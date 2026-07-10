from __future__ import annotations

import logging
import os
from contextlib import ExitStack
from numbers import Number
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr

from insitupy._constants import DEFAULT_CHUNK_SIZE_X, DEFAULT_CHUNK_SIZE_Y
from insitupy._exceptions import InvalidFileTypeError
from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy.containers._zarr_compat import (
    ZARR_V3,
    _get_zarr_store,
    _write_dask_array_to_zarr,
)
from insitupy.images.utils import (
    _efficiently_resize_array,
    _get_scale_factor_from_max_res,
    create_img_pyramid,
    crop_dask_array_or_pyramid,
)
from insitupy.utils._checks import _is_list_of_dask_arrays
from insitupy.utils.utils import convert_to_list, decode_robust_series

logger = logging.getLogger(__name__)


# TODO: Add BoundariesData.read() — loading is complex (zarr pyramids) and
# currently handled through InSituData / CellData.read().

class BoundariesData(DeepCopyMixin):
    '''
    Object to read and load boundaries of cells and nuclei.
    '''
    def __init__(self,
                 cell_names: np.ndarray | list,
                 seg_mask_value: np.ndarray | list | None,
                 nucleus_to_cell_map: dict[int, int] | None = None,
                 nucleus_count: np.ndarray | None = None,
                 ):
        """
        Initialize the BoundariesData object.

        Args:
            cell_names (Union[np.ndarray, List]): Cell names which need to correspond to `.obs_names` in the `.table` of `CellData`.
            seg_mask_value (Optional[Union[np.ndarray, List]]): Segmentation mask values. Required to have the same length as `cell_names`.
                Specifies which values in the "cells" segmentation mask correspond to which cell name.
            nucleus_to_cell_map (Optional[Dict[int, int]]): Mapping from nucleus index (0-indexed) to cell index (0-indexed).
                For Xenium v2.0+ with multinucleated cells, this allows mapping each nucleus to its parent cell.
                To look up a nucleus mask value N, use: `nucleus_to_cell_map[N - 1]` (since mask values are 1-indexed).
                If None, assumes 1:1 mapping between nuclei and cells (Xenium v1.x behavior).
            nucleus_count (Optional[np.ndarray]): Array with the number of nuclei per cell.
                Useful for identifying multinucleated cells. If None, not available.

        For more details on how these values are saved in case of Xenium In Situ, see:
        https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/tutorials/outputs/xoa-output-zarr
        """
        if len(cell_names) != len(seg_mask_value):
            raise ValueError(f"cell_names ({len(cell_names)}) and seg_mask_value ({len(seg_mask_value)}) must have the same length.")

        self._metadata = {}

        # store cell ids
        self._cell_names = da.from_array(np.array(cell_names, dtype=str))

        self._seg_mask_value = seg_mask_value
        if self._seg_mask_value is not None:
            self._seg_mask_value = da.from_array(np.array(seg_mask_value, dtype=np.uint32))
        else:
            raise ValueError("Argument 'seg_mask_value' is None. This argument is required to be set.")

        # Store nucleus-to-cell mapping for multinucleated cell support (Xenium v2.0+)
        self._nucleus_to_cell_map = nucleus_to_cell_map
        self._nucleus_count = nucleus_count

        self._data = dict()
        self._store = None  # zarr store reference for lifecycle management

    def __repr__(self):
        if len(self._data) > 0:
            labels = list(self._metadata.keys())
            if len(labels) == 0:
                repr = "Empty BoundariesData object"
            else:
                ll = len(labels)
                repr = f"BoundariesData object with {ll} {'entry' if ll == 1 else 'entries'}:"
                for l in labels:
                    if self._data[l] is not None:
                        repr += f"\n{tf.SPACER+tf.Bold+l+tf.ResetAll}"
        else:
            repr = "empty"
        return repr

    def close(self):
        """Close the underlying zarr store if one is attached."""
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key: str, item):
        if isinstance(key, str):
            if _is_list_of_dask_arrays(item):
                self._data[key] = item
            else:
                raise TypeError(f"Item for key '{key}' is not a list of dask arrays. Cannot be set.")
        else:
            raise TypeError(f"Key '{key}' is not a string. Cannot be used as key.")

    @property
    def metadata(self):
        """Dict of boundary metadata (pixel size, shape, etc.) keyed by layer name."""
        return self._metadata

    @property
    def cell_names(self):
        """Dask array of cell-name strings, one entry per labelled cell."""
        return self._cell_names

    @property
    def seg_mask_value(self):
        """Dask array of integer mask values corresponding to each cell name."""
        return self._seg_mask_value

    @property
    def nucleus_to_cell_map(self):
        """Mapping from nucleus label ID to cell index. None if not available (v1.x data)."""
        return self._nucleus_to_cell_map

    @property
    def nucleus_count(self):
        """Array with number of nuclei per cell. None if not available."""
        return self._nucleus_count

    @property
    def is_empty(self):
        """True if no boundary masks have been added yet."""
        return len(self._data) == 0

    def update_nucleus_metadata_from_xenium(
        self,
        xenium_path: str | os.PathLike | Path,
        overwrite: bool = False,
    ) -> None:
        """
        Update nucleus_to_cell_map and nucleus_count from raw Xenium data if they are None.

        This method reads the nucleus metadata from the Xenium cells.zarr.zip file
        and updates the BoundariesData object in-place if the properties are missing.
        Useful for updating older saved data that lacks multinucleated cell support.

        Parameters
        ----------
        xenium_path : Union[str, os.PathLike, Path]
            Path to the raw Xenium output directory containing cells.zarr.zip
        overwrite : bool, default False
            If True, update the metadata even if it already exists.
            If False, only update metadata that is currently None.

        Returns
        -------
        None
            Updates the object in-place

        Examples
        --------
        >>> boundaries = read_celldata("path/to/celldata").boundaries
        >>> if boundaries.nucleus_to_cell_map is None:
        ...     boundaries.update_nucleus_metadata_from_xenium("path/to/xenium/output")
        >>>
        >>> # Force update even if metadata exists
        >>> boundaries.update_nucleus_metadata_from_xenium("path/to/xenium/output", overwrite=True)
        """
        from pathlib import Path
        from warnings import warn

        import dask.array as da
        import zarr

        xenium_path = Path(xenium_path)
        cells_zarr_file = xenium_path / "cells.zarr.zip"

        if not cells_zarr_file.exists():
            raise FileNotFoundError(f"Could not find cells.zarr.zip at {cells_zarr_file}")

        # Check if we need to update anything
        needs_nucleus_map = self._nucleus_to_cell_map is None or overwrite
        needs_nucleus_count = self._nucleus_count is None or overwrite

        if not needs_nucleus_map and not needs_nucleus_count:
            logger.info("nucleus_to_cell_map and nucleus_count are already set. No update needed.")
            return

        # Count unique cell and nucleus IDs in current boundaries (excluding background=0)
        cells_data = self._data["cells"][0] if isinstance(self._data["cells"], list) else self._data["cells"]
        ncells_current = len(da.unique(cells_data).compute()) - 1

        nnuclei_current = None
        if "nuclei" in self._data and self._data["nuclei"] is not None:
            nuclei_data = self._data["nuclei"][0] if isinstance(self._data["nuclei"], list) else self._data["nuclei"]
            nnuclei_current = len(da.unique(nuclei_data).compute()) - 1

        # Import helper functions from _io module
        from insitupy._io._xenium import (
            _read_nucleus_count_from_store,
            _read_nucleus_to_cell_map_from_store,
        )

        # Open the Xenium zarr store
        store = zarr.storage.ZipStore(cells_zarr_file, mode='r')

        # Read nucleus_to_cell_map if needed
        if needs_nucleus_map:
            try:
                nucleus_to_cell_map = _read_nucleus_to_cell_map_from_store(store, self._cell_names.compute())

                # Validate that the number of nuclei matches
                if nnuclei_current is not None and len(nucleus_to_cell_map) != nnuclei_current:
                    warn(f"Number of nuclei in nucleus_to_cell_map ({len(nucleus_to_cell_map)}) does not match "
                         f"the number of unique nuclei in boundaries mask ({nnuclei_current}). This may indicate "
                         f"a mismatch between the saved boundaries and the source Xenium data.")

                self._nucleus_to_cell_map = nucleus_to_cell_map
                logger.info(f"Updated nucleus_to_cell_map with {len(nucleus_to_cell_map)} entries.")
            except (KeyError, IndexError, zarr.errors.ArrayNotFoundError) as e:
                warn(f"Could not read nucleus_to_cell_map from Xenium data: {e}")

        # Read nucleus_count if needed
        if needs_nucleus_count:
            try:
                nucleus_count = _read_nucleus_count_from_store(store)
                if nucleus_count is not None:
                    # Validate that the number of cells matches
                    if len(nucleus_count) != ncells_current:
                        warn(f"Number of cells in nucleus_count ({len(nucleus_count)}) does not match "
                             f"the number of unique cells in boundaries mask ({ncells_current}). This may indicate "
                             f"a mismatch between the saved boundaries and the source Xenium data.")

                    self._nucleus_count = nucleus_count
                    logger.info(f"Updated nucleus_count for {len(nucleus_count)} cells.")
                else:
                    warn("nucleus_count not available in Xenium data.")
            except (KeyError, IndexError, zarr.errors.ArrayNotFoundError) as e:
                warn(f"Could not read nucleus_count from Xenium data: {e}")

        store.close()

    def add_boundaries(self,
                       cell_boundaries: da.core.Array | np.ndarray,
                       pixel_size: Number, # required for boundaries that are saved as masks
                       nuclei_boundaries: da.core.Array | np.ndarray | None = None,
                       overwrite: bool = False
                       ):
        """Add cell and optionally nuclei boundary masks to the object.

        Boundary arrays are stored as dask arrays under the ``"cells"``
        and ``"nuclei"`` keys.

        Args:
            cell_boundaries: 2-D segmentation mask array where each cell
                is identified by its segmentation mask value. Can be a
                numpy or dask array.
            pixel_size: Spatial size of one pixel in micrometers. Stored
                in the metadata and used when saving or cropping.
            nuclei_boundaries: Optional 2-D segmentation mask for nuclei,
                following the same convention as ``cell_boundaries``.  If
                None, no nuclei boundaries are stored.
            overwrite: If True, replace existing boundary entries. Raises
                ``KeyError`` when boundaries already exist and
                ``overwrite`` is False.

        Raises:
            ValueError: If ``cell_boundaries`` is None.
            TypeError: If boundary inputs are not dask arrays, numpy
                arrays, or lists.
            KeyError: If boundary labels already exist and ``overwrite``
                is False.
        """
        if cell_boundaries is None:
            raise ValueError("cell_boundaries cannot be None.")

        # make sure the boundaries are a dask array
        if isinstance(cell_boundaries, np.ndarray):
            cell_boundaries = da.from_array(cell_boundaries)

        if isinstance(nuclei_boundaries, np.ndarray):
            nuclei_boundaries = da.from_array(nuclei_boundaries)

        if not (isinstance(cell_boundaries, da.core.Array) or isinstance(cell_boundaries, list) or cell_boundaries is None):
            raise TypeError("cell_boundaries must be a dask/numpy array or a list")

        if not (isinstance(nuclei_boundaries, da.core.Array) or isinstance(nuclei_boundaries, list) or nuclei_boundaries is None):
            raise TypeError("nuclei_boundaries must be a dask/numpy array, a list, or None")

        data = {
            "cells": cell_boundaries,
            "nuclei": nuclei_boundaries
        }

        for l, img in data.items():
            if l not in self._metadata or overwrite:
                # add to object
                self._data[l] = img
                self._metadata[l] = {}
                self._metadata[l]["pixel_size"] = pixel_size
            else:
                raise KeyError(f"Label '{l}' exists already in BoundariesData object. To overwrite, set 'overwrite' argument to True.")

    def crop(self,
             cell_ids: list[str],
             xlim: tuple[int, int],
             ylim: tuple[int, int],
             inplace: bool = False
             ):
        '''
        Crop the BoundariesData object.
        '''

        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        # make sure cell ids are a list
        cell_ids = convert_to_list(cell_ids)

        for n, meta in _self._metadata.items():
            # get dataframe
            data = _self[n]

            if data is not None:
                # get pixel size
                pixel_size = meta["pixel_size"]

                data = crop_dask_array_or_pyramid(
                    data=data,
                    xlim=xlim,
                    ylim=ylim,
                    pixel_size=pixel_size
                )

            # add to object
            _self._data[n] = data

        if not inplace:
            return _self

    def convert_to_shapely_objects(self):
        """Convert raw coordinate DataFrames to GeoDataFrames with Shapely Polygon objects.

        Iterates over all boundary layers that are still stored as plain
        :class:`pandas.DataFrame` objects (with ``vertex_x`` / ``vertex_y``
        columns) and converts them to cell-level
        :class:`~shapely.geometry.Polygon` geometries grouped by
        ``cell_id``.  Layers that are already converted are skipped with a
        warning.
        """
        for n in self._metadata.keys():
            logger.info(f"Converting `{n}` to GeoPandas DataFrame with shapely objects.")
            # retrief dataframe with boundary coordinates
            df = self[n]

            if isinstance(df, pd.DataFrame):
                # convert xy coordinates into shapely Point objects
                df["geometry"] = gpd.points_from_xy(df["vertex_x"], df["vertex_y"])
                del df["vertex_x"], df["vertex_y"]

                # convert points into polygon objects per cell id
                df = df.groupby("cell_id")['geometry'].apply(lambda x: Polygon(x.tolist()))
                df.index = decode_robust_series(df.index)  # convert byte strings in index

                # add to object
                self._data[n] = pd.DataFrame(df)
            else:
                logger.warning(f"Boundaries element `{n}` was no Dataframe. Skipped.")

    def save(self,
             path : str | os.PathLike | Path = "boundaries.zarr.zip",
             save_as_pyramid: bool = True,
             max_resolution: Number | None = None,
             verbose: bool = False
             ):
        """Save boundary masks, cell names, and metadata to a zarr store.

        Writes cell and nuclei masks, cell names, segmentation mask
        values, and (when available) nucleus-to-cell mapping and nucleus
        counts into a zarr or zarr.zip archive.

        Args:
            path: Output file path. Must end with ``.zarr`` or
                ``.zarr.zip``.
            save_as_pyramid: If True, store boundary masks as a
                multi-resolution image pyramid.
            max_resolution: Maximum spatial resolution in micrometers per
                pixel.  Masks with finer resolution are downsampled
                before saving.  If None, masks are saved at their
                original resolution.
            verbose: If True, log progress messages.

        Raises:
            InvalidFileTypeError: If ``path`` does not end with
                ``.zarr`` or ``.zarr.zip``.
        """

        path = Path(path)
        suffix = path.name.split(".", maxsplit=1)[-1]

        if suffix not in ["zarr", "zarr.zip"]:
            raise InvalidFileTypeError(allowed_types=[".zarr", ".zarr.zip"], received_type=suffix)

        zipped = suffix == "zarr.zip"

        # Use ExitStack to handle context manager differences between Zarr v2 and v3
        with ExitStack() as stack:
            dirstore = _get_zarr_store(path, mode="w", zipped=zipped)

            # In Zarr v2, stores are context managers and need to be entered
            if not ZARR_V3:
                dirstore = stack.enter_context(dirstore)

            for n, meta in self._metadata.items():
                bound_data = self[n]

                # determine scale factor
                scale_factor = _get_scale_factor_from_max_res(pixel_size=meta['pixel_size'], max_resolution=max_resolution)

                if bound_data is not None:
                    if scale_factor is not None:
                        if isinstance(bound_data, list):
                            bound_data = bound_data[0]
                        bound_data = _efficiently_resize_array(array=bound_data, scale_factor=scale_factor)
                        bound_data = da.from_array(bound_data) # convert to dask array
                        meta['pixel_size'] = max_resolution # update metadata

                    # check data
                    if isinstance(bound_data, list):
                        if not save_as_pyramid:
                            bound_data = bound_data[0]
                    else:
                        if save_as_pyramid:
                            # create pyramid
                            bound_data = create_img_pyramid(img=bound_data, axes="YX", nsubres=6)


                    if isinstance(bound_data, list):
                        for i, b in enumerate(bound_data):
                            comp = f"masks/{n}/{i}"
                            b = b.rechunk((DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X))
                            b.to_zarr(dirstore, component=comp)
                    else:
                        # Apply chunking for non-pyramid data (YX axes)
                        bound_data = bound_data.rechunk((DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X))
                        bound_data.to_zarr(dirstore, component=f"masks/{n}")

                    # add boundaries metadata to zarr.zip
                    store = zarr.open(dirstore, mode="a")
                    store[f"masks/{n}"].attrs.put(meta)

            # save cell names
            _write_dask_array_to_zarr(dirstore, "cell_names", self.cell_names)

            if self._seg_mask_value is not None:
                _write_dask_array_to_zarr(dirstore, "seg_mask_value", self.seg_mask_value)

            # Save nucleus_to_cell_map if available (for multinucleated cell support)
            if self._nucleus_to_cell_map is not None:
                # Store as 2D array with columns [nucleus_index, cell_index]
                nucleus_map_arr = np.array([[k, v] for k, v in self._nucleus_to_cell_map.items()], dtype=np.int64)
                _write_dask_array_to_zarr(dirstore, "nucleus_to_cell_map", da.from_array(nucleus_map_arr))

            # Save nucleus_count if available
            if self._nucleus_count is not None:
                _write_dask_array_to_zarr(dirstore, "nucleus_count", da.from_array(self._nucleus_count))

from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from numbers import Number
from os.path import relpath
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from shapely import MultiPolygon, Polygon

from insitupy._io.files import check_overwrite_and_remove_if_true, write_dict_to_json
from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy._version import __version__

logger = logging.getLogger(__name__)


class CellData(DeepCopyMixin):
    '''
    Data object containing an AnnData object and a boundary object which are kept in sync.
    '''
    def __init__(self,
               table: AnnData,
               boundaries=None,
               config: dict = {}
               ):
        self._table = table
        self._config = config

        if boundaries is not None:
            self._boundaries = boundaries
            self._entries = ["table", "boundaries"]
        else:
            self._boundaries = None
            self._entries = ["table"]

    @classmethod
    def read(cls, path):
        """Read CellData from a saved directory.

        Args:
            path: Path to the CellData directory.

        Returns:
            CellData: The loaded CellData object.
        """
        from insitupy.containers.io import _read_celldata
        return _read_celldata(path)

    def __getitem__(self, key):
        """Retrieve a subset of the `CellData` object.

        Args:
            key (int, slice, list, np.ndarray, pd.Series): The index, slice, list of indices, boolean mask,
                or Series to retrieve.

        Returns:
            `CellData`: A new `CellData` object with the selected subset of cells.
        """
        new_celldata = self.copy()
        new_celldata._table = new_celldata._table[key].copy()
        new_celldata.sync()
        return new_celldata

    def __len__(self):
        """Return the number of cells in the `CellData` object.

        Returns:
            int: The number of cells.
        """
        return len(self.table)

    def __repr__(self):
        repr = (
            f"{tf.Bold+'table'+tf.ResetAll}\n"
            f"{tf.SPACER+self._table.__repr__()}"
        )

        if self._boundaries is not None:
            bound_repr = self._boundaries.__repr__()

            repr += f"\n{tf.Bold+'boundaries'+tf.ResetAll}\n" + tf.SPACER + bound_repr.replace("\n", f"\n{tf.SPACER}")
        return repr

    @property
    def matrix(self):
        """Deprecated alias for :attr:`table`. Use ``table`` instead."""
        warnings.warn(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._table

    @matrix.setter
    def matrix(self, value: AnnData):
        """Deprecated alias for the ``table`` setter. Use ``table`` instead."""
        warnings.warn(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_table(value=value, allow_partial_overlap=False)

    @property
    def table(self):
        """Alias for matrix property. This is the preferred name going forward."""
        return self._table

    @table.setter
    def table(self, value: AnnData):
        """Alias for matrix setter. This is the preferred name going forward."""
        self._set_table(value=value, allow_partial_overlap=False)

    def set_table(self,
                  value: AnnData,
                  allow_partial_overlap: bool = False):
        """
        Safely set table data while keeping table and boundaries consistent.

        Args:
            value: New AnnData table.
            allow_partial_overlap: If True and boundaries exist, keep only cells present
                in both table and boundaries. If False, fail when table contains cell IDs
                that are not present in boundaries.
        """
        self._set_table(value=value, allow_partial_overlap=allow_partial_overlap)

    def _set_table(self,
                   value: AnnData,
                   allow_partial_overlap: bool = False):
        if not isinstance(value, AnnData):
            raise TypeError(f"Table must be an AnnData object. Instead: {type(value)}.")

        if self._boundaries is not None:
            table_cell_ids = pd.Index(value.obs_names.astype(str))

            if len(table_cell_ids.unique()) != len(table_cell_ids):
                raise ValueError("Table .obs_names must be unique when boundaries are present.")

            boundary_cell_ids = pd.Index(self._boundaries.cell_names.compute().astype(str))
            missing_in_boundaries = table_cell_ids.difference(boundary_cell_ids)

            if len(missing_in_boundaries) > 0 and not allow_partial_overlap:
                missing_preview = ", ".join(map(str, missing_in_boundaries[:5]))
                if len(missing_in_boundaries) > 5:
                    missing_preview += ", ..."
                raise ValueError(
                    "New table contains cell IDs that are not present in boundaries "
                    f"({len(missing_in_boundaries)} missing; examples: {missing_preview}). "
                    "Use `set_table(..., allow_partial_overlap=True)` to keep only overlapping cells."
                )

        self._table = value

        if self._boundaries is not None:
            self.sync(verbose=False)

    @property
    def config(self):
        """Configuration object storing segmentation and data-loading parameters."""
        return self._config

    @property
    def boundaries(self):
        """Associated :class:`~insitupy.containers.boundaries_data.BoundariesData` object, or None."""
        return self._boundaries

    @property
    def is_synced(self) -> bool:
        """True if the cell IDs in the table and boundaries are identical and ordered the same way."""
        if self._boundaries is None:
            return True

        table_cell_ids = pd.Index(self._table.obs_names.astype(str))
        boundary_cell_ids = pd.Index(self._boundaries.cell_names.compute().astype(str))
        seg_mask_values = self._boundaries.seg_mask_value.compute()

        if len(table_cell_ids.unique()) != len(table_cell_ids):
            return False
        if len(boundary_cell_ids.unique()) != len(boundary_cell_ids):
            return False
        if len(boundary_cell_ids) != len(seg_mask_values):
            return False

        return table_cell_ids.equals(boundary_cell_ids)

    @property
    def entries(self):
        """List of data entries (e.g. file sources) associated with this layer."""
        return self._entries

    def copy(self):
        '''
        Function to generate a deep copy of the current object.
        '''

        return deepcopy(self)

    def close(self):
        """Close underlying resources owned by this CellData object."""
        if self._boundaries is not None:
            close_method = getattr(self._boundaries, "close", None)
            if callable(close_method):
                close_method()

    def crop(self,
            xlim: tuple[int, int] | None = None,
            ylim: tuple[int, int] | None = None,
            shape: Polygon | MultiPolygon | None = None,
            inplace: bool = False,
            verbose: bool = True
            ):
        """Crop the CellData object to a spatial region.

        Cells are filtered by their spatial coordinates. Boundaries (if
        present) are cropped accordingly and coordinates are shifted so
        that the origin of the cropped region becomes (0, 0).

        Either ``xlim``/``ylim`` or ``shape`` must be provided.  When
        ``shape`` is given it takes precedence over ``xlim``/``ylim``.

        Args:
            xlim: Tuple of (min_x, max_x) pixel limits for rectangular
                cropping.
            ylim: Tuple of (min_y, max_y) pixel limits for rectangular
                cropping.
            shape: A Shapely Polygon or MultiPolygon used for
                non-rectangular cropping.
            inplace: If True, modify this object in place. Otherwise
                return a cropped copy.
            verbose: If True, log a warning when both xlim/ylim and shape
                are provided.

        Returns:
            CellData or None: A new cropped CellData object if
                ``inplace=False``, otherwise None.
        """
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        # retrieve cell coordinates
        cell_coords = _self.table.obsm['spatial'].copy()

        # Ensure that either both xlim and ylim are not None or shape is not None
        if (xlim is None or ylim is None) and shape is None:
            raise ValueError("Either both xlim and ylim must be provided, or shape must be provided.")

        if shape is not None:
            if xlim is not None and ylim is not None:
                if verbose:
                    logger.warning("Both xlim/ylim and shape are provided. Shape will be used for cropping.")

            # create shapely objects from cell coordinates
            cells = gpd.points_from_xy(cell_coords[:, 0], cell_coords[:, 1])

            # create a mask based on the shape
            mask = shape.contains(cells)

            # get bounding box of shape
            minx, miny, maxx, maxy = shape.bounds # (minx, miny, maxx, maxy)
            # make sure there are no negative values in the limits, consistent
            # with the xlim/ylim branch below and with InSituData.crop, which
            # clips the region bounds before cropping the images
            xlim = (max(0.0, minx), maxx)
            ylim = (max(0.0, miny), maxy)

        else:
            if xlim is None or ylim is None:
                raise ValueError("Either both xlim and ylim must be provided, or shape must be provided.")

            # make sure there are no negative values in the limits
            xlim = tuple(np.clip(xlim, a_min=0, a_max=None))
            ylim = tuple(np.clip(ylim, a_min=0, a_max=None))

            # create a mask based on xlim and ylim
            xmask = (cell_coords[:, 0] >= xlim[0]) & (cell_coords[:, 0] <= xlim[1])
            ymask = (cell_coords[:, 1] >= ylim[0]) & (cell_coords[:, 1] <= ylim[1])
            mask = xmask & ymask

        # select
        _self._table = _self.table[mask, :].copy()

        # crop boundaries
        if _self.boundaries is not None:
            _self.boundaries.crop(
                cell_ids=_self.table.obs_names,
                xlim=xlim, ylim=ylim,
                inplace=True
                )

        # shift coordinates to correct for change of coordinates during cropping
        _self.shift(x=-xlim[0], y=-ylim[0])

        # sync the ids and names
        _self.sync(verbose=verbose)

        if not inplace:
            return _self


    def save(self,
             path: str | os.PathLike | Path,
             max_resolution_boundaries: Number | None = None,
             overwrite: bool = False
             ):
        """Save the CellData object to disk.

        Writes the AnnData table as ``table.h5ad`` and, if boundaries are
        present, saves them as a zarr store inside ``path``.  A
        ``.celldata`` JSON metadata file is written alongside.

        Args:
            path: Directory to save into. Created if it does not exist.
            max_resolution_boundaries: Maximum spatial resolution for
                saved boundaries in micrometers per pixel.  Boundaries
                with finer resolution are downsampled.  If None, boundaries
                are saved at their original resolution.
            overwrite: If True, remove ``path`` first if it already exists.
        """
        path = Path(path)
        celldata_metadata = {}

        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)

        # write table to file
        mtx_file = path / "table.h5ad"
        self._table.write(mtx_file)
        celldata_metadata["table"] = Path(relpath(mtx_file, path)).as_posix()

        # save boundaries
        if self._boundaries is not None:
            boundaries = self._boundaries
            bound_file = path / "boundaries.zarr"

            # save boundaries
            boundaries.save(bound_file, max_resolution=max_resolution_boundaries)

            # add entry for boundaries to metadata
            celldata_metadata["boundaries"] = Path(relpath(bound_file, path)).as_posix()

        # add version to metadata
        celldata_metadata["version"] = __version__

        # add configurations
        if self._config is not None:
            celldata_metadata["config"] = self._config

        # save metadata
        write_dict_to_json(dictionary=celldata_metadata, file=path / ".celldata")


    def sync(self,
             verbose: bool = False,
             return_summary: bool = False):
        '''
        Function to synchronize table and boundaries of CellData.

        Procedure:
        1. Select table cell IDs
        2. Check if all table cell IDs are in boundaries
            - if not all are in boundaries, throw error saying that those will also be removed
        3. Select only table cell IDs which are also in boundaries and filter for them

        Args:
            verbose: Print synchronization changes to stdout.
            return_summary: If True, return a dict describing synchronization actions.
        '''
        if self._boundaries is None:
            summary = {
                "had_boundaries": False,
                "changed": False,
                "removed_table": 0,
                "removed_boundaries": 0,
                "reordered_boundaries": False,
            }
            if verbose:
                logger.info("CellData.sync(): no boundaries present; nothing to synchronize.")
            if return_summary:
                return summary
            return None
        else:
            # get cell IDs from table
            table_cell_ids = pd.Index(self._table.obs_names.astype(str))

            if len(table_cell_ids.unique()) != len(table_cell_ids):
                raise ValueError("Table .obs_names must be unique to synchronize with boundaries.")

            boundaries = self._boundaries

            # create pandas series from seg_mask values and cell_names
            ds = pd.Series(
                data=boundaries.seg_mask_value.compute(),
                index=boundaries.cell_names.compute().astype(str)
                )

            # filter table for IDs that are available in boundaries
            table_mask_in_boundaries = table_cell_ids.isin(ds.index)
            overlapping_table_ids = table_cell_ids[table_mask_in_boundaries]
            boundary_overlap_ids = ds.index[ds.index.isin(overlapping_table_ids)]

            if len(table_cell_ids) > 0 and not np.any(table_mask_in_boundaries):
                raise ValueError("No matching values between `.boundaries.cell_names` and `.table.obs_names`. All table entries would get filtered out.")

            n_removed_table = int(np.sum(~table_mask_in_boundaries))
            if n_removed_table > 0:
                self._table = self._table[table_mask_in_boundaries, :].copy()
                table_cell_ids = pd.Index(self._table.obs_names.astype(str))

            # align boundary metadata to table order
            ds_aligned_to_table = ds.reindex(table_cell_ids)

            boundaries._seg_mask_value = da.from_array(np.array(ds_aligned_to_table.values, dtype=np.uint32))
            boundaries._cell_names = da.from_array(np.array(ds_aligned_to_table.index, dtype=str))

            # keep nucleus_to_cell_map/nucleus_count aligned to the new cell_names order
            old_cell_names = ds.index.to_numpy().astype(str)  # pre-sync boundary/cell order
            new_cell_names = ds_aligned_to_table.index.to_numpy().astype(str)  # new table order
            n_old = len(old_cell_names)

            # nucleus_to_cell_map: value = parent cell name (stable across filter/reorder)
            nmap = boundaries._nucleus_to_cell_map
            if nmap is not None:
                old_name_set = set(old_cell_names)
                unknown = {k: v for k, v in nmap.items() if v not in old_name_set}
                if unknown:
                    warnings.warn(
                        f"nucleus_to_cell_map has {len(unknown)} entr{'y' if len(unknown) == 1 else 'ies'} "
                        "referencing a cell name not present in the current boundaries; dropping "
                        "the map. Re-read from raw data to restore multinucleated-cell mapping.",
                        stacklevel=2)
                    boundaries._nucleus_to_cell_map = None
                else:
                    new_name_set = set(new_cell_names)
                    remapped = {k: v for k, v in nmap.items() if v in new_name_set}
                    boundaries._nucleus_to_cell_map = remapped or None

            # nucleus_count: per-cell, aligned to cell order
            ncount = boundaries._nucleus_count
            if ncount is not None:
                ncount = np.asarray(ncount)
                if len(ncount) != n_old:
                    warnings.warn("nucleus_count length does not match the cell table; dropping it.",
                                  stacklevel=2)
                    boundaries._nucleus_count = None
                else:
                    boundaries._nucleus_count = (
                        pd.Series(ncount, index=old_cell_names).reindex(new_cell_names).to_numpy())

            # find the seg_mask_values which are not anymore present
            seg_mask_values_not_in_table = ds[~ds.index.isin(table_cell_ids)].values
            n_removed_boundaries = int(np.sum(~ds.index.isin(table_cell_ids)))
            was_reordered = not overlapping_table_ids.equals(boundary_overlap_ids)
            changed = any([n_removed_table > 0, n_removed_boundaries > 0, was_reordered])

            # extract boundaries
            cell_bounds = boundaries["cells"]
            nuc_bounds = boundaries["nuclei"]

            if isinstance(cell_bounds, list):
                if nuc_bounds is not None:
                    if not isinstance(nuc_bounds, list):
                        raise TypeError("Cellular boundaries are a image pyramid but nuclear boundaries are not. Both need to be of the same type for the synchronization to work.")
                for i, cell_bound in enumerate(cell_bounds):
                    removed_cells_mask = da.isin(cell_bound, seg_mask_values_not_in_table)
                    cell_bound[removed_cells_mask] = 0 # set all removed cells 0
                    if nuc_bounds is not None:
                        nuc_bounds[i][removed_cells_mask] = 0 # set all nuclei belong to the removed cells 0
            elif isinstance(cell_bounds, da.core.Array):
                if nuc_bounds is not None:
                    if not isinstance(nuc_bounds, da.core.Array):
                        raise TypeError("Cellular boundaries are a dask array but nuclear boundaries are not. Both need to be of the same type for the synchronization to work.")
                # set all non existent cell ids to zero
                removed_cells_mask = da.isin(cell_bounds, seg_mask_values_not_in_table)
                cell_bounds[removed_cells_mask] = 0 # set all removed cells 0

                if nuc_bounds is not None:
                    nuc_bounds[removed_cells_mask] = 0 # set all nuclei belong to the removed cells 0
            else:
                logger.warning(f"Unknown data type for cellular boundaries: {type(cell_bounds)}. Need to be either a dask array or a list of dask arrays. Skipped synchronization of cell ids.")

            summary = {
                "had_boundaries": True,
                "changed": changed,
                "removed_table": n_removed_table,
                "removed_boundaries": n_removed_boundaries,
                "reordered_boundaries": was_reordered,
            }

            if verbose:
                if changed:
                    changes = []
                    if n_removed_table > 0:
                        changes.append(f"removed {n_removed_table} table entries")
                    if n_removed_boundaries > 0:
                        changes.append(f"removed {n_removed_boundaries} boundary entries")
                    if was_reordered:
                        changes.append("reordered boundary metadata to match table order")
                    logger.info(f"CellData.sync(): synchronized table and boundaries ({', '.join(changes)}).")
                else:
                    logger.info("CellData.sync(): no synchronization needed; table and boundaries are already aligned.")

            if return_summary:
                return summary
            return None

    def shift(self,
              x: int | float,
              y: int | float
              ):
        '''
        Function to shift the coordinates of both table and boundaries data by certain values x/y.
        '''

        # move origin again to 0 by subtracting the lower limits from the coordinates
        cell_coords = self._table.obsm['spatial'].copy()
        cell_coords[:, 0] += x
        cell_coords[:, 1] += y
        self._table.obsm['spatial'] = cell_coords

        if self._boundaries is None:
            logger.warning('No `boundaries` attribute found in CellData found.')
        else:
            boundaries = self._boundaries
            for n in boundaries.metadata.keys():
                # get dataframe
                df = boundaries[n]

                if isinstance(df, pd.DataFrame):
                    # re-center to 0
                    df["vertex_x"] += x
                    df["vertex_y"] += y

                    # add to object
                    setattr(self._boundaries, n, df)

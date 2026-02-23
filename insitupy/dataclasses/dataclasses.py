import logging
import os
import warnings
from contextlib import ExitStack
from copy import deepcopy
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from shapely import MultiPoint, MultiPolygon, Point, Polygon, affinity

from insitupy import WITH_NAPARI, __version__
from insitupy._constants import (DEFAULT_CHUNK_SIZE_X, DEFAULT_CHUNK_SIZE_Y,
                                 FORBIDDEN_ANNOTATION_NAMES, RED)
from insitupy._exceptions import InvalidFileTypeError
from insitupy._io.files import (check_overwrite_and_remove_if_true,
                                write_dict_to_json)
from insitupy._io.geo import parse_geopandas, write_qupath_geojson
from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy.dataclasses._segmentations import _read_baysor, _read_proseg
from insitupy.images.axes import (ImageAxes, _transpose_to_standard_axes,
                                  get_height_and_width)
from insitupy.images.io import (get_zarr_source_path, is_from_zarr_disk,
                                read_image, write_ome_tiff, write_zarr)
from insitupy.images.utils import (_efficiently_resize_array,
                                   _get_scale_factor_from_max_res,
                                   create_img_pyramid,
                                   crop_dask_array_or_pyramid, resize_image)
from insitupy.utils._checks import _is_list_of_dask_arrays
from insitupy.utils.utils import convert_to_list, decode_robust_series

logger = logging.getLogger(__name__)

# Detect Zarr version for compatibility
ZARR_V3 = hasattr(zarr.storage, 'LocalStore')


def _get_zarr_store(path, mode: str = "r", zipped: bool = False):
    """
    Get a Zarr store compatible with both Zarr v2 and v3.

    Args:
        path: Path to the zarr store
        mode: Mode to open the store ('r', 'w', 'a')
        zipped: Whether the store is a ZipStore

    Returns:
        For Zarr v3: store object (no context manager needed)
        For Zarr v2: store object (should be used as context manager)
    """
    if ZARR_V3:
        # Zarr v3 API
        if zipped:
            return zarr.storage.ZipStore(path, mode=mode)
        else:
            return zarr.storage.LocalStore(path)
    else:
        # Zarr v2 API
        if zipped:
            return zarr.ZipStore(path, mode=mode)
        else:
            return zarr.DirectoryStore(path)


if WITH_NAPARI:
    from napari.utils.notifications import show_info, show_warning



class ShapesData(DeepCopyMixin):
    '''
    Object to store annotations.
    '''
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None,
                 keys: Optional[List[str]] = None,
                 pixel_size: Optional[float] = None,
                 assert_uniqueness: bool = False,
                 polygons_only: bool = False,
                 forbidden_names: Optional[List[str]] = None,
                 shape_name: Optional[str] = None,
                 ) -> None:
        self._shape_name = shape_name if shape_name is not None else "shapes"

        # add hidden variables
        self._data = {}
        self._assert_uniqueness = assert_uniqueness
        self._polygons_only = polygons_only
        self._forbidden_names = forbidden_names

        if files is not None:
            # make sure files and keys are a list
            assert keys is not None, "If `files` are given, also corresponding `keys` need to be given."
            files = convert_to_list(files)
            keys = convert_to_list(keys)
            assert len(files) == len(keys), "Number of files does not match number of keys."

            assert pixel_size is not None, "If files and `keys` are given, also `pixel_size` needs to be specified"

            if files is not None:
                for key, file in zip(keys, files):
                    # read annotation and store in dictionary
                    self.add_data(data=file,
                                  key=key,
                                  scale_factor=pixel_size,
                                  )

    def __repr__(self):
        if len(self._data) > 0:
            repr_strings = []
            for l, m in self.metadata.items():
                # get metadata
                n = m[f"n_{self._shape_name}"]
                classes = m["classes"]
                classes_str = [f"'{elem}'" for elem in classes]
                lc = len(classes)

                # create string
                r = (
                    f'{tf.Bold}{l}:{tf.ResetAll}\t{n} '
                    f'{self._shape_name}, {lc} '
                    f'{"classes" if lc>1 else "class"} '
                )
                if lc < 10:
                    r += f'({", ".join(classes_str)})'
                repr_strings.append(r)

            s = "\n".join(repr_strings)
        else:
            s = "empty"

        return s

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def metadata(self):
        """Compute metadata on-demand from current data state."""
        meta = {}
        for key, df in self._data.items():
            meta[key] = {
                f"n_{self._shape_name}": len(df),
                "classes": sorted(df['name'].unique().tolist()) if 'name' in df.columns else ["unnamed"],
            }
        return meta

    @property
    def is_empty(self):
        return len(self._data) == 0

    def _check_uniqueness(self,
                          dataframe: Optional[gpd.GeoDataFrame] = None,
                          key: Optional[str] = None,
                          verbose: bool = True
                          ) -> bool:

        if dataframe is None:
            annot_df = self[key]
        else:
            annot_df = dataframe

        if len(annot_df.index.unique()) != len(annot_df.name.unique()):
            warnings.warn(
                message=
                (
                    f"The names of the {self._shape_name} for key '{key}' were not unique and thus "
                    f"the key was skipped. In regions only one geometry per class is allowed."
                    f"To achieve this in the napari viewer select one layer per region and give each layer a unique name."
                    )
                )
            return False
        else:
            if verbose:
                print(f"Names of {self._shape_name} for key '{key}' are unique.")
            return True

    def add_data(self,
                 data: Union[gpd.GeoDataFrame, pd.DataFrame, dict,
                                str, os.PathLike, Path],
                 key: str,
                 scale_factor: Number,
                 verbose: bool = False,
                 in_napari: bool = False
                   ):
        # parse geopandas data from dataframe or file
        new_df = parse_geopandas(data)

        if new_df is None:
            print(f"Data for key '{key}' was empty. Skipped import.", flush=True)
        else:
            if "name" not in new_df.columns:
                new_df["name"] = ["None"] * len(new_df)

            if "color" not in new_df.columns:
                warnings.warn("No 'color' column found in the imported data. Setting all colors to red.", stacklevel=2)
                new_df["color"] = [RED] * len(new_df)

            if self._forbidden_names is not None:
                try:
                    new_names = new_df["name"].tolist()
                except KeyError:
                    pass
                else:
                    if np.any([elem in new_names for elem in self._forbidden_names]):
                        raise ValueError(f"One of the forbidden names for annotations ({self._forbidden_names}) has been used in the imported dataset. Please change the respective change to prevent interference with downstream functions.")

            # convert geometries into unit (e.g. µm) values
            new_df["geometry"] = new_df["geometry"].scale(xfact=scale_factor, yfact=scale_factor, origin=(0,0))

            # determine the type of layer that needs to be used in napari later
            layer_types = []
            for geom in new_df["geometry"]:
                if isinstance(geom, Point) or isinstance(geom, MultiPoint):
                    layer_types.append("Points")
                else:
                    layer_types.append("Shapes")
            new_df["layer_type"] = layer_types

            if key not in self._data.keys():
                # if key does not exist yet, the new df is the whole annotation dataframe
                annot_df = new_df

                # collect additional variables for reporting
                new_geometries_added = True
                existing_str = ""
                old_n = 0
                new_n = len(annot_df)
            else:
                # concatenate old and new annoation dataframe
                annot_df = self[key]
                old_n = len(annot_df)
                annot_df = pd.concat([annot_df, new_df], ignore_index=False)

                # remove all duplicated shapes - leaving only the newly added
                dup_mask = annot_df.index.duplicated(keep="last")
                annot_df = annot_df[~dup_mask]
                new_n = len(annot_df)

                # collect additional variables for reporting
                new_geometries_added = new_n > old_n
                existing_str = "existing "

            if new_geometries_added:
                if self._assert_uniqueness:
                    # check if the shapes data for this key is unique
                    is_unique = self._check_uniqueness(dataframe=annot_df, key=key, verbose=False)

                    if not is_unique:
                        return

                if self._polygons_only:
                    # check if any of the shapes are not shapely Polygons
                    is_not_polygon = np.array([not isinstance(p, Polygon) for p in annot_df.geometry])
                    if np.any(is_not_polygon):
                        annot_df = annot_df.loc[~is_not_polygon]
                        show_warning(f"Some {self._shape_name} were not shapely.Polygon objects and skipped.")

            # check that the dataframe is not empty
            if len(annot_df) > 0:
                # add dataframe to ShapesData object
                self._data[key] = annot_df

                if verbose:
                    # report
                    if in_napari:
                        _show_func = show_info
                    else:
                        _show_func = print
                    if new_geometries_added:
                        _show_func(f"Added {new_n - old_n} new {self._shape_name} to {existing_str}key '{key}'")
                    else:
                        _show_func(f"Updated {self._shape_name} to {existing_str}key '{key}'")

    def crop(self,
             shape,
             xlim,
             ylim,
             verbose: bool = True,
             inplace: bool = False
             ):
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        if shape is None:
            if (xlim is None) or (ylim is None):
                raise ValueError("If shape is None, both xlim and ylim must not be None.")
            else:
                shape = Polygon([(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])])
        else:
            if (xlim is not None) and (ylim is not None):
                if verbose:
                    warnings.warn("Both xlim/ylim and shape are provided. Shape will be used for cropping.")

        keys_to_remove = []
        for key in list(_self._data.keys()):
            shapesdf = _self[key]

            # select annotations that intersect with the selected area
            mask = [shape.intersects(elem) for elem in shapesdf["geometry"]]
            shapesdf = shapesdf.loc[mask, :].copy()

            # move origin to zero after cropping
            shapesdf["geometry"] = shapesdf["geometry"].apply(affinity.translate, xoff=-xlim[0], yoff=-ylim[0])

            # check if there are annotations left or if it has to be deleted
            if len(shapesdf) > 0:
                # add new dataframe back to annotations object
                _self._data[key] = shapesdf
            else:
                # mark for deletion
                keys_to_remove.append(key)

        # delete empty keys
        for key in keys_to_remove:
            del _self._data[key]

        if not inplace:
            return _self

    def keys(self):
        return self._data.keys()

    def remove_key(
        self,
        key_to_remove: str,
        classes_to_remove: Union[Literal["all"], List[str], str] = "all"
        ):
        if classes_to_remove == "all":
            try:
                del self._data[key_to_remove]
            except KeyError:
                print(f"Key '{key_to_remove}' not found in ShapesData object. Nothing to remove.")
        else:
            classes_to_remove = convert_to_list(classes_to_remove)
            geom_df = self[key_to_remove]
            self._data[key_to_remove] = geom_df[~geom_df.name.isin(classes_to_remove)]

    def save(self,
             path: Union[str, os.PathLike, Path],
             overwrite: bool = False
             ):
        path = Path(path)

        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)

        # save each shape layer as geojson
        for key in self.keys():
            df = self[key]
            shapes_file = path / f"{key}.geojson"
            write_qupath_geojson(dataframe=df, file=shapes_file)

        # # save metadata
        # shape_meta_path = path / f"metadata.json"
        # write_dict_to_json(dictionary=self.metadata, file=shape_meta_path)


class AnnotationsData(ShapesData):
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None,
                 keys: Optional[List[str]] = None,
                 pixel_size: Optional[float] = None
                 ) -> None:

        ShapesData.__init__(
            self,
            files=files,
            keys=keys,
            pixel_size=pixel_size,
            assert_uniqueness=False,
            polygons_only=False,
            forbidden_names=FORBIDDEN_ANNOTATION_NAMES,
            shape_name="annotations",
            )


class RegionsData(ShapesData):
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None,
                 keys: Optional[List[str]] = None,
                 pixel_size: Optional[float] = None
                 ) -> None:

        ShapesData.__init__(
            self,
            files=files,
            keys=keys,
            pixel_size=pixel_size,
            assert_uniqueness=True,
            polygons_only=True,
            forbidden_names=None,
            shape_name="regions",
            )

class BoundariesData(DeepCopyMixin):
    '''
    Object to read and load boundaries of cells and nuclei.
    '''
    def __init__(self,
                 cell_names: Union[np.ndarray, List],
                 seg_mask_value: Optional[Union[np.ndarray, List]],
                 nucleus_to_cell_map: Optional[Dict[int, int]] = None,
                 nucleus_count: Optional[np.ndarray] = None,
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
        #self._cell_ids = da.from_array(np.array(cell_ids, dtype=np.uint32))
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
                repr = f"Empty BoundariesData object"
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
                raise ValueError(f"Item for key '{key}' is not a list of dask arrays. Cannot be set.")
        else:
            raise ValueError(f"Key '{key}' is not a string. Cannot be used as key.")

    @property
    def metadata(self):
        return self._metadata

    # @property
    # def cell_ids(self):
    #     return self._cell_ids

    @property
    def cell_names(self):
        return self._cell_names

    @property
    def seg_mask_value(self):
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
        return len(self._data) == 0

    def update_nucleus_metadata_from_xenium(
        self,
        xenium_path: Union[str, os.PathLike, Path],
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
        from zarr.errors import ArrayNotFoundError

        xenium_path = Path(xenium_path)
        cells_zarr_file = xenium_path / "cells.zarr.zip"

        if not cells_zarr_file.exists():
            raise FileNotFoundError(f"Could not find cells.zarr.zip at {cells_zarr_file}")

        # Check if we need to update anything
        needs_nucleus_map = self._nucleus_to_cell_map is None or overwrite
        needs_nucleus_count = self._nucleus_count is None or overwrite

        if not needs_nucleus_map and not needs_nucleus_count:
            print("nucleus_to_cell_map and nucleus_count are already set. No update needed.")
            return

        # Count unique cell and nucleus IDs in current boundaries (excluding background=0)
        cells_data = self._data["cells"][0] if isinstance(self._data["cells"], list) else self._data["cells"]
        ncells_current = len(da.unique(cells_data).compute()) - 1

        nnuclei_current = None
        if "nuclei" in self._data and self._data["nuclei"] is not None:
            nuclei_data = self._data["nuclei"][0] if isinstance(self._data["nuclei"], list) else self._data["nuclei"]
            nnuclei_current = len(da.unique(nuclei_data).compute()) - 1

        # Import helper functions from _io module
        from insitupy._io._xenium import (_read_nucleus_count_from_store,
                                          _read_nucleus_to_cell_map_from_store)

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
                print(f"Updated nucleus_to_cell_map with {len(nucleus_to_cell_map)} entries.")
            except Exception as e:
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
                    print(f"Updated nucleus_count for {len(nucleus_count)} cells.")
                else:
                    warn("nucleus_count not available in Xenium data.")
            except Exception as e:
                warn(f"Could not read nucleus_count from Xenium data: {e}")

        store.close()

    def add_boundaries(self,
                       cell_boundaries: Union[da.core.Array, np.ndarray],
                       pixel_size: Number, # required for boundaries that are saved as masks
                       nuclei_boundaries: Optional[Union[da.core.Array, np.ndarray]] = None,
                       #data: Optional[Union[Dict[str, da.core.Array]]],
                    #    labels: Optional[List[str]] = [],
                       overwrite: bool = False
                       ):
        if cell_boundaries is None:
            raise ValueError("cell_boundaries cannot be None.")

        # make sure the boundaries are a dask array
        #if not isinstance(cell_boundaries, da.core.Array) and cell_boundaries is not None:
        if isinstance(cell_boundaries, np.ndarray):
            cell_boundaries = da.from_array(cell_boundaries)

        #if not isinstance(nuclei_boundaries, da.core.Array) and nuclei_boundaries is not None:
        if isinstance(nuclei_boundaries, np.ndarray):
            nuclei_boundaries = da.from_array(nuclei_boundaries)

        if not (isinstance(cell_boundaries, da.core.Array) or isinstance(cell_boundaries, list) or cell_boundaries is None):
            raise ValueError("cell_boundaries must be a dask/numpy array or a list")

        if not (isinstance(nuclei_boundaries, da.core.Array) or isinstance(nuclei_boundaries, list) or nuclei_boundaries is None):
            raise ValueError("nuclei_boundaries must be a dask/numpy array, a list, or None")

        data = {
            "cells": cell_boundaries,
            "nuclei": nuclei_boundaries
        }

        # if isinstance(data, dict):
        # extract keys from dictionary
        # labels = data.keys()
        # data = data.values()
        # elif isinstance(data, list):
        #     if labels is None:
        #         raise ValueError("Argument 'labels' is None. If 'dataframes' is a list, 'labels' is required to be a list, too.")
        #     else:
        #         # make sure labels is a list
        #         labels = convert_to_list(labels)
        # else:
        #     data = convert_to_list(data)
        #     labels = convert_to_list(labels)
        #     #raise ValueError(f"Argument 'dataframes' has unknown file type ({type(data)}). Expected to be a list or dictionary.")

        #for l, df in zip(labels, data):
        for l, img in data.items():
            #if isinstance(img, pd.DataFrame) or isinstance(img, da.core.Array) or np.all([isinstance(elem, da.core.Array) for elem in img]):
            if l not in self._metadata or overwrite:
                # add to object
                self._data[l] = img
                self._metadata[l] = {}
                self._metadata[l]["pixel_size"] = pixel_size
            else:
                raise KeyError(f"Label '{l}' exists already in BoundariesData object. To overwrite, set 'overwrite' argument to True.")
            # else:
            #     print(f"Boundaries element `{l}` is neither a pandas DataFrame nor a Dask Array. Was not added.")

    def crop(self,
             cell_ids: List[str],
             xlim: Tuple[int, int],
             ylim: Tuple[int, int],
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
                # try:
                # get pixel size
                pixel_size = meta["pixel_size"]

                data = crop_dask_array_or_pyramid(
                    data=data,
                    xlim=xlim,
                    ylim=ylim,
                    pixel_size=pixel_size
                )
                # except InvalidDataTypeError:
                #     # filter dataframe
                #     data = data.loc[data["cell_id"].isin(cell_ids), :]

                #     # re-center to 0
                #     data["vertex_x"] -= xlim[0]
                #     data["vertex_y"] -= ylim[0]

            # add to object
            _self._data[n] = data

        if not inplace:
            return _self

    def convert_to_shapely_objects(self):
        for n in self._metadata.keys():
            print(f"Converting `{n}` to GeoPandas DataFrame with shapely objects.")
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
                print(f"Boundaries element `{n} was no Dataframe. Skipped.")

    def save(self,
             bound_file : Union[str, os.PathLike, Path] = "boundaries.zarr.zip",
             save_as_pyramid: bool = True,
             max_resolution: Optional[Number] = None,
             verbose: bool = False
             ):
        bound_file = Path(bound_file)
        suffix = bound_file.name.split(".", maxsplit=1)[-1]

        if suffix not in ["zarr", "zarr.zip"]:
            raise InvalidFileTypeError(allowed_types=[".zarr", ".zarr.zip"], received_type=suffix)

        zipped = suffix == "zarr.zip"

        # Use ExitStack to handle context manager differences between Zarr v2 and v3
        with ExitStack() as stack:
            dirstore = _get_zarr_store(bound_file, mode="w", zipped=zipped)

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


                    #if isinstance(bound_data, dask.array.core.Array):
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

                # save keys in insitupy metadata
                #metadata["boundaries"]["keys"].append(n)

            # save paths in insitupy metadata
            #metadata["boundaries"]["path"] = Path(relpath(bound_file, path)).as_posix()

            # save cell names
            self.cell_names.to_zarr(
                dirstore,
                component="cell_names",
                overwrite=True
                )

            if self._seg_mask_value is not None:
                self.seg_mask_value.to_zarr(dirstore, component="seg_mask_value", overwrite=True)

            # Save nucleus_to_cell_map if available (for multinucleated cell support)
            if self._nucleus_to_cell_map is not None:
                # Store as 2D array with columns [nucleus_index, cell_index]
                nucleus_map_arr = np.array([[k, v] for k, v in self._nucleus_to_cell_map.items()], dtype=np.int64)
                da.from_array(nucleus_map_arr).to_zarr(dirstore, component="nucleus_to_cell_map", overwrite=True)

            # Save nucleus_count if available
            if self._nucleus_count is not None:
                da.from_array(self._nucleus_count).to_zarr(dirstore, component="nucleus_count", overwrite=True)

        # # add version to metadata
        # metadata_to_save = self.metadata.copy()
        # metadata_to_save["version"] = __version__

        # # save metadata
        # write_dict_to_json(dictionary=metadata_to_save, file=path / ".boundariesdata")

class CellData(DeepCopyMixin):
    '''
    Data object containing an AnnData object and a boundary object which are kept in sync.
    '''
    def __init__(self,
               table: AnnData,
               boundaries: Optional[BoundariesData],
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
        logger.warning(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead."
        )
        return self._table

    @matrix.setter
    def matrix(self, value: AnnData):
        logger.warning(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead."
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
            raise ValueError(f"Table must be an AnnData object. Instead: {type(value)}.")

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
        return self._config

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def entries(self):
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
            xlim: Optional[Tuple[int, int]] = None,
            ylim: Optional[Tuple[int, int]] = None,
            shape: Optional[Union[Polygon, MultiPolygon]] = None,
            inplace: bool = False,
            verbose: bool = True
            ):

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
        # if xlim is not None and ylim is not None and shape is not None:
        #     warnings.warn("Both xlim/ylim and shape are provided. Shape will be used for cropping.")

        if shape is not None:
            if xlim is not None and ylim is not None:
                if verbose:
                    warnings.warn("Both xlim/ylim and shape are provided. Shape will be used for cropping.")

            # create shapely objects from cell coordinates
            cells = gpd.points_from_xy(cell_coords[:, 0], cell_coords[:, 1])

            # create a mask based on the shape
            mask = shape.contains(cells)

            # get bounding box of shape
            minx, miny, maxx, maxy = shape.bounds # (minx, miny, maxx, maxy)
            xlim = (minx, maxx)
            ylim = (miny, maxy)

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
        _self.boundaries.crop(
            cell_ids=_self.table.obs_names,
            xlim=xlim, ylim=ylim,
            inplace=True
            )

        # shift coordinates to correct for change of coordinates during cropping
        if shape is not None:
            minx, miny, _, _ = shape.bounds
            _self.shift(x=-minx, y=-miny)
        else:
            _self.shift(x=-xlim[0], y=-ylim[0])

        # sync the ids and names
        _self.sync()

        if not inplace:
            return _self


    def save(self,
             path: Union[str, os.PathLike, Path],
             boundaries_zipped: bool = False,
             #boundaries_as_pyramid: bool = True,
             max_resolution_boundaries: Optional[Number] = None,
             overwrite: bool = False
             ):

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
            if boundaries_zipped:
                bound_file = path / "boundaries.zarr.zip"
            else:
                bound_file = path / "boundaries.zarr"

            # save boundaries
            boundaries.save(bound_file, save_as_pyramid=True, max_resolution=max_resolution_boundaries)

            # add entry for boundaries to metadata
            celldata_metadata["boundaries"] = Path(relpath(bound_file, path)).as_posix()
            # bound_path.mkdir(parents=True, exist_ok=True) # create directory

        # add version to metadata
        celldata_metadata["version"] = __version__

        # add configurations
        if self._config is not None:
            celldata_metadata["config"] = self._config

        # save metadata
        write_dict_to_json(dictionary=celldata_metadata, file=path / ".celldata")


    def sync(self,
             verbose: bool = False):
        '''
        Function to synchronize table and boundaries of CellData.

        Procedure:
        1. Select table cell IDs
        2. Check if all table cell IDs are in boundaries
            - if not all are in boundaries, throw error saying that those will also be removed
        3. Select only table cell IDs which are also in boundaries and filter for them
        '''
        # get cell IDs from table
        table_cell_ids = pd.Index(self._table.obs_names.astype(str))

        if len(table_cell_ids.unique()) != len(table_cell_ids):
            raise ValueError("Table .obs_names must be unique to synchronize with boundaries.")

        if self._boundaries is None:
            print('No `boundaries` attribute found in CellData found.')
        else:
            boundaries = self._boundaries

            # create pandas series from seg_mask values and cell_names
            ds = pd.Series(
                data=boundaries.seg_mask_value.compute(),
                index=boundaries.cell_names.compute().astype(str)
                )

            # filter table for IDs that are available in boundaries
            table_mask_in_boundaries = table_cell_ids.isin(ds.index)

            if not np.any(table_mask_in_boundaries):
                raise ValueError("No matching values between `.boundaries.cell_names` and `.table.obs_names`. All table entries would get filtered out.")

            n_removed_table = int(np.sum(~table_mask_in_boundaries))
            if n_removed_table > 0:
                self._table = self._table[table_mask_in_boundaries, :].copy()
                table_cell_ids = pd.Index(self._table.obs_names.astype(str))

            # align boundary metadata to table order
            ds_aligned_to_table = ds.reindex(table_cell_ids)

            boundaries._seg_mask_value = da.from_array(np.array(ds_aligned_to_table.values, dtype=np.uint32))
            boundaries._cell_names = da.from_array(np.array(ds_aligned_to_table.index, dtype=str))

            # find the seg_mask_values which are not anymore present
            seg_mask_values_not_in_table = ds[~ds.index.isin(table_cell_ids)].values

            # extract boundaries
            cell_bounds = boundaries["cells"]
            nuc_bounds = boundaries["nuclei"]

            if isinstance(cell_bounds, list):
                if nuc_bounds is not None:
                    assert isinstance (nuc_bounds, list), "Cellular boundaries are a image pyramid but nuclear boundaries are not. Both need to be of the same type for the synchronization to work."
                for i, cell_bound in enumerate(cell_bounds):
                    removed_cells_mask = da.isin(cell_bound, seg_mask_values_not_in_table)
                    cell_bound[removed_cells_mask] = 0 # set all removed cells 0
                    if nuc_bounds is not None:
                        nuc_bounds[i][removed_cells_mask] = 0 # set all nuclei belong to the removed cells 0
            elif isinstance(cell_bounds, da.core.Array):
                if nuc_bounds is not None:
                    assert isinstance (nuc_bounds, da.core.Array), "Cellular boundaries are a dask array but nuclear boundaries are not. Both need to be of the same type for the synchronization to work."
                # set all non existent cell ids to zero
                removed_cells_mask = da.isin(cell_bounds, seg_mask_values_not_in_table)
                cell_bounds[removed_cells_mask] = 0 # set all removed cells 0

                if nuc_bounds is not None:
                    nuc_bounds[removed_cells_mask] = 0 # set all nuclei belong to the removed cells 0
            else:
                warnings.warn(f"Unknown data type for cellular boundaries: {type(cell_bounds)}. Need to be either a dask array or a list of dask arrays. Skipped synchronization of cell ids.")

            if verbose:
                n_removed_boundaries = int(np.sum(~ds.index.isin(table_cell_ids)))
                if n_removed_table > 0:
                    print(f"Filtered out {n_removed_table} table entries not present in boundaries.", flush=True)
                print(f"Filtered out {n_removed_boundaries} boundaries.", flush=True)

    def shift(self,
              x: Union[int, float],
              y: Union[int, float]
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
            print('No `boundaries` attribute found in CellData found.')
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

class MultiCellData(DeepCopyMixin):
    '''
    Data object containing multiple CellData objects.
    '''
    def __init__(self):
        self._layers: Dict[str, CellData] = dict()
        self._main_key: Optional[str] = None

    def __len__(self):
        return len(self._layers)

    def __repr__(self):
        if len(self._layers) > 0:
            if self._main_key is not None:
                indented_repr = self._layers[self._main_key].__repr__().replace('\n', f'\n{tf.SPACER}')
                repr = (
                    f"{tf.Bold}MultiCellData with main layer{tf.ResetAll} '{self._main_key}'\n"
                    f"{tf.SPACER}{indented_repr}"
                )

            non_main_keys = [f"'{k}'" for k in self._layers.keys() if k != self._main_key]
            if len(non_main_keys) > 0:
                repr += f"\n\nAdditional layers with keys: {', '.join(non_main_keys)}"
        else:
            repr = "empty"
        return repr

    def __getitem__(self, key):
        # if key == "main":
        #     return self._data.get(self._key_main)
        # else:
        return self._layers.get(key)

    def __setitem__(self, key: str, item: CellData):
        if isinstance(item, CellData):
            # check whether this is the first data that is added
            is_first_key = True if len(self._layers) == 0 else False

            # add data
            self._layers[key] = item

            # set key as main key if it is the first data to be added to the layer
            if is_first_key:
                self.main_key = key
        else:
            raise ValueError(f"Item must be of type CellData. Instead: {type(item)}.")

    def __delitem__(self, key: str):
        if key in self._layers.keys():
            if key == self._main_key:
                raise KeyError(f"Cannot delete the main key '{self._main_key}'. Please use `set_main()` to set another key as main first.")
            layer = self._layers[key]
            close_method = getattr(layer, "close", None)
            if callable(close_method):
                close_method()
            del self._layers[key]
        else:
            raise KeyError(f"Key '{key}' not found in MultiCellData.")

    def close(self):
        """Close underlying resources owned by all contained CellData layers."""
        for layer in self._layers.values():
            close_method = getattr(layer, "close", None)
            if callable(close_method):
                close_method()

    @property
    def layers(self):
        return self._layers

    @property
    def matrix(self):
        logger.warning(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead."
        )
        try:
            return self._layers[self._main_key].table
        except KeyError:
            print("MultiCellData object is empty.")
            return None
        except AttributeError:
            print("No matrix available.")
            return None

    @property
    def table(self):
        """Alias for matrix property. This is the preferred name going forward."""
        try:
            return self._layers[self._main_key].table
        except KeyError:
            print("MultiCellData object is empty.")
            return None
        except AttributeError:
            print("No table available.")
            return None

    @property
    def boundaries(self):
        try:
            return self._layers[self._main_key].boundaries
        except KeyError:
            print("MultiCellData object is empty.")
            return None
        except AttributeError:
            print("No boundaries available.")
            return None

    @property
    def main_key(self):
        return self._main_key

    @main_key.setter
    def main_key(self, value: str):
        if value not in self._layers.keys():
            raise ValueError(f"Such layer does not exist.")
        self._main_key = value

    @property
    def is_empty(self):
        return len(self._layers) == 0

    def add_celldata(self,
                     cd: CellData,
                     key: str,
                     is_main: bool = False,
                     overwrite: bool = False):
        if not isinstance(cd, CellData):
            raise ValueError(f"cd must be of type CellData. Instead: {type(cd)}.")

        if key in self._layers.keys():
            if not overwrite:
                raise KeyError(
                    f"Key '{key}' already exists in MultiCellData. "
                    f"Set overwrite=True to replace it."
                )
            print(f"Overwriting '{key}' in MultiCellData.")
        self._layers[key] = cd
        if is_main:
            self._main_key = key

    def add_proseg(self,
                   path: Union[str, os.PathLike, Path],
                   counts_file: Optional[str] = None,
                   cell_metadata_file: Optional[str] = None,
                   polygons_file: Optional[str] = None,
                   pixel_size: Number = 1,
                   key: str = "proseg",
                   is_main: bool = False,
                   overwrite: bool = False
                   ):
        """
            Adds output of Proseg https://github.com/dcjones/proseg segmentation to the object.

            Args:
                path (Union[str, os.PathLike, Path]): Path to proseg output. Can be either:
                    - A directory containing individual files (counts, metadata, and polygon files) for legacy proseg output
                    - A .zarr directory containing a SpatialData object with proseg results.
                      The SpatialData object should contain:
                      * tables['table']: AnnData with counts and cell metadata
                      * shapes['cell_boundaries']: GeoDataFrame with cell polygons
                counts_file (Optional[str]): Name of the counts file. Only used with legacy directory input.
                cell_metadata_file (Optional[str]): Name of the cell metadata file. Only used with legacy directory input.
                polygons_file (Optional[str]): Name of the polygons file. Only used with legacy directory input.
                pixel_size (float): Size of the pixel for scaling.
                key (str, optional): Key to store the data. Defaults to "proseg".
                is_main (bool, optional): Flag to indicate if this is the main data. Defaults to False.
                overwrite (bool, optional): If True, allow overwriting an existing key. Defaults to False.
        """
        from ._segmentations import _read_proseg, _read_proseg_from_spatialdata

        # Convert to Path object
        path = Path(path)

        # Check if this is a zarr file containing spatialdata
        if path.suffix == '.zarr' or path.name.endswith('.zarr'):
            # Read SpatialData from zarr
            try:
                import spatialdata
            except ImportError:
                raise ImportError(
                    "Reading proseg output from zarr requires the spatialdata package. "
                    "Please install with `pip install spatialdata`."
                )

            # File-specific parameters are ignored for spatialdata input
            if any([counts_file, cell_metadata_file, polygons_file]):
                import warnings
                warnings.warn(
                    "File-specific parameters (counts_file, cell_metadata_file, polygons_file) "
                    "are ignored when reading from .zarr (SpatialData) format.",
                    UserWarning
                )

            # Read spatialdata from zarr
            sdata = spatialdata.read_zarr(path)

            # Process spatialdata object
            adata, boundaries_mask, cell_names, seg_mask_value = _read_proseg_from_spatialdata(
                sdata,
                pixel_size=pixel_size
            )
            del sdata  # free spatialdata object before heavy allocations
        else:
            # Legacy path-based input (directory with individual files)
            adata, boundaries_mask, cell_names, seg_mask_value = _read_proseg(
                path, counts_file=counts_file, cell_metadata_file=cell_metadata_file,
                polygons_file=polygons_file, pixel_size=pixel_size
            )

        # generate boundaries data object
        boundaries = BoundariesData(
            cell_names=cell_names,
            seg_mask_value=seg_mask_value
            )

        # add cellular boundaries
        boundaries.add_boundaries(
            cell_boundaries=boundaries_mask,
            pixel_size=pixel_size
            )
        del boundaries_mask  # no longer needed after add_boundaries

        # Create cell data and add to object
        celldata = CellData(table=adata, boundaries=boundaries)
        del adata, boundaries  # now owned by celldata

        self.add_celldata(cd=celldata, key=key, is_main=is_main, overwrite=overwrite)


    def add_baysor(
                    self,
                    xd: Union[str, os.PathLike, Path], # XeniumRanger output
                    path: Union[str, os.PathLike, Path], # baysor output
                    counts_file: Optional[str] = None,
                    cell_metadata_file: Optional[str] = None,
                    polygons_file: Optional[str] = None,
                    pixel_size: Number = 1,
                    key: str = "baysor",
                    is_main: bool = False,
                    overwrite: bool = False
                    ):

        from ._segmentations import _read_baysor

        # Convert to Path object
        path = Path(path)
        print(path)
        # Legacy path-based input (directory with individual files)
        adata, boundaries_mask, cell_names, seg_mask_value = _read_baysor(path, xd,
            counts_file=counts_file, cell_metadata_file=cell_metadata_file,
            polygons_file=polygons_file, pixel_size=pixel_size
            )

        # generate boundaries data object
        boundaries = BoundariesData(
            cell_names=cell_names,
            seg_mask_value=seg_mask_value
            )

        # add cellular boundaries
        boundaries.add_boundaries(
            #data={f"cells": img},
            cell_boundaries=boundaries_mask,
            pixel_size=pixel_size
            )
        del boundaries_mask  # no longer needed after add_boundaries

        # Create cell data and add to object
        celldata = CellData(table=adata, boundaries=boundaries)
        del adata, boundaries  # now owned by celldata

        self.add_celldata(cd=celldata, key=key, is_main=is_main, overwrite=overwrite)


    def crop(self,
            xlim: Optional[Tuple[int, int]] = None,
            ylim: Optional[Tuple[int, int]] = None,
            shape: Optional[Union[Polygon, MultiPolygon]] = None,
            inplace: bool = False,
            verbose: bool = True):

        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        for key in _self._layers.keys():
            _self._layers[key].crop(
                xlim=xlim,
                ylim=ylim,
                shape=shape,
                inplace=True,
                verbose=verbose)

        if not inplace:
            return _self

    def keys(self):
        return self._layers.keys()

    def save(self,
             path: Union[str, os.PathLike, Path],
             boundaries_zipped: bool = False,
             overwrite: bool = False,
             max_resolution_boundaries: Optional[Number] = None
             ):

        path = Path(path)
        multicelldata_metadata = {"key_main": self._main_key, "all_keys": list(self._layers.keys())}
        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)
        for key in self._layers.keys():
            save_path = path / key
            self._layers[key].save(
                path=save_path,
                boundaries_zipped=boundaries_zipped,
                max_resolution_boundaries=max_resolution_boundaries,
                overwrite=overwrite)

        # add version to metadata
        multicelldata_metadata["version"] = __version__

        # save metadata
        write_dict_to_json(dictionary=multicelldata_metadata, file=path / ".multicelldata")

    def set_main(self, key):
        if key in self.keys():
            self._main_key = key

    def sync(self):
        current_keys = self._layers.keys()
        for key in current_keys:
            self._layers[key].sync()


class ImageData(DeepCopyMixin):
    '''
    Object to read and load images.
    '''
    def __init__(self,
                 img_files: List[str] = None,
                 img_names: List[str] = None,
                 pixel_size: float = None,
                 ):

        # iterate through files and load them
        self._names = []
        self._metadata = {}
        self._data = {}

        if img_files is not None:
            # convert arguments to lists
            img_files = convert_to_list(img_files)
            img_names = convert_to_list(img_names)

            for n, f in zip(img_names, img_files):
                #impath = path / f
                self.add_image(
                    image=f,
                    channel_names=n,
                    axes=None,
                    pixel_size=pixel_size,
                    ome_meta=None,
                    )

    def __repr__(self):
        if len(self._data) > 0:
            # Calculate the maximum length of the key names for alignment
            max_key_len = max(len(n) for n in self._metadata.keys())
            pad = 3
            repr_strings = [f"{tf.Bold}'{n}':{tf.ResetAll}{' ' * (max_key_len - len(n) + pad)}{metadata['shape']}" for n,metadata in self._metadata.items()]
            s = "\n".join(repr_strings)
        else:
            s = "empty"
        return s

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data.get(key)

    def __delitem__(self, key: str):
        """Remove an image by name using del syntax.

        Args:
            key: Name of the image to remove.

        Raises:
            KeyError: If the image name is not found.

        Example:
            >>> del img_data['DAPI']
        """
        if key not in self._names:
            raise KeyError(f"Image '{key}' not found in ImageData. Available images: {self._names}")

        del self._data[key]
        self._names.remove(key)
        self._metadata.pop(key, None)

    def __contains__(self, key):
        return key in self.keys()

    def keys(self):
        return self._data.keys()

    @property
    def metadata(self):
        return self._metadata

    @property
    def names(self):
        return self._names

    @property
    def is_empty(self):
        return len(self._data) == 0

    def add_image(
        self,
        image: Union[da.core.Array, np.ndarray, str, os.PathLike, Path],
        channel_names: Optional[Union[str, List[str]]] = None,
        axes: Optional[str] = None, # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
        pixel_size: Optional[Number] = None,
        ome_meta: Optional[dict] = {},
        is_rgb: Optional[bool] = None,
        transformation_matrix: Optional[Union[np.ndarray, str, os.PathLike, Path]] = None,
        reference_image: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True
        ):
        """Add image data to the ImageData object.

        For multi-channel images ("CYX" axes), each channel is added as a separate entry.
        RGB images ("YXS" axes with 3 channels) are kept together as a single entry.

        Args:
            image: Either a dask/numpy array or a path to an image file.
            channel_names: Name identifier(s) for the image. For multi-channel images (CYX), provide
                a list of names (one per channel). For single-channel or RGB images, provide
                a string. If None, channel names are automatically extracted from OME metadata.
            axes: Axis specification (e.g., 'YX', 'CYX', 'YXS'). Required if image is an array.
            pixel_size: Physical pixel size in µm/pixel. Required if image is an array.
            ome_meta: OME metadata dictionary.
            is_rgb: Whether the image is RGB. Auto-detected if None.
            transformation_matrix: Optional affine transformation matrix to apply to the image.
                Can be a 2x3 or 3x3 numpy array, or a path to a CSV/Excel file containing the matrix.
                The matrix should be in the form:
                [[a, b, xoff],
                 [d, e, yoff]]
                or with [0, 0, 1] as third row.
            reference_image: Name of the reference image in this ImageData object. If provided,
                the transformation matrix offsets are assumed to be in pixel coordinates at the
                reference image's resolution and will be converted to physical coordinates.
                The pixel size and output canvas size are automatically retrieved from the
                reference image's metadata. If not provided, the transformation matrix is assumed
                to be in physical coordinates (µm) and the output size matches the input image.
            overwrite: If True, overwrite existing image(s) with the same name(s).
            verbose: If True, print status messages.

        Example:
            >>> # Add single-channel image
            >>> img_data.add_image(image_array, name='DAPI', axes='YX', pixel_size=0.5)

            >>> # Add multi-channel IF image (splits into separate channels)
            >>> img_data.add_image(
            ...     multichannel_image,
            ...     channel_names=['DAPI', 'CD45', 'PanCK'],  # One name per channel
            ...     axes='CYX',
            ...     pixel_size=0.5
            ... )

            >>> # Add multi-channel with auto-detected names from OME metadata
            >>> img_data.add_image(ome_tiff_path, name=None)  # Extracts channel names automatically

            >>> # Add RGB image (keeps all 3 channels together)
            >>> img_data.add_image(
            ...     rgb_image,
            ...     name='HE',
            ...     axes='YXS',
            ...     pixel_size=0.5
            ... )
        """
        # Load image data
        if isinstance(image, da.core.Array) or isinstance(image, np.ndarray):
            assert axes is not None, "If `image` is numpy or dask array, `axes` needs to be set."
            assert pixel_size is not None, "If `image` is numpy or dask array, `pixel_size` needs to be set."

            try:
                # convert to dask array before addition
                img = da.from_array(image)
            except ValueError:
                # in this case the array was already a dask array
                img = image
            filename = None

        elif Path(str(image)).exists():
            # read path
            image = Path(image)
            image = image.resolve() # resolve relative path
            filename = image.name
            img, ome_meta, axes, pixel_size = read_image(image) # returns image pyramid as list of dask arrays if possible
        else:
            raise ValueError(f"`image` is neither a dask array nor an existing path. Instead: {type(image)}")

        # Transpose image to standard axis order (YX, CYX, or YXS)
        img, axes = _transpose_to_standard_axes(img, axes)

        # Get image shape
        img_shape = img[0].shape if isinstance(img, list) else img.shape

        # Determine if this is a multi-channel image that should be split
        axes_config = ImageAxes(axes)
        is_multichannel = axes == "CYX" or (axes_config.C is not None and axes != "YXS")

        # Handle channel names for multi-channel images
        if is_multichannel:
            n_channels = img_shape[axes_config.C]

            # Extract or validate channel names
            if channel_names is None:
                # Try to extract from OME metadata
                if ome_meta and 'Image' in ome_meta:
                    try:
                        channels_info = ome_meta['Image']['Pixels']['Channel']
                        # Handle both single channel (dict) and multiple channels (list)
                        if isinstance(channels_info, dict):
                            channel_names = [channels_info.get('Name', f'Channel_0')]
                        else:
                            channel_names = [ch.get('Name', f'Channel_{i}') for i, ch in enumerate(channels_info)]

                        if verbose:
                            print(f"Extracted channel names from OME metadata: {channel_names}")
                    except (KeyError, TypeError):
                        # Fallback to numbered channels
                        channel_names = [f'Channel_{i}' for i in range(n_channels)]
                        if verbose:
                            print(f"Could not extract channel names from OME metadata. Using: {channel_names}")
                else:
                    # Fallback to numbered channels
                    channel_names = [f'Channel_{i}' for i in range(n_channels)]
                    if verbose:
                        print(f"No OME metadata available. Using channel names: {channel_names}")

            elif isinstance(channel_names, str):
                raise ValueError(
                    f"Multi-channel image detected (axes='{axes}', {n_channels} channels) but `name` is a string. "
                    f"Please provide a list of channel names with length {n_channels}, or set name=None to "
                    f"automatically extract channel names from OME metadata."
                )

            elif isinstance(channel_names, list):
                if len(channel_names) != n_channels:
                    raise ValueError(
                        f"Length of `name` list ({len(channel_names)}) does not match number of channels ({n_channels}). "
                        f"Axes: '{axes}', Image shape: {img_shape}"
                    )
                channel_names = channel_names

            else:
                raise TypeError(f"`name` must be a string, list of strings, or None. Got: {type(channel_names)}")

            # Split channels and add each separately
            if verbose:
                print(f"Splitting multi-channel image into {n_channels} separate channels...")

            for i, ch_name in enumerate(channel_names):
                # Extract single channel
                if isinstance(img, list):
                    # Handle image pyramid
                    channel_img = [np.take(level, i, axis=axes_config.C) for level in img]
                else:
                    channel_img = np.take(img, i, axis=axes_config.C)

                # Determine new axes (remove channel dimension)
                channel_axes = "YX"

                # Add this channel as a separate image (recursive call for single channel)
                self._add_single_image(
                    img=channel_img,
                    name=ch_name,
                    axes=channel_axes,
                    pixel_size=pixel_size,
                    filename=filename,
                    ome_meta=ome_meta,
                    is_rgb=False,
                    transformation_matrix=transformation_matrix,
                    reference_image=reference_image,
                    overwrite=overwrite,
                    verbose=verbose
                )

        else:
            # Single-channel or RGB image - add as-is
            if isinstance(channel_names, list):
                raise ValueError(
                    f"Single-channel or RGB image (axes='{axes}') but `name` is a list. "
                    f"Please provide a single string name for this image."
                )

            if channel_names is None:
                # For single-channel images, try to extract name from OME or use default
                if ome_meta and 'Image' in ome_meta:
                    try:
                        channel_names = ome_meta['Image'].get('Name', 'Image_0')
                        if verbose:
                            print(f"Extracted image name from OME metadata: {channel_names}")
                    except (KeyError, TypeError):
                        channel_names = 'Image_0'
                else:
                    channel_names = 'Image_0'

            # Add single image
            self._add_single_image(
                img=img,
                name=channel_names,
                axes=axes,
                pixel_size=pixel_size,
                filename=filename,
                ome_meta=ome_meta,
                is_rgb=is_rgb,
                transformation_matrix=transformation_matrix,
                reference_image=reference_image,
                overwrite=overwrite,
                verbose=verbose
            )

    def _add_single_image(
        self,
        img: Union[da.core.Array, np.ndarray, List],
        name: str,
        axes: str,
        pixel_size: Number,
        filename: Optional[str],
        ome_meta: dict,
        is_rgb: Optional[bool],
        transformation_matrix: Optional[Union[np.ndarray, str, os.PathLike, Path]],
        reference_image: Optional[str],
        overwrite: bool,
        verbose: bool
    ):
        """Internal method to add a single image (used by add_image after channel splitting)."""

        # Check if name already exists
        if name in self._names:
            if not overwrite:
                print(f"`ImageData` object contains already an image with name '{name}'. Image is not added.") if verbose else None
                return
            else:
                # remove attribute with current name
                del self._data[name]
                # remove from name list and metadata
                self._names = [elem for elem in self._names if elem != name]
                self._metadata.pop(name, None)

        # Apply transformation if provided
        if transformation_matrix is not None:
            if verbose:
                print(f"Applying transformation to image '{name}'...")

            # Determine reference_pixel_size and output_size from reference_image if provided
            reference_pixel_size = None
            output_size = None

            if reference_image is not None:
                if reference_image not in self._names:
                    raise ValueError(
                        f"Reference image '{reference_image}' not found in ImageData. "
                        f"Available images: {self._names}"
                    )
                reference_pixel_size = self._metadata[reference_image]['pixel_size']

                # Get output_size from reference image
                ref_shape = self._metadata[reference_image]['shape']
                ref_axes = self._metadata[reference_image]['axes']
                ref_axes_config = ImageAxes(ref_axes)

                # Get height and width from reference image
                ref_height = ref_shape[ref_axes_config.Y]
                ref_width = ref_shape[ref_axes_config.X]

                # Convert to physical coordinates (µm)
                output_size = (
                    ref_height * reference_pixel_size,
                    ref_width * reference_pixel_size
                )

                if verbose:
                    print(f"Using reference image '{reference_image}' (pixel size: {reference_pixel_size} µm/pixel, "
                          f"shape: {ref_height}x{ref_width} pixels = {output_size[0]:.1f}x{output_size[1]:.1f} µm)")

            # Create a temporary ImageData object to use the transform method
            temp_img_data = ImageData()
            temp_img_data._data[name] = img
            temp_img_data._names = [name]
            temp_img_data._metadata[name] = {
                'pixel_size': pixel_size,
                'axes': axes
            }

            # Apply transformation
            # source_pixel_size = pixel_size (the image being added)
            temp_img_data.transform(
                transformation_matrix=transformation_matrix,
                source_pixel_size=pixel_size,
                reference_pixel_size=reference_pixel_size,
                output_size=output_size,
                inplace=True,
                verbose=verbose
            )

            # Get transformed image
            img = temp_img_data._data[name]

            # Update axes if needed (transform maintains axes)
            axes = temp_img_data._metadata[name]['axes']

        # set attribute and add names to object
        self._data[name] = img
        self._names.append(name)

        # retrieve metadata
        img_shape = img[0].shape if isinstance(img, list) else img.shape

        # save metadata
        self._metadata[name] = {}
        self._metadata[name]["filename"] = filename
        self._metadata[name]["shape"] = img_shape  # store shape
        self._metadata[name]["axes"] = axes
        self._metadata[name]["OME"] = ome_meta

        self._metadata[name]['pixel_size'] = pixel_size

        # check whether the image is RGB or not
        if is_rgb is None:
            if len(img_shape) == 3:
                channels = img_shape[2]
                if channels == 3:
                    self._metadata[name]["rgb"] = True
                else:
                    self._metadata[name]["rgb"] = False
            elif len(img_shape) == 2:
                self._metadata[name]["rgb"] = False
            else:
                raise ValueError(f"Unknown image shape: {img_shape}")
        else:
            self._metadata[name]["rgb"] = is_rgb

    def remove_image(
        self,
        names: Union[str, List[str]],
        verbose: bool = True
    ):
        """Remove one or more images from the ImageData object.

        Args:
            names: Name or list of names of images to remove.
            verbose: If True, print status messages for each removed image.

        Raises:
            KeyError: If any of the specified image names is not found.

        Example:
            >>> # Remove single image
            >>> img_data.remove_image('DAPI')

            >>> # Remove multiple images
            >>> img_data.remove_image(['DAPI', 'CD45', 'PanCK'])

            >>> # Remove without verbose output
            >>> img_data.remove_image('DAPI', verbose=False)
        """
        names = convert_to_list(names)

        for name in names:
            if name not in self._names:
                raise KeyError(f"Image '{name}' not found in ImageData. Available images: {self._names}")

            del self[name]  # Uses __delitem__

            if verbose:
                print(f"Removed image '{name}'")

    def load(self,
             which: Union[List[str], str] = "all"
             ):
        '''
        Load images into memory.
        '''
        if which == "all":
            which = self._img_names

        # make sure which is a list
        which = convert_to_list(which)
        for n in which:
            img_loaded = self[n].compute()
            self._data[n] = img_loaded

    def crop(self,
             xlim: Optional[Tuple[int, int]],
             ylim: Optional[Tuple[int, int]],
             inplace: bool = False
             ):
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()
        # extract names from metadata
        names = list(_self._metadata.keys())
        for n in names:
            # extract the image pyramid
            img_data = _self[n]

            # extract pixel size
            pixel_size = _self._metadata[n]['pixel_size']

            cropped_img_data = crop_dask_array_or_pyramid(
                data=img_data,
                xlim=xlim,
                ylim=ylim,
                pixel_size=pixel_size
            )

            # save cropping properties in metadata
            _self._metadata[n]["cropping_xlim"] = xlim
            _self._metadata[n]["cropping_ylim"] = ylim

            try:
                _self._metadata[n]["shape"] = cropped_img_data.shape
            except AttributeError:
                _self._metadata[n]["shape"] = cropped_img_data[0].shape

            # add cropped pyramid to object
            _self._data[n] = cropped_img_data

        if not inplace:
            return _self

    def save(self,
             output_folder: Union[str, os.PathLike, Path],
             keys_to_save: Optional[str] = None,
             as_zarr: bool = True,
             zipped: bool = False,
             save_pyramid: bool = True,
             compression: Literal['jpeg', 'LZW', 'jpeg2000', 'ZLIB', None] = 'ZLIB', # jpeg2000 or ZLIB are recommended in the Xenium documentation - ZLIB is faster
             return_savepaths: bool = False,
             overwrite: bool = False,
             max_resolution: Optional[Number] = None, # in µm per pixel
             verbose: bool = False
             ):
        """
        Save images to the specified output folder in either Zarr or OME-TIFF format.

        Args:
            output_folder (Union[str, os.PathLike, Path]): The directory where images will be saved.
            keys_to_save (Optional[str]): Specific keys of images to save. If None, all images are saved.
            as_zarr (bool): If True, save images in Zarr format. Otherwise, save as OME-TIFF.
            zipped (bool): If True and saving as Zarr, compress the Zarr files into zip archives.
            save_pyramid (bool): If True, save image pyramids (only applicable for Zarr format).
            compression (Literal['jpeg', 'LZW', 'jpeg2000', 'ZLIB', None]): Compression method for OME-TIFF files.
            return_savepaths (bool): If True, return the paths of the saved files.
            overwrite (bool): If True, overwrite existing files in the output folder. Default is False.
            max_resolution (Optional[Number]): Maximum resolution for images in µm per pixel. If the pixel size of an image is larger than `max_resolution`, the image is downscaled. Default is None.
            verbose (bool): If True, print status messages during saving. Default is True.

        Returns:
            Optional[Dict[str, Path]]: A dictionary mapping image keys to their save paths if `return_savepaths` is True. Otherwise, returns None.

        Raises:
            FileExistsError: If `overwrite` is False and a file with the same name already exists.

        """
        output_folder = Path(output_folder)

        if keys_to_save is None:
            keys_to_save = list(self._metadata.keys())
        else:
            keys_to_save = convert_to_list(keys_to_save)

        # create output directory (allow saving to existing directories)
        output_folder.mkdir(parents=True, exist_ok=True)

        if return_savepaths:
            savepaths = {}

        for name, img_metadata in self._metadata.items():
            if name in keys_to_save:
                # extract image
                img = self[name]
                new_img_metadata = img_metadata.copy()

                axes = new_img_metadata['axes']
                pixel_size = new_img_metadata['pixel_size'] # in µm per pixel

                if max_resolution is not None:
                    if max_resolution == pixel_size:
                        warnings.warn(f"`max_pixel_size` ({max_resolution}) equal to `pixel_size` ({pixel_size}). Skipped resizing.")
                        pass
                    if max_resolution < pixel_size:
                        warnings.warn(f"`max_pixel_size` ({max_resolution}) smaller than `pixel_size` ({pixel_size}). Skipped resizing.")
                        pass
                    else:
                        # downscale image
                        if isinstance(img, list):
                            img = img[0]
                        downscale_factor = max_resolution / pixel_size

                        if verbose:
                            print(f"Downscale image to {max_resolution} µm per pixel by factor {downscale_factor}")
                        img = resize_image(img, scale_factor=1/downscale_factor, axes=axes)
                        img = da.from_array(img)

                        # change metadata
                        new_img_metadata['pixel_size'] = max_resolution
                        try:
                            new_img_metadata['OME']['Image']['Pixels']['PhysicalSizeX'] = str(max_resolution)
                        except KeyError:
                            new_img_metadata['OME']['PhysicalSizeX'] = str(max_resolution)

                        try:
                            new_img_metadata['OME']['Image']['Pixels']['PhysicalSizeY'] = str(max_resolution)
                        except KeyError:
                            new_img_metadata['OME']['PhysicalSizeY'] = str(max_resolution)

                if as_zarr:
                    # generate filename
                    if zipped:
                        #filename = Path(img_metadata["file"]).name.split(".")[0] + ".zarr.zip"
                        filename = name + ".zarr.zip"
                    else:
                        # filename = Path(img_metadata["file"]).name.split(".")[0] + ".zarr"
                        filename = name + ".zarr"

                    # write to zarr
                    img_path = output_folder / filename

                    # check if file exists and handle overwrite
                    if img_path.exists() and not overwrite:
                        logger.warning(f"Image '{name}' already exists at {img_path}. Skipping. Set `overwrite=True` to overwrite.")
                        continue

                    # Safety check: prevent overwriting a zarr store that the
                    # dask array is lazily reading from.  Writing to the same
                    # store would destroy the source data before it is read,
                    # resulting in zeros / corrupted data.
                    #
                    # The check is only relevant for arrays that are lazily
                    # backed by a *Zarr* store on disk.  In-memory arrays
                    # (numpy-backed dask arrays) and arrays loaded from
                    # non-Zarr formats (TIFF, HDF5, npy) are safe to save
                    # into a Zarr target because the source is either
                    # entirely in memory or in a different format/location.
                    if overwrite and img_path.exists() and is_from_zarr_disk(img):
                        source_path = get_zarr_source_path(img)
                        target_path = img_path.resolve()
                        if source_path is not None and source_path == target_path:
                            logger.warning(
                                f"Skipping image '{name}': the dask array is lazily backed by the "
                                f"same Zarr store at {img_path}. Writing would destroy the source "
                                f"data before it is read. To update this image, first load it into "
                                f"memory (e.g. via `.persist()` or `.compute()`), or save it under "
                                f"a different name."
                            )
                            continue
                        elif source_path is None:
                            logger.warning(
                                f"Skipping image '{name}': the dask array appears to be backed "
                                f"by a Zarr store but the source path could not be determined. "
                                f"Cannot verify it differs from the target path {img_path}. "
                                f"Overwriting could destroy the source data. To update this image, "
                                f"first load it into memory (e.g. via `.persist()` or `.compute()`), "
                                f"or save it under a different name."
                            )
                            continue

                    write_zarr(image=img, file=img_path,
                               img_metadata=new_img_metadata,
                               save_pyramid=save_pyramid,
                               axes=axes, verbose=verbose,
                               overwrite=overwrite
                               )
                else:
                    # get file name for saving
                    #filename = Path(img_metadata["file"]).name.split(".")[0] + ".ome.tif"
                    filename = name + ".ome.tif"

                    # check if file exists and handle overwrite
                    img_path = output_folder / filename
                    if img_path.exists() and not overwrite:
                        warnings.warn(f"Image '{name}' already exists at {img_path}. Skipping. Set `overwrite=True` to overwrite.")
                        continue

                    # retrieve image metadata for saving
                    photometric = 'rgb' if new_img_metadata['rgb'] else 'minisblack'


                    # retrieve OME metadata
                    ome_meta_to_retrieve = ["SignificantBits", "PhysicalSizeX", "PhysicalSizeY",
                                            "PhysicalSizeXUnit", "PhysicalSizeYUnit"]

                    try:
                        pixel_meta = new_img_metadata["OME"]["Image"]["Pixels"]
                    except KeyError:
                        pixel_meta = new_img_metadata["OME"]

                    selected_metadata = {key: pixel_meta[key] for key in ome_meta_to_retrieve if key in pixel_meta}

                    # write images as OME-TIFF
                    write_ome_tiff(image=img, file=img_path,
                                photometric=photometric, axes=axes,
                                compression=compression,
                                metadata=selected_metadata, overwrite=overwrite,
                                verbose=verbose
                                )

                if return_savepaths:
                    # collect savepaths
                    savepaths[name] = output_folder / filename

        if return_savepaths:
            return savepaths

    def transform(
        self,
        transformation_matrix: Union[np.ndarray, str, os.PathLike, Path],
        source_pixel_size: Optional[Number] = None,
        reference_pixel_size: Optional[Number] = None,
        output_size: Optional[Tuple[Number, Number]] = None,
        inplace: bool = False,
        verbose: bool = False
    ):
        """Apply an affine transformation to all images in the ImageData object.

        Transforms all images using the provided affine transformation matrix.
        The transformation is applied consistently across images of different
        resolutions by converting to physical coordinates.

        Args:
            transformation_matrix: Either a 2x3 or 3x3 numpy array representing
                the affine transformation matrix, or a path to a CSV/Excel file
                containing the matrix. The matrix should be in the form:
                [[a, b, xoff],
                 [d, e, yoff]]
                or
                [[a, b, xoff],
                 [d, e, yoff],
                 [0, 0, 1]]
            source_pixel_size: Pixel size (in µm/pixel) of the source image from
                which the transformation matrix was derived. Only used if
                reference_pixel_size is also provided. If None, it is assumed to
                be equal to reference_pixel_size (i.e., no scaling of the linear
                transformation component is performed).
            reference_pixel_size: Pixel size (in µm/pixel) of the reference image
                used during registration. If provided, the transformation matrix
                offsets (xoff, yoff) are assumed to be in pixel coordinates of the
                reference image and will be converted to physical coordinates (µm).
                If None, the matrix offsets are assumed to already be in physical
                coordinates (µm). This is important when the transformation matrix
                was computed in pixel space for a specific image resolution.
            output_size: Tuple of (height, width) in physical coordinates (µm)
                specifying the desired output canvas size, following NumPy's
                (rows, cols) convention. If None, the output size will match the
                input image size. Use this when transforming images to align with
                a target image of different dimensions.
            inplace: If True, modify the object in place. Otherwise, return a
                transformed copy. Defaults to False.
            verbose: If True, print status messages. Defaults to False.

        Returns:
            ImageData: Transformed ImageData object if inplace=False, else None.

        Raises:
            ValueError: If the transformation matrix has invalid dimensions or format.
            FileNotFoundError: If the provided path does not exist.

        Example:
            >>> # Matrix in physical coordinates (µm)
            >>> images.transform(transformation_matrix=matrix)

            >>> # Matrix in pixel coordinates, computed for reference at 0.2125 µm/pixel
            >>> images.transform(
            ...     transformation_matrix=matrix,
            ...     reference_pixel_size=0.2125
            ... )

            >>> # Transform to match a target image size (3000 µm height x 4000 µm width)
            >>> images.transform(
            ...     transformation_matrix=matrix,
            ...     output_size=(3000, 4000)  # (height, width) in µm
            ... )
        """
        import cv2

        _self = self if inplace else self.copy()

        # Load transformation matrix if it's a file path
        if isinstance(transformation_matrix, (str, os.PathLike, Path)):
            transformation_matrix = Path(transformation_matrix)
            if not transformation_matrix.exists():
                raise FileNotFoundError(f"Transformation matrix file not found: {transformation_matrix}")

            # Read file based on extension
            if transformation_matrix.suffix.lower() in ['.csv', '.txt']:
                M = pd.read_csv(transformation_matrix, header=None).values
            elif transformation_matrix.suffix.lower() in ['.xlsx', '.xls']:
                M = pd.read_excel(transformation_matrix, header=None).values
            else:
                raise ValueError(f"Unsupported file format: {transformation_matrix.suffix}. Use .csv, .txt, .xlsx, or .xls")
        else:
            M = np.array(transformation_matrix)

        # Validate matrix dimensions
        if M.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Transformation matrix must be 2x3 or 3x3, got shape {M.shape}. "
                f"Expected format:\n"
                f"[[a, b, xoff],\n"
                f" [d, e, yoff]] or with [0, 0, 1] as third row."
            )

        # Extract transformation parameters
        if M.shape == (3, 3):
            # Validate that the third row is [0, 0, 1]
            if not np.allclose(M[2, :], [0, 0, 1]):
                raise ValueError("For 3x3 matrix, third row must be [0, 0, 1]")
            M = M[:2, :]

        # Convert pixel-based matrix to physical coordinates if reference_pixel_size is provided
        if reference_pixel_size is not None:
            M = M.copy().astype(np.float64)

            if source_pixel_size is not None:
                M[:2, :2] *= (reference_pixel_size / source_pixel_size)

            M[0, 2] *= reference_pixel_size  # Convert x offset: pixels → µm
            M[1, 2] *= reference_pixel_size  # Convert y offset: pixels → µm
            if verbose:
                print(f"Converted transformation matrix from pixel coordinates "
                      f"(reference: {reference_pixel_size} µm/pixel) to physical coordinates.")

        if verbose:
            print(f"Applying transformation matrix (in physical coordinates):\n{M}")

        # Apply transformation to each image
        for name in list(_self._metadata.keys()):
            img = _self._data[name]
            pixel_size = _self._metadata[name]['pixel_size']
            axes = _self._metadata[name]['axes']

            # Handle image pyramids (list of arrays)
            if isinstance(img, list):
                img_to_transform = img[0]  # Use highest resolution
                is_pyramid = True
            else:
                img_to_transform = img
                is_pyramid = False

            # Convert dask array to numpy for transformation
            if isinstance(img_to_transform, da.Array):
                img_to_transform = img_to_transform.compute()

            # Scale transformation matrix based on pixel size
            # The transformation matrix is now in physical coordinates (µm)
            # We need to convert to pixel coordinates for this specific image
            scaled_M = M.copy().astype(np.float64)
            scaled_M[0, 2] /= pixel_size  # Scale x offset: µm → pixels
            scaled_M[1, 2] /= pixel_size  # Scale y offset: µm → pixels

            # Get image dimensions
            img_axes = ImageAxes(axes)
            if output_size is not None:
                # Convert physical output size (height, width) to pixels for this image
                h = int(round(output_size[0] / pixel_size))
                w = int(round(output_size[1] / pixel_size))
            else:
                # Use input image dimensions
                h = img_to_transform.shape[img_axes.Y]
                w = img_to_transform.shape[img_axes.X]

            if verbose:
                print(f"Transforming image '{name}' with shape {img_to_transform.shape} -> output size ({w}, {h})")

            # Apply transformation based on image type (grayscale, RGB, or multichannel)
            if len(img_to_transform.shape) == 2:
                # Grayscale image (YX)
                transformed = cv2.warpAffine(
                    img_to_transform,
                    scaled_M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            elif len(img_to_transform.shape) == 3:
                if img_axes.is_rgb:
                # if axes == "YXS" or (img_axes.S is not None):
                    # RGB image - transform directly
                    transformed = cv2.warpAffine(
                        img_to_transform,
                        scaled_M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                else:
                    # Multichannel image (CYX) - transform each channel
                    n_channels = img_to_transform.shape[img_axes.C]
                    transformed_channels = []
                    for c in range(n_channels):
                        channel = np.take(img_to_transform, c, axis=img_axes.C)
                        transformed_channel = cv2.warpAffine(
                            channel,
                            scaled_M,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        transformed_channels.append(transformed_channel)
                    # Stack channels back
                    transformed = np.stack(transformed_channels, axis=img_axes.C)
            else:
                raise ValueError(f"Unsupported image shape: {img_to_transform.shape}")

            # Convert back to dask array
            transformed = da.from_array(transformed)

            # Recreate pyramid if needed
            if is_pyramid:
                transformed = create_img_pyramid(transformed, axes=axes, nsubres=len(img))

            # Update data
            _self._data[name] = transformed

            # Update shape in metadata
            if isinstance(transformed, list):
                _self._metadata[name]["shape"] = transformed[0].shape
            else:
                _self._metadata[name]["shape"] = transformed.shape

            if verbose:
                print(f"Transformed image '{name}'")

        if verbose:
            print(f"Transformed {len(_self._metadata)} images.")

        if not inplace:
            return _self


class SpatialUnitsData(DeepCopyMixin):
    """
    Object to store spatial units (e.g., functional tissue units, niches)
    with their associated omics data.

    Geometric information about the spatial units are stored as GeoDataFrames
    with polygon geometries, and their omics readouts are stored
    as AnnData objects. This provides flexibility for defining various spatial units beyond cells.

    Note: All coordinates in the geometries are assumed to be given as physical
    coordinates (usually µm).
    """

    def __init__(
        self,
        shapes: Optional[gpd.GeoDataFrame],
        data: Optional[AnnData],
        unit_type: str = "unit"
    ):
        """
        Initialize SpatialUnitsData object.

        Args:
            shapes: GeoDataFrame containing polygon geometries for spatial units.
                Should have columns: 'geometry', 'name' (unit identifier),
                and optionally 'color', 'type', etc.
                All coordinates are assumed to be in physical units (usually µm).
            data: AnnData object with omics readouts. obs_names should
                match unit names in the GeoDataFrame.
            unit_type: Description of unit type (e.g., 'niche', 'functional_unit').
        """
        self._shapes = shapes.copy() if shapes is not None else gpd.GeoDataFrame()
        self._data = data.copy()
        self._unit_type = unit_type

        # Convert Point geometries with radius to circles
        if not self._shapes.empty and 'radius' in self._shapes.columns:
            # Check if any geometries are Points
            point_mask = self._shapes.geometry.geom_type.isin(['Point', 'MultiPoint'])
            if point_mask.any():
                # Only convert Point geometries that have a valid (non-NA) radius
                radius_valid = ~self._shapes['radius'].isna()
                convert_mask = point_mask & radius_valid

                if convert_mask.any():
                    logger.info(f"Converting {convert_mask.sum()} Point geometries with radius to circular polygons using buffer.")
                    self._shapes.loc[convert_mask, 'geometry'] = self._shapes.loc[convert_mask].apply(
                        lambda row: row.geometry.buffer(row.radius), axis=1
                    )

                # Remove radius column after conversion
                # self._shapes = self._shapes.drop(columns=['radius'])

        # Validate consistency if both features and data are provided
        if not self._shapes.empty and self._data is not None:
            self._validate_consistency()

            # rename feature index to match data.obs_names
            self._shapes.index = self._data.obs_names

    def __repr__(self):
        n_units = len(self._shapes)
        has_data = self._data is not None

        if n_units > 0:
            repr_str = (
                f"{tf.Bold}SpatialUnitsData{tf.ResetAll} (Type: '{self._unit_type}')\n"
            )

            if has_data:
                repr_str += (
                    f"{tf.SPACER}.table: {self._data.n_obs} obs × "
                    f"{self._data.n_vars} vars\n"
                    f"{tf.SPACER}.shapes: {n_units} geometries"
                )

            # if self._pixel_size is not None:
            #     repr_str += f"{tf.SPACER}Pixel size: {self._pixel_size} µm"
        else:
            repr_str = "Empty SpatialUnitsData object"

        return repr_str

    def __len__(self):
        return len(self._shapes)

    def __getitem__(self, key):
        """Subset SpatialUnitsData by unit indices or names."""
        new_obj = self.copy()

        if isinstance(key, (int, slice, list, np.ndarray, pd.Series)):
            new_obj._shapes = new_obj._shapes.iloc[key].copy()
        elif isinstance(key, str):
            # Assume string key is a unit name
            new_obj._shapes = new_obj._shapes[
                new_obj._shapes['name'] == key
            ].copy()
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

        # Sync data if present
        if new_obj._data is not None:
            unit_names = new_obj._shapes.index.tolist()
            new_obj._data = new_obj._data[unit_names, :].copy()

        return new_obj

    @property
    def shapes(self) -> gpd.GeoDataFrame:
        """GeoDataFrame containing geometries of `.shapes`."""
        return self._shapes

    @shapes.setter
    def shapes(self, value: gpd.GeoDataFrame):
        if not isinstance(value, gpd.GeoDataFrame):
            raise TypeError(f"`.shapes` must be GeoDataFrame, not {type(value)}")
        self._shapes = value

    # @property
    # def data(self) -> Optional[AnnData]:
    #     """Alias for table property."""
    #     return self._data

    # @data.setter
    # def data(self, value: Optional[AnnData]):
    #     """Alias for table setter."""
    #     if value is not None and not isinstance(value, AnnData):
    #         raise TypeError(f"data must be AnnData object, not {type(value)}")
    #     self._data = value

    @property
    def table(self) -> Optional[AnnData]:
        """AnnData object with omics readouts. This is the preferred name going forward."""
        return self._data

    @table.setter
    def table(self, value: Optional[AnnData]):
        """Set the AnnData table. This is the preferred name going forward."""
        if value is not None and not isinstance(value, AnnData):
            raise TypeError(f"table must be AnnData object, not {type(value)}")
        self._data = value

    @property
    def unit_type(self) -> str:
        """Type of spatial units stored."""
        return self._unit_type

    @property
    def is_empty(self) -> bool:
        return len(self._shapes) == 0

    def _validate_consistency(self):
        """Validate that shapes and data indices match."""
        if self._data is None:
            return

        unit_names = self._shapes.index
        data_names = self._data.obs_names

        if len(unit_names) != len(data_names):
            raise ValueError(
                f"Number of shapes ({len(unit_names)}) does not match "
                f"number of data obs ({len(data_names)})."
            )

        if not np.all(unit_names == data_names):
            logger.warning(
                f"Indices in `.shapes` do not match `.data.obs_names`. Shapes will be renamed according to the `obs_names`. "
                f"For this to be valid, please make sure that the order of elements in `.shapes` and `.data` matches."
            )

    def crop(
        self,
        xlim: Optional[Tuple[Number, Number]] = None,
        ylim: Optional[Tuple[Number, Number]] = None,
        shape: Optional[Union[Polygon, MultiPolygon]] = None,
        inplace: bool = False,
        verbose: bool = True
    ):
        """
        Crop spatial units to a specified region.

        Args:
            xlim: X-axis limits (min, max).
            ylim: Y-axis limits (min, max).
            shape: Polygon/MultiPolygon to crop to. Takes precedence over xlim/ylim.
            inplace: Modify object in place.
            verbose: Print status messages.

        Returns:
            Cropped SpatialUnitsData if not inplace, else None.
        """
        _self = self if inplace else self.copy()

        # Create crop shape
        if shape is None:
            if xlim is None or ylim is None:
                raise ValueError("Must provide either shape or both xlim and ylim.")
            shape = Polygon([
                (xlim[0], ylim[0]), (xlim[1], ylim[0]),
                (xlim[1], ylim[1]), (xlim[0], ylim[1])
            ])
        else:
            if xlim is not None and ylim is not None and verbose:
                warnings.warn("Both shape and xlim/ylim provided. Using shape.")
            xlim = shape.bounds[0], shape.bounds[2]
            ylim = shape.bounds[1], shape.bounds[3]

        # Filter features that intersect
        mask = _self._shapes.geometry.intersects(shape)
        _self._shapes = _self._shapes[mask].copy()

        # Translate to origin
        _self._shapes["geometry"] = _self._shapes["geometry"].apply(
            affinity.translate, xoff=-xlim[0], yoff=-ylim[0]
        )

        # Crop data if present
        if _self._data is not None:
            feature_names = _self._shapes.index.tolist()
            _self._data = _self._data[feature_names, :].copy()

        if verbose:
            print(f"Cropped to {len(_self._shapes)} features.")

        if not inplace:
            return _self

    def sync(self, verbose: bool = False):
        """
        Synchronize spatial units and data to have matching indices.
        Keeps only units present in both.
        """
        if self._data is None:
            if verbose:
                print("No data to sync.")
            return

        unit_names = set(self._shapes.index)
        data_names = set(self._data.obs_names)
        common_names = unit_names & data_names

        # Filter units
        self._shapes = self._shapes.loc[list(common_names)]

        # Filter data
        self._data = self._data[list(common_names), :].copy()

        if verbose:
            print(f"Synced to {len(common_names)} common features.")

    def transform(
        self,
        transformation_matrix: Union[np.ndarray, str, os.PathLike, Path],
        source_pixel_size: Optional[Number] = None,
        reference_pixel_size: Optional[Number] = None,
        inplace: bool = False,
        verbose: bool = False
    ):
        """Apply an affine transformation to all geometries in the FeatureData object.

        Transforms all feature geometries using the provided affine transformation
        matrix. Since FeatureData stores coordinates in physical units (µm), the
        transformation matrix should also be in physical coordinates.

        Args:
            transformation_matrix: Either a 2x3 or 3x3 numpy array representing
                the affine transformation matrix, or a path to a CSV/Excel file
                containing the matrix. The matrix should be in the form:
                [[a, b, xoff],
                 [d, e, yoff]]
                or
                [[a, b, xoff],
                 [d, e, yoff],
                 [0, 0, 1]]
            source_pixel_size: Pixel size (in µm/pixel) of the source image from
                which the features were derived. Only used if reference_pixel_size
                is also provided. If None, it is assumed to be equal to
                reference_pixel_size (i.e., no scaling of the linear transformation
                component is performed).
            reference_pixel_size: Pixel size (in µm/pixel) of the reference image
                used during registration. If provided, the transformation matrix
                offsets (xoff, yoff) are assumed to be in pixel coordinates of the
                reference image and will be converted to physical coordinates (µm).
                If None, the matrix offsets are assumed to already be in physical
                coordinates (µm). This is important when the transformation matrix
                was computed in pixel space for a specific image resolution.
            inplace: If True, modify the object in place. Otherwise, return a
                transformed copy. Defaults to False.
            verbose: If True, print status messages. Defaults to False.

        Returns:
            SpatialUnitsData: Transformed SpatialUnitsData object if inplace=False, else None.

        Raises:
            ValueError: If the transformation matrix has invalid dimensions or format.
            FileNotFoundError: If the provided path does not exist.

        Example:
            >>> # Matrix in physical coordinates (µm)
            >>> features.transform(transformation_matrix=matrix)

            >>> # Matrix in pixel coordinates, computed for reference at 0.2125 µm/pixel
            >>> features.transform(
            ...     transformation_matrix=matrix,
            ...     reference_pixel_size=0.2125
            ... )
        """
        _self = self if inplace else self.copy()

        # Load transformation matrix if it's a file path
        if isinstance(transformation_matrix, (str, os.PathLike, Path)):
            transformation_matrix = Path(transformation_matrix)
            if not transformation_matrix.exists():
                raise FileNotFoundError(f"Transformation matrix file not found: {transformation_matrix}")

            # Read file based on extension
            if transformation_matrix.suffix.lower() in ['.csv', '.txt']:
                M = pd.read_csv(transformation_matrix, header=None).values
            elif transformation_matrix.suffix.lower() in ['.xlsx', '.xls']:
                M = pd.read_excel(transformation_matrix, header=None).values
            else:
                raise ValueError(f"Unsupported file format: {transformation_matrix.suffix}. Use .csv, .txt, .xlsx, or .xls")
        else:
            M = np.array(transformation_matrix)

        # Validate matrix dimensions
        if M.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Transformation matrix must be 2x3 or 3x3, got shape {M.shape}. "
                f"Expected format:\n"
                f"[[a, b, xoff],\n"
                f" [d, e, yoff]] or with [0, 0, 1] as third row."
            )

        # Extract transformation parameters
        if M.shape == (3, 3):
            # Validate that the third row is [0, 0, 1]
            if not np.allclose(M[2, :], [0, 0, 1]):
                raise ValueError("For 3x3 matrix, third row must be [0, 0, 1]")
            M = M[:2, :]

        # Convert pixel-based matrix to physical coordinates if reference_pixel_size is provided
        if reference_pixel_size is not None:
            M = M.copy().astype(np.float64)

            if source_pixel_size is not None:
                M[:2, :2] *= (reference_pixel_size / source_pixel_size)

            M[0, 2] *= reference_pixel_size  # Convert x offset: pixels → µm
            M[1, 2] *= reference_pixel_size  # Convert y offset: pixels → µm
            if verbose:
                print(f"Converted transformation matrix from pixel coordinates "
                      f"(reference: {reference_pixel_size} µm/pixel) to physical coordinates.")

        # Apply transformation to geometries using shapely's affine_transform
        # Matrix format for shapely: [a, b, d, e, xoff, yoff]
        a, b, xoff = M[0, :]
        d, e, yoff = M[1, :]

        if verbose:
            print(f"Applying transformation (in physical coordinates): "
                  f"a={a}, b={b}, d={d}, e={e}, xoff={xoff}, yoff={yoff}")

        _self._shapes["geometry"] = _self._shapes["geometry"].apply(
            lambda geom: affinity.affine_transform(geom, [a, b, d, e, xoff, yoff])
        )

        if verbose:
            print(f"Transformed {len(_self._shapes)} features.")

        if not inplace:
            return _self

    def save(
        self,
        path: Union[str, os.PathLike, Path],
        overwrite: bool = False
    ):
        """
        Save FeatureData to directory.

        Args:
            path: Output directory path.
            overwrite: If True, overwrite existing files.
        """
        path = Path(path)

        # Check overwrite
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # Create directory
        path.mkdir(parents=True, exist_ok=True)

        # Save features as geojson
        if not self._shapes.empty:
            features_file = path / "features.geojson"
            write_qupath_geojson(dataframe=self._shapes, file=features_file)

        # Save data as h5ad
        if self._data is not None:
            data_file = path / "data.h5ad"
            self._data.write(data_file)

        # Save metadata
        metadata = {
            "version": __version__,
            "feature_type": self._feature_type,
            "n_features": len(self._shapes),
            "has_data": self._data is not None
        }
        write_dict_to_json(dictionary=metadata, file=path / ".featuredata")
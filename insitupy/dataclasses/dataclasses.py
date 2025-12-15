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
from parse import *
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
from insitupy.dataclasses._segmentations import _read_proseg
from insitupy.images.axes import (ImageAxes, _transpose_to_standard_axes,
                                  get_height_and_width)
from insitupy.images.io import read_image, write_ome_tiff, write_zarr
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
                 ):
        """
        Initialize the BoundariesData object.

        Args:
            cell_names (Union[np.ndarray, List]): Cell names which need to correspond to `.obs_names` in the `.table` of `CellData`.
            seg_mask_value (Optional[Union[np.ndarray, List]]): Segmentation mask values. Required to have the same length as `cell_names`.
                Specifies which values in the "cells" segmentation mask correspond to which cell name.

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

        self._data = dict()

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
    def is_empty(self):
        return len(self._data) == 0

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

            #self._cell_ids.to_zarr(dirstore, component="cell_id")
            self.cell_names.to_zarr(dirstore, component="cell_names", overwrite=True)

            if self._seg_mask_value is not None:
                self.seg_mask_value.to_zarr(dirstore, component="seg_mask_value", overwrite=True)

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
               matrix: AnnData,
               boundaries: Optional[BoundariesData],
               config: dict = {}
               ):
        self._matrix = matrix
        self._config = config

        if boundaries is not None:
            self._boundaries = boundaries
            self._entries = ["matrix", "boundaries"]
        else:
            self._boundaries = None
            self._entries = ["matrix"]

    def __getitem__(self, key):
        """Retrieve a subset of the `CellData` object.

        Args:
            key (int, slice, list, np.ndarray, pd.Series): The index, slice, list of indices, boolean mask,
                or Series to retrieve.

        Returns:
            `CellData`: A new `CellData` object with the selected subset of cells.
        """
        new_celldata = self.copy()
        new_celldata._matrix = new_celldata._matrix[key].copy()
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
            f"{tf.Bold+'matrix'+tf.ResetAll}\n"
            f"{tf.SPACER+self._matrix.__repr__()}"
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
        return self._matrix

    @matrix.setter
    def matrix(self, value: AnnData):
        logger.warning(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead."
        )
        if not isinstance(value, AnnData):
            raise ValueError(f"Matrix must be an AnnData object. Instead: {type(value)}.")
        self._matrix = value

    @property
    def table(self):
        """Alias for matrix property. This is the preferred name going forward."""
        return self._matrix

    @table.setter
    def table(self, value: AnnData):
        """Alias for matrix setter. This is the preferred name going forward."""
        if not isinstance(value, AnnData):
            raise ValueError(f"Table must be an AnnData object. Instead: {type(value)}.")
        self._matrix = value

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
        _self.table = _self.table[mask, :].copy()

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

        # write matrix to file
        mtx_file = path / "matrix.h5ad"
        self._matrix.write(mtx_file)
        celldata_metadata["matrix"] = Path(relpath(mtx_file, path)).as_posix()

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
        Function to synchronize matrix and boundaries of CellData.

        Procedure:
        1. Select matrix cell IDs
        2. Check if all matrix cell IDs are in boundaries
            - if not all are in boundaries, throw error saying that those will also be removed
        3. Select only matrix cell IDs which are also in boundaries and filter for them
        '''
        # get cell IDs from matrix
        matrix_cell_ids_hex = self._matrix.obs_names.astype(str)

        if self._boundaries is None:
            print('No `boundaries` attribute found in CellData found.')
        else:
            boundaries = self._boundaries

            # create pandas series from seg_mask values and cell_names
            ds = pd.Series(
                data=boundaries.seg_mask_value,
                index=boundaries.cell_names
                )

            filter_mask_in = ds.index.isin(matrix_cell_ids_hex)

            if not np.any(filter_mask_in):
                raise ValueError("No matching values between boundaries.cell_names and matrix.obs_names. All boundaries would get filtered out.")

            # filter cell names and seg_mask_values
            boundaries._seg_mask_value = da.from_array(np.array(ds[filter_mask_in]))
            boundaries._cell_names = da.from_array(np.array(ds.index[filter_mask_in], dtype=str))

            # find the seg_mask_values which are not anymore present
            seg_mask_values_not_in_matrix = ds[~filter_mask_in].values

            # extract boundaries
            cell_bounds = boundaries["cells"]
            nuc_bounds = boundaries["nuclei"]

            if isinstance(cell_bounds, list):
                if nuc_bounds is not None:
                    assert isinstance (nuc_bounds, list), "Cellular boundaries are a image pyramid but nuclear boundaries are not. Both need to be of the same type for the synchronization to work."
                for i, cell_bound in enumerate(cell_bounds):
                    removed_cells_mask = da.isin(cell_bound, seg_mask_values_not_in_matrix)
                    cell_bound[removed_cells_mask] = 0 # set all removed cells 0
                    if nuc_bounds is not None:
                        nuc_bounds[i][removed_cells_mask] = 0 # set all nuclei belong to the removed cells 0
            elif isinstance(cell_bounds, da.core.Array):
                if nuc_bounds is not None:
                    assert isinstance (nuc_bounds, da.core.Array), "Cellular boundaries are a dask array but nuclear boundaries are not. Both need to be of the same type for the synchronization to work."
                # set all non existent cell ids to zero
                removed_cells_mask = da.isin(cell_bounds, seg_mask_values_not_in_matrix)
                cell_bounds[removed_cells_mask] = 0 # set all removed cells 0

                if nuc_bounds is not None:
                    nuc_bounds[removed_cells_mask] = 0 # set all nuclei belong to the removed cells 0
            else:
                warnings.warn(f"Unknown data type for cellular boundaries: {type(cell_bounds)}. Need to be either a dask array or a list of dask arrays. Skipped synchronization of cell ids.")

            if verbose:
                print(f"Filtered out {np.sum(~filter_mask_in)} boundaries.", flush=True)

    def shift(self,
              x: Union[int, float],
              y: Union[int, float]
              ):
        '''
        Function to shift the coordinates of both matrix and boundaries data by certain values x/y.
        '''

        # move origin again to 0 by subtracting the lower limits from the coordinates
        cell_coords = self._matrix.obsm['spatial'].copy()
        cell_coords[:, 0] += x
        cell_coords[:, 1] += y
        self._matrix.obsm['spatial'] = cell_coords

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
            del self._layers[key]
        else:
            raise KeyError(f"Key '{key}' not found in MultiCellData.")

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
                     is_main: bool = False):
        if not isinstance(cd, CellData):
            raise ValueError(f"cd must be of type CellData. Instead: {type(cd)}.")

        if key in self._layers.keys():
            print(f"Overwriting {key}.")
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
                   is_main: bool = False
                   ):
        """
            Adds output of Proseg https://github.com/dcjones/proseg segmentation to the object.

            Args:
                path_counts (Union[str, os.PathLike, Path]): Path to the counts file (.parquet, .csv or csv.gz).
                path_metadata (Union[str, os.PathLike, Path]): Path to the metadata file (.parquet, .csv or csv.gz).
                path_baysor_polygons (Union[str, os.PathLike, Path]): Path to the Baysor-like polygons file.
                pixel_size (float): Size of the pixel for scaling.
                key (str, optional): Key to store the data. Defaults to "proseg".
                is_main (bool, optional): Flag to indicate if this is the main data. Defaults to False.
        """


        # generate data paths
        path = Path(path)

        adata, boundaries_mask, cell_names, seg_mask_value = _read_proseg(
            path, counts_file=counts_file, cell_metadata_file=cell_metadata_file, polygons_file=polygons_file, pixel_size=pixel_size
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

        # Create cell data and add to object
        celldata = CellData(matrix=adata, boundaries=boundaries)

        self.add_celldata(cd=celldata, key=key, is_main=is_main)


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
                    name=n,
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
        name: str,
        axes: Optional[str] = None, # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
        pixel_size: Optional[Number] = None,
        ome_meta: Optional[dict] = {},
        is_rgb: Optional[bool] = None,
        overwrite: bool = False,
        verbose: bool = True
        ):
        if name in self._names:
            if not overwrite:
                print(f"`ImageData` object contains already an image with name '{name}'. Image is not added.") if verbose else None
                do_addition = False
            else:
                # remove attribute with current name
                del self._data[name]

                # remove from name list and metadata
                self._names = [elem for elem in self._names if elem != name]
                self._metadata.pop(name, None)

                do_addition = True
        else:
            do_addition = True

        if do_addition:
            # check if image is a path or a data array
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

            # set attribute and add names to object
            self._data[name] = img
            self._names.append(name)

            # retrieve metadata
            img_shape = img[0].shape if isinstance(img, list) else img.shape
            # img_max = img[0].max() if isinstance(img, list) else img.max()
            # try:
            #     img_max = img_max.compute()
            # except AttributeError:
            #     img_max = img_max

            # save metadata
            self._metadata[name] = {}
            self._metadata[name]["filename"] = filename
            self._metadata[name]["shape"] = img_shape  # store shape
            self._metadata[name]["axes"] = axes
            self._metadata[name]["OME"] = ome_meta

            # if len(ome_meta) > 0:
            #     # add universal pixel size to metadata
            #     try:
            #         self._metadata[name]['pixel_size'] = float(ome_meta['Image']['Pixels']['PhysicalSizeX'])
            #     except KeyError:
            #         self._metadata[name]['pixel_size'] = float(ome_meta['PhysicalSizeX'])
            # else:
            #     self._metadata[name]['pixel_size'] = pixel_size

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

            # # get image contrast limits
            # if self._metadata[name]["rgb"]:
            #     self._metadata[name]["contrast_limits"] = (0, img_max)
            # else:
            #     self._metadata[name]["contrast_limits"] = (0, img_max)


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
            FileExistsError: If `overwrite` is False and the output folder already contains files.

        """
        output_folder = Path(output_folder)

        if keys_to_save is None:
            keys_to_save = list(self._metadata.keys())
        else:
            keys_to_save = convert_to_list(keys_to_save)

        # check overwrite
        check_overwrite_and_remove_if_true(path=output_folder, overwrite=overwrite)

        # create output directory
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
                    write_zarr(image=img, file=img_path,
                               img_metadata=new_img_metadata,
                               save_pyramid=save_pyramid,
                               axes=axes, verbose=verbose
                               )
                else:
                    # get file name for saving
                    #filename = Path(img_metadata["file"]).name.split(".")[0] + ".ome.tif"
                    filename = name + ".ome.tif"
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
                    write_ome_tiff(image=img, file=output_folder / filename,
                                photometric=photometric, axes=axes,
                                compression=compression,
                                metadata=selected_metadata, overwrite=False,
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
                matrix = pd.read_csv(transformation_matrix, header=None).values
            elif transformation_matrix.suffix.lower() in ['.xlsx', '.xls']:
                matrix = pd.read_excel(transformation_matrix, header=None).values
            else:
                raise ValueError(f"Unsupported file format: {transformation_matrix.suffix}. Use .csv, .txt, .xlsx, or .xls")
        else:
            matrix = np.array(transformation_matrix)

        # Validate matrix dimensions
        if matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Transformation matrix must be 2x3 or 3x3, got shape {matrix.shape}. "
                f"Expected format:\n"
                f"[[a, b, xoff],\n"
                f" [d, e, yoff]] or with [0, 0, 1] as third row."
            )

        # Extract transformation parameters
        if matrix.shape == (3, 3):
            # Validate that the third row is [0, 0, 1]
            if not np.allclose(matrix[2, :], [0, 0, 1]):
                raise ValueError("For 3x3 matrix, third row must be [0, 0, 1]")
            matrix = matrix[:2, :]

        # Convert pixel-based matrix to physical coordinates if reference_pixel_size is provided
        if reference_pixel_size is not None:
            matrix = matrix.copy().astype(np.float64)

            if source_pixel_size is not None:
                matrix[:2, :2] *= (reference_pixel_size / source_pixel_size)

            matrix[0, 2] *= reference_pixel_size  # Convert x offset: pixels → µm
            matrix[1, 2] *= reference_pixel_size  # Convert y offset: pixels → µm
            if verbose:
                print(f"Converted transformation matrix from pixel coordinates "
                      f"(reference: {reference_pixel_size} µm/pixel) to physical coordinates.")

        if verbose:
            print(f"Applying transformation matrix (in physical coordinates):\n{matrix}")

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
            scaled_matrix = matrix.copy().astype(np.float64)
            scaled_matrix[0, 2] /= pixel_size  # Scale x offset: µm → pixels
            scaled_matrix[1, 2] /= pixel_size  # Scale y offset: µm → pixels

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
                    scaled_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            elif len(img_to_transform.shape) == 3:
                if axes == "YXS" or (img_axes.S is not None):
                    # RGB image - transform directly
                    transformed = cv2.warpAffine(
                        img_to_transform,
                        scaled_matrix,
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
                            scaled_matrix,
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


class FeatureData(DeepCopyMixin):
    """
    Object to store spatial features (e.g., functional tissue units, niches)
    with their associated omics data.

    Features are stored as GeoDataFrames with polygon geometries, and their
    omics readouts are stored as AnnData objects. This provides
    flexibility for defining various spatial units beyond cells.

    Note: All coordinates in the geometries are assumed to be given as physical
    coordinates (usually µm).
    """

    def __init__(
        self,
        shapes: Optional[gpd.GeoDataFrame],
        data: Optional[AnnData],
        feature_type: str = "feature"
    ):
        """
        Initialize FeatureData object.

        Args:
            shapes: GeoDataFrame containing polygon geometries for features.
                Should have columns: 'geometry', 'name' (feature identifier),
                and optionally 'color', 'type', etc.
                All coordinates are assumed to be in physical units (usually µm).
            data: AnnData object with omics readouts. obs_names should
                match feature names in the GeoDataFrame.
            feature_type: Description of feature type (e.g., 'niche', 'functional_unit').
        """
        self._shapes = shapes.copy() if shapes is not None else gpd.GeoDataFrame()
        self._data = data.copy()
        self._feature_type = feature_type

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
        n_features = len(self._shapes)
        has_data = self._data is not None

        if n_features > 0:
            repr_str = (
                f"{tf.Bold}FeatureData{tf.ResetAll} (Type: '{self._feature_type}')\n"
            )

            if has_data:
                repr_str += (
                    f"{tf.SPACER}.data: {self._data.n_obs} obs × "
                    f"{self._data.n_vars} vars\n"
                    f"{tf.SPACER}.shapes: {n_features} geometries"
                )

            # if self._pixel_size is not None:
            #     repr_str += f"{tf.SPACER}Pixel size: {self._pixel_size} µm"
        else:
            repr_str = "Empty FeatureData object"

        return repr_str

    def __len__(self):
        return len(self._shapes)

    def __getitem__(self, key):
        """Subset FeatureData by feature indices or names."""
        new_obj = self.copy()

        if isinstance(key, (int, slice, list, np.ndarray, pd.Series)):
            new_obj._shapes = new_obj._shapes.iloc[key].copy()
        elif isinstance(key, str):
            # Assume string key is a feature name
            new_obj._shapes = new_obj._shapes[
                new_obj._shapes['name'] == key
            ].copy()
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

        # Sync data if present
        if new_obj._data is not None:
            feature_names = new_obj._shapes.index.tolist()
            new_obj._data = new_obj._data[feature_names, :].copy()

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

    @property
    def data(self) -> Optional[AnnData]:
        """Alias for table property."""
        return self._data

    @data.setter
    def data(self, value: Optional[AnnData]):
        """Alias for table setter."""
        if value is not None and not isinstance(value, AnnData):
            raise TypeError(f"data must be AnnData object, not {type(value)}")
        self._data = value

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
    def feature_type(self) -> str:
        """Type of features stored."""
        return self._feature_type

    @property
    def is_empty(self) -> bool:
        return len(self._shapes) == 0

    def _validate_consistency(self):
        """Validate that shapes and data indices match."""
        if self._data is None:
            return

        feature_names = self._shapes.index
        data_names = self._data.obs_names

        if len(feature_names) != len(data_names):
            raise ValueError(
                f"Number of shapes ({len(feature_names)}) does not match "
                f"number of data obs ({len(data_names)})."
            )

        if not np.all(feature_names == data_names):
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
        Crop features to a specified region.

        Args:
            xlim: X-axis limits (min, max).
            ylim: Y-axis limits (min, max).
            shape: Polygon/MultiPolygon to crop to. Takes precedence over xlim/ylim.
            inplace: Modify object in place.
            verbose: Print status messages.

        Returns:
            Cropped FeatureData if not inplace, else None.
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
        Synchronize features and data to have matching indices.
        Keeps only features present in both.
        """
        if self._data is None:
            if verbose:
                print("No data to sync.")
            return

        feature_names = set(self._shapes.index)
        data_names = set(self._data.obs_names)
        common_names = feature_names & data_names

        # Filter features
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
            FeatureData: Transformed FeatureData object if inplace=False, else None.

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
                matrix = pd.read_csv(transformation_matrix, header=None).values
            elif transformation_matrix.suffix.lower() in ['.xlsx', '.xls']:
                matrix = pd.read_excel(transformation_matrix, header=None).values
            else:
                raise ValueError(f"Unsupported file format: {transformation_matrix.suffix}. Use .csv, .txt, .xlsx, or .xls")
        else:
            matrix = np.array(transformation_matrix)

        # Validate matrix dimensions
        if matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Transformation matrix must be 2x3 or 3x3, got shape {matrix.shape}. "
                f"Expected format:\n"
                f"[[a, b, xoff],\n"
                f" [d, e, yoff]] or with [0, 0, 1] as third row."
            )

        # Extract transformation parameters
        if matrix.shape == (3, 3):
            # Validate that the third row is [0, 0, 1]
            if not np.allclose(matrix[2, :], [0, 0, 1]):
                raise ValueError("For 3x3 matrix, third row must be [0, 0, 1]")
            matrix = matrix[:2, :]

        # Convert pixel-based matrix to physical coordinates if reference_pixel_size is provided
        if reference_pixel_size is not None:
            matrix = matrix.copy().astype(np.float64)

            if source_pixel_size is not None:
                matrix[:2, :2] *= (reference_pixel_size / source_pixel_size)

            matrix[0, 2] *= reference_pixel_size  # Convert x offset: pixels → µm
            matrix[1, 2] *= reference_pixel_size  # Convert y offset: pixels → µm
            if verbose:
                print(f"Converted transformation matrix from pixel coordinates "
                      f"(reference: {reference_pixel_size} µm/pixel) to physical coordinates.")

        # Apply transformation to geometries using shapely's affine_transform
        # Matrix format for shapely: [a, b, d, e, xoff, yoff]
        a, b, xoff = matrix[0, :]
        d, e, yoff = matrix[1, :]

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
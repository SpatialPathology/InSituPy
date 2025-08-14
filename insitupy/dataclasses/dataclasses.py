import os
import warnings
from copy import deepcopy
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from parse import *
from shapely import MultiPoint, MultiPolygon, Point, Polygon, affinity

from insitupy import WITH_NAPARI, __version__
from insitupy._constants import FORBIDDEN_ANNOTATION_NAMES
from insitupy._core._mixins import DeepCopyMixin
from insitupy._exceptions import InvalidFileTypeError
from insitupy._io.files import (check_overwrite_and_remove_if_true,
                                write_dict_to_json)
from insitupy._io.geo import parse_geopandas, write_qupath_geojson
from insitupy._textformat import textformat as tf
from insitupy.dataclasses._segmentations import _read_proseg
from insitupy.images.io import read_image, write_ome_tiff, write_zarr
from insitupy.images.utils import (_efficiently_resize_array,
                                   _get_scale_factor_from_max_res,
                                   create_img_pyramid,
                                   crop_dask_array_or_pyramid, resize_image)
from insitupy.utils._checks import _is_list_of_dask_arrays
from insitupy.utils.utils import convert_to_list, decode_robust_series

if WITH_NAPARI:
    from napari.utils.notifications import show_info, show_warning

class ShapesData(DeepCopyMixin):
    '''
    Object to store annotations.
    '''
    # default_skip_multipolygons = False
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None,
                 keys: Optional[List[str]] = None,
                 pixel_size: Optional[float] = None,
                 assert_uniqueness: bool = False,
                 polygons_only: bool = False,
                 forbidden_names: Optional[List[str]] = None,
                 shape_name: Optional[str] = None,
                 #repr_color = tf.Cyan
                 ) -> None:
        self._shape_name = shape_name if shape_name is not None else "shapes"

        # add hidden variables
        self._metadata = {} # dictionary for metadata
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
        if len(self._metadata) > 0:
            repr_strings = []
            for l, m in self._metadata.items():
                # add ' to classes
                # classes = [f"'{elem}'" for elem in m["classes"]]
                # lc = len(classes)

                # get metadata
                n = len(self._data[l]) # get number of entries
                classes = sorted(self._data[l]["name"].unique())
                classes_str = [f"'{elem}'" for elem in classes]
                lc = len(classes)

                # create string
                r = (
                    f'{tf.Bold}{l}:{tf.ResetAll}\t{n} '
                    f'{self._shape_name}, {lc} '
                    f'{"classes" if lc>1 else "class"} '
                )
                if lc < 10:
                    r += f'({", ".join(classes_str)}) {m["analyzed"]}'
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
        return self._metadata

    @property
    def is_empty(self):
        return len(self._data) == 0

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError(f'The metadata attribute must be a dictionary.')
        self._metadata = value

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

    def _update_metadata(self,
                         keys: Union[str, Literal["all"]] = "all",
                         analyzed: bool = False,
                         verbose: bool = False
                         ):

        if keys == "all":
            keys = list(self._metadata.keys())

        keys = convert_to_list(keys)
        keys_to_remove = []
        for key in keys:
            if self[key] is None:
                self._metadata.pop(key)
                if verbose:
                    print(f'Removed {key}', flush=True)
            else:
                annot_df = self[key]
                # record metadata information
                self._metadata[key][f"n_{self._shape_name}"] = len(annot_df)  # number of annotations

                try:
                    self._metadata[key]["classes"] = annot_df['name'].unique().tolist()  # annotation classes
                except KeyError:
                    self._metadata[key]["classes"] = ["unnamed"]

                self._metadata[key]["analyzed"] = tf.Tick if analyzed else ""  # whether this annotation has been used in the annotate() function

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

            # # convert pixel coordinates to metric units
            # new_df["geometry"] = new_df.geometry.scale(origin=(0,0), xfact=pixel_size, yfact=pixel_size)

            if key not in self._data.keys():
                # if key does not exist yet, the new df is the whole annotation dataframe
                annot_df = new_df

                # collect additional variables for reporting
                new_geometries_added = True # dataframe will be added later
                existing_str = ""
                old_n = 0
                new_n = len(annot_df)
            else:
                # concatenate old and new annoation dataframe
                annot_df = self[key]
                old_n = len(annot_df)
                annot_df = pd.concat([annot_df, new_df], ignore_index=False)

                # remove all duplicated shapes - leaving only the newly added
                dup_mask = annot_df.index.duplicated()
                annot_df = annot_df[~dup_mask] # filter out duplicates
                new_n = len(annot_df)

                # collect additional variables for reporting
                new_geometries_added = new_n > old_n
                existing_str = "existing "

            if new_geometries_added:
                add = True
                if self._assert_uniqueness:
                    # check if the shapes data for this key is unique (same number of names than indices)
                    is_unique = self._check_uniqueness(dataframe=annot_df, key=key, verbose=False)

                    if not is_unique:
                        add = False

                if self._polygons_only:
                    # check if any of the shapes are not shapely Polygons
                    is_not_polygon = np.array([not isinstance(p, Polygon) for p in annot_df.geometry])
                    if np.any(is_not_polygon):
                        annot_df = annot_df.loc[~is_not_polygon]
                        warnings.warn(
                            f"Some {self._shape_name} were not shapely.Polygon objects and skipped.",
                            stacklevel=2
                            )

                # check that the dataframe is not empty
                if len(annot_df) == 0:
                    add = False

                if add:
                    # add dataframe to AnnotationData object
                    self._data[key] = annot_df

                    # add new entry to metadata
                    self._metadata[key] = {}

                    # update metadata
                    self._update_metadata(keys=key, analyzed=False)

                    if verbose:
                        # report
                        if in_napari:
                            show_info(f"Added {new_n - old_n} new {self._shape_name} to {existing_str}key '{key}'")
                        else:
                            print(f"Added {new_n - old_n} new {self._shape_name} to {existing_str}key '{key}'")

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

        new_metadata = {}
        for i, n in enumerate(_self._metadata.keys()):
            shapesdf = _self[n]

            # select annotations that intersect with the selected area
            mask = [shape.intersects(elem) for elem in shapesdf["geometry"]]
            shapesdf = shapesdf.loc[mask, :].copy()

            # move origin to zero after cropping
            shapesdf["geometry"] = shapesdf["geometry"].apply(affinity.translate, xoff=-xlim[0], yoff=-ylim[0])

            # check if there are annotations left or if it has to be deleted
            if len(shapesdf) > 0:
                # add new dataframe back to annotations object
                _self._data[n] = shapesdf

                # update metadata
                new_metadata[n] = {}
                new_metadata[n][f"n_{_self._shape_name}"] = len(shapesdf)
                new_metadata[n]["classes"] = shapesdf.name.unique().tolist()
                new_metadata[n]["analyzed"] = _self._metadata[n]["analyzed"]  # analyzed information is just copied

            else:
                # delete annotations
                del _self._data[n]

        _self._metadata = new_metadata

        _self._update_metadata()

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
                self.metadata.pop(key_to_remove, None)
            except KeyError:
                print(f"Key '{key_to_remove}' not found in ShapesData object. Nothing to remove.")
        else:
            classes_to_remove = convert_to_list(classes_to_remove)
            geom_df = self[key_to_remove]
            self._data[key_to_remove] = geom_df[~geom_df.name.isin(classes_to_remove)]
            self.metadata[key_to_remove]['classes'] = [c for c in self.metadata[key_to_remove]['classes'] if c not in classes_to_remove]

        self._update_metadata()

    def save(self,
             path: Union[str, os.PathLike, Path],
             overwrite: bool = False
             ):
        path = Path(path)

        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)

        # # create path for matrix
        # annot_path = (path / self.shape_name)
        # annot_path.mkdir(parents=True, exist_ok=True) # create directory

        # if metadata is not None:
        #     metadata["annotations"] = {}
        for n in self._metadata.keys():
            df = self[n]
            # annot_file = annot_path / f"{n}.parquet"
            # annot_df.to_parquet(annot_file)
            shapes_file = path / f"{n}.geojson"
            write_qupath_geojson(dataframe=df, file=shapes_file)

            # if metadata is not None:
            #     metadata["annotations"][n] = Path(relpath(annot_file, path)).as_posix()

        # save AnnotationData metadata
        shape_meta_path = path / f"metadata.json"
        write_dict_to_json(dictionary=self._metadata, file=shape_meta_path)

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
            cell_names (Union[np.ndarray, List]): Cell names which need to correspond to `.obs_names` in the `.matrix` of `CellData`.
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

        with zarr.ZipStore(bound_file, mode='w') if suffix == "zarr.zip" else zarr.DirectoryStore(bound_file) as dirstore:
            # for conditional 'with' see also: https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
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
                            bound_data = create_img_pyramid(img=bound_data, nsubres=6)


                    #if isinstance(bound_data, dask.array.core.Array):
                    if isinstance(bound_data, list):
                        for i, b in enumerate(bound_data):
                            comp = f"masks/{n}/{i}"
                            b.to_zarr(dirstore, component=comp)
                    else:
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
        return len(self.matrix)

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
        return self._matrix

    @matrix.setter
    def matrix(self, value: AnnData):
        if not isinstance(value, AnnData):
            raise ValueError(f"Matrix must be an AnnData object. Instead: {type(value)}.")
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
        cell_coords = _self.matrix.obsm['spatial'].copy()

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
        _self.matrix = _self.matrix[mask, :].copy()

        # crop boundaries
        _self.boundaries.crop(
            cell_ids=_self.matrix.obs_names,
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
        self._layers = dict()
        self._main_key = None

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

    def __setitem__(self, key: str, item):
        if isinstance(item, CellData):
            self._layers[key] = item
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
        try:
            return self._layers[self._main_key].matrix
        except KeyError:
            print("MultiCellData object is empty.")
            return None
        except AttributeError:
            print("No matrix available.")
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

    def get_all_keys(self):
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
        if key in self.get_all_keys():
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
                 #path: Union[str, os.PathLike, Path] = None,
                 img_files: List[str] = None,
                 img_names: List[str] = None,
                 pixel_size: float = None,
                 ):
        # # add path to object
        # self.path = path

        # iterate through files and load them
        self._names = []
        self._metadata = {}
        self._data = dict()

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
            repr_strings = [f"{tf.Bold}{n}:{tf.ResetAll}\t{metadata['shape']}" for n,metadata in self._metadata.items()]
            s = "\n".join(repr_strings)
        else:
            s = "empty"
        #repr = f"{tf.Blue+tf.Bold}images{tf.ResetAll}\n{s}"
        return s

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data.get(key)

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
        ome_meta: Optional[dict] = None,
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
                img, ome_meta, axes = read_image(image)

            else:
                raise ValueError(f"`image` is neither a dask array nor an existing path.")

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

            # add universal pixel size to metadata
            try:
                self._metadata[name]['pixel_size'] = float(ome_meta['Image']['Pixels']['PhysicalSizeX'])
            except KeyError:
                self._metadata[name]['pixel_size'] = float(ome_meta['PhysicalSizeX'])

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


import os
import warnings
from copy import deepcopy
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import cv2
import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from parse import *
from shapely import MultiPoint, MultiPolygon, Point, Polygon, affinity

from insitupy import __version__
from insitupy._constants import FORBIDDEN_ANNOTATION_NAMES
from insitupy.images.utils import resize_image
from insitupy.utils.utils import convert_int_to_xenium_hex

from .._exceptions import InvalidDataTypeError, InvalidFileTypeError
from ..images.io import read_image, write_ome_tiff, write_zarr
from ..images.utils import create_img_pyramid, crop_dask_array_or_pyramid
from ..io.files import check_overwrite_and_remove_if_true, write_dict_to_json
from ..io.geo import parse_geopandas, write_qupath_geojson
from ..utils.utils import convert_to_list, decode_robust_series
from ..utils.utils import textformat as tf
from ._mixins import DeepCopyMixin


class ShapesData(DeepCopyMixin):
    '''
    Object to store annotations.
    '''
    # default_skip_multipolygons = False
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None,
                 keys: Optional[List[str]] = None,
                 pixel_size: Optional[float] = None,
                 assert_uniqueness: Optional[bool] = None,
                #  skip_multipolygons: Optional[bool] = None,
                 polygons_only: Optional[bool] = None,
                 forbidden_names: Optional[List[str]] = None,
                 shape_name: Optional[str] = None
                 ) -> None:

        self._default_assert_uniqueness = False
        self._default_polygons_only = False
        self._repr_color = tf.Cyan
        self._default_forbidden_names = None

        self._shape_name = shape_name if shape_name is not None else "shapes"

        # create dictionary for metadata
        self._metadata = {}

        self._data =dict()

        # set configuration of ShapesData
        if assert_uniqueness is None:
            self._assert_uniqueness = self._default_assert_uniqueness
        else:
            self._assert_uniqueness = assert_uniqueness

        # if skip_multipolygons is None:
        #     self.skip_multipolygons = self.default_skip_multipolygons

        if polygons_only is None:
            self._polygons_only = self._default_polygons_only
        else:
            self._polygons_only = polygons_only

        if forbidden_names is None:
            self._forbidden_names = self._default_forbidden_names
        else:
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
                classes = [f"'{elem}'" for elem in m["classes"]]
                lc = len(classes)

                # create string
                r = (
                    f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self._shape_name}"]} '
                    f'{self._shape_name}, {lc} '
                    f'{"classes" if lc>1 else "class"} '
                )
                if lc < 10:
                    r += f'({",".join(classes)}) {m["analyzed"]}'
                repr_strings.append(r)
            # repr_strings = [
            #     #f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self.shape_name}"]} {self.shape_name}, {len(m["classes"])} classes {*m["classes"],} {m["analyzed"]}' for l, m in self.metadata.items()
            #     f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self.shape_name}"]} {self.shape_name}, {len(m["classes"])} {"classes" if len(m["classes"])>1 else "class"} ({",".join(m["classes"])}) {m["analyzed"]}' for l, m in self.metadata.items()
            #     ]

            s = "\n".join(repr_strings)
        else:
            s = ""
        repr = f"{self._repr_color}{tf.Bold}{self._shape_name}{tf.ResetAll}\n{s}"
        return repr
    

    def __getitem__(self, key):
        return self._data.get(key)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
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
                    default_name: str = "name",
                    verbose: bool = False,
                   ):
        # parse geopandas data from dataframe or file
        new_df = parse_geopandas(data)

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

        # if "scale" not in new_df.columns:
        #     # add scale factor to data
        #     if scale_factor is None:
        #         warnings.warn("No `scale_factor` added to data.")
        #     new_df["scale"] = [scale_factor] * len(new_df)
        # else:
        #     if verbose:
        #         print("Scale inferred from file.", flush=True)

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
            new_annotations_added = True # dataframe will be added later
            existing_str = ""
            old_n = 0
            new_n = len(annot_df)
        else:
            # concatenate old and new annoation dataframe
            annot_df = self[key]
            old_n = len(annot_df)
            annot_df = pd.concat([annot_df, new_df], ignore_index=False)

            # remove all duplicated shapes - leaving only the newly added
            annot_df = annot_df[~annot_df.index.duplicated()]
            new_n = len(annot_df)

            # collect additional variables for reporting
            new_annotations_added = new_n > old_n
            existing_str = "existing "

        if new_annotations_added:
            add = True
            if self._assert_uniqueness:
                # check if the shapes data for this key is unique (same number of names than indices)
                is_unique = self._check_uniqueness(dataframe=annot_df, key=key, verbose=False)

                if not is_unique:
                    add = False

            if self._polygons_only:
                # check if any of the shapes are shapely MultiPolygons
                is_not_polygon = [not isinstance(p, Polygon) for p in annot_df.geometry]
                if np.any(is_not_polygon):
                    annot_df = annot_df.loc[is_not_polygon]
                    warnings.warn(
                        f"Some {self._shape_name} were not pure Polygon objects and skipped.",
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
                    print(f"Added {new_n - old_n} new {self._shape_name} to {existing_str}key '{key}'")

    def crop(self,
             xlim, ylim
             ):
        limit_poly = Polygon([(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])])

        new_metadata = {}
        for i, n in enumerate(self._metadata.keys()):
            shapesdf = self[n]

            # select annotations that intersect with the selected area
            mask = [limit_poly.intersects(elem) for elem in shapesdf["geometry"]]
            shapesdf = shapesdf.loc[mask, :].copy()

            # move origin to zero after cropping
            shapesdf["geometry"] = shapesdf["geometry"].apply(affinity.translate, xoff=-xlim[0], yoff=-ylim[0])

            # check if there are annotations left or if it has to be deleted
            if len(shapesdf) > 0:
                # add new dataframe back to annotations object
                self._data[n] = shapesdf

                # update metadata
                new_metadata[n] = {}
                new_metadata[n][f"n_{self._shape_name}"] = len(shapesdf)
                new_metadata[n]["classes"] = shapesdf.name.unique().tolist()
                new_metadata[n]["analyzed"] = self._metadata[n]["analyzed"]  # analyzed information is just copied

            else:
                # delete annotations
                del self._data[n]

        self._metadata = new_metadata

        self._update_metadata()

    def remove_data(self,
                   key_to_remove: str,
                   classes_to_remove: Union[Literal["all"], List[str], str] = "all"
                   ):
        if classes_to_remove == "all":
            del self._data[key_to_remove]
        else:
            classes_to_remove = convert_to_list(classes_to_remove)
            geom_df = self[key_to_remove]
            self._data[key_to_remove] = geom_df[~geom_df.name.isin(classes_to_remove)]

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
        self._default_assert_uniqueness = False
        # self.default_skip_multipolygons = False
        self._default_polygons_only = False
        self._shape_name = "annotations"
        self._repr_color = tf.Cyan
        self._default_forbidden_names = FORBIDDEN_ANNOTATION_NAMES

        ShapesData.__init__(self, files, keys, pixel_size, shape_name=self._shape_name)

class RegionsData(ShapesData):
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None,
                 keys: Optional[List[str]] = None,
                 pixel_size: Optional[float] = None
                 ) -> None:
        self._default_assert_uniqueness = True
        # self.default_skip_multipolygons = True # MultiPolygons are not allowed in regions
        self._default_polygons_only = True
        self._shape_name = "regions"
        self._repr_color = tf.Yellow

        ShapesData.__init__(self, files, keys, pixel_size, shape_name=self._shape_name)

class BoundariesData(DeepCopyMixin):
    '''
    Object to read and load boundaries of cells and nuclei.
    '''
    def __init__(self,
                 cell_ids: Optional[da.core.Array],
                 seg_mask_value: Optional[da.core.Array],
                 #pixel_size: Number = 1, # required for boundaries that are saved as masks
                 ):
        """_summary_
        For details on `cell_ids` and `seg_mask_value` see: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/1.9/tutorials/outputs/xoa-output-zarr

        Args:
            cell_ids (Optional[da.core.Array]): _description_
            seg_mask_value (Optional[da.core.Array]): _description_
        """
        self._metadata = {}

        # store cell ids
        self._cell_ids = cell_ids
        self._seg_mask_value = seg_mask_value

        self._data = dict()

    def __repr__(self):
        labels = list(self._metadata.keys())
        if len(labels) == 0:
            repr = f"Empty BoundariesData object"
        else:
            ll = len(labels)
            repr = f"BoundariesData object with {ll} {'entry' if ll == 1 else 'entries'}:"
            for l in labels:
                repr += f"\n{tf.SPACER+tf.Bold+l+tf.ResetAll}"
        return repr
    
    def __getitem__(self, key):
        return self._data.get(key)

    def __setitem__(self, key: str, item):
        self._data[key] = item

    @property
    def metadata(self):
        return self._metadata

    @property
    def cell_ids(self):
        return self._cell_ids

    @property
    def seg_mask_value(self):
        return self._seg_mask_value

    def add_boundaries(self,
                       data: Optional[Union[dict, List[str]]],
                       pixel_size: Number, # required for boundaries that are saved as masks
                       labels: Optional[List[str]] = [],
                       overwrite: bool = False
                       ):
        if isinstance(data, dict):
            # extract keys from dictionary
            labels = data.keys()
            data = data.values()
        elif isinstance(data, list):
            if labels is None:
                raise ValueError("Argument 'labels' is None. If 'dataframes' is a list, 'labels' is required to be a list, too.")
            else:
                # make sure labels is a list
                labels = convert_to_list(labels)
        else:
            data = convert_to_list(data)
            labels = convert_to_list(labels)
            #raise ValueError(f"Argument 'dataframes' has unknown file type ({type(data)}). Expected to be a list or dictionary.")

        for l, df in zip(labels, data):
            if isinstance(df, pd.DataFrame) or isinstance(df, da.core.Array) or np.all([isinstance(elem, da.core.Array) for elem in df]):
                if l not in self._metadata or overwrite:
                    # add to object
                    self._data[l] = df
                    self._metadata[l] = {}
                    self._metadata[l]["pixel_size"] = pixel_size
                else:
                    raise KeyError(f"Label '{l}' exists already in BoundariesData object. To overwrite, set 'overwrite' argument to True.")
            else:
                print(f"Boundaries element `{l}` is neither a pandas DataFrame nor a Dask Array. Was not added.")

    def crop(self,
             cell_ids: List[str],
             xlim: Tuple[int, int],
             ylim: Tuple[int, int]
             ):
        '''
        Crop the BoundariesData object.
        '''
        # make sure cell ids are a list
        cell_ids = convert_to_list(cell_ids)

        for n, meta in self._metadata.items():
            # get dataframe
            data = self[n]

            try:
                # get pixel size
                pixel_size = meta["pixel_size"]

                data = crop_dask_array_or_pyramid(
                    data=data,
                    xlim=xlim,
                    ylim=ylim,
                    pixel_size=pixel_size
                )
            except InvalidDataTypeError:
                # filter dataframe
                data = data.loc[data["cell_id"].isin(cell_ids), :]

                # re-center to 0
                data["vertex_x"] -= xlim[0]
                data["vertex_y"] -= ylim[0]

            # add to object
            self._data[n] = data

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
             save_as_pyramid: bool = True
             ):

        suffix = bound_file.name.split(".", maxsplit=1)[-1]

        if suffix not in ["zarr", "zarr.zip"]:
            raise InvalidFileTypeError(allowed_types=[".zarr", ".zarr.zip"], received_type=suffix)

        with zarr.ZipStore(bound_file, mode='w') if suffix == "zarr.zip" else zarr.DirectoryStore(bound_file) as dirstore:
            # for conditional 'with' see also: https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
            for n in self._metadata.keys():
                bound_data = self[n]

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
                store[f"masks/{n}"].attrs.put(self._metadata[n])

                # save keys in insitupy metadata
                #metadata["boundaries"]["keys"].append(n)

            # save paths in insitupy metadata
            #metadata["boundaries"]["path"] = Path(relpath(bound_file, path)).as_posix()

            self._cell_ids.to_zarr(dirstore, component="cell_id")

            if self._seg_mask_value is not None:
                self._seg_mask_value.to_zarr(dirstore, component="seg_mask_value")

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

    def save(self,
             path: Union[str, os.PathLike, Path],
             boundaries_zipped: bool = False,
             boundaries_as_pyramid: bool = True,
             overwrite: bool = False,
             ):

        path = Path(path)
        celldata_metadata = {}

        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)

        # # create path for matrix
        # mtx_path = path / "matrix"
        # mtx_path.mkdir(parents=True, exist_ok=True) # create directory

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
            boundaries.save(bound_file, save_as_pyramid=True)

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


    def sync_cell_ids(self):
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
            # retrieve cell_ids of boundaries
            try:
                # this assumes the new version of data with integers representing hex code
                bound_cell_ids_int = boundaries.cell_ids[:, 0].compute()
                data_suffices = boundaries.cell_ids[:, 1].compute()

            except IndexError:
                # this means that the data has an older type where the cell ids are simply integers
                # since in this old data there is as far as I know no seg_mask_value, the integers are also the final ids
                assert boundaries.seg_mask_value is None, "Cell IDs had only one column but also a seg_mask_value was given. Needs to be checked."
                bound_cell_ids_final = boundaries.cell_ids.compute().astype(str)

                # check which ids are not in matrix cell ids
                seg_mask_values_not_in_matrix = np.array([elem for elem in bound_cell_ids_final if elem not in matrix_cell_ids_hex])

            else:
                # generate from integers and data suffix a xenium-style hex code
                bound_cell_ids_final = [convert_int_to_xenium_hex(ci, dataset_suffix=ds) for ci, ds in zip(bound_cell_ids_int, data_suffices)]

                # check which ids are not in matrix cell ids
                seg_mask_values = boundaries.seg_mask_value.compute()
                seg_mask_values_not_in_matrix = [smv for smv, cif in zip(seg_mask_values, bound_cell_ids_final) if cif not in matrix_cell_ids_hex]


            for n in boundaries.metadata.keys():
                # get data
                bound_data = boundaries[n]

                if isinstance(bound_data, list):
                    synced_bound_data = []
                    # iterate through list
                    for d in bound_data:
                        # set all non existent cell ids to zero
                        d[da.isin(d, seg_mask_values_not_in_matrix)] = 0
                        synced_bound_data.append(d)

                elif isinstance(bound_data, da.core.Array):
                    # set all non existent cell ids to zero
                    bound_data[da.isin(bound_data, seg_mask_values_not_in_matrix)] = 0
                    synced_bound_data = bound_data

                elif isinstance(bound_data, pd.DataFrame):
                    # filter dataframe
                    bound_data = bound_data.loc[bound_data["cell_id"].astype(str).isin(matrix_cell_ids_hex), :]
                    synced_bound_data = bound_data

                else:
                    warnings.warn(f"Unknown data type for boundaries key '{n}'. Skipped synchronization of cell ids.")
                # add to object
                self._boundaries[n] = synced_bound_data

            print(f"Successfully synchronized.")

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
        self._data = dict()
        self._key_main = None

    def __repr__(self):
        if self._key_main is not None:
            repr = (
                f"{tf.Bold+'Main CellData: '+tf.ResetAll}"
                f"{self._key_main}\n"
                f"{self._data[self._key_main].__repr__()}\n"
            )
        for key in self._data.keys():
            if key != self._key_main:
                repr += f"{tf.Bold+'Alternative CellData: '+tf.ResetAll}"
                repr += f"{key}\n"
                repr += f"{self._data[key].__repr__()}\n"
        return repr
    
    def __getitem__(self, key):
        if key == "main":
            return self._data.get(self._key_main)
        else:
            return self._data.get(key)

    def __setitem__(self, key: str, item):
        self._data[key] = item

    @property
    def is_main(self):
        return self._is_main

    def add_celldata(self,
                     cd: CellData,
                     key: str,
                     is_main: bool = False):
        if key in self._data.keys():
            print(f"Overwriting {key}.")
        self._data[key] = cd
        if is_main:
            self._key_main = key

    def save(self,
             path: Union[str, os.PathLike, Path],
             boundaries_zipped: bool = False,
             boundaries_as_pyramid: bool = True,
             overwrite: bool = False,
             ):

        path = Path(path)
        multicelldata_metadata = {"key_main": self._key_main, "all_keys": list(self._data.keys())}
        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)
        for key in self._data.keys():
            save_path = path / key
            self._data[key].save(save_path, boundaries_zipped, boundaries_as_pyramid, overwrite)


        # add version to metadata
        multicelldata_metadata["version"] = __version__

        # save metadata
        write_dict_to_json(dictionary=multicelldata_metadata, file=path / ".multicelldata")

    def get_all_keys(self):
        return self._data.keys()


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
                    ome_meta=None
                    )

    def __repr__(self):
        if len(self._metadata) > 0:
            repr_strings = [f"{tf.Bold}{n}:{tf.ResetAll}\t{metadata['shape']}" for n,metadata in self._metadata.items()]
            s = "\n".join(repr_strings)
        else:
            s = "empty"
        repr = f"{tf.Blue+tf.Bold}images{tf.ResetAll}\n{s}"
        return repr
    
    def __getitem__(self, key):
        return self._data.get(key)

    @property
    def metadata(self):
        return self._metadata

    @property
    def names(self):
        return self._names

    def add_image(
        self,
        image: Union[da.core.Array, np.ndarray, str, os.PathLike, Path],
        name: str,
        axes: Optional[str] = None, # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
        pixel_size: Optional[Number] = None,
        ome_meta: Optional[dict] = None,
        overwrite: bool = False
        ):
        if name in self._names:
            if not overwrite:
                print(f"`ImageData` object contains already an image with name '{name}'. Image is not added.")
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
            img_max = img[0].max() if isinstance(img, list) else img.max()
            img_max = img_max.compute() if isinstance(img, da.Array) else img_max
            #img_max = int(img_max)

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
            if len(img_shape) == 3:
                self._metadata[name]["rgb"] = True
            elif len(img_shape) == 2:
                self._metadata[name]["rgb"] = False
            else:
                raise ValueError(f"Unknown image shape: {img_shape}")

            # get image contrast limits
            if self._metadata[name]["rgb"]:
                self._metadata[name]["contrast_limits"] = (0, 255)
            else:
                self._metadata[name]["contrast_limits"] = (0, img_max)


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
             ylim: Optional[Tuple[int, int]]
             ):
        # extract names from metadata
        names = list(self._metadata.keys())
        for n in names:
            # extract the image pyramid
            img_data = self[n]

            # extract pixel size
            pixel_size = self._metadata[n]['pixel_size']

            cropped_img_data = crop_dask_array_or_pyramid(
                data=img_data,
                xlim=xlim,
                ylim=ylim,
                pixel_size=pixel_size
            )

            # save cropping properties in metadata
            self._metadata[n]["cropping_xlim"] = xlim
            self._metadata[n]["cropping_ylim"] = ylim

            try:
                self._metadata[n]["shape"] = cropped_img_data.shape
            except AttributeError:
                self._metadata[n]["shape"] = cropped_img_data[0].shape

            # add cropped pyramid to object
            self._data[n] = cropped_img_data

    def save(self,
             output_folder: Union[str, os.PathLike, Path],
             keys_to_save: Optional[str] = None,
             as_zarr: bool = True,
             zipped: bool = False,
             save_pyramid: bool = True,
             compression: Literal['jpeg', 'LZW', 'jpeg2000', 'ZLIB', None] = 'ZLIB', # jpeg2000 or ZLIB are recommended in the Xenium documentation - ZLIB is faster
             return_savepaths: bool = False,
             overwrite: bool = False,
             max_resolution: Optional[Number] = None # in µm per pixel
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
            overwrite (bool): If True, overwrite existing files in the output folder.

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

        for n, img_metadata in self._metadata.items():
            if n in keys_to_save:
                # extract image
                img = self[n]
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
                        filename = n + ".zarr.zip"
                    else:
                        # filename = Path(img_metadata["file"]).name.split(".")[0] + ".zarr"
                        filename = n + ".zarr"

                    # write to zarr
                    img_path = output_folder / filename
                    write_zarr(image=img, file=img_path,
                               img_metadata=new_img_metadata,
                               save_pyramid=save_pyramid,
                               axes=axes
                               )
                else:
                    # get file name for saving
                    #filename = Path(img_metadata["file"]).name.split(".")[0] + ".ome.tif"
                    filename = n + ".ome.tif"
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

                    print("start saving")
                    # write images as OME-TIFF
                    write_ome_tiff(image=img, file=output_folder / filename,
                                photometric=photometric, axes=axes,
                                compression=compression,
                                metadata=selected_metadata, overwrite=False)
                    print("end saving")

                if return_savepaths:
                    # collect savepaths
                    savepaths[n] = output_folder / filename

        if return_savepaths:
            return savepaths

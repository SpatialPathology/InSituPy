
import functools as ft
import os
import shutil
from copy import deepcopy
from datetime import datetime
from numbers import Number
from os.path import abspath
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4
from warnings import warn

import dask.dataframe as dd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from parse import *
from pyarrow import ArrowInvalid
from tqdm import tqdm

from insitupy import __version__
from insitupy._constants import (CACHE, ISPY_METADATA_FILE, LOAD_FUNCS,
                                 MODALITIES, MODALITIES_COLOR_DICT)
from insitupy._exceptions import (InSituDataRepeatedCropError,
                                  ModalityNotFoundError,
                                  ModalityNotFoundWarning)
from insitupy._io.files import (check_overwrite_and_remove_if_true, read_json,
                                write_dict_to_json)
from insitupy._textformat import textformat as tf
from insitupy._warnings import NoProjectLoadWarning
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.dataclasses.dataclasses import (AnnotationsData, ImageData,
                                              MultiCellData, RegionsData,
                                              SpatialUnitsData)
from insitupy.dataclasses.io import (_save_annotations, _save_cells,
                                     _save_images, _save_regions,
                                     _save_transcripts, _save_units,
                                     read_multicelldata, read_shapesdata)
from insitupy.utils._helpers import sort_paths_by_datetime
from insitupy.utils.geo import fast_query_points_within_polygon
from insitupy.utils.utils import _crop_transcripts, convert_to_list


class InSituData:
    """
    InSituData class for managing and analyzing spatially resolved transcriptomics data.

    .. figure:: ../../_static/img/insitudata_overview.svg
       :width: 500px
       :align: right
       :class: dark-light

    It provides methods for loading, saving, visualizing, and manipulating various modalities
    of data, such as images, cells, annotations, regions, and transcripts.

    Attributes:
        images (ImageData): Image data associated with the object.
        cells (MultiCellData): Cell data associated with the object.
        annotations (AnnotationsData): Annotation data associated with the object.
        regions (RegionsData): Region data associated with the object.
        transcripts (pd.DataFrame): Transcript data associated with the object.

        path (Union[str, os.PathLike, Path]): Path to the data directory.
        metadata (dict): Metadata associated with the InSituData object.
        slide_id (str): Identifier for the slide.
        sample_id (str): Identifier for the sample.
        from_insitudata (bool): Indicates whether the object was loaded from an InSituData project.

        viewer (napari.Viewer): Napari viewer for visualizing the data.
        quicksave_dir (Path): *Experimental feature!* Directory for quicksave operations.

    Methods:
        assign_geometries(geometry_type, keys, add_masks, add_to_obs, overwrite, cells_layer):
            Assigns geometries (annotations or regions) to the cell data.
        assign_annotations(keys, add_masks, overwrite):
            Assigns annotations to the cell data.
        assign_regions(keys, add_masks, overwrite):
            Assigns regions to the cell data.
        copy(keep_path):
            Creates a deep copy of the InSituData object.
        crop(region_tuple, xlim, ylim, inplace, verbose):
            Crops the data based on the provided parameters.
        plot_dimred(save):
            Plots dimensionality reduction results.
        load_all(skip, verbose):
            Loads all available modalities.
        load_annotations(verbose):
            Loads annotation data.
        import_annotations(files, keys, scale_factor, verbose):
            Imports annotation data from external files.
        load_regions(verbose):
            Loads region data.
        import_regions(files, keys, scale_factor, verbose):
            Imports region data from external files.
        load_cells(verbose):
            Loads cell data.
        load_images(names, overwrite, verbose):
            Loads image data.
        load_transcripts(verbose, mode):
            Loads transcript data.
        read(path):
            Reads an InSituData object from a specified folder.
        saveas(path, overwrite, zip_output, images_as_zarr, zarr_zipped, images_max_resolution, verbose):
            Saves the InSituData object to a specified path.
        save(path, zarr_zipped, verbose, keep_history):
            Saves the InSituData object to its current path or a specified path.
        save_colorlegends(savepath, from_canvas, max_per_col):
            Saves color legends from the viewer.
        quicksave(note):
            *Experimental feature!* Saves a quick snapshot of the annotations.
        list_quicksaves():
            *Experimental feature!* Lists all available quicksaves.
        load_quicksave(uid):
            *Experimental feature!* Loads a quicksave by its unique identifier.
        show(keys, cells_layer, point_size, scalebar, unit, grayscale_colormap, return_viewer, widgets_max_width):
            Visualizes the data using a napari viewer.
        store_geometries(name_pattern, uid_col):
            Extracts geometric layers from the viewer and stores them as annotations or regions.
        reload(skip, verbose):
            Reloads the loaded modalities.
        get_loaded_modalities():
            Returns a list of currently loaded modalities.
        remove_history(verbose):
            Removes the history of saved modalities.
        remove_modality(modality):
            Removes a specific modality from the object.

    """

    # import deprecated functions
    from ._deprecated import (add_alt, add_baysor, normalize_and_transform,
                              read_all, read_annotations, read_cells,
                              read_images, read_regions, read_transcripts,
                              reduce_dimensions, save_colorlegends,
                              save_current_colorlegend, store_geometries,
                              sync_geometries)

    def __init__(self,
                 path: Optional[Union[str, os.PathLike, Path]] = None,
                 metadata: Optional[dict] = None,
                 slide_id: Optional[str] = None,
                 sample_id: Optional[str] = None,
                 method_name: str = "not specified",
                 method_params: dict = dict(),
                 pixel_size: Number = 1
                 ):
        """
        """
        # metadata
        if path is not None:
            self._path = Path(path)
        else:
            self._path = None
        self._slide_id = slide_id
        self._sample_id = sample_id

        if metadata is None:
            # initialize metadata
            self._metadata = {}
            self._metadata["data"] = {}
            self._metadata["history"] = {}
            self._metadata["history"]["cells"] = []
            self._metadata["history"]["annotations"] = []
            self._metadata["history"]["regions"] = []
            self._metadata["uids"] = [str(uuid4())] # initialize the uid section
            self._metadata["method"] = method_name
        else:
            self._metadata = metadata

        # add method parameters
        assert isinstance(method_params, dict), "`method_params` must be a dictionary."
        self._metadata["method_params"] = method_params

        # modalities
        self._images = ImageData()
        self._cells = MultiCellData()
        self._units = None
        self._annotations = AnnotationsData()
        self._regions = RegionsData()
        self._transcripts = None

        # other
        #self._viewer = None
        self._quicksave_dir = None

    def __repr__(self):
        # if len(self._metadata) == 0:
        #     method = "unknown"
        # else:
        try:
            method = self._metadata["method"]
        except KeyError:
            method = "unknown"

        if self._path is not None:
            self._path = self._path.resolve()

        # check if all modalities are empty
        empty_checks = [elem.is_empty for elem in [
            self._images, self._cells, self._annotations, self._regions
            ]] + [self._transcripts is None, self._units is None] # transcripts and units do not have is_empty property since they are dataframes
        all_empty = np.all(empty_checks)

        repr = (
            f"{tf.Bold+tf.Red}InSituData{tf.ResetAll}\n"
            f"{tf.Bold}Method:{tf.ResetAll}\t\t{method}\n"
            f"{tf.Bold}Slide ID:{tf.ResetAll}\t{self._slide_id}\n"
            f"{tf.Bold}Sample ID:{tf.ResetAll}\t{self._sample_id}\n"
            f"{tf.Bold}Path:{tf.ResetAll}\t\t{self._path}\n"
        )

        # #if self._metadata is not None:
        # if "metadata_file" in self._metadata:
        #     mfile = self._metadata["metadata_file"]
        # else:
        #     mfile = None
        # # else:
        # #     mfile = None

        # repr += f"{tf.Bold}Metadata file:{tf.ResetAll}\t{mfile}"

        if all_empty:
            repr += "\n\nNo modalities loaded."
        else:
            if not self._images.is_empty:
                images_repr = self._images.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['images']+tf.Bold} images{tf.ResetAll}\n{tf.SPACER}   " + images_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if not self._cells.is_empty:
                cells_repr = self._cells.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['cells']+tf.Bold} cells{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if self._units is not None:
                units_repr = self._units.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['units']+tf.Bold} units{tf.ResetAll}\n{tf.SPACER}   " + units_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if not self._annotations.is_empty:
                annot_repr = self._annotations.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['annotations']+tf.Bold} annotations{tf.ResetAll}\n{tf.SPACER}   " + annot_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if not self._regions.is_empty:
                region_repr = self._regions.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['regions']+tf.Bold} regions{tf.ResetAll}\n{tf.SPACER}   " + region_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if self._transcripts is not None:
                trans_repr = f"DataFrame with shape {self._transcripts.shape[0]} x {self._transcripts.shape[1]}"

                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['transcripts']+tf.Bold} transcripts{tf.ResetAll}\n{tf.SPACER}   " + trans_repr
                )
        return repr


    @property
    def path(self):
        """Return save path of the InSituData object.
        Returns:
            str: Save path.
        """
        return self._path

    @property
    def metadata(self):
        """Return metadata of the InSituData object.
        Returns:
            dict: Metadata.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        raise AttributeError("Cannot modify 'metadata' attribute after initialization.")

    @property
    def slide_id(self):
        """Return slide id of the InSituData object.
        Returns:
            str: Slide id.
        """
        return self._slide_id

    @property
    def sample_id(self):
        """Return sample id of the InSituData object.
        Returns:
            str: Sample id.
        """
        return self._sample_id

    @property
    def from_insitudata(self):
        if self._path is not None:
            if Path(self._path).exists():
                return True
            else:
                print(f"Path {str(self._path)} does not exist.")
                return False
        else:
            return False

    @property
    def images(self):
        """Return images of the InSituData object.
        Returns:
            insitupy._core.dataclasses.ImageData: Images.
        """
        return self._images

    @images.setter
    def images(self, value):
        raise AttributeError("Cannot modify 'cells' attribute after initialization.")

    @images.deleter
    def images(self):
        self._images = ImageData()
        print("Cleared all data from 'images'.")

    @property
    def cells(self):
        """Return cell data of the InSituData object.
        Returns:
            insitupy._core.dataclasses.MultiCellData: Cell data.
        """
        return self._cells

    @cells.setter
    def cells(self, value):
        raise AttributeError("Cannot modify 'cells' attribute after initialization.")

    @cells.deleter
    def cells(self):
        self._cells = MultiCellData()
        print("Cleared all data from 'cells'.")

    @property
    def units(self):
        """Return spatial units data of the InSituData object.
        Returns:
            insitupy._core.dataclasses.SpatialUnitsData: Spatial units data.
        """
        return self._units

    @units.setter
    def units(self, value):
        raise AttributeError("Cannot modify 'units' attribute after initialization.")

    @units.deleter
    def units(self):
        self._units = None
        print("Cleared all data from 'units'.")

    def add_units(self, data: SpatialUnitsData):
        """
        Add spatial units data to the InSituData object.

        Args:
            data (SpatialUnitsData): The spatial units data to add.

        Raises:
            TypeError: If data is not of type SpatialUnitsData.
        """
        if not isinstance(data, SpatialUnitsData):
            raise TypeError(f"Data must be of type SpatialUnitsData, but got {type(data).__name__} instead.")

        self._units = data

    @property
    def annotations(self):
        """Return annotations of the InSituData object.
        Returns:
            insitupy._core.dataclasses.AnnotationsData: Annotations.
        """
        return self._annotations

    @annotations.setter
    def annotations(self, value):
        raise AttributeError("Cannot modify 'annotations' attribute after initialization.")

    @annotations.deleter
    def annotations(self):
        self._annotations = AnnotationsData()
        print("Cleared all data from 'annotations'.")

    @property
    def regions(self):
        """Return regions of the InSituData object.
        Returns:
            insitupy._core.dataclasses.RegionsData: Regions.
        """
        return self._regions

    @regions.setter
    def regions(self, value):
        raise AttributeError("Cannot modify 'regions' attribute after initialization.")

    @regions.deleter
    def regions(self):
        self._regions = RegionsData()
        print("Cleared all data from 'regions'.")

    @property
    def transcripts(self):
        """Return transcripts of the InSituData object.
        Returns:
            pd.DataFrame: Transcripts.
        """
        return self._transcripts

    @transcripts.setter
    def transcripts(self, value: dd.DataFrame):
        if isinstance(value, dd.DataFrame):
            self._transcripts = value
        elif isinstance(value, pd.DataFrame):
            self._transcripts = dd.from_pandas(value, npartitions=8)
        else:
            raise ValueError(f"Value must be of type dask.dataframe.DataFrame, but got {type(value)} instead.")

    @transcripts.deleter
    def transcripts(self):
        self._transcripts = None


    def assign_geometries(self,
                          geometry_type: Literal["annotations", "regions"],
                          keys: Union[str, Literal["all"]] = "all",
                          add_masks: bool = False,
                          add_to_obs: bool = False,
                          overwrite: bool = True,
                          cells_layer: str = None
                          ):
        '''
        Function to assign geometries (annotations or regions) to the anndata object in
        InSituData.cells[layer].table. Assignment information is added to the DataFrame in `.obs`.
        '''
        # assert that prerequisites are met
        try:
            geom_attr = getattr(self, geometry_type)
        except AttributeError:
            raise ModalityNotFoundError(modality=geometry_type)

        # get the right cells layer
        celldata, cells_layer_name = _get_cell_layer(
            cells=self.cells, cells_layer=cells_layer,
            verbose=True, return_layer_name=True
            )
        name = f".cells['{cells_layer_name}']"

        if keys == "all":
            keys = geom_attr.metadata.keys()

        # make sure annotation keys are a list
        keys = convert_to_list(keys)

        # convert coordinates into shapely Point objects
        x = celldata.table.obsm["spatial"][:, 0]
        y = celldata.table.obsm["spatial"][:, 1]
        cells = gpd.points_from_xy(x, y)
        cells = gpd.GeoSeries(cells)

        # iterate through annotation keys
        for key in keys:
            print(f"Assigning key '{key}'...")
            if key not in geom_attr.keys():
                raise KeyError(f"Key '{key}' not found in {geometry_type}.")

            # extract pandas dataframe of current key
            geom_df = geom_attr[key]

            # make sure the geom names do not contain any ampersand string (' % '),
            # since this would interfere with the downstream analysis
            if geom_df["name"].str.contains(' & ').any():
                raise ValueError(
                    f"The {geometry_type} with key '{key}' contains names with the ampersand string ' & '. "
                    f"This is not allowed as it would interfere with downstream analysis."
                    )

            # get unique list of annotation names
            geom_names = geom_df.name.unique()

            # initiate dataframe as dictionary
            data = {}

            # iterate through names
            for n in tqdm(geom_names):
                polygons = geom_df[geom_df["name"] == n]["geometry"].tolist()

                #in_poly = [poly.contains(cells) for poly in polygons]
                in_poly = [fast_query_points_within_polygon(poly, cells) for poly in polygons]

                # check if points were in any of the polygons
                in_poly_res = np.array(in_poly).any(axis=0)

                # collect results
                data[n] = in_poly_res

            # convert into pandas dataframe
            data = pd.DataFrame(data)
            data.index = celldata.table.obs_names

            # transform data into one column
            column_to_add = [" & ".join(geom_names[row.values]) if np.any(row.values) else "unassigned" for _, row in data.iterrows()]

            if add_to_obs:
                # create annotation from annotation masks
                col_name = f"{geometry_type}-{key}"
                data[col_name] = column_to_add
                if col_name in celldata.table.obs:
                    if overwrite:
                        celldata.table.obs.drop(col_name, axis=1, inplace=True)
                        print(f'Existing column "{col_name}" is overwritten.', flush=True)
                        add = True
                    else:
                        warn(f'Column "{col_name}" exists already in `{name}.table.obs`. Assignment of key "{key}" was skipped. To force assignment, select `overwrite=True`.')
                        add = False
                else:
                    add = True

                if add:
                    if add_masks:
                        celldata.table.obs = pd.merge(left=celldata.table.obs, right=data, left_index=True, right_index=True)
                    else:
                        celldata.table.obs = pd.merge(left=celldata.table.obs, right=data.iloc[:, -1], left_index=True, right_index=True)

                    # save that the current key was analyzed
                    geom_attr.metadata[key]["analyzed"] = tf.TICK
            else:
                # add to obsm
                obsm_keys = celldata.table.obsm.keys()
                if geometry_type not in obsm_keys:
                    # add empty pandas dataframe with obs_names as index
                    celldata.table.obsm[geometry_type] = pd.DataFrame(index=celldata.table.obs_names)

                celldata.table.obsm[geometry_type][key] = column_to_add

                # save that the current key was analyzed
                geom_attr.metadata[key]["analyzed"] = tf.TICK

                print(f"Added results to `{name}.table.obsm['{geometry_type}']", flush=True)


    def assign_annotations(
        self,
        keys: Union[str, Literal["all"]] = "all",
        cells_layers: Optional[Union[List[str], str]] = None,
        add_masks: bool = False,
        overwrite: bool = True
    ):
        if cells_layers is None:
            layers_list = self._cells.keys()
        else:
            layers_list = convert_to_list(cells_layers)

        for l in layers_list:
            self.assign_geometries(
                geometry_type="annotations",
                keys=keys,
                add_masks=add_masks,
                overwrite=overwrite,
                cells_layer=l
            )

    def assign_regions(
        self,
        keys: Union[str, Literal["all"]] = "all",
        cells_layers: Optional[Union[List[str], str]] = None,
        add_masks: bool = False,
        overwrite: bool = True
    ):
        if cells_layers is None:
            layers_list = self._cells.keys()
        else:
            layers_list = convert_to_list(cells_layers)

        for l in layers_list:
            self.assign_geometries(
                geometry_type="regions",
                keys=keys,
                add_masks=add_masks,
                overwrite=overwrite,
                cells_layer=l
            )

    def copy(self, keep_path: bool = False):
        '''
        Function to generate a deep copy of the InSituData object.
        '''
        self_copy = deepcopy(self)

        if not keep_path:
            self_copy._path = None
            self_copy.metadata["path"] = None
        return self_copy

    def crop(self,
             region_tuple: Optional[Tuple[str, str]] = None,
             xlim: Optional[Tuple[int, int]] = None,
             ylim: Optional[Tuple[int, int]] = None,
             inplace: bool = False,
             verbose: bool = False
            ):
        """
        Crop the data based on the provided parameters.

        Args:
            region_tuple (Optional[Tuple[str, str]]): A tuple specifying the region to crop.
            xlim (Optional[Tuple[int, int]]): The x-axis limits for cropping.
            ylim (Optional[Tuple[int, int]]): The y-axis limits for cropping.
            inplace (bool): If True, modify the data in place. Otherwise, return a new cropped data.

        Raises:
            ValueError: If none of region_tuple, layer_name, or xlim/ylim are provided.
        """
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        if region_tuple is None:
            if xlim is None or ylim is None:
                raise ValueError("If shape is None, both xlim and ylim must not be None.")

            # make sure there are no negative values in the limits
            xlim = tuple(np.clip(xlim, a_min=0, a_max=None))
            ylim = tuple(np.clip(ylim, a_min=0, a_max=None))
            shape = None
        else:
            # extract regions dataframe
            region_key = region_tuple[0]
            region_name = region_tuple[1]
            region_df = self._regions[region_key]

            if region_name in region_df["name"].unique():
                # extract geometry
                shape = region_df[region_df["name"] == region_name]["geometry"].item()
                #use_shape = True
            else:
                raise ValueError(f"Region name '{region_name}' not found in regions with key '{region_key}'.")

            # extract x and y limits from the geometry
            minx, miny, maxx, maxy = shape.bounds # (minx, miny, maxx, maxy)
            xlim = (minx, maxx)
            ylim = (miny, maxy)

        try:
            # if the object was previously cropped, check if the current window is identical with the previous one
            if np.all([elem in _self.metadata["method_params"].keys() for elem in ["cropping_xlim", "cropping_ylim"]]):
                # test whether the limits are identical
                if (xlim == _self.metadata["method_params"]["cropping_xlim"]) & (ylim == _self.metadata["method_params"]["cropping_ylim"]):
                    raise InSituDataRepeatedCropError(xlim, ylim)
        except TypeError:
            pass

        if not _self.cells.is_empty:
            _self.cells.crop(
                shape=shape,
                xlim=xlim, ylim=ylim,
                inplace=True, verbose=False
            )

        if _self.transcripts is not None:
            _self.transcripts = _crop_transcripts(
                transcript_df=_self.transcripts,
                shape=shape,
                xlim=xlim, ylim=ylim, verbose=verbose
            )

        if not self._images.is_empty:
            _self.images.crop(xlim=xlim, ylim=ylim, inplace=True)

        if not self._annotations.is_empty:

            _self.annotations.crop(
                shape=shape,
                xlim=tuple([elem for elem in xlim]),
                ylim=tuple([elem for elem in ylim]),
                verbose=verbose, inplace=True
                )

        if not self._regions.is_empty:
            _self.regions.crop(
                shape=shape,
                xlim=tuple([elem for elem in xlim]),
                ylim=tuple([elem for elem in ylim]),
                verbose=verbose, inplace=True
            )

        #if _self.metadata is not None:
        # add information about cropping to metadata
        if "cropping_history" not in _self.metadata:
            _self.metadata["cropping_history"] = {}
            _self.metadata["cropping_history"]["xlim"] = []
            _self.metadata["cropping_history"]["ylim"] = []
        _self.metadata["cropping_history"]["xlim"].append(tuple([int(elem) for elem in xlim]))
        _self.metadata["cropping_history"]["ylim"].append(tuple([int(elem) for elem in ylim]))

        # add new uid to uid history
        _self.metadata["uids"].append(str(uuid4()))

        # empty current data and data history entry in metadata
        _self.metadata["data"] = {}
        for k in _self.metadata["history"].keys():
            _self.metadata["history"][k] = []

        if not inplace:
            return _self

    def transform(
        self,
        transformation_matrix: Union[np.ndarray, str, os.PathLike, Path],
        source_pixel_size: Optional[Number] = None,
        reference_pixel_size: Optional[Number] = None,
        output_size: Optional[Tuple[Number, Number]] = None,
        inplace: bool = False,
        verbose: bool = False
    ):
        """
        Apply an affine transformation to the InSituData object (Images and Features).

        Args:
            transformation_matrix: Either a 2x3 or 3x3 numpy array representing
                the affine transformation matrix, or a path to a CSV/Excel file
                containing the matrix.
            source_pixel_size: Pixel size (in µm/pixel) of the source image from
                which the transformation matrix was derived.
            reference_pixel_size: Pixel size (in µm/pixel) of the reference image
                used during registration.
            output_size: Tuple of (height, width) in physical coordinates (µm)
                specifying the desired output canvas size.
            inplace: If True, modify the object in place. Otherwise, return a
                transformed copy. Defaults to False.
            verbose: If True, print status messages. Defaults to False.

        Returns:
            InSituData: Transformed InSituData object if inplace=False, else None.
        """
        if inplace:
            _self = self
        else:
            _self = self.copy()

        # Transform images
        if not _self.images.is_empty:
            if verbose:
                print("Transforming images...")
            _self.images.transform(
                transformation_matrix=transformation_matrix,
                reference_pixel_size=reference_pixel_size,
                source_pixel_size=source_pixel_size,
                output_size=output_size,
                inplace=True,
                verbose=verbose
            )

        # Transform units
        if _self.units is not None:
            if verbose:
                print("Transforming units...")
            _self.units.transform(
                transformation_matrix=transformation_matrix,
                reference_pixel_size=reference_pixel_size,
                source_pixel_size=source_pixel_size,
                inplace=True,
                verbose=verbose
            )

        if not inplace:
            return _self

    def align_units(
        self,
        other: "InSituData",
        transformation_matrix: Union[np.ndarray, str, os.PathLike, Path],
        source_image_name: Optional[str] = None,
        reference_image_name: Optional[str] = None,
        source_pixel_size: Optional[Number] = None,
        reference_pixel_size: Optional[Number] = None,
        transfer_images: bool = False,
        verbose: bool = False
    ):
        """
        Align units from another InSituData object to this one.

        This function takes units from another InSituData object, applies a
        transformation, and adds them to the current object. It is designed for
        integrating units (e.g., from Visium) onto a high-resolution dataset
        (e.g., Xenium) after registration.

        If `transfer_images` is True and the source object (other) contains images,
        they are also transformed and added to the current object.

        Args:
            other (InSituData): The InSituData object containing the units to align.
            transformation_matrix: Transformation matrix to align the units.
            source_image_name: Name of the source image in `other.images` to infer `source_pixel_size`.
            reference_image_name: Name of the reference image in `self.images` to infer `reference_pixel_size`.
            source_pixel_size: Pixel size (in µm/pixel) of the source image (origin of units).
            reference_pixel_size: Pixel size (in µm/pixel) of the reference image (target).
            transfer_images: If True, transfer images from `other` to `self`. Defaults to False.
            verbose: If True, print status messages.

        Raises:
            ValueError: If the configuration of self or other is invalid.
        """
        # Check configuration of self
        if self.cells.is_empty:
            warn("The target InSituData object (self) has no cells. "
                 "Alignment is typically done onto a dataset with cells.")

        if self.units is not None:
            raise ValueError("The target InSituData object (self) already has spatial units. "
                             "Please remove them before aligning new units.")

        # Check configuration of other
        if other.units is None:
            raise ValueError("The source InSituData object (other) has no spatial units to align.")

        if not other.cells.is_empty:
            warn("The source InSituData object (other) has cells. "
                 "Typically, the source object should only contain spatial units to be aligned.")

        # Determine reference pixel size
        if reference_pixel_size is None and reference_image_name is not None:
            try:
                reference_pixel_size = self.images.metadata[reference_image_name]["pixel_size"]
            except KeyError:
                raise ValueError(f"Reference image '{reference_image_name}' not found in self.images.")

        # Determine source pixel size
        if source_pixel_size is None and source_image_name is not None:
            try:
                source_pixel_size = other.images.metadata[source_image_name]["pixel_size"]
            except KeyError:
                raise ValueError(f"Source image '{source_image_name}' not found in other.images.")

        # Copy units from other
        units_to_add = other.units.copy()

        # Transform units
        if verbose:
            print("Transforming and aligning spatial units...")

        units_to_add.transform(
            transformation_matrix=transformation_matrix,
            reference_pixel_size=reference_pixel_size,
            source_pixel_size=source_pixel_size,
            inplace=True,
            verbose=verbose
        )

        # Add to self
        self._units = units_to_add

        if verbose:
            print("Spatial units aligned and added to InSituData object.")

        # Align images
        if transfer_images and not other.images.is_empty:
            if verbose:
                print("Transforming and aligning images...")

            images_to_add = other.images.copy()
            images_to_add.transform(
                transformation_matrix=transformation_matrix,
                reference_pixel_size=reference_pixel_size,
                source_pixel_size=source_pixel_size,
                inplace=True,
                verbose=verbose
            )

            for name in images_to_add.names:
                img = images_to_add[name]
                if isinstance(img, list):
                    img = img[0]
                self.images.add_image(
                    image=img,
                    channel_names=name,
                    axes=images_to_add.metadata[name]["axes"],
                    pixel_size=images_to_add.metadata[name]["pixel_size"],
                    ome_meta=images_to_add.metadata[name].get("OME", {}),
                    is_rgb=images_to_add.metadata[name].get("rgb", None),
                    overwrite=True,
                    verbose=verbose
                )

            if verbose:
                print("Images aligned and added to InSituData object.")

    def plot_dimred(self, save: Optional[str] = None):
        '''
        Read dimensionality reduction plots.
        '''
        # construct paths
        analysis_path = self._path / "analysis"
        umap_file = analysis_path / "umap" / "gene_expression_2_components" / "projection.csv"
        pca_file = analysis_path / "pca" / "gene_expression_10_components" / "projection.csv"
        cluster_file = analysis_path / "clustering" / "gene_expression_graphclust" / "clusters.csv"


        # read data
        umap_data = pd.read_csv(umap_file)
        pca_data = pd.read_csv(pca_file)
        cluster_data = pd.read_csv(cluster_file)

        # merge dimred data with clustering data
        data = ft.reduce(lambda left, right: pd.merge(left, right, on='Barcode'), [umap_data, pca_data.iloc[:, :3], cluster_data])
        data["Cluster"] = data["Cluster"].astype('category')

        # plot
        nrows = 1
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
        sns.scatterplot(data=data, x="PC-1", y="PC-2", hue="Cluster", palette="tab20", ax=axs[0])
        sns.scatterplot(data=data, x="UMAP-1", y="UMAP-2", hue="Cluster", palette="tab20", ax=axs[1])
        if save is not None:
            plt.savefig(save)
        plt.show()

    def load_all(self,
                 skip: Optional[str] = None,
                 verbose: bool = False
                 ):
        # # extract read functions
        # read_funcs = [elem for elem in dir(self) if elem.startswith("load_")]
        # read_funcs = [elem for elem in read_funcs if elem not in ["load_all", "load_quicksave"]]
        for f in LOAD_FUNCS:
            if skip is None or skip not in f:
                func = getattr(self, f)
                # try:
                func(verbose=verbose)
                # except ModalityNotFoundError as err:
                #     if verbose:
                #         print(err)

    def load_annotations(self, verbose: bool = False):
        if verbose:
            print("Loading annotations...", flush=True)
        # try:
        #     p = self._metadata["data"]["annotations"]
        # except KeyError:
        #     if verbose:
        #         raise ModalityNotFoundError(modality="annotations")
        # extract available paths
        paths = [p for p in (self.path / "annotations").glob("[!.]*") if p.is_dir()]

        if len(paths) == 0:
            if verbose:
                # Example usage
                warn(ModalityNotFoundWarning("annotations"), stacklevel=2)
        else:
            # extract the latest entry
            path = sort_paths_by_datetime(paths)[0]
            self._annotations = read_shapesdata(path=path, mode="annotations")


    def import_annotations(self,
                           files: Optional[Union[str, os.PathLike, Path]],
                           keys: Optional[str],
                           scale_factor: Number, # µm/pixel - can be used to convert the pixel coordinates into µm coordinates
                           verbose: bool = False
                           ):
        if verbose:
            print("Importing annotations...", flush=True)

        # add annotations object
        files = convert_to_list(files)
        keys = convert_to_list(keys)

        if len(files) != len(keys):
            raise ValueError("Length of files and keys must be the same.")

        # if self._annotations is None:
        #     self._annotations = AnnotationsData()

        for key, file in zip(keys, files):
            # read annotation and store in dictionary
            self._annotations.add_data(
                data=file,
                key=key,
                scale_factor=scale_factor
                )

        #self._remove_empty_modalities()

    def load_regions(self, verbose: bool = False):
        if verbose:
            print("Loading regions...", flush=True)
        # try:
        #     p = self._metadata["data"]["regions"]
        # except KeyError:
        #     if verbose:
        #         raise ModalityNotFoundError(modality="regions")

        # extract available paths
        paths = [p for p in (self.path / "regions").glob("[!.]*") if p.is_dir()]

        if len(paths) == 0:
            if verbose:
                warn(ModalityNotFoundWarning("regions"), stacklevel=2)
        else:
            # extract the latest entry
            path = sort_paths_by_datetime(paths)[0]
            self._regions = read_shapesdata(path=path, mode="regions")

    def import_regions(self,
                    files: Optional[Union[str, os.PathLike, Path]],
                    keys: Optional[str],
                    scale_factor: Number, # µm/pixel - used to convert the pixel coordinates into µm coordinates
                    verbose: bool = False
                    ):
        if verbose:
            print("Importing regions...", flush=True)

        # add regions object
        files = convert_to_list(files)
        keys = convert_to_list(keys)

        if len(files) != len(keys):
            raise ValueError("Length of files and keys must be the same.")


        # if self._regions is None:
        #     self._regions = RegionsData()

        for key, file in zip(keys, files):
            # read annotation and store in dictionary
            self._regions.add_data(data=file,
                                key=key,
                                scale_factor=scale_factor
                                )

        #self._remove_empty_modalities()


    def load_cells(self, verbose: bool = False):
        if verbose:
            print("Loading cells...", flush=True)

        if self.from_insitudata:
            # try:
            #     cells_path = self._metadata["data"]["cells"]
            # except KeyError:
            #     if verbose:
            #         raise ModalityNotFoundError(modality="cells")

            # extract available paths
            paths = [p for p in (self.path / "cells").glob("[!.]*") if p.is_dir()]

            if len(paths) == 0:
                if verbose:
                    warn(ModalityNotFoundWarning("cells"), stacklevel=2)
            else:
                # extract the latest entry
                path = sort_paths_by_datetime(paths)[0]
                self._cells = read_multicelldata(path=path)
        else:
            NoProjectLoadWarning()

    def load_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    overwrite: bool = False,
                    verbose: bool = False
                    ):
        # load image into ImageData object
        if verbose:
            print("Loading images...", flush=True)

        if self.from_insitudata:
            # check if image data is stored in this InSituData
            # try:
            #     images_dict = self._metadata["data"]["images"]
            # except KeyError:
            #     if verbose:
            #         raise ModalityNotFoundError(modality="images")

            img_paths = list((self.path / "images").glob("[!.]*.zarr"))
            if len(img_paths) == 0:
                if verbose:
                    warn(ModalityNotFoundWarning("images"), stacklevel=2)
            else:
                img_names = [p.stem for p in img_paths]

                if names != "all":
                    names = convert_to_list(names)
                    if not np.all([elem in img_names for elem in names]):
                        not_available = [elem for elem in names if elem not in img_names]
                        raise ValueError(f"Following 'names' are not available: {not_available}")
                    img_names = names

                # if names == "all":
                #     img_names = list(images_dict.keys())
                # else:
                #     img_names = convert_to_list(names)

                # # get file paths and names
                # img_files = [v for k,v in images_dict.items() if k in img_names]
                # img_names = [k for k,v in images_dict.items() if k in img_names]

                # # create imageData object
                # img_paths = [self._path / elem for elem in img_files]

                # if self._images is None:
                #     self._images = ImageData(img_paths, img_names)
                # else:
                for im, n in zip(img_paths, img_names):
                    self._images.add_image(im, n, overwrite=overwrite, verbose=verbose)

        else:
            NoProjectLoadWarning()

    def load_transcripts(self,
                        verbose: bool = False,
                        mode: Literal["pandas", "dask"] = "dask",
                        ):
        # read transcripts
        if verbose:
            print("Loading transcripts...", flush=True)

        if self.from_insitudata:
            # # check if transcript data is stored in this InSituData
            # try:
            #     transcripts_path = self._metadata["data"]["transcripts"]
            # except KeyError:
            #     if verbose:
            #         raise ModalityNotFoundError(modality="transcripts")

            # extract available paths
            transcripts_path = Path(self.path) / "transcripts/transcripts.parquet"

            if not transcripts_path.exists():
                if verbose:
                    warn(ModalityNotFoundWarning("transcripts"), stacklevel=2)
            else:
                if mode == "pandas":
                    self._transcripts = pd.read_parquet(transcripts_path)
                elif mode == "dask":
                    # Load the transcript data using Dask
                    try:
                        self._transcripts = dd.read_parquet(transcripts_path)
                    except ArrowInvalid:
                        parquet_files = list(Path(transcripts_path).glob("part*.parquet"))
                        self._transcripts = dd.read_parquet(parquet_files)
                else:
                    raise ValueError(f"Invalid value for `mode`: {mode}")
        else:
            NoProjectLoadWarning()

    def load_units(self,
                     verbose: bool = False
                     ):
        # read units
        if verbose:
            print("Loading spatial units...", flush=True)

        if self.from_insitudata:
            # extract available paths
            units_path = Path(self.path) / "units"

            if not units_path.exists():
                if verbose:
                    warn(ModalityNotFoundWarning("units"), stacklevel=2)
            else:
                import json

                import geopandas as gpd
                from anndata import read_h5ad

                # Load shapes
                shapes_file = units_path / "shapes.parquet"
                shapes = gpd.read_parquet(shapes_file)

                # Load data if present
                data_file = units_path / "data.h5ad"
                data = read_h5ad(data_file) if data_file.exists() else None

                # Load metadata
                meta_file = units_path / "metadata.json"
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta_dict = json.load(f)
                    unit_type = meta_dict.get("unit_type", "unit")
                else:
                    unit_type = "unit"

                # Create SpatialUnitsData object and assign
                self._units = SpatialUnitsData(
                    shapes=shapes,
                    data=data,
                    unit_type=unit_type
                )
        else:
            NoProjectLoadWarning()

    @classmethod
    def read(cls, path: Union[str, os.PathLike, Path]):
        """Read an InSituData object from a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where data is saved.

        Returns:
            InSituData: A new InSituData object with the loaded data.
        """
        path = Path(path) # make sure the path is a pathlib path

        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Path does not exist or is not a directory: {str(path)}")

        if not (path / ISPY_METADATA_FILE).exists():
            raise FileNotFoundError(f"No InSituPy metadata file found in the specified directory: {str(path)}")

        # read InSituData metadata
        insitupy_metadata_file = path / ISPY_METADATA_FILE
        metadata = read_json(insitupy_metadata_file)

        # retrieve slide_id and sample_id
        slide_id = metadata["slide_id"]
        sample_id = metadata["sample_id"]

        # save paths of this project in metadata
        metadata["path"] = abspath(path).replace("\\", "/")
        metadata["metadata_file"] = ISPY_METADATA_FILE

        data = cls(path=path,
                   metadata=metadata,
                   slide_id=slide_id,
                   sample_id=sample_id
                   )
        return data


    def saveas(self,
            path: Union[str, os.PathLike, Path],
            overwrite: bool = False,
            zip_output: bool = False,
            images_as_zarr: bool = True,
            zarr_zipped: bool = False,
            images_max_resolution: Optional[Number] = None, # in µm per pixel
            verbose: bool = True
            ):
        '''
        Function to save the InSituData object.

        Args:
            path: Path to save the data to.
        '''
        # check if the path already exists
        path = Path(path)

        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)

        if zip_output:
            zippath = path / (path.stem + ".zip")
            check_overwrite_and_remove_if_true(path=zippath, overwrite=overwrite)

        print(f"Saving data to {str(path)}") if verbose else None

        # create output directory if it does not exist yet
        path.mkdir(parents=True, exist_ok=True)

        # store basic information about experiment
        self._metadata["slide_id"] = self._slide_id
        self._metadata["sample_id"] = self._sample_id

        # clean old entries in data metadata
        self._metadata["data"] = {}

        # save images
        if not self._images.is_empty:
            images = self._images
            _save_images(
                imagedata=images,
                path=path,
                metadata=self._metadata,
                images_as_zarr=images_as_zarr,
                zipped=zarr_zipped,
                max_resolution=images_max_resolution,
                verbose=False
                )

        # save cells
        if not self._cells.is_empty:
            cells = self._cells
            _save_cells(
                cells=cells,
                path=path,
                metadata=self._metadata,
                boundaries_zipped=zarr_zipped,
                max_resolution_boundaries=images_max_resolution
            )

        # save transcripts
        if self._transcripts is not None:
            transcripts = self._transcripts
            _save_transcripts(
                transcripts=transcripts,
                path=path,
                metadata=self._metadata
                )

        # save units
        if self._units is not None:
            units = self._units
            _save_units(
                units=units,
                path=path,
                metadata=self._metadata
                )

        # save annotations
        if not self._annotations.is_empty:
            annotations = self._annotations
            _save_annotations(
                annotations=annotations,
                path=path,
                metadata=self._metadata
            )

        # save regions
        if not self._regions.is_empty:
            regions = self._regions
            _save_regions(
                regions=regions,
                path=path,
                metadata=self._metadata
            )

        # save version of InSituPy
        self._metadata["version"] = __version__

        if "method_params" in self._metadata:
            # move method_param key to end of metadata
            self._metadata["method_params"] = self._metadata.pop("method_params")

        # write Xeniumdata metadata to json file
        xd_metadata_path = path / ISPY_METADATA_FILE
        write_dict_to_json(dictionary=self._metadata, file=xd_metadata_path)

        # Optionally: zip the resulting directory
        if zip_output:
            shutil.make_archive(path, 'zip', path, verbose=False)
            shutil.rmtree(path) # delete directory

        # # change path to the new one
        # self._path = path.resolve()

        # # reload the modalities
        # self.reload(verbose=False)

        print("Saved.") if verbose else None

    def save(self,
             path: Optional[Union[str, os.PathLike, Path]] = None,
             zarr_zipped: bool = False,
             verbose: bool = True,
             keep_history: bool = False
             ):

        # check path
        if path is not None:
            path = Path(path)
        else:
            if self.from_insitudata:
                #path = Path(self._metadata["path"])
                path = self.path
            else:
                warn(
                    f"Data is not linked to an InSituPy project folder (link can be lost by copy for example). "
                    f"Use `saveas()` instead to save the data to a new project folder."
                    )
                return

        if path.exists():
            if verbose:
                print(f"Saving to existing path: {str(path)}", flush=True)

            # check if path is a valid directory
            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {str(path)}")

            # check if the folder is a InSituPy project
            metadata_file = path / ISPY_METADATA_FILE

            if metadata_file.exists():
                # read metadata file and check uid
                project_meta = read_json(metadata_file)

                # check uid
                project_uid = project_meta["uids"][-1]  # [-1] to select latest uid
                current_uid = self._metadata["uids"][-1]
                if current_uid == project_uid:
                    self._update_to_existing_project(path=path,
                                                     zarr_zipped=zarr_zipped,
                                                     verbose=verbose
                                                     )

                    # reload the modalities
                    self.reload(verbose=False, skip=["transcripts", "images"])

                    if not keep_history:
                        self.remove_history(verbose=False)
                else:
                    warn(
                        f"UID of current object {current_uid} not identical with UID in project path {path}: {project_uid}.\n"
                        f"Project is neither saved nor updated. Try `saveas()` instead to save the data to a new project folder. "
                        f"A reason for this could be the data has been cropped in the meantime."
                    )
            else:
                warn(
                    f"No `.ispy` metadata file in {path}. Directory is probably no valid InSituPy project. "
                    f"Use `saveas()` instead to save the data to a new InSituPy project."
                    )


        else:
            if verbose:
                print(f"Saving to new path: {str(path)}", flush=True)

            # save to the respective directory
            self.saveas(path=path)

    def quantify_signal(
        self,
        image_name: str,
        cells_layer: Optional[str] = None,
        cells_compartment: Literal["cells", "nuclei"] = "cells",
        method: Literal["mean", "median"] = "median",
        downsample_factor: Optional[int] = None,
        tile_size: Optional[int] = None,
        add_to_obs: bool = True
    ):
        from insitupy.utils._calc import (create_tiles, quantify_fluorescence,
                                          summarize_tile_measurements)
        img = self.images[image_name]
        pixel_size = self.images.metadata[image_name]["pixel_size"]
        if isinstance(img, list):
            img = img[0]
        cellsdata = _get_cell_layer(self.cells, cells_layer=cells_layer)
        mask = cellsdata.boundaries[cells_compartment]
        if isinstance(mask, list):
            mask = mask[0]

        if tile_size is None:
            measurements, cell_ids = quantify_fluorescence(
                image_dask=img,
                mask_dask=mask,
                method=method,
                downsample_factor=downsample_factor
            )
        else:

            # Tiled approach
            overlap = int(100 / pixel_size)
            print(f"Quantification using tiled approach with overlap {overlap}...", flush=True)
            img_tiles = create_tiles(img, tile_size=tile_size, overlap=overlap)
            mask_tiles = create_tiles(mask, tile_size=tile_size, overlap=overlap)

            quant_results = []
            for i in tqdm(range(len(img_tiles)), desc="Processing tiles"):
                img_tile = img_tiles[i][0]
                mask_tile = mask_tiles[i][0]
                quant_results.append(quantify_fluorescence(
                    image_dask=img_tile,
                    mask_dask=mask_tile,
                    method=method,
                    return_area=True
                ))

            # extract measurements from tiled results
            print("Collecting results...", flush=True)
            measurements, cell_ids = summarize_tile_measurements(quant_results)

        name_mapping = dict(zip(
            cellsdata.boundaries.seg_mask_value.compute(),
            cellsdata.boundaries.cell_names.compute()))

        res_series = pd.Series(
            measurements,
            index=list(map(name_mapping.get, cell_ids))
        )

        if add_to_obs:
            obs_col = f"{image_name}_signal_{cells_compartment}_{method}"
            cellsdata.table.obs[obs_col] = res_series
            print(f"Added quantification results to `.cells['{cells_layer}'].table.obs['{obs_col}']`.", flush=True)
        else:
            return res_series



    def quicksave(self,
                  note: Optional[str] = None
                  ):
        # create quicksave directory if it does not exist already
        self._quicksave_dir = CACHE / "quicksaves"
        self._quicksave_dir.mkdir(parents=True, exist_ok=True)

        # save annotations
        if self._annotations.is_empty:
            print("No annotations found. Quicksave skipped.", flush=True)
        else:
            annotations = self._annotations
            # create filename
            current_datetime = datetime.now().strftime("%y%m%d_%H-%M-%S")
            slide_id = self._slide_id
            sample_id = self._sample_id
            uid = str(uuid4())[:8]

            # create output directory
            outname = f"{slide_id}__{sample_id}__{current_datetime}__{uid}"
            outdir = self._quicksave_dir / outname

            _save_annotations(
                annotations=annotations,
                path=outdir,
                metadata=None
            )

            if note is not None:
                with open(outdir / "note.txt", "w") as notefile:
                    notefile.write(note)

            # # # zip the output
            # shutil.make_archive(outdir, format='zip', root_dir=outdir, verbose=False)
            # shutil.rmtree(outdir) # delete directory


    def list_quicksaves(self):
        pattern = "{slide_id}__{sample_id}__{savetime}__{uid}"

        # collect results
        res = {
            "slide_id": [],
            "sample_id": [],
            "savetime": [],
            "uid": [],
            "note": []
        }
        for d in self._quicksave_dir.glob("[!.]*"):
            parse_res = parse(pattern, d.stem).named
            for key, value in parse_res.items():
                res[key].append(value)

            notepath = d / "note.txt"
            if notepath.exists():
                with open(notepath, "r") as notefile:
                    res["note"].append(notefile.read())
            else:
                res["note"].append("")

        # create and return dataframe
        return pd.DataFrame(res)

    def load_quicksave(self,
                       uid: str
                       ):
        # find files with the uid
        files = list(self._quicksave_dir.glob(f"*{uid}*"))

        if len(files) == 1:
            ad = read_shapesdata(files[0] / "annotations", mode="annotations")
        elif len(files) == 0:
            print(f"No quicksave with uid '{uid}' found. Use `.list_quicksaves()` to list all available quicksaves.")
        else:
            raise ValueError(f"More than one quicksave with uid '{uid}' found.")

        # add annotations to existing annotations attribute or add a new one
        # if self._annotations is None:
        #     self._annotations = AnnotationsData()
        # else:
        for k in ad.metadata.keys():
            self._annotations.add_data(ad[k], k, verbose=True)

    def show(self,
        keys: Optional[str] = None,
        key_type: Literal["genes", "obs", "obsm"] = "genes",
        cells_layer: Optional[str] = None,
        point_size: int = 8,
        scalebar: bool = True,
        unit: str = "µm",
        return_viewer: bool = False,
        widgets_max_width: int = 500,
        verbose: bool = False
        ):
        # check whether napari is installed
        try:
            import napari

            from insitupy._core._napari import _show
        except ImportError:
            raise ImportError("Napari is not installed. Please install napari with `pip install napari[all]` to use this functionality.")

        _show(
            data=self,
            keys=keys,
            key_type=key_type,
            cells_layer=cells_layer,
            point_size=point_size,
            scalebar=scalebar,
            unit=unit,
            return_viewer=return_viewer,
            widgets_max_width=widgets_max_width,
            verbose=verbose
        )

    def reload(
        self,
        skip: Optional[List] = None,
        verbose: bool = True
        ):
        data_meta = self._metadata["data"]
        loaded_modalities = [elem for elem in self.get_loaded_modalities() if elem in data_meta]

        if skip is not None:
            # remove the modalities which are supposed to be skipped during reload
            skip = convert_to_list(skip)
            for s in skip:
                try:
                    loaded_modalities.remove(s)
                except ValueError:
                    pass

        if len(loaded_modalities) > 0:
            print(f"Reloading following modalities: {', '.join(loaded_modalities)}") if verbose else None
            for cm in loaded_modalities:
                func = getattr(self, f"load_{cm}")
                func(verbose=verbose)
        else:
            print("No modalities with existing save path found. Consider saving the data with `saveas()` first.")

    def get_modality(self, modality: str):
        return getattr(self, modality)

    def get_loaded_modalities(self):
        loaded_modalities = []
        for m in MODALITIES:
            try:
                if not getattr(self, m).is_empty:
                    loaded_modalities.append(m)
            except AttributeError:
                # exception for transcripts
                if getattr(self, m) is not None:
                    loaded_modalities.append(m)

        #loaded_modalities = [m for m in MODALITIES if getattr(self, m) is not None]
        return loaded_modalities

    def remove_history(self,
                       verbose: bool = True
                       ):

        for cat in ["annotations", "cells", "regions"]:
            dirs_to_remove = []
            #if hasattr(self, cat):
            files = sorted((self._path / cat).glob("[!.]*"))
            if len(files) > 1:
                dirs_to_remove = files[:-1]

                for d in dirs_to_remove:
                    shutil.rmtree(d)

                print(f"Removed {len(dirs_to_remove)} entries from '.{cat}'.") if verbose else None
            else:
                print(f"No history found for '{cat}'.") if verbose else None

    def remove_modality(self,
                        modality: str
                        ):
        if hasattr(self, modality):
            # delete attribute from InSituData object
            delattr(self, modality)

            # delete metadata
            self.metadata["data"].pop(modality, None) # returns None if key does not exist

        else:
            print(f"No modality '{modality}' found. Nothing removed.")

    def _update_to_existing_project(self,
                                    path: Optional[Union[str, os.PathLike, Path]],
                                    zarr_zipped: bool = False,
                                    verbose: bool = True
                                    ):
        if verbose:
            print(f"Updating project in {path}")

        # save cells
        if not self._cells.is_empty:
            cells = self._cells
            if verbose:
                print("\tUpdating cells...", flush=True)
            _save_cells(
                cells=cells,
                path=path,
                metadata=self._metadata,
                boundaries_zipped=zarr_zipped,
                overwrite=True
            )


        # save annotations
        if not self._annotations.is_empty:
            annotations = self._annotations
            if verbose:
                print("\tUpdating annotations...", flush=True)
            _save_annotations(
                annotations=annotations,
                path=path,
                metadata=self._metadata
            )

        # save regions
        if not self._regions.is_empty:
            regions = self._regions
            if verbose:
                print("\tUpdating regions...", flush=True)
            _save_regions(
                regions=regions,
                path=path,
                metadata=self._metadata
            )

        # save version of InSituPy
        self._metadata["version"] = __version__

        if "method_params" in self._metadata:
            # move method_params key to end of metadata
            self._metadata["method_params"] = self._metadata.pop("method_params")

        # write Xeniumdata metadata to json file
        xd_metadata_path = path / ISPY_METADATA_FILE
        write_dict_to_json(dictionary=self._metadata, file=xd_metadata_path)

        if verbose:
            print("Saved.")



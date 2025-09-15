
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

import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from parse import *
from pyarrow import ArrowInvalid
from scipy.sparse import issparse
from tqdm import tqdm

from insitupy import WITH_NAPARI, __version__
from insitupy._constants import (CACHE, FLUO_CMAP, ISPY_METADATA_FILE,
                                 LOAD_FUNCS, MODALITIES, MODALITIES_COLOR_DICT)
from insitupy._exceptions import (InSituDataMissingObject,
                                  InSituDataRepeatedCropError,
                                  ModalityNotFoundError,
                                  ModalityNotFoundWarning)
from insitupy._io.files import (check_overwrite_and_remove_if_true, read_json,
                                write_dict_to_json)
from insitupy._textformat import textformat as tf
from insitupy._warnings import NoProjectLoadWarning
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.dataclasses.dataclasses import (AnnotationsData, ImageData,
                                              MultiCellData, RegionsData)
from insitupy.dataclasses.io import (_save_annotations, _save_cells,
                                     _save_images, _save_regions,
                                     _save_transcripts, read_multicelldata,
                                     read_shapesdata)
from insitupy.images.axes import ImageAxes
from insitupy.images.utils import _get_contrast_limits, create_img_pyramid
from insitupy.utils._helpers import (_get_expression_values,
                                     sort_paths_by_datetime)
from insitupy.utils.geo import fast_query_points_within_polygon
from insitupy.utils.utils import _crop_transcripts, convert_to_list

# optional packages that are not always installed
if WITH_NAPARI:
    import napari
    from napari.layers import Layer, Points, Shapes
    from napari.utils.notifications import show_info, show_warning

    from insitupy.interactive._configs import _get_viewer_uid, config_manager
    from insitupy.interactive._layers import _create_points_layer
    from insitupy.interactive._widgets import SaveWidget, SyncButton

    #from napari.layers.shapes.shapes import Shapes
    from ..interactive._widgets import (_initialize_widgets,
                                        add_new_geometries_widget)


class InSituData:
    """
    InSituData class for managing and analyzing spatially resolved transcriptomics data.

    .. figure:: ../_static/img/insitudata_overview.svg
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
            ]] + [self._transcripts is None] # transcripts doe not have is_empty property since they are a dataframe
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

            if self._transcripts is not None:
                trans_repr = f"DataFrame with shape {self._transcripts.shape[0]} x {self._transcripts.shape[1]}"

                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['transcripts']+tf.Bold} transcripts{tf.ResetAll}\n{tf.SPACER}   " + trans_repr
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
        InSituData.cells[layer].matrix. Assignment information is added to the DataFrame in `.obs`.
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
        x = celldata.matrix.obsm["spatial"][:, 0]
        y = celldata.matrix.obsm["spatial"][:, 1]
        cells = gpd.points_from_xy(x, y)
        cells = gpd.GeoSeries(cells)

        # iterate through annotation keys
        for key in keys:
            print(f"Assigning key '{key}'...")
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
            data.index = celldata.matrix.obs_names

            # transform data into one column
            column_to_add = [" & ".join(geom_names[row.values]) if np.any(row.values) else "unassigned" for _, row in data.iterrows()]

            if add_to_obs:
                # create annotation from annotation masks
                col_name = f"{geometry_type}-{key}"
                data[col_name] = column_to_add
                if col_name in celldata.matrix.obs:
                    if overwrite:
                        celldata.matrix.obs.drop(col_name, axis=1, inplace=True)
                        print(f'Existing column "{col_name}" is overwritten.', flush=True)
                        add = True
                    else:
                        warn(f'Column "{col_name}" exists already in `{name}.matrix.obs`. Assignment of key "{key}" was skipped. To force assignment, select `overwrite=True`.')
                        add = False
                else:
                    add = True

                if add:
                    if add_masks:
                        celldata.matrix.obs = pd.merge(left=celldata.matrix.obs, right=data, left_index=True, right_index=True)
                    else:
                        celldata.matrix.obs = pd.merge(left=celldata.matrix.obs, right=data.iloc[:, -1], left_index=True, right_index=True)

                    # save that the current key was analyzed
                    geom_attr.metadata[key]["analyzed"] = tf.TICK
            else:
                # add to obsm
                obsm_keys = celldata.matrix.obsm.keys()
                if geometry_type not in obsm_keys:
                    # add empty pandas dataframe with obs_names as index
                    celldata.matrix.obsm[geometry_type] = pd.DataFrame(index=celldata.matrix.obs_names)

                celldata.matrix.obsm[geometry_type][key] = column_to_add

                # save that the current key was analyzed
                geom_attr.metadata[key]["analyzed"] = tf.TICK

                print(f"Added results to `{name}.matrix.obsm['{geometry_type}']", flush=True)


    def assign_annotations(
        self,
        keys: Union[str, Literal["all"]] = "all",
        cells_layers: Optional[Union[List[str], str]] = None,
        add_masks: bool = False,
        overwrite: bool = True
    ):
        if cells_layers is None:
            layers_list = self._cells.get_all_keys()
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
            layers_list = self._cells.get_all_keys()
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

            # extract geometry
            print(region_name)
            shape = region_df[region_df["name"] == region_name]["geometry"].item()
            #use_shape = True

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

    @classmethod
    def read(cls, path: Union[str, os.PathLike, Path]):
        """Read an InSituData object from a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where data is saved.

        Returns:
            InSituData: A new InSituData object with the loaded data.
        """
        path = Path(path) # make sure the path is a pathlib path

        assert (path / ISPY_METADATA_FILE).exists(), "No insitupy metadata file found."
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

        # check if napari is installed
        try:
            import napari
        except ImportError:
            raise ImportError("Napari is not installed. Please install napari with `pip install napari[all]` to use this functionality.")

        # initialize a config class manager with new ID
        uid_viewer = config_manager.add_config(data=self)
        current_viewer_config = config_manager[uid_viewer] # get current viewer config
        if verbose:
            current_viewer_config.verbose = True

        # create viewer
        current_viewer = napari.Viewer(title=f"{self.slide_id}: {self.sample_id} #{uid_viewer}")

        # IMAGES
        if self.images.is_empty:
            warn("No images found.")
        else:
            self._add_images_to_viewer(viewer=current_viewer)

        # CELLS
        if keys is not None:
            self._add_cells_to_viewer(
                viewer=current_viewer,
                keys=keys,
                key_type=key_type,
                cells_layer=cells_layer,
                point_size=point_size
                )

        # WIDGETS
        self._add_widgets_to_viewer(
            viewer=current_viewer,
            widgets_max_width=widgets_max_width
        )

        # BUTTONS
        self._add_buttons_to_viewer(
            viewer=current_viewer
        )

        # EVENTS
        self._add_events_to_viewer(viewer=current_viewer)

        # COLOR LEGEND
        self._add_color_legend_to_viewer(viewer=current_viewer)

        # NAPARI SETTINGS
        if scalebar:
            # add scale bar
            current_viewer.scale_bar.visible = True
            current_viewer.scale_bar.unit = unit

        napari.run()

        if return_viewer:
            return current_viewer


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

    ################################
    ### NAPARI-RELATED FUNCTIONS ###
    ################################
    if WITH_NAPARI:
        def _add_images_to_viewer(
            self,
            viewer: napari.Viewer,
            grayscale_colormap: List[str] = FLUO_CMAP,
            ):
            images_attr = self._images
            n_images = len(images_attr.metadata)
            n_grayscales = 0 # number of grayscale images
            for i, (img_name, img_metadata) in enumerate(images_attr.metadata.items()):
                # get image
                img = images_attr[img_name]

                # only last image is set visible
                is_visible = False if i < n_images - 1 else True
                pixel_size = img_metadata['pixel_size']

                # check if the current image is RGB
                #is_rgb = self._images.metadata[img_name]["rgb"]
                axes_str = self._images.metadata[img_name]["axes"]
                shape = self._images.metadata[img_name]["shape"]
                if self._images.metadata[img_name]["axes"] == "CYX":
                    if not len(shape) == 3:
                        warn((
                            f"Axes information ({axes_str}) and shape ({shape}) do not fit together. Assumed grayscale image with axes 'YX'.\n"
                            f"Error is likely caused by inconsistencies in the metadata file occuring in insitupy versions < 0.9.0."
                            )
                            )
                        axes_str = "YX"

                axes = ImageAxes(axes_str)
                is_rgb = axes.is_rgb

                if not is_rgb and axes.C is not None:
                    if not isinstance(img, list):
                        n_channels = img.shape[axes.C]
                    else:
                        n_channels = img[0].shape[axes.C]

                    try:
                        # get channel names
                        channel_names = [
                            elem["Name"]
                            for elem
                            in self._images.metadata[img_name]['OME']['Image']['Pixels']['Channel']
                            ]
                    except KeyError:
                        channel_names = [f"Channel {i+1}" for i in range(n_channels)]

                    # Multichannel grayscale image
                    for ch in range(n_channels):
                        # get channel name
                        ch_name = channel_names[ch]

                        # select channel
                        if not isinstance(img, list):
                            channel_img = da.take(img, indices=ch, axis=axes.C)
                        else:
                            channel_img = [da.take(elem, indices=ch, axis=axes.C) for elem in img]
                        #channel_img = img[ch]

                        # select color map
                        if ch_name in ["nuclei", "nucleus"] or "DAPI" in ch_name:
                            cmap = "blue"
                        else:
                            cmap = grayscale_colormap[n_grayscales % len(grayscale_colormap)]
                            n_grayscales += 1

                        if not isinstance(channel_img, list):
                            # create image pyramid for lazy loading
                            img_pyramid = create_img_pyramid(img=channel_img, nsubres=6)
                        else:
                            img_pyramid = channel_img

                        # get contrast limits
                        contrast_limits = _get_contrast_limits(img_pyramid)

                        if contrast_limits[1] == 0:
                            warn("The maximum value of the image is 0. Is the image really completely empty?")
                            contrast_limits = (0, 255)

                        # add image to viewer
                        viewer.add_image(
                            img_pyramid,
                            name=f"{img_name}: {ch_name}",
                            colormap=cmap,
                            blending="additive",
                            rgb=False,
                            contrast_limits=contrast_limits,
                            scale=(pixel_size, pixel_size),
                            visible=is_visible
                        )

                else:
                    if is_rgb:
                        cmap = None  # default value of cmap
                        blending = "translucent_no_depth"  # set blending mode
                    else:
                        if img_name in ["nuclei", "nucleus"] or "DAPI" in img_name:
                            cmap = "blue"
                        else:
                            cmap = grayscale_colormap[n_grayscales]
                            n_grayscales += 1
                        blending = "additive"  # set blending mode

                    if not isinstance(img, list):
                        # create image pyramid for lazy loading
                        img_pyramid = create_img_pyramid(img=img, nsubres=6)
                    else:
                        img_pyramid = img

                    # infer contrast limits
                    contrast_limits = _get_contrast_limits(img_pyramid)

                    if contrast_limits[1] == 0:
                        warn("The maximum value of the image is 0. Is the image really completely empty?")
                        contrast_limits = (0, 255)

                    # add img pyramid to napari viewer
                    viewer.add_image(
                            img_pyramid,
                            name=img_name,
                            colormap=cmap,
                            blending=blending,
                            rgb=is_rgb,
                            contrast_limits=contrast_limits,
                            scale=(pixel_size, pixel_size),
                            visible=is_visible
                        )

        def _add_cells_to_viewer(
            self,
            viewer: napari.viewer,
            keys: str,
            key_type: Literal["genes", "obs", "obsm"] = "genes",
            cells_layer: Optional[str] = None,
            point_size: int = 8
            ):
            if self.cells.is_empty:
                raise InSituDataMissingObject("cells")
            else:
                celldata = _get_cell_layer(cells=self.cells, cells_layer=cells_layer)

                if cells_layer is None:
                    cells_layer_name = self.cells.main_key
                else:
                    cells_layer_name = cells_layer

                # convert keys to list
                keys = convert_to_list(keys)

                # get point coordinates
                points = np.flip(celldata.matrix.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])
                #points *= pixel_size # convert to length unit (e.g. µm)

                # get expression matrix
                if issparse(celldata.matrix.X):
                    X = celldata.matrix.X.toarray()
                else:
                    X = celldata.matrix.X

                for i, k in enumerate(keys):
                    # get expression values
                    color_value = _get_expression_values(
                        adata=celldata.matrix,
                        X=X,
                        key_type=key_type, key=k
                    )

                    # extract names of cells
                    cell_names = celldata.matrix.obs_names.values

                    # create points layer
                    layer = _create_points_layer(
                        points=points,
                        color_values=color_value,
                        name=f"{cells_layer_name}-{k}",
                        point_names=cell_names,
                        point_size=point_size,
                        visible=True
                    )

                    # add layer programmatically - does not work for all types of layers
                    # see: https://forum.image.sc/t/add-layerdatatuple-to-napari-viewer-programmatically/69878
                    #self._viewer.add_layer(Layer.create(*layer))
                    viewer.add_layer(Layer.create(*layer))

        def _add_widgets_to_viewer(
            self,
            viewer: napari.Viewer,
            widgets_max_width: int = 500
            ):
            # get viewer configuration from configuration manager
            viewer_config = config_manager[_get_viewer_uid(viewer)]

            if self.cells.is_empty:
                # add annotation widget to napari
                add_geom_widget = add_new_geometries_widget()
                add_geom_widget.max_height = 120
                add_geom_widget.max_width = widgets_max_width
                viewer.window.add_dock_widget(add_geom_widget, name="Add geometries", area="right")
            else:
                #celldata = self._cells
                # initialize the widgets
                (
                    show_points_widget,
                    locate_cells_widget,
                    show_geometries_widget,
                    show_boundaries_widget,
                    select_data,
                    filter_cells_widget,
                ) = _initialize_widgets(
                    viewer=viewer,
                    viewer_config=viewer_config
                    )

                # add widgets to napari window
                if select_data is not None:
                    viewer.window.add_dock_widget(select_data, name="Select data", area="right", tabify=False)
                    select_data.max_height = 80
                    select_data.max_width = widgets_max_width

                if show_points_widget is not None:
                    viewer.window.add_dock_widget(show_points_widget, name="Show data", area="right", tabify=False)
                    show_points_widget.max_height = 170
                    show_points_widget.max_width = widgets_max_width

                if show_boundaries_widget is not None:
                    viewer.window.add_dock_widget(show_boundaries_widget, name="Show boundaries", area="right", tabify=False)
                    #show_boundaries_widget.max_height = 80
                    show_boundaries_widget.max_width = widgets_max_width

                if locate_cells_widget is not None:
                    viewer.window.add_dock_widget(locate_cells_widget, name="Navigate to cell", area="right", tabify=False)
                    #locate_cells_widget.max_height = 130
                    locate_cells_widget.max_width = widgets_max_width

                if filter_cells_widget is not None:
                    viewer.window.add_dock_widget(filter_cells_widget, name="Filter cells", area="right", tabify=True)
                    filter_cells_widget.max_height = 150
                    show_points_widget.max_width = widgets_max_width

                # add annotation widget to napari
                add_geom_widget = add_new_geometries_widget()
                #annot_widget.max_height = 100
                add_geom_widget.max_width = widgets_max_width
                viewer.window.add_dock_widget(add_geom_widget, name="Add geometries", area="right", tabify=False, #add_vertical_stretch=True
                                                    )

                if show_geometries_widget is not None:
                    viewer.window.add_dock_widget(show_geometries_widget, name="Show geometries", area="right", tabify=True)
                    show_geometries_widget.max_width = widgets_max_width

        def _add_events_to_viewer(
            self,
            viewer: napari.Viewer
            ):
            # get viewer configuration from configuration manager
            viewer_config = config_manager[_get_viewer_uid(viewer)]

            # Assign function to an layer addition event
            def _update_uid(event):
                global uids_before_removal
                if event is not None:
                    layer = event.source
                    print(event.action) if viewer_config.verbose else None
                    if event.action == "added" and viewer_config._auto_set_uid:
                        if isinstance(layer, Shapes):
                            type_last = layer.shape_type[-1]
                            if type_last in ["polygon", "rectangle", "ellipse"]:
                                geom_type = "polygon_exterior"
                            elif type_last in ["path", "line"]:
                                geom_type = "line"
                            else:
                                show_warning(f"Unsupported shape type '{type_last}' for UID assignment. Only 'polygon' and 'path' are supported.")
                        elif isinstance(layer, Points):
                            geom_type = "point"
                        #if 'uid' in layer.properties:
                        uid = str(uuid4())
                        print(f"Added '{type_last}' with UID '{uid}'") if viewer_config.verbose else None
                        try:
                            layer.properties['uid'][-1] = uid
                            layer.properties['type'][-1] = geom_type
                        except KeyError:
                            layer.properties['uid'] = np.array([uid], dtype='object')
                            layer.properties['uid'] = np.array([geom_type], dtype='object')

                    elif event.action == "removing":
                        uids_before_removal = set(layer.properties['uid'])
                    elif event.action == "removed":
                        removed_uids = uids_before_removal ^ set(layer.properties['uid'])
                        print(f"Removed following UIDs: {removed_uids}") if viewer_config.verbose else None
                        viewer_config._removal_tracker += list(removed_uids)
                    else:
                        pass

            # Assign the function to data of all existing layers
            for layer in viewer.layers:
                if isinstance(layer, Shapes) or isinstance(layer, Points):
                    layer.events.data.connect(_update_uid)

            # Connect the function to the data of existing shapes and points layers in the viewer
            def connect_to_all_shapes_layers(event):
                layer = event.source[event.index]
                if event is not None:
                    if isinstance(layer, Shapes) or isinstance(layer, Points):
                        layer.events.data.connect(_update_uid)

            # Connect the function to any new layers added to the viewer
            viewer.layers.events.inserted.connect(connect_to_all_shapes_layers)

        def _add_color_legend_to_viewer(
            self,
            viewer: napari.Viewer
            ):
            # # add color legend widget
            config = config_manager[_get_viewer_uid(viewer)]
            viewer.window.add_dock_widget(config.static_canvas, area='left', name='Color legend')

            # add save widget for color legends
            save_widget = SaveWidget()
            viewer.window.add_dock_widget(save_widget, area='left', name="Save color legend")

        def _add_buttons_to_viewer(
            self,
            viewer: napari.Viewer
        ):
            # create sync button
            sync_button = SyncButton()

            # add the sync button to viewer
            viewer.window.add_dock_widget(sync_button, area='right', name="Sync")
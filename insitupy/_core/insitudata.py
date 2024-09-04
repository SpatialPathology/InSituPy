import functools as ft
import gc
import json
import os
import shutil
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4
from warnings import warn

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from dask_image.imread import imread
from geopandas import GeoDataFrame
from parse import *
from scipy.sparse import issparse
from shapely import Point, Polygon
from shapely.affinity import scale as scale_func
from tqdm import tqdm

from insitupy import WITH_NAPARI, __version__
from insitupy._constants import ISPY_METADATA_FILE, REGIONS_SYMBOL
from insitupy._core._save import _save_images
from insitupy._core._xenium import (_read_binned_expression,
                                    _read_boundaries_from_xenium,
                                    _read_matrix_from_xenium,
                                    _restructure_transcripts_dataframe)
from insitupy._exceptions import UnknownOptionError
from insitupy.images import ImageRegistration, deconvolve_he, resize_image
from insitupy.io.files import read_json, write_dict_to_json
from insitupy.io.io import (read_baysor_cells, read_baysor_transcripts,
                            read_celldata, read_shapesdata)
from insitupy.io.plots import save_and_show_figure
from insitupy.utils.preprocessing import (normalize_anndata,
                                          reduce_dimensions_anndata)
from insitupy.utils.utils import get_nrows_maxcols

from .._constants import CACHE, ISPY_METADATA_FILE, MODALITIES
from .._exceptions import (ModalityNotFoundError, NotOneElementError,
                           WrongNapariLayerTypeError, XeniumDataMissingObject,
                           XeniumDataRepeatedCropError)
from ..images.utils import create_img_pyramid
from ..io.files import check_overwrite_and_remove_if_true, read_json
from ..plotting import expr_along_obs_val
from ..utils.utils import (convert_napari_shape_to_polygon_or_line,
                           convert_to_list)
from ..utils.utils import textformat as tf
from ._layers import _create_points_layer
from ._save import (_save_alt, _save_annotations, _save_cells, _save_images,
                    _save_regions, _save_transcripts)
from .dataclasses import AnnotationsData, CellData, ImageData, RegionsData

# optional packages that are not always installed
if WITH_NAPARI:
    import napari
    from napari.layers import Layer, Points, Shapes

    #from napari.layers.shapes.shapes import Shapes
    from ._layers import _add_annotations_as_layer
    from ._widgets import _initialize_widgets, add_new_geometries_widget


class InSituData:
    #TODO: Docstring of InSituData

    # import deprecated functions
    from ._deprecated import (read_all, read_annotations, read_cells,
                              read_images, read_regions, read_transcripts)

    def __init__(self,
                 path: Union[str, os.PathLike, Path],
                 metadata: dict,
                 slide_id: str,
                 sample_id: str,
                 from_insitudata: bool,
                 ):
        """_summary_

        Args:
            path (Union[str, os.PathLike, Path]): _description_
            pattern_xenium_folder (str, optional): _description_. Defaults to "output-{ins_id}__{slide_id}__{sample_id}".
            matrix (Optional[AnnData], optional): _description_. Defaults to None.

        Raises:
            FileNotFoundError: _description_
        """
        self.path = path
        self.metadata = metadata
        self.slide_id = slide_id
        self.sample_id = sample_id
        self.from_insitudata = from_insitudata

    def __repr__(self):
        try:
            method = self.metadata["method"]
        except KeyError:
            method = "unknown"

        repr = (
            f"{tf.Bold+tf.Red}InSituData{tf.ResetAll}\n"
            f"{tf.Bold}Method:{tf.ResetAll}\t\t{method}\n"
            f"{tf.Bold}Slide ID:{tf.ResetAll}\t{self.slide_id}\n"
            f"{tf.Bold}Sample ID:{tf.ResetAll}\t{self.sample_id}\n"
            f"{tf.Bold}Path:{tf.ResetAll}\t\t{self.path.resolve()}\n"
        )

        mfile = self.metadata["metadata_file"]

        repr += f"{tf.Bold}Metadata file:{tf.ResetAll}\t{mfile}"

        if hasattr(self, "images"):
            images_repr = self.images.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + images_repr.replace("\n", f"\n{tf.SPACER}   ")
            )

        if hasattr(self, "cells"):
            cells_repr = self.cells.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} cells{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
            )

        if hasattr(self, "transcripts"):
            trans_repr = f"DataFrame with shape {self.transcripts.shape[0]} x {self.transcripts.shape[1]}"

            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Purple+tf.Bold} transcripts{tf.ResetAll}\n{tf.SPACER}   " + trans_repr
            )

        if hasattr(self, "annotations"):
            annot_repr = self.annotations.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + annot_repr.replace("\n", f"\n{tf.SPACER}   ")
            )

        if hasattr(self, "regions"):
            region_repr = self.regions.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + region_repr.replace("\n", f"\n{tf.SPACER}   ")
            )

        if hasattr(self, "alt"):
            cells_repr = self.alt.__repr__()
            altseg_keys = self.alt.keys()
            repr = (
                #repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} alt{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} alt{tf.ResetAll}\n"
                f"{tf.SPACER}   Alternative CellData objects with following keys: {','.join(altseg_keys)}"
            )
        return repr

    def _save_metadata_after_registration(self,
                      metadata_path: Union[str, os.PathLike, Path] = None
                      ):
        # if there is no specific path given, the metadata is written to the default path for modified metadata
        if metadata_path is None:
            metadata_path = self.path / "experiment_modified.xenium"

        # write to json file
        metadata_json = json.dumps(self.metadata["xenium"], indent=4)
        print(f"\t\tSave metadata to {metadata_path}", flush=True)
        with open(metadata_path, "w") as metafile:
            metafile.write(metadata_json)

    def _remove_empty_modalities(self):
        try:
            # check if anything really added to regions and if not, remove it again
            if len(self.regions.metadata) == 0:
                self.remove_modality("regions")
        except AttributeError:
            pass
        try:
            # check if anything really added to annotations and if not, remove it again
            if len(self.annotations.metadata) == 0:
                self.remove_modality("annotations")
        except AttributeError:
            pass

    def assign_geometries(self,
                          modality: Literal["annotations", "regions"],
                          keys: Union[str, Literal["all"]] = "all",
                          add_masks: bool = False,
                          overwrite: bool = True
                          ):
        '''
        Function to assign the annotations to the anndata object in XeniumData.matrix.
        Annotation information is added to the DataFrame in `.obs`.
        '''
        # assert that prerequisites are met
        try:
            geom_attr = getattr(self, modality)
        except AttributeError:
            raise ModalityNotFoundError(modality=modality)

        try:
            cell_attr = self.cells
        except AttributeError:
            raise ModalityNotFoundError("cells")

        if keys == "all":
            keys = geom_attr.metadata.keys()

        # make sure annotation keys are a list
        keys = convert_to_list(keys)

        # convert coordinates into shapely Point objects
        x = cell_attr.matrix.obsm["spatial"][:, 0]
        y = cell_attr.matrix.obsm["spatial"][:, 1]
        cells = gpd.points_from_xy(x, y)

        # iterate through annotation keys
        for key in keys:
            print(f"Assigning key '{key}'...")
            # extract pandas dataframe of current key
            geom_df = getattr(geom_attr, key)

            # get unique list of annotation names
            geom_names = geom_df.name.unique()

            # initiate dataframe as dictionary
            df = {}

            # iterate through names
            for n in geom_names:
                polygons = geom_df[geom_df["name"] == n]["geometry"].tolist()
                scales = geom_df[geom_df["name"] == n]["scale"].tolist()

                in_poly = []
                for poly, scale in zip(polygons, scales):
                    # scale the polygon
                    poly = scale_func(poly, xfact=scale[0], yfact=scale[1], origin=(0,0))

                    # check if which of the points are inside the current annotation polygon
                    in_poly.append(poly.contains(cells))

                # check if points were in any of the polygons
                in_poly_res = np.array(in_poly).any(axis=0)

                # collect results
                df[n] = in_poly_res

            # convert into pandas dataframe
            df = pd.DataFrame(df)
            df.index = cell_attr.matrix.obs_names

            # create annotation from annotation masks
            col_name = f"{modality}-{key}"
            df[col_name] = [" & ".join(geom_names[row.values]) if np.any(row.values) else np.nan for i, row in df.iterrows()]

            add = True
            if col_name in self.cells.matrix.obs:
                if overwrite:
                    self.cells.matrix.obs.drop(col_name, axis=1, inplace=True)
                    print(f'Existing column "{col_name}" is overwritten.', flush=True)
                    add = True
                else:
                    warn(f'Column "{col_name}" exists already in `xd.cells.matrix.obs`. Assignment of key "{key}" was skipped. To force assignment, select `overwrite=True`.')
                    add = False

            if add:
                if add_masks:
                    self.cells.matrix.obs = pd.merge(left=self.cells.matrix.obs, right=df, left_index=True, right_index=True)
                else:
                    self.cells.matrix.obs = pd.merge(left=self.cells.matrix.obs, right=df.iloc[:, -1], left_index=True, right_index=True)

                # save that the current key was analyzed
                geom_attr.metadata[key]["analyzed"] = tf.TICK

    def assign_annotations(
        self,
        keys: Union[str, Literal["all"]] = "all",
        add_masks: bool = False,
        overwrite: bool = True
    ):
        self.assign_geometries(
            modality="annotations",
            keys=keys,
            add_masks=add_masks,
            overwrite=overwrite
        )

    def assign_regions(
        self,
        keys: Union[str, Literal["all"]] = "all",
        add_masks: bool = False,
        overwrite: bool = True
    ):
        self.assign_geometries(
            modality="regions",
            keys=keys,
            add_masks=add_masks,
            overwrite=overwrite
        )

    def copy(self):
        '''
        Function to generate a deep copy of the XeniumData object.
        '''
        from copy import deepcopy
        had_viewer = False
        if hasattr(self, "viewer"):
            had_viewer = True

            # make copy of viewer to add it later again
            viewer_copy = self.viewer.copy()
            # remove viewer because there is otherwise a error during deepcopy
            del self.viewer

        # make copy
        self_copy = deepcopy(self)

        # add viewer again to original object if necessary
        if had_viewer:
            self.viewer = viewer_copy

        return self_copy

    def crop(self,
            shape_layer: Optional[str] = None,
            xlim: Optional[Tuple[int, int]] = None,
            ylim: Optional[Tuple[int, int]] = None,
            inplace: bool = False
            ):
        '''
        Function to crop the XeniumData object.
        '''
        if shape_layer is not None:
            try:
                # extract shape layer for cropping from napari viewer
                crop_shape = self.viewer.layers[shape_layer]
            except KeyError:
                raise KeyError(f"Shape layer selected for cropping ('{shape_layer}') was not found in layers.")

            # check the type of the element
            if not isinstance(crop_shape, napari.layers.Shapes):
                raise WrongNapariLayerTypeError(found=type(crop_shape), wanted=napari.layers.Shapes)

            use_shape = True
        else:
            # if xlim or ylim is not none, assert that both are not None
            if xlim is not None or ylim is not None:
                assert np.all([elem is not None for elem in [xlim, ylim]])
                use_shape = False

        # assert that either shape_layer is given or xlim/ylim
        assert np.any([elem is not None for elem in [shape_layer, xlim, ylim]]), "No values given for either `shape_layer` or `xlim/ylim`."

        if use_shape:
            # extract shape layer for cropping from napari viewer
            crop_shape = self.viewer.layers[shape_layer]

            # check the structure of the shape object
            if len(crop_shape.data) != 1:
                raise NotOneElementError(crop_shape.data)

            # select the shape from list
            crop_window = crop_shape.data[0].copy()
            crop_window *= self.metadata["xenium"]["pixel_size"] # convert to metric unit (normally µm)

            # extract x and y limits from the shape (assuming a rectangle)
            xlim = (crop_window[:, 1].min(), crop_window[:, 1].max())
            ylim = (crop_window[:, 0].min(), crop_window[:, 0].max())

        # make sure there are no negative values in the limits
        xlim = tuple(np.clip(xlim, a_min=0, a_max=None))
        ylim = tuple(np.clip(ylim, a_min=0, a_max=None))

        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        # if the object was previously cropped, check if the current window is identical with the previous one
        if np.all([elem in _self.metadata["xenium"].keys() for elem in ["cropping_xlim", "cropping_ylim"]]):
            # test whether the limits are identical
            if (xlim == _self.metadata["xenium"]["cropping_xlim"]) & (ylim == _self.metadata["xenium"]["cropping_ylim"]):
                raise XeniumDataRepeatedCropError(xlim, ylim)

        try:
            # infer mask from cell coordinates
            cells = _self.cells
        except AttributeError:
            pass
        else:
            cell_coords = cells.matrix.obsm['spatial'].copy()
            xmask = (cell_coords[:, 0] >= xlim[0]) & (cell_coords[:, 0] <= xlim[1])
            ymask = (cell_coords[:, 1] >= ylim[0]) & (cell_coords[:, 1] <= ylim[1])
            mask = xmask & ymask

            # select
            _self.cells.matrix = _self.cells.matrix[mask, :].copy()

            # crop boundaries
            _self.cells.boundaries.crop(
                cell_ids=_self.cells.matrix.obs_names, xlim=xlim, ylim=ylim
                )

            # shift coordinates to correct for change of coordinates during cropping
            _self.cells.shift(x=-xlim[0], y=-ylim[0])

        try:
            alt = _self.alt
        except AttributeError:
            pass
        else:
            for k, cells in alt.items():
                cell_coords = cells.matrix.obsm['spatial'].copy()
                xmask = (cell_coords[:, 0] >= xlim[0]) & (cell_coords[:, 0] <= xlim[1])
                ymask = (cell_coords[:, 1] >= ylim[0]) & (cell_coords[:, 1] <= ylim[1])
                mask = xmask & ymask

                # select
                cells.matrix = cells.matrix[mask, :].copy()

                # crop boundaries
                cells.boundaries.crop(
                    cell_ids=_self.cells.matrix.obs_names, xlim=xlim, ylim=ylim
                    )

                # shift coordinates to correct for change of coordinates during cropping
                cells.shift(x=-xlim[0], y=-ylim[0])

        if hasattr(_self, "transcripts"):
            # infer mask for selection
            xmask = (_self.transcripts["coordinates", "x"] >= xlim[0]) & (_self.transcripts["coordinates", "x"] <= xlim[1])
            ymask = (_self.transcripts["coordinates", "y"] >= ylim[0]) & (_self.transcripts["coordinates", "y"] <= ylim[1])
            mask = xmask & ymask

            # select
            _self.transcripts = _self.transcripts.loc[mask, :].copy()

            # move origin again to 0 by subtracting the lower limits from the coordinates
            _self.transcripts["coordinates", "x"] -= xlim[0]
            _self.transcripts["coordinates", "y"] -= ylim[0]

        if hasattr(_self, "images"):
            _self.images.crop(xlim=xlim, ylim=ylim)

        if hasattr(_self, "annotations"):
            _self.annotations.crop(xlim=xlim, ylim=ylim)

        if hasattr(_self, "regions"):
            _self.regions.crop(xlim=xlim, ylim=ylim)

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
            if k != "alt":
                _self.metadata["history"][k] = []
            else:
                empty_alt_hist_dict = {k: [] for k in _self.metadata["history"]["alt"].keys()}
                _self.metadata["history"]["alt"] = empty_alt_hist_dict

        # sometimes modalities like annotations or regions can be empty in the meantime
        # here such empty modalities are removed
        _self._remove_empty_modalities()

        if inplace:
            if hasattr(self, "viewer"):
                del _self.viewer # delete viewer
        else:
            return _self




    def hvg(self,
            hvg_batch_key: Optional[str] = None,
            hvg_flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = 'seurat',
            hvg_n_top_genes: Optional[int] = None,
            verbose: bool = True
            ) -> None:
        """
        Calculate highly variable genes (HVGs) using specified flavor and parameters.

        Args:
            hvg_batch_key (str, optional):
                Batch key for computing HVGs separately for each batch. Default is None, indicating all samples are considered.
            hvg_flavor (Literal["seurat", "cell_ranger", "seurat_v3"], optional):
                Flavor of the HVG computation method. Choose between "seurat", "cell_ranger", or "seurat_v3".
                Default is 'seurat'.
            hvg_n_top_genes (int, optional):
                Number of top highly variable genes to identify. Mandatory if `hvg_flavor` is set to "seurat_v3".
                Default is None.
            verbose (bool, optional):
                If True, print progress messages during HVG computation. Default is True.

        Raises:
            ValueError: If `hvg_n_top_genes` is not specified for "seurat_v3" flavor or if an invalid `hvg_flavor` is provided.

        Returns:
            None: This method modifies the input matrix in place, identifying highly variable genes based on the specified
                flavor and parameters. It does not return any value.
        """

        if hvg_flavor in ["seurat", "cell_ranger"]:
            hvg_layer = None
        elif hvg_flavor == "seurat_v3":
            hvg_layer = "counts" # seurat v3 method expects counts data

            # n top genes must be specified for this method
            if hvg_n_top_genes is None:
                raise ValueError(f"HVG computation: For flavor {hvg_flavor} `hvg_n_top_genes` is mandatory")
        else:
            raise ValueError(f'Unknown value for `hvg_flavor`: {hvg_flavor}. Possible values: {["seurat", "cell_ranger", "seurat_v3"]}')

        if hvg_batch_key is None:
            print("Calculate highly-variable genes across all samples using {} flavor...".format(hvg_flavor)) if verbose else None
        else:
            print("Calculate highly-variable genes per batch key {} using {} flavor...".format(hvg_batch_key, hvg_flavor)) if verbose else None

        sc.pp.highly_variable_genes(self.cells.matrix, batch_key=hvg_batch_key, flavor=hvg_flavor, layer=hvg_layer, n_top_genes=hvg_n_top_genes)


    def normalize(self,
                transformation_method: Literal["log1p", "sqrt"] = "log1p",
                normalize_alt: bool = True,
                verbose: bool = True
                ) -> None:
        """
        Normalize the data using either log1p or square root transformation.

        Args:
            transformation_method (Literal["log1p", "sqrt"], optional):
                The method used for data transformation. Choose between "log1p" for logarithmic transformation
                and "sqrt" for square root transformation. Default is "log1p".
            normalize_alt (bool, optional):
                If True, `.alt` modalities are also normalized, if available.
            verbose (bool, optional):
                If True, print progress messages during normalization. Default is True.

        Raises:
            ValueError: If `transformation_method` is not one of ["log1p", "sqrt"].

        Returns:
            None: This method modifies the input matrix in place, normalizing the data based on the specified method.
                It does not return any value.
        """
        try:
            cells = self.cells
        except AttributeError:
            raise ModalityNotFoundError(modality="cells")

        normalize_anndata(adata=cells.matrix, transformation_method=transformation_method, verbose=verbose)

        try:
            alt = self.alt
        except AttributeError:
            pass
        else:
            print("Found `.alt` modality.")
            for k, cells in alt.items():
                print(f"\tNormalizing {k}...")
                normalize_anndata(adata=cells.matrix, transformation_method=transformation_method, verbose=verbose)

    def add_alt(self,
                celldata_to_add: CellData,
                key_to_add: str
                ) -> None:
        # check if the current self has already an alt object and add a empty one if not
        alt_attr_name = "alt"
        try:
            alt_attr = getattr(self, alt_attr_name)
        except AttributeError:
            setattr(self, alt_attr_name, {})
            alt_attr = getattr(self, alt_attr_name)

        # add the celldata to the given key
        alt_attr[key_to_add] = celldata_to_add

    def add_baysor(self,
                   path: Union[str, os.PathLike, Path],
                   read_transcripts: bool = False,
                   key_to_add: str = "baysor",
                   pixel_size: Number = 1 # the pixel size is usually 1 since baysor runs on the µm coordinates
                   ):

        # # convert to pathlib path
        path = Path(path)

        # read baysor data
        celldata = read_baysor_cells(baysor_output=path, pixel_size=pixel_size)

        # add celldata to alt attribute
        self.add_alt(celldata_to_add=celldata, key_to_add=key_to_add)

        if read_transcripts:
            trans_attr_name = "transcripts"
            try:
                trans_attr = getattr(self, trans_attr_name)
            except AttributeError:
                print("No transcript layer found. Addition of Baysor transcript data is skipped.", flush=True)
                pass
            else:
                # read baysor transcripts
                baysor_results = read_baysor_transcripts(baysor_output=path)
                baysor_results = baysor_results[["cell"]]

                # merge transcripts with existing transcripts
                baysor_results.columns = pd.MultiIndex.from_tuples([("cell_id", key_to_add)])
                trans_attr = pd.merge(left=trans_attr,
                                    right=baysor_results,
                                    left_index=True,
                                    right_index=True
                                    )

                # add resulting dataframe to XeniumData
                setattr(self, trans_attr_name, trans_attr)


    def plot_dimred(self, save: Optional[str] = None):
        '''
        Read dimensionality reduction plots.
        '''
        # construct paths
        analysis_path = self.path / "analysis"
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
                 ):
        # extract read functions
        read_funcs = [elem for elem in dir(self) if elem.startswith("load_")]
        read_funcs = [elem for elem in read_funcs if elem not in ["load_all", "load_quicksave"]]

        for f in read_funcs:
            if skip is None or skip not in f:
                func = getattr(self, f)
                try:
                    func()
                except ModalityNotFoundError as err:
                    print(err)

    def load_annotations(self):
        print("Loading annotations...", flush=True)
        try:
            p = self.metadata["data"]["annotations"]
        except KeyError:
            raise ModalityNotFoundError(modality="annotations")
        self.annotations = read_shapesdata(path=self.path / p, mode="annotations")


    def import_annotations(self,
                           files: Optional[Union[str, os.PathLike, Path]],
                           keys: Optional[str]
                           ):
        print("Importing annotations...", flush=True)

        # add annotations object
        files = convert_to_list(files)
        keys = convert_to_list(keys)
        pixel_size = self.metadata["xenium"]['pixel_size']

        if not hasattr(self, "annotations"):
            self.annotations = AnnotationsData()

        for key, file in zip(keys, files):
            # read annotation and store in dictionary
            self.annotations.add_data(data=file,
                                      key=key,
                                      scale_factor=(pixel_size, pixel_size)
                                      )

        # # check if anything really added to annotations and if not, remove it again
        # if len(self.annotations.metadata) == 0:
        #     self.remove_modality("annotations")

        self._remove_empty_modalities()

    def load_regions(self):
        print("Loading regions...", flush=True)
        try:
            p = self.metadata["data"]["regions"]
        except KeyError:
            raise ModalityNotFoundError(modality="regions")
        self.regions = read_shapesdata(path=self.path / p, mode="regions")

    def import_regions(self,
                    files: Optional[Union[str, os.PathLike, Path]],
                    keys: Optional[str]
                    ):
        print("Importing regions...", flush=True)

        # add regions object
        files = convert_to_list(files)
        keys = convert_to_list(keys)
        pixel_size = self.metadata["xenium"]['pixel_size']

        if not hasattr(self, "regions"):
            self.regions = RegionsData()

        for key, file in zip(keys, files):
            # read annotation and store in dictionary
            self.regions.add_data(data=file,
                                key=key,
                                scale_factor=(pixel_size, pixel_size),
                                )

        # self.regions = RegionsData(files=files,
        #                            keys=keys,
        #                            pixel_size=self.metadata["xenium"]['pixel_size'])

        # # check if anything really added to regions and if not, remove it again
        # if len(self.regions.metadata) == 0:
        #     self.remove_modality("regions")

        self._remove_empty_modalities()


    def load_cells(self):
        print("Loading cells...", flush=True)
        pixel_size = self.metadata["xenium"]["pixel_size"]
        if self.from_insitudata:
            try:
                cells_path = self.metadata["data"]["cells"]
            except KeyError:
                raise ModalityNotFoundError(modality="cells")
            else:
                self.cells = read_celldata(path=self.path / cells_path)

            # check if alt data is there and read if yes
            try:
                alt_path_dict = self.metadata["data"]["alt"]
            except KeyError:
                print("\tNo alternative cells found...")
            else:
                print("\tFound alternative cells...")
                alt_dict = {}
                for k, p in alt_path_dict.items():
                    alt_dict[k] = read_celldata(path=self.path / p)

                # add attribute
                setattr(self, "alt", alt_dict)

        else:
            # read celldata
            matrix = matrix = _read_matrix_from_xenium(path=self.path)
            boundaries = _read_boundaries_from_xenium(path=self.path, pixel_size=pixel_size)
            self.cells = CellData(matrix=matrix, boundaries=boundaries)

            try:
                # read binned expression
                arr = _read_binned_expression(path=self.path, gene_names_to_select=self.cells.matrix.var_names)
                self.cells.matrix.varm["binned_expression"] = arr
            except ValueError:
                warn("Loading of binned expression did not work. Skipped it.")
                pass


    def load_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    nuclei_type: Literal["focus", "mip", ""] = "mip",
                    load_cell_segmentation_images: bool = True
                    ):
        # load image into ImageData object
        print("Loading images...", flush=True)

        if self.from_insitudata:
            # check if matrix data is stored in this XeniumData
            if "images" not in self.metadata["data"]:
                raise ModalityNotFoundError(modality="images")

            if names == "all":
                img_names = list(self.metadata["data"]["images"].keys())
            else:
                img_names = convert_to_list(names)

            # get file paths and names
            img_files = [v for k,v in self.metadata["data"]["images"].items() if k in img_names]
            img_names = [k for k,v in self.metadata["data"]["images"].items() if k in img_names]
        else:
            nuclei_file_key = f"morphology_{nuclei_type}_filepath"

            # In v2.0 the "mip" image was removed due to better focusing of the machine.
            # For <v2.0 the function still tries to retrieve the "mip" image but in case this is not found
            # it will retrieve the "focus" image
            if nuclei_type == "mip" and nuclei_file_key not in self.metadata["xenium"]["images"].keys():
                warn(
                    f"Nuclei image type '{nuclei_type}' not found. Used 'focus' instead. This is the normal behavior for data analyzed with Xenium Ranger >=v2.0",
                    UserWarning, stacklevel=2
                     )

                nuclei_type = "focus"
                nuclei_file_key = f"morphology_{nuclei_type}_filepath"

            if names == "nuclei":
                img_keys = [nuclei_file_key]
                img_names = ["nuclei"]
            else:
                # get available keys for registered images in metadata
                img_keys = [elem for elem in self.metadata["xenium"]["images"] if elem.startswith("registered")]

                # extract image names from keys and add nuclei
                img_names = ["nuclei"] + [elem.split("_")[1] for elem in img_keys]

                # add dapi image key
                img_keys = [nuclei_file_key] + img_keys

                if names != "all":
                    # make sure keys is a list
                    names = convert_to_list(names)
                    # select the specified keys
                    mask = [elem in names for elem in img_names]
                    img_keys = [elem for m, elem in zip(mask, img_keys) if m]
                    img_names = [elem for m, elem in zip(mask, img_names) if m]

            # get path of image files
            img_files = [self.metadata["xenium"]["images"][k] for k in img_keys]

            if load_cell_segmentation_images:
                # get cell segmentation images if available
                if "morphology_focus/" in self.metadata["xenium"]["images"][nuclei_file_key]:
                    seg_files = ["morphology_focus/morphology_focus_0001.ome.tif",
                                 "morphology_focus/morphology_focus_0002.ome.tif",
                                 "morphology_focus/morphology_focus_0003.ome.tif"
                                 ]
                    seg_names = ["cellseg1", "cellseg2", "cellseg3"]

                    # check which segmentation files exist and append to image list
                    seg_file_exists_list = [(self.path / f).is_file() for f in seg_files]
                    #print(seg_file_exists_list)
                    img_files += [f for f, exists in zip(seg_files, seg_file_exists_list) if exists]
                    img_names += [n for n, exists in zip(seg_names, seg_file_exists_list) if exists]

        # create imageData object
        img_paths = [self.path / elem for elem in img_files]
        self.images = ImageData(img_paths, img_names, pixel_size=self.metadata["xenium"]['pixel_size'])

    def load_transcripts(self,
                        transcript_filename: str = "transcripts.parquet"
                        ):
        if self.from_insitudata:
            # check if matrix data is stored in this XeniumData
            if "transcripts" not in self.metadata["data"]:
                raise ModalityNotFoundError(modality="transcripts")

            # read transcripts
            print("Loading transcripts...", flush=True)
            self.transcripts = pd.read_parquet(self.path / self.metadata["data"]["transcripts"])
        else:
            # read transcripts
            print("Loading transcripts...", flush=True)
            transcript_dataframe = pd.read_parquet(self.path / transcript_filename)

            self.transcripts = _restructure_transcripts_dataframe(transcript_dataframe)


    def reduce_dimensions(self,
                        umap: bool = True,
                        tsne: bool = True,
                        batch_correction_key: Optional[str] = None,
                        perform_clustering: bool = True,
                        verbose: bool = True,
                        tsne_lr: int = 1000,
                        tsne_jobs: int = 8,
                        **kwargs
                        ):
        """
        Reduce the dimensionality of the data using PCA, UMAP, and t-SNE techniques, optionally performing batch correction.

        Args:
            umap (bool, optional):
                If True, perform UMAP dimensionality reduction. Default is True.
            tsne (bool, optional):
                If True, perform t-SNE dimensionality reduction. Default is True.
            batch_correction_key (str, optional):
                Batch key for performing batch correction using scanorama. Default is None, indicating no batch correction.
            verbose (bool, optional):
                If True, print progress messages during dimensionality reduction. Default is True.
            tsne_lr (int, optional):
                Learning rate for t-SNE. Default is 1000.
            tsne_jobs (int, optional):
                Number of CPU cores to use for t-SNE computation. Default is 8.
            **kwargs:
                Additional keyword arguments to be passed to scanorama function if batch correction is performed.

        Raises:
            ValueError: If an invalid `batch_correction_key` is provided.

        Returns:
            None: This method modifies the input matrix in place, reducing its dimensionality using specified techniques and
                batch correction if applicable. It does not return any value.
        """
        try:
            cells = self.cells
        except AttributeError:
            raise ModalityNotFoundError(modality="cells")

        reduce_dimensions_anndata(adata=cells.matrix,
                                  umap=umap, tsne=tsne,
                                  batch_correction_key=batch_correction_key,
                                  perform_clustering=perform_clustering,
                                  verbose=verbose,
                                  tsne_lr=tsne_lr, tsne_jobs=tsne_jobs
                                  )

        try:
            alt = self.alt
        except AttributeError:
            pass
        else:
            print("Found `.alt` modality.")
            for k, cells in alt.items():
                print(f"\tReducing dimensions in `.alt['{k}']...")
                reduce_dimensions_anndata(adata=cells.matrix,
                                        umap=umap, tsne=tsne,
                                        batch_correction_key=batch_correction_key,
                                        perform_clustering=perform_clustering,
                                        verbose=verbose,
                                        tsne_lr=tsne_lr, tsne_jobs=tsne_jobs
                                        )


    def saveas(self,
            path: Union[str, os.PathLike, Path],
            overwrite: bool = False,
            zip_output: bool = False,
            images_as_zarr: bool = True,
            zarr_zipped: bool = False
            ):
        '''
        Function to save the XeniumData object.

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

        print(f"Saving data to {str(path)}")

        # create output directory if it does not exist yet
        path.mkdir(parents=True, exist_ok=True)

        # store basic information about experiment
        self.metadata["slide_id"] = self.slide_id
        self.metadata["sample_id"] = self.sample_id

        # clean old entries in data metadata
        self.metadata["data"] = {}

        # save images
        try:
            images = self.images
        except AttributeError:
            pass
        else:
            _save_images(
                imagedata=images,
                path=path,
                metadata=self.metadata,
                images_as_zarr=images_as_zarr,
                zipped=zarr_zipped
                )

        # save cells
        try:
            cells = self.cells
        except AttributeError:
            pass
        else:
            _save_cells(
                cells=cells,
                path=path,
                metadata=self.metadata,
                boundaries_zipped=zarr_zipped
            )

        # save alternative cell data
        try:
            alt = self.alt
        except AttributeError:
            pass
        else:
            _save_alt(
                attr=alt,
                path=path,
                metadata=self.metadata,
                boundaries_zipped=zarr_zipped
            )

        # save transcripts
        try:
            transcripts = self.transcripts
        except AttributeError:
            pass
        else:
            _save_transcripts(
                transcripts=transcripts,
                path=path,
                metadata=self.metadata
                )

        # save annotations
        try:
            annotations = self.annotations
        except AttributeError:
            pass
        else:
            _save_annotations(
                annotations=annotations,
                path=path,
                metadata=self.metadata
            )

        # save regions
        try:
            regions = self.regions
        except AttributeError:
            pass
        else:
            _save_regions(
                regions=regions,
                path=path,
                metadata=self.metadata
            )

        # save version of InSituPy
        self.metadata["version"] = __version__

        # move xenium key to end of metadata
        self.metadata["xenium"] = self.metadata.pop("xenium")

        # write Xeniumdata metadata to json file
        xd_metadata_path = path / ISPY_METADATA_FILE
        write_dict_to_json(dictionary=self.metadata, file=xd_metadata_path)

        # Optionally: zip the resulting directory
        if zip_output:
            shutil.make_archive(path, 'zip', path, verbose=False)
            shutil.rmtree(path) # delete directory

        print("Saved.")

    def save(self,
             path: Optional[Union[str, os.PathLike, Path]] = None,
             zarr_zipped: bool = False
             ):

        # check path
        if path is not None:
            path = Path(path)
        else:
            if self.from_insitudata:
                path = Path(self.metadata["path"])
            else:
                warn(
                    f"Data as not loaded from an InSituPy project. "
                    f"Use `saveas()` instead to save the data to a new project folder."
                    )

        if path.exists():
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
                current_uid = self.metadata["uids"][-1]
                if current_uid == project_uid:
                    self._update_to_existing_project(path=path, zarr_zipped=zarr_zipped)

                    # reload the modalities
                    self.reload()
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
            # save to the respective directory
            self.saveas(path=path)



    def _update_to_existing_project(self,
                                    path: Optional[Union[str, os.PathLike, Path]],
                                    zarr_zipped: bool = False
                                    ):
        print(f"Updating project in {path}")

        # save cells
        try:
            cells = self.cells
        except AttributeError:
            pass
        else:
            print("\tUpdating cells...", flush=True)
            _save_cells(
                cells=cells,
                path=path,
                metadata=self.metadata,
                boundaries_zipped=zarr_zipped,
                overwrite=True
            )

        # save alternative cell data
        try:
            alt = self.alt
        except AttributeError:
            pass
        else:
            print("\tUpdating alternative segmentations...", flush=True)
            _save_alt(
                attr=alt,
                path=path,
                metadata=self.metadata,
                boundaries_zipped=zarr_zipped
            )

        # save annotations
        try:
            annotations = self.annotations
        except AttributeError:
            pass
        else:
            print("\tUpdating annotations...", flush=True)
            _save_annotations(
                annotations=annotations,
                path=path,
                metadata=self.metadata
            )

        # save regions
        try:
            regions = self.regions
        except AttributeError:
            pass
        else:
            print("\tUpdating regions...", flush=True)
            _save_regions(
                regions=regions,
                path=path,
                metadata=self.metadata
            )

        # save version of InSituPy
        self.metadata["version"] = __version__

        # move xenium key to end of metadata
        self.metadata["xenium"] = self.metadata.pop("xenium")

        # write Xeniumdata metadata to json file
        xd_metadata_path = path / ISPY_METADATA_FILE
        write_dict_to_json(dictionary=self.metadata, file=xd_metadata_path)

        print("Saved.")


    def quicksave(self,
                  note: Optional[str] = None
                  ):
        # create quicksave directory if it does not exist already
        self.quicksave_dir = CACHE / "quicksaves"
        self.quicksave_dir.mkdir(parents=True, exist_ok=True)

        # save annotations
        try:
            annotations = self.annotations
        except AttributeError:
            print("No annotations found. Quicksave skipped.", flush=True)
            pass
        else:
            # create filename
            current_datetime = datetime.now().strftime("%y%m%d_%H-%M-%S")
            slide_id = self.slide_id
            sample_id = self.sample_id
            uid = str(uuid4())[:8]

            # create output directory
            outname = f"{slide_id}__{sample_id}__{current_datetime}__{uid}"
            outdir = self.quicksave_dir / outname

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
        for d in self.quicksave_dir.glob("*"):
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
        files = list(self.quicksave_dir.glob(f"*{uid}*"))

        if len(files) == 1:
            ad = read_shapesdata(files[0] / "annotations", mode="annotations")
        elif len(files) == 0:
            print(f"No quicksave with uid '{uid}' found. Use `.list_quicksaves()` to list all available quicksaves.")
        else:
            raise ValueError(f"More than one quicksave with uid '{uid}' found.")

        # add annotations to existing annotations attribute or add a new one
        try:
            annotations = self.annotations
        except AttributeError:
            annotations = self.annotations = AnnotationsData()
        else:
            for k in ad.metadata.keys():
                annotations.add_data(getattr(ad, k), k, verbose=True)


    def show(self,
        keys: Optional[str] = None,
        # annotation_keys: Optional[str] = None,
        point_size: int = 6,
        scalebar: bool = True,
        pixel_size: float = None, # if none, extract from metadata
        unit: str = "µm",
        # cmap_annotations: str ="Dark2",
        grayscale_colormap: List[str] = ["red", "green", "cyan", "magenta", "yellow", "gray"],
        return_viewer: bool = False,
        widgets_max_width: int = 200
        ):
        # get information about pixel size
        if (pixel_size is None) & (scalebar):
            # extract pixel_size
            pixel_size = float(self.metadata["xenium"]["pixel_size"])
        else:
            pixel_size = 1

        # create viewer
        self.viewer = napari.Viewer()

        try:
            image_keys = self.images.metadata.keys()
        except AttributeError:
            warn("No attribute `.images` found.")
        else:
            n_grayscales = 0 # number of grayscale images
            for i, img_name in enumerate(image_keys):
                img = getattr(self.images, img_name)
                is_visible = False if i < len(image_keys) - 1 else True # only last image is set visible

                # check if the current image is RGB
                is_rgb = self.images.metadata[img_name]["rgb"]

                if is_rgb:
                    cmap = None  # default value of cmap
                    blending = "translucent_no_depth"  # set blending mode
                else:
                    if img_name == "nuclei":
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

                # add img pyramid to napari viewer
                self.viewer.add_image(
                        img_pyramid,
                        name=img_name,
                        colormap=cmap,
                        blending=blending,
                        rgb=is_rgb,
                        contrast_limits=self.images.metadata[img_name]["contrast_limits"],
                        scale=(pixel_size, pixel_size),
                        visible=is_visible
                    )

        # optionally: add cells as points
        #if show_cells or keys is not None:
        if keys is not None:
            try:
                cells = self.cells
            except AttributeError:
                raise XeniumDataMissingObject("cells")
            else:
                # convert keys to list
                keys = convert_to_list(keys)

                # get point coordinates
                points = np.flip(cells.matrix.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])
                #points *= pixel_size # convert to length unit (e.g. µm)

                # get expression matrix
                if issparse(cells.matrix.X):
                    X = cells.matrix.X.toarray()
                else:
                    X = cells.matrix.X

                for i, k in enumerate(keys):
                    pvis = False if i < len(keys) - 1 else True # only last image is set visible
                    # get expression values
                    if k in cells.matrix.obs.columns:
                        color_value = cells.matrix.obs[k].values

                    else:
                        geneid = cells.matrix.var_names.get_loc(k)
                        color_value = X[:, geneid]

                    # extract names of cells
                    cell_names = cells.matrix.obs_names.values

                    # create points layer
                    layer = _create_points_layer(
                        points=points,
                        color_values=color_value,
                        name=k,
                        point_names=cell_names,
                        point_size=point_size,
                        visible=pvis
                    )

                    # add layer programmatically - does not work for all types of layers
                    # see: https://forum.image.sc/t/add-layerdatatuple-to-napari-viewer-programmatically/69878
                    self.viewer.add_layer(Layer.create(*layer))

        # WIDGETS
        try:
            cells = self.cells
        except AttributeError:
            # add annotation widget to napari
            annot_widget = add_new_geometries_widget()
            annot_widget.max_height = 100
            annot_widget.max_width = widgets_max_width
            self.viewer.window.add_dock_widget(annot_widget, name="Add geometries", area="right")
        else:
            # initialize the widgets
            show_points_widget, locate_cells_widget, show_geometries_widget, show_boundaries_widget, select_data = _initialize_widgets(xdata=self)

            # add widgets to napari window
            if select_data is not None:
                self.viewer.window.add_dock_widget(select_data, name="Select data", area="right")
                select_data.max_height = 50
                select_data.max_width = widgets_max_width

            if show_points_widget is not None:
                self.viewer.window.add_dock_widget(show_points_widget, name="Show data", area="right")
                show_points_widget.max_height = 130
                show_points_widget.max_width = widgets_max_width

            if show_boundaries_widget is not None:
                self.viewer.window.add_dock_widget(show_boundaries_widget, name="Show boundaries", area="right")
                show_boundaries_widget.max_height = 80
                show_boundaries_widget.max_width = widgets_max_width

            if locate_cells_widget is not None:
                self.viewer.window.add_dock_widget(locate_cells_widget, name="Navigate to cell", area="right")
                #locate_cells_widget.max_height = 130
                locate_cells_widget.max_width = widgets_max_width

            # add annotation widget to napari
            annot_widget = add_new_geometries_widget()
            #annot_widget.max_height = 100
            annot_widget.max_width = widgets_max_width
            self.viewer.window.add_dock_widget(annot_widget, name="Add geometries", area="right")

            # if show_region_widget is not None:
            #     self.viewer.window.add_dock_widget(show_region_widget, name="Show regions", area="right")
            #     show_region_widget.max_height = 100
            #     show_region_widget.max_width = widgets_max_width

            if show_geometries_widget is not None:
                self.viewer.window.add_dock_widget(show_geometries_widget, name="Show geometries", area="right", tabify=True)
                #show_annotations_widget.max_height = 100
                show_geometries_widget.max_width = widgets_max_width


        # EVENTS
        # Function assign to an layer addition event
        def _update_uid(event):
            if event is not None:

                layer = event.source
                # print(event.action) # print what event.action returns
                # print(event.data_indices) # print index of event
                if event.action == "add":
                    # print(f'Added to {layer}')
                    if 'uid' in layer.properties:
                        layer.properties['uid'][-1] = str(uuid4())
                    else:
                        layer.properties['uid'] = np.array([str(uuid4())], dtype='object')

                elif event.action == "remove":
                    pass
                    # print(f"Removed from {layer}")
                else:
                    raise ValueError("Unexpected value '{event.action}' for `event.action`. Expected 'add' or 'remove'.")

                # print(layer.properties)

        for layer in self.viewer.layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                layer.events.data.connect(_update_uid)
                #layer.metadata = layer.properties

        # Connect the function to all shapes layers in the viewer
        def connect_to_all_shapes_layers(event):
            layer = event.source[event.index]
            if event is not None:
                if isinstance(layer, Shapes) or isinstance(layer, Points):
                    # print('Annotation layer added')
                    layer.events.data.connect(_update_uid)

        # Connect the function to any new layers added to the viewer
        self.viewer.layers.events.inserted.connect(connect_to_all_shapes_layers)


        # NAPARI SETTINGS
        if scalebar:
            # add scale bar
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = unit

        napari.run()
        if return_viewer:
            return self.viewer

    def store_geometries(self,
                         name_pattern = "{type_symbol} {class_name} ({annot_key})",
                         uid_col: str = "id"
                         ):
        """
        Extracts geometric layers from shapes and points layers in the napari viewer
        and stores them in the XeniumData object as annotations or regions.

        Args:
            name_pattern (str): A format string used to parse the layer names.
                It should contain placeholders for 'type_symbol', 'class_name',
                and 'annot_key'.
            uid_col (str): The name of the column used to store unique identifiers
                for the geometries. Default is "id".

        Raises:
            AttributeError: If the viewer is not initialized, an error message
                prompts the user to open a napari viewer using the `.show()` method.

        Notes:
            - The function iterates through the layers in the viewer and checks if
            they are instances of Shapes or Points.
            - It extracts the geometric data, colors, and other relevant properties
            to create a GeoDataFrame.
            - The GeoDataFrame is then added to the annotations or regions of the
            XeniumData object based on the type of layer.
            - If the layer is classified as a region but is a point layer, a warning
            is issued, and the layer is skipped.
        """
        try:
            viewer = self.viewer
        except AttributeError as e:
            print(f"{str(e)}. Use `.show()` first to open a napari viewer.")

        # iterate through layers and save them as annotation or region if they meet requirements
        layers = viewer.layers
        #collection_dict = {}
        for layer in layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                name_parsed = parse(name_pattern, layer.name)
                if name_parsed is not None:
                    type_symbol = name_parsed.named["type_symbol"]
                    annot_key = name_parsed.named["annot_key"]
                    class_name = name_parsed.named["class_name"]

                    # if the XeniumData object does not has an annotations attribute, initialize it
                    if not hasattr(self, "annotations"):
                        self.annotations = AnnotationsData() # initialize empty object

                    # extract shapes coordinates and colors
                    layer_data = layer.data
                    colors = layer.edge_color.tolist()
                    scale = layer.scale

                    checks_passed = True
                    is_region_layer = False
                    object_type = "annotation"
                    if type_symbol == REGIONS_SYMBOL:
                        is_region_layer = True
                        object_type = "region"
                        if isinstance(layer, Points):
                            warn(f'Layer "{layer.name}" is a point layer and at the same time classified as "Region". This is not allowed. Skipped this layer.')
                            checks_passed = False

                    if checks_passed:
                        if isinstance(layer, Shapes):
                            # extract shape types
                            shape_types = layer.shape_type
                            # build annotation GeoDataFrame
                            geom_df = {
                                uid_col: layer.properties["uid"],
                                "objectType": object_type,
                                #"geometry": [Polygon(np.stack([ar[:, 1], ar[:, 0]], axis=1)) for ar in layer_data],  # switch x/y
                                "geometry": [convert_napari_shape_to_polygon_or_line(napari_shape_data=ar, shape_type=st) for ar, st in zip(layer_data, shape_types)],
                                "name": class_name,
                                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
                                #"scale": [scale] * len(layer_data),
                                #"layer_type": ["Shapes"] * len(layer_data)
                            }

                        elif isinstance(layer, Points):
                            # build annotation GeoDataFrame
                            geom_df = {
                                uid_col: layer.properties["uid"],
                                "objectType": object_type,
                                "geometry": [Point(d[1], d[0]) for d in layer_data],  # switch x/y
                                "name": class_name,
                                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
                                #"scale": [scale] * len(layer_data),
                                #"layer_type": ["Points"] * len(layer_data)
                            }

                        # generate GeoDataFrame
                        geom_df = GeoDataFrame(geom_df, geometry="geometry")

                        if is_region_layer:
                            if not hasattr(self, "regions"):
                                self.regions = RegionsData()

                            # add regions
                            self.regions.add_data(data=geom_df,
                                                  key=annot_key,
                                                  verbose=True,
                                                  scale_factor=scale)
                        else:
                            if not hasattr(self, "annotations"):
                                self.annotations = AnnotationsData()

                            # add annotations
                            self.annotations.add_data(data=geom_df,
                                                      key=annot_key,
                                                      verbose=True,
                                                      scale_factor=scale)

            else:
                pass

        self._remove_empty_modalities()

    def plot_binned_expression(
        self,
        genes: Union[List[str], str],
        maxcols: int = 4,
        figsize: Tuple[int, int] = (8,6),
        savepath: Union[str, os.PathLike, Path] = None,
        save_only: bool = False,
        dpi_save: int = 300,
        show: bool = True,
        fontsize: int = 28
        ):
        # extract binned expression matrix and gene names
        binex = self.cells.matrix.varm["binned_expression"]
        gene_names = self.cells.matrix.var_names

        genes = convert_to_list(genes)

        nplots, nrows, ncols = get_nrows_maxcols(genes, max_cols=maxcols)

        # setup figure
        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))

        # scale font sizes
        plt.rcParams.update({'font.size': fontsize})

        if nplots > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

        for i, gene in enumerate(genes):
            # retrieve binned expression
            img = binex[gene_names.get_loc(gene)]

            # determine upper limit for color
            vmax = np.percentile(img[img>0], 95)

            # plot expression
            axs[i].imshow(img, cmap="viridis", vmax=vmax)

            # set title
            axs[i].set_title(gene)

        if nplots > 1:

            # check if there are empty plots remaining
            while i < nrows * maxcols - 1:
                i+=1
                # remove empty plots
                axs[i].set_axis_off()

        if show:
            fig.tight_layout()
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save)
        else:
            return fig, axs

    def plot_expr_along_obs_val(
        self,
        keys: str,
        obs_val: str,
        groupby: Optional[str] = None,
        method: Literal["lowess", "loess"] = 'loess',
        stderr: bool = False,
        savepath=None,
        return_data=False,
        **kwargs
        ):
        # retrieve anndata object from InSituData
        adata = self.cells.matrix

        results = expr_along_obs_val(
            adata=adata,
            keys=keys,
            obs_val=obs_val,
            groupby=groupby,
            method=method,
            stderr=stderr,
            savepath=savepath,
            return_data=return_data
            **kwargs
            )

        if return_data:
            return results

    def reload(self):
        data_meta = self.metadata["data"]
        current_modalities = [m for m in MODALITIES if hasattr(self, m) and m in data_meta]
        # # check if there is a path for the modalities in self.metadata
        # data_meta = self.metadata["data"]
        # print(data_meta)
        # for cm in current_modalities:
        #     print(cm)
        #     if cm in data_meta:
        #         print('blubb')
        #         pass
        #     else:
        #         print(f"No data path found for modality '{cm}'. Modality skipped during reload and needs to be saved first.")
        #         #current_modalities.remove(cm)

        if len(current_modalities) > 0:
            print(f"Reloading following modalities: {','.join(current_modalities)}")
            for cm in current_modalities:
                func = getattr(self, f"load_{cm}")
                func()
        else:
            print("No modalities with existing save path found. Consider saving the data with `saveas()` first.")

    def remove_history(self,
                       verbose: bool = True
                       ):

        for cat in ["annotations", "cells", "regions"]:
            dirs_to_remove = []
            #if hasattr(self, cat):
            files = sorted((self.path / cat).glob("*"))
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



def register_images(
    data: InSituData,
    image_to_be_registered: Union[str, os.PathLike, Path],
    image_type: Literal["histo", "IF"],
    channel_names: Union[str, List[str]],
    channel_name_for_registration: Optional[str] = None,  # name used for the nuclei image. Only required for IF images.
    template_image_name: str = "nuclei",
    save_results: bool = True,
    add_registered_image: bool = True,
    decon_scale_factor: float = 0.2,
    physicalsize: str = 'µm',
    prefix: str = ""
    ):
    '''
    Register images stored in XeniumData object.
    '''

    # if image type is IF, the channel name for registration needs to be given
    if image_type == "IF" and channel_name_for_registration is None:
        raise ValueError(f'If `image_type" is "IF", `channel_name_for_registration is not allowed to be `None`.')

    # define output directory
    output_dir = data.path.parent / "registered_images"

    # if output_dir.is_dir() and not force:
    #     raise FileExistsError(f"Output directory {output_dir} exists already. If you still want to run the registration, set `force=True`.")

    # check if image path exists
    image_to_be_registered = Path(image_to_be_registered)
    if not image_to_be_registered.is_file():
        raise FileNotFoundError(f"No such file found: {str(image_to_be_registered)}")

    # make sure the given image names are in a list
    channel_names = convert_to_list(channel_names)

    # determine the structure of the image axes and check other things
    axes_template = "YX"
    if image_type == "histo":
        axes_image = "YXS"

        # make sure that there is only one image name given
        if len(channel_names) > 1:
            raise ValueError(f"More than one image name retrieved ({channel_names})")

        if len(channel_names) == 0:
            raise ValueError(f"No image name found in file {image_to_be_registered}")

    elif image_type == "IF":
        axes_image = "CYX"
    else:
        raise UnknownOptionError(image_type, available=["histo", "IF"])

    print(f'\tProcessing following {image_type} images: {tf.Bold}{", ".join(channel_names)}{tf.ResetAll}', flush=True)

    # read images
    print("\t\tLoading images to be registered...", flush=True)
    image = imread(image_to_be_registered) # e.g. HE image

    # sometimes images are read with an empty time dimension in the first axis.
    # If this is the case, it is removed here.
    if len(image.shape) == 4:
        image = image[0]

    # read images in XeniumData object
    data.load_images(names=template_image_name)
    template = data.images.nuclei[0] # usually the nuclei/DAPI image is the template. Use highest resolution of pyramid.

    # extract OME metadata
    ome_metadata_template = data.images.metadata["nuclei"]["OME"]

    # extract pixel size for x and y from metadata
    pixelsizes = {key: ome_metadata_template['Image']['Pixels'][key] for key in ['PhysicalSizeX', 'PhysicalSizeY']}

    # generate OME metadata for saving
    ome_metadata = {
        **{'SignificantBits': 8,
        'PhysicalSizeXUnit': physicalsize,
        'PhysicalSizeYUnit': physicalsize
        },
        **pixelsizes
    }

    # determine one pixel direction as universal pixel size
    pixel_size = pixelsizes['PhysicalSizeX']

    # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
    axes_selected = "YX"
    if image_type == "histo":
        print("\t\tRun color deconvolution", flush=True)
        # deconvolve HE - performed on resized image to save memory
        # TODO: Scale to max width instead of using a fixed scale factor before deconvolution (`scale_to_max_width`)
        nuclei_img, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes=axes_selected),
                                    return_type="grayscale", convert=True)

        # bring back to original size
        nuclei_img = resize_image(nuclei_img, scale_factor=1/decon_scale_factor, axes=axes_selected)

        # set nuclei_channel and nuclei_axis to None
        channel_name_for_registration = channel_axis = None
    else:
        # image_type is "IF" then
        # get index of nuclei channel
        channel_name_for_registration = channel_names.index(channel_name_for_registration)
        channel_axis = axes_image.find("C")

        if channel_axis == -1:
            raise ValueError(f"No channel indicator `C` found in image axes ({axes_image})")

        print(f"\t\tSelect image with nuclei from IF image (channel: {channel_name_for_registration})", flush=True)
        # select nuclei channel from IF image
        if channel_name_for_registration is None:
            raise TypeError("Argument `nuclei_channel` should be an integer and not NoneType.")

        # select dapi channel for registration
        nuclei_img = np.take(image, channel_name_for_registration, channel_axis)
        #selected = image[nuclei_channel]

    # Setup image registration objects - is important to load and scale the images.
    # The reason for this are limits in C++, not allowing to perform certain OpenCV functions on big images.

    # First: Setup the ImageRegistration object for the whole image (before deconvolution in histo images and multi-channel in IF)
    imreg_complete = ImageRegistration(
        image=image,
        template=template,
        axes_image=axes_image,
        axes_template=axes_template,
        verbose=False
        )
    # load and scale the whole image
    imreg_complete.load_and_scale_images()

    # setup ImageRegistration object with the nucleus image (either from deconvolution or just selected from IF image)
    imreg_selected = ImageRegistration(
        image=nuclei_img,
        template=imreg_complete.template,
        axes_image=axes_selected,
        axes_template=axes_template,
        max_width=4000,
        convert_to_grayscale=False,
        perspective_transform=False
    )

    # run all steps to extract features and get transformation matrix
    imreg_selected.load_and_scale_images()

    print("\t\tExtract common features from image and template", flush=True)
    # perform registration to extract the common features ptsA and ptsB
    imreg_selected.extract_features()
    imreg_selected.calculate_transformation_matrix()

    if image_type == "histo":
        # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
        if imreg_complete.image_resized is None:
            imreg_selected.image = imreg_complete.image  # use original image
        else:
            imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image

        # perform registration
        imreg_selected.perform_registration()

        if save_results:
            # save files
            identifier = f"{prefix}__{data.slide_id}__{data.sample_id}__{channel_names[0]}"
            #current_outfile = output_dir / f"{identifier}__registered.ome.tif"
            imreg_selected.save(
                output_dir=output_dir,
                identifier = identifier,
                axes=axes_image,
                photometric='rgb',
                ome_metadata=ome_metadata
                )

            # save metadata
            data.metadata["xenium"]['images'][f'registered_{channel_names[0]}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
            write_dict_to_json(data.metadata["xenium"], data.path / "experiment_modified.xenium")
            #self._save_metadata_after_registration()
        if add_registered_image:
            data.images.add_image(
                image=imreg_selected.registered,
                name=channel_names[0],
                axes=axes_image,
                pixel_size=pixel_size,
                ome_meta=ome_metadata
                )

        del imreg_complete, imreg_selected, image, template, nuclei_img, eo, dab
    else:
        # image_type is IF
        # In case of IF images the channels are normally in the first axis and each channel is registered separately
        # Further, each channel is then saved separately as grayscale image.

        # iterate over channels
        for i, n in enumerate(channel_names):
            # skip the DAPI image
            if n == channel_name_for_registration:
                break

            if imreg_complete.image_resized is None:
                # select one channel from non-resized original image
                imreg_selected.image = np.take(imreg_complete.image, i, channel_axis)
            else:
                # select one channel from resized original image
                imreg_selected.image_resized = np.take(imreg_complete.image_resized, i, channel_axis)

            # perform registration
            imreg_selected.perform_registration()

            if save_results:
                # save files
                identifier = f"{data.slide_id}__{data.sample_id}__{n}"
                #current_outfile = output_dir / f"{filename}__registered.ome.tif",
                imreg_selected.save(
                    #outfile = current_outfile,
                    output_dir=output_dir,
                    identifier=identifier,
                    #output_dir=self.path.parent / "registered_images",
                    #filename=f"{self.slide_id}__{self.sample_id}__{n}",
                    axes='YX',
                    photometric='minisblack',
                    ome_metadata=ome_metadata
                    )

                # save metadata
                data.metadata["xenium"]['images'][f'registered_{n}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
                write_dict_to_json(data.metadata["xenium"], data.path / "experiment_modified.xenium")
                #self._save_metadata_after_registration()
            if add_registered_image:
                data.images.add_image(
                    image=imreg_selected.registered,
                    name=n,
                    axes=axes_image,
                    pixel_size=pixel_size,
                    ome_meta=ome_metadata
                    )

        # free RAM
        del imreg_complete, imreg_selected, image, template, nuclei_img
    gc.collect()


def calc_distance_of_cells_from(
    data: InSituData,
    annotation_key: str,
    annotation_class: str,
    region_key: Optional[str] = None,
    region_name: Optional[str] = None,
    key_to_save: Optional[str] = None
    ):

    """
    Calculate the distance of cells from a specified annotation class within a given region and save the results.

    This function calculates the distance of each cell in the spatial data to the closest point
    of a specified annotation class. The distances are then saved in the cell data matrix.

    Args:
        data (InSituData): The input data containing cell and annotation information.
        annotation_key (str): The key to retrieve the annotation information.
        annotation_class (Optional[str]): The specific annotation class to calculate distances from.
        region_key: (Optional[str]): If not None, `region_key` is used together with `region_name` to determine the region in which cells are considered
                                     for the analysis.
        region_name: (Optional[str]): If not None, `region_name` is used together with `region_key` to determine the region in which cells are considered
                                     for the analysis.
        key_to_save (Optional[str]): The key under which to save the calculated distances in the cell data matrix.
                                     If None, a default key is generated based on the annotation class.

    Returns:
        None
    """
    if region_name is None:
        print(f'Calculate the distance of cells from the annotation "{annotation_class}"')
        region_mask = [True] * len(data.cells.matrix)
    else:
        assert region_key is not None, "`region_key` must not be None if `region_name` is not None."
        print(f'Calculate the distance of cells from the annotation "{annotation_class}" within region "{region_name}"')
        region_col_name = f'regions-{region_key}'

        if region_col_name not in data.cells.matrix.obs.columns:
            data.assign_regions(keys=region_key)

        # generate mask for selected region
        region_mask = data.cells.matrix.obs[region_col_name] == region_name

    # create geopandas points from cells
    x = data.cells.matrix.obsm["spatial"][:, 0][region_mask]
    y = data.cells.matrix.obsm["spatial"][:, 1][region_mask]
    indices = data.cells.matrix.obs_names[region_mask]
    cells = gpd.points_from_xy(x, y)

    # retrieve annotation information
    annot_df = data.annotations.get(annotation_key)
    class_df = annot_df[annot_df["name"] == annotation_class]

    # calculate distance of cells to their closest point
    scaled_geometries = [
        scale_func(geometry, xfact=scale[0], yfact=scale[1], origin=(0,0))
        for geometry, scale in zip(class_df["geometry"], class_df["scale"])
        ]
    dists = np.array([cells.distance(geometry) for geometry in scaled_geometries])
    min_dists = dists.min(axis=0)

    # add indices to minimum distances
    min_dists = pd.Series(min_dists, index=indices)

    # add results to CellData
    if key_to_save is None:
        key_to_save = f"dist_from_{annotation_class}"
    data.cells.matrix.obs[key_to_save] = min_dists
    print(f'Save distances to `.cells.matrix.obs["{key_to_save}"]`')

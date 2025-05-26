#from __future__ import annotations  # this prevents circular imports

import gc
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
from warnings import warn

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.api.types import is_numeric_dtype

from insitupy._constants import (DEFAULT_CATEGORICAL_CMAP,
                                 DEFAULT_CONTINUOUS_CMAP)
from insitupy._core._checks import _is_experiment, check_raw
from insitupy._core._utils import _get_cell_layer
from insitupy._core.dataclasses import ImageData, RegionsData
from insitupy._core.insitudata import InSituData
from insitupy._core.insituexperiment import InSituExperiment
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting._colors import (_add_colorlegend_to_axis,
                                       _extract_color_values,
                                       create_cmap_mapping)
from insitupy.utils._adata import filter_anndata
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)


class _ColorConfigMultiPlot:
    def __init__(
        self,
        data: Union[InSituData, InSituExperiment],
        cells_layer: Optional[str] = None,
        keys: Union[str, List[str]] = None,
        raw: bool = False,
        layer: Optional[str] = None,
        palette = DEFAULT_CATEGORICAL_CMAP
    ):
        # add properties
        self._dict = {}

        # get parameters
        self.cells_layer = cells_layer
        self.raw = raw
        self.layer = layer
        self.palette = palette

        if _is_experiment(data):
            data_list = data.data
            exp_color_dict = data.colors
            is_experiment = True
        else:
            data_list = [data]
            exp_color_dict = {}
            is_experiment = False

        for key in keys:
            if key in exp_color_dict:
                # use color_dict from InSituExperiment
                color_entry = {
                    "color_dict": exp_color_dict[key],
                    "max_value": None,
                    "is_categorical": True,
                    "crange": None
                }
            else:
                # EITHER because key is continuous
                # OR because data was InSituData
                # OR because sync_colors was not called on InSituExperiment
                color_entry = self._infer_color_entry(data_list=data_list, key=key)

            # add entry to dictionary
            self._dict[key] = color_entry

    def __getitem__(self, key):
        return self._dict.get(key)

    @property
    def dict(self):
        return self._dict

    def keys(self):
        return self._dict.keys()

    def _infer_color_entry(
        self, data_list, key
        ):
        # preconfigure entry
        color_entry = {
            "color_dict": None,
            "max_value": None,
            "is_categorical": False,
            "crange": None
        }
        if len(data_list) == 1:
            # one dataset
            xd = data_list[0]
            celldata = _get_cell_layer(
                cells=xd.cells,
                cells_layer=self.cells_layer
                )
            ad = celldata.matrix

            # extract the data
            color_values, is_categorical = _extract_color_values(
                adata=ad, key=key, raw=self.raw, layer=self.layer
            )

            if is_categorical:
                color_entry["is_categorical"] = True
                # check if colors were saved in uns
                uns_key = f"{key}_colors"
                if uns_key in ad.uns.keys() and self.palette is None:
                    hex_list = ad.uns[uns_key]
                    color_entry["color_dict"] = {
                        c: hex_list[i] for i, c in enumerate(color_values.cat.categories)
                        }
                else:
                    # create a new color mapping
                    color_entry["color_dict"] = create_cmap_mapping(
                        color_values.values, cmap=self.palette
                        )

            else:
                # no values are categorical - collect the maximum values
                color_entry["max_value"] = np.max(color_values)
                color_entry["is_categorical"] = False
                color_entry["crange"] = [0, color_entry["max_value"]]

        else:
            # multiple datasets
            value_list = []
            categorical_list = []
            for xd in data_list:
                celldata = _get_cell_layer(
                    cells=xd.cells,
                    cells_layer=self.cells_layer
                    )
                ad = celldata.matrix

                # extract the data
                color_values, is_categorical = _extract_color_values(
                    adata=ad, key=key, raw=self.raw, layer=self.layer
                )

                if is_categorical:
                    value_list.append(np.unique(color_values))
                else:
                    value_list.append(np.max(color_values))

                categorical_list.append(is_categorical)

            if np.all(categorical_list):
                # all values are categorical - concatenate all values
                all_values = np.unique(np.concat(value_list))
                color_entry["color_dict"] = create_cmap_mapping(
                    all_values, cmap=self.palette
                    )
                color_entry["is_categorical"] = True

            elif not np.any(categorical_list):
                # no values are categorical - collect the maximum values
                color_entry["max_value"] = np.max(value_list)
                color_entry["is_categorical"] = False
                color_entry["crange"] = [0, color_entry["max_value"]]
            else:
                raise ValueError(f"Values found for key {key} showed mixed type (categorical/numeric).")

        return color_entry

class _SinglePlotConfig:
    '''
    Object extracting spatial coordinates and expression data from anndata object.
    '''
    def __init__(
        self,
        adata: AnnData,
        key: List[str],
        ax: plt.Axes,
        name: str,
        idx_key: int,
        color_config: dict,
        add_legend: bool,
        RegionDataObject: Optional[RegionsData],
        region_tuple: Optional[Tuple[str, str]],
        ImageDataObject: Optional[ImageData],
        image_key: Optional[str] = None,
        pixelwidth_per_subplot: int = 200,
        raw: bool = False,
        layer: Optional[str] = None,
        obsm_key: str = 'spatial',
        origin_zero: bool = False, # whether to start axes ticks at 0
        xlim_general: Optional[Tuple[int, int]] = None,
        ylim_general: Optional[Tuple[int, int]] = None,
        histogram_setting: Optional[Union[Literal["auto"], Tuple[int, int]]] = "auto",

        ):

        # add arguments to object
        self.key = key
        self.ax = ax
        self.name = name
        self.idx_key = idx_key
        self.add_legend = add_legend

        # retrieve color dictionary
        self.color_dict = color_config[key]["color_dict"]
        self.crange = color_config[key]["crange"]
        self.categorical = color_config[key]["is_categorical"] # True if color_dict is not None
        self.crange = color_config[key]["crange"]

        # prepare limits using region or lim arguments
        if region_tuple is not None:
            if xlim_general is not None or ylim_general is not None:
                raise ValueError("If region_tuple is given, xlim and ylim need to be None.")
            else:
                region_df = RegionDataObject[region_tuple[0]]
                geom = region_df[region_df["name"] == region_tuple[1]]["geometry"].item()
                self.xlim = [geom.bounds[0], geom.bounds[2]]
                self.ylim = [geom.bounds[1], geom.bounds[3]]
        else:
            # make sure limits are lists
            self.xlim = list(xlim_general) if xlim_general is not None else xlim_general
            self.ylim = list(ylim_general) if ylim_general is not None else ylim_general

        ## Extract coordinates
        # extract x and y pixel coordinates and convert to micrometer
        self.x_coords = adata.obsm[obsm_key][:, 0].copy()
        self.y_coords = adata.obsm[obsm_key][:, 1].copy()

        # shift coordinates that they start at (0,0)
        if origin_zero:
            self.x_offset = self.x_coords.min()
            self.y_offset = self.y_coords.min()
            self.x_coords -= self.x_offset
            self.y_coords -= self.y_offset
        else:
            self.x_offset = self.y_offset = 0

        if self.xlim is None:
            # xmin = np.min([self.x_coords.min(), self.y_coords.min()]) # make sure that result is always a square
            # xmax = np.max([self.x_coords.max(), self.y_coords.max()])
            xmin = self.x_coords.min()
            xmax = self.x_coords.max()

            # include margin
            #self.xlim = (xmin - spot_size, xmax + spot_size)
            self.xlim = (xmin, xmax)

        if self.ylim is None:
            # ymin = np.min([self.x_coords.min(), self.y_coords.min()])
            # ymax = np.max([self.x_coords.max(), self.y_coords.max()])
            ymin = self.y_coords.min()
            ymax = self.y_coords.max()

            # include margin
            #self.ylim = (ymin - spot_size, ymax + spot_size)
            self.ylim = (ymin, ymax)

        # extract image information
        if ImageDataObject is not None:
            # pick the image with the right resolution for plotting
            max_pixel_size = np.max([self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]]) / pixelwidth_per_subplot
            orig_pixel_size = ImageDataObject.metadata[image_key]["pixel_size"]
            img_pyramid = ImageDataObject[image_key]
            pixel_sizes_levels = np.array([orig_pixel_size * (2**i) for i in range(len(img_pyramid))])

            try:
                selected_level = np.where(pixel_sizes_levels <= max_pixel_size)[0][-1].item()
                selected_pixel_size = pixel_sizes_levels[selected_level].item()
            except IndexError:
                selected_level = 0
                selected_pixel_size = pixel_sizes_levels[selected_level].item()

            # extract parameters from ImageDataObject
            self.pixel_size = selected_pixel_size
            self.image = img_pyramid[selected_level]

            ywidth = self.image.shape[0]
            xwidth = self.image.shape[1]

            # determine limits for selected pyramid image - clip to maximum image dims (important for extent of image during plotting)
            self.pixel_xlim = np.clip([int(elem / selected_pixel_size) for elem in self.xlim], a_min=0, a_max=xwidth).tolist()
            self.pixel_ylim = np.clip([int(elem / selected_pixel_size) for elem in self.ylim], a_min=0, a_max=ywidth).tolist()

            # crop image
            self.image = self.image[
                self.pixel_ylim[0]:self.pixel_ylim[1],
                self.pixel_xlim[0]:self.pixel_xlim[1]
                ]

            if histogram_setting is None:
                self.vmin = self.vmax = None
            elif histogram_setting == "auto":
                self.vmin = da.percentile(self.image.ravel(), 30).compute().item()
                self.vmax = da.percentile(self.image.ravel(), 99.5).compute().item()
            elif isinstance(histogram_setting, tuple):
                self.vmin = histogram_setting[0]
                self.vmax = histogram_setting[1]
            else:
                raise ValueError(f"Unknown type for histogram_setting: {type(histogram_setting)}")

        else:
            self.image = None

        # get color values for expression data or categories
        self.color_values, self.categorical = _extract_color_values(
            adata=adata, key=self.key, raw=raw, layer=layer
        )

class MultiSpatialPlot:
    '''
    Class to render scatter plots of single-cell spatial transcriptomics data.
    '''
    def __init__(self, **kwargs):
        # Automatically assign all keyword arguments to the instance
        for key, value in kwargs.items():
            setattr(self, key, value)

        # convert arguments to lists
        self.keys = convert_to_list(self.keys)

        # check if cmap is supposed to be centered
        if self.cmap_center is None:
            self.normalize=None
        else:
            self.normalize = colors.CenteredNorm(vcenter=self.cmap_center)

        # set multiplot variables
        self.multikeys = False
        self.multidata = False
        if len(self.keys) > 1:
            self.multikeys = True

        # check whether the data is an InSituExperiment or a single InSituData
        if _is_experiment(self.data):
            self.n_data = len(self.data)
            self.is_experiment = True

            if self.sync_colors:
                # synchronize colors before plotting
                self.data.sync_colors(
                    obs_cols=self.keys,
                    cells_layer=self.cells_layer,
                    palette=self.palette
                )
            else:
                if self.is_experiment:
                    # Todo: Need to make sure that the colors are already synchronized if data is categorical and an InSituExperiment
                    warn(f"`sync_colors=False`, meaning that colors are not being synchronized. "
                         f"In case of categorical data, this might lead to different color mappings "
                         f"between the datasets. Set `sync_colors=True` or use InSituExperiments's `.sync_colors` function "
                         f"to synchronize colors", UserWarning, 2)

        else:
            self.n_data = 1
            self.is_experiment = False

        if self.n_data > 1:
            self.multidata = True
        elif self.n_data == 1:
            self.multidata = False
        else:
            raise ValueError(f"n_data < 1: {self.n_data}")

        self.color_config = _ColorConfigMultiPlot(
            data=self.data,
            cells_layer=self.cells_layer,
            keys=self.keys,
            raw=self.raw,
            layer=self.layer,
            palette=self.palette
        )

    def setup_subplots(self):
        print("Setup subplots.") if self.verbose else None
        self.add_legend_to_last_subplot = False
        if self.multidata:
            if self.multikeys:
                # determine the layout of the subplots
                self.n_rows = self.n_data
                self.max_cols = len(self.keys)
                self.n_plots = self.n_rows * self.max_cols

                # create subplots
                self.fig, self.axs = plt.subplots(
                    self.n_rows, self.max_cols,
                    figsize=(6 * self.max_cols, 6 * self.n_rows),
                    dpi=self.dpi_display)
                #self.fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.
            else:
                if self.color_config[self.keys[0]]["color_dict"] is None:
                    # continuous data
                    n_subplots = self.n_data
                    self.add_legend_to_last_subplot = False
                else:
                    # categorical data
                    n_subplots = self.n_data+1
                    self.add_legend_to_last_subplot = True # is the case for multidata=True and multikeys=False for categorical data

                # determine the layout of the subplots
                self.n_plots, self.n_rows, self.max_cols = get_nrows_maxcols(
                    n_keys=n_subplots,
                    max_cols=self.max_cols
                    )
                self.fig, self.axs = plt.subplots(
                    self.n_rows, self.max_cols,
                    figsize=(6 * self.max_cols, 6 * self.n_rows),
                    dpi=self.dpi_display)
                #self.fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.

                if self.n_plots > 1:
                    self.axs = self.axs.ravel()
                else:
                    self.axs = [self.axs]

                remove_empty_subplots(
                    axes=self.axs,
                    nplots=self.n_plots,
                    nrows=self.n_rows,
                    ncols=self.max_cols
                    )

        else:
            self.n_plots = len(self.keys)
            if self.max_cols is None:
                self.max_cols = self.n_plots
                self.n_rows = 1
            else:
                if self.n_plots > self.max_cols:
                    self.n_rows = math.ceil(self.n_plots / self.max_cols)
                else:
                    self.n_rows = 1
                    self.max_cols = self.n_plots

            self.fig, self.axs = plt.subplots(
                self.n_rows, self.max_cols,
                figsize=(6 * self.max_cols, 6 * self.n_rows),
                dpi=self.dpi_display)

            if self.n_plots > 1:
                self.axs = self.axs.ravel()
            else:
                self.axs = np.array([self.axs])

            # remove axes from empty plots
            remove_empty_subplots(
                axes=self.axs,
                nplots=self.n_plots,
                nrows=self.n_rows,
                ncols=self.max_cols,
                )

        if self.header is not None:
            plt.suptitle(self.header, fontsize=18, x=0.5, y=0.98)

    def _set_axis(
        self,
        ConfigData: _SinglePlotConfig
        ):
        ax = ConfigData.ax

        # set axis
        ax.set_xlim(ConfigData.xlim[0], ConfigData.xlim[1])
        ax.set_ylim(ConfigData.ylim[0], ConfigData.ylim[1])
        ax.set_xlabel('µm', fontsize=self.label_size)
        ax.set_ylabel('µm', fontsize=self.label_size)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_aspect(1)
        ax.set_facecolor(self.background_color)
        ax.tick_params(labelsize=self.tick_label_size)

        if self.multidata and not self.multikeys:
            ax.set_title(
                ConfigData.name + "\n" + ConfigData.key,
                fontsize=self.title_size, #fontweight='bold',
                pad=10,
                rotation=0
                )
        else:
            # set titles
            ax.set_title(
                ConfigData.key,
                fontsize=self.title_size, #fontweight='bold'
                pad=10
                )

            if ConfigData.idx_key == 0:
                ax.annotate(
                    ConfigData.name,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label,
                    textcoords='offset points',
                    size=14,
                    rotation=90,
                    ha='right', va='center',
                    weight='bold'
                    )

    def _determine_axes(self, idx, idx_key):
        if len(self.axs.shape) == 2:
            ax = self.axs[idx, idx_key]
            if idx == (self.n_rows - 1):
                add_legend = True
            else:
                add_legend = False
        elif len(self.axs.shape) == 1:
            if self.multikeys:
                ax = self.axs[idx_key]
                add_legend = True
            else:
                ax = self.axs[idx]

                if len(self.axs) == 1:
                    add_legend = True
                else:
                    add_legend = False
        else:
            raise ValueError("`len(self.axs.shape)` has wrong shape {}. Requires 1 or 2.".format(len(self.axs.shape)))

        return ax, add_legend

    def _get_data(self, idx):
        # extract the InSituData object
        try:
            xd = self.data.data[idx]
            meta = self.data.metadata.iloc[idx]
        except AttributeError:
            xd = self.data
            meta = None

        # retrieve the right cell data
        celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)

        # extract anndata
        adata = celldata.matrix

        # filter anndata
        if self.filter_mode is not None and self.filter_tuple is not None:
            adata = filter_anndata(
                adata=adata, filter_mode=self.filter_mode, filter_tuple=self.filter_tuple
                )

        if self.name_column is None or meta is None:
            sample_name = xd.sample_id
        else:
            sample_name = meta[self.name_column]

        if self.image_key is not None:
            imagedata = xd.images
        else:
            imagedata = None

        # get regions
        regions = xd.regions

        return adata, sample_name, imagedata, regions

    def plot_to_subplots(self):
        print("Do plotting.") if self.verbose else None
        #i = 0
        for idx in range(self.n_data):
            # # extract the InSituData object
            # try:
            #     xd = self.data.data[idx]
            #     meta = self.data.metadata.iloc[idx]
            # except AttributeError:
            #     xd = self.data
            #     meta = None

            # # retrieve the right cell data
            # celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)

            # # extract anndata
            # ad = celldata.matrix

            # # filter anndata
            # ad = ad[ad.obs["cell_type"].str.contains("Cancer")]

            # if self.name_column is None or meta is None:
            #     sample_name = xd.sample_id
            # else:
            #     sample_name = meta[self.name_column]

            # if self.image_key is not None:
            #     imagedata = xd.images
            # else:
            #     imagedata = None

            ad, sample_name, imagedata, regions = self._get_data(idx)


            for idx_key, key in enumerate(self.keys):
                # get axis to plot
                if self.ax is None:
                    ax, add_legend = self._determine_axes(idx, idx_key)
                else:
                    ax = self.ax

                # get data
                ConfigData = _SinglePlotConfig(
                    adata=ad,
                    key=key,
                    ax=ax,
                    name=sample_name,
                    idx_key=idx_key,
                    color_config=self.color_config,
                    add_legend=add_legend,
                    RegionDataObject=regions,
                    region_tuple=self.region_tuple,
                    ImageDataObject=imagedata,
                    image_key=self.image_key,
                    pixelwidth_per_subplot=self.pixelwidth_per_subplot,
                    raw=self.raw,
                    layer=self.layer,
                    obsm_key=self.obsm_key,
                    origin_zero=self.origin_zero,
                    xlim_general=self.xlim,
                    ylim_general=self.ylim,
                    histogram_setting=self.histogram_setting
                )

                if ConfigData.color_values is not None:
                    # set the axes (add titles, set limits, etc.)
                    self._set_axis(ConfigData)

                    # plot single spatial plot in given axis
                    self.single_spatial(
                        ConfigData=ConfigData,
                        # color_dict=color_dict,
                        # crange=ConfigData.crange,
                        # add_legend=add_legend,
                        )
                else:
                    print("Key '{}' not found.".format(key), flush=True)
                    ax.set_axis_off()

                # free RAM
                del ConfigData
                gc.collect()

            # free RAM
            del imagedata
            gc.collect()

        if self.add_legend_to_last_subplot:
            # get axis of last subplots for color legend
            ax = self.axs[self.n_plots-1]

            k = list(self.color_config.keys())[0]
            color_config_key = self.color_config[k]
            # is_categorical = color_config_key["is_categorical"]
            # if is_categorical:
            color_dict = self.color_config[k]["color_dict"]
            _add_colorlegend_to_axis(color_dict=color_dict, ax=ax)

            # else:
            #     #if ConfigData.categorical:
            #     ax.set_axis_off()


    def single_spatial(
        self,
        ConfigData: _SinglePlotConfig,
        #axis: plt.Axes,
        # color_dict: Dict,
        # crange: Optional[Tuple[float, float]],
        # add_legend: bool,
        ):

        # calculate marker size
        pixels_per_unit = ConfigData.ax.transData.transform(
            [(0, 1), (1, 0)]) - ConfigData.ax.transData.transform((0, 0))
        # x_ppu = pixels_per_unit[1, 0]
        y_ppu = pixels_per_unit[0, 1]
        pxs = y_ppu * self.spot_size
        size = (72. / self.fig.dpi * pxs)**2

        if ConfigData.image is not None:
            # plot image data
            extent = (
                ConfigData.pixel_xlim[0] * ConfigData.pixel_size - 0.5,
                ConfigData.pixel_xlim[1] * ConfigData.pixel_size - 0.5,
                ConfigData.pixel_ylim[1] * ConfigData.pixel_size - 0.5,
                ConfigData.pixel_ylim[0] * ConfigData.pixel_size - 0.5
                )

            ConfigData.ax.imshow(
                ConfigData.image,
                extent=extent,
                origin='upper', cmap='gray', vmin=ConfigData.vmin, vmax=ConfigData.vmax)

        # plot transcriptomic data
        if ConfigData.categorical:
            sns.scatterplot(
                x=ConfigData.x_coords, y=ConfigData.y_coords,
                hue=ConfigData.color_values,
                marker=self.spot_type,
                s=size,
                linewidth=0,
                palette=ConfigData.color_dict,
                alpha=self.alpha,
                ax=ConfigData.ax
                )
            # add legend
            # divide axis to fit legend
            divider = make_axes_locatable(ConfigData.ax)
            lax = divider.append_axes("bottom", size="2%", pad=0)

            if ConfigData.add_legend:
                _add_colorlegend_to_axis(
                    color_dict=ConfigData.color_dict,
                    ax=lax,
                    max_per_col=10,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -10)
                    )

            # Remove the axis ticks and labels
            lax.set_xticks([])
            lax.set_yticks([])
            lax.axis('off')

            # Remove the legend from the main axis
            ConfigData.ax.legend().remove()
        else:
            s = ConfigData.ax.scatter(
                ConfigData.x_coords,
                ConfigData.y_coords,
                c=ConfigData.color_values,
                marker=self.spot_type,
                s=size,
                alpha=self.alpha,
                linewidths=0,
                cmap=self.cmap,
                norm=self.normalize
                )

            # divide axis to fit colorbar
            divider = make_axes_locatable(ConfigData.ax)
            cax = divider.append_axes("right", size="4%", pad=0.1)

            # add colorbar
            clb = self.fig.colorbar(s, cax=cax, orientation='vertical')
            # set colorbar
            clb.ax.tick_params(labelsize=self.tick_label_size)

            if self.clb_title is not None:
                clb.ax.set_xlabel(
                    self.clb_title,  # Change to xlabel for horizontal orientation
                    fontdict={"fontsize": self.label_size},
                    labelpad=20
                    )

            if ConfigData.crange is not None:
                clb.mappable.set_clim(
                    ConfigData.crange[0],
                    ConfigData.crange[1]
                    )
            else:
                if self.crange_type == 'percentile':
                    clb.mappable.set_clim(0, np.percentile(ConfigData.color_values, 99))

FilterMode = Literal[
    "contains", "not contains", "starts with", "ends with",
    "is equal", "is not", "in", "not in",
    "greater than", "less than", "greater or equal", "less or equal"
    ]

def plot_spatial(
    data: Union[InSituData, InSituExperiment],
    keys: Union[str, List[str]],
    cells_layer: Optional[str] = None,
    raw: bool = False,
    layer: Optional[str] = None,
    filter_mode: Optional[FilterMode] = None,
    filter_tuple: Optional[Tuple[str, Union[str, int, float, List[Union[str, int, float]]]]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    max_cols: int = 4,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    region_tuple: Tuple[str, str] = None,
    crange: Optional[List[int]] = None,
    crange_type: Literal['minmax', 'percentile'] = 'minmax',
    palette: str = DEFAULT_CATEGORICAL_CMAP,
    cmap_center: Optional[float] = None,
    dpi_display: int = 80,
    obsm_key: str = 'spatial',
    origin_zero: bool = False,
    spot_size: float = 10,
    spot_type: str = 'o',
    cmap: str = DEFAULT_CONTINUOUS_CMAP,
    sync_colors: bool = True,
    background_color: str = 'white',
    alpha: float = 1,
    colorbar: bool = True,
    clb_title: Optional[str] = None,
    header: Optional[str] = None,
    name_column: Optional[str] = None,
    title_size: int = 24,
    label_size: int = 16,
    tick_label_size: int = 14,
    image_key: Optional[str] = None,
    pixelwidth_per_subplot: int = 200,
    histogram_setting: Optional[Union[Literal["auto"], Tuple[int, int]]] = "auto",
    savepath: Optional[str] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    show: bool = True,
    verbose: bool = False,
):
    plotter = MultiSpatialPlot(
        data=data,
        keys=keys,
        cells_layer=cells_layer,
        raw=raw,
        layer=layer,
        filter_mode=filter_mode,
        filter_tuple=filter_tuple,
        fig=fig,
        ax=ax,
        max_cols=max_cols,
        xlim=xlim,
        ylim=ylim,
        region_tuple=region_tuple,
        crange=crange,
        crange_type=crange_type,
        palette=palette,
        cmap_center=cmap_center,
        dpi_display=dpi_display,
        obsm_key=obsm_key,
        origin_zero=origin_zero,
        spot_size=spot_size,
        spot_type=spot_type,
        cmap=cmap,
        sync_colors=sync_colors,
        background_color=background_color,
        alpha=alpha,
        colorbar=colorbar,
        clb_title=clb_title,
        header=header,
        name_column=name_column,
        title_size=title_size,
        label_size=label_size,
        tick_label_size=tick_label_size,
        image_key=image_key,
        pixelwidth_per_subplot=pixelwidth_per_subplot,
        histogram_setting=histogram_setting,
        savepath=savepath,
        save_only=save_only,
        dpi_save=dpi_save,
        show=show,
        verbose=verbose,
    )

    # # Now use the config object as before
    # plotter.prepare_colors()

    if plotter.ax is None:
        plotter.setup_subplots()
    else:
        assert plotter.fig is not None, "If axis for plotting is given, also a figure object needs to be provided via `fig`"
        assert len(plotter.keys) == 1, "If single axis is given not more than one key is allowed."

    plotter.plot_to_subplots()

    save_and_show_figure(
        savepath=plotter.savepath,
        fig=plotter.fig,
        save_only=plotter.save_only,
        show=plotter.show,
        dpi_save=plotter.dpi_save
    )

    gc.collect()


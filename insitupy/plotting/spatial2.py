
import gc
import math
from dataclasses import dataclass, fields
from typing import List, Literal, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from insitupy._constants import (DEFAULT_CATEGORICAL_CMAP,
                                 DEFAULT_CONTINUOUS_CMAP)
from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy._io.plots import save_and_show_figure
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.dataclasses.dataclasses import (AnnotationsData, ImageData,
                                              RegionsData)
from insitupy.experiment.data import InSituExperiment
from insitupy.utils._adata import filter_anndata
from insitupy.utils._colors import (_add_colorlegend_to_axis,
                                    _extract_color_values, _rgb2hex_robust,
                                    create_cmap_mapping)
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)

FilterMode = Literal[
    "contains", "not contains", "starts with", "ends with",
    "is equal", "is not", "in", "not in",
    "greater than", "less than", "greater or equal", "less or equal"
    ]


# -------------------------------
# CONFIG OBJECTS
# -------------------------------

class UpdatableConfig:
    def update_values(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}.")

    def show_all(self):
        print(f"Configuration parameters for {self.__class__.__name__}:")
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            print(f"\t{name}: {value}")


@dataclass
class DataConfig(UpdatableConfig):
    # data extraction config
    layer: Optional[str] = None
    raw: bool = False
    obsm_key: str = 'spatial'
    name_column: Optional[str] = None

    # data attribute keys
    region_tuple: Optional[Tuple[str, str]] = None
    annotations_key: Optional[Tuple[str, Optional[Union[str, List[str]]]]] = None
    image_key: Optional[str] = None

    # filters
    filter_mode: Optional[str] = None
    filter_tuple: Optional[Tuple] = None

@dataclass
class PlotConfig(UpdatableConfig):
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    spot_size: float = 10
    alpha: float = 1.0
    cmap: str = DEFAULT_CONTINUOUS_CMAP
    palette: str = DEFAULT_CATEGORICAL_CMAP
    spot_type: str = "o"
    background_color: str = "white"
    cmap_center: Optional[float] = None
    normalize: Optional[colors.Normalize] = None
    legend_max_per_col: int = 10
    clb_title: Optional[str] = None
    annotations_mode: Literal["outlined", "filled"] = "outlined"
    crange: Optional[List[int]] = None
    crange_type: Literal['minmax', 'max', 'upper_percentile', 'percentile'] = 'upper_percentile'
    origin_zero: bool = False
    label_size: int = 16
    title_size: int = 18
    tick_label_size: int = 14
    pixelwidth_per_subplot: int = 200
    histogram_setting: Union[Literal["auto"], Tuple[int, int], None] = "auto"

    def __post_init__(self):
        # check if cmap is supposed to be centered
        if self.cmap_center is not None:
            self.normalize = colors.CenteredNorm(vcenter=self.cmap_center)

@dataclass
class LayoutConfig(UpdatableConfig):
    max_cols: Optional[int] = 4
    header: Optional[str] = None
    multikeys: bool = False
    multidata: bool = False
    n_rows: int = None
    n_cols: int = None
    n_plots: int = None
    subplot_width: int = 6
    subplot_height: int = 6
    figsize: Optional[Tuple] = None
    add_legend_to_last_subplot: bool = False
    dpi_display: int = 80
    def calc_subplot_params(self, keys, n_data, color_config):
        # set multiplot variables
        if len(keys) > 1:
            self.multikeys = True

        if n_data > 1:
            self.multidata = True
        elif n_data == 1:
            self.multidata = False
        else:
            raise ValueError(f"n_data < 1: {n_data}")

        if self.multidata:
            if self.multikeys:
                # determine the layout of the subplots
                self.n_rows = n_data
                self.n_cols = len(keys)
                self.n_plots = self.n_rows * self.n_cols

            else:
                if color_config[keys[0]]["color_dict"] is None:
                    # continuous data
                    n_subplots = n_data
                    self.add_legend_to_last_subplot = False
                else:
                    # categorical data
                    n_subplots = n_data+1
                    self.add_legend_to_last_subplot = True # is the case for multidata=True and multikeys=False for categorical data

                if self.max_cols is None:
                    self.max_cols = n_subplots

                # determine the layout of the subplots
                self.n_plots, self.n_rows, self.n_cols = get_nrows_maxcols(
                    n_keys=n_subplots,
                    max_cols=self.max_cols
                    )

        else:
            self.n_plots = len(keys)
            if self.max_cols is None:
                self.n_cols = self.n_plots
                self.n_rows = 1
            else:
                if self.n_plots > self.max_cols:
                    self.n_cols = self.max_cols
                    self.n_rows = math.ceil(self.n_plots / self.max_cols)
                else:
                    self.n_rows = 1
                    self.n_cols = self.n_plots

        if self.figsize is None:
            self.figsize = (self.subplot_width * self.n_cols, self.subplot_height * self.n_rows)

def _get_crange(color_values, crange_type):
    if crange_type == 'max':
        crange = [0, np.max(color_values)]
    elif crange_type == 'minmax':
        crange = [np.min(color_values), np.max(color_values)]
    elif crange_type == 'upper_percentile':
        crange = [0, np.percentile(color_values, 99)]
    elif crange_type == 'percentile':
        crange = [np.percentile(color_values, 1), np.percentile(color_values, 99)]
    else:
        raise ValueError(f"Unknown crange_type: {crange_type}. Must be one of 'max', 'minmax', 'upper_percentile' or 'percentile'.")

    return crange

class ColorConfigMultiPlot:
    def __init__(
        self,
        data: Union[InSituData, InSituExperiment],
        data_config: DataConfig,
        plot_config: PlotConfig,
        cells_layer: Optional[str] = None,
        keys: Union[str, List[str]] = None,
        ):
        # add properties
        self._dict = {}

        if _is_experiment(data):
            data_list = data.data
            exp_color_dict = data.colors
            # is_experiment = True
        else:
            data_list = [data]
            exp_color_dict = {}
            # is_experiment = False

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
                color_entry = self._add_color_entry(
                    data_list=data_list,
                    key=key,
                    cells_layer=cells_layer,
                    data_config=data_config,
                    plot_config=plot_config
                    )

            # add entry to dictionary
            self._dict[key] = color_entry

    def __getitem__(self, key):
        return self._dict.get(key)

    @property
    def dict(self):
        return self._dict

    def keys(self):
        return self._dict.keys()

    def _add_color_entry(
        self, data_list, key, cells_layer,
        data_config, plot_config
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
                cells_layer=cells_layer
                )
            ad = celldata.matrix

            # extract the data
            color_values, is_categorical = _extract_color_values(
                adata=ad, key=key, raw=data_config.raw, layer=data_config.layer
            )

            if is_categorical:
                color_entry["is_categorical"] = True
                # check if colors were saved in uns
                uns_key = f"{key}_colors"
                if uns_key in ad.uns.keys() and plot_config.palette is None:
                    hex_list = ad.uns[uns_key]
                    color_entry["color_dict"] = {
                        c: hex_list[i] for i, c in enumerate(color_values.cat.categories)
                        }
                else:
                    # create a new color mapping
                    color_entry["color_dict"] = create_cmap_mapping(
                        color_values.values, cmap=plot_config.palette
                        )

            else:
                # no values are categorical - collect the maximum values
                color_entry["max_value"] = np.max(color_values)
                color_entry["is_categorical"] = False

                if plot_config.crange is not None:
                    color_entry["crange"] = plot_config.crange
                else:
                    color_entry["crange"] = _get_crange(
                    color_values=color_values,
                    crange_type=plot_config.crange_type
                    )

        else:
            # multiple datasets
            value_list = []
            categorical_list = []
            for xd in data_list:
                celldata = _get_cell_layer(
                    cells=xd.cells,
                    cells_layer=cells_layer
                    )
                ad = celldata.matrix

                # extract the data
                color_values, is_categorical = _extract_color_values(
                    adata=ad, key=key, raw=data_config.raw, layer=data_config.layer
                )

                if is_categorical:
                    value_list.append(np.unique(color_values))
                else:
                    value_list.append(np.max(color_values))

                categorical_list.append(is_categorical)

            if np.all(categorical_list):
                # all values are categorical - concatenate all values
                all_values = np.unique(np.concatenate(value_list))
                color_entry["color_dict"] = create_cmap_mapping(
                    all_values, cmap=plot_config.palette
                    )
                color_entry["is_categorical"] = True

            elif not np.any(categorical_list):
                # no values are categorical - collect the maximum values
                color_entry["max_value"] = np.max(value_list)
                color_entry["is_categorical"] = False
                # color_entry["crange"] = [0, color_entry["max_value"]]

                if plot_config.crange is not None:
                    color_entry["crange"] = plot_config.crange
                else:
                    color_entry["crange"] = _get_crange(
                        color_values=value_list,
                        crange_type=plot_config.crange_type
                        )
            else:
                raise ValueError(f"Values found for key {key} showed mixed type (categorical/numeric).")

        return color_entry


def plot_spatial2(
    data: Union[InSituData, InSituExperiment],
    keys: Union[str, List[str]],
    cells_layer: Optional[str] = None,
    layer: Optional[str] = None,

    # data attribute keys
    region_tuple: Optional[Tuple[str, str]] = None,
    annotations_key: Optional[Tuple[str, Optional[Union[str, List[str]]]]] = None,
    image_key: Optional[str] = None,

    # filters
    filter_mode: Optional[str] = None,
    filter_tuple: Optional[Tuple] = None,

    # plotting configs
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    spot_size: float = 10,
    alpha: float = 1.0,

    # layout configs
    max_cols: Optional[int] = 4,

    # save configs
    savepath: Optional[str] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    show: bool = True,

    # init config classes
    plot_config: PlotConfig = None,
    layout_config: LayoutConfig = None,
    data_config: DataConfig = None,

    # others
    verbose: bool = False,
    ):
    # convert arguments to lists
    keys = convert_to_list(keys)

    # init config classes
    if plot_config is None:
        plot_config = PlotConfig()
    if layout_config is None:
        layout_config = LayoutConfig()
    if data_config is None:
        data_config = DataConfig()

    # update some values depending on function arguments
    data_config.update_values(
        layer=layer,
        region_tuple=region_tuple, annotations_key=annotations_key, image_key=image_key,
        filter_mode=filter_mode, filter_tuple=filter_tuple
        )
    plot_config.update_values(
        xlim=xlim, ylim=ylim,
        spot_size=spot_size, alpha=alpha
    )
    layout_config.update_values(
        max_cols=max_cols
    )

    # check whether the data is an InSituExperiment or a single InSituData
    if _is_experiment(data):
        n_data = len(data)

        # synchronize colors before plotting
        data.sync_colors(
            keys=keys,
            cells_layer=cells_layer,
            palette=plot_config.palette
        )
    else:
        n_data = 1
        # is_experiment = False

    color_config = ColorConfigMultiPlot(
        data=data,
        cells_layer=cells_layer,
        keys=keys,
        data_config=data_config,
        plot_config=plot_config
    )

    layout_config.calc_subplot_params(
        keys=keys,
        n_data=n_data,
        color_config=color_config
        )

    # setup the subplots
    fig, axs = setup_subplots(
        layout_config=layout_config,
        verbose=verbose
    )

    plot_to_subplots(
        data,
        keys,
        cells_layer,
        fig,
        axs,
        plot_config,
        layout_config,
        data_config,
        color_config
    )

    save_and_show_figure(
        savepath=savepath,
        fig=fig,
        save_only=save_only,
        show=show,
        dpi_save=dpi_save
    )

    gc.collect()

def setup_subplots(
    layout_config: LayoutConfig,
    verbose: bool = False
    ):
    print("Setup subplots.") if verbose else None

    fig, axs = plt.subplots(
        layout_config.n_rows, layout_config.n_cols,
        figsize=layout_config.figsize,
        dpi=layout_config.dpi_display
        )

    if not layout_config.multidata or (layout_config.multidata and not layout_config.multikeys):
        if layout_config.n_plots > 1:
            axs = axs.ravel()
        else:
            axs = np.array([axs])

    if not (layout_config.multidata and layout_config.multikeys):
        remove_empty_subplots(
            axes=axs,
            nplots=layout_config.n_plots,
            nrows=layout_config.n_rows,
            ncols=layout_config.n_cols
            )

    if layout_config.header is not None:
        plt.suptitle(layout_config.header, fontsize=18, x=0.5, y=0.98)

    return fig, axs

def _configure_axis_and_title(
    ax,
    key,
    idx_key,
    sample_name,
    plot_config,
    layout_config
    ):
    # set axis
    ax.set_xlim(plot_config.xlim[0], plot_config.xlim[1])
    ax.set_ylim(plot_config.ylim[0], plot_config.ylim[1])
    ax.set_xlabel('µm', fontsize=plot_config.label_size)
    ax.set_ylabel('µm', fontsize=plot_config.label_size)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_aspect(1)
    ax.set_facecolor(plot_config.background_color)
    ax.tick_params(labelsize=plot_config.tick_label_size)

    if layout_config.multidata and not layout_config.multikeys:
        ax.set_title(
            sample_name + "\n" + key,
            fontsize=plot_config.title_size, #fontweight='bold',
            pad=10,
            rotation=0
            )
    else:
        # set titles
        ax.set_title(
            key,
            fontsize=plot_config.title_size, #fontweight='bold'
            pad=10
            )

        if idx_key == 0:
            ax.annotate(
                sample_name,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label,
                textcoords='offset points',
                size=14,
                rotation=90,
                ha='right', va='center',
                weight='bold'
                )

def _determine_axes(axs, idx, idx_key, layout_config):
    if len(axs.shape) == 2:
        ax = axs[idx, idx_key]
        if idx == (layout_config.n_rows - 1):
            add_legend = True
        else:
            add_legend = False
    elif len(axs.shape) == 1:
        if layout_config.multikeys:
            ax = axs[idx_key]
            add_legend = True
        else:
            ax = axs[idx]

            if len(axs) == 1:
                add_legend = True
            else:
                add_legend = False
    else:
        raise ValueError("`len(self.axs.shape)` has wrong shape {}. Requires 1 or 2.".format(len(axs.shape)))

    return ax, add_legend

def _get_data(
    data,
    idx,
    cells_layer,
    data_config
    ):
    # extract the InSituData object
    try:
        xd = data.data[idx]
        meta = data.metadata.iloc[idx]
    except AttributeError:
        xd = data
        meta = None

    # retrieve the right cell data
    celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

    # extract anndata
    adata = celldata.matrix

    # filter anndata
    if data_config.filter_mode is not None and data_config.filter_tuple is not None:
        adata = filter_anndata(
            adata=adata, filter_mode=data_config.filter_mode, filter_tuple=data_config.filter_tuple
            )

    if data_config.name_column is None or meta is None:
        sample_name = xd.sample_id
    else:
        sample_name = meta[data_config.name_column]

    if data_config.image_key is not None:
        if not xd.images.is_empty:
            imagedata = xd.images
        else:
            imagedata = None
    else:
        imagedata = None

    # get regions
    regions = xd.regions

    # get annotations
    annotations = xd.annotations

    return adata, sample_name, imagedata, regions, annotations

def plot_to_subplots(
    data,
    keys,
    cells_layer,
    fig,
    axs,
    plot_config,
    layout_config,
    data_config,
    color_config,
    verbose: bool = False
):
    print("Do plotting.") if verbose else None

    if _is_experiment(data):
        n_data = len(data)
    else:
        n_data = 1

    #i = 0
    for idx in range(n_data):

        # retrieve data
        ad, sample_name, image_data, regions_data, annotations_data = _get_data(
            data, idx, cells_layer, data_config,
            )

        for idx_key, key in enumerate(keys):
            # get axis to plot
            ax, add_legend = _determine_axes(axs, idx, idx_key, layout_config)

            # plot single spatial plot in given axis
            single_spatial(
                adata=ad,
                key=key, idx_key=idx_key, name=sample_name,
                fig=fig, ax=ax, add_legend=add_legend,
                color_config=color_config, data_config=data_config,
                layout_config=layout_config, plot_config=plot_config,
                regions_data=regions_data, annotations_data=annotations_data, image_data=image_data
            )

    if layout_config.add_legend_to_last_subplot:
        # get axis of last subplots for color legend
        ax = axs[layout_config.n_plots-1]

        k = list(color_config.keys())[0]
        color_config_key = color_config[k]
        # is_categorical = color_config_key["is_categorical"]
        # if is_categorical:
        color_dict = color_config[k]["color_dict"]
        _add_colorlegend_to_axis(
            color_dict=color_dict,
            max_per_col=plot_config.legend_max_per_col,
            ax=ax)

        # else:
        #     #if ConfigData.categorical:
        #     ax.set_axis_off()

def _prepare_limits_and_coordinates(
    adata,
    regions_data,
    region_tuple,
    xlim,
    ylim,
    obsm_key,
    origin_zero
    ):

    # prepare limits using region or lim arguments
    if region_tuple is not None:
        if xlim is not None or ylim is not None:
            raise ValueError("If region_tuple is given, xlim and ylim need to be None.")
        else:
            region_df = regions_data[region_tuple[0]]
            geom = region_df[region_df["name"] == region_tuple[1]]["geometry"].item()
            xlim = [geom.bounds[0], geom.bounds[2]]
            ylim = [geom.bounds[1], geom.bounds[3]]
    else:
        # make sure limits are lists
        xlim = list(xlim) if xlim is not None else xlim
        ylim = list(ylim) if ylim is not None else ylim

    ## Extract coordinates
    # extract x and y pixel coordinates and convert to micrometer
    x_coords = adata.obsm[obsm_key][:, 0].copy()
    y_coords = adata.obsm[obsm_key][:, 1].copy()

    # shift coordinates that they start at (0,0)
    if origin_zero:
        x_offset = x_coords.min()
        y_offset = y_coords.min()
        x_coords -= x_offset
        y_coords -= y_offset
    else:
        x_offset = y_offset = 0

    if xlim is None:
        xmin = x_coords.min()
        xmax = x_coords.max()

        # include margin
        xlim = (xmin, xmax)

    if ylim is None:
        ymin = y_coords.min()
        ymax = y_coords.max()

        # include margin
        ylim = (ymin, ymax)

    return x_coords, y_coords, xlim, ylim

def extract_image_information(
    ImageDataObject,
    image_key,
    xlim,
    ylim,
    pixelwidth_per_subplot,
    histogram_setting
    ):
    # extract image information
    if ImageDataObject is None:
        image = None
        selected_pixel_size = None
        vmin = vmax = None
        pixel_xlim = pixel_ylim = None
    else:
        # pick the image with the right resolution for plotting
        max_pixel_size = np.max([xlim[1] - xlim[0],
                                 ylim[1] - ylim[0]]) / pixelwidth_per_subplot
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
        image = img_pyramid[selected_level]

        ywidth = image.shape[0]
        xwidth = image.shape[1]

        # determine limits for selected pyramid image - clip to maximum image dims (important for extent of image during plotting)
        pixel_xlim = np.clip(
            [int(elem / selected_pixel_size) for elem in xlim],
            a_min=0, a_max=xwidth).tolist()
        pixel_ylim = np.clip(
            [int(elem / selected_pixel_size) for elem in ylim],
            a_min=0, a_max=ywidth).tolist()

        # crop image
        image = image[
            pixel_ylim[0]:pixel_ylim[1],
            pixel_xlim[0]:pixel_xlim[1]
            ]

        if histogram_setting is None:
            vmin = vmax = None
        elif histogram_setting == "auto":
            vmin = da.percentile(image.ravel(), 30).compute().item()
            vmax = da.percentile(image.ravel(), 99.5).compute().item()
        elif isinstance(histogram_setting, tuple):
            vmin = histogram_setting[0]
            vmax = histogram_setting[1]
        else:
            raise ValueError(f"Unknown type for histogram_setting: {type(histogram_setting)}")

    return image, selected_pixel_size, vmin, vmax, pixel_xlim, pixel_ylim

def extract_annotations_information(
    annotations_data,
    annotations_key
    ):
    if annotations_key is not None:
        if isinstance(annotations_key, tuple):
            ankey = annotations_key[0]
            anvalues = annotations_key[1]

            # get annotations dataframe
            annotations_df = annotations_data[ankey]

            if anvalues not in ("all", None):
                # filter them by what is provided in the values of the tuple
                anvalues = convert_to_list(anvalues)
                mask = annotations_df["name"].isin(anvalues)
                annotations_df = annotations_df[mask]
        elif isinstance(annotations_key, str):
            # get annotations dataframe
            annotations_df = annotations_data[annotations_key]
        else:
            raise ValueError(f"Unknown type for annotations_key: {type(annotations_key)}. Must be either a tuple or a string.")

    else:
        annotations_df = None

    return annotations_df

def _calculate_marker_size(
    ax,
    plot_config,
    fig
    ):
    # calculate marker size
    pixels_per_unit = ax.transData.transform(
        [(0, 1), (1, 0)]) - ax.transData.transform((0, 0))
    # x_ppu = pixels_per_unit[1, 0]
    y_ppu = pixels_per_unit[0, 1]
    pxs = y_ppu * plot_config.spot_size
    size = (72. / fig.dpi * pxs)**2

    return size

def single_spatial(
    adata: AnnData,
    key: List[str],
    idx_key: int,
    name: str,

    # figure
    fig: plt.Figure,
    ax: plt.Axes,
    add_legend: bool,

    # configs
    color_config: dict,
    data_config: DataConfig,
    layout_config: LayoutConfig,
    plot_config: PlotConfig,

    # data attributes
    regions_data: Optional[RegionsData] = None,
    annotations_data: Optional[AnnotationsData] = None,
    image_data: Optional[ImageData] = None,
    ):

    # get color values for expression data or categories
    color_values, categorical = _extract_color_values(
        adata=adata, key=key, raw=data_config.raw, layer=data_config.layer
    )

    if color_values is None:
        print("Key '{}' not found.".format(key), flush=True)
        ax.set_axis_off()

    else:
        # retrieve color dictionary
        color_dict = color_config[key]["color_dict"]
        crange = color_config[key]["crange"]
        categorical = color_config[key]["is_categorical"] # True if color_dict is not None

        x_coords, y_coords, plot_config.xlim, plot_config.ylim = _prepare_limits_and_coordinates(
            adata=adata,
            regions_data=regions_data,
            region_tuple=data_config.region_tuple,
            xlim=plot_config.xlim, ylim=plot_config.ylim,
            obsm_key=data_config.obsm_key,
            origin_zero=plot_config.origin_zero
            )

        image, pixel_size, vmin, vmax, pixel_xlim, pixel_ylim = extract_image_information(
            image_data, image_key=data_config.image_key,
            xlim=plot_config.xlim, ylim=plot_config.ylim,
            pixelwidth_per_subplot=plot_config.pixelwidth_per_subplot,
            histogram_setting=plot_config.histogram_setting
        )

        annotations_df = extract_annotations_information(
            annotations_data,
            annotations_key=data_config.annotations_key
        )

        # set the axes (add titles, set limits, etc.)
        _configure_axis_and_title(
            ax=ax,
            key=key, idx_key=idx_key, sample_name=name,
            plot_config=plot_config, layout_config=layout_config
        )

        # calculate the marker size - must be done AFTER the axes are configured!
        size = _calculate_marker_size(
            ax=ax,
            plot_config=plot_config,
            fig=fig)

        # size=5

        if image is not None:
            # plot image data
            extent = (
                pixel_xlim[0] * pixel_size - 0.5,
                pixel_xlim[1] * pixel_size - 0.5,
                pixel_ylim[1] * pixel_size - 0.5,
                pixel_ylim[0] * pixel_size - 0.5
                )

            ax.imshow(
                image,
                extent=extent,
                origin='upper', cmap='gray', vmin=vmin, vmax=vmax)

        # plot transcriptomic data
        if categorical:
            sns.scatterplot(
                x=x_coords, y=y_coords,
                hue=color_values,
                marker=plot_config.spot_type,
                s=size,
                linewidth=0,
                palette=color_dict,
                alpha=plot_config.alpha,
                ax=ax
                )

            # add legend
            # divide axis to fit legend
            divider = make_axes_locatable(ax)
            lax = divider.append_axes("bottom", size="2%", pad=0)

            if add_legend:
                _add_colorlegend_to_axis(
                    color_dict=color_dict,
                    ax=lax,
                    max_per_col=plot_config.legend_max_per_col,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -10)
                    )

            # Remove the axis ticks and labels
            lax.set_xticks([])
            lax.set_yticks([])
            lax.axis('off')

            # Remove the legend from the main axis
            ax.legend().remove()
        else:
            s = ax.scatter(
                x_coords,
                y_coords,
                c=color_values,
                marker=plot_config.spot_type,
                s=size,
                alpha=plot_config.alpha,
                linewidths=0,
                cmap=plot_config.cmap,
                norm=plot_config.normalize
                )

            # divide axis to fit colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.1)

            # add colorbar
            clb = fig.colorbar(s, cax=cax, orientation='vertical')
            # set colorbar
            clb.ax.tick_params(labelsize=plot_config.tick_label_size)

            if plot_config.clb_title is not None:
                clb.ax.set_xlabel(
                    plot_config.clb_title,  # Change to xlabel for horizontal orientation
                    fontdict={"fontsize": plot_config.label_size},
                    labelpad=20
                    )

            # if crange is not None:
            clb.mappable.set_clim(
                crange[0],
                crange[1]
                )

        if annotations_df is not None:
            # convert rgb colors to hex colors
            hex_colors = [
                _rgb2hex_robust(elem, scale_to_one=True, max_value=255)
                for elem
                in annotations_df.color
                ]

            if plot_config.annotations_mode == "outlined":
                # plot the annotations as outlines
                annotations_df.plot(
                    edgecolor=hex_colors,
                    linewidth=4,
                    facecolor="none",
                    ax=ax,
                    aspect=1
                    )
            elif plot_config.annotations_mode == "filled":
                # plot the annotations filled with transparent colors
                annotations_df.plot(
                    color=hex_colors,
                    edgecolor="none",
                    alpha=0.3,
                    ax=ax,
                    aspect=1
                    )

                # plot outlines in black
                annotations_df.plot(
                    facecolor="none",
                    edgecolor="black",
                    linewidth=2,
                    ax=ax,
                    aspect=1
                    )
            else:
                raise ValueError(f"Unknown type for annotations_mode: {type(plot_config.annotations_mode)}. Must be a string that is either 'outlined' or 'filled'.")

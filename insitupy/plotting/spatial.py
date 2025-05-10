#from __future__ import annotations  # this prevents circular imports

import gc
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from insitupy._core._utils import _get_cell_layer
from insitupy._core.insitudata import InSituData
from insitupy._core.insituexperiment import InSituExperiment
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting._colors import (_add_colorlegend_to_axis,
                                       create_cmap_mapping)
from insitupy.plotting._objects import (_ConfigSpatialPlot,
                                        _extract_color_values)
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)


class MultiSpatialPlot:
    '''
    Class to render scatter plots of single-cell spatial transcriptomics data.
    '''
    def __init__(self,
                 data: Union[InSituData, InSituExperiment],
                 keys: Union[str, List[str]],
                 cells_layer: Optional[str] = None,
                 data_ids: Optional[List[int]] = None,
                 raw: bool = False,
                 layer: Optional[str] = None,
                 fig: plt.figure = None,
                 ax: plt.Axes = None,
                 max_cols: int = 4,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 normalize_crange_not_for: List = [],
                 crange: Optional[List[int]] = None,
                 crange_type: Literal['minmax', 'percentile'] = 'minmax',
                 palette: str = 'tab10',
                 cmap_center: Optional[float] = None,
                 dpi_display: int = 80,
                 obsm_key: str = 'spatial',
                 origin_zero: bool = False,
                 spot_size: float = 10,
                 spot_type: str = 'o',
                 cmap: str = 'viridis',
                 background_color: str = 'white',
                 alpha: float = 1,
                 colorbar: bool = True,
                 clb_title: Optional[str] = None,
                 header: Optional[str] = None,
                 name_column: str = None,

                 # font sizes
                 title_size: int = 24,
                 label_size: int = 16,
                 tick_label_size: int = 14,

                 # image stuff
                 image_key: Optional[str] = None,
                 pixelwidth_per_subplot: int = 200,
                 histogram_setting: Optional[Union[Literal["auto"], Tuple[int, int]]] = "auto",

                 # saving
                 savepath: Optional[str] = None,
                 save_only: bool = False,
                 dpi_save: int = 300,
                 show: bool = True,

                 # other
                 verbose: bool = False
                 ):



        # Assign all kwargs to instance variables
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)


    def check_arguments(self):
        print("Check arguments.") if self.verbose else None
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

        try:
            self.n_data = len(self.data)
        except TypeError:
            # if the data is an InSituData, it raises a TypeError
            self.n_data = 1

        if self.n_data > 1:
            self.multidata = True
        elif self.n_data == 1:
            self.multidata = False
        else:
            raise ValueError(f"n_data < 1: {self.n_data}")


    def setup_subplots(self):
        print("Setup subplots.") if self.verbose else None
        self.separate_categorical_legend = False
        if self.multidata:
            if self.multikeys:
                # determine the layout of the subplots
                self.n_rows = self.n_data
                self.max_cols = len(self.keys)
                self.n_plots = self.n_rows * self.max_cols

                # create subplots
                self.fig, self.axs = plt.subplots(self.n_rows, self.max_cols,
                                                  figsize=(8 * self.max_cols, 8 * self.n_rows),
                                                  dpi=self.dpi_display)
                self.fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.
            else:
                self.separate_categorical_legend = True
                # determine the layout of the subplots
                self.n_plots, self.n_rows, self.max_cols = get_nrows_maxcols(n_keys=self.n_data+1, max_cols=self.max_cols)
                self.fig, self.axs = plt.subplots(self.n_rows, self.max_cols,
                                        figsize=(7.6 * self.max_cols, 6 * self.n_rows),
                                        dpi=self.dpi_display)
                self.fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.

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
                figsize=(8 * self.max_cols, 8 * self.n_rows),
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

    def prepare_colorlegends(self):
        print("Prepare color legends.") if self.verbose else None
        self.cmap_dict = {}
        self.maxval_dict = {}
        for key in self.keys:
            value_list = []
            categorical_list = []
            for idx in range(self.n_data):
                # extract the InSituData
                try:
                    xd = self.data.data[idx]
                except AttributeError:
                    xd = self.data
                celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)
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
                self.cmap_dict[key] = create_cmap_mapping(all_values)
            elif not np.any(categorical_list):
                # no values are categorical - collect the maximum values
                self.maxval_dict[key] = np.max(value_list)
            else:
                raise ValueError(f"Values found for key {key} showed mixed type (categorical/numeric).")

    def plot_to_subplots(self):
        print("Do plotting.") if self.verbose else None
        #i = 0
        for idx in range(self.n_data):
            # extract the InSituData
            try:
                xd = self.data.data[idx]
                meta = self.data.metadata.iloc[idx]
            except AttributeError:
                xd = self.data
                meta = None
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)
            ad = celldata.matrix
            annot_df = xd.annotations

            if self.name_column is None or meta is None:
                name = xd.sample_id
            else:
                name = meta[self.name_column]

            if self.image_key is not None:
                imagedata = xd.images
            else:
                imagedata = None

            for idx_key, key in enumerate(self.keys):
                # get axis to plot
                if self.ax is None:
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
                            add_legend = False
                            # if idx == (self.n_data - 1):
                            #     separate_categorical_legend = True
                    else:
                        raise ValueError("`len(self.axs.shape)` has wrong shape {}. Requires 1 or 2.".format(len(self.axs.shape)))
                else:
                    ax = self.ax

                # get data
                ConfigData = _ConfigSpatialPlot(
                    adata=ad,
                    key=key,
                    ImageDataObject=imagedata,
                    image_key=self.image_key,
                    pixelwidth_per_subplot=self.pixelwidth_per_subplot,
                    raw=self.raw,
                    layer=self.layer,
                    obsm_key=self.obsm_key,
                    origin_zero=self.origin_zero,
                    xlim=self.xlim,
                    ylim=self.ylim,
                    spot_size=self.spot_size,
                    histogram_setting=self.histogram_setting
                )

                if ConfigData.color_values is not None:
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
                        ax.set_title(name + "\n" + ConfigData.key,
                                     fontsize=self.title_size, #fontweight='bold',
                                     pad=10,
                                     rotation=0)
                    else:
                        # set titles
                        ax.set_title(ConfigData.key,
                                     fontsize=self.title_size, #fontweight='bold'
                                     pad=10,
                                     )

                        if idx_key == 0:
                            ax.annotate(name,
                                        xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                                        xycoords=ax.yaxis.label, textcoords='offset points',
                                        size=14, rotation=90,
                                        ha='right', va='center', weight='bold')

                    if ConfigData.categorical:
                        #color_dict = create_cmap_mapping(data=ad.obs[key])
                        color_dict = self.cmap_dict[key]
                        crange = None
                    else:
                        #color_dict = self.palette
                        color_dict = None
                        crange = [0, self.maxval_dict[key]]

                    # plot single spatial plot in given axis
                    self.single_spatial(
                        ConfigData=ConfigData,
                        axis=ax,
                        color_dict=color_dict,
                        crange=crange,
                        add_legend=add_legend,
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

        if self.separate_categorical_legend:
            # get axis of last subplots for color legend
            ax = self.axs[self.n_plots-1]
            if len(self.cmap_dict) == 1:
                k = list(self.cmap_dict.keys())[0]
                color_dict = self.cmap_dict[k]

                _add_colorlegend_to_axis(color_dict=color_dict, ax=ax)

            else:
                ax.set_axis_off()


    def single_spatial(
        self,
        ConfigData: Type[_ConfigSpatialPlot],
        axis: plt.Axes,
        color_dict: Dict,
        crange: Optional[Tuple[float, float]],
        add_legend: bool,
        ):

        # calculate marker size
        pixels_per_unit = axis.transData.transform(
            [(0, 1), (1, 0)]) - axis.transData.transform((0, 0))
        # x_ppu = pixels_per_unit[1, 0]
        y_ppu = pixels_per_unit[0, 1]
        pxs = y_ppu * ConfigData.spot_size
        size = (72. / self.fig.dpi * pxs)**2

        if ConfigData.image is not None:
            # plot image data
            extent = (
                ConfigData.pixel_xlim[0] * ConfigData.pixel_size - 0.5,
                ConfigData.pixel_xlim[1] * ConfigData.pixel_size - 0.5,
                ConfigData.pixel_ylim[1] * ConfigData.pixel_size - 0.5,
                ConfigData.pixel_ylim[0] * ConfigData.pixel_size - 0.5
                )

            axis.imshow(
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
                palette=color_dict,
                alpha=self.alpha,
                ax=axis
                )
            # add legend
            # divide axis to fit legend
            divider = make_axes_locatable(axis)
            lax = divider.append_axes("bottom", size="2%", pad=0)

            if add_legend:
                _add_colorlegend_to_axis(
                    color_dict=color_dict, ax=lax, max_per_row=6,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -5)
                    )

            # Remove the axis ticks and labels
            lax.set_xticks([])
            lax.set_yticks([])
            lax.axis('off')

            # Remove the legend from the main axis
            axis.legend().remove()
        else:
            s = axis.scatter(
                ConfigData.x_coords, ConfigData.y_coords,
                c=ConfigData.color_values,
                marker=self.spot_type,
                #s=ConfigData.spot_size,
                s=size,
                alpha=self.alpha,
                linewidths=0,
                cmap=self.cmap,
                norm=self.normalize
                )

            # divide axis to fit colorbar
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="4%", pad=0.1)

            # add colorbar
            clb = self.fig.colorbar(s, cax=cax, orientation='vertical')
            # set colorbar
            clb.ax.tick_params(labelsize=self.tick_label_size)

            if self.clb_title is not None:
                clb.ax.set_xlabel(self.clb_title,  # Change to xlabel for horizontal orientation
                                fontdict={"fontsize": self.label_size},
                                labelpad=20)

            if crange is not None:
                clb.mappable.set_clim(crange[0], crange[1])
            else:
                if self.crange_type == 'percentile':
                    clb.mappable.set_clim(0, np.percentile(ConfigData.color_values, 99))

def plot_spatial(
    data: Union[InSituData, InSituExperiment],
    keys: Union[str, List[str]],
    cells_layer: Optional[str] = None,
    data_ids: Optional[List[int]] = None,
    raw: bool = False,
    layer: Optional[str] = None,
    fig: plt.figure = None,
    ax: plt.Axes = None,
    max_cols: int = 4,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    normalize_crange_not_for: List = [],
    crange: Optional[List[int]] = None,
    crange_type: Literal['minmax', 'percentile'] = 'minmax',
    palette: str = 'tab10',
    cmap_center: Optional[float] = None,
    dpi_display: int = 80,
    obsm_key: str = 'spatial',
    origin_zero: bool = False,
    spot_size: float = 10,
    spot_type: str = 'o',
    cmap: str = 'viridis',
    background_color: str = 'white',
    alpha: float = 1,
    colorbar: bool = True,
    clb_title: Optional[str] = None,
    header: Optional[str] = None,
    name_column: str = None,

    # font sizes
    title_size: int = 24,
    label_size: int = 16,
    tick_label_size: int = 14,

    # image stuff
    image_key: Optional[str] = None,
    pixelwidth_per_subplot: int = 200,
    histogram_setting: Optional[Union[Literal["auto"], Tuple[int, int]]] = "auto",

    # saving
    savepath: Optional[str] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    show: bool = True,

    # other
    verbose: bool = False,
    ):

    msp = MultiSpatialPlot(**locals())

    # check arguments
    msp.check_arguments()

    # prepare color legends
    msp.prepare_colorlegends()

    # plotting
    if msp.ax is None:
        msp.setup_subplots()
    else:
        assert msp.fig is not None, "If axis for plotting is given, also a figure object needs to be provided via `fig`"
        assert len(msp.keys) == 1, "If single axis is given not more than one key is allowed."

    msp.plot_to_subplots()

    save_and_show_figure(
        savepath=msp.savepath,
        fig=msp.fig,
        save_only=msp.save_only,
        show=msp.show,
        dpi_save=msp.dpi_save
        )

    gc.collect()


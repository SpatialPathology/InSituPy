import gc
import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from insitupy._core._checks import _is_experiment
from insitupy._io.plots import save_and_show_figure
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.utils._adata import filter_anndata
from insitupy.utils._colors import (_add_colorlegend_to_axis,
                                    _extract_color_values, _rgb2hex_robust)
from insitupy.utils.utils import convert_to_list, remove_empty_subplots


# -------------------------------
# CONFIG OBJECTS
# -------------------------------
@dataclass
class PlotAppearanceConfig:
    spot_size: float = 10
    spot_type: str = "o"
    alpha: float = 1.0
    cmap: str = "viridis"
    palette: str = "tab20"
    background_color: str = "white"
    cmap_center: Optional[float] = None
    legend_max_per_col: int = 10
    clb_title: Optional[str] = None


@dataclass
class LayoutConfig:
    max_cols: int = 4
    dpi_display: int = 80
    title_size: int = 18
    label_size: int = 16
    tick_label_size: int = 14
    header: Optional[str] = None


@dataclass
class DataFilterConfig:
    raw: bool = False
    layer: Optional[str] = None
    filter_mode: Optional[str] = None
    filter_tuple: Optional[Tuple] = None


@dataclass
class ImageConfig:
    image_key: Optional[str] = None
    pixelwidth_per_subplot: int = 200
    histogram_setting: Union[Literal["auto"], Tuple[int, int], None] = "auto"

class _ColorConfigMultiPlot:
    """
    Ensures consistent color mapping across multiple datasets and keys.
    Stores for each key:
        - color_dict (for categorical values)
        - crange (min/max for continuous values)
        - is_categorical (bool)
    """
    def __init__(
        self,
        data,
        cells_layer: Optional[str],
        keys: List[str],
        raw: bool = False,
        layer: Optional[str] = None,
        palette: str = "tab20"
    ):
        self._dict = {}
        self.cells_layer = cells_layer
        self.raw = raw
        self.layer = layer
        self.palette = palette

        # experiment vs single dataset
        if _is_experiment(data):
            data_list = data.data
            exp_color_dict = data.colors
        else:
            data_list = [data]
            exp_color_dict = {}

        for key in keys:
            if key in exp_color_dict:
                # use experiment-wide predefined colors
                self._dict[key] = {
                    "color_dict": exp_color_dict[key],
                    "crange": None,
                    "is_categorical": True,
                }
            else:
                # infer colors from data
                self._dict[key] = self._infer_color_entry(data_list, key)

    def __getitem__(self, key):
        return self._dict[key]

    def keys(self):
        return self._dict.keys()

    def _infer_color_entry(self, data_list, key):
        """Infer color settings across multiple datasets."""
        value_list = []
        categorical_list = []

        for xd in data_list:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)
            ad = celldata.matrix
            color_values, is_categorical = _extract_color_values(
                adata=ad, key=key, raw=self.raw, layer=self.layer
            )
            if color_values is None:
                continue
            if is_categorical:
                value_list.append(np.unique(color_values))
            else:
                value_list.append(np.max(color_values))
            categorical_list.append(is_categorical)

        if np.all(categorical_list):
            # all categorical → build shared color_dict
            all_values = np.unique(np.concatenate(value_list))
            color_dict = {v: c for v, c in zip(all_values, sns.color_palette(self.palette, len(all_values)))}
            return {"color_dict": color_dict, "crange": None, "is_categorical": True}

        elif not np.any(categorical_list):
            # all continuous → record global max
            max_value = np.max(value_list)
            return {"color_dict": None, "crange": [0, max_value], "is_categorical": False}

        else:
            raise ValueError(f"Mixed categorical and continuous values for key {key}.")


class _SinglePlotConfig:
    """
    Collects all per-subplot data: coordinates, limits, images, annotations, colors.
    Keeps preparation logic separate from plotting.
    """
    def __init__(
        self,
        adata,
        key: str,
        ax: plt.Axes,
        sample_name: str,
        idx_key: int,
        color_config: _ColorConfigMultiPlot,
        regions=None,
        region_tuple: Optional[Tuple[str, str]] = None,
        annotations=None,
        annotations_key: Optional[Union[str, Tuple[str, Union[str, List[str]]]]] = None,
        annotations_mode: Literal["outlined", "filled"] = "outlined",
        imagedata=None,
        image_key: Optional[str] = None,
        pixelwidth_per_subplot: int = 200,
        histogram_setting: Union[Literal["auto"], Tuple[int, int], None] = "auto",
        raw: bool = False,
        layer: Optional[str] = None,
        obsm_key: str = "spatial",
        origin_zero: bool = False,
    ):
        self.key = key
        self.ax = ax
        self.sample_name = sample_name
        self.idx_key = idx_key
        self.annotations_mode = annotations_mode

        # -------------------------------
        # Coordinates
        # -------------------------------
        self.x_coords = adata.obsm[obsm_key][:, 0].copy()
        self.y_coords = adata.obsm[obsm_key][:, 1].copy()

        if origin_zero:
            self.x_coords -= self.x_coords.min()
            self.y_coords -= self.y_coords.min()

        # -------------------------------
        # Limits (region-based or full extent)
        # -------------------------------
        if region_tuple is not None:
            region_df = regions[region_tuple[0]]
            geom = region_df[region_df["name"] == region_tuple[1]]["geometry"].item()
            self.xlim = [geom.bounds[0], geom.bounds[2]]
            self.ylim = [geom.bounds[1], geom.bounds[3]]
        else:
            self.xlim = (self.x_coords.min(), self.x_coords.max())
            self.ylim = (self.y_coords.min(), self.y_coords.max())

        # -------------------------------
        # Image pyramid selection
        # -------------------------------
        self.image, self.pixel_size, self.vmin, self.vmax = None, None, None, None
        if imagedata is not None and image_key is not None:
            img_pyramid = imagedata[image_key]
            orig_pixel_size = imagedata.metadata[image_key]["pixel_size"]
            pixel_sizes_levels = np.array([orig_pixel_size * (2**i) for i in range(len(img_pyramid))])

            max_pixel_size = max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]) / pixelwidth_per_subplot
            try:
                selected_level = np.where(pixel_sizes_levels <= max_pixel_size)[0][-1]
            except IndexError:
                selected_level = 0
            self.pixel_size = pixel_sizes_levels[selected_level].item()

            # crop image
            self.image = img_pyramid[selected_level]
            ywidth, xwidth = self.image.shape[:2]
            pixel_xlim = np.clip([int(v / self.pixel_size) for v in self.xlim], 0, xwidth).tolist()
            pixel_ylim = np.clip([int(v / self.pixel_size) for v in self.ylim], 0, ywidth).tolist()
            self.image = self.image[pixel_ylim[0]:pixel_ylim[1], pixel_xlim[0]:pixel_xlim[1]]

            # histogram-based scaling
            if histogram_setting == "auto":
                self.vmin = np.percentile(self.image.ravel(), 30)
                self.vmax = np.percentile(self.image.ravel(), 99.5)
            elif isinstance(histogram_setting, tuple):
                self.vmin, self.vmax = histogram_setting

        # -------------------------------
        # Annotations
        # -------------------------------
        self.annotations_df = None
        if annotations_key is not None and annotations is not None:
            if isinstance(annotations_key, tuple):
                key, values = annotations_key
                df = annotations[key]
                if values not in ("all", None):
                    values = convert_to_list(values)
                    df = df[df["name"].isin(values)]
                self.annotations_df = df
            elif isinstance(annotations_key, str):
                self.annotations_df = annotations[annotations_key]

        # -------------------------------
        # Color values
        # -------------------------------
        self.color_dict = color_config[key]["color_dict"]
        self.crange = color_config[key]["crange"]
        self.categorical = color_config[key]["is_categorical"]

        # color values
        self.color_values, _ = _extract_color_values(
            adata=adata, key=key, raw=raw, layer=layer
        )
        # self.color_values, self.categorical = _extract_color_values(
        #     adata=adata, key=key, raw=raw, layer=layer
        # )



# -------------------------------
# MULTI PLOT CLASS
# -------------------------------
class MultiSpatialPlot:
    def __init__(self, data, keys, cells_layer,
                 appearance: PlotAppearanceConfig,
                 layout: LayoutConfig,
                 filters: DataFilterConfig,
                 images: ImageConfig,
                 savepath=None, show=True, save_only=False, dpi_save=300, verbose=False):

        self.data = data
        self.keys = convert_to_list(keys)
        self.cells_layer = cells_layer
        self.appearance = appearance
        self.layout = layout
        self.filters = filters
        self.images = images
        self.savepath = savepath
        self.show = show
        self.save_only = save_only
        self.dpi_save = dpi_save
        self.verbose = verbose

        # detect experiment vs. single dataset
        self.is_experiment = _is_experiment(data)
        self.n_data = len(data) if self.is_experiment else 1

        # normalization for continuous colormaps
        self.normalize = None
        if self.appearance.cmap_center is not None:
            self.normalize = colors.CenteredNorm(vcenter=self.appearance.cmap_center)

        self.color_config = _ColorConfigMultiPlot(
            data=self.data,
            cells_layer=self.cells_layer,
            keys=self.keys,
            raw=self.filters.raw,
            layer=self.filters.layer,
            palette=self.appearance.palette
        )

    # -------------------------------
    # SUBPLOT LAYOUT
    # -------------------------------
    def setup_subplots(self):
        n_keys = len(self.keys)
        self.n_plots = self.n_data * n_keys
        self.n_rows = math.ceil(self.n_plots / self.layout.max_cols)

        self.fig, axs = plt.subplots(
            self.n_rows, self.layout.max_cols,
            figsize=(6 * self.layout.max_cols, 6 * self.n_rows),
            dpi=self.layout.dpi_display,
        )
        self.axs = axs.ravel()
        remove_empty_subplots(self.axs, self.n_plots, self.n_rows, self.layout.max_cols)

        if self.layout.header:
            self.fig.suptitle(self.layout.header, fontsize=self.layout.title_size)

    def _set_axis(self, config):
        ax = config.ax

        # axis limits
        ax.set_xlim(config.xlim[0], config.xlim[1])
        ax.set_ylim(config.ylim[0], config.ylim[1])

        # labels
        ax.set_xlabel("µm", fontsize=self.layout.label_size)
        ax.set_ylabel("µm", fontsize=self.layout.label_size)

        # title
        ax.set_title(
            config.key,
            fontsize=self.layout.title_size,
            pad=10
        )

        # background and ticks
        ax.set_facecolor(self.appearance.background_color)
        ax.tick_params(labelsize=self.layout.tick_label_size)
        ax.invert_yaxis()
        ax.set_aspect(1)

        # add sample name to the first subplot in each row
        if config.idx_key == 0:
            ax.annotate(
                config.sample_name,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size=12,
                rotation=90,
                ha="right", va="center",
                weight="bold"
            )


    # -------------------------------
    # DATA EXTRACTION
    # -------------------------------
    def _get_data(self, idx):
        try:
            xd = self.data.data[idx]
            meta = self.data.metadata.iloc[idx]
        except AttributeError:
            xd = self.data
            meta = None

        celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)
        adata = celldata.matrix

        # filter
        if self.filters.filter_mode and self.filters.filter_tuple:
            adata = filter_anndata(
                adata=adata,
                filter_mode=self.filters.filter_mode,
                filter_tuple=self.filters.filter_tuple,
            )

        sample_name = getattr(xd, "sample_id", f"sample_{idx}")
        imagedata = xd.images if self.images.image_key and not xd.images.is_empty else None
        return adata, sample_name, imagedata, xd.regions, xd.annotations

    def _calculate_marker_size(self, ax):
        """Scale marker size relative to axis units and figure DPI."""
        pixels_per_unit = ax.transData.transform([(0, 1), (1, 0)]) - ax.transData.transform((0, 0))
        y_ppu = pixels_per_unit[0, 1]
        pxs = y_ppu * self.appearance.spot_size
        size = (72.0 / self.fig.dpi * pxs) ** 2
        return size

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    def plot_to_subplots(self):
        for idx in range(self.n_data):
            adata, sample_name, imagedata, regions, annotations = self._get_data(idx)

            for idx_key, key in enumerate(self.keys):
                ax = self.axs[idx * len(self.keys) + idx_key]
                
                config = _SinglePlotConfig(
                    adata=adata,
                    key=key,
                    ax=ax,
                    sample_name=sample_name,
                    idx_key=idx_key,
                    color_config=self.color_config,
                    regions=regions,
                    region_tuple=None,   # could be passed from plot_spatial args
                    annotations=annotations,
                    annotations_key=None,  # could be passed from args
                    annotations_mode="outlined",
                    imagedata=imagedata,
                    image_key=self.images.image_key,
                    pixelwidth_per_subplot=self.images.pixelwidth_per_subplot,
                    histogram_setting=self.images.histogram_setting,
                    raw=self.filters.raw,
                    layer=self.filters.layer,
                    obsm_key="spatial",
                    origin_zero=False,
                )

                if config.color_values is None:
                    ax.set_axis_off()
                    continue

                self._set_axis(config)

                # plot background image if present
                if config.image is not None:
                    extent = (
                        config.xlim[0], config.xlim[1],
                        config.ylim[1], config.ylim[0]
                    )
                    ax.imshow(config.image, extent=extent, origin="upper",
                            cmap="gray", vmin=config.vmin, vmax=config.vmax)

                # plot categorical or continuous
                if config.categorical:
                    self._plot_points_categorical(ax, config)
                else:
                    self._plot_points_continuous(ax, config)

                # add annotations
                if config.annotations_df is not None:
                    self._plot_annotations(ax, config)

        gc.collect()

    # -------------------------------
    # PLOTTING HELPERS
    # -------------------------------
    def _plot_points_categorical(self, ax, config: _SinglePlotConfig):
        marker_size = self._calculate_marker_size(ax)
        sns.scatterplot(
            x=config.x_coords, y=config.y_coords,
            hue=config.color_values,
            marker=self.appearance.spot_type,
            s=marker_size,
            linewidth=0,
            palette=self.appearance.palette,
            alpha=self.appearance.alpha,
            ax=ax,
        )
        ax.legend().remove()

        # add custom color legend
        divider = make_axes_locatable(ax)
        lax = divider.append_axes("bottom", size="3%", pad=0.05)
        _add_colorlegend_to_axis(
            color_dict=dict(zip(config.color_values.cat.categories, sns.color_palette(self.appearance.palette))),
            ax=lax,
            max_per_col=self.appearance.legend_max_per_col,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1)
        )
        lax.set_axis_off()

    def _plot_points_continuous(self, ax, config: _SinglePlotConfig):
        marker_size = self._calculate_marker_size(ax)
        sc = ax.scatter(
            config.x_coords, config.y_coords,
            c=config.color_values,
            marker=self.appearance.spot_type,
            s=marker_size,
            alpha=self.appearance.alpha,
            linewidths=0,
            cmap=self.appearance.cmap,
            norm=self.normalize,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        clb = self.fig.colorbar(sc, cax=cax)
        clb.ax.tick_params(labelsize=self.layout.tick_label_size)
        if self.appearance.clb_title:
            clb.set_label(self.appearance.clb_title, fontsize=self.layout.label_size)

    def _plot_annotations(self, ax, config: _SinglePlotConfig):
        hex_colors = [_rgb2hex_robust(c, scale_to_one=True, max_value=255)
                    for c in config.annotations_df.color]
        if config.annotations_mode == "outlined":
            config.annotations_df.plot(edgecolor=hex_colors, linewidth=2,
                                    facecolor="none", ax=ax, aspect=1)
        elif config.annotations_mode == "filled":
            config.annotations_df.plot(color=hex_colors, alpha=0.3,
                                    edgecolor="none", ax=ax, aspect=1)
            config.annotations_df.plot(facecolor="none", edgecolor="black",
                                    linewidth=1, ax=ax, aspect=1)


    # -------------------------------
    # SAVE
    # -------------------------------
    def save(self):
        save_and_show_figure(
            savepath=self.savepath,
            fig=self.fig,
            save_only=self.save_only,
            show=self.show,
            dpi_save=self.dpi_save,
        )

def plot_spatial_new(
    data,
    keys,
    cells_layer: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    save_only: bool = False,
    dpi_save: int = 300,
    verbose: bool = False,
    appearance: PlotAppearanceConfig = PlotAppearanceConfig(),
    layout: LayoutConfig = LayoutConfig(),
    filters: DataFilterConfig = DataFilterConfig(),
    images: ImageConfig = ImageConfig(),
):
    plotter = MultiSpatialPlot(
        data=data,
        keys=keys,
        cells_layer=cells_layer,
        appearance=appearance,
        layout=layout,
        filters=filters,
        images=images,
        savepath=savepath,
        show=show,
        save_only=save_only,
        dpi_save=dpi_save,
        verbose=verbose,
    )
    plotter.setup_subplots()
    plotter.plot_to_subplots()
    plotter.save()

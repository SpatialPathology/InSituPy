
import copy
import gc
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colors
from matplotlib.collections import PathCollection
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.axes_grid1 import make_axes_locatable

from insitupy._constants import (
    DEFAULT_CATEGORICAL_CMAP,
    DEFAULT_CONTINUOUS_CMAP,
    NA_CATEGORY,
    with_insitupy_style,
)
from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy._mixins import _UpdatablePlottingConfig
from insitupy.containers import AnnotationsData, ImageData, RegionsData
from insitupy.containers._utils import _get_table_layer, _has_table_layer
from insitupy.experiment.data import InSituExperiment
from insitupy.plotting.save import save_and_show_figure
from insitupy.utils._adata import filter_anndata
from insitupy.utils._colors import (
    _add_colorlegend_to_axis,
    _coerce_na_for_plot,
    _extract_color_values,
    _fallback_category_colors,
    _merge_stored_color_dicts,
    _parse_unique_categories,
    _rgb2hex_robust,
    _stored_color_dict,
    _warn_na_cells_hidden,
    create_cmap_mapping,
)
from insitupy.utils.utils import (
    convert_to_list,
    get_nrows_maxcols,
    remove_empty_subplots,
)

logger = logging.getLogger(__name__)

FilterMode = Literal[
    "contains", "not contains", "starts with", "ends with",
    "is equal", "is not", "in", "not in",
    "greater than", "less than", "greater or equal", "less or equal"
    ]

# -------------------------------
# CONFIG OBJECTS
# -------------------------------

@dataclass
class DataConfig(_UpdatablePlottingConfig):
    """Configuration class for data extraction and preprocessing.

    Attributes:
        layer (Optional[str]): The data layer to extract from, if applicable.
        raw (bool): Whether to use raw data (default: False).
        obsm_key (str): The key in `.obsm` containing spatial coordinates (default: 'spatial').
        name_column (Optional[str]): Column name for sample/observation names.
        region_tuple (Optional[Tuple[str, str]]): Tuple specifying a region or subset of the data.
        annotations_key (Optional[Tuple[str, Optional[Union[str, List[str]]]]]): Key(s) for annotations to extract.
        image_key (Optional[str]): Key for accessing associated images.
        filter_mode (Optional[str]): Filtering mode, if any (e.g., 'threshold', 'percentile').
        filter_tuple (Optional[Tuple]): Parameters used for filtering data.

    Methods:
        update_values(**kwargs): Update one or more configuration attributes dynamically.
        show_all(): Display all current configuration values.
    """

    # data extraction config
    layer: str | None = None
    raw: bool = False
    obsm_key: str = 'spatial'
    name_column: str | None = None

    # data attribute keys
    region_tuple: tuple[str, str] | None = None
    annotations_key: tuple[str, str | list[str] | None] | None = None
    image_key: str | None = None

    # filters
    filter_mode: FilterMode | None = None
    filter_tuple: tuple | None = None

@dataclass
class PlotConfig(_UpdatablePlottingConfig):
    """Configuration class for plot appearance and rendering.

    Attributes:
        xlim (Optional[Tuple[float, float]]): X-axis limits.
        ylim (Optional[Tuple[float, float]]): Y-axis limits.
        spot_size (float): Marker extent in um - diameter of circles, edge of squares
            (default: 10).
        alpha (float): Transparency level for plotted elements (default: 1.0).
        cmap (str): Colormap for continuous variables.
        palette (Optional[ListedColormap]): Color palette for categorical variables. ``None``
            (default) uses complete colors stored in ``uns['{key}_colors']``, else the default
            categorical palette. An explicit palette always wins; for an ``InSituExperiment``
            it also wins over ``exp.colors`` and is used for that figure only (it is not
            saved - use ``exp.sync_colors(key, palette=...)`` for that).
        spot_type (str): Marker type for scatter plots (default: "o").
        background_color (str): Background color of the plot (default: "white").
        cmap_center (Optional[float]): Center value for diverging colormaps.
        normalize (Optional[colors.Normalize]): Normalization strategy for color scaling.
        legend_max_per_col (int): Maximum number of legend items per column (default: 10).
        clb_title (Optional[str]): Title for the colorbar.
        annotations_mode (Literal["outlined", "filled"]): How annotations are displayed (default: "outlined").
        crange (Optional[List[int]]): Value range for color scaling.
        crange_type (Literal["minmax", "max", "upper_percentile", "percentile"]): Method for computing color ranges (default: "upper_percentile").
        origin_zero (bool): Whether to enforce origin at zero (default: False).
        label_size (int): Font size for labels (default: 16).
        title_size (int): Font size for titles (default: 18).
        tick_label_size (int): Font size for tick labels (default: 14).
        pixelwidth_per_subplot (int): Pixel width per subplot (default: 200).
        histogram_setting (Union[Literal["auto"], Tuple[int, int], None]): Histogram bin settings (default: "auto").

    Methods:
        update_values(**kwargs): Update one or more configuration attributes dynamically.
        show_all(): Display all current configuration values.
    """

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    spot_size: float = 10
    alpha: float = 1.0
    cmap: str = DEFAULT_CONTINUOUS_CMAP
    palette: ListedColormap | None = None
    spot_type: str = "o"
    background_color: str = "white"
    nan_color: str | None = None
    cmap_center: float | None = None
    normalize: colors.Normalize | None = None
    show_legend: bool = True
    legend_max_per_col: int = 10
    clb_title: str | None = None
    annotations_mode: Literal["outlined", "filled"] = "outlined"
    crange: list[int] | None = None
    crange_type: Literal['minmax', 'max', 'upper_percentile', 'percentile'] = 'upper_percentile'
    origin_zero: bool = False
    label_size: int = 16
    show_title: bool = True
    title_size: int = 18
    show_scale: bool = True
    tick_label_size: int = 14
    pixelwidth_per_subplot: int = 200
    histogram_setting: Literal["auto"] | tuple[int, int] | None = "auto"

    def __post_init__(self):
        # check if cmap is supposed to be centered
        if self.cmap_center is not None:
            self.normalize = colors.CenteredNorm(vcenter=self.cmap_center)

@dataclass
class LayoutConfig(_UpdatablePlottingConfig):
    """Configuration class for subplot layout and figure arrangement.

    Attributes:
        max_cols (Optional[int]): Maximum number of columns in the subplot grid (default: 4).
        header (Optional[str]): Figure header or title.
        multikeys (bool): Whether multiple keys are plotted simultaneously (default: False).
        multidata (bool): Whether multiple datasets are plotted simultaneously (default: False).
        n_rows (int): Number of subplot rows.
        n_cols (int): Number of subplot columns.
        n_plots (int): Total number of subplots.
        subplot_width (int): Width of each subplot in inches (default: 6).
        subplot_height (int): Height of each subplot in inches (default: 6).
        figsize (Optional[Tuple]): Overall figure size, computed if not provided.
        add_legend_to_last_subplot (bool): Whether to add a legend to the last subplot (default: False).
        dpi_display (int): Display resolution in DPI (default: 80).

    Methods:
        update_values(**kwargs): Update one or more configuration attributes dynamically.
        show_all(): Display all current configuration values.
        calc_subplot_params(keys, n_data, color_config): Compute subplot grid parameters
            (rows, columns, number of plots) based on the given keys and datasets.
    """

    max_cols: int | None = 4
    header: str | None = None
    multikeys: bool = False
    multidata: bool = False
    n_rows: int = None
    n_cols: int = None
    n_plots: int = None
    subplot_width: int = 6
    subplot_height: int = 6
    wspace: float | None = 0.4
    hspace: float | None = 0.2
    figsize: tuple | None = None
    add_legend_to_last_subplot: bool = False
    dpi_display: int = 80
    def calc_subplot_params(self, keys, n_data, color_config):
        """Compute subplot grid dimensions and layout flags from the number of keys and data objects.

        Sets ``self.n_rows``, ``self.n_cols``, ``self.n_plots``,
        ``self.multikeys``, ``self.multidata``, and
        ``self.add_legend_to_last_subplot`` in-place based on the combination
        of *keys* and *n_data*.

        Args:
            keys: Sequence of colour/feature keys to plot.
            n_data: Number of data objects (samples / experiments) to show.
            color_config: Mapping from key to colour configuration dict; used
                to detect categorical vs. continuous data when laying out
                categorical legends.
        """
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

class _Unset:
    """Sentinel marking a ``spatial()`` argument the caller did not pass.

    Lets ``_explicit_overrides`` tell "argument omitted" apart from "argument
    passed a value that happens to equal a config default", so an explicitly
    passed top-level kwarg always overrides a caller-provided config.
    """
    __slots__ = ()

    def __repr__(self):
        return "UNSET"

_UNSET = _Unset()

def _explicit_overrides(**kwargs) -> dict:
    """Keep only kwargs the caller explicitly passed at the top level.

    A value equal to the ``_UNSET`` sentinel means the corresponding ``spatial()``
    argument was omitted, so it is dropped and the caller-provided
    plot_config/layout_config/data_config keeps its own value for that field.
    Any other value - including one equal to a config default, or a numpy array -
    is forwarded to ``update_values()``.
    """
    return {name: value for name, value in kwargs.items() if value is not _UNSET}

@with_insitupy_style
def spatial(
    data: InSituData | InSituExperiment,
    keys: str | list[str],
    cells_layer: str | None = None,
    units_layer: str | None = None,
    layer: str | None = _UNSET,

    # data attribute keys
    region_tuple: tuple[str, str] | None = _UNSET,
    annotation_tuple: tuple[str, str | list[str] | None] | None = _UNSET,
    annotations_key: tuple[str, str | list[str] | None] | None = None,
    image_key: str | None = _UNSET,

    # filters
    filter_mode: FilterMode | None = _UNSET,
    filter_tuple: tuple | None = _UNSET,

    # plotting configs
    xlim: tuple[float, float] | None = _UNSET,
    ylim: tuple[float, float] | None = _UNSET,
    spot_size: float = _UNSET,
    alpha: float = _UNSET,
    nan_color: str | None = _UNSET,

    # layout configs
    max_cols: int | None = _UNSET,
    figsize: tuple[float, float] | None = _UNSET,
    subplot_width: float | None = _UNSET,
    subplot_height: float | None = _UNSET,

    # save configs
    savepath: str | None = None,
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
    """Plot spatial omics data with optional images, annotations, and regions.

    This function generates spatial scatter plots of spatially-resolved omics data from one or multiple
    :class:`~insitupy._core.data.InSituData` or :class:`~insitupy.experiment.data.InSituExperiment` objects.
    It supports categorical and continuous features, overlays images and annotations, and provides flexible configuration
    for plotting, layout, and saving.

    Args:
        data (InSituData or InSituExperiment): Input dataset or experiment.
        keys (str or list of str): Feature key(s) to plot (e.g., gene names or annotations).
        cells_layer (str, optional): Name of the cell layer to extract data from.
        units_layer (str, optional): Name of a units layer (e.g. Visium HD bins such as
            ``"square_008um"``, or Visium spots) to plot instead of a cell layer. Mutually exclusive
            with ``cells_layer``. Unless ``spot_size``/``plot_config`` are given, bins are drawn as
            squares with the bin edge length as marker size (tiling the tissue) and spots as circles
            with the spot diameter. Plotting 2 um layers of a full slide is slow: use
            ``xlim``/``ylim`` or crop first. Colors of an ``InSituExperiment`` (``exp.colors``) are
            not used for units layers.
        layer (str, optional): AnnData layer to extract values from.
        region_tuple (tuple of (str, str), optional): Region identifier (dataset key, region name).
        annotation_tuple (tuple of (str, str or list of str), optional):
            Annotation overlay specifier as ``(key, name)`` where ``key`` is the
            annotation category and ``name`` is the specific annotation class (or
            a list of classes) to overlay. Pass just the key as a plain string to
            overlay all classes in that category.
        annotations_key (tuple or str, optional): Deprecated. Use ``annotation_tuple`` instead.
        image_key (str, optional): Key for associated images to overlay. In an experiment, samples
            without this image are plotted without one (with a warning); a single sample without
            it raises a ``ValueError``.
        filter_mode (str, optional): Mode used for filtering cells (e.g., "contains", "greater than").
        filter_tuple (tuple, optional): Parameters for filtering (depends on ``filter_mode``).
        xlim (tuple of float, optional): X-axis limits.
        ylim (tuple of float, optional): Y-axis limits.
        spot_size (float): Marker extent in um (diameter of circles, edge of squares),
            drawn at this true size at any zoom level or dpi; markers smaller than one
            pixel are drawn at one pixel. Default is 10.
        alpha (float): Transparency for plotted markers. Default is 1.0.
        nan_color (str, optional): Color for cells with missing values (NaN) in
            categorical columns. If None (default), NaN cells are excluded from the
            plot. Has no effect on continuous columns.
        max_cols (int, optional): Maximum number of subplot columns. Default is 4.
        figsize (tuple[float, float], optional): Overall figure size (width, height) in
            inches. Overrides ``subplot_width``/``subplot_height`` when provided. Default
            None (derived from ``subplot_width``/``subplot_height``).
        subplot_width (float, optional): Width of each individual subplot panel in inches.
            Used to derive the total figure width when ``figsize`` is None. Default None
            (falls back to 6).
        subplot_height (float, optional): Height of each individual subplot panel in
            inches. Used to derive the total figure height when ``figsize`` is None.
            Default None (falls back to 6).
        savepath (str, optional): Path to save the figure (if None, figure is not saved).
        save_only (bool): If True, save figure without displaying. Default is False.
        dpi_save (int): Resolution in DPI for saving the figure. Default is 300.
        show (bool): Whether to display the plot. Default is True.
        plot_config (PlotConfig, optional): Plot appearance/rendering configuration
            (cmap, palette, spot_type, crange, ...) for options with no dedicated
            top-level kwarg. Fields it shares with a top-level kwarg above (xlim, ylim,
            spot_size, alpha, nan_color) are preserved unless that kwarg is also passed
            explicitly - see the precedence note below.
        layout_config (LayoutConfig, optional): Figure/subplot layout configuration
            (header, wspace, hspace, dpi_display, ...) for options with no dedicated
            top-level kwarg. Fields it shares with a top-level kwarg above (max_cols,
            figsize, subplot_width, subplot_height) follow the same precedence note.
        data_config (DataConfig, optional): Data-extraction configuration (raw,
            obsm_key, name_column, ...) for options with no dedicated top-level kwarg.
            Fields it shares with a top-level kwarg above (layer, region_tuple,
            annotation_tuple, image_key, filter_mode, filter_tuple) follow the same
            precedence note.
        verbose (bool): If True, print progress messages. Default is False.

    Note:
        Precedence between a shared top-level kwarg and a caller-provided
        ``plot_config``/``layout_config``/``data_config`` object: the top-level kwarg
        wins whenever it is explicitly passed - even when the value it is given
        equals a config default; otherwise the config object's value is left
        untouched. E.g. ``spatial(data, keys, plot_config=PlotConfig(spot_size=20))``
        keeps ``spot_size=20``; adding ``spot_size=8`` at the top level overrides it to 8.

    Returns:
        None: Displays and/or saves the generated spatial plot(s).

    Raises:
        ValueError: If filter parameters or layout arguments are invalid.
        ValueError: If mixed categorical and continuous values are encountered for a key.

    Examples:
    >>> import insitupy as isp
    >>> isp.pl.spatial(data, keys="GeneA")
    >>> isp.pl.spatial(exp, keys=["GeneA", "GeneB"], image_key="lowres", savepath="plots/spatial.png")
    """

    if annotations_key is not None:
        warnings.warn(
            "'annotations_key' is deprecated, use 'annotation_tuple' instead.",
            DeprecationWarning,
            stacklevel=3,  # spatial is wrapped by @with_insitupy_style
        )
        annotation_tuple = annotations_key

    # convert arguments to lists
    keys = convert_to_list(keys)

    if cells_layer is not None and units_layer is not None:
        raise ValueError("cells_layer and units_layer are mutually exclusive")

    # experiments: samples without the requested layer are skipped (empty panel, one warning);
    # single samples keep raising the usual error from the layer lookup
    present = None
    if _is_experiment(data):
        present = [_has_table_layer(xd, cells_layer, units_layer) for xd in data.data]
        desc = _layer_desc(cells_layer, units_layer)
        if not any(present):
            available = {}
            for xd in data.data:
                container = xd.units if units_layer is not None else xd.cells
                available[xd.sample_id] = list(container.keys()) if container is not None else []
            raise ValueError(f"No sample has {desc}. Available per sample: {available}")
        if not all(present):
            missing = [xd.sample_id for xd, p in zip(data.data, present, strict=True) if not p]
            warnings.warn(
                f"{desc} not found in sample(s) {missing} (absent or not loaded); "
                "their panels are left empty.",
                UserWarning, stacklevel=3,  # spatial is wrapped by @with_insitupy_style
            )

    # whether the caller gave a plot_config (units marker defaults apply only without one)
    user_plot_config = plot_config is not None

    # init config classes
    # Copy any caller-provided config so we never mutate the caller's object.
    # calc_subplot_params() writes derived layout state (figsize, n_rows, ...)
    # back into layout_config; reusing the same config across calls would
    # otherwise leak a stale figsize (e.g. one sized for the full experiment
    # onto a later subset plot), which also inflates the computed marker size.
    plot_config = PlotConfig() if plot_config is None else copy.copy(plot_config)
    layout_config = LayoutConfig() if layout_config is None else copy.copy(layout_config)
    data_config = DataConfig() if data_config is None else copy.copy(data_config)

    # Update config objects with function arguments - only kwargs the caller
    # explicitly passed (i.e. not left at the `_UNSET` sentinel) are forwarded, so a
    # caller-provided plot_config/layout_config/data_config is not silently clobbered
    # by defaults for fields the caller never touched at the top level.
    data_config.update_values(**_explicit_overrides(
        layer=layer,
        region_tuple=region_tuple,
        annotations_key=annotation_tuple,
        image_key=image_key,
        filter_mode=filter_mode,
        filter_tuple=filter_tuple,
    ))
    plot_config.update_values(**_explicit_overrides(
        xlim=xlim, ylim=ylim,
        spot_size=spot_size, alpha=alpha,
        nan_color=nan_color,
    ))
    layout_config.update_values(**_explicit_overrides(
        max_cols=max_cols,
        figsize=figsize,
        subplot_width=subplot_width,
        subplot_height=subplot_height,
    ))

    # figsize wins over subplot_width/height inside calc_subplot_params, so warn when
    # the caller asked for a per-panel size that an effective figsize - top-level OR
    # from a caller-provided layout_config - will silently override. Must run BEFORE
    # calc_subplot_params, which fills layout_config.figsize with a *computed* size.
    if layout_config.figsize is not None and (
        subplot_width is not _UNSET or subplot_height is not _UNSET
    ):
        warnings.warn(
            "subplot_width/subplot_height are ignored when figsize is provided; "
            "figsize sets the total figure size and takes precedence.",
            UserWarning, stacklevel=3,
        )

    _check_image_key(data, data_config.image_key, present=present)

    # marker defaults for units layers (bins tile as squares, spots are circles)
    if units_layer is not None and spot_size is _UNSET and not user_plot_config:
        _set_units_marker_defaults(data, units_layer, plot_config, present=present)

    # check whether the data is an InSituExperiment or a single InSituData
    if _is_experiment(data):
        n_data = len(data)

        # synchronize colors before plotting; implicit because keys that are genes,
        # numeric or already colored are expected here and need no sync, and stored
        # uns colors are adopted, never overwritten. An explicit palette is plot-only
        # and wins over exp.colors, so it skips the sync.
        # (experiment colors are cells-only, so units layers skip this)
        if units_layer is None and plot_config.palette is None:
            # with samples missing the layer, sync on a view of the others (sync loops over
            # every sample and would raise); the view shares the samples, so their uns
            # colors are written, only the parent's experiment color store is not updated
            sync_target = data
            if not all(present):
                sync_target = data[[i for i, p in enumerate(present) if p]]
            sync_target._sync_colors(
                keys=keys,
                cells_layer=cells_layer,
                palette=None,
                overwrite=False,
                implicit=True,
            )
    else:
        n_data = 1
        # is_experiment = False

    color_config = _ColorConfigMultiPlot(
        data=data,
        cells_layer=cells_layer,
        units_layer=units_layer,
        keys=keys,
        data_config=data_config,
        plot_config=plot_config,
        present=present,
    )

    layout_config.calc_subplot_params(
        keys=keys,
        n_data=n_data,
        color_config=color_config
        )

    # setup the subplots
    fig, axs = _setup_subplots(
        layout_config=layout_config,
        verbose=verbose
    )

    _plot_to_subplots(
        data,
        keys,
        cells_layer,
        fig,
        axs,
        plot_config,
        layout_config,
        data_config,
        color_config,
        units_layer=units_layer,
        present=present,
    )

    save_and_show_figure(
        savepath=savepath,
        fig=fig,
        save_only=save_only,
        show=show,
        dpi_save=dpi_save,
        tight=False
    )

    gc.collect()

# deprecated version
def plot_spatial(*args, **kwargs):
    """Deprecated. Use :func:`spatial` instead."""
    from insitupy._warnings import plot_functions_deprecations_warning
    plot_functions_deprecations_warning(name="spatial")

def _layer_desc(cells_layer, units_layer) -> str:
    if units_layer is not None:
        return f"units_layer '{units_layer}'"
    if cells_layer is not None:
        return f"cells_layer '{cells_layer}'"
    return "a cell layer"

def _setup_subplots(
    layout_config: LayoutConfig,
    verbose: bool = False
    ):
    logger.info("Setup subplots.") if verbose else None

    fig, axs = plt.subplots(
        layout_config.n_rows, layout_config.n_cols,
        figsize=layout_config.figsize,
        dpi=layout_config.dpi_display
        )

    # adjust the subplots
    fig.subplots_adjust(wspace=layout_config.wspace, hspace=layout_config.hspace)

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

def _plot_to_subplots(
    data,
    keys,
    cells_layer,
    fig,
    axs,
    plot_config,
    layout_config,
    data_config,
    color_config,
    verbose: bool = False,
    units_layer: str | None = None,
    present: list[bool] | None = None,
):
    logger.info("Do plotting.") if verbose else None

    if _is_experiment(data):
        n_data = len(data)
    else:
        n_data = 1

    total_hidden = 0
    affected_keys = set()

    for idx in range(n_data):

        if present is not None and not present[idx]:
            # sample without the requested layer: empty, titled panel per key
            sample_name = _sample_name(data, idx, data_config)
            for idx_key, key in enumerate(keys):
                ax, _ = _determine_axes(axs, idx, idx_key, layout_config)
                _set_panel_title(ax, key, idx_key, sample_name, plot_config, layout_config)
                ax.text(0.5, 0.5, f"no {_layer_desc(cells_layer, units_layer)}",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue

        # retrieve data
        ad, sample_name, image_data, regions_data, annotations_data = _get_data(
            data, idx, cells_layer, data_config, units_layer=units_layer,
            )

        for idx_key, key in enumerate(keys):
            # get axis to plot
            ax, add_legend = _determine_axes(axs, idx, idx_key, layout_config)

            # only add legend if this is allowed by config class
            add_legend = plot_config.show_legend and add_legend

            # plot single spatial plot in given axis
            n_hidden = _single_spatial(
                adata=ad,
                key=key, idx_key=idx_key, name=sample_name,
                fig=fig, ax=ax, add_legend=add_legend,
                color_config=color_config, data_config=data_config,
                layout_config=layout_config, plot_config=plot_config,
                regions_data=regions_data, annotations_data=annotations_data, image_data=image_data
            )
            if n_hidden:
                total_hidden += n_hidden
                affected_keys.add(key)

    if total_hidden:
        _warn_na_cells_hidden(total_hidden, sorted(affected_keys))

    if layout_config.add_legend_to_last_subplot and plot_config.show_legend:
        # get axis of last subplots for color legend
        ax = axs[layout_config.n_plots-1]

        k = list(color_config.keys())[0]
        color_config_key = color_config[k]
        # is_categorical = color_config_key["is_categorical"]
        # if is_categorical:
        color_dict = color_config[k]["color_dict"]
        if plot_config.nan_color is not None and color_config[k].get("has_na"):
            color_dict = {**color_dict, NA_CATEGORY: plot_config.nan_color}
        _add_colorlegend_to_axis(
            color_dict=color_dict,
            max_per_col=plot_config.legend_max_per_col,
            ax=ax)

def _single_spatial(
    adata: AnnData,
    key: list[str],
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
    regions_data: RegionsData | None = None,
    annotations_data: AnnotationsData | None = None,
    image_data: ImageData | None = None,
    ):

    # get color values for expression data or categories
    color_values, categorical = _extract_color_values(
        adata=adata, key=key, raw=data_config.raw, layer=data_config.layer
    )

    n_hidden = 0

    if color_values is None:
        logger.warning(f"Key '{key}' not found.")
        ax.set_axis_off()

    else:
        # retrieve color dictionary
        color_dict = color_config[key]["color_dict"]
        crange = color_config[key]["crange"]
        categorical = color_config[key]["is_categorical"] # True if color_dict is not None

        x_coords, y_coords, xlim, ylim, (x_offset, y_offset) = _prepare_limits_and_coordinates(
            adata=adata,
            regions_data=regions_data,
            region_tuple=data_config.region_tuple,
            xlim=plot_config.xlim, ylim=plot_config.ylim,
            obsm_key=data_config.obsm_key,
            origin_zero=plot_config.origin_zero,
            margin=plot_config.spot_size / 2,
            )

        image, pixel_size, vmin, vmax, pixel_xlim, pixel_ylim = _extract_image_information(
            image_data, image_key=data_config.image_key,
            # the image lives in absolute coordinates: undo the origin_zero shift
            xlim=(xlim[0] + x_offset, xlim[1] + x_offset),
            ylim=(ylim[0] + y_offset, ylim[1] + y_offset),
            pixelwidth_per_subplot=plot_config.pixelwidth_per_subplot,
            histogram_setting=plot_config.histogram_setting
        )

        annotations_df = _extract_annotations_information(
            annotations_data,
            annotations_key=data_config.annotations_key
        )
        if annotations_df is not None and (x_offset or y_offset):
            # annotations are stored in um, like the cell coordinates: shift them too
            annotations_df = annotations_df.copy()
            annotations_df["geometry"] = annotations_df.geometry.translate(
                xoff=-x_offset, yoff=-y_offset
            )

        # set the axes (add titles, set limits, etc.)
        _configure_axis_and_title(
            ax=ax,
            key=key, idx_key=idx_key, sample_name=name,
            plot_config=plot_config, layout_config=layout_config,
            xlim=xlim, ylim=ylim
        )

        if image is not None:
            # plot image data
            extent = (
                pixel_xlim[0] * pixel_size - x_offset,
                pixel_xlim[1] * pixel_size - x_offset,
                pixel_ylim[1] * pixel_size - y_offset,
                pixel_ylim[0] * pixel_size - y_offset
                )

            ax.imshow(
                image,
                extent=extent,
                origin='upper', cmap='gray', vmin=vmin, vmax=vmax)

        # plot transcriptomic data
        if categorical:
            color_values = pd.Series(color_values)
            na_mask = color_values.isna().to_numpy()
            n_na = int(na_mask.sum())
            if n_na:
                if plot_config.nan_color is None:
                    keep = ~na_mask
                    x_coords = x_coords[keep]
                    y_coords = y_coords[keep]
                    color_values = color_values[keep]
                    n_hidden = n_na
                else:
                    color_values = _coerce_na_for_plot(color_values)
                    color_dict = {**color_dict, NA_CATEGORY: plot_config.nan_color}

            # one RGBA row per category, indexed by category code (vectorized for millions of bins)
            codes = pd.Categorical(np.asarray(color_values, dtype=object),
                                   categories=list(color_dict)).codes
            if (codes < 0).any():
                missing = sorted({str(v) for v in np.asarray(color_values, dtype=object)[codes < 0]})
                raise ValueError(f"No color defined for categories {missing} of key '{key}'.")
            _scatter_data_units(
                ax, x_coords, y_coords, plot_config,
                facecolors=colors.to_rgba_array(list(color_dict.values()))[codes],
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
        else:
            # NaN values are masked, i.e. drawn in the colormap's "bad" color (transparent)
            s = _scatter_data_units(
                ax, x_coords, y_coords, plot_config,
                array=np.ma.masked_invalid(np.asarray(color_values, dtype=float).ravel()),
                cmap=plot_config.cmap,
                norm=plot_config.normalize,
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

    return n_hidden

class _ColorConfigMultiPlot:
    def __init__(
        self,
        data: InSituData | InSituExperiment,
        data_config: DataConfig,
        plot_config: PlotConfig,
        cells_layer: str | None = None,
        keys: str | list[str] = None,
        units_layer: str | None = None,
        present: list[bool] | None = None,
        ):
        # add properties
        self._dict = {}

        if _is_experiment(data):
            data_list = data.data
            if present is not None:
                # samples without the layer are not plotted and must not be resolved
                data_list = [xd for xd, p in zip(data_list, present, strict=True) if p]
            # experiment colors are cells-only; an explicit palette wins over
            # exp.colors (plot-only, nothing is saved)
            exp_colors = (
                data.colors if units_layer is None and plot_config.palette is None else None
            )
            # is_experiment = True
        else:
            data_list = [data]
            exp_colors = None
            # is_experiment = False

        for key in keys:
            layer_color_dict = (
                exp_colors.get(key, cells_layer=cells_layer) if exp_colors is not None else None
            )
            use_layer_dict = False
            if layer_color_dict is not None:
                # resolve the key once per dataset; the experiment colors are only valid
                # if the key is categorical everywhere it is found (a key that resolves
                # to a gene must be drawn as expression, not with a categorical palette)
                resolved = [
                    _extract_color_values(
                        adata=_get_table_layer(xd, cells_layer=cells_layer, units_layer=units_layer).table,
                        key=key, raw=data_config.raw, layer=data_config.layer
                    )
                    for xd in data_list
                ]
                found = [(values, cat) for values, cat in resolved if values is not None]
                use_layer_dict = all(cat is True for _, cat in found)
                if not use_layer_dict:
                    logger.debug(
                        "'%s' does not resolve to a categorical obs column in all samples "
                        "(e.g. it is a gene); ignoring its experiment color dict.", key
                    )

            if use_layer_dict:
                # use color_dict from InSituExperiment (copy: never mutate exp.colors);
                # categories missing from a partial dict get the same fallback colors
                # that the write-through stored in uns
                has_na = any(pd.isna(values).any() for values, _ in found)
                color_dict = dict(layer_color_dict)
                fallback = _fallback_category_colors(
                    [values for values, _ in found], DEFAULT_CATEGORICAL_CMAP
                )
                for cat, color in fallback.items():
                    color_dict.setdefault(cat, color)
                color_entry = {
                    "color_dict": color_dict,
                    "max_value": None,
                    "is_categorical": True,
                    "crange": None,
                    "has_na": has_na,
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
                    plot_config=plot_config,
                    units_layer=units_layer,
                    )

            # add entry to dictionary
            self._dict[key] = color_entry

    def __getitem__(self, key):
        return self._dict.get(key)

    @property
    def dict(self):
        """Return the underlying colour configuration dictionary."""
        return self._dict

    def keys(self):
        """Return the keys of the colour configuration dictionary."""
        return self._dict.keys()

    def _add_color_entry(
        self, data_list, key, cells_layer,
        data_config, plot_config, units_layer=None
        ):
        # preconfigure entry
        color_entry = {
            "color_dict": None,
            "max_value": None,
            "is_categorical": False,
            "crange": None,
            "has_na": False,
        }
        if len(data_list) == 1:
            # one dataset
            xd = data_list[0]
            celldata = _get_table_layer(
                xd, cells_layer=cells_layer, units_layer=units_layer
                )
            ad = celldata.table

            # extract the data
            color_values, is_categorical = _extract_color_values(
                adata=ad, key=key, raw=data_config.raw, layer=data_config.layer
            )

            if color_values is None:
                raise ValueError(f"Key '{key}' not found in the data.")

            if is_categorical:
                color_entry["is_categorical"] = True
                color_entry["has_na"] = bool(pd.isna(color_values).any())
                # complete colors saved in uns are used unless a palette was given explicitly
                stored = (
                    _stored_color_dict(ad, key, color_values)
                    if plot_config.palette is None else None
                )
                if stored is not None:
                    color_entry["color_dict"] = stored
                else:
                    # create a new color mapping
                    color_entry["color_dict"] = create_cmap_mapping(
                        color_values.values, cmap=plot_config.palette
                        )

            else:
                # no values are categorical - collect the maximum values
                color_values = np.asarray(color_values, dtype=float).ravel()
                color_entry["max_value"] = np.nanmax(color_values)
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
            has_na = False
            stored_dicts = []  # stored uns colors per sample (None if unusable)
            for xd in data_list:
                celldata = _get_table_layer(
                    xd, cells_layer=cells_layer, units_layer=units_layer
                    )
                ad = celldata.table

                # extract the data
                color_values, is_categorical = _extract_color_values(
                    adata=ad, key=key, raw=data_config.raw, layer=data_config.layer
                )

                if color_values is not None:
                    if is_categorical:
                        has_na |= bool(pd.isna(color_values).any())
                        value_list.append(np.asarray(_parse_unique_categories(color_values)))
                        stored_dicts.append(_stored_color_dict(ad, key, color_values))
                    else:
                        value_list.append(np.asarray(color_values, dtype=float).ravel())

                categorical_list.append(is_categorical)

            if len(value_list) == 0:
                raise ValueError(f"Key '{key}' not found in any of the datasets.")

            if np.all(categorical_list):
                # all values are categorical - concatenate all values
                all_values = np.unique(np.concatenate(value_list))
                merged = (
                    _merge_stored_color_dicts(stored_dicts, all_values)
                    if plot_config.palette is None else None
                )
                if merged is not None:
                    color_entry["color_dict"] = merged
                else:
                    color_entry["color_dict"] = create_cmap_mapping(
                        all_values, cmap=plot_config.palette
                        )
                color_entry["is_categorical"] = True
                color_entry["has_na"] = has_na

            elif not np.any(categorical_list):
                # no values are categorical - pool the values of all samples
                pooled = np.concatenate(value_list)
                color_entry["max_value"] = np.nanmax(pooled)
                color_entry["is_categorical"] = False

                if plot_config.crange is not None:
                    color_entry["crange"] = plot_config.crange
                else:
                    color_entry["crange"] = _get_crange(
                        color_values=pooled,
                        crange_type=plot_config.crange_type
                        )
            else:
                raise ValueError(f"Values found for key {key} showed mixed type (categorical/numeric).")

        return color_entry

def _get_crange(color_values, crange_type):
    if crange_type == 'max':
        crange = [0, np.nanmax(color_values)]
    elif crange_type == 'minmax':
        crange = [np.nanmin(color_values), np.nanmax(color_values)]
    elif crange_type == 'upper_percentile':
        crange = [0, np.nanpercentile(color_values, 99)]
    elif crange_type == 'percentile':
        crange = [np.nanpercentile(color_values, 1), np.nanpercentile(color_values, 99)]
    else:
        raise ValueError(f"Unknown crange_type: {crange_type}. Must be one of 'max', 'minmax', 'upper_percentile' or 'percentile'.")

    return crange

def _configure_axis_and_title(
    ax,
    key,
    idx_key,
    sample_name,
    plot_config,
    layout_config,
    xlim,
    ylim
    ):

    # configure axis
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_aspect(1)
    ax.set_facecolor(plot_config.background_color)

    if plot_config.show_scale:
        ax.set_xlabel('µm', fontsize=plot_config.label_size)
        ax.set_ylabel('µm', fontsize=plot_config.label_size)
        ax.tick_params(labelsize=plot_config.tick_label_size)

    else:
        # hide axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    _set_panel_title(ax, key, idx_key, sample_name, plot_config, layout_config)

def _set_panel_title(ax, key, idx_key, sample_name, plot_config, layout_config):
    """Panel title and (first key) sample label; shared with the empty panels."""
    if layout_config.multidata and not layout_config.multikeys:
        if plot_config.show_title:
            ax.set_title(
                sample_name + "\n" + key,
                fontsize=plot_config.title_size, #fontweight='bold',
                pad=10,
                rotation=0
                )
    else:
        if plot_config.show_title:
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
        raise ValueError(f"`len(self.axs.shape)` has wrong shape {len(axs.shape)}. Requires 1 or 2.")

    return ax, add_legend

def _sample_name(data, idx, data_config) -> str:
    """Display name of sample ``idx`` (single InSituData or experiment)."""
    try:
        xd = data.data[idx]
        meta = data._metadata.iloc[idx]
    except AttributeError:
        xd = data
        meta = None

    if data_config.name_column is None or meta is None:
        name = xd.sample_id
    else:
        name = meta[data_config.name_column]
    # some readers leave sample_id None (e.g. read_visium_hd); titles need a string
    return f"sample {idx}" if name is None else str(name)

def _get_data(
    data,
    idx,
    cells_layer,
    data_config,
    units_layer=None,
    ):
    # extract the InSituData object
    try:
        xd = data.data[idx]
    except AttributeError:
        xd = data

    # retrieve the right cell data
    celldata = _get_table_layer(xd, cells_layer=cells_layer, units_layer=units_layer)

    # extract anndata
    adata = celldata.table

    if units_layer is not None and data_config.obsm_key not in adata.obsm:
        raise ValueError(
            f"Units layer '{units_layer}' has no `.obsm['{data_config.obsm_key}']` coordinates. "
            f"Set `table.obsm['{data_config.obsm_key}']` from `shapes.centroid` first, "
            "e.g. `np.c_[shapes.centroid.x, shapes.centroid.y]`."
        )

    # filter anndata
    if data_config.filter_mode is not None and data_config.filter_tuple is not None:
        adata = filter_anndata(
            adata=adata, filter_mode=data_config.filter_mode, filter_tuple=data_config.filter_tuple
            )

    sample_name = _sample_name(data, idx, data_config)

    # samples without the requested image are plotted without one (spatial() has
    # already validated the key and warned/raised)
    if data_config.image_key is not None and data_config.image_key in xd.images:
        imagedata = xd.images
    else:
        imagedata = None

    # get regions
    regions = xd.regions

    # get annotations
    annotations = xd.annotations

    return adata, sample_name, imagedata, regions, annotations



        # else:
        #     #if ConfigData.categorical:
        #     ax.set_axis_off()

def _check_image_key(data, image_key, present=None):
    """Validate ``image_key`` before plotting.

    A single sample without the image raises. For an experiment, samples without it
    are plotted without an image and one warning lists them; it raises only if no
    sample has the image. Samples skipped for a missing layer (``present[i]`` False)
    are not plotted and therefore not checked.
    """
    if image_key is None:
        return

    def _not_found(available):
        msg = f"image_key '{image_key}' not found. Available images: {list(available)}."
        if not available:
            msg += " Call load_images() first if the list is empty."
        return msg

    if _is_experiment(data):
        plotted = [
            xd for i, xd in enumerate(data.data) if present is None or present[i]
        ]
        missing = [xd.sample_id for xd in plotted if image_key not in xd.images]
        if len(missing) == len(plotted):
            available = sorted({k for xd in plotted for k in xd.images.keys()})
            raise ValueError(_not_found(available) + " No sample has this image.")
        if missing:
            warnings.warn(
                f"image_key '{image_key}' not found in sample(s) {missing}; "
                "plotting them without an image.",
                # helper -> spatial -> @with_insitupy_style wrapper -> caller
                UserWarning, stacklevel=4,
            )
    elif image_key not in data.images:
        raise ValueError(_not_found(data.images.keys()))


def _prepare_limits_and_coordinates(
    adata,
    regions_data,
    region_tuple,
    xlim,
    ylim,
    obsm_key,
    origin_zero,
    margin=0.0,
    ):
    """Return marker coordinates and axis limits in um.

    ``margin`` (um) is added on each side of the data extent for limits that are
    neither given nor derived from ``region_tuple``.
    """

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

    if origin_zero:
        if xlim is not None:
            xlim = [xlim[0] - x_offset, xlim[1] - x_offset]
        if ylim is not None:
            ylim = [ylim[0] - y_offset, ylim[1] - y_offset]

    # default limits: data extent plus a margin, so that edge markers are not cut in half
    if xlim is None:
        xlim = (x_coords.min() - margin, x_coords.max() + margin)

    if ylim is None:
        ylim = (y_coords.min() - margin, y_coords.max() + margin)

    return x_coords, y_coords, xlim, ylim, (x_offset, y_offset)

def _extract_image_information(
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
        if not isinstance(img_pyramid, (list, tuple)):
            # a single array (e.g. an image added in memory) is a one-level pyramid
            img_pyramid = [img_pyramid]
        # level pixel sizes from the level widths (not assuming a factor of 2 between levels)
        base_w = img_pyramid[0].shape[1]
        pixel_sizes_levels = np.array(
            [orig_pixel_size * base_w / lvl.shape[1] for lvl in img_pyramid]
        )

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

        # draw at most ~2 pixels per display pixel: stride the array only (the crop
        # still covers the same area, so pixel size and pixel limits stay unchanged)
        k = int(max_pixel_size // selected_pixel_size)
        if k > 1:
            image = image[::k, ::k]

        if image.size == 0:
            # limits fully outside the image: nothing to draw
            return None, selected_pixel_size, None, None, pixel_xlim, pixel_ylim

        if histogram_setting is None:
            vmin = vmax = None
        elif histogram_setting == "auto":
            if ImageDataObject.metadata[image_key].get("rgb", False):
                # RGB images are drawn without a value range
                vmin = vmax = None
            else:
                vmin = da.percentile(image.ravel(), 30).compute().item()
                vmax = da.percentile(image.ravel(), 99.5).compute().item()
        elif isinstance(histogram_setting, tuple):
            vmin = histogram_setting[0]
            vmax = histogram_setting[1]
        else:
            raise ValueError(f"Unknown type for histogram_setting: {type(histogram_setting)}")

    return image, selected_pixel_size, vmin, vmax, pixel_xlim, pixel_ylim

def _extract_annotations_information(
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

def _set_units_marker_defaults(data, units_layer, plot_config, present=None):
    """Set marker type/size so that units tile the tissue (bins) or match their extent (spots).

    On an experiment the first sample that has the layer (``present[i]``) is used.
    """
    if _is_experiment(data):
        idx = next((i for i, p in enumerate(present) if p), 0) if present is not None else 0
        xd = data.data[idx]
    else:
        xd = data
    layer = _get_table_layer(xd, units_layer=units_layer)
    bin_size = layer.table.uns.get("bin_size_um")
    if bin_size:
        plot_config.update_values(spot_type="s", spot_size=float(bin_size))
    elif not layer.shapes.empty:
        median_area = float(np.median(layer.shapes.geometry.area))
        plot_config.update_values(spot_type="o", spot_size=2 * float(np.sqrt(median_area / np.pi)))


class _DataUnitMarkers(PathCollection):
    """Scatter markers whose size is given in data units (um) instead of points.

    Matplotlib sizes scatter markers in points^2, so a size in um has to be converted
    with the pixels per um of the axes. That value is only final at draw time: the
    equal aspect shrinks the axes box when the figure is drawn, and colorbar/legend
    axes appended with ``make_axes_locatable`` take space afterwards. Converting
    earlier draws the markers too large (1.2-4x, depending on the data aspect ratio),
    so this collection resets its sizes from the current ``transData`` on every
    draw. That also keeps them right after zooming, resizing or saving at another dpi.

    Args:
        ax: Axes to draw into (offsets are in its data coordinates).
        offsets: (n, 2) marker centers in data units.
        size: Marker extent in data units - square edge for ``"s"``, diameter for ``"o"``.
        marker: Matplotlib marker specification.
        min_px: Lower bound of the drawn marker extent in display pixels, so that
            markers smaller than a pixel (whole-slide views of small bins) stay visible.
        **kwargs: Forwarded to :class:`~matplotlib.collections.PathCollection`
            (``facecolors``, ``array``, ``cmap``, ``norm``, ``alpha``, ...).
    """

    def __init__(self, ax, offsets, size, marker="o", min_px=1.0, **kwargs):
        marker_obj = marker if isinstance(marker, MarkerStyle) else MarkerStyle(marker)
        path = marker_obj.get_path().transformed(marker_obj.get_transform())
        kwargs.setdefault("linewidths", 0)
        super().__init__(
            (path,), sizes=[1.0], offsets=offsets, offset_transform=ax.transData, **kwargs
        )
        # marker paths are scaled in display space (as in Axes.scatter); without this,
        # add_collection would set transData and draw the paths in data units
        self.set_transform(IdentityTransform())
        self.marker = marker
        self.data_size = float(size)
        self.min_px = float(min_px)
        self._clamp_logged = False

    def _size_in_px(self):
        """Marker extent in display pixels for the current axes layout (before clamping)."""
        trans = self.axes.transData
        px_per_unit = abs(trans.transform((0, 1))[1] - trans.transform((0, 0))[1])
        return self.data_size * px_per_unit

    def draw(self, renderer):
        px = self._size_in_px()
        if px < self.min_px:
            if not self._clamp_logged:
                logger.info(
                    f"Markers of {self.data_size:g} um are {px:.2f} px wide at this zoom level and "
                    f"are drawn at {self.min_px:g} px. Zoom in (xlim/ylim), enlarge figsize or "
                    "save with a higher dpi_save to see them at their true size."
                )
                self._clamp_logged = True
            px = self.min_px
        # sizes are in points^2; points = pixels * 72 / dpi (dpi is the save dpi while saving)
        self.set_sizes([(px * 72.0 / self.figure.dpi) ** 2])
        super().draw(renderer)


def _scatter_data_units(ax, x, y, plot_config, **kwargs):
    """Draw markers of ``plot_config.spot_size`` um at (x, y) and return the collection."""
    coll = _DataUnitMarkers(
        ax, np.column_stack([x, y]), size=plot_config.spot_size,
        marker=plot_config.spot_type, alpha=plot_config.alpha, **kwargs,
    )
    ax.add_collection(coll, autolim=False)
    return coll

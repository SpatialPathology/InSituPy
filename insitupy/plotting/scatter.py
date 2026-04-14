"""
Fast embedding plots for large-scale single-cell data using datashader.

Usage:
    from fast_embedding_plot import umap, embedding

    # Static UMAP
    umap(adata, color="celltype")

    # Interactive with multiple colors
    umap(adata, color=["celltype", "n_counts"], interactive=True)

    # General embedding
    embedding(adata, basis="X_pca", color="leiden")

Optional dependencies (install as needed):
    pip install datashader holoviews bokeh  # For datashader rendering
    pip install jupyter-scatter             # For jscatter rendering
    pip install plotly                      # For plotly rendering
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    import holoviews as hv
    import jscatter
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from insitupy._constants import with_insitupy_style


def _check_datashader():
    """Check if datashader and matplotlib are available."""
    try:
        import datashader
        import matplotlib.pyplot
        return True
    except ImportError:
        return False


def _check_holoviews():
    """Check if holoviews is available."""
    try:
        import holoviews
        return True
    except ImportError:
        return False


def _check_jscatter():
    """Check if jupyter-scatter is available."""
    try:
        import jscatter
        return True
    except ImportError:
        return False


def _check_plotly():
    """Check if plotly is available."""
    try:
        import plotly
        return True
    except ImportError:
        return False


def _check_scanpy():
    """Check if scanpy is available (for color palettes)."""
    try:
        import scanpy
        return True
    except ImportError:
        return False


def _get_color_values(
    adata: ad.AnnData,
    key: str
) -> tuple[np.ndarray, Literal["categorical", "continuous"]]:
    """
    Retrieve color values from adata.obs or adata.X.

    Returns array of values and type ("categorical" or "continuous").
    """
    if key in adata.obs.columns:
        values = adata.obs[key]
        if (
            isinstance(values.dtype, pd.CategoricalDtype)
            or values.dtype == object
            or pd.api.types.is_bool_dtype(values)
        ):
            return values.astype("category"), "categorical"
        else:
            return values.values, "continuous"

    if key in adata.var_names:
        expr = adata[:, key].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        return np.asarray(expr).flatten(), "continuous"

    raise KeyError(f"'{key}' not found in adata.obs or adata.var_names")


def _get_default_palette(n_cats: int) -> list[str]:
    """Get default color palette, using scanpy's if available."""
    if _check_scanpy():
        import scanpy as sc
        palettes = sc.pl.palettes
        if n_cats <= 10:
            return palettes.vega_10
        elif n_cats <= 20:
            return palettes.vega_20
        elif n_cats <= 28:
            return palettes.zeileis_28
        else:
            return palettes.godsnot_102
    else:
        # Fallback palette (tab10 + tab20)
        tab10 = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        tab20 = tab10 + [
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
        ]
        if n_cats <= 10:
            return tab10
        else:
            return tab20


def _get_colormap(
    values: np.ndarray | pd.Categorical,
    color_type: Literal["categorical", "continuous"],
    cmap: str | None = None
) -> tuple[dict | str, None]:
    """
    Generate colormap for values.

    Returns (colormap, None) where:
    - For categorical: colormap is dict mapping categories to colors
    - For continuous: colormap is string cmap name
    """
    if color_type == "categorical":
        categories = values.cat.categories
        n_cats = len(categories)
        palette = _get_default_palette(n_cats)
        color_dict = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
        return color_dict, None
    else:
        return cmap or "viridis", None


def _get_vmin_vmax(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    vmax_percentile: float | None = None
) -> tuple[float, float]:
    """Determine vmin and vmax for continuous color scale."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.bool_):
        values = values.astype(np.float32)

    if vmin is None:
        vmin = float(np.nanmin(values))

    if vmax_percentile is not None:
        vmax = float(np.nanpercentile(values, vmax_percentile))
    elif vmax is None:
        vmax = float(np.nanmax(values))

    return vmin, vmax


def _plot_static_categorical(
    ax: "plt.Axes",
    df: pd.DataFrame,
    color_key: str,
    color_dict: dict,
    point_size: float
) -> None:
    """Plot categorical data with datashader."""
    import datashader as ds
    from datashader.mpl_ext import dsshow

    df["color"] = df[color_key].map(color_dict)
    dsshow(
        df,
        ds.Point("x", "y"),
        ds.count_cat(color_key),
        color_key=color_dict,
        ax=ax
    )


def _plot_static_continuous(
    ax: "plt.Axes",
    df: pd.DataFrame,
    color_key: str,
    cmap: str,
    point_size: float,
    vmin: float,
    vmax: float
) -> None:
    """Plot continuous data with datashader."""
    import datashader as ds
    from datashader.mpl_ext import dsshow

    # Clip values to vmin/vmax range for proper color mapping
    df = df.copy()
    df[color_key] = df[color_key].clip(lower=vmin, upper=vmax)

    dsshow(
        df,
        ds.Point("x", "y"),
        ds.mean(color_key),
        cmap=cmap,
        ax=ax
    )


def _add_legend(
    ax: "plt.Axes",
    color_dict: dict,
    legend_mode: Literal["full", "truncate", "separate", "none"],
    max_categories: int = 20,
    legend_entries_per_col: int = 10
) -> "plt.Figure | None":
    """Add legend to plot based on mode."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if legend_mode == "none":
        return None

    categories = list(color_dict.keys())
    n_cats = len(categories)

    if legend_mode == "truncate" and n_cats > max_categories:
        categories = categories[:max_categories]
        truncated = True
    else:
        truncated = False

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_dict[cat],
               markersize=8, label=cat)
        for cat in categories
    ]

    if truncated:
        handles.append(Line2D([0], [0], marker="", color="w", label=f"... +{n_cats - max_categories} more"))

    n_legend_entries = len(handles)
    legend_ncols = max(1, (n_legend_entries + legend_entries_per_col - 1) // legend_entries_per_col)

    if legend_mode == "separate":
        fig_width = legend_ncols * 2.5
        fig_height = min(n_legend_entries, legend_entries_per_col) * 0.25 + 0.5
        fig_legend = plt.figure(figsize=(fig_width, fig_height))
        fig_legend.legend(
            handles=handles,
            loc="center",
            frameon=False,
            ncol=legend_ncols
        )
        return fig_legend
    else:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            ncol=legend_ncols
        )
        return None


def _plot_plotly(
    df: pd.DataFrame,
    color_key: str | None,
    color_type: Literal["categorical", "continuous"] | None,
    color_dict: dict | None,
    cmap: str | None,
    title: str,
    width: int,
    height: int,
    point_size: float,
    vmin: float | None = None,
    vmax: float | None = None,
    plotly_renderer: str | None = None
) -> "go.Figure":
    """Create interactive plot with Plotly WebGL."""
    import plotly.express as px
    import plotly.io as pio

    if color_key is None or color_key == "_density":
        fig = px.scatter(
            df, x="x", y="y",
            render_mode="webgl",
            title=title
        )
    elif color_type == "categorical":
        fig = px.scatter(
            df, x="x", y="y", color=color_key,
            color_discrete_map=color_dict,
            render_mode="webgl",
            title=title
        )
    else:
        fig = px.scatter(
            df, x="x", y="y", color=color_key,
            color_continuous_scale=cmap or "viridis",
            range_color=[vmin, vmax] if vmin is not None and vmax is not None else None,
            render_mode="webgl",
            title=title
        )

    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        plot_bgcolor="white"
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)

    if plotly_renderer is not None:
        pio.renderers.default = plotly_renderer

    return fig


def _plot_jscatter(
    df: pd.DataFrame,
    color_key: str | None,
    color_type: Literal["categorical", "continuous"] | None,
    color_dict: dict | None,
    cmap: str | None,
    width: int,
    height: int,
    point_size: float,
    tooltip_keys: list[str] | None = None
) -> "jscatter.Scatter":
    """Create interactive plot with jupyter-scatter."""
    import jscatter

    kwargs = {
        "x": "x",
        "y": "y",
        "width": width,
        "height": height,
        "size": point_size,
        "legend": True,
    }

    if color_key is not None and color_key != "_density":
        kwargs["color_by"] = color_key
        if color_type == "categorical":
            kwargs["color_map"] = color_dict
        else:
            kwargs["color_map"] = cmap or "viridis"

    scatter = jscatter.Scatter(data=df, **kwargs)

    if tooltip_keys:
        scatter.tooltip(enable=True, properties=tooltip_keys)
    elif color_key is not None and color_key != "_density":
        scatter.tooltip(enable=True, properties=[color_key])

    return scatter


def _plot_interactive_bokeh(
    df: pd.DataFrame,
    color_key: str,
    color_type: Literal["categorical", "continuous"],
    color_dict: dict | None,
    cmap: str | None,
    title: str,
    width: int,
    height: int
) -> "hv.Element":
    """Create interactive plot with bokeh backend."""
    import datashader as ds
    import holoviews as hv
    from holoviews.operation.datashader import datashade, spread

    hv.extension("bokeh")

    points = hv.Points(df, kdims=["x", "y"], vdims=[color_key])

    if color_type == "categorical":
        shaded = datashade(points, aggregator=ds.count_cat(color_key), color_key=color_dict,
                          width=width, height=height)
    else:
        shaded = datashade(points, aggregator=ds.mean(color_key), cmap=cmap or "viridis",
                          width=width, height=height)

    shaded = spread(shaded, px=1)
    return shaded.opts(width=width, height=height, title=title)


def _plot_interactive_matplotlib(
    df: pd.DataFrame,
    color_key: str,
    color_type: Literal["categorical", "continuous"],
    color_dict: dict | None,
    cmap: str | None,
    title: str,
    width: int,
    height: int
) -> "hv.Element":
    """Create interactive plot with matplotlib backend."""
    import datashader as ds
    import holoviews as hv
    from holoviews.operation.datashader import datashade

    hv.extension("matplotlib")

    points = hv.Points(df, kdims=["x", "y"], vdims=[color_key])

    if color_type == "categorical":
        shaded = datashade(points, aggregator=ds.count_cat(color_key), color_key=color_dict,
                          width=width, height=height)
    else:
        shaded = datashade(points, aggregator=ds.mean(color_key), cmap=cmap or "viridis",
                          width=width, height=height)

    return shaded.opts(fig_size=200, title=title)


@with_insitupy_style
def embedding(
    adata: ad.AnnData,
    basis: str = "X_umap",
    color: str | Sequence[str] | None = None,
    nan_color: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vmax_percentile: float | None = None,
    point_size: float = 1.0,
    interactive: bool = False,
    interactive_backend: Literal["bokeh", "matplotlib"] = "bokeh",
    interactive_resolution: int = 800,
    render_mode: Literal["datashader", "jscatter", "plotly"] = "datashader",
    plotly_renderer: str | None = "notebook",
    tooltip: str | Sequence[str] | None = None,
    legend_mode: Literal["full", "truncate", "separate", "none"] = "full",
    legend_max_categories: int = 20,
    legend_entries_per_col: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ncols: int = 3,
    wspace: float | None = None,
    hspace: float | None = None,
    save: str | Path | None = None,
    show: bool | None = None,
    return_fig: bool = False
) -> "plt.Figure | hv.Layout | jscatter.Scatter | list[jscatter.Scatter] | go.Figure | list[go.Figure] | None":
    """
    Fast embedding plot using datashader for large datasets.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Key in adata.obsm for coordinates (e.g., "X_umap", "X_pca").
    color
        Key(s) for color encoding. Searches adata.obs first, then adata.var_names.
        Can be single key or list of keys for multiple panels.
    nan_color
        Color for cells with missing values (NaN) in categorical columns. If
        None (default), NaN cells are excluded from the plot. If a color string
        (e.g. "lightgray"), NaN cells are shown in that color with a "NaN"
        legend entry. Has no effect on continuous color columns.
    cmap
        Colormap for continuous values. Default: "viridis".
    vmin
        Minimum value for continuous color scale. Default: data minimum.
    vmax
        Maximum value for continuous color scale. Default: data maximum.
        Ignored if vmax_percentile is set.
    vmax_percentile
        Percentile (0-100) to use for vmax. Useful for clipping outliers.
        E.g., 95 uses the 95th percentile as vmax. Overrides vmax if set.
    point_size
        Point size. For jscatter, values 1-10 work well.
    interactive
        If True, return interactive plot.
    interactive_backend
        Backend for datashader interactive plots: "bokeh" or "matplotlib".
        Ignored when render_mode="jscatter" or "plotly".
    interactive_resolution
        Pixel resolution for interactive plots (default: 800).
    render_mode
        Rendering mode for interactive plots:
        - "datashader": Rasterized, fastest for static/overview
        - "jscatter": WebGL vector, best for zooming/selection (Jupyter)
        - "plotly": WebGL vector, works on clusters/remote servers
    plotly_renderer
        Plotly renderer to use (default: "notebook"). Options include:
        "iframe", "notebook", "jupyterlab", "browser", "png", "svg".
        Only used when render_mode="plotly".
    tooltip
        Column(s) to show in tooltip (jscatter only).
    legend_mode
        How to handle legends for categorical data:
        - "full": Show all categories
        - "truncate": Show max_categories, indicate remaining
        - "separate": Create separate legend figure
        - "none": No legend
    legend_max_categories
        Maximum categories to show when legend_mode="truncate".
    legend_entries_per_col
        Maximum legend entries per column.
    title
        Plot title. If None, uses color key.
    figsize
        Figure size (width, height) in inches.
    ncols
        Number of columns for multi-panel plots.
    wspace
        Horizontal spacing between subplots (fraction of subplot width).
        Default: None (uses matplotlib default).
    hspace
        Vertical spacing between subplots (fraction of subplot height).
        Default: None (uses matplotlib default).
    save
        Path to save figure. If None, not saved.
    show
        Whether to show figure. Default: True if save is None.
    return_fig
        If True, return the figure object.

    Returns
    -------
    Figure object if return_fig=True, else None.
    For interactive mode with datashader, returns holoviews object.
    For interactive mode with jscatter, returns Scatter widget(s).
    For interactive mode with plotly, returns Figure or list of Figures.

    Raises
    ------
    ImportError
        If required optional dependencies are not installed.
    """
    # Validate basis
    if basis not in adata.obsm:
        raise KeyError(f"'{basis}' not found in adata.obsm")

    coords = adata.obsm[basis]
    if coords.shape[1] < 2:
        raise ValueError(f"'{basis}' must have at least 2 dimensions")

    # Normalize color to list
    if color is None:
        color = [None]
    elif isinstance(color, str):
        color = [color]
    else:
        color = list(color)

    n_panels = len(color)

    # Normalize tooltip to list
    tooltip_keys = None
    if tooltip is not None:
        if isinstance(tooltip, str):
            tooltip_keys = [tooltip]
        else:
            tooltip_keys = list(tooltip)

    # Interactive mode
    if interactive:
        # plotly mode
        if render_mode == "plotly":
            if not _check_plotly():
                raise ImportError(
                    "plotly is required for render_mode='plotly'. "
                    "Install with: pip install plotly"
                )

            figs = []
            for c in color:
                df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})

                if c is not None:
                    values, color_type = _get_color_values(adata, c)
                    _had_nan = False
                    if color_type == "categorical" and values.isna().any():
                        _had_nan = True
                        if nan_color is None:
                            keep = ~values.isna().values
                            df = df[keep]
                            values = values[keep]
                        else:
                            if "NaN" not in values.cat.categories:
                                values = values.cat.add_categories(["NaN"])
                            values = values.fillna("NaN")
                    df[c] = values.values if hasattr(values, "values") else values
                    colormap, _ = _get_colormap(values, color_type, cmap)

                    if color_type == "categorical":
                        color_dict = colormap
                        if _had_nan and nan_color is not None:
                            color_dict["NaN"] = nan_color
                        cmap_use = None
                        df[c] = df[c].astype(str)
                        vmin_use, vmax_use = None, None
                    else:
                        color_dict = None
                        cmap_use = colormap
                        vmin_use, vmax_use = _get_vmin_vmax(df[c].values, vmin, vmax, vmax_percentile)
                else:
                    color_type = None
                    color_dict = None
                    cmap_use = None
                    vmin_use, vmax_use = None, None

                plot_title = title or c or basis
                fig = _plot_plotly(
                    df, c, color_type, color_dict, cmap_use,
                    plot_title, interactive_resolution, interactive_resolution, point_size,
                    vmin_use, vmax_use, plotly_renderer
                )
                figs.append(fig)

            return figs[0] if len(figs) == 1 else figs

        # jscatter mode
        if render_mode == "jscatter":
            if not _check_jscatter():
                raise ImportError(
                    "jupyter-scatter is required for render_mode='jscatter'. "
                    "Install with: pip install jupyter-scatter"
                )
            import jscatter

            plots = []
            for c in color:
                df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})

                if c is not None:
                    values, color_type = _get_color_values(adata, c)
                    _had_nan = False
                    if color_type == "categorical" and values.isna().any():
                        _had_nan = True
                        if nan_color is None:
                            keep = ~values.isna().values
                            df = df[keep]
                            values = values[keep]
                        else:
                            if "NaN" not in values.cat.categories:
                                values = values.cat.add_categories(["NaN"])
                            values = values.fillna("NaN")
                    df[c] = values.values if hasattr(values, "values") else values
                    colormap, _ = _get_colormap(values, color_type, cmap)

                    if color_type == "categorical":
                        color_dict = colormap
                        if _had_nan and nan_color is not None:
                            color_dict["NaN"] = nan_color
                        cmap_use = None
                        df[c] = df[c].astype(str)
                    else:
                        color_dict = None
                        cmap_use = colormap
                else:
                    color_type = None
                    color_dict = None
                    cmap_use = None

                p = _plot_jscatter(
                    df, c, color_type, color_dict, cmap_use,
                    interactive_resolution, interactive_resolution, point_size,
                    tooltip_keys
                )
                plots.append(p)

            if len(plots) == 1:
                return plots[0].show()
            else:
                jscatter.link(plots)
                for p in plots:
                    p.show()
                return plots

        # datashader mode (default)
        if not _check_holoviews():
            raise ImportError(
                "holoviews is required for interactive datashader plots. "
                "Install with: pip install holoviews bokeh datashader"
            )
        if not _check_datashader():
            raise ImportError(
                "datashader is required for interactive datashader plots. "
                "Install with: pip install datashader"
            )
        import holoviews as hv

        plots = []
        for c in color:
            df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})

            if c is not None:
                values, color_type = _get_color_values(adata, c)
                _had_nan = False
                _nan_keep = None
                if color_type == "categorical" and values.isna().any():
                    _had_nan = True
                    if nan_color is None:
                        _nan_keep = ~values.isna().values
                        df = df[_nan_keep]
                        values = values[_nan_keep]
                    else:
                        if "NaN" not in values.cat.categories:
                            values = values.cat.add_categories(["NaN"])
                        values = values.fillna("NaN")
                df[c] = values.values if hasattr(values, "values") else values
                colormap, _ = _get_colormap(values, color_type, cmap)

                if color_type == "categorical":
                    color_dict = colormap
                    if _had_nan and nan_color is not None:
                        color_dict["NaN"] = nan_color
                    cmap_use = None
                else:
                    color_dict = None
                    cmap_use = colormap

                if tooltip_keys:
                    for tk in tooltip_keys:
                        if tk not in df.columns and tk in adata.obs.columns:
                            tk_vals = adata.obs[tk].values
                            if _nan_keep is not None:
                                tk_vals = tk_vals[_nan_keep]
                            df[tk] = tk_vals
            else:
                c = "_density"
                df[c] = 1
                color_type = "continuous"
                color_dict = None
                cmap_use = "viridis"

            plot_title = title or c

            if interactive_backend == "bokeh":
                p = _plot_interactive_bokeh(df, c, color_type, color_dict, cmap_use, plot_title,
                                           interactive_resolution, interactive_resolution)
            else:
                p = _plot_interactive_matplotlib(df, c, color_type, color_dict, cmap_use, plot_title,
                                                interactive_resolution, interactive_resolution)

            plots.append(p)

        return plots[0] if len(plots) == 1 else hv.Layout(plots).cols(ncols)

    # Static mode - requires datashader and matplotlib
    if not _check_datashader():
        raise ImportError(
            "datashader and matplotlib are required for static plots. "
            "Install with: pip install datashader matplotlib"
        )

    import datashader as ds
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from datashader.mpl_ext import dsshow

    if figsize is None:
        panel_size = 5
        nrows = (n_panels + ncols - 1) // ncols
        figsize = (min(n_panels, ncols) * panel_size + 2, nrows * panel_size)

    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    legend_figs = []

    for i, c in enumerate(color):
        ax = axes[i]
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})

        if c is not None:
            values, color_type = _get_color_values(adata, c)
            _had_nan = False
            if color_type == "categorical" and values.isna().any():
                _had_nan = True
                if nan_color is None:
                    keep = ~values.isna().values
                    df = df[keep]
                    values = values[keep]
                else:
                    if "NaN" not in values.cat.categories:
                        values = values.cat.add_categories(["NaN"])
                    values = values.fillna("NaN")
            df[c] = values.values if hasattr(values, "values") else values
            colormap, _ = _get_colormap(values, color_type, cmap)

            if color_type == "categorical":
                if _had_nan and nan_color is not None:
                    colormap["NaN"] = nan_color
                _plot_static_categorical(ax, df, c, colormap, point_size)
                legend_fig = _add_legend(ax, colormap, legend_mode, legend_max_categories, legend_entries_per_col)
                if legend_fig:
                    legend_figs.append(legend_fig)
            else:
                vmin_use, vmax_use = _get_vmin_vmax(df[c].values, vmin, vmax, vmax_percentile)
                _plot_static_continuous(ax, df, c, colormap, point_size, vmin_use, vmax_use)
                sm = plt.cm.ScalarMappable(
                    cmap=colormap,
                    norm=mcolors.Normalize(vmin=vmin_use, vmax=vmax_use)
                )
                plt.colorbar(sm, ax=ax, shrink=0.6)
        else:
            dsshow(df, ds.Point("x", "y"), ds.count(), cmap="viridis", ax=ax)
            c = "density"

        ax.set_title(title or c)
        ax.set_xlabel(f"{basis.replace('X_', '').upper()}1")
        ax.set_ylabel(f"{basis.replace('X_', '').upper()}2")
        ax.set_aspect("equal")

        x_range = df["x"].max() - df["x"].min()
        y_range = df["y"].max() - df["y"].min()
        margin = 0.05
        ax.set_xlim(df["x"].min() - margin * x_range, df["x"].max() + margin * x_range)
        ax.set_ylim(df["y"].min() - margin * y_range, df["y"].max() + margin * y_range)

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if show is None:
        show = save is None

    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=150, bbox_inches="tight")

        for j, leg_fig in enumerate(legend_figs):
            leg_path = save.parent / f"{save.stem}_legend_{j}{save.suffix}"
            leg_fig.savefig(leg_path, dpi=150, bbox_inches="tight")
            plt.close(leg_fig)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig
    return None


@with_insitupy_style
def umap(
    adata: ad.AnnData,
    color: str | Sequence[str] | None = None,
    **kwargs
) -> "plt.Figure | hv.Layout | jscatter.Scatter | list[jscatter.Scatter] | go.Figure | list[go.Figure] | None":
    """
    Fast UMAP plot using datashader for large datasets.

    Wrapper around embedding() with basis="X_umap".
    See embedding() for full parameter documentation.
    """
    return embedding(adata=adata, basis="X_umap", color=color, **kwargs)


def pca(
    adata: ad.AnnData,
    color: str | Sequence[str] | None = None,
    **kwargs
) -> "plt.Figure | hv.Layout | None":
    """
    Fast PCA plot using datashader.

    Wrapper around embedding() with basis="X_pca".
    See embedding() for full parameter documentation.
    """
    return embedding(adata=adata, basis="X_pca", color=color, **kwargs)


def tsne(
    adata: ad.AnnData,
    color: str | Sequence[str] | None = None,
    **kwargs
) -> "plt.Figure | hv.Layout | None":
    """
    Fast t-SNE plot using datashader.

    Wrapper around embedding() with basis="X_tsne".
    See embedding() for full parameter documentation.
    """
    return embedding(adata=adata, basis="X_tsne", color=color, **kwargs)
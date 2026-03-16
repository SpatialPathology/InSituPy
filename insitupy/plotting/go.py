import textwrap

import matplotlib.pyplot as plt
import numpy as np

from insitupy.plotting.save import save_and_show_figure
from insitupy.utils.utils import check_list, get_nrows_maxcols


def go_plot(
    enrichment,
    style="dot",
    color_key="Gene ratio",
    size_key=None, groups=None,
    fig=None, axis=None,
    max_to_plot=10,
    max_cols=4, cmap='viridis', cmin=None, cmax=None,
    ax_label_size=16, markersize=240,
    figsize=(8,6),
    remove_yticklabels=False,
    xtick_label_size=16, ytick_label_size=16,
    clb_label_size=16, clb_tick_label_size=16,
    clb_pos=None, clb_norm=False,
    title_size=16, max_line_length=30, custom_headers=None,
    max_name_length=60,
    value_to_plot='Enrichment score',
    colorcol=None,
    sortby=None, sort=True, additional_sortby=None,
    x_margin=None, y_margin=None,
    name_key='name', libraries=None, ascending=False,
    savepath=None, save_only=False, show=True, dpi_save=300
    ):
    """
    Plot GO / pathway enrichment results as dot, fixed-dot, or bar charts.

    Creates one subplot per group found in the first index level of ``enrichment``.
    Each subplot shows the top enriched terms ranked by ``value_to_plot``.

    Args:
        enrichment: Multi-index DataFrame of enrichment results. The first index
            level contains group names; columns must include ``'source'``,
            ``'native'``, and ``name_key``. Typically the output of
            :meth:`~insitupy.utils.go.GOEnrichment.gprofiler`.
        style (str, optional): Plot style. One of ``'dot'`` (scatter with colorbar),
            ``'dot_fixed'`` (dot size encodes score, no x-axis), or ``'bar'``
            (horizontal bar chart). Defaults to ``'dot'``.
        color_key (str, optional): Column used to color dots in ``'dot'`` and
            ``'dot_fixed'`` styles. Defaults to ``'Gene ratio'``.
        size_key (str, optional): Column used to scale dot size. Defaults to None
            (uniform size given by ``markersize``).
        groups (list or str, optional): Subset of groups (first index level) to
            plot. None plots all groups. Defaults to None.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw into.
            Defaults to None (a new figure is created).
        axis (matplotlib.axes.Axes, optional): Existing axis to draw into. If
            provided, only a single group may be present. Defaults to None.
        max_to_plot (int, optional): Maximum number of terms to show per group.
            Defaults to 10.
        max_cols (int, optional): Maximum number of subplot columns. Defaults to 4.
        cmap (str, optional): Colormap for dot color encoding. Defaults to
            ``'viridis'``.
        cmin (float, optional): Minimum value for colorbar normalization. Defaults
            to None (auto).
        cmax (float, optional): Maximum value for colorbar normalization. Defaults
            to None (auto).
        ax_label_size (int, optional): Font size for axis labels. Defaults to 16.
        markersize (int, optional): Dot size when ``size_key`` is None. Defaults
            to 240.
        figsize (tuple, optional): ``(width, height)`` of a single subplot panel.
            Total figure size scales with the number of panels. Defaults to
            ``(8, 6)``.
        remove_yticklabels (bool, optional): If True, hide y-axis tick labels.
            Defaults to False.
        xtick_label_size (int, optional): Font size for x-axis tick labels.
            Defaults to 16.
        ytick_label_size (int, optional): Font size for y-axis tick labels.
            Defaults to 16.
        clb_label_size (int, optional): Font size for the colorbar label. Defaults
            to 16.
        clb_tick_label_size (int, optional): Font size for colorbar tick labels.
            Defaults to 16.
        clb_pos (int, optional): Index of the subplot that receives the colorbar
            when using shared colorbars. None places the colorbar in each subplot.
            Defaults to None.
        clb_norm (bool, optional): If True, normalize the colorbar to the range
            ``[cmin, cmax]``. Defaults to False.
        title_size (int, optional): Font size for subplot titles. Defaults to 16.
        max_line_length (int, optional): Maximum line length (characters) before a
            line break is inserted in term names. Defaults to 30.
        custom_headers (list or str, optional): Custom subplot titles replacing the
            auto-generated ``'<group> <libraries>'`` header. Must match the number
            of plotted groups. Defaults to None.
        max_name_length (int, optional): Maximum characters of a term name before
            it is truncated and appended with the GO native ID. Defaults to 60.
        value_to_plot (str, optional): Column used as the x-axis (dot plot) or bar
            width. Defaults to ``'Enrichment score'``.
        colorcol (str, optional): Column whose values color the y-axis tick labels.
            Defaults to None.
        sortby (str, optional): Column used to sort terms within each group. Defaults
            to None (sorts by ``value_to_plot``).
        sort (bool, optional): If True, sort terms before plotting. Defaults to True.
        additional_sortby (str, optional): Secondary sort column applied after
            ``sortby``. Defaults to None.
        x_margin (float, optional): Fractional x-axis margin passed to
            ``ax.margins()``. Defaults to None.
        y_margin (float, optional): Fractional y-axis margin passed to
            ``ax.margins()``. Defaults to None.
        name_key (str, optional): Column containing the display name of each term.
            Defaults to ``'name'``.
        libraries (list or str, optional): Subset of GO/pathway libraries (from the
            ``'source'`` column) to include. None includes all. Defaults to None.
        ascending (bool, optional): Sort order for ``sortby``. Defaults to False
            (descending).
        savepath (str, optional): Path to save the figure. Defaults to None.
        save_only (bool, optional): If True, save without displaying. Defaults to
            False.
        show (bool, optional): If True, display the figure. If False, return the
            figure and axes. Defaults to True.
        dpi_save (int, optional): DPI for saved figure. Defaults to 300.

    Returns:
        None: Displays and/or saves the figure when ``show=True``.

        Tuple[matplotlib.figure.Figure, np.ndarray]: ``(fig, axs)`` when
            ``show=False``.
    """

    # get groups from index of enrichment dataframe
    extracted_groups = enrichment.index.unique(level=0).tolist()

    if groups is not None:
        groups = [groups] if isinstance(groups, str) else list(groups)

        groups = check_list(groups, extracted_groups)

        if len(groups) == 0:
            return
    else:
        groups = extracted_groups

    # check possible custom headers
    if custom_headers is not None:
        custom_headers = [custom_headers] if isinstance(custom_headers, str) else list(custom_headers)

    # check for libraries
    if libraries is None:
        libraries = list(enrichment['source'].unique())
    else:
        # convert libraries to list if they are string
        libraries = [libraries] if isinstance(libraries, str) else list(libraries)

        # check if all libraries are in dataframe
        libs_available = enrichment['source'].unique()
        notin = [elem for elem in libraries if elem not in libs_available]
        assert len(notin) == 0, (
            f"Following libraries could not be found in the `source` column: {notin}\n"
            f"Following libraries are possible: {libs_available}"
        )
        # filter for libraries
        enrichment = enrichment[enrichment['source'].isin(libraries)].copy()

    # sort dataframe
    if sortby is None:
        sortby = value_to_plot

    if sort:
        enrichment.sort_values(sortby, ascending=ascending, inplace=True)

    # shorten to max length if necessary
    if max_to_plot is not None:
        enrichment = enrichment.groupby(level=0).head(max_to_plot).copy()

    if additional_sortby is not None:
        enrichment.sort_values([additional_sortby, sortby], ascending=[True, False], inplace=True)
        #enrichment[name_key] = ["{} ({})".format(a,b) for a,b in zip(enrichment[name_key], enrichment[additional_sortby])]

    # Prepare names for plotting
    # Shorten name and add GO term name if too short
    if max_name_length is not None:
        enrichment[name_key] = ["{}...({})".format(n[:max_name_length], go) if len(n)>max_name_length else n for go, n in zip(enrichment['native'], enrichment[name_key])]
    # Insert line breaks if sentence too long
    #enrichment[name_key] = [nth_repl_all(elem, ' ', '\n', 1) for elem in enrichment[name_key]]

    # get min and max for the colorbar
    if color_key is not None:
        try:
            if clb_norm:
                cmax = enrichment[color_key].max() if cmax is None else cmax
                cmin = enrichment[color_key].min() if cmin is None else cmin
            else:
                cmax = None
                cmin = None
        except KeyError:
            print(f"color_key '{color_key}' not found in columns.")

    # Plotting
    if axis is None:
        n_plots, n_rows, n_cols = get_nrows_maxcols(len(groups), max_cols=max_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows))

    else:
        assert len(groups) == 1, "Enrichment dataframe contains more than one group in first index level 0."
        axs = axis
        n_plots = 1
        show = False

    axs = axs.ravel() if n_plots > 1 else [axs]

    for i, group in enumerate(groups):
        if group in enrichment.index.unique(level=0):
            # select data
            df = enrichment.xs(group).copy()

            # if max_to_plot is not None and len(df) > max_to_plot:
            #         df = df[:max_to_plot]
            if color_key is not None:
                try:
                    color = df[color_key]
                except KeyError:
                    print(f"color_key '{color_key}' not found in columns.")
                    color = 'k'
            else:
                color = 'k'

            if size_key is not None:
                markersize=df[size_key]*5

            if max_line_length is not None:
                # introduce line breaks if the names are too long
                df[name_key] = [textwrap.fill(elem, width=max_line_length, break_long_words=True) for elem in df[name_key]]

            # plotting
            if style == "dot":
                s = axs[i].scatter(
                    df[value_to_plot], df[name_key],
                    c=color, cmap=cmap,
                    s=markersize,
                    edgecolors='k')
            elif style == "dot_fixed":
                s = axs[i].scatter(
                    x = [0] * len(df[name_key]),
                    y = df[name_key],
                    s=df[value_to_plot] * 50,
                    c=color,
                    cmap=cmap,
                    edgecolors='k')
            elif style == "bar":
                ys = df[name_key]
                ys_pos = np.arange(len(ys))
                s = axs[i].barh(
                    ys_pos,
                    width=df[value_to_plot],
                    height=0.8,
                    color='k',
                    #cmap=cmap,
                    #s=markersize,
                    #edgecolors='k'
                )
                axs[i].set_yticks(ys_pos, labels=ys)
            else:
                raise ValueError('Invalid `style` parameter ("{}"). Possible options'.format(style))

            if custom_headers is None:
                axs[i].set_title("{}\n{}".format(group, libraries), fontsize=title_size)
            else:
                axs[i].set_title(custom_headers[i], fontsize=title_size)

            axs[i].invert_yaxis()
            axs[i].set_xlabel(value_to_plot, fontsize=ax_label_size)
            axs[i].tick_params(axis='x', which='major', labelsize=xtick_label_size)
            axs[i].tick_params(axis='y', which='major', labelsize=ytick_label_size)
            axs[i].margins(x=x_margin, y=y_margin)

            if style == "dot":
                axs[i].grid(axis='y')

            if style == "dot_fixed":
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
                axs[i].spines['left'].set_visible(False)
                axs[i].get_xaxis().set_ticks([])
                axs[i].set_xlabel("")

            if remove_yticklabels:
                axs[i].set_yticklabels([])

            # cbar_ax = fig.add_axes([0.95, 0.12, 0.02, 0.75])
            # clb = plt.colorbar(s, cax=cbar_ax, orientation='vertical')
            # clb.set_label('Fraction of patterned genes', loc='center')

            if style in ["dot", "dot_fixed"]:
                if color_key is not None:
                    if clb_pos is None:
                        clb = fig.colorbar(s, ax=axs[i], fraction=0.2, aspect=40, pad=0.15)
                        clb.set_label(color_key, loc='center', fontsize=clb_label_size)
                        clb.ax.tick_params(labelsize=clb_tick_label_size)
                        if clb_norm:
                            clb.mappable.set_clim(cmin, cmax)
                    else:
                        if i == clb_pos:
                            clb = fig.colorbar(s, ax=axs[clb_pos])
                            clb.set_label(color_key, loc='center', fontsize=clb_label_size)
                            clb.ax.tick_params(labelsize=clb_tick_label_size)
                            if clb_norm:
                                clb.mappable.set_clim(cmin, cmax)



            if size_key is not None:
                kw = dict(prop="sizes", num=5,
                    #color=s.cmap(0.7),
                    #fmt="$ {x:.2f}",
                    #func=lambda s: np.sqrt(s/.3)/3
                    #func=lambda s: np.sqrt(s)
                    )
                size_legend = axs[i].legend(*s.legend_elements(**kw, alpha=0.6),
                                    #markerscale=0.5,
                                    loc="lower right", title=size_key)

            if colorcol is not None:
                color_dict = {a: b for a, b in zip(df["name"], df[colorcol])}
                for xtick in axs[i].get_yticklabels():
                    xtick.set_color(color_dict[xtick.get_text()])
                # for xtick, color in zip(axs[i].get_xticklabels(), enrichment[colorcol]):
                #     xtick.set_color(color)


        else:
            #print('No significant results for selected libraries {} in group {}'.format(libraries, group))
            axs[i].set_title("{}\n{}".format(group, libraries), fontsize=12, fontweight='bold')
            axs[i].text(0.5,0.5, 'No significant results for selected\nlibraries {}\nin group {}'.format(libraries, group),
                        va='center', ha='center', fontsize=20)
            axs[i].set_axis_off()

    # check if there are empty plots remaining
    if n_plots > 1:
        while i < n_rows * n_cols - 1:
            i += 1
            # remove empty plots
            axs[i].set_axis_off()

        fig.tight_layout()

    fig.tight_layout()
    if show:
        save_and_show_figure(savepath=savepath, save_only=save_only, dpi_save=dpi_save, fig=fig)
    else:
        return fig, axs
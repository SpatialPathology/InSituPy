"""
Dual fold change comparison plots for differential gene expression analysis.

This module provides functions to create scatter plots comparing log2 fold changes
between main DGE results and neighborhood-based comparisons.
"""

import os
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from insitupy.dataclasses.results import (DiffExprConfigCollector,
                                          DiffExprResults)
from insitupy.plotting.config import _add_config_table
from insitupy.plotting.save import save_and_show_figure

# Constants
DEFAULT_LABEL_FONTSIZE = 12
DEFAULT_LABEL_FONTSIZE_NONSIG = 10
DEFAULT_POINT_SIZE_SIG = 20
DEFAULT_POINT_SIZE_NONSIG = 15
DEFAULT_POINT_ALPHA = 1.0
COLOR_SIGNIFICANT = 'black'
COLOR_NOT_SIGNIFICANT = 'gray'
AXIS_MARGIN_FACTOR = 0.1
MIN_PVALUE = 1e-300
TABLE_ROW_HEIGHT = 0.075


def dual_foldchange_plot(
    results: DiffExprResults,
    significance_threshold: Number = 0.05,
    foldchange_threshold: Number = 1,
    logfoldchanges_col: str = "log2foldchange",
    pval_col: str = "padj",
    patch_colors: List[str] = ["lightgreen", "lightcoral"],
    adjust_labels: bool = True,
    label_top_n: Union[int, Literal["all"]] = "all",
    label_sortby: str = "padj",
    size_by_pvalue: bool = True,
    size_range: Tuple[Number, Number] = (10, 40),
    show_nonsignificant: bool = True,
    show_config: bool = False,
    figsize: Tuple[Number, Number] = (6, 6),
    savepath: Union[str, os.PathLike, Path, None] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    show: bool = True,
) -> None:
    """
    Create scatter plots comparing log2 fold changes between main DGE results
    and neighborhood-based comparisons.

    This function generates two side-by-side plots showing how genes upregulated
    or downregulated in the main comparison behave in their respective neighborhood
    comparisons.

    Parameters
    ----------
    results : DiffExprResults
        Result container holding main and neighborhood differential expression data.
        Must include both target_neighborhood and ref_neighborhood.
    significance_threshold : Number, default=0.05
        Adjusted p-value threshold for significance in the main comparison.
    foldchange_threshold : Number, default=1
        Minimum absolute fold change (not log-transformed) to include genes in the plot.
        Genes must exceed this threshold in the main comparison to be plotted.
    logfc_col : str, default='log2foldchange'
        Column name for log2 fold change values.
    pval_col : str, default='padj'
        Column name for adjusted p-values.
    patch_colors : list of str, default=['lightgreen', 'lightcoral']
        Background colors for [positive, negative] y-axis regions, indicating
        whether genes are higher or lower in the neighborhood.
    adjust_labels : bool, default=True
        If True, automatically adjust overlapping gene labels using adjustText.
    label_top_n : int or 'all', default='all'
        Number of top genes to label based on label_sortby.
        If 'all', labels all genes in the plot.
        If int, labels the top N most significant genes.
    label_sortby : str, default='padj'
        Column name to use for ranking genes when label_top_n is an integer.
        Typically 'padj' (ascending) or 'log2foldchange' (absolute value).
    size_by_pvalue : bool, default=False
        If True, scale point sizes based on -log10(padj) values.
        More significant genes (lower p-values) will have larger points.
    size_range : tuple of Number, default=(10, 40)
        Range of point sizes (min, max) when size_by_pvalue is True.
        Ignored if size_by_pvalue is False.
    show_nonsignificant : bool, default=True
        If True, plots both significant and non-significant genes.
        If False, only plots significant genes (padj < significance_threshold).
    show_config : bool, default=False
        If True, displays a configuration table below the plots showing analysis
        parameters and DEG counts.
    figsize : tuple of Number, default=(6, 6)
        Size of each subplot in inches (width, height).
        Total figure width will be figsize[0] * 2.
    savepath : str, os.PathLike, Path, optional
        Path to save the plot. If None, plot is not saved.
    save_only : bool, default=False
        If True, saves plot without displaying it.
    dpi_save : int, default=300
        Resolution in dots per inch for saved figure.
    show : bool, default=True
        If True, displays the plot after creation.

    Returns
    -------
    None
        Displays and/or saves the plot as specified.

    Raises
    ------
    ValueError
        If results object has no neighborhood comparisons or if parameters are invalid.
    KeyError
        If required columns are missing from the DataFrames.

    Notes
    -----
    The function creates two subplots:
    - Left: Downregulated genes in main comparison vs. their neighborhood behavior
    - Right: Upregulated genes in main comparison vs. their neighborhood behavior

    The y-axis shows log2 fold change in the neighborhood comparison:
    - Positive values: gene is higher in the cell type than in the neighborhood
    - Negative values: gene is lower in the cell type than in the neighborhood

    Background shading helps interpret the biological meaning:
    - Green region (y > 0): genes enriched in cell type vs. neighborhood
    - Red region (y < 0): genes depleted in cell type vs. neighborhood

    When size_by_pvalue is True, point sizes are scaled by -log10(padj):
    - Larger points = more significant (smaller p-values)
    - Smaller points = less significant (larger p-values)

    Examples
    --------
    >>> # Label all genes with default point sizes
    >>> dual_foldchange_plot(
    ...     results=deg_results,
    ...     significance_threshold=0.01,
    ...     foldchange_threshold=1.5,
    ...     label_top_n="all"
    ... )

    >>> # Show only significant genes with p-value-scaled sizes
    >>> dual_foldchange_plot(
    ...     results=deg_results,
    ...     label_top_n=20,
    ...     size_by_pvalue=True,
    ...     size_range=(20, 150),
    ...     show_nonsignificant=False
    ... )
    """
    # Input validation
    if not results.has_neighbors():
        raise ValueError("The provided results object has no neighbor comparisons.")

    if label_top_n != "all" and (not isinstance(label_top_n, int) or label_top_n < 0):
        raise ValueError(f"label_top_n must be 'all' or a non-negative integer, got {label_top_n}")

    df_main = results.main
    df_nb_target = results.target_neighborhood
    df_nb_ref = results.ref_neighborhood

    for df, name in [(df_main, "main"), (df_nb_target, "target_neighborhood"),
                     (df_nb_ref, "ref_neighborhood")]:
        if logfoldchanges_col not in df.columns:
            raise KeyError(f"Column '{logfoldchanges_col}' not found in {name} DataFrame")
        if pval_col not in df.columns:
            raise KeyError(f"Column '{pval_col}' not found in {name} DataFrame")

    logfc_threshold = np.log2(foldchange_threshold)

    filtered_data_x_up = df_main[df_main[logfoldchanges_col] >= logfc_threshold].copy()
    filtered_data_y_up = df_nb_target[df_nb_target.index.isin(filtered_data_x_up.index)].copy()

    filtered_data_x_down = df_main[df_main[logfoldchanges_col] <= -logfc_threshold].copy()
    filtered_data_y_down = df_nb_ref[df_nb_ref.index.isin(filtered_data_x_down.index)].copy()

    # determine subplot layout
    ncols = 3 if show_config else 2
    fig, axs = plt.subplots(1, ncols, figsize=(figsize[0] * ncols, figsize[1]))

    # If only 2 axes, convert to list for consistent indexing
    if ncols == 2:
        axs = list(axs)
    else:
        axs = list(axs)
        ax_config = axs[2]

    xs = [filtered_data_x_down, filtered_data_x_up]
    ys = [filtered_data_y_down, filtered_data_y_up]
    fcs = [-logfc_threshold, logfc_threshold]
    titles = ["Downregulated", "Upregulated"]

    for i, (x, y, fc, t) in enumerate(zip(xs, ys, fcs, titles)):
        if len(x) > 0 and len(y) > 0:
            _plot_single_nb_plot(
                df_main=x,
                df_neighbor=y,
                logfc_col=logfoldchanges_col,
                pval_col=pval_col,
                fold_change_threshold=fc,
                significance_threshold=significance_threshold,
                patch_colors=patch_colors,
                label_top_n=label_top_n,
                label_sortby=label_sortby,
                adjust_labels=adjust_labels,
                size_by_pvalue=size_by_pvalue,
                size_range=size_range,
                show_nonsignificant=show_nonsignificant,
                config=None,
                n_upreg=None,
                n_downreg=None,
                ax=axs[i],
                show_legend=(i == 1),
                title=t,
                ylabel="Paired Log2FoldChange\nwith neighborhood" if i == 0 else None,
            )
        else:
            axs[i].text(0.5, 0.5, "No genes passed the threshold",
                        ha="center", va="center", fontsize=14)
            axs[i].set_xticks([])
            axs[i].set_yticks([])

    # Add config table to the right panel
    if show_config:
        n_upreg = np.sum((df_main[pval_col] <= significance_threshold) &
                         (df_main[logfoldchanges_col] > logfc_threshold))
        n_downreg = np.sum((df_main[pval_col] <= significance_threshold) &
                           (df_main[logfoldchanges_col] < -logfc_threshold))
        _add_config_table(
            config=results.config,
            ax=ax_config,
            position="inside",
            n_upreg=n_upreg, n_downreg=n_downreg)

    plt.tight_layout()

    save_and_show_figure(
        savepath=savepath,
        fig=fig,
        save_only=save_only,
        dpi_save=dpi_save,
        show=show
    )


def _plot_single_nb_plot(
    df_main: pd.DataFrame,
    df_neighbor: pd.DataFrame,
    logfc_col: str,
    pval_col: str,
    fold_change_threshold: Number,
    significance_threshold: Number,
    patch_colors: List[str],
    label_top_n: Union[int, Literal["all"]],
    label_sortby: str,
    adjust_labels: bool,
    size_by_pvalue: bool,
    size_range: Tuple[Number, Number],
    show_nonsignificant: bool,
    config: Optional[DiffExprConfigCollector],
    n_upreg: Optional[int],
    n_downreg: Optional[int],
    ax: plt.Axes,
    show_legend: bool,
    xlabel: Optional[str] = "Log2FoldChange",
    ylabel: Optional[str] = "Log2FoldChange (with neighborhood)",
    title: Optional[str] = None
) -> None:
    """
    Helper function for plotting one dual fold change comparison subplot.

    Parameters
    ----------
    df_main : pd.DataFrame
        Main comparison data (already filtered by fold change threshold).
    df_neighbor : pd.DataFrame
        Neighborhood comparison data for the same genes.
    logfc_col : str
        Column name for log2 fold change values.
    pval_col : str
        Column name for adjusted p-values.
    fold_change_threshold : Number
        Log2 fold change threshold (already transformed).
    significance_threshold : Number
        Adjusted p-value threshold for significance.
    patch_colors : list of str
        Background colors for [positive, negative] regions.
    label_top_n : int or 'all'
        Number of top genes to label, or 'all' for all genes.
    label_sortby : str
        Column to sort by for selecting top genes.
    adjust_labels : bool
        Whether to adjust overlapping labels.
    size_by_pvalue : bool
        Whether to scale point sizes by -log10(padj).
    size_range : tuple of Number
        Range of point sizes (min, max) when size_by_pvalue is True.
    show_nonsignificant : bool
        Whether to plot non-significant genes.
    config : DiffExprConfigCollector, optional
        Configuration object containing analysis metadata to display in table below plot.
    n_upreg : int, optional
        Number of significantly upregulated genes.
    n_downreg : int, optional
        Number of significantly downregulated genes.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    show_legend : bool
        Whether to show the legend.
    show_ylabel : bool
        Whether to show the y-axis label.

    Returns
    -------
    None
        Plots on the provided axes.
    """
    # CRITICAL: Merge dataframes to ensure x and y values are aligned by gene
    # Keep only genes present in both dataframes
    common_genes = df_main.index.intersection(df_neighbor.index)
    df_main_aligned = df_main.loc[common_genes]
    df_neighbor_aligned = df_neighbor.loc[common_genes]

    # Extract values for plotting - now guaranteed to be aligned
    x_values = df_main_aligned[logfc_col]
    y_values = df_neighbor_aligned[logfc_col]
    p_values = df_main_aligned[pval_col]
    genes = df_main_aligned.index

    # Determine significance
    sig = p_values < significance_threshold

    # Filter data if show_nonsignificant is False
    if not show_nonsignificant:
        # Keep only significant genes
        x_values = x_values[sig]
        y_values = y_values[sig]
        p_values = p_values[sig]
        genes = genes[sig]
        # All remaining genes are significant
        sig = pd.Series(True, index=genes)

        if len(genes) == 0:
            # No significant genes to plot
            ax.text(
                0.5, 0.5,
                "No significant genes",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

    # Calculate point sizes
    if size_by_pvalue:
        # Clip p-values to avoid log(0) and very extreme values
        p_values_clipped = p_values.clip(lower=MIN_PVALUE, upper=1.0)
        neg_log_pvals = -np.log10(p_values_clipped)

        # Normalize to size_range
        min_size, max_size = size_range
        pval_min = neg_log_pvals.min()
        pval_max = neg_log_pvals.max()

        if pval_max > pval_min:
            # Scale linearly from min_size to max_size
            sizes = min_size + (neg_log_pvals - pval_min) / (pval_max - pval_min) * (max_size - min_size)
        else:
            # All p-values are the same, use middle of range
            sizes = np.full(len(p_values), (min_size + max_size) / 2)

        sizes_sig = sizes[sig]
        sizes_nonsig = sizes[~sig]
    else:
        # Use default fixed sizes
        sizes_sig = DEFAULT_POINT_SIZE_SIG
        sizes_nonsig = DEFAULT_POINT_SIZE_NONSIG

    # Calculate axis limits with margin
    xmin, xmax = x_values.min(), x_values.max()
    ymin, ymax = y_values.min(), y_values.max()
    xmincorr = xmin - AXIS_MARGIN_FACTOR * abs(xmin)
    xmaxcorr = xmax + AXIS_MARGIN_FACTOR * abs(xmax)
    ymincorr = ymin - AXIS_MARGIN_FACTOR * abs(ymin)
    ymaxcorr = ymax + AXIS_MARGIN_FACTOR * abs(ymax)

    ax.set_xlim(xmincorr, xmaxcorr)
    ax.set_ylim(ymincorr, ymaxcorr)

    # Add background patches
    ax.axhspan(0, ymaxcorr, facecolor=patch_colors[0], alpha=0.3)
    ax.axhspan(ymincorr, 0, facecolor=patch_colors[1], alpha=0.3)

    # Scatter significant and non-significant points
    if show_nonsignificant:
        # Plot both significant and non-significant genes
        ax.scatter(
            x_values[sig],
            y_values[sig],
            c=COLOR_SIGNIFICANT,
            label="significant",
            alpha=DEFAULT_POINT_ALPHA,
            s=sizes_sig
        )
        ax.scatter(
            x_values[~sig],
            y_values[~sig],
            c=COLOR_NOT_SIGNIFICANT,
            label="not significant",
            alpha=DEFAULT_POINT_ALPHA,
            s=sizes_nonsig
        )
    else:
        # Only plot significant genes (all genes at this point are significant)
        ax.scatter(
            x_values,
            y_values,
            c=COLOR_SIGNIFICANT,
            label="significant",
            alpha=DEFAULT_POINT_ALPHA,
            s=sizes_sig
        )

    # Add reference lines
    objects_to_avoid = []
    objects_to_avoid.append(
        ax.axhline(0, color=COLOR_NOT_SIGNIFICANT, linestyle="--", linewidth=1)
    )
    objects_to_avoid.append(
        ax.axhline(-1, color=COLOR_NOT_SIGNIFICANT, linestyle="--", linewidth=1)
    )
    objects_to_avoid.append(
        ax.axvline(
            fold_change_threshold,
            color=COLOR_NOT_SIGNIFICANT,
            linestyle="--",
            linewidth=1
        )
    )

    # Select genes to label
    if label_top_n == "all":
        genes_to_label = genes
    else:
        # Need to work with a dataframe for sorting
        if show_nonsignificant:
            df_for_selection = df_main_aligned
        else:
            # Reconstruct aligned dataframe with only significant genes
            df_for_selection = pd.DataFrame({
                logfc_col: x_values,
                pval_col: p_values
            }, index=genes)

        # Validate label_sortby column
        if label_sortby not in df_for_selection.columns:
            warn(
                f"The specified label_sortby column '{label_sortby}' does not exist. "
                f"Using '{pval_col}' instead.",
                UserWarning
            )
            label_sortby = pval_col

        # Select top N genes
        if label_sortby == pval_col:
            # For p-values, sort ascending (smallest p-values first)
            top_genes_df = df_for_selection.nsmallest(
                min(label_top_n, len(df_for_selection)),
                label_sortby
            )
        else:
            # For other columns (e.g., log2foldchange), sort by absolute value
            df_with_abs = df_for_selection.copy()
            df_with_abs['_abs_sort'] = df_with_abs[label_sortby].abs()
            top_genes_df = df_with_abs.nlargest(
                min(label_top_n, len(df_for_selection)),
                '_abs_sort'
            )

        genes_to_label = top_genes_df.index

    # Add text labels for selected genes
    texts = []
    for gene in genes_to_label:
        x = x_values.loc[gene]
        y = y_values.loc[gene]
        pval = p_values.loc[gene]

        is_sig = pval < significance_threshold
        color = COLOR_SIGNIFICANT if is_sig else COLOR_NOT_SIGNIFICANT
        style = "normal" if is_sig else "oblique"
        fontsize = DEFAULT_LABEL_FONTSIZE if is_sig else DEFAULT_LABEL_FONTSIZE_NONSIG

        t = ax.text(
            x, y, gene,
            fontsize=fontsize,
            color=color,
            fontstyle=style
        )
        texts.append(t)

    # Adjust label overlap
    if adjust_labels and texts:
        adjust_text(
            texts,
            ax=ax,
            objects=objects_to_avoid,
            only_move={"text": "xy"},
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    # Add axis labels and legend
    if show_legend:
        if size_by_pvalue:
            # Create custom legend with size information
            from matplotlib.lines import Line2D

            # Legend elements for color
            legend_elements = []
            if show_nonsignificant:
                legend_elements.extend([
                    Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COLOR_SIGNIFICANT, markersize=8,
                       label=f'high'),
                    Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COLOR_SIGNIFICANT, markersize=5,
                       label=f'low'),
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=COLOR_NOT_SIGNIFICANT, markersize=5,
                           label='no')
                ])
            else:
                legend_elements.extend([
                    Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COLOR_SIGNIFICANT, markersize=8,
                       label=f'high'),
                    Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COLOR_SIGNIFICANT, markersize=5,
                       label=f'low')
                ])

            # # Add size information
            # min_size, max_size = size_range
            # legend_elements.extend([
            #     Line2D([0], [0], marker='o', color='w',
            #            markerfacecolor='k', markersize=8,
            #            label=f'high sig.'),
            #     Line2D([0], [0], marker='o', color='w',
            #            markerfacecolor='gray', markersize=4,
            #            label=f'low sig.')
            # ])

            ax.legend(
                handles=legend_elements,
                title=f"significance",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )
        else:
            # Standard legend without size information
            ax.legend(
                title=f"{pval_col} (target vs. reference)",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    # Add configuration table if provided
    if config is not None and n_upreg is not None and n_downreg is not None:
        _add_config_table(config, n_upreg, n_downreg, ax)



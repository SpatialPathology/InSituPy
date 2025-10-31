import os
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.font_manager import FontProperties

from insitupy.dataclasses.results import (DiffExprConfigCollector,
                                          DiffExprResults)
from insitupy.plotting.config import _add_config_table
from insitupy.plotting.save import save_and_show_figure

# Constants
DEFAULT_ALPHA = 0.5
DEFAULT_LABEL_FONTSIZE = 14
DEFAULT_AXIS_FONTSIZE = 14
DEFAULT_TITLE_FONTSIZE = 16
AXIS_MARGIN_FACTOR = 1.1
COLOR_UPREGULATED = 'maroon'
COLOR_DOWNREGULATED = 'royalblue'
COLOR_NOT_SIGNIFICANT = 'black'
MIN_PVALUE = 1e-300


def single_volcano(
    data: pd.DataFrame,
    logfoldchanges_column: str = 'log2foldchange',
    pval_column: str = 'padj',
    significance_threshold: Number = 0.05,
    foldchange_threshold: Number = 2,
    title: Optional[str] = None,
    adjust_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    savepath: Union[str, os.PathLike, Path, None] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    show: bool = True,
    genes_to_label: Union[int, List[str]] = 20,
    label_sortby: str = "log2foldchange",
    figsize: Tuple[int, int] = (8, 6),
    config: Optional[DiffExprConfigCollector] = None,
    up_label: Optional[str] = "Target",
    down_label: Optional[str] = "Reference"
) -> None:
    """
    Create a volcano plot with labeled top differentially expressed genes.

    This function generates a volcano plot displaying log2 fold changes vs. -log10
    adjusted p-values, with automatic labeling of the most significant genes and
    customizable thresholds for significance.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing gene expression data with genes as index.
        Must contain columns for log fold changes and p-values.
    logfoldchanges_column : str, default='log2foldchange'
        Name of the column containing log2 fold change values.
    pval_column : str, default='padj'
        Name of the column containing adjusted p-values.
    significance_threshold : Number, default=0.05
        Adjusted p-value threshold below which genes are considered significant.
    foldchange_threshold : Number, default=2
        Absolute fold change threshold (not log-transformed) for biological significance.
        Will be log2-transformed internally.
    title : str, optional
        Title for the plot. If None, no title is displayed.
    adjust_labels : bool, default=True
        If True, automatically adjust label positions to minimize overlap.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates new figure.
    savepath : str, os.PathLike, Path, optional
        Path to save the plot. If None, plot is not saved.
    save_only : bool, default=False
        If True, saves plot without displaying it.
    dpi_save : int, default=300
        Resolution in dots per inch for saved figure.
    show : bool, default=True
        If True, displays the plot. Ignored if ax is provided.
    label_top_n : int or list of str, default=20
        If int: number of top up- and down-regulated genes to label.
        If list: specific gene names to label (if present in significant genes).
    label_sortby : str, default='log2foldchange'
        Column name to use for sorting genes when selecting top N to label.
        Falls back to logfoldchanges_column if specified column doesn't exist.
    figsize : tuple of int, default=(8, 6)
        Figure size in inches (width, height). Only used if ax is None.
    config : DiffExprConfigCollector, optional
        Configuration object containing analysis metadata to display in table below plot.
    up_label : str, optional, default='Target'
        Label for upregulated genes section. Set to None to hide.
    down_label : str, optional, default='Reference'
        Label for downregulated genes section. Set to None to hide.

    Returns
    -------
    None
        Displays and/or saves the plot as specified.

    Raises
    ------
    ValueError
        If data is empty or thresholds are invalid.
    KeyError
        If required columns are missing from the DataFrame.

    Notes
    -----
    - Genes are colored as:
        * Maroon: significantly upregulated (padj < threshold, log2FC > threshold)
        * Royal blue: significantly downregulated (padj < threshold, log2FC < -threshold)
        * Black: not significant
    - P-values are clipped at 1e-300 to avoid log(0) errors.
    - Axis limits are automatically adjusted to show all significant genes plus margin.

    Examples
    --------
    >>> single_volcano(
    ...     data=deg_results,
    ...     significance_threshold=0.01,
    ...     foldchange_threshold=1.5,
    ...     label_top_n=10,
    ...     title="Treatment vs Control"
    ... )
    """
    # Input validation
    if data.empty:
        raise ValueError("Input data is empty")

    if logfoldchanges_column not in data.columns:
        raise KeyError(f"Column '{logfoldchanges_column}' not found in data")

    if pval_column not in data.columns:
        raise KeyError(f"Column '{pval_column}' not found in data")

    if significance_threshold <= 0 or significance_threshold >= 1:
        raise ValueError(f"significance_threshold must be between 0 and 1, got {significance_threshold}")

    if foldchange_threshold <= 0:
        raise ValueError(f"foldchange_threshold must be positive, got {foldchange_threshold}")

    # Copy data to avoid modifying original DataFrame
    data = data.copy()

    # Validate label_sortby column
    if label_sortby not in data.columns:
        warn(
            f"The specified label_sortby column '{label_sortby}' does not exist. "
            f"Using '{logfoldchanges_column}' instead.",
            UserWarning
        )
        label_sortby = logfoldchanges_column

    # Prepare data: clip p-values to avoid log(0)
    data[pval_column] = data[pval_column].clip(lower=MIN_PVALUE)

    # Transform p-values to -log10 scale
    neg_log_pval_column = "neg_log10_pvals"
    data[neg_log_pval_column] = -np.log10(data[pval_column])
    neg_log_sig_thresh = -np.log10(significance_threshold)
    lfc_threshold = np.log2(foldchange_threshold)

    # Determine colors based on significance and fold change
    is_significant = data[neg_log_pval_column] > neg_log_sig_thresh
    is_upregulated = data[logfoldchanges_column] > lfc_threshold
    is_downregulated = data[logfoldchanges_column] < -lfc_threshold

    colors = np.where(
        is_significant & is_upregulated,
        COLOR_UPREGULATED,
        np.where(
            is_significant & is_downregulated,
            COLOR_DOWNREGULATED,
            COLOR_NOT_SIGNIFICANT
        )
    )

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        show = False  # Don't show if ax is provided

    # Create scatter plot
    ax.scatter(
        data[logfoldchanges_column],
        data[neg_log_pval_column],
        alpha=DEFAULT_ALPHA,
        color=colors
    )

    # Add title and labels
    if title is not None:
        ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    ax.set_xlabel('$\\mathregular{Log_2}$ fold change', fontsize=DEFAULT_AXIS_FONTSIZE)
    ax.set_ylabel('$\\mathregular{-Log_{10}}$ adjusted p-value', fontsize=DEFAULT_AXIS_FONTSIZE)

    # Add threshold lines
    ax.axhline(y=neg_log_sig_thresh, color=COLOR_NOT_SIGNIFICANT, linestyle='--')
    ax.axvline(x=lfc_threshold, color=COLOR_NOT_SIGNIFICANT, linestyle='--')
    ax.axvline(x=-lfc_threshold, color=COLOR_NOT_SIGNIFICANT, linestyle='--')

    # Identify significant genes
    sig_mask = data[neg_log_pval_column] > neg_log_sig_thresh
    up_mask = (data[logfoldchanges_column] > lfc_threshold) & sig_mask
    down_mask = (data[logfoldchanges_column] < -lfc_threshold) & sig_mask

    up_data = data[up_mask]
    down_data = data[down_mask]

    # Select genes to label
    if isinstance(genes_to_label, int):
        top_up_genes = up_data.nlargest(genes_to_label, label_sortby) if not up_data.empty else pd.DataFrame()
        top_down_genes = down_data.nsmallest(genes_to_label, label_sortby) if not down_data.empty else pd.DataFrame()
    elif isinstance(genes_to_label, list):
        top_up_genes = up_data[up_data.index.isin(genes_to_label)] if not up_data.empty else pd.DataFrame()
        top_down_genes = down_data[down_data.index.isin(genes_to_label)] if not down_data.empty else pd.DataFrame()
    else:
        raise TypeError(f"label_top_n must be int or list, got {type(genes_to_label)}")

    # Calculate axis limits with margins
    if not down_data.empty:
        xmin = min(
            down_data[logfoldchanges_column].min() * AXIS_MARGIN_FACTOR,
            -(lfc_threshold * AXIS_MARGIN_FACTOR)
        )
    else:
        xmin = -(lfc_threshold * AXIS_MARGIN_FACTOR)

    if not up_data.empty:
        xmax = max(
            up_data[logfoldchanges_column].max() * AXIS_MARGIN_FACTOR,
            lfc_threshold * AXIS_MARGIN_FACTOR
        )
        ymax = max(
            up_data[neg_log_pval_column].max() * AXIS_MARGIN_FACTOR,
            down_data[neg_log_pval_column].max() * AXIS_MARGIN_FACTOR if not down_data.empty else 0,
            neg_log_sig_thresh * AXIS_MARGIN_FACTOR
        )
    else:
        xmax = lfc_threshold * AXIS_MARGIN_FACTOR
        ymax = (
            down_data[neg_log_pval_column].max() * AXIS_MARGIN_FACTOR
            if not down_data.empty
            else neg_log_sig_thresh * AXIS_MARGIN_FACTOR
        )

    # Set axis limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)

    # Annotate top genes
    top_genes = pd.concat([top_up_genes, top_down_genes])
    texts = []
    for gene, row in top_genes.iterrows():
        texts.append(
            ax.annotate(
                gene,
                (row[logfoldchanges_column], row[neg_log_pval_column]),
                fontsize=DEFAULT_LABEL_FONTSIZE,
                alpha=0.75
            )
        )

    # Adjust text positions to avoid overlap
    if adjust_labels and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            max_move=None
        )

    # Add directional labels
    if up_label is not None:
        ax.annotate(
            up_label,
            xy=(1, 1.04),
            xycoords='axes fraction',
            xytext=(0.95, 1.04),
            textcoords='axes fraction',
            ha='right',
            va='center',
            fontsize=DEFAULT_LABEL_FONTSIZE,
            color=COLOR_NOT_SIGNIFICANT,
            arrowprops=dict(arrowstyle='->', color=COLOR_NOT_SIGNIFICANT)
        )

    if down_label is not None:
        ax.annotate(
            down_label,
            xy=(0, 1.04),
            xycoords='axes fraction',
            xytext=(0.05, 1.04),
            textcoords='axes fraction',
            ha='left',
            va='center',
            fontsize=DEFAULT_LABEL_FONTSIZE,
            color=COLOR_NOT_SIGNIFICANT,
            arrowprops=dict(arrowstyle='->', color=COLOR_NOT_SIGNIFICANT)
        )

    # Add configuration table if provided
    if config is not None:
        n_upreg = np.sum((data[pval_column] <= significance_threshold) &
                        (data[logfoldchanges_column] > lfc_threshold))
        n_downreg = np.sum((data[pval_column] <= significance_threshold) &
                          (data[logfoldchanges_column] < -lfc_threshold))

        _add_config_table(config, n_upreg, n_downreg, ax)

    # Save and show figure
    save_and_show_figure(
        savepath=savepath,
        fig=plt.gcf(),
        save_only=save_only,
        dpi_save=dpi_save,
        show=show
    )


def volcano(
    results: DiffExprResults,
    significance_threshold: Number = 0.05,
    foldchange_threshold: Number = 2,
    label_top_n: int = 20,
    label_sortby: str = "log2foldchange",
    figsize: Tuple[Number, Number] = (6, 6),
    show: bool = True,
    show_config: bool = False,
    title: Optional[str] = None,
    savepath: Union[str, os.PathLike, Path, None] = None,
    save_only: bool = False,
    dpi_save: int = 300
) -> None:
    """
    Generate multi-panel volcano plots for differential expression results.

    Creates a comprehensive visualization of differential gene expression analysis,
    including the main comparison and optional neighborhood comparisons if available
    in the results object.

    Parameters
    ----------
    results : DiffExprResults
        Container object holding differential expression results with:
        - main: primary comparison results
        - target_neighborhood: target neighborhood comparison (optional)
        - ref_neighborhood: reference neighborhood comparison (optional)
    significance_threshold : Number, default=0.05
        Adjusted p-value threshold for statistical significance.
    foldchange_threshold : Number, default=2
        Minimum absolute fold change (not log-transformed) for biological significance.
    label_top_n : int, default=20
        Number of top differentially expressed genes to label in each plot.
    label_sortby : str, default='log2foldchange'
        Column name used to rank genes for labeling.
        Must be present in the results DataFrame.
    figsize : tuple of Number, default=(6, 6)
        Size of each subplot in inches (width, height).
        Total figure width will be figsize[0] * number_of_panels.
    show : bool, default=True
        If True, displays the plot after creation.
    show_config : bool, default=False
        If True, displays a configuration table below the main plot
        showing analysis parameters and DEG counts.
    title : str, optional
        Overall title for the figure. If None, no title is displayed.
    savepath : str, os.PathLike, Path, optional
        Path to save the figure. If None, figure is not saved.
    save_only : bool, default=False
        If True, saves the figure without displaying it (overrides show).
    dpi_save : int, default=300
        Resolution in dots per inch for saved figure.

    Returns
    -------
    None
        Displays and/or saves the multi-panel volcano plot as specified.

    Raises
    ------
    ValueError
        If parameters are out of valid range.

    Notes
    -----
    The function creates 1-3 subplots depending on available data:
    - Panel 1 (always present): Main comparison volcano plot
    - Panel 2 (if available): Target neighborhood comparison
    - Panel 3 (if available): Reference neighborhood comparison

    Neighborhood comparisons show genes that are downregulated in the neighborhood
    compared to the cell type of interest (log2FC < 0), using adjusted p-values
    with a fold change threshold of 1 (no fold change filtering beyond significance).

    Examples
    --------
    >>> volcano(
    ...     results=deg_results,
    ...     significance_threshold=0.01,
    ...     foldchange_threshold=1.5,
    ...     label_top_n=15,
    ...     show_config=True,
    ...     title="Cell Type A vs Cell Type B"
    ... )

    >>> # Save without displaying
    >>> volcano(
    ...     results=deg_results,
    ...     savepath="volcano_plots.png",
    ...     save_only=True,
    ...     dpi_save=600
    ... )
    """
    # Determine number of panels needed
    ncols = 3 if results.has_neighbors() else 1

    # Create figure with appropriate layout
    fig, axs = plt.subplots(1, ncols, figsize=(figsize[0] * ncols, figsize[1]))
    if ncols == 1:
        axs = [axs]  # Make iterable for consistent indexing

    config = results.config if show_config else None

    # Plot main comparison
    single_volcano(
        data=results.main,
        logfoldchanges_column='log2foldchange',
        pval_column='padj',
        foldchange_threshold=foldchange_threshold,
        significance_threshold=significance_threshold,
        genes_to_label=label_top_n,
        label_sortby=label_sortby,
        config=config,
        down_label="Reference",
        up_label="Target",
        ax=axs[0]
    )

    axs[0].set_title("Target vs. Reference", pad=30, fontsize=18)

    # Plot neighborhood comparisons if available
    if results.has_neighbors():
        neighborhood_data = [
            results.target_neighborhood,
            results.ref_neighborhood
        ]

        for idx, data in enumerate(neighborhood_data, start=1):
            # Filter for downregulated genes in neighborhood
            # (genes lower in neighborhood vs. cell type)
            filtered_data = data[data["log2foldchange"] < 0].copy()

            if filtered_data.empty:
                warn(
                    f"No downregulated genes in neighborhood comparison {idx}. "
                    "Panel will be empty.",
                    UserWarning
                )

            single_volcano(
                data=filtered_data,
                logfoldchanges_column='log2foldchange',
                pval_column='padj',
                foldchange_threshold=1,  # No FC threshold for neighborhoods
                significance_threshold=significance_threshold,
                genes_to_label=label_top_n,
                label_sortby=label_sortby,
                config=None,  # Don't show config on neighborhood plots
                down_label="Neighborhood",
                up_label=None,
                ax=axs[idx]
            )

        axs[1].set_title("Target vs. Neighborhood", pad=30, fontsize=18)
        axs[2].set_title("Reference vs. Neighborhood", pad=30, fontsize=18)

    # Add overall title if provided
    if title is not None:
        plt.suptitle(title, fontsize=DEFAULT_TITLE_FONTSIZE)

    # Save and show figure
    save_and_show_figure(
        savepath=savepath,
        fig=fig,
        save_only=save_only,
        dpi_save=dpi_save,
        show=show,
        tight=True
    )

# Deprecated functions
def plot_volcano(*args, **kwargs) -> None:
    """
    Deprecated: Use single_volcano() or volcano() instead.

    This function is maintained for backward compatibility but will be removed
    in a future version.

    Raises
    ------
    DeprecationWarning
        Always raised when this function is called.
    """
    warn(
        "plot_volcano() is deprecated and will be removed in a future version. "
        "Use single_volcano() for single plots or volcano() for multi-panel plots instead.",
        DeprecationWarning,
        stacklevel=2
    )


import os
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from warnings import warn

import decoupler as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.font_manager import FontProperties

from insitupy._checks import try_import
from insitupy._io.plots import save_and_show_figure
from insitupy.dataclasses.results import DiffExprResults


def single_volcano(
    data,
    logfoldchanges_column: str = 'log2FoldChange',
    pval_column: str = 'pvalue',
    significance_threshold: Number = 0.05,
    foldchange_threshold: Number = 2,
    title: str = None,
    adjust_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    show: bool = True,
    label_top_n: Union[int, List[str]] = 20,
    label_sortby: str = "scores",
    figsize: Tuple[int, int] = (8, 6),
    config_table=None
    ):
    """
    Create a volcano plot from the DataFrame and label the top 20 most significant up and down-regulated genes.
    For the generation of the input data `insitupy.utils.deg.create_deg_dataframe` can be used

    Args:
        data (pd.DataFrame): DataFrame containing gene names, log fold changes, and p-values.
        logfoldchanges_column (str): Column name for log fold changes (default is 'logfoldchanges').
        pval_column (str): Column name for negative log10 p-values (default is 'neg_log10_pvals').
        significance_threshold (float): P-value threshold for significance (default is 0.05).
        foldchange_threshold (float): Fold change threshold for up/down regulation (default is 2).
        title (str): Title of the plot (default is "Volcano Plot").
        adjust_labels (bool, optional): If True, adjusts the labels to avoid overlap. Default is False.
        savepath (Union[str, os.PathLike, Path], optional): Path to save the plot (default is None).
        save_only (bool): If True, only save the plot without displaying it (default is False).
        dpi_save (int): Dots per inch (DPI) for saving the plot (default is 300).
        label_top_n (int): Number of top up- and downregulated genes to label in the plot (default is 20).
        figsize (Tuple[int, int]): Size of the figure in inches (default is (8, 6)).

    Returns:
        None
    """
    # copy data to avoid modifying original DataFrame
    data = data.copy()

    # Validate if the label_sortby column exists in the DataFrame
    if label_sortby not in data.columns:
        print(f"The specified label_sortby column '{label_sortby}' does not exist in the input DataFrame. Using '{logfoldchanges_column}' instead.")
        label_sortby = logfoldchanges_column

    # prepare data
    neg_log_pval_column = "neg_log10_pvals"
    data[neg_log_pval_column] = -np.log10(data[pval_column])
    neg_log_sig_thresh = -np.log10(significance_threshold)
    lfc_threshold = np.log2(foldchange_threshold)

    # Determine colors based on significance and fold change
    colors = []
    for index, row in data.iterrows():
        if row[neg_log_pval_column] > neg_log_sig_thresh:
            if row[logfoldchanges_column] > lfc_threshold:
                colors.append('maroon')  # Up-regulated
            elif row[logfoldchanges_column] < -lfc_threshold:
                colors.append('royalblue')  # Down-regulated
            else:
                colors.append('black')  # Not significant
        else:
            colors.append('black')  # Not significant

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
    else:
        show = False  # if ax is provided, do not show by default

    # Scatter plot
    ax.scatter(data[logfoldchanges_column], data[neg_log_pval_column],
                alpha=0.5, color=colors)

    # Add labels and title
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('$\mathregular{Log_2}$ fold change', fontsize=14)
    ax.set_ylabel('$\mathregular{-Log_{10}}$ p-value', fontsize=14)

    # Add horizontal line for significance threshold
    ax.axhline(y=-np.log10(significance_threshold), color='black', linestyle='--')

    # Add vertical lines for fold change thresholds
    ax.axvline(x=lfc_threshold, color='black', linestyle='--')
    ax.axvline(x=-lfc_threshold, color='black', linestyle='--')

    # # Calculate mixed score and get top 20 up and down-regulated genes
    # volcano_data['mixed_score'] = -np.log10(volcano_data['pvals']) * volcano_data[logfoldchanges_column]

    # determine top up- and down-regulated genes for adding the names
    sig_mask = (data[neg_log_pval_column] > neg_log_sig_thresh)
    up_mask = (data[logfoldchanges_column] > lfc_threshold) & sig_mask
    down_mask = (data[logfoldchanges_column] < -lfc_threshold) & sig_mask

    # select data
    up_data = data[up_mask]
    down_data = data[down_mask]

    # select genes
    if isinstance(label_top_n, int):
        top_up_genes = up_data.nlargest(label_top_n, label_sortby)
        top_down_genes = down_data.nsmallest(label_top_n, label_sortby)
    elif isinstance(label_top_n, list):
        top_up_genes = up_data
        top_up_genes = top_up_genes[top_up_genes.index.isin(label_top_n)]
        top_down_genes = down_data
        top_down_genes = top_down_genes[top_down_genes.index.isin(label_top_n)]

    # infer x and y limits
    if len(down_data) > 0:
        xmin = min(
            down_data[logfoldchanges_column].min()*1.1,
            -(lfc_threshold*1.1))
        ymin = 0 #down_data[pval_column].min()*1.1
    else:
        xmin = -(lfc_threshold*1.1)
        ymin = 0

    if len(up_data) > 0:
        xmax = max(
            up_data[logfoldchanges_column].max()*1.1,
            lfc_threshold*1.1
            )
        ymax = max(
            up_data[neg_log_pval_column].max()*1.1,
            down_data[neg_log_pval_column].max()*1.1,
            neg_log_sig_thresh*1.1
            )
    else:
        xmax = lfc_threshold*1.1
        ymax = neg_log_sig_thresh*1.1

    xlims = (xmin, xmax)
    ylims = (ymin, ymax)

    # Combine top genes for annotation
    top_genes = pd.concat([top_up_genes, top_down_genes])

    # Adjust y-axis limits to provide space for text
    ax.set_ylim(0, ylims[1])

    # set x-axis limits to remove non-significant outliers
    ax.set_xlim(xlims[0], xlims[1])

    # Annotate top genes
    texts = []
    for gene, row in top_genes.iterrows():
        texts.append(ax.annotate(
            gene,
            (row[logfoldchanges_column], row[neg_log_pval_column]),
            fontsize=14,  # Increased font size
            alpha=0.75))

    if adjust_labels:
        # Adjust text to avoid overlap
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            max_move=None # this helped with some annotations remaining overlapping
            )

    if config_table is not None:
        _add_config_table(config_table, ax)

    # save and show figure
    save_and_show_figure(
        savepath=savepath,
        fig=plt.gcf(),
        save_only=save_only,
        dpi_save=dpi_save,
        show=show
        )
    #plt.show()

# deprecated functions
def plot_volcano(*args, **kwargs):
    from .._warnings import plot_functions_deprecations_warning
    plot_functions_deprecations_warning(name="volcano")


def _add_config_table(config_table, ax):
    # Add labels to the top of the plot, outside the plot area
    ax.annotate('Target', xy=(1, 1.04), xycoords='axes fraction',
                xytext=(-65, 0), textcoords='offset points',
                ha='left', va='center', fontsize=14, color='black',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.annotate('Reference', xy=(0, 1.04), xycoords='axes fraction',
                xytext=(93, 0), textcoords='offset points',
                ha='right', va='center', fontsize=14, color='black',
                arrowprops=dict(arrowstyle='->', color='black'))

    # Create table data
    # Add table at the bottom of the plot
    table = ax.table(
        cellText=config_table.values,
        colLabels=config_table.columns,
        cellLoc='center',
        colWidths=[.2,.4,.4],
        loc='bottom',
        bbox=[-0.12, -0.2-(0.1*(len(config_table)+1)), 1.12, 0.1*(len(config_table)+1)]
        )

    # make first row and first column bold
    for (row, col), cell in table.get_celld().items():
        if (row == 0) | (col == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    table.scale(xscale=2, yscale=1)
    # adjust position of axes (alternative to subplots_adjust above)
    pos = ax.get_position()
    new_pos = [pos.x0, pos.y0 - 0.05, pos.width, pos.height*0.7]
    ax.set_position(new_pos)

def volcano(
    results: DiffExprResults,
    significance_threshold: Number = 0.05,
    foldchange_threshold: Number = 2,
    label_top_n: int = 20,
    label_sortby: str = "log2FoldChange",
    figsize_per_plot: int = 6,
    show: bool = True,
    backend: Literal["insitupy", "decoupler"] = "insitupy"
):

    """
    Generate volcano plots for differential gene expression (DGE) results.

    Parameters
    ----------
    results : DiffExprResults
        Container object holding DGE results, including main and neighborhood comparisons.
    significance_threshold : float, optional
        P-value threshold for statistical significance. Default is 0.05.
    foldchange_threshold : float, optional
        Minimum absolute log2 fold change to consider a gene biologically significant. Default is 2.
    label_top_n : int, optional
        Number of top genes to label in each volcano plot. Default is 20.
    label_sortby : str, optional
        Column name used to sort genes for labeling. Default is "log2FoldChange".
    figsize_per_plot : int, optional
        Width (in inches) allocated per subplot. Default is 6.
    show : bool, optional
        Whether to display the figure immediately. Default is True.
    backend : {"insitupy", "decoupler"}, optional
        Plotting backend to use. "insitupy" uses a custom plotting function; "decoupler" uses decoupler's plotting utilities. Default is "insitupy".

    Raises
    ------
    ValueError
        If an unsupported backend is specified.

    Notes
    -----
    - The function automatically detects and includes neighborhood comparisons if available.
    - Titles are generated based on the DGE setup metadata.
    - Uses matplotlib for plotting.
    """

    # Collect data and titles
    results_data = [results.main]
    dge_setup = results.metadata["dge_setup"]
    condition, cond1, cond2 = dge_setup
    titles = [f"Cells\n{condition}\n{cond1} vs. {cond2}"]

    if results.has_neighbors():
        results_data += [results.nb_condition_a, results.nb_condition_b]
        titles += ["Neighborhoods (condition A)", "Neighborhoods (condition B)"]

    # Create figure
    ncols = len(results_data)
    fig, axs = plt.subplots(1, ncols, figsize=(figsize_per_plot * ncols, 6))
    if ncols == 1:
        axs = [axs]  # make iterable for consistent loop

    for ax, df, title in zip(axs, results_data, titles):

        if backend == "decoupler":
            dc.pl.volcano(
                df,
                x="log2FoldChange",
                y="pvalue",
                thr_stat=np.log2(foldchange_threshold),
                thr_sign=significance_threshold,
                top=int(label_top_n*2), ax=ax)

        elif backend == "insitupy":
            single_volcano(
                data=df,
                logfoldchanges_column='log2FoldChange',
                pval_column='pvalue',
                foldchange_threshold=foldchange_threshold,
                significance_threshold=significance_threshold,
                label_top_n=label_top_n,
                label_sortby=label_sortby,
                ax=ax,
            )
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'insitupy' or 'decoupler'.")

        ax.set_title(title)

    plt.tight_layout()
    if show:
        plt.show()
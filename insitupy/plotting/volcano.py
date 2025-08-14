import os
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

from insitupy._io.plots import save_and_show_figure


def plot_volcano(data,
                 logfoldchanges_column: str = 'logfoldchanges',
                 pval_column: str = 'neg_log10_pvals',
                 significance_threshold: Number = 0.05,
                 fold_change_threshold: Number = 2,
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
        fold_change_threshold (float): Fold change threshold for up/down regulation (default is 2).
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
    # Validate if the label_sortby column exists in the DataFrame
    if label_sortby not in data.columns:
        warn(f"The specified label_sortby column '{label_sortby}' does not exist in the input DataFrame. Using '{logfoldchanges_column}' instead.")
        label_sortby = logfoldchanges_column


    if adjust_labels:
        try:
            from adjustText import adjust_text
        except ImportError:
            raise ImportError("The 'adjustText' module is required for label adjustment. Please install it with `pip install adjusttext` or select adjust_labels=False.")

    # plt.figure(figsize=figsize)
    neg_log_sig_thresh = -np.log10(significance_threshold)
    lfc_threshold = np.log2(fold_change_threshold)

    # Determine colors based on significance and fold change
    colors = []
    for index, row in data.iterrows():
        if row[pval_column] > neg_log_sig_thresh:
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

    # Scatter plot
    # plt.scatter(data[logfoldchanges_column], data[pval_column],
    #             alpha=0.5, color=colors)
    ax.scatter(data[logfoldchanges_column], data[pval_column],
                alpha=0.5, color=colors)

    # Add labels and title
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('$\mathregular{Log_2}$ fold change', fontsize=14)
    ax.set_ylabel('$\mathregular{-Log_10}$ p-value', fontsize=14)

    # Add horizontal line for significance threshold
    ax.axhline(y=-np.log10(significance_threshold), color='black', linestyle='--')

    # Add vertical lines for fold change thresholds
    ax.axvline(x=lfc_threshold, color='black', linestyle='--')
    ax.axvline(x=-lfc_threshold, color='black', linestyle='--')

    # # Calculate mixed score and get top 20 up and down-regulated genes
    # volcano_data['mixed_score'] = -np.log10(volcano_data['pvals']) * volcano_data[logfoldchanges_column]

    # determine top up- and down-regulated genes for adding the names
    # create masks
    sig_mask = (data[pval_column] > neg_log_sig_thresh)
    up_mask = (data[logfoldchanges_column] > lfc_threshold) & sig_mask
    down_mask = (data[logfoldchanges_column] < -lfc_threshold) & sig_mask

    # select data
    up_data = data[up_mask]
    down_data = data[down_mask]

    # select genes
    if isinstance(label_top_n, int):
        # top_up_genes = up_data.nlargest(label_top_n, logfoldchanges_column)
        # top_down_genes = down_data.nsmallest(label_top_n, logfoldchanges_column)
        top_up_genes = up_data.nlargest(label_top_n, label_sortby)
        top_down_genes = down_data.nsmallest(label_top_n, label_sortby)
    elif isinstance(label_top_n, list):
        top_up_genes = up_data
        top_up_genes = top_up_genes[top_up_genes["gene"].isin(label_top_n)]
        top_down_genes = down_data
        top_down_genes = top_down_genes[top_down_genes["gene"].isin(label_top_n)]

    # infer x and y limits
    #print(down_data)
    if len(down_data) > 0:
        xmin = min(down_data[logfoldchanges_column].min()*1.1, -(lfc_threshold*2))
        ymin = 0 #down_data[pval_column].min()*1.1
    else:
        xmin = -(lfc_threshold*2)
        ymin = 0

    if len(up_data) > 0:
        xmax = max(
            up_data[logfoldchanges_column].max()*1.1,
            lfc_threshold*2
            )
        ymax = max(
            up_data[pval_column].max()*1.1,
            down_data[pval_column].max()*1.1,
            neg_log_sig_thresh*2
            )
    else:
        xmax = lfc_threshold*2
        ymax = neg_log_sig_thresh*2

    xlims = (xmin, xmax)
    ylims = (ymin, ymax)

    # Combine top genes for annotation
    #return top_up_genes
    top_genes = pd.concat([top_up_genes, top_down_genes])

    # Adjust y-axis limits to provide space for text
    #ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # Increase the upper limit of the y-axis to make space for the annotations
    ax.set_ylim(0, ylims[1])

    # set x-axis limits to remove non-significant outliers
    #print(xlims)
    ax.set_xlim(xlims[0], xlims[1])

    # Annotate top genes
    texts = []
    for i, row in top_genes.iterrows():
        texts.append(ax.annotate(
            row['gene'],
            (row[logfoldchanges_column], row[pval_column]),
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
        # table.auto_set_font_size(True)
        # table.set_fontsize(12)

        # Adjust layout to make room for the table
        # plt.subplots_adjust(left=0.2, bottom=0.1*(1+len(config_table)))

        # adjust position of axes (alternative to subplots_adjust above)
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 - 0.05, pos.width, pos.height*0.7]
        ax.set_position(new_pos)


    # save and show figure
    save_and_show_figure(
        savepath=savepath,
        fig=plt.gcf(),
        save_only=save_only,
        dpi_save=dpi_save,
        show=show
        )
    #plt.show()
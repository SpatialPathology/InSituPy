import os
from numbers import Number
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from insitupy.io.plots import save_and_show_figure


def volcano_plot(data,
                 significance_threshold: Number = 0.05,
                 fold_change_threshold: Number = 1,
                 title: str = "Volcano Plot",
                 savepath: Union[str, os.PathLike, Path] = None,
                 save_only: bool = False,
                 dpi_save: int = 300,
                 ):
    """
    Create a volcano plot from the DataFrame and label the top 20 most significant up and down-regulated genes.

    Args:
        data (pd.DataFrame): DataFrame containing gene names, log fold changes, and p-values.
        significance_threshold (float): P-value threshold for significance (default is 0.05).
        fold_change_threshold (float): Log2 fold change threshold for up/down regulation (default is 1).
    """
    plt.figure(figsize=(10, 6))

    # Determine colors based on significance and fold change
    colors = []
    for index, row in data.iterrows():
        if row['pvals'] < significance_threshold:
            if row['logfoldchanges'] > fold_change_threshold:
                colors.append('maroon')  # Up-regulated
            elif row['logfoldchanges'] < -fold_change_threshold:
                colors.append('royalblue')  # Down-regulated
            else:
                colors.append('black')  # Not significant
        else:
            colors.append('black')  # Not significant

    # Scatter plot
    plt.scatter(data['logfoldchanges'], data['neg_log10_pvals'],
                alpha=0.5, color=colors)

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('$\mathregular{Log_2}$ fold change', fontsize=14)
    plt.ylabel('$\mathregular{-Log_10}$ p-value', fontsize=14)

    # Add horizontal line for significance threshold
    plt.axhline(y=-np.log10(significance_threshold), color='black', linestyle='--')

    # Add vertical lines for fold change thresholds
    plt.axvline(x=fold_change_threshold, color='black', linestyle='--')
    plt.axvline(x=-fold_change_threshold, color='black', linestyle='--')

    # # Calculate mixed score and get top 20 up and down-regulated genes
    # volcano_data['mixed_score'] = -np.log10(volcano_data['pvals']) * volcano_data['logfoldchanges']

    top_up_genes = data[data['logfoldchanges'] > fold_change_threshold].nlargest(20, 'scores')
    top_down_genes = data[data['logfoldchanges'] < -fold_change_threshold].nsmallest(20, 'scores')

    # Combine top genes for annotation
    top_genes = pd.concat([top_up_genes, top_down_genes])

    # Adjust y-axis limits to provide space for text
    plt.ylim(0, plt.ylim()[1] * 1.2)  # Increase the upper limit of the y-axis to make space for the annotations

    # Annotate top genes
    texts = []
    for i, row in top_genes.iterrows():
        texts.append(plt.annotate(row['gene'],
                                   (row['logfoldchanges'], row['neg_log10_pvals']),
                                   fontsize=14,  # Increased font size
                                   alpha=0.75))

    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # save and show figure
    save_and_show_figure(savepath=savepath, fig=plt.gcf(), save_only=save_only, dpi_save=dpi_save)
    plt.show()
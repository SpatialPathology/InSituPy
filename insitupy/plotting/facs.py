import os
import warnings
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties

from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.plotting.save import save_and_show_figure


def facs_plot(data,
              gene1: str = 'gene1',
              gene2: str = 'gene2',
              cluster_key: str = 'None',
              threshold_gene1: Number = 1,
              threshold_gene2: Number = 1,
              cells_layer: str = None,
              layer: str = None,
              ):
    """
    Create a FACS-style scatter plot of two genes and classify double-positive cells.

    Plots expression of ``gene1`` (x-axis) against ``gene2`` (y-axis) as a scatter
    plot, with dashed threshold lines. Points can optionally be colored by a
    cluster annotation column.

    **Side effect:** Adds a boolean column named ``'{gene1}/{gene2} double pos.'``
    to ``data.cells[layer].table.obs`` marking cells that exceed both thresholds.

    Args:
        data (InSituData): The data object containing cell expression tables.
        gene1 (str, optional): Name of the gene to plot on the x-axis.
            Defaults to ``'gene1'``.
        gene2 (str, optional): Name of the gene to plot on the y-axis.
            Defaults to ``'gene2'``.
        cluster_key (str, optional): Name of the ``obs`` column to use for
            coloring points. Pass ``None`` to plot all points in a single color.
            Defaults to ``'None'``.
        threshold_gene1 (Number, optional): Expression threshold for ``gene1``.
            Cells above this value are counted as gene1-positive. Defaults to 1.
        threshold_gene2 (Number, optional): Expression threshold for ``gene2``.
            Cells above this value are counted as gene2-positive. Defaults to 1.
        cells_layer (str, optional): Name of the cell segmentation layer to use.
            Defaults to None (main layer).
        layer (str, optional): Deprecated. Use ``cells_layer`` instead.

    Returns:
        None: Displays the plot and modifies the cell table's ``obs`` in place.
    """
    if layer is not None:
        warnings.warn("'layer' is deprecated, use 'cells_layer' instead.",
                      DeprecationWarning, stacklevel=2)
        cells_layer = layer

    celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
    adata = celldata.table

    expr1 = adata[:, gene1].X.toarray().flatten()
    expr2 = adata[:, gene2].X.toarray().flatten()

    plt.figure(figsize=(7,7))

    if cluster_key is None:
        sns.scatterplot(x=expr1, y=expr2,s=8, alpha=0.6, linewidth=0)
        plt.title(f"{data.sample_id}: {gene1} vs. {gene2} expression")
    else:
        cluster=adata.obs[cluster_key]
        palette = sns.color_palette("tab10", cluster.nunique())
        sns.scatterplot(x=expr1, y=expr2,hue=cluster, palette=palette,s=8, alpha=0.6, linewidth=0)

    plt.axvline(x=threshold_gene1, color="red", linestyle="--")
    plt.axhline(y=threshold_gene2, color="red", linestyle="--")

    plt.xlabel(gene1)
    plt.ylabel(gene2)
    plt.title(f"{data.sample_id}: {gene1} vs. {gene2} expression, colored by {cluster_key}")

    plt.tight_layout()
    plt.show()

    adata.obs[f'{gene1}/{gene2} double pos.']=(expr1 > threshold_gene1) & (expr2 > threshold_gene2)
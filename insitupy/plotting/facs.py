import os
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from typing import Optional


from insitupy.plotting.save import save_and_show_figure

def facs_plot(data,
              gene1: str = 'gene1',
              gene2: str = 'gene2',
              cluster_key: str = 'None',
              threshold_gene1: Number = 1,
              threshold_gene2: Number =1,
              layer: str = 'main'
              ):

    adata=data.cells[layer].matrix

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
import os
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from adjustText import adjust_text
from matplotlib.patches import Patch

from insitupy._io.plots import save_and_show_figure


def volcano_two_sides(data_x,
            data_y,
            logfoldchanges_column: str = 'logfoldchanges',
            pval_column: str = 'neg_log10_pvals',
            significance_threshold: Number = 0.05,
            fold_change_threshold: Number = 1.0,
            title: str = None,
            adjust_labels: bool = True,
            ax: Optional[plt.Axes] = None,
            savepath: Union[str, os.PathLike, Path] = None,
            save_only: bool = False,
            dpi_save: int = 300,
            show: bool = True,
            label_top_n: Union[int, List[str]] = 20,
            figsize: Tuple[int, int] = (8, 6),
            config_table=None
):
    
    #valid_genes=data_x[data_x['logfoldchanges'] > 1.0 ]['gene'].tolist()
    valid_genes = data_x[(data_x['logfoldchanges'] > 1.0) & (data_x['logfoldchanges'] < 10.0)]['gene'].tolist()

    
    filtered_data_y=data_y[data_y['gene'].isin(valid_genes)]
    filtered_data_x=data_x[data_x['gene'].isin(valid_genes)]
    
    merged = pd.merge(filtered_data_x[["gene", "logfoldchanges"]],filtered_data_y[["gene", "logfoldchanges"]],on="gene",suffixes=("_target_vs_reference", "_target_vs_neighbors"))
    merged['pvals']=filtered_data_x['pvals']
    
    plt.figure(figsize=(7,7))
    plt.scatter(merged['logfoldchanges_target_vs_reference'], merged['logfoldchanges_target_vs_neighbors'], alpha=0.5, color="gray")
    
    plt.axhspan(1, merged['logfoldchanges_target_vs_neighbors'].max(), facecolor="lightgreen", alpha=0.3)
    plt.axhspan(merged['logfoldchanges_target_vs_neighbors'].min(), -1, facecolor="lightcoral", alpha=0.3)
    plt.axhspan(-1, 1, facecolor="white", alpha=1.0)
    
    plt.axhline(1, color="black", linestyle="--", linewidth=1)
    plt.axhline(-1, color="black", linestyle="--", linewidth=1)
    plt.axvline(fold_change_threshold, color="black", linestyle="--", linewidth=1)

    subset_green = merged[(merged["logfoldchanges_target_vs_neighbors"] > 1) & (merged["logfoldchanges_target_vs_reference"] > fold_change_threshold)]
    plt.scatter(subset_green["logfoldchanges_target_vs_reference"], subset_green["logfoldchanges_target_vs_neighbors"], color="green", alpha=0.8)
    
    texts=[]
    for _, row in subset_green.iterrows():
        if row['pvals'] <=0.05:
            t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
        else:
            t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=10, color="gray",fontstyle="oblique")
        texts.append(t)
    adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)) 
        
    subset_red = merged[(merged["logfoldchanges_target_vs_neighbors"] < -1) & (merged["logfoldchanges_target_vs_reference"] > fold_change_threshold)]
    plt.scatter(subset_red["logfoldchanges_target_vs_reference"], subset_red["logfoldchanges_target_vs_neighbors"], color="red", alpha=0.8)
    
    texts = []
    for _, row in subset_red.iterrows():
        if row['pvals'] <=0.05:
            t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
            
        else:
            t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=10, color="gray")
        texts.append(t)
    adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)) 
      
    #handles = [
       # Line2D([0], [0], color='black', linestyle='-', label='significant (p<=0.05)'),
       # Line2D([0], [0], color='gray', linestyle='-', label='not significant (p>0.05)')]
    
    handles = [
        Patch(facecolor='black', edgecolor='black', label='GENE (p <= 0.05)'),
        Patch(facecolor='gray', edgecolor='gray', label='GENE (p > 0.05)')]
    
    plt.legend(handles=handles, loc='upper right', fontsize=10)    
    plt.xlabel("pos. log2FC target_vs_reference")
    plt.ylabel("log2FC target_vs_neighbors")
    plt.title("DEG")
    plt.show()
    

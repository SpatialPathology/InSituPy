# import os
# from numbers import Number
# from pathlib import Path
# from typing import List, Optional, Tuple, Union
# from warnings import warn

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import pandas as pd
# from matplotlib.font_manager import FontProperties
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# from adjustText import adjust_text
# from matplotlib.patches import Patch

# from insitupy.plotting.save import save_and_show_figure


# def volcano_two_sides(data_x,
#             data_y,
#             logfoldchanges_column: str = 'log2foldchange',
#             pval_column: str = 'neg_log10_pvals',
#             fold_change_threshold: Number = 1.0,
#             title: str = None,
# ):

#     #valid_genes=data_x[data_x['log2foldchange'] > 1.0 ]['gene'].tolist()
#     valid_genes = data_x[(data_x['log2foldchange'] > 1.0) & (data_x['log2foldchange'] < 10.0)]['gene'].tolist()

#     filtered_data_y=data_y[data_y['gene'].isin(valid_genes)]
#     filtered_data_x=data_x[data_x['gene'].isin(valid_genes)]

#     merged = pd.merge(filtered_data_x[["gene", "log2foldchange"]],filtered_data_y[["gene", "log2foldchange"]],
#                       on="gene",suffixes=("_target_vs_reference", "_target_vs_neighbors"))
#     merged['neg_log10_pvals']=filtered_data_x[pval_column]
#     sig = merged["neg_log10_pvals"] > 1.3

#     plt.figure(figsize=(7,7))
#     plt.axhspan(1, merged['logfoldchanges_target_vs_neighbors'].max(), facecolor="lightgreen", alpha=0.3)
#     plt.axhspan(merged['logfoldchanges_target_vs_neighbors'].min(), -1, facecolor="lightcoral", alpha=0.3)
#     plt.scatter(merged['logfoldchanges_target_vs_reference'][sig], merged['logfoldchanges_target_vs_neighbors'][sig],c="black", label="significant",alpha=1.0,s=20)
#     plt.scatter(merged['logfoldchanges_target_vs_reference'][~sig], merged['logfoldchanges_target_vs_neighbors'][~sig],c="gray", label="non significant",alpha=1.0, s=15)

#     plt.axhline(1, color="black", linestyle="--", linewidth=1)
#     plt.axhline(-1, color="black", linestyle="--", linewidth=1)
#     plt.axvline(fold_change_threshold, color="black", linestyle="--", linewidth=1)

#     subset_green = merged[(merged["logfoldchanges_target_vs_neighbors"] > 1) & (merged["logfoldchanges_target_vs_reference"] > fold_change_threshold)]
#     texts=[]
#     for _, row in subset_green.iterrows():
#         if row['neg_log10_pvals']  > 1.3:
#             t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
#         else:
#             t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=10, color="gray",fontstyle="oblique")
#         texts.append(t)
#     adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

#     subset_red = merged[(merged["logfoldchanges_target_vs_neighbors"] < -1) & (merged["logfoldchanges_target_vs_reference"] > fold_change_threshold)]
#     texts = []
#     for _, row in subset_red.iterrows():
#         if row['neg_log10_pvals']  > 1.3:
#             t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
#         else:
#             t=plt.text(row["logfoldchanges_target_vs_reference"], row["logfoldchanges_target_vs_neighbors"], row["gene"], fontsize=10, color="gray")
#         texts.append(t)
#     adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

#     plt.legend(title='pvalue (target_vs_reference)',loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.xlabel("pos. log2FC target_vs_reference")
#     plt.ylabel("log2FC target_vs_neighbors")
#     plt.title(title)
#     plt.show()

#     return merged




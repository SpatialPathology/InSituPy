"""
Functions in this module were adapted from the decoupler package v1.9.0 (https://github.com/scverse/decoupler):
Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D., Müller-Dott S.,
Taus P., Dugourd A., Holland C.H., Ramirez Flores R.O. and Saez-Rodriguez J. 2022.
decoupleR: Ensemble of computational methods to infer biological activities from omics data.
Bioinformatics Advances. https://doi.org/10.1093/bioadv/vbac016

"""
import copy
import glob
import os
import sys
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings, warn

import anndata as ad
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm

import insitupy
from insitupy._core.data import CACHE, ImageData, InSituData
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment
#from insitupy import InSituExperiment
from insitupy.io import read_xenium

# def extract_adata_from_InSituExperiment(exp, cell_layer:str, sample_col):

#     adatas={}
#     for id, data in exp.iterdata():
#         layer= _get_cell_layer(cells=data.cells,cells_layer=cell_layer)
#         adata = layer.matrix
#         adatas[layer.matrix.obs[sample_col].unique().tolist()[0]]=adata

#     return adatas


# def concatenate_adatas_for_pseudobulk(adatas: dict,label:str):

#     adata=ad.concat(adatas,label=label)

#     return adata


def get_neighborhood(
    exp: InSituExperiment,
    sample_col: str,
    cells_layer: Optional[str] = None,
    radius: Number = 30
    ):

    matrices = {}

    for id, data in exp.iterdata():
        layer = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        coords = layer.matrix.obsm["spatial"]
        A = radius_neighbors_graph(coords, radius=radius, mode="connectivity", include_self=False)
        matrices[layer.matrix.obs[sample_col].unique().tolist()[0]] = A

    return matrices


#def neighborhoods_pseudobulk(matrices, exp,cells_layers,groups_col,raw_counts,sample_col):
def neighborhoods_pseudobulk(
    celldata,
    groups_col,
    counts_layer,
    sample_col,
    radius: int = 20,
    mode: Literal["sum", "mean", "median"] = "sum"
    ):
    try:
        import decoupler as dc

    except ImportError:
        print(("Decoupler is not installed. Interactive visualization using `.show()` will not be possible. If you want to use these features, install decoupler with `pip install decoupler`"))

    # pseudobulks={}
    # for id, data in exp.iterdata():
    # layer= _get_cell_layer(cells=data.cells,cells_layer=cells_layers)
    # print(layer.matrix.obs[sample_col].unique().tolist()[0])

    # celldata.matrix.obs["cell_id"] = [f"{i}" for i in range(celldata.matrix.n_obs)]
    adata = celldata.matrix
    # adata.X = adata.layers[raw_counts].copy()
    coords = celldata.matrix.obsm["spatial"]
    A = radius_neighbors_graph(coords, radius=radius, mode="connectivity", include_self=False)

    celltype_pdata={}
    for celltype in celldata.matrix.obs[groups_col].unique():
        # select cells of that cell type
        mask = adata.obs[groups_col] == celltype

        # check which of the neighboring cells it neighbor to at least one cell of that type
        any_mask = A.toarray()[mask].any(axis=0)

        # filter for such neighboring cells
        filtered = adata[any_mask]
        # cell_names = celldata.matrix[celldata.matrix.obs[groups_col] == celltype].obs['cell_id']
        # cell_names = cell_names.tolist()
        # cell_names = [int(x) for x in cell_names]

        # #A = matrices[layer.matrix.obs[sample_col].unique().tolist()[0]]
        # neighbors= []
        # for i in cell_names:
        #     neighbors.append(A[i].nonzero()[1])

        # all_indices = np.concatenate(neighbors)
        # unique_indices = np.unique(all_indices)
        # unique_list = unique_indices.tolist()
        # unique_list = [str(x) for x in unique_list]


        # filtered=celldata.matrix[celldata.matrix.obs['cell_id'].isin(unique_list)].copy()

        # calculate pseudobulk for neighboring cells that are in the neighborhood of at least one cell of that type
        pdata = dc.pp.pseudobulk(
            adata=filtered,
            sample_col=sample_col,
            groups_col=None,
            layer=counts_layer,
            mode=mode
            )

        celltype_pdata[celltype] = pdata

    pdata_big = ad.concat(celltype_pdata, label=groups_col)
    # pdata_big=concatenate_adatas_for_pseudobulk(celltype_pdata,label=groups_col)

    pdata_big.obs['pseudo_neighbors'] = (pdata_big.obs.index.astype(str)+ "_" + pdata_big.obs[groups_col].astype(str)  + "_neighbors" )
    pdata_big.obs = pdata_big.obs.set_index("pseudo_neighbors")
    # pseudobulks[layer.matrix.obs[sample_col].unique().tolist()[0]]=pdata_big

    # pdata_neighbors=concatenate_adatas_for_pseudobulk(pseudobulks,label=sample_col)

    # return pdata_neighbors
    return pdata_big


def concatenate_pdata_and_pdata_neighbors(pdata,pdata_neighbors):

    pdata_neighbors.obs = pdata_neighbors.obs.set_index("pseudo_neighbors")

    ps={'pdata_neighbors':pdata_neighbors,
        'pdata':pdata}

    pdata_final=ad.concat(ps)
    pdata_final.obs['neighbors'] = np.where(pdata_final.obs.index.str.contains("neighbors"),'neighbors','cell')

    return pdata_final


def generate_pseudobulk(
    exp,
    celltype_col: Optional[str] = None,
    cells_layer: Optional[str] = None,
    counts_layer: Optional[str] = None,
    mode: Literal["sum", "mean", "median"] = "sum",
    calculate_neighbors: bool = False,
    neighbors_radius: int = 20,
    **kwargs
    ):

    try:
        import decoupler as dc

    except ImportError:
        print(("Decoupler is not installed. Interactive visualization using `.show()` will not be possible. If you want to use these features, install decoupler with `pip install decoupler`"))

    # adatas=extract_adata_from_InSituExperiment(
    #     exp=exp,
    #     cell_layer=cell_layer,
    #     sample_col=sample_col
    #     )

    pseudobulks={}
    # matrices = {}
    for meta, data in exp.iterdata():
        # get UID
        uid = meta["uid"]

        # extract anndata
        celldata= _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        adata = celldata.matrix

        # add batch information
        adata.obs["uid"] = uid

        # create pseudobulk from anndata
        pdata = dc.pp.pseudobulk(
            adata=adata,
            sample_col="uid",
            groups_col=celltype_col,
            layer=counts_layer,
            mode=mode,
            )

        if calculate_neighbors:
            pdata_neighbors = neighborhoods_pseudobulk(
                celldata=celldata,
                groups_col=celltype_col,
                counts_layer=counts_layer,
                sample_col=uid,
                radius=neighbors_radius,
                mode=mode,
                **kwargs
            )

            # concatenate pseudobulk and neighbor pseudobulk
            pdata = ad.concat(
                {
                'pdata_neighbors': pdata_neighbors,
                'pdata': pdata
                }
            )
            pdata.obs['neighbors'] = np.where(
                pdata.obs.index.str.contains("neighbors"),
                'neighbors',
                'cell'
                )

        # collect data
        pseudobulks[uid] = pdata

    # concatenate all pseudobulks
    pdata_final = ad.concat(pseudobulks, label=uid)

    return pdata_final

    # if calculate_neighbors:

    #     # matrices=get_neighborhood(exp=exp,cells_layer=cells_layer,radius=radius, sample_col=sample_col)

    #     pdata_neighbors=neighborhoods_pseudobulk(
    #         matrices=matrices,
    #         exp=exp,
    #         cells_layers=cells_layer,
    #         groups_col=groups_col,
    #         counts_layer=counts_layer,
    #         sample_col=sample_col
    #         )

    #     pdata_final=concatenate_pdata_and_pdata_neighbors(pdata_big,pdata_neighbors)
    #     return pdata_final

    # else:
    #     return pdata_big


def two_sides_volcano(results_df_normal,results_df_neighbors,significance_threshold,fold_change_threshold):

    n_upreg_target = np.sum((results_df_normal["pvalue"] <= significance_threshold) & (results_df_normal["log2FoldChange"] > np.log2(fold_change_threshold)))
    n_downreg_target = np.sum((results_df_normal["pvalue"] <= significance_threshold) & (results_df_normal["log2FoldChange"] < -np.log2(fold_change_threshold)))

    n_upreg_target = np.sum((results_df_neighbors["pvalue"] <= significance_threshold) & (results_df_neighbors["log2FoldChange"] > np.log2(fold_change_threshold)))
    n_downreg_target = np.sum((results_df_neighbors["pvalue"] <= significance_threshold) & (results_df_neighbors["log2FoldChange"] < -np.log2(fold_change_threshold)))

    #valid_genes = de_genes_df[(de_genes_df['logfoldchanges'] > 1.0) & (de_genes_df['logfoldchanges'] < 10.0)].index.tolist()
    valid_genes = results_df_normal[(results_df_normal['log2FoldChange'] > 1.0)].index.tolist()
    print(valid_genes)

    filtered_data_y=results_df_neighbors[results_df_neighbors.index.isin(valid_genes)]
    filtered_data_x=results_df_normal[results_df_normal.index.isin(valid_genes)]

    merged = pd.merge(
    filtered_data_x[["log2FoldChange","pvalue"]],
    filtered_data_y[["log2FoldChange"]],
    left_index=True,
    right_index=True,
    suffixes=("_target_vs_reference", "_target_vs_neighbors"))
    merged = merged.reset_index().rename(columns={"index": "gene"})

    merged["neg_log10_pvals"]=-np.log10(merged['pvalue'])
    sig = merged["neg_log10_pvals"] > 1.3


    plt.figure(figsize=(7,7))
    plt.axhspan(0, merged['log2FoldChange_target_vs_neighbors'].max(), facecolor="lightgreen", alpha=0.3)
    plt.axhspan(-1, 0, facecolor="lightyellow", alpha=0.3)

    if merged['log2FoldChange_target_vs_neighbors'].min() < -1:
        plt.axhspan(merged['log2FoldChange_target_vs_neighbors'].min(),-1, facecolor="lightcoral", alpha=0.3)



    plt.scatter(merged['log2FoldChange_target_vs_reference'][sig], merged['log2FoldChange_target_vs_neighbors'][sig],c="black", label="significant",alpha=1.0,s=20)
    plt.scatter(merged['log2FoldChange_target_vs_reference'][~sig], merged['log2FoldChange_target_vs_neighbors'][~sig],c="gray", label="non significant",alpha=1.0, s=15)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axhline(-1, color="black", linestyle="--", linewidth=1)
    plt.axvline(fold_change_threshold, color="black", linestyle="--", linewidth=1)

    subset_green = merged[(merged["log2FoldChange_target_vs_neighbors"] > 0) & (merged["log2FoldChange_target_vs_reference"] > fold_change_threshold)]
    texts=[]
    for _, row in subset_green.iterrows():
        if row['neg_log10_pvals']  > 1.3:
            t=plt.text(row["log2FoldChange_target_vs_reference"], row["log2FoldChange_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
        else:
            t=plt.text(row["log2FoldChange_target_vs_reference"], row["log2FoldChange_target_vs_neighbors"], row["gene"], fontsize=10, color="gray",fontstyle="oblique")
        texts.append(t)
    adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    subset_red = merged[(merged["log2FoldChange_target_vs_neighbors"] < -1) & (merged["log2FoldChange_target_vs_reference"] > fold_change_threshold)]
    texts = []
    for _, row in subset_red.iterrows():
        if row['neg_log10_pvals']  > 1.3:
            t=plt.text(row["log2FoldChange_target_vs_reference"], row["log2FoldChange_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
        else:
            t=plt.text(row["log2FoldChange_target_vs_reference"], row["log2FoldChange_target_vs_neighbors"], row["gene"], fontsize=10, color="gray")
        texts.append(t)
    adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))


    subset_yellow = merged[(merged["log2FoldChange_target_vs_neighbors"] > -1) & (merged["log2FoldChange_target_vs_neighbors"] < 0)]
    texts = []
    for _, row in subset_yellow.iterrows():
        if row['neg_log10_pvals']  > 1.3:
            t=plt.text(row["log2FoldChange_target_vs_reference"], row["log2FoldChange_target_vs_neighbors"], row["gene"], fontsize=12, color="black")
        else:
            t=plt.text(row["log2FoldChange_target_vs_reference"], row["log2FoldChange_target_vs_neighbors"], row["gene"], fontsize=10, color="gray")
        texts.append(t)

    adjust_text(texts,arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.legend(title='pvals (target_vs_reference)',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("pos. log2FC target_vs_reference")
    plt.ylabel("log2FC target_vs_neighbors")
    #plt.title(title)
    plt.show()

    return merged
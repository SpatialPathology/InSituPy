"""
Functions in this module were adapted from the decoupler package v1.9.0 (https://github.com/scverse/decoupler):
Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D., Müller-Dott S.,
Taus P., Dugourd A., Holland C.H., Ramirez Flores R.O. and Saez-Rodriguez J. 2022.
decoupleR: Ensemble of computational methods to infer biological activities from omics data.
Bioinformatics Advances. https://doi.org/10.1093/bioadv/vbac016

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack

from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment

import insitupy
from insitupy._core.data import InSituData,ImageData, CACHE
#from insitupy import InSituExperiment
from insitupy.io import read_xenium
import scanpy as sc

from pathlib import Path
import dask
import os
import glob
import copy

from insitupy.io import read_xenium
import anndata as ad
import sys

from tqdm import tqdm
from sklearn.neighbors import radius_neighbors_graph
from numbers import Number
from typing import List, Literal, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings, warn

from sklearn.neighbors import radius_neighbors_graph
from typing import Literal, Optional
import os
from adjustText import adjust_text
from insitupy.dataclasses._utils import _get_cell_layer


def extract_adata_from_InSituExperiment(exp, cell_layer:str):
    
    adatas={}
    for id, data in exp.iterdata():
        layer= _get_cell_layer(cells=data.cells,cells_layer=cell_layer)
        adata = layer.matrix
        adatas[data.sample_id]=adata
    
    return adatas


def concatenate_adatas_for_pseudobulk(adatas: dict,label:str):
    
    adata=ad.concat(adatas,label=label)
    
    return adata


def get_neighborhood(exp: InSituExperiment, cells_layer: Optional[str] = None, radius: Number = 30):
    
    matrices={}
    #  calculation of neighborhood for each sample
    for id, data in exp.iterdata():
        layer= _get_cell_layer(cells=data.cells,cells_layer=cells_layer)
        coords = layer.matrix.obsm["spatial"]
        A = radius_neighbors_graph(coords, radius=radius, mode="connectivity", include_self=False)
        matrices[data.sample_id]=A
        
    return matrices


def neighborhoods_pseudobulk(matrices, exp,cells_layers,groups_col,raw_counts,sample_col):
    try:
        import decoupler as dc
    
    except ImportError:
        print(("Decoupler is not installed. Interactive visualization using `.show()` will not be possible. If you want to use these features, install decoupler with `pip install decoupler`"))
    
    pseudobulks={}
    for id, data in exp.iterdata():
        print(data.sample_id)
        layer= _get_cell_layer(cells=data.cells,cells_layer=cells_layers)
        layer.matrix.obs["cell_id"] = [f"{i}" for i in range(layer.matrix.n_obs)]
        layer.matrix.X=layer.matrix.layers[raw_counts].copy()
        celltype_pdata={}
        for celltype in layer.matrix.obs[groups_col].unique():
            cell_names=layer.matrix[layer.matrix.obs[groups_col]==celltype].obs['cell_id']
            cell_names=cell_names.tolist()
            cell_names = [int(x) for x in cell_names]
        
            A = matrices[data.sample_id] 
            neighbors= []
            for i in cell_names:  
                neighbors.append(A[i].nonzero()[1])
            
            all_indices = np.concatenate(neighbors)
            unique_indices = np.unique(all_indices)
            unique_list = unique_indices.tolist()
            unique_list = [str(x) for x in unique_list]
            
           
            filtered=layer.matrix[layer.matrix.obs['cell_id'].isin(unique_list)].copy()
           
            pdata = dc.pp.pseudobulk(adata=filtered,sample_col=sample_col,groups_col=None, mode="sum") 
            
            celltype_pdata[celltype]=pdata
        pdata_big=concatenate_adatas_for_pseudobulk(celltype_pdata,label=groups_col)
        
        pdata_big.obs['pseudo_neighbors'] = (pdata_big.obs.index.astype(str)+ "_" + pdata_big.obs[groups_col].astype(str)  + "_neighbors" )
        pseudobulks[data.sample_id]=pdata_big 
        pdata_neighbors=concatenate_adatas_for_pseudobulk(pseudobulks,label=sample_col)
        
          
    return pdata_neighbors


def concatenate_pdata_and_pdata_neighbors(pdata,pdata_neighbors):
    
    pdata_neighbors.obs = pdata_neighbors.obs.set_index("pseudo_neighbors")

    ps={'pdata_neighbors':pdata_neighbors,
        'pdata':pdata}
    
    pdata_final=ad.concat(ps)
    pdata_final.obs['neighbors'] = np.where(pdata_final.obs.index.str.contains("neighbors"),'neighbors','cell')
    
    return pdata_final


def generate_pseudobulk_neighbors(exp,cell_layer,raw_counts,sample_col,groups_col,radius):
    
    try:
        import decoupler as dc
    
    except ImportError:
        print(("Decoupler is not installed. Interactive visualization using `.show()` will not be possible. If you want to use these features, install decoupler with `pip install decoupler`"))
    
    adatas=extract_adata_from_InSituExperiment(exp=exp, cell_layer=cell_layer)
    
    pseudobulks={}
    for id, adata in adatas.items():
        adata.X=adata.layers[raw_counts].copy()
        pdata = dc.pp.pseudobulk(adata=adata,sample_col=sample_col,groups_col=groups_col, mode="sum") 
        pseudobulks[id]=pdata
    
    pdata_big=concatenate_adatas_for_pseudobulk(pseudobulks,label=sample_col)
    
    matrices=get_neighborhood(exp=exp,cells_layer=cell_layer,radius=radius)
    
    pdata_neighbors=neighborhoods_pseudobulk(matrices=matrices,exp=exp,cells_layers=cell_layer,groups_col=groups_col,raw_counts=raw_counts,sample_col=sample_col)
    
    pdata_final=concatenate_pdata_and_pdata_neighbors(pdata_big,pdata_neighbors)
    
    return pdata_final   
    
    
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
from typing import Optional, Tuple

import anndata as ad
import decoupler as dc
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

from insitupy.dataclasses.results import (DiffExprConfigCollector,
                                          DiffExprResults)
from insitupy.utils._helpers import suppress_output


def _obs_qc_plot(
    pdata,
    pdata_nb,
    celltype_col,
    condition_str
):
    if pdata_nb is not None:
        data_list = [pdata, pdata_nb]
        data_names = ["Pseudobulk of cells", "Pseudobulk of neighborhood"]
    else:
        data_list = [pdata]
        data_names = ["Pseudobulk of cells"]

    groups = [celltype_col, condition_str]
    ncols = len(groups)
    nrows = len(data_list)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 4*nrows))

    for r, d in enumerate(data_list):
        for c, g in enumerate(groups):
            dc.pl.filter_samples(
                adata=d,
                groupby=g,
                min_cells=10,
                min_counts=1000,
                ax=axs[r,c]
            )

            axs[r,c].set_title(f"{data_names[r]}")

    plt.tight_layout()
    plt.show()

def _feature_qc_plot(
    pdata_ct
):
    fig, axs = plt.subplots(1,2, figsize=(8*2, 6))
    dc.pl.filter_by_expr(
        adata=pdata_ct,
        group="condition",
        min_count=10,
        min_total_count=15,
        large_n=10,
        min_prop=0.7,
        ax=axs[0]
    )
    dc.pl.filter_by_prop(
        adata=pdata_ct,
        min_prop=0.1,
        min_smpls=2,
        ax=axs[1]
    )
    plt.show()

def _preprocess_psbulk_data(adata):
    # Store raw counts in layers
    adata.layers["counts"] = adata.X.copy()

    # Normalize, scale and compute pca
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)

    # Return raw counts to X
    dc.pp.swap_layer(adata=adata, key="counts", inplace=True)

    return adata

def _run_deseq2_pseudobulk(adata, dge_setup, return_params: bool = False):
    try:
        from pydeseq2.dds import DefaultInference, DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "The package `pydeseq2` is not installed but is required for pseudobulk differential gene expression analysis.\n"
            "Please install it via `pip install pydeseq2`."
        )

    with suppress_output():
        # Build DESeq2 object
        inference = DefaultInference(n_cpus=8)
        dds = DeseqDataSet(
            adata=adata,
            design=f"~{dge_setup[0]}",
            refit_cooks=True,
            inference=inference,
        )

        # Compute LFCs
        dds.deseq2()

        # Extract contrast between conditions
        stat_res = DeseqStats(dds, contrast=dge_setup, inference=inference)

        # Compute Wald test
        stat_res.summary()

    if return_params:
        params = extract_all_params(ds=stat_res, dds=dds)
        return stat_res, params
    else:
        return stat_res

def _verbose_filter_samples(pdata, min_cells, min_counts, verbose: bool = True):
    # do filtering of pseudobulk samples
    before = pdata.shape[0]
    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)
    after = pdata.shape[0]

    if verbose:
        print(f"Filtered pseudobulk samples: {before - after} removed, {after} remaining (out of {before} total).", flush=True)

def _verbose_filter_features(
    pdata,
    verbose: bool = True
    ):
    before = pdata.shape[1]
    # do filtering of features
    dc.pp.filter_by_expr(
        adata=pdata,
        group="condition",
        min_count=10,
        min_total_count=15,
        large_n=10,
        min_prop=0.7,
    )
    dc.pp.filter_by_prop(
        adata=pdata,
        min_prop=0.1,
        min_smpls=2,
    )
    after = pdata.shape[1]

    if verbose:
        print(f"Filtered features: {before - after} removed, {after} remaining (out of {before} total).", flush=True)


def pseudobulk_dge(
    pdata,
    dge_setup: Tuple[str, str, str],
    celltype_col: str,
    celltype: str,
    pdata_nb: Optional[AnnData] = None,
    plot_qc: bool = True,
    min_cells: int = 10,
    min_counts: int = 1000,
    verbose: bool = True
    ):

    if plot_qc:
        # plot QC
        print("Sample filtering QC:", flush=True)
        _obs_qc_plot(
            pdata=pdata, pdata_nb=pdata_nb,
            celltype_col=celltype_col,
            condition_str=dge_setup[0]
        )

    # do filtering of pseudobulk samples
    _verbose_filter_samples(pdata, min_cells, min_counts, verbose)

    if pdata_nb is not None:
        _verbose_filter_samples(pdata_nb, min_cells, min_counts, verbose)

    # select cell type
    pdata_ct = pdata[pdata.obs[celltype_col] == celltype, :].copy()

    if pdata_nb is not None:
        pdata_ct_nb = pdata_nb[pdata_nb.obs[celltype_col] == celltype, :].copy()

    if plot_qc:
        # plot feature QC
        print("Feature filtering QC:", flush=True)
        _feature_qc_plot(pdata_ct)

    _verbose_filter_features(pdata=pdata_ct, verbose=verbose)

    if pdata_nb is not None:
        pdata_ct_nb = pdata_ct_nb[:, pdata_ct_nb.var_names.isin(pdata_ct.var_names)].copy()

    # do preprocessing
    pdata_ct = _preprocess_psbulk_data(pdata_ct)

    if pdata_nb is not None:
        pdata_ct_nb = _preprocess_psbulk_data(pdata_ct_nb)

    # prepare data for differential gene expression analysis
    pdata_ct.obs["obs_type"] = "cells"

    if pdata_nb is not None:
        pdata_first_condition = ad.concat({
            "cells": pdata_ct[pdata_ct.obs[dge_setup[0]] == dge_setup[1]],
            "neighbors": pdata_ct_nb[pdata_ct_nb.obs[dge_setup[0]] == dge_setup[1]],
        }, label="obs_type")

        pdata_second_condition = ad.concat({
            "cells": pdata_ct[pdata_ct.obs[dge_setup[0]] == dge_setup[2]],
            "neighbors": pdata_ct_nb[pdata_ct_nb.obs[dge_setup[0]] == dge_setup[2]],
        }, label="obs_type")


    # run DESeq2 for conditions and return results
    stat_res, params = _run_deseq2_pseudobulk(pdata_ct, dge_setup=dge_setup, return_params=True)
    results_df = stat_res.results_df.rename({"log2FoldChange": "log2foldchange"}, axis=1)

    if pdata_nb is not None:
        # run DESeq2 for neighborhood data and return results
        stat_res_first = _run_deseq2_pseudobulk(pdata_first_condition, dge_setup=["obs_type", "cells", "neighbors"])
        stat_res_second = _run_deseq2_pseudobulk(pdata_second_condition, dge_setup=["obs_type", "cells", "neighbors"])
        results_df_nb_first = stat_res_first.results_df.rename({"log2FoldChange": "log2foldchange"}, axis=1)
        results_df_nb_second = stat_res_second.results_df.rename({"log2FoldChange": "log2foldchange"}, axis=1)

    # collect the configurations
    config = DiffExprConfigCollector(
        mode="pseudobulk",
        method_params={
            "pseudobulk": {
                "min_cells": min_cells,
                "min_counts": min_counts
            }.update(pdata.uns['pseudobulk_settings']),
            "deseq2": params
        }
    )

    results = DiffExprResults(
        main=results_df,
        config=config,
        target_neighborhood=results_df_nb_first if pdata_nb is not None else None,
        ref_neighborhood=results_df_nb_second if pdata_nb is not None else None,
    )

    return results


def extract_deseqstats_params(ds):
    """Extract parameters from DeseqStats object.

    Parameters
    ----------
    ds : DeseqStats
        Fitted DeseqStats object

    Returns
    -------
    dict
        Dictionary containing DeseqStats parameters
    """
    params = {
        'contrast': ds.contrast,
        'alpha': ds.alpha,
        'cooks_filter': ds.cooks_filter,
        'shrunk_LFCs': ds.shrunk_LFCs,
    }
    return params


def extract_deseqdataset_params(dds):
    """Extract parameters from DeseqDataSet object.

    Parameters
    ----------
    dds : DeseqDataSet
        Fitted DeseqDataSet object

    Returns
    -------
    dict
        Dictionary containing DeseqDataSet parameters
    """
    params = {
        'design': str(dds.design),
        'refit_cooks': dds.refit_cooks,
    }
    return params


def extract_uns_params(dds):
    """Extract parameters from .uns attribute in DeseqDataSet object.

    Parameters
    ----------
    dds : DeseqDataSet
        Fitted DeseqDataSet object

    Returns
    -------
    dict
        Dictionary containing parameters from .uns
    """
    params = dict(dds.uns)
    return params


def extract_all_params(ds, dds):
    """Extract all parameters from DeseqStats and DeseqDataSet objects.

    Parameters
    ----------
    ds : DeseqStats
        Fitted DeseqStats object
    dds : DeseqDataSet
        Fitted DeseqDataSet object

    Returns
    -------
    dict
        Dictionary containing all extracted parameters
    """
    all_params = {}
    all_params.update(extract_deseqstats_params(ds))
    all_params.update(extract_deseqdataset_params(dds))
    all_params.update(extract_uns_params(dds))
    return all_params

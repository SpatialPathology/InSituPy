import os
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings, warn

import anndata
import anndata as ad
import decoupler as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.dataclasses.results import DiffExprResults
from insitupy.experiment.data import InSituExperiment
from insitupy.plotting import single_volcano
from insitupy.plotting.dge import dual_foldchange_plot
from insitupy.utils._dge import _select_data_for_dge
from insitupy.utils._helpers import suppress_output
from insitupy.utils.dge import create_deg_dataframe


def differential_gene_expression(
    target: InSituData,
    target_annotation_tuple: Optional[Tuple[str, str]] = None,
    target_cell_type_tuple: Optional[Tuple[str, str]] = None,
    target_region_tuple: Optional[Tuple[str, str]] = None,
    ref: Optional[Union[InSituData, List[InSituData]]] = None,
    ref_annotation_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
    ref_cell_type_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
    ref_region_tuple: Optional[Tuple[str, str]] = "same",
    cells_layer: Optional[str] = None,
    significance_threshold: Number = 0.05,
    fold_change_threshold: Number = 2,
    show_volcano: bool = True,
    return_results: bool = False,
    method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 't-test',
    exclude_ambiguous_assignments: bool = False,
    force_assignment: bool = False,
    title: Optional[str] = None,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    verbose: bool = False,
    **volcano_kwargs
):
    """
    Perform differential gene expression analysis on in situ sequencing data.

    This function compares gene expression between specified annotations within a single
    InSituData object or between two InSituData objects. It supports various statistical
    methods for differential expression analysis and can generate a volcano plot of the results.

    Args:
        target (InSituData): The primary in situ data object.
        target_annotation_tuple (Optional[Tuple[str, str]]): Tuple containing the annotation key and name for the target data.
        target_cell_type_tuple (Optional[Tuple[str, str]]): Tuple specifying an observation key and value to filter the target data by cell type.
        target_region_tuple (Optional[Tuple[str, str]]): Tuple specifying a region key and name to restrict the analysis to a specific region in the target data.
        ref (Optional[Union[InSituData, List[InSituData]]]): Reference in situ data object(s) for comparison. Defaults to None.
        ref_annotation_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Tuple containing the reference annotation key and name, or "rest" to use the rest of the data as reference, or "same" to use the same annotation as the target. Defaults to "same".
        ref_cell_type_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Tuple specifying an observation key and value to filter the reference data by cell type, or "rest" to use the rest of the data, or "same" to use the same cell type as the target. Defaults to "same".
        ref_region_tuple (Optional[Tuple[str, str]]): Tuple specifying a region key and name to restrict the analysis to a specific region in the reference data. Defaults to None.
        significance_threshold (float): P-value threshold for significance (default is 0.05).
        fold_change_threshold (float): Fold change threshold for up/down regulation (default is 1).
        show_volcano (bool): Whether to generate a volcano plot of the results. Defaults to True.
        return_results (bool): Whether to return the results as dictionary including the dataframe differentially expressed genes and the parameters.
        method (Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']]): Statistical method to use for differential expression analysis. Defaults to 't-test'.
        exclude_ambiguous_assignments (bool): Whether to exclude ambiguous assignments in the data. Defaults to False.
        force_assignment (bool): Whether to force assignment of annotations and regions even if it has been done before already. Defaults to False.
        title (Optional[str]): Title for the volcano plot. Defaults to None.
        savepath (Union[str, os.PathLike, Path]): Path to save the plot. Defaults to None.
        save_only (bool): If True, only save the plot without displaying it. Defaults to False.
        dpi_save (int): Dots per inch (DPI) for saving the plot. Defaults to 300.
        verbose (bool): Whether to print detailed information during the analysis. Defaults to False.
        **volcano_kwargs: Additional keyword arguments for the volcano plot.

    Returns:
        Union[None, Dict[str, Any]]: If `plot_volcano` is True, returns None. Otherwise, returns a dictionary with the results DataFrame and parameters used for the analysis.

    Raises:
        ValueError: If `ref_annotation_tuple` is neither 'rest' nor a 2-tuple.
        AssertionError: If `ref` is provided when `ref_annotation_tuple` is 'rest'.
        AssertionError: If `target_region_tuple` is provided when `ref` is not None.
        AssertionError: If the specified region or annotation is not found in the data.

    Example:
        >>> result = differential_gene_expression(
                target=my_data,
                target_annotation_tuple=("pathologist", "tumor"),
                ref=my_ref_data,
                ref_annotation_tuple=("cell_type", "astrocyte"),
                plot_volcano=True,
                method='wilcoxon'
            )
    """
    if not (show_volcano | return_results):
        raise ValueError("Both `show_volcano` and `return_results` are False. At least one of them must be True.")

    dge_comparison_column = "DGE_COMPARISON_COLUMN"

    # pre-flight checks
    if ref_annotation_tuple is not None:
        if ref_annotation_tuple == "rest":
            if ref is not None:
                raise ValueError("Value 'rest' for `ref_annotation_tuple` is only allowed if no reference data is given (`ref=None`).")
        elif ref_annotation_tuple == "same":
            ref_annotation_tuple = target_annotation_tuple
        elif not isinstance(ref_annotation_tuple, tuple):
            raise ValueError(f"Unknown type of `ref_annotation_tuple`: {type(ref_annotation_tuple)}. Must be either tuple, 'rest', 'same' or None.")
        else:
            pass

    if ref_region_tuple is not None:
        if ref_region_tuple == "rest":
            if ref is not None:
                raise ValueError("Value 'rest' for `ref_region_tuple` is only allowed if no reference data is given (`ref=None`).")
        elif ref_region_tuple == "same":
            ref_region_tuple = target_region_tuple
        elif not isinstance(ref_region_tuple, tuple):
            raise ValueError(f"Unknown type of `ref_region_tuple`: {type(ref_region_tuple)}. Must be either tuple, 'rest', 'same' or None.")
        else:
            pass

    if ref_cell_type_tuple is not None:
        if ref_cell_type_tuple == "rest":
            if ref is not None:
                raise ValueError("Value 'rest' for `ref_cell_type_tuple` is only allowed if no reference data is given (`ref=None`).")
        elif ref_cell_type_tuple == "same":
            ref_cell_type_tuple = target_cell_type_tuple
        elif not isinstance(ref_cell_type_tuple, tuple):
            raise ValueError(f"Unknown type of `ref_cell_type_tuple`: {type(ref_cell_type_tuple)}. Must be either tuple, 'rest', 'same' or None.")
        else:
            pass

    # select data for analysis
    adata_data = _select_data_for_dge(
        data=target,
        cells_layer=cells_layer,
        annotation_tuple=target_annotation_tuple,
        cell_type_tuple=target_cell_type_tuple,
        region_tuple=target_region_tuple,
        force_assignment=force_assignment,
        verbose=verbose
    )

    # original tuples for plotting the configuration table
    orig_ref_annotation_tuple = ref_annotation_tuple
    orig_ref_cell_type_tuple = ref_cell_type_tuple

    if ref is None:
        ref = target.copy()
        ref_celldata = _get_cell_layer(cells=ref.cells, cells_layer=cells_layer)

        # TODO: Implement behavior for "rest"
        # The "rest" argument is only implemented if ref_data is None in the beginning
        if ref_annotation_tuple == "rest":
            rest_annotations = [
                elem
                for elem in ref_celldata.matrix.obsm["annotations"][target_annotation_tuple[0]].unique()
                if elem != target_annotation_tuple[1]
                ]
            ref_annotation_tuple = (target_annotation_tuple[0], rest_annotations)

        if ref_region_tuple == "rest":
            rest_regions = [
                elem
                for elem in ref_celldata.matrix.obsm["regions"][target_region_tuple[0]].unique()
                if elem != target_region_tuple[1]
                ]
            ref_region_tuple = (target_region_tuple[0], rest_regions)

        if ref_cell_type_tuple == "rest":
            rest_cell_types = [
                elem
                for elem in ref_celldata.matrix.obs[target_cell_type_tuple[0]].unique()
                if elem != target_cell_type_tuple[1]
                ]
            ref_cell_type_tuple = (target_cell_type_tuple[0], rest_cell_types)

    if isinstance(ref, InSituData):
        # generate a list from ref_dta
        ref = [ref]
    elif isinstance(ref, list):
        assert np.all([isinstance(elem, InSituData) for elem in ref]), "Not all elements of list given in `ref` are InSituData objects."
    else:
        raise ValueError("`ref` must be an InSituData object or a list of InSituData objects.")

    adata_ref_list = []
    for rd in ref:
        # select reference data for analysis
        ad_ref = _select_data_for_dge(
            data=rd,
            cells_layer=cells_layer,
            annotation_tuple=ref_annotation_tuple,
            cell_type_tuple=ref_cell_type_tuple,
            region_tuple=ref_region_tuple,
            force_assignment=force_assignment,
            verbose=verbose
        )
        adata_ref_list.append(ad_ref)

    if len(adata_ref_list) > 1:
        adata_ref = anndata.concat(adata_ref_list)
    else:
        adata_ref = adata_ref_list[0]

    # concatenate and ignore user warning about observations being not unique since we take care of this later by filtering out duplicate values if wanted.
    with catch_warnings():
        filterwarnings("ignore", message="Observation names are not unique. To make them unique, call `.obs_names_make_unique`.")
        adata_combined = anndata.concat(
            {
                "DATA": adata_data,
                "REFERENCE": adata_ref
            },
            label=dge_comparison_column
        )

    if not exclude_ambiguous_assignments:
        # check whether cells with identical names are found in both data and reference and if yes give a warning
        if not set(adata_data.obs_names).isdisjoint(set(adata_ref.obs_names)):
            n_duplicated_cells = len(set(adata_data.obs_names).intersection(set(adata_ref.obs_names)))
            pct_duplicated_cells = round((n_duplicated_cells / 2) / (len(adata_data) + len(adata_data)) * 100, 1)

            warn(
                f"{n_duplicated_cells} ({pct_duplicated_cells}%) cells with identical names were found to belong to both data and reference. "
                "This can happen due to overlapping annotations or non-unique cell names in the individual datasets. "
                "If you are sure that the same cell cannot be found in both data and reference, you can ignore this warning. "
                "To exclude ambiguously assigned cells from the analysis, use `exclude_ambiguous_assignments=True`."
            )

    else:
        # check whether some cells are in both data and reference
        duplicated_mask = adata_combined.obs_names.duplicated(keep=False)

        if np.any(duplicated_mask):
            print("Exclude ambiguously assigned cells...")
            # remove duplicated values
            adata_combined = adata_combined[~duplicated_mask].copy()

    # add column to .obs for its use in rank_genes_groups()
    #adata_combined.obs = adata_combined.obs.filter([dge_comparison_column]) # empty obs

    print(f"Calculate differentially expressed genes with Scanpy's `rank_genes_groups` using '{method}'.")
    sc.tl.rank_genes_groups(adata=adata_combined,
                            groupby=dge_comparison_column,
                            groups=["DATA"],
                            reference="REFERENCE",
                            method=method,
                            )

    # create dataframe from results
    res_dict = create_deg_dataframe(
        adata=adata_combined, groups="DATA")
    df = res_dict["DATA"]

    if show_volcano:
        cell_counts = adata_combined.obs[dge_comparison_column].value_counts()
        data_counts = cell_counts["DATA"]
        ref_counts = cell_counts["REFERENCE"]

        n_upreg = np.sum((df["pvals"] <= significance_threshold) & (df["logfoldchanges"] > np.log2(fold_change_threshold)))
        n_downreg = np.sum((df["pvals"] <= significance_threshold) & (df["logfoldchanges"] < -np.log2(fold_change_threshold)))

        config_table = pd.DataFrame({
            "": ["Annotation", "Cell type", "Region", "Cell number", "DEG number"],
            "Reference": [elem[1] if isinstance(elem, tuple) else elem
                          for elem in [orig_ref_annotation_tuple, orig_ref_cell_type_tuple, ref_region_tuple]] + [ref_counts, n_downreg],
            "Target": [elem[1] if isinstance(elem, tuple) else elem
                       for elem in [target_annotation_tuple, target_cell_type_tuple, target_region_tuple]] + [data_counts, n_upreg]
        })

        # remove empty rows
        config_table = config_table.set_index("").dropna(how="all").reset_index()

        single_volcano(
            data=df,
            significance_threshold=significance_threshold,
            fold_change_threshold=fold_change_threshold,
            title=title,
            savepath = savepath,
            save_only = save_only,
            dpi_save = dpi_save,
            config_table = config_table,
            adjust_labels=True,
            **volcano_kwargs
            )
    if return_results:
        return {
            "results": df,
            "params": adata_combined.uns["rank_genes_groups"]["params"]
        }


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

def _run_deseq2_pseudobulk(adata, dge_setup):
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
    plot_qc: bool = False,
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
    stat_res = _run_deseq2_pseudobulk(pdata_ct, dge_setup=dge_setup)
    results_df = stat_res.results_df

    if pdata_nb is not None:
        # run DESeq2 for neighborhood data and return results
        stat_res_first = _run_deseq2_pseudobulk(pdata_first_condition, dge_setup=["obs_type", "cells", "neighbors"])
        stat_res_second = _run_deseq2_pseudobulk(pdata_second_condition, dge_setup=["obs_type", "cells", "neighbors"])
        results_df_nb_first = stat_res_first.results_df
        results_df_nb_second = stat_res_second.results_df

    # # plot volcano plot
    # if pdata_nb is not None:
    #     results_data = [results_df, results_df_nb_first, results_df_nb_second]
    #     titles = ["Cells", "Neighborhoods (condition A)", "Neighborhoods (condition B)"]
    # else:
    #     results_data = [results_df]
    #     titles = ["Cells"]

    # ncols = len(results_data)
    # fig, axs = plt.subplots(1, ncols, figsize=(6*ncols, 6))
    # for i, d in enumerate(results_data):
    #     axs[i].set_title(titles[i])
    #     dc.pl.volcano(
    #         d,
    #         x="log2FoldChange",
    #         y="pvalue",
    #         top=40,
    #         ax=axs[i]
    #         )
    # plt.tight_layout()
    # plt.show()

    results = DiffExprResults(
        main=results_df,
        nb_condition_a=results_df_nb_first if pdata_nb is not None else None,
        nb_condition_b=results_df_nb_second if pdata_nb is not None else None,
        metadata={
            "celltype": celltype,
            "dge_setup": dge_setup,
            "min_cells": min_cells,
            "min_counts": min_counts
        }
    )

    return results

    # if pdata_nb is not None:
    #     return results_df, results_df_nb_first, results_df_nb_second
    # else:
    #     return results_df

    # if pdata_nb is not None:

    #     volcano_nb(
    #         results_df_normal=results_df,
    #         results_df_nb_first=results_df_nb_first,
    #         significance_threshold=0.05,
    #         fold_change_threshold=0.5)
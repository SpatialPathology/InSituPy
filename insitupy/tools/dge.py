from numbers import Number
from typing import List, Literal, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings, warn

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from insitupy._core.data import InSituData
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.dataclasses.results import (DiffExprConfigCollector,
                                          DiffExprResults)
from insitupy.plotting.volcano import single_volcano
from insitupy.tools.neighbors import mean_gex_diff_to_neighbors
from insitupy.utils._dge import _select_data_for_dge
from insitupy.utils.dge import create_deg_dataframe

DGE_COMPARISON_COLUMN = "DGE_COMPARISON_COLUMN"

def dge(
    target: InSituData,
    target_annotation_tuple: Optional[Tuple[str, str]] = None,
    target_cell_type_tuple: Optional[Tuple[str, str]] = None,
    target_region_tuple: Optional[Tuple[str, str]] = None,
    target_name: Optional[str] = None,
    target_metadata: Optional[dict] = None,
    ref: Optional[Union[InSituData, List[InSituData]]] = None,
    ref_annotation_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
    ref_cell_type_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
    ref_region_tuple: Optional[Tuple[str, str]] = "same",
    ref_name: Optional[str] = None,
    ref_metadata: Optional[dict] = None,
    cells_layer: Optional[str] = None,
    consider_neighbors: bool = False,
    method: Optional[Literal['t-test', 'wilcoxon', 'logreg', 't-test_overestim_var']] = 't-test',
    exclude_ambiguous_assignments: bool = False,
    force_assignment: bool = False,
    verbose: bool = False,
    ) -> DiffExprResults:
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
        verbose (bool): Whether to print detailed information during the analysis. Defaults to False.

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

    # if not (show_volcano | return_results):
    #     raise ValueError("Both `show_volcano` and `return_results` are False. At least one of them must be True.")

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
    adata_target, adata_target_full = _select_data_for_dge(
        data=target,
        cells_layer=cells_layer,
        annotation_tuple=target_annotation_tuple,
        cell_type_tuple=target_cell_type_tuple,
        region_tuple=target_region_tuple,
        force_assignment=force_assignment,
        return_all_celltypes=True,
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
    adata_ref_full_list = []
    for rd in ref:
        # select reference data for analysis
        ad_ref, ad_ref_full = _select_data_for_dge(
            data=rd,
            cells_layer=cells_layer,
            annotation_tuple=ref_annotation_tuple,
            cell_type_tuple=ref_cell_type_tuple,
            region_tuple=ref_region_tuple,
            force_assignment=force_assignment,
            return_all_celltypes=True,
            verbose=verbose
        )
        adata_ref_list.append(ad_ref)
        adata_ref_full_list.append(ad_ref_full)

    if len(adata_ref_list) > 1:
        adata_ref = anndata.concat(adata_ref_list)
        adata_ref_full = anndata.concat(adata_ref_full_list)
    else:
        adata_ref = adata_ref_list[0]
        adata_ref_full = adata_ref_full_list[0]

    # concatenate and ignore user warning about observations being not unique since we take care of this later by filtering out duplicate values if wanted.
    with catch_warnings():
        filterwarnings("ignore", message="Observation names are not unique. To make them unique, call `.obs_names_make_unique`.")
        adata_combined = anndata.concat(
            {
                "DATA": adata_target,
                "REFERENCE": adata_ref
            },
            label=DGE_COMPARISON_COLUMN
        )

    if not exclude_ambiguous_assignments:
        # check whether cells with identical names are found in both data and reference and if yes give a warning
        if not set(adata_target.obs_names).isdisjoint(set(adata_ref.obs_names)):
            n_duplicated_cells = len(set(adata_target.obs_names).intersection(set(adata_ref.obs_names)))
            pct_duplicated_cells = round((n_duplicated_cells / 2) / (len(adata_target) + len(adata_target)) * 100, 1)

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
                            groupby=DGE_COMPARISON_COLUMN,
                            groups=["DATA"],
                            reference="REFERENCE",
                            method=method,
                            )

    # create dataframe from results
    res_dict = create_deg_dataframe(
        adata=adata_combined, groups="DATA")
    df = res_dict["DATA"]
    df = df.set_index("gene")

    # collect configuration
    method_params = adata_combined.uns["rank_genes_groups"]["params"]
    cell_counts = adata_combined.obs[DGE_COMPARISON_COLUMN].value_counts()
    data_counts = cell_counts["DATA"]
    ref_counts = cell_counts["REFERENCE"]

    config = DiffExprConfigCollector(
        mode="single-cell",
        method_params=method_params,
        cells_layer=cells_layer,
        exclude_ambiguous_assignments=exclude_ambiguous_assignments,

        target_annotation=target_annotation_tuple[1] if isinstance(target_annotation_tuple, tuple) else target_annotation_tuple,
        target_cell_type=target_cell_type_tuple[1] if isinstance(target_cell_type_tuple, tuple) else target_cell_type_tuple,
        target_region=target_region_tuple[1] if isinstance(target_region_tuple, tuple) else target_region_tuple,
        target_cell_number=data_counts,
        target_name=target_name,
        target_metadata=target_metadata,

        ref_annotation=orig_ref_annotation_tuple[1] if isinstance(orig_ref_annotation_tuple, tuple) else orig_ref_annotation_tuple,
        ref_cell_type=orig_ref_cell_type_tuple[1] if isinstance(orig_ref_cell_type_tuple, tuple) else orig_ref_cell_type_tuple,
        ref_region=ref_region_tuple[1] if isinstance(ref_region_tuple, tuple) else ref_region_tuple,
        ref_cell_number=ref_counts,
        ref_name=ref_name,
        ref_metadata=ref_metadata
    )

    if consider_neighbors:
        nb_results_target, _, _, _ = mean_gex_diff_to_neighbors(
            adata=adata_target_full,
            radius=20,
            celltype_tuple=target_cell_type_tuple,
            test=method,
        )

        nb_results_ref, _, _, _ = mean_gex_diff_to_neighbors(
            adata=adata_ref_full,
            radius=20,
            celltype_tuple=ref_cell_type_tuple,
            test=method,
        )
    else:
        nb_results_target = nb_results_ref = None


    res = DiffExprResults(
        main=df,
        config=config,
        target_neighborhood=nb_results_target,
        ref_neighborhood=nb_results_ref
        )

    return res

    # if show_volcano:
    #     cell_counts = adata_combined.obs[DGE_COMPARISON_COLUMN].value_counts()
    #     data_counts = cell_counts["DATA"]
    #     ref_counts = cell_counts["REFERENCE"]

    #     n_upreg = np.sum((df["pvalue"] <= significance_threshold) & (df["log2foldchange"] > np.log2(foldchange_threshold)))
    #     n_downreg = np.sum((df["pvalue"] <= significance_threshold) & (df["log2foldchange"] < -np.log2(foldchange_threshold)))

    #     config_table = pd.DataFrame({
    #         "": ["Annotation", "Cell type", "Region", "Cell number", "DEG number"],
    #         "Reference": [elem[1] if isinstance(elem, tuple) else elem
    #                       for elem in [orig_ref_annotation_tuple, orig_ref_cell_type_tuple, ref_region_tuple]] + [ref_counts, n_downreg],
    #         "Target": [elem[1] if isinstance(elem, tuple) else elem
    #                    for elem in [target_annotation_tuple, target_cell_type_tuple, target_region_tuple]] + [data_counts, n_upreg]
    #     })

    #     # remove empty rows
    #     config_table = config_table.set_index("").dropna(how="all").reset_index()

    #     single_volcano(
    #         data=df,
    #         significance_threshold=significance_threshold,
    #         foldchange_threshold=foldchange_threshold,
    #         title=title,
    #         savepath = savepath,
    #         save_only = save_only,
    #         dpi_save = dpi_save,
    #         config = config_table,
    #         adjust_labels=True,
    #         **volcano_kwargs
    #         )
    # if return_results:
    #     return {
    #         "results": df,
    #         "params": adata_combined.uns["rank_genes_groups"]["params"]
    #     }



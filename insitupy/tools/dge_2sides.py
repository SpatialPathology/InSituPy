import os
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings, warn

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import radius_neighbors_graph

from insitupy._core.data import InSituData
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.plotting import volcano
from insitupy.utils._dge import _select_data_for_dge
from insitupy.utils.dge import create_deg_dataframe
from insitupy.plotting import volcano_two_sides


def differential_gene_expression_two_sides(
    target: InSituData,
    target_annotation_tuple: Optional[Tuple[str, str]] = None,
    target_cell_type_tuple: Optional[Tuple[str, str]] = None,
    target_region_tuple: Optional[Tuple[str, str]] = None,
    radius: Number = 100,
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

    #defining neighborhood
    coords = target.cells[cells_layer].matrix.obsm["spatial"]
    A = radius_neighbors_graph(coords, radius=radius, mode="connectivity", include_self=False)
    
    target_mask = target.cells[cells_layer].matrix.obs[target_cell_type_tuple[0]] == target_cell_type_tuple[1]
    target_idx = np.where(target_mask)[0]
    
    neighbors = A[target_idx].nonzero()[1]   
    neighbors = np.unique(neighbors)
    
    neighbors_non_target_celltype = [i for i in neighbors if target.cells[cells_layer].matrix.obs.iloc[i][target_cell_type_tuple[0]] != target_cell_type_tuple[1]]
    
    target.cells[cells_layer].matrix.obs["neighbors"] = "other"
    target.cells[cells_layer].matrix.obs.iloc[target_idx, target.cells[cells_layer].matrix.obs.columns.get_loc("neighbors")] = target_cell_type_tuple[1]
    target.cells[cells_layer].matrix.obs.iloc[neighbors_non_target_celltype,  target.cells[cells_layer].matrix.obs.columns.get_loc("neighbors")] = "neighbors"
    
    print(target.cells[cells_layer].matrix.obs['neighbors'].unique())
    
    print(f"Calculate differentially expressed genes with Scanpy's `rank_genes_groups` using '{method}' for target dataset and neighbors")
    sc.tl.rank_genes_groups(adata=target.cells[cells_layer].matrix,
                            groupby="neighbors",
                            groups=[target_cell_type_tuple[1]],
                            reference="neighbors",
                            method=method,
                            )
    
    
    # create dataframe from neighbor_results
    neighbors_dict = create_deg_dataframe(
        adata=target.cells[cells_layer].matrix, groups=target_cell_type_tuple[1])
    df_neighbors = neighbors_dict[target_cell_type_tuple[1]]

    if show_volcano:
        cell_counts_neighbors = target.cells[cells_layer].matrix.obs['neighbors'].value_counts()
        data_counts_neighbors = cell_counts_neighbors[target_cell_type_tuple[1]]
        ref_counts_neighbors = cell_counts_neighbors["neighbors"]

        n_upreg_neigh = np.sum((df_neighbors["pvals"] <= significance_threshold) & (df_neighbors["logfoldchanges"] > np.log2(fold_change_threshold)))
        n_downreg_neigh = np.sum((df_neighbors["pvals"] <= significance_threshold) & (df_neighbors["logfoldchanges"] < -np.log2(fold_change_threshold)))

        config_table_neighbors = pd.DataFrame({
            "": ["Cell number", "D [EG number"],
            "Neighbors": [ref_counts_neighbors, n_downreg_neigh],
            "Target": [data_counts_neighbors, n_upreg_neigh]
        })
        config_table_neighbors = config_table_neighbors.set_index("").dropna(how="all").reset_index()
    
    
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
    

    # plotting
        volcano(
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
        
        volcano(
            data=df_neighbors,
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
        
        merged=volcano_two_sides(
            data_x=df,
            data_y=df_neighbors,
            fold_change_threshold=fold_change_threshold,
            title=title,
            **volcano_kwargs) 
                
    if return_results:
        return {
            "results": df,
            "params": adata_combined.uns["rank_genes_groups"]["params"],
            "merged_df":merged
        }
    
    
      
       
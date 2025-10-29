from numbers import Number
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import radius_neighbors_graph
from statsmodels.stats import multitest
from tqdm import tqdm

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_inputs(
    adata,
    radius: Number,
    obs_key: str,
    celltype_col: Optional[str],
    celltype: Optional[str],
    test: str,
    min_cells: Number,
) -> None:
    """Validate common input parameters."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if min_cells < 2:
        raise ValueError("min_cells must be at least 2")
    if test not in ["wilcoxon", "t-test"]:
        raise ValueError("test must be 'wilcoxon' or 't-test'")
    if obs_key not in adata.obsm:
        raise KeyError(f"obs_key '{obs_key}' not found in adata.obsm")

    # Cell type validation
    if celltype_col is not None:
        if celltype_col not in adata.obs.columns:
            raise KeyError(f"celltype_col '{celltype_col}' not found in adata.obs")
        if celltype is None:
            raise ValueError("Must specify 'celltype' when 'celltype_col' is provided")
        if celltype not in adata.obs[celltype_col].values:
            raise ValueError(f"celltype '{celltype}' not found in adata.obs['{celltype_col}']")
    elif celltype is not None:
        raise ValueError("Cannot specify 'celltype' without 'celltype_col'")


def _prepare_data(
    adata,
    genes_subset: Optional[List[str]],
    verbose: bool
) -> Tuple[List[str], np.ndarray, int]:
    """Prepare gene data and indices."""
    all_genes = adata.var_names.tolist()

    if genes_subset is not None:
        gene_mask = adata.var_names.isin(genes_subset)
        if gene_mask.sum() == 0:
            raise ValueError("None of the genes in genes_subset found in adata.var_names")
        genes = [g for g in all_genes if g in genes_subset]
        gene_indices = np.where(gene_mask)[0]
        if verbose:
            print(f"Testing subset of {len(genes)} genes (out of {len(all_genes)} total)")
    else:
        genes = all_genes
        gene_indices = np.arange(len(all_genes))

    n_genes_total = len(all_genes)
    return genes, gene_indices, n_genes_total


def _build_spatial_graph(
    coords: np.ndarray,
    radius: float,
    celltype_col: Optional[str],
    celltype: Optional[str],
    adata,
    exclude_self: bool,
    use_distance_weighting: bool,
    verbose: bool
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, int]:
    """Build spatial adjacency graph with optional cell type filtering."""
    n_cells = coords.shape[0]

    if verbose:
        print(f"Building spatial graph (radius={radius}, weighted={use_distance_weighting})...")

    # Build radius-based adjacency
    mode = "distance" if use_distance_weighting else "connectivity"
    A = radius_neighbors_graph(coords, radius=radius, mode=mode, include_self=not exclude_self)

    # Get target cells mask
    if celltype_col is not None:
        target_mask = (adata.obs[celltype_col] == celltype).values
        n_target_cells = target_mask.sum()
        if verbose:
            print(f"Selected {n_target_cells} '{celltype}' cells as targets")

        # Exclude same-type neighbors
        if verbose:
            print("Excluding same-celltype neighbors from adjacency matrix...")
        A_ct = A.copy()
        A_ct = A_ct.tocoo()

        ct_labels = adata.obs[celltype_col].values
        keep = []
        for i, (r, c) in enumerate(zip(A_ct.row, A_ct.col)):
            if not (target_mask[r] and ct_labels[r] == ct_labels[c] and r != c):
                keep.append(i)

        if keep:
            keep = np.array(keep)
            A_ct = sparse.coo_matrix(
                (A_ct.data[keep], (A_ct.row[keep], A_ct.col[keep])),
                shape=A_ct.shape
            )
        else:
            A_ct = sparse.coo_matrix(A_ct.shape)

        A_ct = A_ct.tocsr()
        A_ct.eliminate_zeros()
    else:
        target_mask = np.ones(n_cells, dtype=bool)
        n_target_cells = n_cells
        A_ct = A

    # Apply distance weighting if requested
    if use_distance_weighting:
        A_ct = A_ct.copy()
        A_ct.data = 1.0 / (A_ct.data + 1e-10)

    # Normalize rows to sum to 1
    deg = np.array(A_ct.sum(axis=1)).squeeze()
    deg[deg == 0] = 1
    D_inv = sparse.diags(1.0 / deg)
    A_ct = D_inv @ A_ct

    return A_ct, target_mask, deg, n_target_cells


def _compute_neighbor_mean(
    A_ct: sparse.csr_matrix,
    X: np.ndarray,
    gene_indices: np.ndarray,
    verbose: bool
) -> np.ndarray:
    """Compute mean neighbor expression using matrix multiplication.

    Note: A_ct is already normalized (rows sum to 1), but cells without neighbors
    will have all-zero rows even after normalization.
    """
    if verbose:
        print("Computing mean neighbor expression...")

    is_sparse = sparse.issparse(X)
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_subset = X[:, gene_indices]
    else:
        X_subset = X

    # Matrix multiplication for mean
    neighbor_expr = A_ct @ X_subset

    if is_sparse:
        neighbor_expr = neighbor_expr.toarray()

    # Set NaN for cells without neighbors
    # After normalization, cells without neighbors have row sum = 0
    # (because they started with 0 and got divided by 1)
    row_sums = np.array(A_ct.sum(axis=1)).squeeze()
    cells_without_neighbors = row_sums == 0
    neighbor_expr[cells_without_neighbors, :] = np.nan

    return neighbor_expr


def _compute_neighbor_max(
    A_ct: sparse.csr_matrix,
    X: np.ndarray,
    gene_indices: np.ndarray,
    target_mask: np.ndarray,
    exclude_zeros: bool,
    verbose: bool
) -> np.ndarray:
    """Compute maximum neighbor expression per gene."""
    n_cells = A_ct.shape[0]
    print(n_cells)
    n_genes = len(gene_indices) if gene_indices is not None else X.shape[1]

    if verbose:
        print("Computing max neighbor expression (per-gene maximum)...")

    is_sparse = sparse.issparse(X)
    neighbor_expr = np.full((n_cells, n_genes), np.nan)

    # Only compute for target cells
    target_indices = np.where(target_mask)[0]

    for idx, i in enumerate(tqdm(target_indices, desc="cells") if verbose else target_indices):
        neighbors = A_ct[i].indices
        if len(neighbors) == 0:
            continue

        if gene_indices is not None:
            neighbor_exprs = X[neighbors][:, gene_indices]
        else:
            neighbor_exprs = X[neighbors]

        if is_sparse:
            neighbor_exprs = neighbor_exprs.toarray()

        if exclude_zeros:
            max_vals = np.zeros(n_genes)
            for g in range(n_genes):
                expr_g = neighbor_exprs[:, g]
                nonzero = expr_g[expr_g != 0]
                if len(nonzero) > 0:
                    max_vals[g] = np.max(nonzero)
                else:
                    max_vals[g] = 0
            max_vals[np.isinf(max_vals)] = np.nan
            neighbor_expr[i] = max_vals
        else:
            neighbor_expr[i] = np.max(neighbor_exprs, axis=0)

    return neighbor_expr


def _run_statistical_tests(
    gex_diff: np.ndarray,
    target_expr: np.ndarray,
    neighbor_expr: np.ndarray,
    target_mask: np.ndarray,
    genes: List[str],
    test: str,
    min_cells: int,
    batch_size: Optional[int],
    verbose: bool
) -> Dict[str, np.ndarray]:
    """Run per-gene statistical tests and compute fold changes."""
    n_genes = len(genes)

    if verbose:
        print(f"Running paired tests ({test}) per gene and computing effect sizes...")

    min_samples_needed = max(min_cells, 2 if test == "wilcoxon" else 1)

    # Initialize result arrays
    results = {
        'n_target_cells': np.zeros(n_genes, dtype=int),
        'n_cells_used': np.zeros(n_genes, dtype=int),
        'n_expressed': np.zeros(n_genes, dtype=int),
        'mean_target': np.full(n_genes, np.nan),
        'mean_neighbor': np.full(n_genes, np.nan),
        'log2_fold_change': np.full(n_genes, np.nan),
        'log2_fold_change_paired': np.full(n_genes, np.nan),
        'stat_vals': np.full(n_genes, np.nan),
        'pvals': np.full(n_genes, np.nan)
    }

    # Process in batches if requested
    if batch_size is None:
        batch_ranges = [(0, n_genes)]
    else:
        batch_ranges = [(i, min(i + batch_size, n_genes))
                       for i in range(0, n_genes, batch_size)]

    for batch_start, batch_end in batch_ranges:
        if verbose and len(batch_ranges) > 1:
            print(f"Processing batch {batch_start}-{batch_end} of {n_genes} genes...")

        iterator = tqdm(range(batch_start, batch_end), desc="genes") if verbose else range(batch_start, batch_end)

        for g in iterator:
            # Only use target cells for analysis
            diffs = gex_diff[target_mask, g]

            # Total number of target cells for this gene
            n_total = len(diffs)
            results['n_target_cells'][g] = n_total

            # Number of target cells with neighbors (finite diff values)
            ok = np.isfinite(diffs)
            n = ok.sum()
            results['n_cells_used'][g] = n

            if n < min_samples_needed:
                continue

            d = diffs[ok]

            # Track which cells actually express this gene
            expr_g = target_expr[ok, g]
            expressed_mask = expr_g > 0
            results['n_expressed'][g] = expressed_mask.sum()

            # Get target and neighbor expression for fold change
            target_expr_g = target_expr[ok, g]
            neighbor_expr_g = neighbor_expr[target_mask][ok, g]

            # PAIRED FOLD CHANGE
            mean_d = np.mean(d)
            results['log2_fold_change_paired'][g] = mean_d / np.log(2)

            # Calculate mean expression in log space
            mean_target_log = np.mean(target_expr_g)
            mean_neighbor_log = np.mean(neighbor_expr_g)
            results['mean_target'][g] = mean_target_log
            results['mean_neighbor'][g] = mean_neighbor_log

            # UNPAIRED FOLD CHANGE
            mean_target_linear = np.expm1(mean_target_log)
            mean_neighbor_linear = np.expm1(mean_neighbor_log)
            fc = (mean_target_linear + 1e-9) / (mean_neighbor_linear + 1e-9)
            results['log2_fold_change'][g] = np.log2(fc)

            try:
                if test == "t-test":
                    tstat, pval = stats.ttest_1samp(
                        d, 0.0,
                        alternative="two-sided",
                        nan_policy="omit"
                    )
                    #results['stat_vals'][g], results['pvals'][g] = tstat, max(pval, 1e-300)
                    results['stat_vals'][g], results['pvals'][g] = tstat, pval

                elif test == "wilcoxon":
                    if n >= 2:
                        wstat, pval = stats.wilcoxon(
                            d, zero_method="wilcox",
                            alternative="two-sided"
                        )
                        results['stat_vals'][g], results['pvals'][g] = wstat, pval
                        #results['stat_vals'][g], results['pvals'][g] = wstat, max(pval, 1e-300)

            except Exception as e:
                if verbose:
                    print(f"Warning: Test failed for gene {genes[g]}: {e}")
                pass

    return results


def _apply_multiple_testing_correction(
    pvals: np.ndarray,
    correction_method: str,
    verbose: bool
) -> np.ndarray:
    """Apply multiple testing correction."""
    if verbose:
        print("Applying multiple-testing correction...")

    finite = np.isfinite(pvals)
    p_adj = np.full_like(pvals, np.nan)
    if finite.sum() > 0:
        _, p_adj[finite], _, _ = multitest.multipletests(
            pvals[finite], alpha=0.05, method=correction_method
        )

    return p_adj


def _create_qc_stats(
    n_cells: int,
    n_target_cells: int,
    celltype: Optional[str],
    celltype_col: Optional[str],
    n_genes: int,
    n_genes_total: int,
    deg: np.ndarray,
    target_mask: np.ndarray,
    finite_pvals: int,
    test: str,
    correction_method: str,
    radius: float,
    min_cells: int,
    use_distance_weighting: bool,
    neighbor_summary: str,
    exclude_zeros_from_max: Optional[bool] = None
) -> Dict:
    """Create QC statistics dictionary."""
    target_deg = deg[target_mask]

    return {
        "n_cells_total": n_cells,
        "n_cells_target": n_target_cells,
        "target_celltype": celltype if celltype_col is not None else None,
        "n_genes_tested": n_genes,
        "n_genes_total": n_genes_total,
        "target_cells_without_neighbors": int((target_deg == 0).sum()),
        "fraction_target_cells_without_neighbors": float((target_deg == 0).mean()),
        "median_neighbors_target": float(np.median(target_deg[target_deg > 0])) if (target_deg > 0).any() else 0,
        "mean_neighbors_target": float(target_deg.mean()),
        "max_neighbors_target": int(target_deg.max()) if len(target_deg) > 0 else 0,
        "fraction_testable_genes": float(finite_pvals / n_genes),
        "test_used": test,
        "correction_method": correction_method,
        "radius": float(radius),
        "min_cells": int(min_cells),
        "distance_weighted": use_distance_weighting,
        "neighbor_summary": neighbor_summary,
        "exclude_zeros_from_max": exclude_zeros_from_max if neighbor_summary == "max" else None,
    }


def _print_summary(qc_stats: Dict, verbose: bool) -> None:
    """Print analysis summary."""
    if not verbose:
        return

    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    if qc_stats['target_celltype'] is not None:
        print(f"Target cell type: '{qc_stats['target_celltype']}'")
        print(f"Target cells: {qc_stats['n_cells_target']}/{qc_stats['n_cells_total']} "
              f"({qc_stats['n_cells_target']/qc_stats['n_cells_total']:.1%})")
    else:
        print(f"Cells analyzed: {qc_stats['n_cells_total']}")
    print(f"Genes tested: {qc_stats['n_genes_tested']}")
    print(f"Target cells without neighbors: {qc_stats['target_cells_without_neighbors']} "
          f"({qc_stats['fraction_target_cells_without_neighbors']:.1%})")
    print(f"Median neighbors per target cell: {qc_stats['median_neighbors_target']:.1f}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def mean_gex_diff_to_neighbors(
    adata,
    radius: Number = 20.0,
    obs_key: str = "spatial",
    celltype_tuple: Optional[Tuple[str, str]] = None,
    # celltype_col: Optional[str] = None,
    # celltype: Optional[str] = None,
    exclude_self: bool = True,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: str = "fdr_bh",
    min_cells: Number = 3,
    genes_subset: Optional[List[str]] = None,
    use_distance_weighting: bool = False,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray, Dict]:
    """
    Cell-type-specific spatial gene expression contamination analysis using mean neighbor expression.

    Identifies potential spatial contamination by comparing each cell's gene expression
    to the mean expression of its spatial neighborhood.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial coordinates and gene expression.
    radius : Number, default=20.0
        Radius in coordinate units for defining spatial neighbors.
    obs_key : str, default="spatial"
        Key in adata.obsm containing spatial coordinates.
    celltype_tuple : Tuple[str, str], optional
        Tuple specifying an observation key and value to filter the data by cell type.
    exclude_self : bool, default=True
        Whether to exclude the cell itself from its neighborhood.
    test : {"wilcoxon", "t-test"}, default="wilcoxon"
        Statistical test to use.
    correction_method : str, default="fdr_bh"
        Multiple testing correction method.
    min_cells : Number, default=3
        Minimum number of cells required for a valid test.
    genes_subset : List[str], optional
        List of gene names to analyze.
    use_distance_weighting : bool, default=False
        Weight neighbor contributions by inverse distance.
    batch_size : int, optional
        Process genes in batches to reduce memory usage.
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    results : pd.DataFrame
        Per-gene statistics with contamination metrics.
    adjacency_matrix : sparse.csr_matrix
        Neighbor adjacency matrix.
    diff_matrix : np.ndarray
        Per-cell gene expression differences.
    qc_stats : dict
        Quality control statistics.
    """
    # Validate inputs
    celltype_col, celltype = celltype_tuple if celltype_tuple is not None else (None, None)
    _validate_inputs(adata, radius, obs_key, celltype_col, celltype, test, min_cells)

    # Prepare data
    coords = np.asarray(adata.obsm[obs_key])
    X = adata.X
    genes, gene_indices, n_genes_total = _prepare_data(adata, genes_subset, verbose)
    n_genes = len(genes)
    n_cells = coords.shape[0]

    # Build spatial graph
    A_ct, target_mask, deg, n_target_cells = _build_spatial_graph(
        coords, radius, celltype_col, celltype, adata,
        exclude_self, use_distance_weighting, verbose
    )

    # Compute mean neighbor expression
    neighbor_expr = _compute_neighbor_mean(A_ct, X, gene_indices, verbose)

    # Compute differences
    if verbose:
        print("Computing gex_diff = gex_target - gex_neighbor ...")

    X_dense = X.toarray() if sparse.issparse(X) else X
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_dense = X_dense[:, gene_indices]

    gex_diff = np.full((n_cells, n_genes), np.nan)
    gex_diff[target_mask] = X_dense[target_mask] - neighbor_expr[target_mask]
    target_expr = X_dense[target_mask]

    # Run statistical tests
    test_results = _run_statistical_tests(
        gex_diff, target_expr, neighbor_expr, target_mask,
        genes, test, min_cells, batch_size, verbose
    )

    # Apply correction
    p_adj = _apply_multiple_testing_correction(
        test_results['pvals'], correction_method, verbose
    )

    # Create results DataFrame
    results = pd.DataFrame({
        "gene": genes,
        "n_target_cells": test_results['n_target_cells'],
        "n_cells_used": test_results['n_cells_used'],
        "n_cells_expressed": test_results['n_expressed'],
        "mean_target": test_results['mean_target'],
        "mean_neighbor": test_results['mean_neighbor'],
        "log2foldchange": test_results['log2_fold_change_paired'],
        "log2foldchange_unpaired": test_results['log2_fold_change'],
        "pvalue": test_results['pvals'],
        "padj": p_adj
    }).set_index("gene")

    # Add contamination score
    epsilon = 0.1
    results['contamination_score'] = -results['log2foldchange'] / (results['mean_target'] + epsilon)
    results['contamination_score'] = results['contamination_score'].replace([np.inf, -np.inf], np.nan)

    # Create QC stats
    qc_stats = _create_qc_stats(
        n_cells, n_target_cells, celltype, celltype_col,
        n_genes, n_genes_total, deg, target_mask,
        np.isfinite(test_results['pvals']).sum(),
        test, correction_method, radius, min_cells,
        use_distance_weighting, "mean"
    )

    _print_summary(qc_stats, verbose)

    return results, A_ct, gex_diff, qc_stats


def max_gex_diff_to_neighbors(
    adata,
    radius: Number = 20.0,
    obs_key: str = "spatial",
    celltype_col: Optional[str] = None,
    celltype: Optional[str] = None,
    exclude_self: bool = True,
    test: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: str = "fdr_bh",
    min_cells: Number = 3,
    genes_subset: Optional[List[str]] = None,
    exclude_zeros_from_max: bool = True,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray, Dict]:
    """
    Estimate maximum gene expression gradients in spatial neighborhoods.

    Compares each cell's expression to the maximum expression in its spatial neighborhood
    for each gene independently. More sensitive to localized contamination sources as it
    creates a "worst-case neighbor" profile.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial coordinates and gene expression.
    radius : Number, default=20.0
        Radius in coordinate units for defining spatial neighbors.
    obs_key : str, default="spatial"
        Key in adata.obsm containing spatial coordinates.
    celltype_col : str, optional
        Column name in adata.obs containing cell type labels.
    celltype : str, optional
        Target cell type to analyze.
    exclude_self : bool, default=True
        Whether to exclude the cell itself from its neighborhood.
    test : {"wilcoxon", "t-test"}, default="wilcoxon"
        Statistical test to use.
    correction_method : str, default="fdr_bh"
        Multiple testing correction method.
    min_cells : Number, default=3
        Minimum number of cells required for a valid test.
    genes_subset : List[str], optional
        List of gene names to analyze.
    exclude_zeros_from_max : bool, default=True
        Exclude zero values when computing maximum (zeros cannot be contamination sources).
    batch_size : int, optional
        Process genes in batches to reduce memory usage.
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    results : pd.DataFrame
        Per-gene statistics with gradient metrics.
    adjacency_matrix : sparse.csr_matrix
        Neighbor adjacency matrix.
    diff_matrix : np.ndarray
        Per-cell gene expression differences from maximum neighbor.
    qc_stats : dict
        Quality control statistics.
    """
    # Validate inputs
    _validate_inputs(adata, radius, obs_key, celltype_col, celltype, test, min_cells)

    # Prepare data
    coords = np.asarray(adata.obsm[obs_key])
    X = adata.X
    genes, gene_indices, n_genes_total = _prepare_data(adata, genes_subset, verbose)
    n_genes = len(genes)
    n_cells = coords.shape[0]

    # Build spatial graph (no distance weighting for max)
    A_ct, target_mask, deg, n_target_cells = _build_spatial_graph(
        coords, radius, celltype_col, celltype, adata,
        exclude_self, use_distance_weighting=False, verbose=verbose
    )

    # Compute maximum neighbor expression
    neighbor_expr = _compute_neighbor_max(
        A_ct, X, gene_indices, target_mask, exclude_zeros_from_max, verbose
    )

    # Compute differences
    if verbose:
        print("Computing gex_diff = gex_target - gex_neighbor_max ...")

    X_dense = X.toarray() if sparse.issparse(X) else X
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_dense = X_dense[:, gene_indices]

    gex_diff = np.full((n_cells, n_genes), np.nan)
    gex_diff[target_mask] = X_dense[target_mask] - neighbor_expr[target_mask]
    target_expr = X_dense[target_mask]

    # Run statistical tests
    test_results = _run_statistical_tests(
        gex_diff, target_expr, neighbor_expr, target_mask,
        genes, test, min_cells, batch_size, verbose
    )

    # Apply correction
    p_adj = _apply_multiple_testing_correction(
        test_results['pvals'], correction_method, verbose
    )

    # Create results DataFrame with gradient-specific naming
    results = pd.DataFrame({
        "gene": genes,
        "n_target_cells": test_results['n_target_cells'],
        "n_cells_used": test_results['n_cells_used'],
        "n_cells_expressed": test_results['n_expressed'],
        "mean_target": test_results['mean_target'],
        "mean_neighbor_max": test_results['mean_neighbor'],
        "log2_gradient_paired": test_results['log2_fold_change_paired'],
        "log2_gradient": test_results['log2_fold_change'],
        "p_value": test_results['pvals'],
        "p_adj": p_adj
    }).set_index("gene")

    # Add contamination score
    epsilon = 0.1
    results['contamination_score'] = -results['log2_gradient_paired'] / (results['mean_target'] + epsilon)
    results['contamination_score'] = results['contamination_score'].replace([np.inf, -np.inf], np.nan)

    # Create QC stats
    qc_stats = _create_qc_stats(
        n_cells, n_target_cells, celltype, celltype_col,
        n_genes, n_genes_total, deg, target_mask,
        np.isfinite(test_results['pvals']).sum(),
        test, correction_method, radius, min_cells,
        use_distance_weighting=False, neighbor_summary="max",
        exclude_zeros_from_max=exclude_zeros_from_max
    )

    _print_summary(qc_stats, verbose)

    return results, A_ct, gex_diff, qc_stats
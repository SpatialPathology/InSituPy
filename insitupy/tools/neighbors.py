import logging
import warnings
from numbers import Number
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import radius_neighbors_graph
from statsmodels.stats import multitest
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_inputs(
    adata,
    radius: Number,
    obs_key: str,
    celltype_col: Optional[str],
    celltype: Optional[str],
    method: str,
    min_cells: Number,
    strategy: str,
) -> None:
    """Validate common input parameters."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if min_cells < 2:
        raise ValueError("min_cells must be at least 2")
    if method not in ["wilcoxon", "t-test"]:
        raise ValueError("method must be 'wilcoxon' or 't-test'")
    if strategy not in ["mean", "max"]:
        raise ValueError("strategy must be 'mean' or 'max'")
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
            logger.info("Testing subset of %d genes (out of %d total)", len(genes), len(all_genes))
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
    strategy: str,
    verbose: bool
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, int]:
    """Build spatial adjacency graph with optional cell type filtering."""
    n_cells = coords.shape[0]

    if verbose:
        weighted_str = f", weighted={use_distance_weighting}" if strategy == "mean" else ""
        logger.info("Building spatial graph (radius=%s%s)...", radius, weighted_str)

    # Build radius-based adjacency
    # For mean with distance weighting, use distance mode; otherwise connectivity
    mode = "distance" if (strategy == "mean" and use_distance_weighting) else "connectivity"
    A = radius_neighbors_graph(coords, radius=radius, mode=mode, include_self=not exclude_self)

    # Get target cells mask
    if celltype_col is not None:
        target_mask = (adata.obs[celltype_col] == celltype).values
        n_target_cells = target_mask.sum()
        if verbose:
            logger.info("Selected %d '%s' cells as targets", n_target_cells, celltype)

        # Exclude same-type neighbors
        if verbose:
            logger.info("Excluding same-celltype neighbors from adjacency matrix...")
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

    # Apply distance weighting if requested (only for mean strategy)
    if strategy == "mean" and use_distance_weighting:
        A_ct = A_ct.copy()
        A_ct.data = 1.0 / (A_ct.data + 1e-10)

    # Normalize rows to sum to 1 (only for mean strategy)
    deg = np.array(A_ct.sum(axis=1)).squeeze()
    if strategy == "mean":
        deg_norm = deg.copy()
        deg_norm[deg_norm == 0] = 1
        D_inv = sparse.diags(1.0 / deg_norm)
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
        logger.info("Computing mean neighbor expression...")

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
    n_genes = len(gene_indices) if gene_indices is not None else X.shape[1]

    if verbose:
        logger.info("Computing max neighbor expression (per-gene maximum)...")

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
    method: str,
    min_cells: int,
    batch_size: Optional[int],
    verbose: bool
) -> Dict[str, np.ndarray]:
    """Run per-gene statistical tests and compute fold changes."""
    n_genes = len(genes)

    if verbose:
        logger.info("Running paired tests (%s) per gene and computing effect sizes...", method)

    min_samples_needed = max(min_cells, 2 if method == "wilcoxon" else 1)

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
            logger.info("Processing batch %d-%d of %d genes...", batch_start, batch_end, n_genes)

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
                if method == "t-test":
                    tstat, pval = stats.ttest_1samp(
                        d, 0.0,
                        alternative="two-sided",
                        nan_policy="omit"
                    )
                    results['stat_vals'][g], results['pvals'][g] = tstat, pval

                elif method == "wilcoxon":
                    if n >= 2:
                        wstat, pval = stats.wilcoxon(
                            d, zero_method="wilcox",
                            alternative="two-sided"
                        )
                        results['stat_vals'][g], results['pvals'][g] = wstat, pval

            except Exception as e:
                if verbose:
                    logger.warning("Test failed for gene %s: %s", genes[g], e)
                pass

    return results


def _apply_multiple_testing_correction(
    pvals: np.ndarray,
    correction_method: str,
    verbose: bool
) -> np.ndarray:
    """Apply multiple testing correction."""
    if verbose:
        logger.info("Applying multiple-testing correction...")

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
    method: str,
    correction_method: str,
    radius: float,
    min_cells: int,
    use_distance_weighting: bool,
    strategy: str,
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
        "test_used": method,
        "correction_method": correction_method,
        "radius": float(radius),
        "min_cells": int(min_cells),
        "distance_weighted": use_distance_weighting if strategy == "mean" else False,
        "strategy": strategy,
        "exclude_zeros_from_max": exclude_zeros_from_max if strategy == "max" else None,
    }


def _print_summary(qc_stats: Dict, verbose: bool) -> None:
    """Print analysis summary."""
    if not verbose:
        return

    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    if qc_stats['target_celltype'] is not None:
        logger.info("Target cell type: '%s'", qc_stats['target_celltype'])
        logger.info("Target cells: %d/%d (%.1f%%)",
                     qc_stats['n_cells_target'], qc_stats['n_cells_total'],
                     qc_stats['n_cells_target'] / qc_stats['n_cells_total'] * 100)
    else:
        logger.info("Cells analyzed: %d", qc_stats['n_cells_total'])
    logger.info("Strategy: %s", qc_stats['strategy'])
    logger.info("Genes tested: %d", qc_stats['n_genes_tested'])
    logger.info("Target cells without neighbors: %d (%.1f%%)",
                 qc_stats['target_cells_without_neighbors'],
                 qc_stats['fraction_target_cells_without_neighbors'] * 100)
    logger.info("Median neighbors per target cell: %.1f", qc_stats['median_neighbors_target'])
    logger.info("=" * 60)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def calculate_gex_diff_to_neighbors(
    adata,
    radius: Number = 20.0,
    obs_key: str = "spatial",
    celltype_tuple: Optional[Tuple[str, str]] = None,
    exclude_self: bool = True,
    strategy: Literal["mean", "max"] = "mean",
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    test: Literal["wilcoxon", "t-test"] = None,
    correction_method: str = "fdr_bh",
    min_cells: Number = 3,
    genes_subset: Optional[List[str]] = None,
    use_distance_weighting: bool = False,
    exclude_zeros_from_max: bool = True,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray, Dict]:
    """
    Cell-type-specific spatial gene expression contamination analysis.

    Identifies potential spatial contamination by comparing each cell's gene expression
    to its spatial neighborhood using either mean or maximum neighbor expression.

    Args:
        adata (AnnData): Annotated data object containing spatial coordinates and gene
            expression.
        radius (Number): Radius in coordinate units for defining spatial neighbors.
            Default is 20.0.
        obs_key (str): Key in adata.obsm containing spatial coordinates.
            Default is "spatial".
        celltype_tuple (Tuple[str, str], optional): Tuple specifying (celltype_col,
            celltype) to filter by cell type.
        exclude_self (bool): Whether to exclude the cell itself from its neighborhood.
            Default is True.
        strategy (str): Strategy for computing neighbor expression: "mean" (compare to
            mean of all neighbors) or "max" (compare to maximum expression across
            neighbors per gene). Default is "mean".
        method (str): Statistical test to use: "wilcoxon" or "t-test".
            Default is "wilcoxon".
        test (str, optional): Deprecated. Use ``method`` instead.
        correction_method (str): Multiple testing correction method. Default is "fdr_bh".
        min_cells (Number): Minimum number of cells required for a valid test.
            Default is 3.
        genes_subset (List[str], optional): List of gene names to analyze.
        use_distance_weighting (bool): Weight neighbor contributions by inverse distance
            (only for mean strategy). Default is False.
        exclude_zeros_from_max (bool): Exclude zero values when computing maximum (only
            for max strategy). Zeros cannot be contamination sources. Default is True.
        batch_size (int, optional): Process genes in batches to reduce memory usage.
        verbose (bool): Print progress messages. Default is True.

    Returns:
        tuple: A tuple containing:
            - results (pd.DataFrame): Per-gene statistics with contamination metrics.
              Columns depend on strategy: for "mean" includes mean_neighbor,
              log2foldchange, log2foldchange_unpaired; for "max" includes
              mean_neighbor_max, log2foldchange, log2foldchange_unpaired.
            - adjacency_matrix (sparse.csr_matrix): Neighbor adjacency matrix.
            - diff_matrix (np.ndarray): Per-cell gene expression differences.
            - qc_stats (dict): Quality control statistics.

    Examples:
    # Mean strategy (default)
    results, A, diffs, qc = calculate_gex_diff_to_neighbors(
        adata,
        celltype_tuple=("cell_type", "T cells"),
        strategy="mean"
    )

    # Max strategy with distance weighting
    results, A, diffs, qc = calculate_gex_diff_to_neighbors(
        adata,
        celltype_tuple=("cell_type", "T cells"),
        strategy="max",
        exclude_zeros_from_max=True
    )
    """
    if test is not None:
        warnings.warn("'test' is deprecated, use 'method' instead.",
                      DeprecationWarning, stacklevel=2)
        method = test

    # Validate inputs
    celltype_col, celltype = celltype_tuple if celltype_tuple is not None else (None, None)
    _validate_inputs(adata, radius, obs_key, celltype_col, celltype, method, min_cells, strategy)

    # Prepare data
    coords = np.asarray(adata.obsm[obs_key])
    X = adata.X
    genes, gene_indices, n_genes_total = _prepare_data(adata, genes_subset, verbose)
    n_genes = len(genes)
    n_cells = coords.shape[0]

    # Build spatial graph
    A_ct, target_mask, deg, n_target_cells = _build_spatial_graph(
        coords, radius, celltype_col, celltype, adata,
        exclude_self, use_distance_weighting, strategy, verbose
    )

    # Compute neighbor expression based on strategy
    if strategy == "mean":
        neighbor_expr = _compute_neighbor_mean(A_ct, X, gene_indices, verbose)
    else:  # max
        neighbor_expr = _compute_neighbor_max(
            A_ct, X, gene_indices, target_mask, exclude_zeros_from_max, verbose
        )

    # Compute differences
    if verbose:
        suffix = "_max" if strategy == "max" else ""
        logger.info("Computing gex_diff = gex_target - gex_neighbor%s ...", suffix)

    X_dense = X.toarray() if sparse.issparse(X) else X
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_dense = X_dense[:, gene_indices]

    gex_diff = np.full((n_cells, n_genes), np.nan)
    gex_diff[target_mask] = X_dense[target_mask] - neighbor_expr[target_mask]
    target_expr = X_dense[target_mask]

    # Run statistical tests
    test_results = _run_statistical_tests(
        gex_diff, target_expr, neighbor_expr, target_mask,
        genes, method, min_cells, batch_size, verbose
    )

    # Apply correction
    p_adj = _apply_multiple_testing_correction(
        test_results['pvals'], correction_method, verbose
    )

    # Create results DataFrame with strategy-specific naming
    neighbor_col = "mean_neighbor_max" if strategy == "max" else "mean_neighbor"

    results = pd.DataFrame({
        "gene": genes,
        "n_target_cells": test_results['n_target_cells'],
        "n_cells_used": test_results['n_cells_used'],
        "n_cells_expressed": test_results['n_expressed'],
        "mean_target": test_results['mean_target'],
        neighbor_col: test_results['mean_neighbor'],
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
        method, correction_method, radius, min_cells,
        use_distance_weighting, strategy,
        exclude_zeros_from_max=exclude_zeros_from_max if strategy == "max" else None
    )

    _print_summary(qc_stats, verbose)

    return results, A_ct, gex_diff, qc_stats


# ============================================================================
# CONVENIENCE WRAPPERS (for backward compatibility)
# ============================================================================

def mean_gex_diff_to_neighbors(
    adata,
    radius: Number = 20.0,
    obs_key: str = "spatial",
    celltype_tuple: Optional[Tuple[str, str]] = None,
    exclude_self: bool = True,
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: str = "fdr_bh",
    min_cells: Number = 3,
    genes_subset: Optional[List[str]] = None,
    use_distance_weighting: bool = False,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray, Dict]:
    """
    Wrapper for calculate_gex_diff_to_neighbors with strategy="mean".

    See calculate_gex_diff_to_neighbors() for full documentation.
    """
    return calculate_gex_diff_to_neighbors(
        adata=adata,
        radius=radius,
        obs_key=obs_key,
        celltype_tuple=celltype_tuple,
        exclude_self=exclude_self,
        strategy="mean",
        method=method,
        correction_method=correction_method,
        min_cells=min_cells,
        genes_subset=genes_subset,
        use_distance_weighting=use_distance_weighting,
        batch_size=batch_size,
        verbose=verbose
    )


def max_gex_diff_to_neighbors(
    adata,
    radius: Number = 20.0,
    obs_key: str = "spatial",
    celltype_tuple: Optional[Tuple[str, str]] = None,
    exclude_self: bool = True,
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    correction_method: str = "fdr_bh",
    min_cells: Number = 3,
    genes_subset: Optional[List[str]] = None,
    exclude_zeros_from_max: bool = True,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray, Dict]:
    """
    Wrapper for calculate_gex_diff_to_neighbors with strategy="max".

    See calculate_gex_diff_to_neighbors() for full documentation.
    """
    return calculate_gex_diff_to_neighbors(
        adata=adata,
        radius=radius,
        obs_key=obs_key,
        celltype_tuple=celltype_tuple,
        exclude_self=exclude_self,
        strategy="max",
        method=method,
        correction_method=correction_method,
        min_cells=min_cells,
        genes_subset=genes_subset,
        exclude_zeros_from_max=exclude_zeros_from_max,
        batch_size=batch_size,
        verbose=verbose
    )
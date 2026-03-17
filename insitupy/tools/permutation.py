import logging
from numbers import Number
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse, stats
from sklearn.neighbors import radius_neighbors_graph
from statsmodels.stats import multitest
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================================
# CORE COMPUTATION FUNCTIONS
# ============================================================================

def _compute_gex_diff_mean(
    A_ct: sparse.csr_matrix,
    X: np.ndarray,
    target_mask: np.ndarray,
    gene_indices: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gene expression differences using mean neighbor expression.

    Returns
    -------
    gex_diff : np.ndarray
        Gene expression differences (target - neighbor mean)
    neighbor_expr : np.ndarray
        Mean neighbor expression
    """
    n_cells = A_ct.shape[0]
    n_genes = len(gene_indices) if gene_indices is not None else X.shape[1]

    # Compute neighbor mean
    is_sparse = sparse.issparse(X)
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_subset = X[:, gene_indices]
    else:
        X_subset = X

    neighbor_expr = A_ct @ X_subset

    if is_sparse:
        neighbor_expr = neighbor_expr.toarray()

    # Set NaN for cells without neighbors
    row_sums = np.array(A_ct.sum(axis=1)).squeeze()
    cells_without_neighbors = row_sums == 0
    neighbor_expr[cells_without_neighbors, :] = np.nan

    # Compute differences
    X_dense = X.toarray() if sparse.issparse(X) else X
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_dense = X_dense[:, gene_indices]

    gex_diff = np.full((n_cells, n_genes), np.nan)
    gex_diff[target_mask] = X_dense[target_mask] - neighbor_expr[target_mask]

    return gex_diff, neighbor_expr


def _compute_gex_diff_max(
    A_ct: sparse.csr_matrix,
    X: np.ndarray,
    target_mask: np.ndarray,
    gene_indices: np.ndarray,
    exclude_zeros: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gene expression differences using maximum neighbor expression per gene.

    Returns
    -------
    gex_diff : np.ndarray
        Gene expression differences (target - neighbor max)
    neighbor_expr : np.ndarray
        Maximum neighbor expression per gene
    """
    n_cells = A_ct.shape[0]
    n_genes = len(gene_indices) if gene_indices is not None else X.shape[1]

    is_sparse = sparse.issparse(X)
    neighbor_expr = np.full((n_cells, n_genes), np.nan)

    # Only compute for target cells
    target_indices = np.where(target_mask)[0]

    for i in target_indices:
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

    # Compute differences
    X_dense = X.toarray() if sparse.issparse(X) else X
    if gene_indices is not None and len(gene_indices) < X.shape[1]:
        X_dense = X_dense[:, gene_indices]

    gex_diff = np.full((n_cells, n_genes), np.nan)
    gex_diff[target_mask] = X_dense[target_mask] - neighbor_expr[target_mask]

    return gex_diff, neighbor_expr


def _filter_and_normalize_graph(
    A_base: sparse.csr_matrix,
    shuffled_labels: np.ndarray,
    celltype: str,
    use_distance_weighting: bool
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Filter pre-computed spatial graph based on shuffled cell type labels.

    Parameters
    ----------
    A_base : sparse.csr_matrix
        Pre-computed base spatial adjacency matrix
    shuffled_labels : np.ndarray
        Shuffled cell type labels
    celltype : str
        Target cell type
    use_distance_weighting : bool
        Whether to apply distance weighting

    Returns
    -------
    A_ct : sparse.csr_matrix
        Filtered and normalized adjacency matrix (for mean) or just filtered (for max)
    target_mask : np.ndarray
        Boolean mask of target cells
    """
    # Get shuffled target mask
    target_mask = (shuffled_labels == celltype)

    # Exclude same-type neighbors
    A_ct = A_base.copy().tocoo()
    keep = []
    for i, (r, c) in enumerate(zip(A_ct.row, A_ct.col)):
        if not (target_mask[r] and shuffled_labels[r] == shuffled_labels[c] and r != c):
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

    # Apply distance weighting if requested
    if use_distance_weighting:
        A_ct = A_ct.copy()
        A_ct.data = 1.0 / (A_ct.data + 1e-10)

    # Normalize rows (only needed for mean strategy)
    # For max strategy, this normalization will be skipped by the caller
    deg = np.array(A_ct.sum(axis=1)).squeeze()
    deg[deg == 0] = 1
    D_inv = sparse.diags(1.0 / deg)
    A_ct = D_inv @ A_ct

    return A_ct, target_mask


def _single_permutation(
    perm_idx: int,
    A_base: sparse.csr_matrix,
    cell_labels: np.ndarray,
    X: np.ndarray,
    genes: List[str],
    gene_indices: np.ndarray,
    celltype: str,
    strategy: Literal["mean", "max"],
    use_distance_weighting: bool,
    exclude_zeros_from_max: bool,
    genes_to_test_mask: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Run a single permutation: shuffle cell types and compute mean differences.

    Parameters
    ----------
    perm_idx : int
        Permutation index
    A_base : sparse.csr_matrix
        Pre-computed base spatial adjacency matrix
    cell_labels : np.ndarray
        Original cell type labels (will be shuffled)
    strategy : {"mean", "max"}
        Strategy for computing neighbor expression
    use_distance_weighting : bool
        Whether to use distance weighting (only for mean)
    exclude_zeros_from_max : bool
        Whether to exclude zeros from max computation
    genes_to_test_mask : np.ndarray, optional
        Boolean mask indicating which genes to compute (for speed optimization)
    random_seed : int, optional
        Random seed base

    Returns
    -------
    mean_diffs : np.ndarray
        Mean difference for each gene in this permutation
    """
    # Set seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed + perm_idx)

    # Shuffle cell type labels
    shuffled_labels = cell_labels.copy()
    np.random.shuffle(shuffled_labels)

    # Filter graph based on shuffled labels
    if strategy == "mean":
        # For mean: use normalized graph with distance weighting
        A_ct, target_mask = _filter_and_normalize_graph(
            A_base, shuffled_labels, celltype, use_distance_weighting
        )
        gex_diff, _ = _compute_gex_diff_mean(A_ct, X, target_mask, gene_indices, verbose=False)
    else:  # max
        # For max: use unnormalized graph (no distance weighting)
        A_ct, target_mask = _filter_and_normalize_graph(
            A_base, shuffled_labels, celltype, use_distance_weighting=False
        )
        # Need to "undo" the normalization for max strategy
        deg = np.array(A_ct.sum(axis=1)).squeeze()
        deg[deg == 0] = 1
        D = sparse.diags(deg)
        A_ct_unnorm = D @ A_ct

        gex_diff, _ = _compute_gex_diff_max(
            A_ct_unnorm, X, target_mask, gene_indices,
            exclude_zeros=exclude_zeros_from_max, verbose=False
        )

    # Compute mean differences per gene
    n_genes = len(genes)
    mean_diffs = np.full(n_genes, np.nan)

    # Only compute for requested genes if mask provided
    genes_to_compute = range(n_genes) if genes_to_test_mask is None else np.where(genes_to_test_mask)[0]

    for g in genes_to_compute:
        diffs = gex_diff[target_mask, g]
        ok = np.isfinite(diffs)
        if ok.sum() > 0:
            mean_diffs[g] = np.mean(diffs[ok])

    return mean_diffs


# ============================================================================
# MAIN PERMUTATION TEST FUNCTIONS
# ============================================================================

def permutation_test_gex_diff(
    adata,
    observed_results: pd.DataFrame,
    radius: float = 20.0,
    obs_key: str = "spatial",
    celltype_tuple: Tuple[str, str] = None,
    exclude_self: bool = True,
    strategy: Literal["mean", "max"] = "mean",
    use_distance_weighting: bool = False,
    exclude_zeros_from_max: bool = True,
    genes_subset: Optional[List[str]] = None,
    genes_to_test: Optional[List[str]] = None,
    n_permutations: int = 1000,
    n_jobs: int = -1,
    random_seed: Optional[int] = 42,
    show_progress: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform permutation test to assess significance of spatial gene expression differences.

    Shuffles cell type labels while keeping spatial positions fixed to generate null distribution.
    Works for both mean and max neighbor strategies.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial coordinates and gene expression.
    observed_results : pd.DataFrame
        Results from mean_gex_diff_to_neighbors() or max_gex_diff_to_neighbors().
    radius : float, default=20.0
        Radius for spatial neighborhood (must match observed analysis).
    obs_key : str, default="spatial"
        Key in adata.obsm containing spatial coordinates.
    celltype_tuple : Tuple[str, str]
        Tuple of (celltype_col, celltype) for target cell type.
    exclude_self : bool, default=True
        Whether to exclude cell itself from neighborhood.
    strategy : {"mean", "max"}, default="mean"
        Strategy for computing neighbor expression (must match observed analysis).
    use_distance_weighting : bool, default=False
        Weight neighbors by inverse distance (only for mean strategy).
    exclude_zeros_from_max : bool, default=True
        Exclude zeros when computing maximum (only for max strategy).
    genes_subset : List[str], optional
        List of genes to test (must match observed analysis).
    genes_to_test : List[str], optional
        Subset of genes to run permutation test on (for speed optimization).
        If None, tests all genes. Useful for testing only significant genes.
    n_permutations : int, default=1000
        Number of permutations to run.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all cores).
    random_seed : int, optional
        Random seed for reproducibility.
    show_progress : bool, default=True
        Show progress bar for permutations.
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    results : pd.DataFrame
        Original results with added columns:
        - perm_pvalue: Empirical p-value from permutation test
          (two-sided for mean strategy, one-sided lower-tail for max strategy)
        - perm_padj: FDR-corrected permutation p-values
        - perm_mean: Mean of null distribution
        - perm_std: Standard deviation of null distribution
        - perm_zscore: Z-score = (observed_mean - perm_mean) / perm_std

    Notes
    -----
    The permutation test uses different approaches for each strategy:
    - **Mean strategy**: Two-sided test - tests for both enrichment and depletion
      relative to neighborhood mean
    - **Max strategy**: One-sided lower-tail test - tests only for contamination
      (target < neighbor maximum), as positive values are not biologically meaningful
      for contamination detection
    """
    if celltype_tuple is None:
        raise ValueError("celltype_tuple must be provided")

    if strategy not in ["mean", "max"]:
        raise ValueError("strategy must be 'mean' or 'max'")

    celltype_col, celltype = celltype_tuple

    if verbose:
        logger.info("Running permutation test (%s strategy) with %d permutations...", strategy, n_permutations)
        if n_jobs == -1:
            import os
            n_cores = os.cpu_count()
            logger.info("Using all %d cores for parallel processing", n_cores)
        else:
            logger.info("Using %d cores for parallel processing", n_jobs)

    # Prepare data
    coords = np.asarray(adata.obsm[obs_key])
    X = adata.X
    cell_labels = adata.obs[celltype_col].values

    # Get genes
    if genes_subset is not None:
        genes = [g for g in adata.var_names if g in genes_subset]
        gene_indices = np.array([i for i, g in enumerate(adata.var_names) if g in genes_subset])
    else:
        genes = adata.var_names.tolist()
        gene_indices = np.arange(len(genes))

    n_genes = len(genes)

    # Create mask for genes to test (optimization)
    genes_to_test_mask = None
    if genes_to_test is not None:
        genes_to_test_mask = np.array([g in genes_to_test for g in genes])
        n_genes_to_test = genes_to_test_mask.sum()

        if n_genes_to_test == 0:
            raise ValueError("None of the genes in genes_to_test found in genes list")

        if verbose:
            logger.info("Testing only %d/%d genes (speed optimization)", n_genes_to_test, n_genes)
    else:
        n_genes_to_test = n_genes

    # Extract observed mean differences
    observed_mean_diff = observed_results.loc[genes, 'log2foldchange'].values

    # PRE-COMPUTE BASE SPATIAL GRAPH
    if verbose:
        logger.info("Pre-computing base spatial graph...")

    mode = "distance" if (strategy == "mean" and use_distance_weighting) else "connectivity"
    A_base = radius_neighbors_graph(coords, radius=radius, mode=mode, include_self=not exclude_self)

    if verbose:
        logger.info("Base graph computed: %d edges for %d cells", A_base.nnz, A_base.shape[0])

    # Run permutations in parallel
    if verbose:
        logger.info("Running permutations...")

    if show_progress:
        perm_results = []
        with tqdm(total=n_permutations, desc="Permutations", unit="perm") as pbar:
            for result in Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_single_permutation)(
                    perm_idx=i,
                    A_base=A_base,
                    cell_labels=cell_labels,
                    X=X,
                    genes=genes,
                    gene_indices=gene_indices,
                    celltype=celltype,
                    strategy=strategy,
                    use_distance_weighting=use_distance_weighting,
                    exclude_zeros_from_max=exclude_zeros_from_max,
                    genes_to_test_mask=genes_to_test_mask,
                    random_seed=random_seed
                )
                for i in range(n_permutations)
            ):
                perm_results.append(result)
                pbar.update(1)
    else:
        perm_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_single_permutation)(
                perm_idx=i,
                A_base=A_base,
                cell_labels=cell_labels,
                X=X,
                genes=genes,
                gene_indices=gene_indices,
                celltype=celltype,
                strategy=strategy,
                use_distance_weighting=use_distance_weighting,
                exclude_zeros_from_max=exclude_zeros_from_max,
                genes_to_test_mask=genes_to_test_mask,
                random_seed=random_seed
            )
            for i in range(n_permutations)
        )

    # Stack results into matrix (n_permutations x n_genes)
    perm_matrix = np.array(perm_results)

    if verbose:
        logger.info("Computing empirical p-values...")

    # Compute empirical p-values (two-sided test)
    perm_pvalues = np.full(n_genes, np.nan)
    perm_means = np.full(n_genes, np.nan)
    perm_stds = np.full(n_genes, np.nan)
    perm_zscores = np.full(n_genes, np.nan)

    # Only compute statistics for tested genes
    genes_to_compute = range(n_genes) if genes_to_test_mask is None else np.where(genes_to_test_mask)[0]

    for g in genes_to_compute:
        perm_diffs = perm_matrix[:, g]
        valid_perms = np.isfinite(perm_diffs)

        if valid_perms.sum() > 0 and np.isfinite(observed_mean_diff[g]):
            perm_diffs_valid = perm_diffs[valid_perms]
            obs_val = observed_mean_diff[g]

            # Calculate p-value based on strategy
            if strategy == "mean":
                # Two-sided test: count how many permutations are as extreme or more extreme
                # NOTE: This implementation assumes null distribution is centered at zero.
                # It compares absolute values, so values far from zero in either direction
                # are considered extreme. For non-zero-centered nulls, a distance-based
                # approach (|obs - null_mean| vs |perm - null_mean|) would be more appropriate.
                n_extreme = np.sum(np.abs(perm_diffs_valid) >= np.abs(obs_val))
            else:  # max
                # One-sided lower-tail test: count how many permutations are as negative or more negative
                # We only care about contamination (target < neighbor max)
                n_extreme = np.sum(perm_diffs_valid <= obs_val)

            n_valid = len(perm_diffs_valid)

            perm_pvalues[g] = (n_extreme + 1) / (n_valid + 1)
            perm_means[g] = np.mean(perm_diffs_valid)
            perm_stds[g] = np.std(perm_diffs_valid)

            # Z-score: (observed - null_mean) / null_std
            if perm_stds[g] > 0:
                perm_zscores[g] = (obs_val - perm_means[g]) / perm_stds[g]

    # FDR correction
    finite_pvals = np.isfinite(perm_pvalues)
    perm_padj = np.full(n_genes, np.nan)

    if finite_pvals.sum() > 0:
        _, perm_padj[finite_pvals], _, _ = multitest.multipletests(
            perm_pvalues[finite_pvals], alpha=0.05, method='fdr_bh'
        )

    # Add results to observed dataframe
    results = observed_results.copy()
    results['perm_pvalue'] = perm_pvalues
    results['perm_padj'] = perm_padj
    results['perm_mean'] = perm_means
    results['perm_std'] = perm_stds
    results['perm_zscore'] = perm_zscores

    if verbose:
        n_sig = (perm_padj < 0.05).sum()
        n_tested = finite_pvals.sum()
        logger.info("Permutation test complete!")
        logger.info("Genes tested: %d/%d", n_tested, n_genes)
        logger.info("Significant genes (FDR < 0.05): %d/%d", n_sig, n_tested)
        if n_tested > 0:
            logger.info("Mean null distribution mean: %.4f", np.nanmean(perm_means))
            logger.info("Mean null distribution std: %.4f", np.nanmean(perm_stds))

    return results


# Convenience wrappers
def permutation_test_mean_gex_diff(
    adata,
    observed_results: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Permutation test for mean neighbor strategy.

    Wrapper around permutation_test_gex_diff() with strategy="mean".
    See permutation_test_gex_diff() for full documentation.
    """
    return permutation_test_gex_diff(
        adata, observed_results, strategy="mean", **kwargs
    )


def permutation_test_max_gex_diff(
    adata,
    observed_results: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Permutation test for max neighbor strategy.

    Wrapper around permutation_test_gex_diff() with strategy="max".
    See permutation_test_gex_diff() for full documentation.
    """
    # Set default for exclude_zeros_from_max if not provided
    if 'exclude_zeros_from_max' not in kwargs:
        kwargs['exclude_zeros_from_max'] = True

    return permutation_test_gex_diff(
        adata, observed_results, strategy="max", **kwargs
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_permutation_summary(
    results: pd.DataFrame,
    alpha: float = 0.05,
    top_n: int = 20
) -> None:
    """
    Print summary of permutation test results.

    Parameters
    ----------
    results : pd.DataFrame
        Results from permutation_test_gex_diff()
    alpha : float, default=0.05
        Significance threshold
    top_n : int, default=20
        Number of top genes to display
    """
    logger.info("=" * 70)
    logger.info("PERMUTATION TEST SUMMARY")
    logger.info("=" * 70)

    # Overall statistics
    total_genes = len(results)
    tested_genes = results['perm_pvalue'].notna().sum()
    sig_genes = (results['perm_padj'] < alpha).sum()

    logger.info("Total genes: %d", total_genes)
    logger.info("Genes tested: %d", tested_genes)
    logger.info("Significant genes (FDR < %s): %d (%.1f%%)", alpha, sig_genes, sig_genes / tested_genes * 100)

    # Top significant genes
    sig_results = results[results['perm_padj'] < alpha].copy()

    if len(sig_results) > 0:
        sig_results = sig_results.sort_values('perm_zscore')

        logger.info("Top %d significant genes:", min(top_n, len(sig_results)))
        logger.info("%-15s %10s %10s %10s %10s", "Gene", "log2FC", "z-score", "perm_p", "perm_padj")
        logger.info("-" * 70)

        for idx, (gene, row) in enumerate(sig_results.head(top_n).iterrows()):
            logger.info("%-15s %10.3f %10.2f %10.4f %10.4f",
                         gene, row['log2foldchange'], row['perm_zscore'],
                         row['perm_pvalue'], row['perm_padj'])
    else:
        logger.info("No significant genes found.")

    logger.info("=" * 70)


def calculate_contamination_threshold(
    results_perm: pd.DataFrame,
    percentile: float = 95,
    direction: str = "negative",
    return_details: bool = False
) -> float:
    """
    Calculate empirical threshold from permutation null distribution.

    How it works:
    1. For each gene, calculate: threshold = perm_mean + z × perm_std
    2. Take median of all gene-specific thresholds
    3. This gives a single global threshold

    Parameters
    ----------
    results_perm : pd.DataFrame
        Results from permutation test
    percentile : float
        Confidence level (e.g., 95 = 95% confidence)
        For contamination, this means "only 5% of random permutations
        would be more extreme"
    direction : str
        "negative" for contamination (target < neighbor)
        "positive" for enrichment (target > neighbor)
    return_details : bool
        If True, return dict with threshold and per-gene thresholds

    Returns
    -------
    threshold : float or dict
        Global log2FC threshold (or dict if return_details=True)

    Examples
    --------
    # 95% confidence: only 5% of random shuffles would be more extreme
    threshold_95 = calculate_contamination_threshold(
        results_perm, percentile=95
    )

    # 99% confidence: only 1% of random shuffles would be more extreme
    threshold_99 = calculate_contamination_threshold(
        results_perm, percentile=99
    )
    """
    # Get genes with valid null distributions
    valid = results_perm['perm_std'].notna() & (results_perm['perm_std'] > 0)

    if valid.sum() == 0:
        raise ValueError("No genes with valid null distribution")

    if direction == "negative":
        # For contamination: find lower tail threshold
        tail_prob = (100 - percentile) / 100  # e.g., 0.05 for 95th percentile
        z_score = stats.norm.ppf(tail_prob)
    else:
        # For enrichment: find upper tail threshold
        tail_prob = percentile / 100  # e.g., 0.95 for 95th percentile
        z_score = stats.norm.ppf(tail_prob)

    # Calculate gene-specific thresholds
    gene_thresholds = (
        results_perm.loc[valid, 'perm_mean'] +
        z_score * results_perm.loc[valid, 'perm_std']
    )

    # Global threshold = median of gene-specific thresholds
    global_threshold = gene_thresholds.median()

    if return_details:
        return {
            'global_threshold': global_threshold,
            'gene_thresholds': gene_thresholds,
            'z_score': z_score,
            'percentile': percentile,
            'direction': direction,
            'n_genes': valid.sum(),
            'threshold_range': (gene_thresholds.min(), gene_thresholds.max()),
            'threshold_std': gene_thresholds.std()
        }

    return global_threshold
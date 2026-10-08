"""Tests for tl.permutation_test_gex_diff (AC-B5 - permutation null unit fix).

The permutation null was computed in natural-log (log1p) units while the
observed statistic (`log2foldchange`) is in log2 units, making p-values
anti-conservative and z-scores inflated by ~1.44x (1 / ln(2)). This test
pins the corrected unit: with permutation shuffling neutralised (identity
permutation), the null must reproduce the observed log2foldchange, not be
off by a factor of ln(2).
"""

import numpy as np
import pandas as pd
from anndata import AnnData

from insitupy.tools.neighbors import calculate_gex_diff_to_neighbors
from insitupy.tools.permutation import permutation_test_gex_diff

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_paired_adata(n_pairs=5, n_genes=4, seed=0):
    """Cells arranged as (A, B) pairs; each A cell's only neighbor within the
    default radius (20.0) is its paired B cell, and vice versa. This makes
    the neighbor-mean computation deterministic and identical between
    `calculate_gex_diff_to_neighbors` (observed) and `_single_permutation`
    (null) once permutation shuffling is neutralised.
    """
    rng = np.random.default_rng(seed)
    coords = []
    labels = []
    for i in range(n_pairs):
        coords.append([i * 25.0, 0.0])
        labels.append("A")
        coords.append([i * 25.0, 4.0])
        labels.append("B")
    coords = np.array(coords)
    labels = np.array(labels)

    n = len(labels)
    counts = rng.integers(1, 10, size=(n, n_genes)).astype(float)
    X = np.log1p(counts)

    obs = pd.DataFrame({"celltype": labels}, index=pd.Index([f"c{i}" for i in range(n)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    return adata


# ── AC-B5: null and observed share log2 units ───────────────────────────────

def test_identity_permutation_reproduces_log2_observed(monkeypatch):
    # Neutralise shuffling: shuffled_labels stays equal to cell_labels, so
    # the single permutation's neighbor graph and gex_diff match the
    # observed computation exactly.
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    adata = _make_paired_adata()

    observed_results, _, _, _ = calculate_gex_diff_to_neighbors(
        adata,
        celltype_tuple=("celltype", "A"),
        strategy="mean",
        assert_log1p=False,
        verbose=False,
    )

    perm_results = permutation_test_gex_diff(
        adata,
        observed_results=observed_results,
        celltype_tuple=("celltype", "A"),
        strategy="mean",
        n_permutations=1,
        n_jobs=1,
        random_seed=42,
        show_progress=False,
        verbose=False,
    )

    tested = perm_results["log2foldchange"].notna() & perm_results["perm_mean"].notna()
    assert tested.any()

    # Before the fix, this identity held only up to a factor of ln(2)
    # (perm_mean == log2foldchange * ln(2)); the fix makes both log2.
    np.testing.assert_allclose(
        perm_results.loc[tested, "perm_mean"].values,
        perm_results.loc[tested, "log2foldchange"].values,
        rtol=1e-8,
        atol=1e-10,
    )

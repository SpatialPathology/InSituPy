"""Regression tests for non-finite embedding coordinates in pl.embedding / pl.umap.

Reported bug: ``isp.pl.umap(adata, keys=["celltype_final", "uid"])`` raised
``ValueError: Axis limits cannot be NaN or Inf`` from datashader. Root cause was
NaN/Inf coordinates in ``adata.obsm[basis]``: datashader derives axis limits from
the min/max of the x/y columns, so a single non-finite coordinate poisons the
limit. The crash surfaced only for color keys *without* NaN values (e.g. ``uid``);
keys whose NaN cells coincided with the bad coordinates silently dropped them.

``embedding`` now filters non-finite coordinates up front (warning once), so every
render path is protected regardless of the color key.
"""

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import insitupy as isp
from insitupy.plotting.scatter import _check_datashader

N_TOTAL = 20
N_BAD = 4
N_FINITE = N_TOTAL - N_BAD


def _make_adata(bad_value=np.nan, n_cells=N_TOTAL, n_bad=N_BAD):
    """AnnData whose UMAP has `n_bad` non-finite rows and a NaN-free `uid` column.

    `uid` is a single-value categorical with no NaN, reproducing the reported key
    that never dropped the bad-coordinate cells.
    """
    rng = np.random.default_rng(0)
    X = rng.random((n_cells, 4)).astype("float32")
    umap = rng.random((n_cells, 2)).astype("float32")
    umap[:n_bad, 0] = bad_value
    obs = pd.DataFrame(
        {"uid": pd.Categorical(["sample_1"] * n_cells)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(4)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = umap
    return adata


def _total_scatter_points(fig):
    return sum(
        len(c.get_offsets())
        for a in fig.get_axes()
        for c in a.collections
    )


# ── Regression: the exact reported datashader crash ──────────────────────────

@pytest.mark.skipif(not _check_datashader(), reason="datashader not installed")
class TestDatashaderNonfiniteCoords:
    def test_umap_nan_coords_does_not_raise(self):
        adata = _make_adata(bad_value=np.nan)
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # non-finite warning is asserted elsewhere
            isp.pl.umap(adata, keys=["uid"], show=False)
        plt.close("all")

    def test_umap_inf_coords_does_not_raise(self):
        adata = _make_adata(bad_value=np.inf)
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            isp.pl.umap(adata, keys=["uid"], show=False)
        plt.close("all")


# ── Behavioral: warns once and hides only the non-finite cells ───────────────

class TestNonfiniteCoordsMatplotlib:
    def test_warns_on_nonfinite_coords(self):
        adata = _make_adata(bad_value=np.nan)
        plt.close("all")
        with pytest.warns(UserWarning, match="non-finite coordinates"):
            isp.pl.embedding(
                adata, basis="X_umap", keys=["uid"],
                render_mode="matplotlib", show=False,
            )
        plt.close("all")

    def test_only_finite_cells_plotted(self):
        adata = _make_adata(bad_value=np.nan)
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = isp.pl.embedding(
                adata, basis="X_umap", keys=["uid"],
                render_mode="matplotlib", show=False, return_fig=True,
            )
        assert _total_scatter_points(fig) == N_FINITE
        plt.close("all")

    def test_inf_coords_also_hidden(self):
        adata = _make_adata(bad_value=np.inf)
        plt.close("all")
        with pytest.warns(UserWarning, match="non-finite coordinates"):
            fig = isp.pl.embedding(
                adata, basis="X_umap", keys=["uid"],
                render_mode="matplotlib", show=False, return_fig=True,
            )
        assert _total_scatter_points(fig) == N_FINITE
        plt.close("all")


# ── Regression: all-finite coordinates are untouched ─────────────────────────

class TestFiniteCoordsUnchanged:
    def test_no_warning_and_all_points_plotted(self):
        adata = _make_adata(bad_value=np.nan, n_bad=0)
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fig = isp.pl.embedding(
                adata, basis="X_umap", keys=["uid"],
                render_mode="matplotlib", show=False, return_fig=True,
            )
        assert _total_scatter_points(fig) == N_TOTAL
        plt.close("all")

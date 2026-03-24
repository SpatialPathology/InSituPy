"""Smoke tests for plotting functions: single_volcano, volcano, go_plot,
plot_qc_metrics, facs, umap, pca, tsne.

All tests use the Agg backend so no display is required and plt.show() is a no-op.
"""

import matplotlib

matplotlib.use("Agg")  # must be set before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.results import (DiffExprConfigCollector,
                                         DiffExprResults)
from insitupy.plotting.facs import facs
from insitupy.plotting.go import go_plot
from insitupy.plotting.qc import plot_qc_metrics
from insitupy.plotting.scatter import pca, tsne, umap
from insitupy.plotting.volcano import single_volcano, volcano

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dge_df(n_genes=20, seed=0):
    """DataFrame with required DGE result columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "log2foldchange": rng.uniform(-4, 4, n_genes),
            "padj": rng.uniform(0.001, 1, n_genes),
            "scores": rng.uniform(-3, 3, n_genes),
            "neg_log10_pvals": rng.uniform(0, 5, n_genes),
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )


def _make_dge_results(n_genes=20):
    df = _make_dge_df(n_genes)
    config = DiffExprConfigCollector(mode="single-cell", method_params={})
    return DiffExprResults(main=df, config=config)


def _make_enrichment_df():
    """Minimal multi-index enrichment DataFrame matching go_plot's expected format."""
    data = {
        "source": ["GO:BP"] * 5,
        "native": [f"GO:000{i}" for i in range(5)],
        "name": [f"term_{i}" for i in range(5)],
        "Enrichment score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "Gene ratio": [0.4, 0.3, 0.25, 0.2, 0.15],
    }
    df = pd.DataFrame(data)
    df.index = pd.MultiIndex.from_tuples(
        [("groupA", i) for i in range(5)], names=["group", "idx"]
    )
    return df


def _make_adata_with_embeddings(n_cells=20, n_genes=8, seed=0):
    """AnnData with X_umap, X_pca, X_tsne in obsm and a categorical obs column."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, n_genes))
    obs = pd.DataFrame(
        {"celltype": pd.Categorical(["A", "B"] * (n_cells // 2))},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.random((n_cells, 2))
    adata.obsm["X_pca"] = rng.random((n_cells, 4))
    adata.obsm["X_tsne"] = rng.random((n_cells, 2))
    return adata


def _make_insitudata_sparse(n_cells=20, n_genes=4, seed=0):
    """InSituData with a sparse count matrix (required by pl.facs)."""
    rng = np.random.default_rng(seed)
    X_sparse = sp.csr_matrix(rng.integers(0, 20, size=(n_cells, n_genes)).astype(float))
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_genes)]))
    table = AnnData(X=X_sparse, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="test", sample_id="s1",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


# ── single_volcano ────────────────────────────────────────────────────────────

class TestSingleVolcano:
    def test_runs_without_error(self):
        df = _make_dge_df()
        plt.close("all")
        single_volcano(df, show=False, adjust_labels=False)
        plt.close("all")

    def test_with_provided_ax(self):
        df = _make_dge_df()
        fig, ax = plt.subplots()
        single_volcano(df, ax=ax, adjust_labels=False)
        # Function should have drawn on the supplied axis
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        plt.close("all")

    def test_custom_thresholds_accepted(self):
        df = _make_dge_df()
        plt.close("all")
        single_volcano(
            df, significance_threshold=0.1, foldchange_threshold=1.5,
            show=False, adjust_labels=False,
        )
        plt.close("all")


# ── volcano ───────────────────────────────────────────────────────────────────

class TestVolcano:
    def test_runs_without_error(self):
        results = _make_dge_results()
        plt.close("all")
        volcano(results, show=False)
        plt.close("all")

    def test_accepts_custom_thresholds(self):
        results = _make_dge_results()
        plt.close("all")
        volcano(results, significance_threshold=0.1, foldchange_threshold=1.5, show=False)
        plt.close("all")


# ── go_plot ───────────────────────────────────────────────────────────────────

class TestGoPlot:
    def test_returns_fig_axes_when_show_false(self):
        enrichment = _make_enrichment_df()
        result = go_plot(enrichment, show=False)
        assert result is not None
        fig, axs = result
        assert fig is not None
        plt.close("all")

    def test_bar_style_runs_without_error(self):
        enrichment = _make_enrichment_df()
        result = go_plot(enrichment, style="bar", show=False)
        assert result is not None
        plt.close("all")


# ── plot_qc_metrics ───────────────────────────────────────────────────────────

class TestPlotQcMetrics:
    def _make_adata_with_qc(self, n_cells=20, n_genes=8):
        rng = np.random.default_rng(0)
        X = rng.integers(1, 50, size=(n_cells, n_genes)).astype(float)
        obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n_cells)]))
        var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
        adata = AnnData(X=X, obs=obs, var=var)
        sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
        return adata

    def test_runs_without_error_on_adata(self):
        adata = self._make_adata_with_qc()
        plt.close("all")
        plot_qc_metrics(adata)
        plt.close("all")

    def test_obs_only_runs_without_error(self):
        adata = self._make_adata_with_qc()
        plt.close("all")
        plot_qc_metrics(adata, plot_var=False)
        plt.close("all")


# ── pl.facs ─────────────────────────────────────────────────────────────────

class TestFacsPlot:
    def test_runs_without_error(self):
        xd = _make_insitudata_sparse(n_genes=4)
        plt.close("all")
        facs(xd, gene1="gene_0", gene2="gene_1", cluster_key=None)
        plt.close("all")

    def test_double_positive_column_added(self):
        xd = _make_insitudata_sparse(n_genes=4)
        plt.close("all")
        facs(
            xd, gene1="gene_0", gene2="gene_1",
            cluster_key=None, threshold_gene1=5, threshold_gene2=5,
        )
        plt.close("all")
        assert "gene_0/gene_1 double pos." in xd.cells.table.obs.columns

    def test_double_positive_column_is_boolean(self):
        xd = _make_insitudata_sparse(n_genes=4)
        plt.close("all")
        facs(xd, gene1="gene_0", gene2="gene_1", cluster_key=None)
        plt.close("all")
        col = xd.cells.table.obs["gene_0/gene_1 double pos."]
        assert col.dtype == bool


# ── umap / pca / tsne wrappers ────────────────────────────────────────────────

class TestEmbeddingWrappers:
    def test_umap_returns_figure(self):
        adata = _make_adata_with_embeddings()
        fig = umap(adata, keys="celltype", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_pca_returns_figure(self):
        adata = _make_adata_with_embeddings()
        fig = pca(adata, keys="celltype", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_tsne_returns_figure(self):
        adata = _make_adata_with_embeddings()
        fig = tsne(adata, keys="celltype", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_umap_uses_x_umap_basis(self):
        """umap() must delegate to embedding() with basis='X_umap', not X_pca."""
        adata = _make_adata_with_embeddings()
        adata_umap_only = adata.copy()
        del adata_umap_only.obsm["X_pca"]
        del adata_umap_only.obsm["X_tsne"]
        # Should succeed because X_umap is present
        fig = umap(adata_umap_only, show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_pca_uses_x_pca_basis(self):
        """pca() must delegate to embedding() with basis='X_pca', not X_umap."""
        adata = _make_adata_with_embeddings()
        adata_pca_only = adata.copy()
        del adata_pca_only.obsm["X_umap"]
        del adata_pca_only.obsm["X_tsne"]
        fig = pca(adata_pca_only, show=False, return_fig=True)
        assert fig is not None
        plt.close("all")
        plt.close("all")

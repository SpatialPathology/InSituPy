"""Smoke tests for plotting functions: single_volcano, volcano, go_plot,
plot_qc_metrics, facs, umap, pca, tsne.

All tests use the Agg backend so no display is required and plt.show() is a no-op.
"""

import matplotlib

matplotlib.use("Agg")  # must be set before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from matplotlib.colors import to_rgb

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.results import DiffExprConfigCollector, DiffExprResults
from insitupy.plotting.dge import BACKGROUND_ALPHA, dual_foldchange_plot
from insitupy.plotting.facs import facs
from insitupy.plotting.go import go_plot
from insitupy.plotting.qc import plot_qc_metrics
from insitupy.plotting.scatter import _apply_dim, _apply_highlight, pca, tsne, umap
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


# ── dual_foldchange_plot ──────────────────────────────────────────────────────

def _make_dge_results_with_neighbors(n_genes=20, nb_offset=0.0):
    """DiffExprResults with target/ref neighborhood frames sharing the main gene index.

    `nb_offset` shifts the neighborhood log2 fold changes, e.g. to put all of them above 0.
    """
    main = _make_dge_df(n_genes, seed=0)
    nb_target = _make_dge_df(n_genes, seed=1)
    nb_ref = _make_dge_df(n_genes, seed=2)
    for nb in (nb_target, nb_ref):
        nb["log2foldchange"] = nb["log2foldchange"] * 0.75 + nb_offset
    config = DiffExprConfigCollector(mode="single-cell", method_params={})
    return DiffExprResults(main=main, config=config,
                           target_neighborhood=nb_target, ref_neighborhood=nb_ref)


def _dual_plot_axes(results, **kwargs):
    """Draw dual_foldchange_plot without showing it and return the two data panels."""
    plt.close("all")
    dual_foldchange_plot(results, adjust_labels=False, show=False, **kwargs)
    return plt.gcf().axes[:2]


class TestDualFoldchangePlot:
    def test_gradient_is_transparent_at_zero_and_below_points(self):
        ax = _dual_plot_axes(_make_dge_results_with_neighbors())[1]
        images = ax.get_images()
        assert len(images) == 1
        image = images[0]
        assert all(image.get_zorder() < c.get_zorder() for c in ax.collections)

        ymin, ymax = ax.get_ylim()
        assert ymin < -1 and ymax > 1  # the fixture must span both saturation points
        y = np.linspace(ymin, ymax, image.get_array().shape[0])
        alpha = np.asarray(image.get_array())[:, 0, 3]
        assert alpha[np.argmin(np.abs(y))] < 0.01
        np.testing.assert_allclose(alpha[np.abs(y) >= 1], BACKGROUND_ALPHA)
        plt.close("all")

    def test_background_does_not_change_axis_limits(self):
        results = _make_dge_results_with_neighbors()
        limits = {}
        for mode in ("gradient", "split"):
            axs = _dual_plot_axes(results, background=mode)
            limits[mode] = [(ax.get_xlim(), ax.get_ylim()) for ax in axs]
        assert limits["gradient"] == limits["split"]
        plt.close("all")

    def test_gradient_with_all_points_above_zero(self):
        results = _make_dge_results_with_neighbors(nb_offset=5.0)
        for ax in _dual_plot_axes(results):
            assert ax.get_ylim()[0] > 0
            rgba = np.asarray(ax.get_images()[0].get_array())[:, 0, :]
            np.testing.assert_allclose(rgba[:, :3], np.tile(to_rgb("lightgreen"), (len(rgba), 1)))
        plt.close("all")

    @pytest.mark.parametrize("reference_lfc, expected", [(1, {-1, 0, 1}), (2.5, {-2.5, 0, 2.5}),
                                                         (None, {0})])
    def test_reference_lines_follow_reference_lfc_not_saturation(self, reference_lfc, expected):
        axs = _dual_plot_axes(_make_dge_results_with_neighbors(),
                              reference_lfc=reference_lfc, background_saturation=3)
        for ax in axs:
            # axhline spans x in axes coordinates [0, 1]; axvline spans y instead
            hlines = {line.get_ydata()[0] for line in ax.lines if list(line.get_xdata()) == [0, 1]}
            assert hlines == expected
        plt.close("all")

    def test_split_mode_draws_no_gradient_image(self):
        for ax in _dual_plot_axes(_make_dge_results_with_neighbors(), background="split"):
            assert ax.get_images() == []
        plt.close("all")

    @pytest.mark.parametrize("kwargs", [
        {"background": "foo"},
        {"background_saturation": 0},
        {"background_saturation": np.inf},
        {"patch_colors": ["lightgreen"]},
        {"reference_lfc": 0},
        {"reference_lfc": -1},
    ])
    def test_invalid_background_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            _dual_plot_axes(_make_dge_results_with_neighbors(), **kwargs)
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


# ── highlight ─────────────────────────────────────────────────────────────────

class TestApplyHighlight:
    def test_recolors_and_trims_legend(self):
        color_dict = {"A": "#111111", "B": "#222222", "C": "#333333"}
        plot_dict, legend_dict = _apply_highlight(color_dict, ["A"], "#E0E0E0")

        assert plot_dict["A"] == "#111111"
        assert plot_dict["B"] == "#E0E0E0"
        assert plot_dict["C"] == "#E0E0E0"
        assert legend_dict == {"A": "#111111", "Other": "#E0E0E0"}
        assert list(plot_dict)[-1] == "A"

    def test_warns_on_missing_category(self):
        color_dict = {"A": "#111111", "B": "#222222"}
        with pytest.warns(UserWarning, match="highlight categories not found"):
            _apply_highlight(color_dict, ["Z"], "#E0E0E0")

    def test_multiple_highlights(self):
        color_dict = {"A": "#111111", "B": "#222222", "C": "#333333"}
        plot_dict, legend_dict = _apply_highlight(color_dict, ["A", "C"], "#E0E0E0")

        assert plot_dict["A"] == "#111111"
        assert plot_dict["C"] == "#333333"
        assert plot_dict["B"] == "#E0E0E0"
        assert set(legend_dict.keys()) == {"A", "C", "Other"}


class TestEmbeddingHighlight:
    def test_umap_highlight_returns_figure(self):
        adata = _make_adata_with_embeddings()
        fig = umap(adata, keys="celltype", highlight="A", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_umap_highlight_matplotlib_backend(self):
        adata = _make_adata_with_embeddings()
        fig = umap(
            adata, keys="celltype", highlight="A",
            render_mode="matplotlib", show=False, return_fig=True
        )
        assert fig is not None
        plt.close("all")

    def test_highlight_warns_on_continuous_key(self):
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="highlight has no effect on continuous"):
            fig = umap(adata, keys="g0", highlight="A", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_highlight_warns_on_no_key(self):
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="highlight has no effect when no color key"):
            fig = umap(adata, keys=None, highlight="A", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_highlight_warns_when_interactive(self):
        pytest.importorskip("holoviews")
        pytest.importorskip("datashader")
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="highlight is only supported for static plots"):
            umap(adata, keys="celltype", highlight="A", interactive=True, show=False)

    def test_highlight_absent_category_warns(self):
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="highlight categories not found"):
            fig = umap(adata, keys="celltype", highlight="Z", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")


# ── dim ────────────────────────────────────────────────────────────────────────

class TestApplyDim:
    def test_recolors_and_trims_legend(self):
        color_dict = {"A": "#111111", "B": "#222222", "C": "#333333"}
        plot_dict, legend_dict = _apply_dim(color_dict, ["A"], "#E0E0E0")

        assert plot_dict["A"] == "#E0E0E0"
        assert plot_dict["B"] == "#222222"
        assert plot_dict["C"] == "#333333"
        assert legend_dict == {"B": "#222222", "C": "#333333", "Other": "#E0E0E0"}
        assert list(plot_dict)[0] == "A"

    def test_warns_on_missing_category(self):
        color_dict = {"A": "#111111", "B": "#222222"}
        with pytest.warns(UserWarning, match="dim categories not found"):
            _apply_dim(color_dict, ["Z"], "#E0E0E0")

    def test_multiple_dims(self):
        color_dict = {"A": "#111111", "B": "#222222", "C": "#333333"}
        plot_dict, legend_dict = _apply_dim(color_dict, ["A", "C"], "#E0E0E0")

        assert plot_dict["A"] == "#E0E0E0"
        assert plot_dict["C"] == "#E0E0E0"
        assert plot_dict["B"] == "#222222"
        assert set(legend_dict.keys()) == {"B", "Other"}


class TestEmbeddingDim:
    def test_umap_dim_returns_figure(self):
        adata = _make_adata_with_embeddings()
        fig = umap(adata, keys="celltype", dim="B", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_umap_dim_matplotlib_backend(self):
        adata = _make_adata_with_embeddings()
        fig = umap(
            adata, keys="celltype", dim="B",
            render_mode="matplotlib", show=False, return_fig=True
        )
        assert fig is not None
        plt.close("all")

    def test_dim_warns_on_continuous_key(self):
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="dim has no effect on continuous"):
            fig = umap(adata, keys="g0", dim="B", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_dim_warns_on_no_key(self):
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="dim has no effect when no color key"):
            fig = umap(adata, keys=None, dim="B", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_dim_warns_when_interactive(self):
        pytest.importorskip("holoviews")
        pytest.importorskip("datashader")
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="dim is only supported for static plots"):
            umap(adata, keys="celltype", dim="B", interactive=True, show=False)

    def test_dim_absent_category_warns(self):
        adata = _make_adata_with_embeddings()
        with pytest.warns(UserWarning, match="dim categories not found"):
            fig = umap(adata, keys="celltype", dim="Z", show=False, return_fig=True)
        assert fig is not None
        plt.close("all")

    def test_highlight_and_dim_mutually_exclusive(self):
        adata = _make_adata_with_embeddings()
        with pytest.raises(ValueError, match="mutually exclusive"):
            umap(adata, keys="celltype", highlight="A", dim="B", show=False)

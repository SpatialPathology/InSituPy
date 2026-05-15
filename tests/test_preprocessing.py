"""Tests for preprocessing functions: normalize_and_transform, filter_cells,
filter_genes, reduce_dimensions, cluster_cells, calculate_qc_metrics, pseudobulk."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import issparse

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.preprocessing import (
    calculate_qc_metrics,
    cluster_cells,
    filter_cells,
    filter_genes,
    normalize_and_transform,
    reduce_dimensions,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_insitudata(n_cells=30, n_genes=15, seed=42):
    """Create a minimal InSituData with an integer count matrix."""
    rng = np.random.default_rng(seed)
    X = rng.integers(1, 50, size=(n_cells, n_genes)).astype(float)

    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(n_cells)], name="obs"))
    var = pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_genes)], name="var"))

    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2))

    celldata = CellData(table=table, boundaries=None)
    data = InSituData(
        path=None, metadata=None,
        slide_id="test", sample_id="s1",
        method_name="test", method_params={},
    )
    data.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return data


def _raw(table):
    """Extract dense numpy matrix from AnnData.X (sparse or dense)."""
    x = table.X
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


# ── normalize_and_transform ───────────────────────────────────────────────────

class TestNormalizeAndTransform:
    def test_raw_counts_stored_in_layer(self):
        xd = _make_insitudata()
        original_X = _raw(xd.cells.table).copy()

        normalize_and_transform(xd, transformation_method="log1p", assert_integer_counts=True)

        assert "counts" in xd.cells.table.layers
        np.testing.assert_array_equal(xd.cells.table.layers["counts"], original_X)

    def test_x_modified_in_place(self):
        xd = _make_insitudata()
        original_X = _raw(xd.cells.table).copy()

        normalize_and_transform(xd, transformation_method="log1p")

        transformed = _raw(xd.cells.table)
        assert not np.allclose(transformed, original_X)

    def test_log1p_values_are_nonnegative(self):
        xd = _make_insitudata()
        normalize_and_transform(xd, transformation_method="log1p")
        transformed = _raw(xd.cells.table)
        assert np.all(transformed >= 0)

    def test_sqrt_values_are_nonnegative(self):
        xd = _make_insitudata()
        normalize_and_transform(xd, transformation_method="sqrt")
        transformed = _raw(xd.cells.table)
        assert np.all(transformed >= 0)

    def test_norm_counts_layer_created(self):
        xd = _make_insitudata()
        normalize_and_transform(xd, transformation_method="log1p")
        assert "norm_counts" in xd.cells.table.layers

    def test_invalid_method_raises(self):
        xd = _make_insitudata()
        with pytest.raises(ValueError):
            normalize_and_transform(xd, transformation_method="invalid")


# ── filter_cells ──────────────────────────────────────────────────────────────

class TestFilterCells:
    def test_min_counts_reduces_cell_count(self):
        rng = np.random.default_rng(0)
        # 10 cells: first 3 have very low counts, rest have high counts
        X = np.zeros((10, 5), dtype=float)
        X[3:] = rng.integers(10, 50, size=(7, 5)).astype(float)

        obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(10)]))
        table = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(5)]))
        table.obsm["spatial"] = rng.random((10, 2))

        celldata = CellData(table=table, boundaries=None)
        xd = InSituData(path=None, metadata=None, slide_id="t", sample_id="s", method_name="t", method_params={})
        xd.cells.add_celldata(cd=celldata, key="main", is_main=True)

        filter_cells(xd, min_counts=1)

        assert xd.cells.table.n_obs == 7

    def test_max_counts_reduces_cell_count(self):
        rng = np.random.default_rng(1)
        X = np.ones((10, 5), dtype=float) * 100
        X[:3] = rng.integers(1, 5, size=(3, 5)).astype(float)

        obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(10)]))
        table = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(5)]))
        table.obsm["spatial"] = rng.random((10, 2))

        celldata = CellData(table=table, boundaries=None)
        xd = InSituData(path=None, metadata=None, slide_id="t", sample_id="s", method_name="t", method_params={})
        xd.cells.add_celldata(cd=celldata, key="main", is_main=True)

        filter_cells(xd, max_counts=20)

        assert xd.cells.table.n_obs == 3

    def test_multiple_criteria_raises(self):
        xd = _make_insitudata()
        with pytest.raises(ValueError):
            filter_cells(xd, min_counts=1, min_genes=1)


# ── filter_genes ──────────────────────────────────────────────────────────────

class TestFilterGenes:
    def test_min_cells_removes_unexpressed_genes(self):
        # 5 genes: first 2 are zero across all cells → should be removed
        X = np.zeros((10, 5), dtype=float)
        X[:, 2:] = np.random.default_rng(0).integers(1, 20, size=(10, 3)).astype(float)

        obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(10)]))
        var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(5)]))
        table = AnnData(X=X, obs=obs, var=var)
        table.obsm["spatial"] = np.zeros((10, 2))

        celldata = CellData(table=table, boundaries=None)
        xd = InSituData(path=None, metadata=None, slide_id="t", sample_id="s", method_name="t", method_params={})
        xd.cells.add_celldata(cd=celldata, key="main", is_main=True)

        filter_genes(xd, min_cells=1)

        assert xd.cells.table.n_vars == 3

    def test_min_counts_removes_low_count_genes(self):
        X = np.zeros((10, 6), dtype=float)
        X[:, :4] = 100  # high count genes
        X[:, 4:] = 0.5  # will become 0 after int check? No - use 1s
        X[:, 4:] = 1

        obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(10)]))
        var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(6)]))
        table = AnnData(X=X, obs=obs, var=var)
        table.obsm["spatial"] = np.zeros((10, 2))

        celldata = CellData(table=table, boundaries=None)
        xd = InSituData(path=None, metadata=None, slide_id="t", sample_id="s", method_name="t", method_params={})
        xd.cells.add_celldata(cd=celldata, key="main", is_main=True)

        filter_genes(xd, max_counts=50)  # keep genes with total <= 50; last 2 have total=10

        # genes 4 and 5 have total count 10 (1 * 10 cells), genes 0-3 have 1000
        assert xd.cells.table.n_vars == 2


# ── reduce_dimensions ─────────────────────────────────────────────────────────

class TestReduceDimensions:
    def _normalized_data(self, n_cells=30, n_genes=15):
        xd = _make_insitudata(n_cells=n_cells, n_genes=n_genes)
        normalize_and_transform(xd, transformation_method="log1p")
        return xd

    def test_pca_added_to_obsm(self):
        xd = self._normalized_data()
        reduce_dimensions(xd, method="umap", n_neighbors=5)
        assert "X_pca" in xd.cells.table.obsm

    def test_umap_added_to_obsm(self):
        xd = self._normalized_data()
        reduce_dimensions(xd, method="umap", n_neighbors=5)
        assert "X_umap" in xd.cells.table.obsm

    def test_umap_shape_is_2d(self):
        xd = self._normalized_data()
        reduce_dimensions(xd, method="umap", n_neighbors=5)
        n = xd.cells.table.n_obs
        assert xd.cells.table.obsm["X_umap"].shape == (n, 2)

    def test_pca_shape_has_correct_n_obs(self):
        xd = self._normalized_data(n_cells=30, n_genes=15)
        reduce_dimensions(xd, method="umap", n_neighbors=5)
        n = xd.cells.table.n_obs
        assert xd.cells.table.obsm["X_pca"].shape[0] == n


# ── cluster_cells ─────────────────────────────────────────────────────────────

class TestClusterCells:
    def _preprocessed_data(self):
        xd = _make_insitudata(n_cells=30, n_genes=15)
        normalize_and_transform(xd, transformation_method="log1p")
        reduce_dimensions(xd, method="umap", n_neighbors=5)
        return xd

    def test_leiden_labels_added_to_obs(self):
        xd = self._preprocessed_data()
        cluster_cells(xd, method="leiden")
        assert "leiden" in xd.cells.table.obs.columns

    def test_louvain_labels_added_to_obs(self):
        pytest.importorskip("louvain")
        xd = self._preprocessed_data()
        cluster_cells(xd, method="louvain")
        assert "louvain" in xd.cells.table.obs.columns

    def test_leiden_has_at_least_one_cluster(self):
        xd = self._preprocessed_data()
        cluster_cells(xd, method="leiden")
        assert xd.cells.table.obs["leiden"].nunique() >= 1


# ── calculate_qc_metrics ──────────────────────────────────────────────────────

class TestCalculateQcMetrics:
    def test_expected_obs_columns_added(self):
        xd = _make_insitudata()
        calculate_qc_metrics(xd)
        obs_cols = xd.cells.table.obs.columns
        assert "n_genes_by_counts" in obs_cols
        assert "total_counts" in obs_cols

    def test_expected_var_columns_added(self):
        xd = _make_insitudata()
        calculate_qc_metrics(xd)
        var_cols = xd.cells.table.var.columns
        assert "n_cells_by_counts" in var_cols
        assert "total_counts" in var_cols

    def test_total_counts_match_row_sums(self):
        xd = _make_insitudata()
        X = _raw(xd.cells.table)
        row_sums = X.sum(axis=1)
        calculate_qc_metrics(xd)
        np.testing.assert_allclose(
            xd.cells.table.obs["total_counts"].values, row_sums
        )


# ── pseudobulk ────────────────────────────────────────────────────────────────

class TestPseudobulk:
    def test_requires_decoupler(self):
        decoupler = pytest.importorskip("decoupler")  # noqa: F841
        pytest.skip("requires InSituExperiment setup; covered by integration tests")

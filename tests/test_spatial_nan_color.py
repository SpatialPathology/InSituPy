"""Tests for NaN handling in pl.spatial and pl.embedding (nan_color parameter).

Covers:
 - _parse_unique_categories: drops NaN, no stringify
 - _coerce_na_for_plot: NaN -> "NaN" string
 - experiment builder crash-proof (reported TypeError)
 - pl.spatial: default hides NaN + warns; nan_color shows NaN
 - pl.embedding: warns when dropping NaN cells
 - no-NaN regression: clean columns unchanged
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
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment
from insitupy.utils._colors import _coerce_na_for_plot, _parse_unique_categories


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_TOTAL = 10
N_NAN = 3
N_CLEAN = N_TOTAL - N_NAN


def _make_xd_with_nan(seed=0, n_cells=N_TOTAL, n_nan=N_NAN, col="celltype"):
    """InSituData with an object-dtype obs column mixing str labels and NaN."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, 4))
    labels = np.array(["A", "B"] * (n_cells // 2), dtype=object)
    labels[:n_nan] = np.nan  # object dtype: np.nan stores as float NaN alongside str values
    obs = pd.DataFrame({col: labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(4)])
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100.0
    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="s1", sample_id="s1",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_xd_clean(seed=1, n_cells=N_TOTAL, col="celltype"):
    """InSituData with no NaN in the obs column."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, 4))
    labels = np.array(["A", "B"] * (n_cells // 2))
    obs = pd.DataFrame({col: labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(4)])
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100.0
    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="s2", sample_id="s2",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_experiment_with_nan():
    """Two-dataset InSituExperiment where the celltype column has NaN in both."""
    xd1 = _make_xd_with_nan(seed=0)
    xd2 = _make_xd_with_nan(seed=2)
    exp = InSituExperiment()
    exp._data.append(xd1)
    exp._data.append(xd2)
    exp._metadata = pd.DataFrame({"uid": ["s1", "s2"]})
    return exp


def _total_scatter_points(fig):
    """Count total data points across all scatter collections in the figure."""
    return sum(
        len(c.get_offsets())
        for a in fig.get_axes()
        for c in a.collections
    )


def _all_legend_labels(fig):
    """Collect all legend text labels across all axes in the figure."""
    labels = set()
    for a in fig.get_axes():
        leg = a.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                labels.add(t.get_text())
    return labels


# ── Unit: _parse_unique_categories ────────────────────────────────────────────

class TestParseUniqueCategories:
    def test_object_array_drops_nan_no_stringify(self):
        data = np.array(["A", "B", np.nan], dtype=object)
        result = _parse_unique_categories(data)
        assert list(result) == ["A", "B"]
        assert "nan" not in result
        assert "NaN" not in result

    def test_categorical_series_returns_categories(self):
        s = pd.Categorical(["A", "B", "A", np.nan])
        series = pd.Series(s)
        result = _parse_unique_categories(series)
        assert list(result) == list(series.cat.categories)
        assert len(result) == 2  # only "A" and "B"

    def test_numeric_array_drops_nan(self):
        data = np.array([1.0, 2.0, np.nan, 3.0])
        result = _parse_unique_categories(data)
        assert set(result) == {1.0, 2.0, 3.0}


# ── Unit: _coerce_na_for_plot ─────────────────────────────────────────────────

class TestCoerceNaForPlot:
    def test_object_series_nan_replaced(self):
        s = pd.Series(["A", "B", np.nan, "A"], dtype=object)
        result = _coerce_na_for_plot(s)
        assert result[2] == "NaN"
        assert result[0] == "A"
        assert result[1] == "B"
        assert result[3] == "A"

    def test_categorical_series_nan_replaced(self):
        s = pd.Categorical(["A", "B", np.nan])
        series = pd.Series(s)
        result = _coerce_na_for_plot(series)
        assert result[2] == "NaN"
        assert "NaN" in result.cat.categories
        assert result[0] == "A"

    def test_series_without_nan_unchanged(self):
        s = pd.Series(["A", "B", "A"])
        result = _coerce_na_for_plot(s)
        assert list(result) == ["A", "B", "A"]
        assert "NaN" not in result.values


# ── Unit: experiment color builder no longer crashes ─────────────────────────

class TestExperimentColorBuilder:
    def test_does_not_crash_with_nan(self):
        exp = _make_experiment_with_nan()
        result = exp._create_categorical_color_dict("celltype")
        assert result is not None
        assert "NaN" not in result
        assert "nan" not in result
        # only real categories
        assert set(result.keys()) == {"A", "B"}

    def test_returns_none_when_key_absent(self):
        exp = _make_experiment_with_nan()
        result = exp._create_categorical_color_dict("nonexistent_key")
        assert result is None


# ── Regression: pl.spatial on experiment no longer raises TypeError ──────────

class TestSpatialExperimentNoTypeerror:
    def test_experiment_with_nan_does_not_raise(self):
        exp = _make_experiment_with_nan()
        plt.close("all")
        # This was the reported failing call
        isp.pl.spatial(exp, keys=["celltype"], show=False)
        plt.close("all")


# ── Behavioral: default hides NaN + warns ─────────────────────────────────────

class TestSpatialNanDefault:
    def test_warns_when_nan_cells_hidden(self):
        xd = _make_xd_with_nan()
        plt.close("all")
        with pytest.warns(UserWarning, match="cell\\(s\\) with missing values"):
            isp.pl.spatial(xd, keys=["celltype"], show=False)
        plt.close("all")

    def test_only_non_nan_cells_plotted(self):
        xd = _make_xd_with_nan()
        plt.close("all")
        with pytest.warns(UserWarning):
            isp.pl.spatial(xd, keys=["celltype"], show=False)
        fig = plt.gcf()
        total_pts = _total_scatter_points(fig)
        assert total_pts == N_CLEAN
        plt.close("all")

    def test_no_nan_legend_entry(self):
        xd = _make_xd_with_nan()
        plt.close("all")
        with pytest.warns(UserWarning):
            isp.pl.spatial(xd, keys=["celltype"], show=False)
        fig = plt.gcf()
        labels = _all_legend_labels(fig)
        assert "NaN" not in labels
        plt.close("all")


# ── Behavioral: nan_color shows NaN, no warning ───────────────────────────────

class TestSpatialNanColor:
    def test_no_warning_when_nan_color_set(self):
        xd = _make_xd_with_nan()
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            isp.pl.spatial(xd, keys=["celltype"], nan_color="lightgray", show=False)
        plt.close("all")

    def test_all_cells_plotted(self):
        xd = _make_xd_with_nan()
        plt.close("all")
        isp.pl.spatial(xd, keys=["celltype"], nan_color="lightgray", show=False)
        fig = plt.gcf()
        total_pts = _total_scatter_points(fig)
        assert total_pts == N_TOTAL
        plt.close("all")

    def test_nan_legend_entry_present(self):
        xd = _make_xd_with_nan()
        plt.close("all")
        isp.pl.spatial(xd, keys=["celltype"], nan_color="lightgray", show=False)
        fig = plt.gcf()
        labels = _all_legend_labels(fig)
        assert "NaN" in labels
        plt.close("all")


# ── Smoke: embedding warns on drop ───────────────────────────────────────────

class TestEmbeddingNanWarn:
    def _make_adata_with_nan(self, n_cells=N_TOTAL, n_nan=N_NAN):
        rng = np.random.default_rng(42)
        X = rng.random((n_cells, 4))
        labels = np.array(["A", "B"] * (n_cells // 2))
        labels[:n_nan] = None
        obs = pd.DataFrame(
            {"celltype": pd.Categorical(labels, categories=["A", "B"])},
            index=[f"c{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"g{j}" for j in range(4)])
        adata = AnnData(X=X, obs=obs, var=var)
        adata.obsm["X_umap"] = rng.random((n_cells, 2))
        return adata

    def test_warns_by_default(self):
        adata = self._make_adata_with_nan()
        plt.close("all")
        with pytest.warns(UserWarning, match="cell\\(s\\) with missing values"):
            isp.pl.embedding(
                adata, keys=["celltype"],
                render_mode="matplotlib", show=False
            )
        plt.close("all")

    def test_no_warn_with_nan_color(self):
        adata = self._make_adata_with_nan()
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            isp.pl.embedding(
                adata, keys=["celltype"], nan_color="lightgray",
                render_mode="matplotlib", show=False
            )
        plt.close("all")


# ── Regression: clean column unchanged ───────────────────────────────────────

class TestNoNanRegression:
    def test_single_insitudata_no_warning(self):
        xd = _make_xd_clean()
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            isp.pl.spatial(xd, keys=["celltype"], show=False)
        plt.close("all")

    def test_single_insitudata_all_cells_plotted(self):
        xd = _make_xd_clean()
        plt.close("all")
        isp.pl.spatial(xd, keys=["celltype"], show=False)
        fig = plt.gcf()
        total_pts = _total_scatter_points(fig)
        assert total_pts == N_TOTAL
        plt.close("all")

    def test_single_insitudata_no_nan_legend_entry(self):
        xd = _make_xd_clean()
        plt.close("all")
        isp.pl.spatial(xd, keys=["celltype"], show=False)
        fig = plt.gcf()
        labels = _all_legend_labels(fig)
        assert "NaN" not in labels
        assert "nan" not in labels
        plt.close("all")

    def test_experiment_no_warning(self):
        xd1 = _make_xd_clean(seed=1)
        xd2 = _make_xd_clean(seed=3)
        exp = InSituExperiment()
        exp._data.append(xd1)
        exp._data.append(xd2)
        exp._metadata = pd.DataFrame({"uid": ["s1", "s2"]})
        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            isp.pl.spatial(exp, keys=["celltype"], show=False)
        plt.close("all")

"""Value extraction and color ranges in pl.spatial.

Covers pooled continuous color ranges across samples, NaN handling, and sparse
expression storage (X, layer, raw) which must neither crash nor be densified wholesale.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.plotting.spatial import DataConfig, PlotConfig, _ColorConfigMultiPlot
from insitupy.utils._colors import _extract_color_values


def _make_xd(values, sample_id="s1", sparse=None, seed=0):
    """Small sample with a numeric obs column ``total_counts`` and a 3-gene matrix.

    ``sparse`` is None (dense X) or one of "X", "layer", "raw" (that storage is csr_matrix).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    X = rng.random((n, 3))
    table = AnnData(
        X=csr_matrix(X) if sparse == "X" else X,
        obs=pd.DataFrame({"total_counts": values}, index=[f"{sample_id}_c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    if sparse == "layer":
        table.layers["counts"] = csr_matrix(X)
    if sparse == "raw":
        table.raw = AnnData(X=csr_matrix(X), obs=table.obs.copy(), var=table.var.copy())
    table.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    xd = InSituData(
        path=None, metadata=None, slide_id=sample_id, sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    return xd, X


def _experiment(*xds):
    exp = isp.InSituExperiment()
    for xd in xds:
        exp.add(xd)
    return exp


@pytest.mark.parametrize(
    "crange_type, expected",
    [
        ("percentile", lambda p: [np.percentile(p, 1), np.percentile(p, 99)]),
        ("upper_percentile", lambda p: [0, np.percentile(p, 99)]),
        ("minmax", lambda p: [p.min(), p.max()]),
        ("max", lambda p: [0, p.max()]),
    ],
)
def test_multi_sample_crange_uses_pooled_values(crange_type, expected):
    a = np.arange(100, dtype=float)
    b = np.arange(1000, 1010, dtype=float)
    exp = _experiment(_make_xd(a, "a")[0], _make_xd(b, "b", seed=1)[0])
    cfg = _ColorConfigMultiPlot(
        data=exp, data_config=DataConfig(), plot_config=PlotConfig(crange_type=crange_type),
        keys=["total_counts"],
    )
    pooled = np.concatenate([a, b])
    np.testing.assert_allclose(cfg["total_counts"]["crange"], expected(pooled))
    assert cfg["total_counts"]["max_value"] == pooled.max()


@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("crange_type", ["percentile", "minmax", "upper_percentile", "max"])
def test_crange_ignores_nan_in_obs_column(n_samples, crange_type):
    a = np.arange(50, dtype=float)
    a[3] = np.nan
    xds = [_make_xd(a, "a")[0]]
    if n_samples == 2:
        xds.append(_make_xd(np.arange(60, 80, dtype=float), "b", seed=1)[0])
    data = xds[0] if n_samples == 1 else _experiment(*xds)
    cfg = _ColorConfigMultiPlot(
        data=data, data_config=DataConfig(), plot_config=PlotConfig(crange_type=crange_type),
        keys=["total_counts"],
    )
    entry = cfg["total_counts"]
    assert np.all(np.isfinite(entry["crange"]))
    assert np.isfinite(entry["max_value"])


@pytest.mark.parametrize("storage", ["X", "layer", "raw"])
def test_sparse_storage_values_and_plotting(storage):
    xd, dense = _make_xd(np.arange(20, dtype=float), sparse=storage)
    kwargs = {"raw": storage == "raw", "layer": "counts" if storage == "layer" else None}

    values, categorical = _extract_color_values(
        xd.cells["main"].table, key="g1", raw=kwargs["raw"], layer=kwargs["layer"]
    )
    assert not categorical
    assert isinstance(values, np.ndarray) and values.ndim == 1
    np.testing.assert_allclose(values, dense[:, 1])

    isp.pl.spatial(xd, keys="g1", data_config=DataConfig(**kwargs), show=False)
    plt.close("all")


def test_extract_color_values_returns_copy_of_dense_column():
    xd, dense = _make_xd(np.arange(20, dtype=float))
    values, _ = _extract_color_values(xd.cells["main"].table, key="g0", raw=False, layer=None)
    values[:] = -1
    np.testing.assert_allclose(xd.cells["main"].table.X[:, 0], dense[:, 0])
    assert not issparse(values)

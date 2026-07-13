"""Regression tests: pl.spatial and pl.cellular_composition accept an
InSituExperimentView instead of raising ValueError via _is_experiment()'s
exact-class check.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment, InSituExperimentView


def _make_xd(seed, n_cells=10, col="celltype"):
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
        slide_id=f"slide{seed}", sample_id=f"s{seed}",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_experiment(n_samples=3):
    exp = InSituExperiment()
    for i in range(n_samples):
        exp._data.append(_make_xd(seed=i))
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(n_samples)]})
    return exp


class TestSpatialAcceptsView:
    def test_spatial_on_view_does_not_raise(self):
        exp = _make_experiment(n_samples=3)
        view = exp[:2]
        assert isinstance(view, InSituExperimentView)

        plt.close("all")
        isp.pl.spatial(view, keys=["celltype"], show=False)
        plt.close("all")


class TestCellularCompositionAcceptsView:
    def test_cellular_composition_on_view_does_not_raise(self):
        exp = _make_experiment(n_samples=3)
        view = exp[:2]
        assert isinstance(view, InSituExperimentView)

        plt.close("all")
        isp.pl.cellular_composition(view, cell_type_col="celltype", save_only=True)
        plt.close("all")

"""Tests for InSituData.show(return_viewer=True) / InSituExperiment.show(return_viewer=True)
(U-B8/U-H17).

Verifies the napari Viewer is threaded through both layers without requiring
a live napari GUI: `insitupy._core._napari._show` is monkeypatched to return
a sentinel instead of opening a real viewer, and `napari` is stubbed into
sys.modules so the import guard in `InSituData.show` is satisfied even if
napari is not installed in the test environment.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

SENTINEL = object()


def _make_insitudata(slide_id="s", sample_id="x"):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 20, size=(5, 3)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(5)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(3)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((5, 2)) * 100
    cd = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id=slide_id, sample_id=sample_id,
        method_name="t", method_params={},
    )
    xd.cells.add_celldata(cd=cd, key="main", is_main=True)
    return xd


def _stub_show(**kwargs):
    # mirrors the real `_show` contract: only return the viewer when asked
    return SENTINEL if kwargs.get("return_viewer") else None


@pytest.fixture
def stub_napari(monkeypatch):
    if "napari" not in sys.modules:
        monkeypatch.setitem(sys.modules, "napari", types.ModuleType("napari"))
    monkeypatch.setattr("insitupy._core._napari._show", _stub_show)


def test_insitudata_show_returns_viewer_when_requested(stub_napari):
    xd = _make_insitudata()
    assert xd.show(return_viewer=True) is SENTINEL


def test_insitudata_show_returns_none_by_default(stub_napari):
    xd = _make_insitudata()
    assert xd.show() is None


def test_experiment_show_returns_viewer_when_requested(stub_napari):
    exp = InSituExperiment()
    exp.add(_make_insitudata())
    assert exp.show(0, return_viewer=True, auto_sync_colors=False) is SENTINEL


def test_experiment_show_returns_none_by_default(stub_napari):
    exp = InSituExperiment()
    exp.add(_make_insitudata())
    assert exp.show(0, auto_sync_colors=False) is None

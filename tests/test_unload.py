"""Tests for InSituData.unload() / InSituExperiment.unload() guard predicate.

Covers U-B3/U-H10: unload() must guard on `from_insitudata` (whether the
object is backed by a saved project), not on `_path is None`. A
`read_xenium()`-style object has a vendor-bundle path but is not backed by a
saved InSituPy project, so it must still refuse to unload loaded modalities
(previously it silently discarded them because `_path` was not None).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_insitudata(slide_id="slide1", sample_id="s1", path=None):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 20, size=(5, 3)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(5)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(3)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((5, 2)) * 100
    cd = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=path, metadata=None,
        slide_id=slide_id, sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=cd, key="main", is_main=True)
    return xd


# ── InSituData.unload ────────────────────────────────────────────────────────

def test_unload_raises_on_in_memory_object():
    """Loaded cells, from_insitudata False -> unload() raises and keeps the data."""
    xd = _make_insitudata()
    assert xd.from_insitudata is False
    assert not xd.cells.is_empty

    with pytest.raises(ValueError, match="from_insitudata"):
        xd.unload("cells")

    # guard must run before any clearing happens
    assert not xd.cells.is_empty
    assert xd.cells["main"].table.n_obs == 5


def test_unload_raises_on_nonproject_path(tmp_path):
    """A read_xenium()-style object (path set, but no .ispy) must still refuse."""
    vendor_dir = tmp_path / "vendor_bundle"
    vendor_dir.mkdir()
    xd = _make_insitudata(path=vendor_dir)
    assert xd._path is not None
    assert xd.from_insitudata is False

    with pytest.raises(ValueError, match="from_insitudata"):
        xd.unload("cells")

    assert not xd.cells.is_empty


def test_unload_after_saveas_then_reload_roundtrips(tmp_path):
    xd = _make_insitudata()
    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)
    assert xd.from_insitudata is True

    xd.unload("cells")
    assert xd.cells.is_empty

    xd.load_cells()
    assert not xd.cells.is_empty
    assert xd.cells["main"].table.n_obs == 5


def test_unload_noop_when_nothing_loaded():
    """Early-exit path: unloading with nothing loaded must not raise, even in-memory."""
    xd = InSituData(
        path=None, metadata=None,
        slide_id="s", sample_id="x",
        method_name="t", method_params={},
    )
    assert xd.from_insitudata is False
    xd.unload("cells")  # nothing loaded -> no-op, must not raise


# ── InSituExperiment.unload ──────────────────────────────────────────────────

def test_experiment_unload_raises_when_any_dataset_pathless():
    exp = InSituExperiment()
    exp._data.append(_make_insitudata(slide_id="a", sample_id="1"))
    exp._metadata = pd.DataFrame({"uid": ["a"]})

    with pytest.raises(ValueError, match=r"index \[0\]"):
        exp.unload()

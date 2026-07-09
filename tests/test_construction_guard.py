"""Tests for the InSituData constructor guard and from_insitudata semantics."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy._exceptions import InSituDataConstructorPathError
from insitupy.containers.cell_data import CellData

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_insitudata(slide_id="slide1", sample_id="s1"):
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
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=cd, key="main", is_main=True)
    return xd


# ── constructor guard (change #1) ───────────────────────────────────────────

def test_constructor_rejects_saved_project_path(tmp_path):
    xd = _make_insitudata()
    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)

    with pytest.raises(InSituDataConstructorPathError, match="InSituData.read"):
        InSituData(proj_dir)


def test_read_still_loads_saved_project(tmp_path):
    xd = _make_insitudata()
    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)

    xd2 = InSituData.read(proj_dir)
    assert xd2.from_insitudata is True


def test_constructor_allows_in_memory_object():
    xd = InSituData(
        path=None, metadata=None,
        slide_id="s", sample_id="x",
        method_name="t", method_params={},
    )
    assert xd is not None


def test_constructor_allows_nonproject_path(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    xd = InSituData(
        path=empty_dir, metadata=None,
        slide_id="s", sample_id="x",
        method_name="t", method_params={},
    )
    assert xd is not None


# ── from_insitudata (change #2) ─────────────────────────────────────────────

def test_from_insitudata_false_for_in_memory():
    assert _make_insitudata().from_insitudata is False


def test_from_insitudata_false_for_nonproject_path(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    xd = InSituData(
        path=empty_dir, metadata=None,
        slide_id="s", sample_id="x",
        method_name="t", method_params={},
    )
    assert xd.from_insitudata is False


def test_from_insitudata_true_after_read(tmp_path):
    xd = _make_insitudata()
    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)

    xd2 = InSituData.read(proj_dir)
    assert xd2.from_insitudata is True


def test_from_insitudata_true_after_saveas(tmp_path):
    xd = _make_insitudata()
    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)

    assert xd.from_insitudata is True

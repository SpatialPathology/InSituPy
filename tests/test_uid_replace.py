"""Tests for InSituData.uid, InSituExperiment.replace(), and metadata reclassification."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment


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


def _make_saved_experiment(tmp_path):
    """Return (exp, save_dir) with one saved+reloaded dataset."""
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    save_dir = tmp_path / "exp"
    exp.saveas(save_dir)
    exp = InSituExperiment.read(save_dir)
    return exp, save_dir


# ── Phase 1: InSituData.uid ───────────────────────────────────────────────────

def test_uid_is_none_before_add():
    xd = _make_insitudata()
    assert xd.uid is None


def test_uid_assigned_after_add():
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    assert xd.uid is not None
    assert isinstance(xd.uid, str)


def test_uid_matches_experiment_metadata():
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    assert xd.uid == exp.metadata.loc[0, "uid"]


def test_uid_persists_on_save_and_load(tmp_path):
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    uid_before = xd.uid

    save_dir = tmp_path / "exp"
    exp.saveas(save_dir)
    loaded = InSituData.read(save_dir / "data-000")
    assert loaded.uid == uid_before


def test_cross_experiment_add_assigns_fresh_uid():
    exp1 = InSituExperiment()
    exp2 = InSituExperiment()
    xd = _make_insitudata()
    exp1.add(xd)
    uid1 = xd.uid

    exp2.add(xd)
    uid2 = xd.uid

    # New experiment gets a fresh uid
    assert uid2 != uid1
    assert uid2 == exp2.metadata.loc[0, "uid"]


def test_idempotent_readd_same_experiment():
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    uid_first = xd.uid

    exp.add(xd)  # re-add: same uid, same slot
    assert xd.uid == uid_first
    assert len(exp.data) == 1  # no duplicate appended


# ── Phase 2: metadata reclassification ──────────────────────────────────────

def test_experiment_metadata_has_no_slide_sample_columns():
    exp = InSituExperiment()
    xd = _make_insitudata(slide_id="sl1", sample_id="s1")
    exp.add(xd)
    assert "slide_id" not in exp.metadata.columns
    assert "sample_id" not in exp.metadata.columns
    assert "uid" in exp.metadata.columns


def test_update_metadata_emits_futurewarning():
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    with pytest.warns(FutureWarning, match="deprecated"):
        exp.update_metadata()


def test_add_metadata_column_reserved_name_warns():
    exp = InSituExperiment()
    xd = _make_insitudata()
    exp.add(xd)
    with pytest.warns(UserWarning, match="intrinsic attribute"):
        exp.add_metadata_column("slide_id", ["myslide"])


# ── Phase 4: legacy load-path migration ─────────────────────────────────────

def test_legacy_metadata_csv_with_slide_sample_loads_without_error(tmp_path):
    """Old metadata.csv with slide_id/sample_id columns loads and drops them."""
    metadata = pd.DataFrame({
        "uid": ["abc", "def"],
        "slide_id": ["sl1", "sl2"],
        "sample_id": ["s1", "s2"],
        "n_cells": [10, 20],
    })
    metadata.to_csv(tmp_path / "metadata.csv")

    loaded = InSituExperiment._read_insitupy(tmp_path)
    out = loaded.metadata
    assert "slide_id" not in out.columns
    assert "sample_id" not in out.columns
    assert "uid" in out.columns
    assert "n_cells" in out.columns


# ── Phase 3: replace() ───────────────────────────────────────────────────────

def test_replace_swaps_memory_by_int(tmp_path):
    exp, _ = _make_saved_experiment(tmp_path)
    slot_uid = exp.metadata.loc[0, "uid"]

    xd_new = _make_insitudata(slide_id="new", sample_id="n1")
    exp.replace(0, xd_new, confirm=False)

    assert exp.data[0] is xd_new
    assert xd_new.uid == slot_uid


def test_replace_swaps_memory_by_uid(tmp_path):
    exp, _ = _make_saved_experiment(tmp_path)
    uid = exp.metadata.loc[0, "uid"]

    xd_new = _make_insitudata(slide_id="new", sample_id="n2")
    exp.replace(uid, xd_new, confirm=False)

    assert exp.data[0] is xd_new
    assert xd_new.uid == uid


def test_replace_writes_to_disk(tmp_path):
    exp, save_dir = _make_saved_experiment(tmp_path)

    xd_new = _make_insitudata(slide_id="replacement")
    exp.replace(0, xd_new, confirm=False)

    # Reload and verify the replacement is on disk
    reloaded = InSituData.read(save_dir / "data-000")
    assert reloaded.slide_id == "replacement"


def test_replace_warns_if_new_data_has_uid(tmp_path):
    exp, _ = _make_saved_experiment(tmp_path)

    exp2 = InSituExperiment()
    xd2 = _make_insitudata()
    exp2.add(xd2)  # gives xd2 a uid

    with pytest.warns(UserWarning, match="uid"):
        exp.replace(0, xd2, confirm=False)


def test_replace_bad_int_index_raises(tmp_path):
    exp, _ = _make_saved_experiment(tmp_path)

    with pytest.raises(IndexError):
        exp.replace(99, _make_insitudata(), confirm=False)


def test_replace_bad_uid_raises(tmp_path):
    exp, _ = _make_saved_experiment(tmp_path)

    with pytest.raises(KeyError):
        exp.replace("nonexistent-uid", _make_insitudata(), confirm=False)

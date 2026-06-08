"""Tests for the atomic-replace swap in InSituExperiment.saveas.

Covers acceptance criteria from the 260608 saveas-atomic-replace report:
- no temp/backup dirs left on success (new save and overwrite)
- failure during write loop leaves old destination intact
- failure at the final rename restores the old destination
"""

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_xd(seed, n_cells=5, n_genes=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    cd = CellData(table=table, boundaries=None)
    xd = InSituData(path=None, metadata=None,
                    slide_id="s", sample_id="s", method_name="t", method_params={})
    xd.cells.add_celldata(cd=cd, key="main", is_main=True)
    return xd


def _make_exp(n=2):
    exp = InSituExperiment()
    for i in range(n):
        exp._data.append(_make_xd(i))
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(n)]})
    return exp


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_saveas_new_leaves_no_temp_dirs(tmp_path):
    """saveas to a fresh directory must leave no .__ispy_tmp__ or .__ispy_bak__ siblings."""
    exp = _make_exp()
    dest = tmp_path / "exp"
    exp.saveas(dest)

    assert dest.exists()
    assert not (tmp_path / "exp.__ispy_tmp__").exists()
    assert not (tmp_path / "exp.__ispy_bak__").exists()


def test_saveas_overwrite_replaces_and_cleans(tmp_path):
    """saveas twice with overwrite=True must replace content and leave no temp/backup dirs."""
    exp = _make_exp()
    dest = tmp_path / "exp"
    exp.saveas(dest)

    # Write a sentinel file inside so we can confirm replacement.
    sentinel = dest / "sentinel.txt"
    sentinel.write_text("old")

    exp2 = _make_exp(n=3)
    exp2.saveas(dest, overwrite=True)

    assert dest.exists()
    assert not sentinel.exists(), "old sentinel must be gone after overwrite"
    assert not (tmp_path / "exp.__ispy_tmp__").exists()
    assert not (tmp_path / "exp.__ispy_bak__").exists()


def test_saveas_overwrite_false_raises_without_writing(tmp_path):
    """saveas with overwrite=False must raise FileExistsError before any write."""
    exp = _make_exp()
    dest = tmp_path / "exp"
    exp.saveas(dest)

    sentinel = dest / "sentinel.txt"
    sentinel.write_text("old")

    with pytest.raises(FileExistsError):
        exp.saveas(dest, overwrite=False)

    assert sentinel.exists(), "sentinel must be untouched when overwrite=False raises"


def test_saveas_failure_in_write_loop_keeps_old(tmp_path, monkeypatch):
    """A failure during the dataset write loop must leave the old destination intact."""
    exp = _make_exp()
    dest = tmp_path / "exp"
    exp.saveas(dest)

    sentinel = dest / "sentinel.txt"
    sentinel.write_text("old")

    call_count = [0]
    original_saveas = InSituData.saveas

    def _fail_on_second(self, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("simulated disk failure")
        return original_saveas(self, *args, **kwargs)

    monkeypatch.setattr(InSituData, "saveas", _fail_on_second)

    with pytest.raises(RuntimeError, match="simulated disk failure"):
        exp.saveas(dest, overwrite=True)

    assert sentinel.exists(), "old destination must survive a write-loop failure"
    assert not (tmp_path / "exp.__ispy_tmp__").exists(), "staging must be cleaned up"
    assert not (tmp_path / "exp.__ispy_bak__").exists(), "no backup should exist (swap never reached)"


def test_saveas_failure_at_swap_restores_old(tmp_path, monkeypatch):
    """A failure at the final os.rename(staging, path) must restore the old destination."""
    exp = _make_exp()
    dest = tmp_path / "exp"
    exp.saveas(dest)

    sentinel = dest / "sentinel.txt"
    sentinel.write_text("old")

    original_rename = os.rename

    def _fail_on_staging_rename(src, dst):
        # Let path→backup pass; fail only when staging (.__ispy_tmp__) is the source.
        if str(src).endswith(".__ispy_tmp__"):
            raise OSError("simulated rename failure")
        return original_rename(src, dst)

    monkeypatch.setattr("insitupy.experiment.data.os.rename", _fail_on_staging_rename)

    with pytest.raises(OSError, match="simulated rename failure"):
        exp.saveas(dest, overwrite=True)

    assert dest.exists(), "old destination must be restored after swap failure"
    assert sentinel.exists(), "old sentinel must survive after swap failure"
    assert not (tmp_path / "exp.__ispy_tmp__").exists(), "staging must be cleaned up"
    assert not (tmp_path / "exp.__ispy_bak__").exists(), "backup must be cleaned up after restore"

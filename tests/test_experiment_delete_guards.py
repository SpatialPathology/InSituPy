"""Tests for the destructive-operation guards of InSituExperiment.

Covers the ROAD-8 batch (report 260924 "track-b-atomic-commit"):
- ``remove(delete_from_disk=True)`` refuses datasets outside the experiment folder (B5)
- ``concat(mode="move")`` refuses a non-empty destination (T2) and unexpected content in a
  source experiment root unless ``force=True`` (B5), before anything is moved; a source
  ``tables/`` folder is derived data that is not carried over (T18)
- staging/backup leftovers of an interrupted save are not mistaken for datasets
- experiment-level ``zip_output`` is rejected (its result could not be read back)
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_xd(i):
    rng = np.random.default_rng(i)
    table = AnnData(
        X=rng.integers(0, 10, size=(5, 3)).astype(float),
        obs=pd.DataFrame(index=pd.Index([f"s{i}c{j}" for j in range(5)])),
        var=pd.DataFrame(index=pd.Index([f"g{j}" for j in range(3)])),
    )
    table.obsm["spatial"] = rng.random((5, 2)) * 100
    xd = InSituData(path=None, metadata=None,
                    slide_id=f"slide{i}", sample_id=f"sample{i}", method_name="t", method_params={})
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    return xd


def _saved_experiment(tmp_path, name="exp", n=2, offset=0):
    """Save an n-dataset experiment to ``tmp_path/name`` and read it back."""
    exp = InSituExperiment()
    for i in range(n):
        exp.add(_make_xd(offset + i))
    exp.saveas(tmp_path / name)
    return InSituExperiment.read(tmp_path / name)


# ── Interrupted-save leftovers ────────────────────────────────────────────────


def test_read_skips_staging_leftovers(tmp_path):
    exp = _saved_experiment(tmp_path)
    root = tmp_path / "exp"
    (root / "data-000.__ispy_tmp__").mkdir()  # partial dataset: no .ispy inside
    (root / "data-000.__ispy_tmp__" / "half-written.bin").write_text("x")
    (root / "data-001.__ispy_bak__").mkdir()

    with pytest.warns(UserWarning, match="left over from an interrupted save"):
        reread = InSituExperiment.read(root)

    assert len(reread) == len(exp) == 2


# ── remove(delete_from_disk=True) only deletes what the experiment owns ──────


def test_remove_delete_outside_root_refused(tmp_path):
    exp = _saved_experiment(tmp_path, n=1)
    external = tmp_path / "external"
    _make_xd(9).saveas(external, verbose=False)
    exp.add(InSituData.read(external))
    assert len(exp) == 2

    with pytest.raises(ValueError, match="outside this experiment's directory"):
        exp.remove(1, delete_from_disk=True, confirm=False)

    assert (external / ".ispy").exists(), "the external dataset must survive"
    assert len(exp) == 2, "the guard fires before the in-memory removal"


def test_remove_delete_without_experiment_dir_refused(tmp_path):
    external = tmp_path / "external"
    _make_xd(9).saveas(external, verbose=False)
    exp = InSituExperiment()
    exp.add(InSituData.read(external))  # in-memory experiment: it owns no directory

    with pytest.raises(ValueError, match="does not own it"):
        exp.remove(0, delete_from_disk=True, confirm=False)
    assert external.exists() and len(exp) == 1


# ── concat(mode="move"): refusals happen before anything moves ───────────────


def _slot_names(root):
    return sorted(p.name for p in root.glob("data-*"))


def test_concat_move_unexpected_content_refused(tmp_path):
    a = _saved_experiment(tmp_path, "expA", offset=0)
    b = _saved_experiment(tmp_path, "expB", offset=10)
    (tmp_path / "expA" / "notes.txt").write_text("my notes")
    dst = tmp_path / "merged"

    with pytest.raises(ValueError, match=r"notes\.txt"):
        InSituExperiment.concat([a, b], path=dst, mode="move")

    assert _slot_names(tmp_path / "expA") == ["data-000", "data-001"]
    assert _slot_names(tmp_path / "expB") == ["data-000", "data-001"]
    assert (tmp_path / "expA" / "notes.txt").read_text() == "my notes"
    assert not dst.exists()
    assert a.path == tmp_path / "expA" and b.path == tmp_path / "expB"


def test_concat_move_force_deletes(tmp_path):
    a = _saved_experiment(tmp_path, "expA", offset=0)
    b = _saved_experiment(tmp_path, "expB", offset=10)
    (tmp_path / "expA" / "notes.txt").write_text("my notes")
    dst = tmp_path / "merged"

    with pytest.warns(UserWarning, match="Removing source experiment root"):
        merged = InSituExperiment.concat([a, b], path=dst, mode="move", force=True)

    assert not (tmp_path / "expA").exists() and not (tmp_path / "expB").exists()
    assert len(merged) == 4
    assert _slot_names(dst) == ["data-000", "data-001", "data-002", "data-003"]
    assert len(InSituExperiment.read(dst)) == 4


def test_concat_move_into_nonempty_path_refused(tmp_path):
    a = _saved_experiment(tmp_path, "expA", offset=0)
    b = _saved_experiment(tmp_path, "expB", offset=10)
    dst = tmp_path / "merged"
    dst.mkdir()
    (dst / "keep.txt").write_text("x")

    for force in (False, True):  # a non-empty destination has no override
        with pytest.raises(ValueError, match="absent or an empty directory"):
            InSituExperiment.concat([a, b], path=dst, mode="move", force=force)

    assert [p.name for p in dst.iterdir()] == ["keep.txt"]
    assert _slot_names(tmp_path / "expA") == ["data-000", "data-001"]
    assert _slot_names(tmp_path / "expB") == ["data-000", "data-001"]


def test_concat_move_into_source_experiment_refused(tmp_path):
    """A destination inside a source root would be deleted together with that root."""
    a = _saved_experiment(tmp_path, "expA", offset=0)
    b = _saved_experiment(tmp_path, "expB", offset=10)

    with pytest.raises(ValueError, match="cannot write into a source experiment"):
        InSituExperiment.concat([a, b], path=tmp_path / "expA" / "merged", mode="move", force=True)

    assert _slot_names(tmp_path / "expA") == ["data-000", "data-001"]
    assert not (tmp_path / "expA" / "merged").exists()


def test_concat_move_drops_source_tables(tmp_path):
    a = _saved_experiment(tmp_path, "expA", offset=0)
    b = _saved_experiment(tmp_path, "expB", offset=10)
    a.load_cells()
    a.build_table()
    assert (tmp_path / "expA" / "tables").is_dir()
    dst = tmp_path / "merged"
    dst.mkdir()  # an existing EMPTY destination is fine

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InSituExperiment.concat([a, b], path=dst, mode="move")  # no force needed for tables/

    # tables/ is InSituPy-owned derived data: not "unexpected content" that needs consent
    assert not [w for w in caught if "unexpected" in str(w.message) or "did not write" in str(w.message)]
    assert not (tmp_path / "expA").exists()
    assert not (dst / "tables").exists(), "the derived union table is not carried over"
    assert len(InSituExperiment.read(dst)) == 4


# ── experiment-level zip_output ───────────────────────────────────────────────


def test_experiment_zip_output_rejected(tmp_path):
    exp = _saved_experiment(tmp_path, "exp", n=1)
    target = tmp_path / "zipped"

    with pytest.raises(ValueError, match="zip_output is not supported"):
        exp.saveas(target, zip_output=True)
    with pytest.raises(ValueError, match="zip_output is not supported"):
        exp.save(zip_output=True)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["exp"], "nothing may be written"

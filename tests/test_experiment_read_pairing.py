"""Tests for InSituExperiment dataset/metadata pairing on read() (T1).

Covers the 260916 T1 fix: InSituExperiment._read_insitupy() now pairs on-disk
data-* directories to metadata rows by uid instead of by directory order, so a
stray directory left behind by remove(delete_from_disk=False) can no longer be
silently mispaired with the wrong metadata row. save() also warns about orphan
data-* directories that are not part of the experiment.
"""

import json
import shutil
import types
import warnings

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


def _make_three_sample_experiment(tmp_path):
    """Build and save a 3-dataset experiment (G1/G2/G3). Returns (exp, dest, uids)."""
    exp = InSituExperiment()
    uids = []
    for i, label in enumerate(["G1", "G2", "G3"]):
        xd = _make_xd(seed=i)
        exp.add(xd, metadata={"label": label})
        uids.append(xd.uid)
    dest = tmp_path / "exp"
    exp.saveas(dest)
    return exp, dest, uids


def _stub(uid):
    """Minimal stand-in for a loaded InSituData: the helper only reads ._uid."""
    return types.SimpleNamespace(_uid=uid)


# ── 1. Primary regression: remove middle sample, save, re-read ─────────────────


def test_read_pairs_by_uid_after_remove_middle(tmp_path):
    exp, dest, uids = _make_three_sample_experiment(tmp_path)
    uid_g1, uid_g2, uid_g3 = uids

    exp.remove(1, confirm=False)  # drop G2 in memory, default delete_from_disk=False
    exp.save()

    # The removed dataset's directory must still be on disk (not pruned)
    assert (dest / "data-001").exists()

    with pytest.warns(UserWarning, match="orphan|Skipping"):
        exp2 = InSituExperiment.read(dest)

    assert len(exp2) == 2
    for i in range(len(exp2)):
        assert exp2._data[i].uid == exp2._metadata.iloc[i]["uid"]

    surviving_uids = set(exp2._metadata["uid"])
    assert uid_g2 not in surviving_uids
    assert uid_g2 not in {d.uid for d in exp2._data}
    assert surviving_uids == {uid_g1, uid_g3}

    labels_by_uid = dict(zip(exp2._metadata["uid"], exp2._metadata["label"]))
    assert labels_by_uid[uid_g1] == "G1"
    assert labels_by_uid[uid_g3] == "G3"


# ── 2. Metadata uid with no matching directory raises ──────────────────────────


def test_read_raises_on_metadata_uid_without_dir(tmp_path):
    exp, dest, uids = _make_three_sample_experiment(tmp_path)
    missing_uid = uids[2]

    shutil.rmtree(dest / "data-002")

    with pytest.raises(ValueError, match=missing_uid):
        InSituExperiment.read(dest)


# ── 3. Helper-level branch coverage for _pair_loaded_datasets ──────────────────


def test_pair_loaded_datasets_orphan_skipped_with_warning():
    loaded = [_stub("a"), _stub("b")]
    metadata = pd.DataFrame({"uid": ["a"]})

    with pytest.warns(UserWarning, match="orphan|Skipping"):
        result = InSituExperiment._pair_loaded_datasets(loaded, metadata, "path")

    assert [d._uid for d in result] == ["a"]


def test_pair_loaded_datasets_missing_uid_raises():
    loaded = [_stub("a")]
    metadata = pd.DataFrame({"uid": ["a", "b"]})

    with pytest.raises(ValueError, match="b"):
        InSituExperiment._pair_loaded_datasets(loaded, metadata, "path")


def test_pair_loaded_datasets_duplicate_uid_raises():
    loaded = [_stub("a"), _stub("a")]
    metadata = pd.DataFrame({"uid": ["a"]})

    with pytest.raises(ValueError, match="share the uid"):
        InSituExperiment._pair_loaded_datasets(loaded, metadata, "path")


def test_pair_loaded_datasets_legacy_positional_ok():
    loaded = [_stub(None), _stub(None)]
    metadata = pd.DataFrame({"label": ["x", "y"]})  # no uid column, equal counts

    result = InSituExperiment._pair_loaded_datasets(loaded, metadata, "path")

    assert result is loaded


def test_pair_loaded_datasets_legacy_count_mismatch_raises():
    loaded = [_stub(None), _stub(None), _stub(None)]
    metadata = pd.DataFrame({"label": ["x", "y"]})  # no uid column, 2 rows vs 3 dirs

    with pytest.raises(ValueError, match="predates per-dataset uids"):
        InSituExperiment._pair_loaded_datasets(loaded, metadata, "path")


def test_pair_loaded_datasets_mixed_store_raises():
    loaded = [_stub("a"), _stub(None)]
    metadata = pd.DataFrame({"uid": ["a", "b"]})

    with pytest.raises(ValueError, match="Re-save the experiment"):
        InSituExperiment._pair_loaded_datasets(loaded, metadata, "path")


# ── 4. save() warns about orphan directories ────────────────────────────────────


def test_save_warns_on_orphan_dirs(tmp_path):
    exp, dest, uids = _make_three_sample_experiment(tmp_path)

    exp.remove(1, confirm=False)

    with pytest.warns(UserWarning, match="orphan"):
        exp.save()

    assert (dest / "data-001").exists()


# ── 5. remove(delete_from_disk=True): consistent disk state and trash ──────────
# Report 260929 "remove-delete-disk-consistency": the directory used to be deleted while
# metadata.parquet / filters.json still listed the dataset until save(), so an interrupted
# session left an experiment that read() refused.


def _make_filtered_experiment(tmp_path):
    exp, dest, uids = _make_three_sample_experiment(tmp_path)
    exp._filters["keep"] = {"mask": [True, False, True], "note": ""}
    exp.save_filters()
    return exp, dest, uids


def _read_recording_warnings(dest):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exp = InSituExperiment.read(dest)
    return exp, [str(w.message) for w in caught]


def test_remove_delete_from_disk_keeps_store_readable_without_save(tmp_path):
    exp, dest, (uid_g1, uid_g2, uid_g3) = _make_filtered_experiment(tmp_path)

    exp.remove(uid_g2, confirm=False, delete_from_disk=True)  # no save()

    exp2, messages = _read_recording_warnings(dest)

    assert list(exp2._metadata["uid"]) == [uid_g1, uid_g3]
    assert [d.uid for d in exp2._data] == [uid_g1, uid_g3]
    assert list(exp2._metadata["label"]) == ["G1", "G3"]
    assert exp2._filters["keep"]["mask"] == [True, True]
    assert not [m for m in messages if "does not match metadata length" in m]
    assert not [m for m in messages if "orphan" in m or "Skipping" in m]
    assert [m for m in messages if "empty_trash" in m], "read() must report the trash"


def test_remove_delete_from_disk_does_not_write_unsaved_add(tmp_path):
    exp, dest, (uid_g1, uid_g2, uid_g3) = _make_three_sample_experiment(tmp_path)
    exp.add(_make_xd(seed=9), metadata={"label": "unsaved"})  # in memory only

    exp.remove(uid_g2, confirm=False, delete_from_disk=True)

    assert len(exp) == 3, "the unsaved dataset stays in memory"
    exp2, _ = _read_recording_warnings(dest)
    assert list(exp2._metadata["uid"]) == [uid_g1, uid_g3]


def test_remove_delete_from_disk_moves_directory_to_trash(tmp_path):
    exp, dest, (_, uid_g2, _) = _make_filtered_experiment(tmp_path)

    exp.remove(uid_g2, confirm=False, delete_from_disk=True)

    assert not (dest / "data-001").exists()
    folders = [p for p in (dest / ".trash").iterdir() if p.is_dir()]
    assert len(folders) == 1
    assert (folders[0] / ".ispy").exists(), "the dataset is moved, not deleted"
    assert InSituData.read(folders[0]).uid == uid_g2

    (entry,) = json.loads((dest / ".trash" / "manifest.json").read_text())["entries"]
    assert entry["folder"] == folders[0].name
    assert entry["uid"] == uid_g2
    assert entry["sample_id"] == "s"
    assert entry["original_dir"] == "data-001"
    assert entry["size_bytes"] > 0
    assert entry["metadata_row"]["label"] == "G2"
    assert entry["filters"] == {"keep": False}


def test_failed_move_to_trash_leaves_a_readable_store(tmp_path):
    """If the directory cannot be moved, it stays whole as an orphan and the store reads."""
    exp, dest, (uid_g1, uid_g2, uid_g3) = _make_three_sample_experiment(tmp_path)
    (dest / ".trash").write_text("not a directory")  # makes the move fail on every platform

    with pytest.raises(RuntimeError, match="could not be moved to the trash"):
        exp.remove(uid_g2, confirm=False, delete_from_disk=True)

    assert (dest / "data-001" / ".ispy").exists(), "the directory must be left untouched"
    assert list(exp._metadata["uid"]) == [uid_g1, uid_g3]
    exp2, messages = _read_recording_warnings(dest)
    assert list(exp2._metadata["uid"]) == [uid_g1, uid_g3]
    assert [m for m in messages if "Skipping" in m], "the directory is reported as an orphan"


def test_empty_trash_deletes_listed_and_unlisted_folders(tmp_path):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)
    unlisted = dest / ".trash" / "leftover"
    unlisted.mkdir()
    (unlisted / "file.bin").write_text("x")

    exp.empty_trash(confirm=False)

    assert not (dest / ".trash").exists()
    exp2, messages = _read_recording_warnings(dest)
    assert len(exp2) == 2
    assert not [m for m in messages if "trash" in m]


def test_empty_trash_prompt_defaults_to_no(tmp_path, monkeypatch):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)
    monkeypatch.setattr("builtins.input", lambda _: "")

    exp.empty_trash()

    assert len([p for p in (dest / ".trash").iterdir() if p.is_dir()]) == 1


def test_save_after_remove_delete_from_disk_roundtrips(tmp_path):
    exp, dest, (uid_g1, uid_g2, uid_g3) = _make_filtered_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exp.save()
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "orphan" in m]
    assert [m for m in messages if "empty_trash" in m], "save() must report the trash"

    exp2, _ = _read_recording_warnings(dest)
    assert list(exp2._metadata["uid"]) == [uid_g1, uid_g3]
    assert exp2._filters["keep"]["mask"] == [True, True]


def test_remove_delete_from_disk_unprompted_needs_uid(tmp_path, monkeypatch):
    exp, dest, (uid_g1, uid_g2, uid_g3) = _make_three_sample_experiment(tmp_path)

    with pytest.raises(TypeError, match="uid"):
        exp.remove(1, confirm=False, delete_from_disk=True)
    assert len(exp) == 3
    assert (dest / "data-001").exists() and not (dest / ".trash").exists()

    # with the prompt on, a person sees the summary, so a position is still allowed
    monkeypatch.setattr("builtins.input", lambda _: "y")
    exp.remove(1, delete_from_disk=True)
    assert list(exp._metadata["uid"]) == [uid_g1, uid_g3]
    assert not (dest / "data-001").exists()


def test_removed_dataset_is_detached_from_its_directory(tmp_path):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    xd = exp.data[1]

    exp.remove(uid_g2, confirm=False, delete_from_disk=True)

    assert xd.path is None
    with pytest.raises(RuntimeError, match="no project is linked"):
        xd.save()
    assert sorted(p.name for p in dest.glob("data-*")) == ["data-000", "data-002"]


def test_remove_delete_from_disk_refuses_pre_uid_store(tmp_path):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    parquet = dest / "metadata.parquet"
    pd.read_parquet(parquet).drop(columns="uid").to_parquet(parquet, index=False)

    with pytest.raises(ValueError, match="per-dataset uids"):
        exp.remove(uid_g2, confirm=False, delete_from_disk=True)

    assert len(exp) == 3
    assert (dest / "data-001").exists() and not (dest / ".trash").exists()
    assert len(pd.read_parquet(parquet)) == 3


def test_saveas_over_own_path_keeps_trash(tmp_path):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)
    manifest_before = (dest / ".trash" / "manifest.json").read_text()

    exp.saveas(dest, overwrite=True)

    assert (dest / ".trash" / "manifest.json").read_text() == manifest_before
    (entry,) = json.loads(manifest_before)["entries"]
    assert (dest / ".trash" / entry["folder"] / ".ispy").exists()
    assert not (tmp_path / "exp.__ispy_bak__").exists()


def _strand_trash_in_backup(tmp_path, dest):
    """Leave the state of a save that could not carry the trash over: it sits in the backup."""
    backup = tmp_path / "exp.__ispy_bak__"
    backup.mkdir()
    shutil.move(str(dest / ".trash"), str(backup / ".trash"))
    return backup


def test_saveas_recovers_trash_stranded_in_backup(tmp_path):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)
    manifest = (dest / ".trash" / "manifest.json").read_text()
    backup = _strand_trash_in_backup(tmp_path, dest)

    exp.saveas(dest, overwrite=True)

    assert (dest / ".trash" / "manifest.json").read_text() == manifest
    assert not backup.exists()


def test_saveas_refuses_to_drop_backup_with_unmergeable_trash(tmp_path):
    exp, dest, (uid_g1, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)
    backup = _strand_trash_in_backup(tmp_path, dest)
    exp.remove(uid_g1, confirm=False, delete_from_disk=True)  # a second trash in the experiment

    with pytest.raises(RuntimeError, match="still holds the trash"):
        exp.saveas(dest, overwrite=True)

    assert (backup / ".trash" / "manifest.json").exists(), "the stranded trash must survive"
    assert (dest / ".trash" / "manifest.json").exists()


# ── 6. replace(): the prompt defaults to no ────────────────────────────────────


def test_replace_prompt_defaults_to_no(tmp_path, monkeypatch):
    exp, dest, _ = _make_three_sample_experiment(tmp_path)
    new = _make_xd(seed=7)
    new.slide_id = "replacement"
    monkeypatch.setattr("builtins.input", lambda _: "")

    old = exp.data[0]

    exp.replace(0, new)

    assert InSituData.read(dest / "data-000").slide_id == "s", "the directory is unchanged"
    # no swap in memory either: a later save() must not write `new` into the slot
    assert exp.data[0] is old
    assert new.path is None and new.uid is None
    exp.save()
    assert InSituData.read(dest / "data-000").slide_id == "s"


def test_replace_pathless_slot_swaps_in_memory_without_prompt(tmp_path, monkeypatch):
    """A never-saved slot has nothing to overwrite: no prompt, no error, written by saveas()."""
    exp = InSituExperiment()
    exp.add(_make_xd(seed=0))
    slot_uid = exp.metadata.loc[0, "uid"]
    new = _make_xd(seed=1)
    new.slide_id = "replacement"

    def _no_prompt(_):
        raise AssertionError("replace() must not prompt for a slot without a path")

    monkeypatch.setattr("builtins.input", _no_prompt)

    exp.replace(0, new)

    assert exp.data[0] is new and new.uid == slot_uid
    exp.saveas(tmp_path / "exp")
    reread = InSituExperiment.read(tmp_path / "exp")
    assert reread.data[0].slide_id == "replacement" and reread.data[0].uid == slot_uid


def test_empty_trash_prompt_marks_unlisted_items(tmp_path, monkeypatch, capsys):
    exp, dest, (_, uid_g2, _) = _make_three_sample_experiment(tmp_path)
    exp.remove(uid_g2, confirm=False, delete_from_disk=True)
    (dest / ".trash" / "leftover").mkdir()
    monkeypatch.setattr("builtins.input", lambda _: "n")
    capsys.readouterr()

    exp.empty_trash()

    out = capsys.readouterr().out
    assert "'leftover'" in out and "not listed in the trash manifest" in out
    assert f"uid='{uid_g2}'" in out
    assert (dest / ".trash" / "leftover").exists(), "nothing is deleted after a no"

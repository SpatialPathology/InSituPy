"""Tests for InSituExperiment dataset/metadata pairing on read() (T1).

Covers the 260916 T1 fix: InSituExperiment._read_insitupy() now pairs on-disk
data-* directories to metadata rows by uid instead of by directory order, so a
stray directory left behind by remove(delete_from_disk=False) can no longer be
silently mispaired with the wrong metadata row. save() also warns about orphan
data-* directories that are not part of the experiment.
"""

import shutil
import types

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

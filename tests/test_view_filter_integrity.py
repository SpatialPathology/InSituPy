"""Regression tests for InSituExperimentView filter and data integrity bugs.

Covers: C1-C5 (critical), M1, M3 (major), m1, m2 (minor), _parent_indices invariant.
"""

import json

import pandas as pd
import pytest

from insitupy._core.data import InSituData
from insitupy.experiment.data import InSituExperiment, InSituExperimentView


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_datasets(n):
    """Return n minimal in-memory InSituData objects with sequential UIDs."""
    datasets = []
    for i in range(n):
        xd = InSituData()
        xd._uid = f"uid{i:02d}"
        datasets.append(xd)
    return datasets


def _make_metadata_experiment(tmp_path, n=10):
    """Experiment with metadata/filters/colors on disk; datasets are in-memory only."""
    exp = InSituExperiment()
    uids = [f"uid{i:02d}" for i in range(n)]
    exp._metadata = pd.DataFrame({
        "uid": uids,
        "group": [chr(65 + i % 3) for i in range(n)],
    })
    exp._data = _make_datasets(n)
    qc_mask = [i < n // 2 for i in range(n)]
    exp._filters = {"qc": {"mask": list(qc_mask), "note": None}}
    exp._colors = {"main": {"A": {"a": "#FF0000"}, "B": {"b": "#0000FF"}}}
    exp._path = tmp_path
    exp.save_metadata(path=tmp_path)
    exp.save_filters(path=tmp_path)
    exp.save_colors(path=tmp_path)
    return exp


def _make_full_experiment(parent_path, n=10):
    """Create and saveas a full experiment (datasets written to disk); return reloaded."""
    exp = InSituExperiment()
    uids = [f"uid{i:02d}" for i in range(n)]
    exp._data = _make_datasets(n)
    exp._metadata = pd.DataFrame({
        "uid": uids,
        "group": [chr(65 + i % 3) for i in range(n)],
    })
    qc_mask = [i < n // 2 for i in range(n)]
    exp._filters = {"qc": {"mask": list(qc_mask), "note": None}}
    exp._colors = {"main": {"A": {"a": "#FF0000"}, "B": {"b": "#0000FF"}}}
    exp.saveas(parent_path, overwrite=True)
    return InSituExperiment._read_insitupy(parent_path)


# ── C1 ────────────────────────────────────────────────────────────────────────


def test_c1_save_filters_preserves_parent_length(tmp_path):
    """view.save_filters() must merge masks into the full-length parent file."""
    n = 10
    exp = _make_metadata_experiment(tmp_path, n)
    original_qc = list(exp._filters["qc"]["mask"])

    view = exp._subset(slice(0, 5), as_view=True)
    assert isinstance(view, InSituExperimentView)
    assert view._parent_indices == list(range(5))

    # Flip view's mask to make the test meaningful.
    view._filters["qc"]["mask"] = [False] * 5
    view.save_filters()

    with open(tmp_path / "filters.json") as f:
        payload = json.load(f)

    on_disk = payload["filters"]["qc"]["mask"]
    assert len(on_disk) == n, f"mask length should be {n}, got {len(on_disk)}"
    assert on_disk[:5] == [False] * 5, "view positions should reflect view mask"
    assert on_disk[5:] == original_qc[5:], "non-view positions must be unchanged"


def test_c1_new_filter_on_view_pads_with_false_outside(tmp_path):
    """A filter created only on the view fills False for non-view parent rows."""
    n = 10
    exp = _make_metadata_experiment(tmp_path, n)

    view = exp._subset(slice(0, 5), as_view=True)
    view._filters["view_only"] = {"mask": [True, False, True, False, True], "note": None}
    view.save_filters()

    with open(tmp_path / "filters.json") as f:
        payload = json.load(f)

    assert "view_only" in payload["filters"]
    mask = payload["filters"]["view_only"]["mask"]
    assert len(mask) == n
    assert mask[:5] == [True, False, True, False, True]
    assert mask[5:] == [False] * 5


# ── C2 ────────────────────────────────────────────────────────────────────────


def test_c2_saveas_parent_path_overwrites_cleanly(tmp_path):
    """view.saveas(view.path, overwrite=True) produces a clean 5-dataset experiment."""
    exp = _make_full_experiment(tmp_path / "parent", n=10)
    parent_path = exp.path

    view = exp._subset(slice(0, 5), as_view=True)
    assert isinstance(view, InSituExperimentView)

    view.saveas(parent_path, overwrite=True)

    reloaded = InSituExperiment._read_insitupy(parent_path)
    assert len(reloaded._metadata) == 5
    assert not reloaded.is_view
    assert len(reloaded._filters["qc"]["mask"]) == 5
    assert "A" in reloaded._colors["main"]


# ── C3 ────────────────────────────────────────────────────────────────────────


def test_c3_saveas_fresh_path_leaves_view_path_unchanged(tmp_path):
    """view.saveas(export_path) does not mutate view.path."""
    exp = _make_full_experiment(tmp_path / "parent", n=10)
    export_path = tmp_path / "export"
    original_view_path = exp.path

    view = exp._subset(slice(0, 5), as_view=True)
    view.saveas(export_path)

    assert view.path == original_view_path, "view._path must not be mutated"

    reloaded = InSituExperiment._read_insitupy(export_path)
    assert len(reloaded._metadata) == 5
    assert not reloaded.is_view
    assert len(reloaded._filters["qc"]["mask"]) == 5
    assert reloaded.path == export_path


# ── C4 ────────────────────────────────────────────────────────────────────────


def test_c4_remove_delete_from_disk_raises_on_view():
    """remove(delete_from_disk=True) on a view must raise ValueError."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(5)]})
    exp._data = _make_datasets(5)
    exp._filters = {}
    # No exp._path → view._path = None → no disk ops attempted.

    view = exp._subset(slice(0, 3), as_view=True)

    with pytest.raises(ValueError, match="delete_from_disk=True is not allowed"):
        view.remove(0, delete_from_disk=True, confirm=False)


def test_c4_remove_in_memory_on_view_works():
    """remove(delete_from_disk=False) on a view only shrinks the view, not the parent."""
    n = 5
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(n)]})
    exp._data = _make_datasets(n)
    exp._filters = {"qc": {"mask": [True] * n, "note": None}}
    # exp._path deliberately None so remove() doesn't call save().

    view = exp._subset(slice(0, 3), as_view=True)
    initial_view_len = len(view._data)

    view.remove(0, delete_from_disk=False, confirm=False)

    assert len(view._data) == initial_view_len - 1
    assert len(exp._data) == n, "parent _data must not shrink"


# ── C5 ────────────────────────────────────────────────────────────────────────


def test_c5_replace_raises_on_view():
    """replace() on a view must raise ValueError."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(5)]})
    exp._data = _make_datasets(5)
    exp._filters = {}

    view = exp._subset(slice(0, 3), as_view=True)
    new_data = InSituData()

    with pytest.raises(ValueError, match="replace\\(\\) is not supported"):
        view.replace(0, new_data, confirm=False)


# ── M1 ────────────────────────────────────────────────────────────────────────


def test_m1_save_colors_preserves_parent_categories(tmp_path):
    """view.save_colors() must not drop color keys absent from the view."""
    exp = _make_metadata_experiment(tmp_path)
    assert "B" in exp._colors["main"]

    view = exp._subset(slice(0, 5), as_view=True)
    # Simulate view missing "B" (e.g. sync_colors only saw "A" datasets).
    view._colors = {"main": {"A": {"a": "#FF0000"}}}
    view.save_colors()

    with open(tmp_path / "colors.json") as f:
        colors = json.load(f)

    assert "B" in colors["main"], "parent's 'B' color key must survive view.save_colors()"
    assert "A" in colors["main"]


# ── M3 ────────────────────────────────────────────────────────────────────────


def test_m3_add_raises_on_view():
    """view.add() must raise NotImplementedError (M2/R5 fix).

    Adding a dataset to a view would corrupt the parent experiment's state
    (desync metadata, data, and filter masks).  The view override rejects
    add() entirely and tells the user to add on the parent instead.
    """
    n = 5
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(n)]})
    exp._data = _make_datasets(n)
    exp._filters = {"qc": {"mask": [True] * n, "note": None}}

    view = exp._subset(slice(0, 3), as_view=True)

    with pytest.raises(NotImplementedError, match="parent experiment"):
        view.add(InSituData())


# ── m1 ────────────────────────────────────────────────────────────────────────


def test_m1_negative_index_returns_last_row():
    """exp[-1] must return the last dataset, not an empty subset."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["a", "b", "c"]})
    exp._data = _make_datasets(3)

    subset = exp._subset(-1, as_view=False)
    assert len(subset._metadata) == 1
    assert subset._metadata["uid"].iloc[0] == "c"


# ── m2 ────────────────────────────────────────────────────────────────────────


def test_m2_bool_series_foreign_index_selects_by_position():
    """A bool pd.Series with a foreign index must select by position, not label."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["a", "b", "c"]})
    exp._data = _make_datasets(3)

    key = pd.Series([True, False, True], index=["x", "y", "z"])
    subset = exp._subset(key, as_view=False)

    assert len(subset._metadata) == 2
    assert list(subset._metadata["uid"]) == ["a", "c"]


# ── _parent_indices invariant ─────────────────────────────────────────────────


def test_parent_indices_composition():
    """_parent_indices for a view-of-view must index into the root experiment."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"u{i}" for i in range(8)]})
    exp._data = _make_datasets(8)

    view = exp._subset([2, 5, 7], as_view=True)
    assert view._parent_indices == [2, 5, 7]

    view_of_view = view._subset([0, 2], as_view=True)
    assert view_of_view._parent_indices == [2, 7], (
        "view-of-view indices must reference root experiment positions"
    )


# ── New fix tests (R1–R8, R11) ────────────────────────────────────────────────


def test_r1_build_table_raises_on_view(tmp_path):
    """C1/R1: view.build_table() must raise NotImplementedError."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["u0", "u1"]})
    exp._data = _make_datasets(2)
    exp._path = tmp_path

    view = exp._subset(slice(0, 1), as_view=True)
    with pytest.raises(NotImplementedError, match="build_table"):
        view.build_table()


def test_r2_concat_move_rejects_view(tmp_path):
    """C2/R2: concat([view], mode='move') must raise early before any file ops."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["u0", "u1"]})
    exp._data = _make_datasets(2)
    for i, xd in enumerate(exp._data):
        xd._path = tmp_path / f"data-{i:03d}"
        xd._path.mkdir()
    exp._path = tmp_path

    view = exp._subset(slice(0, 1), as_view=True)

    dest = tmp_path / "concat_out"
    with pytest.raises(ValueError, match="InSituExperimentView"):
        InSituExperiment.concat([view], mode="move", path=dest)

    # Parent paths must be untouched.
    assert (tmp_path / "data-000").exists(), "parent data dir must not be moved"


def test_r4_reload_on_view_does_not_re_read_metadata(tmp_path):
    """M1/R4: view.reload() must not overwrite view metadata with full-experiment data."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["u0", "u1", "u2"], "group": ["A", "B", "C"]})
    exp._data = _make_datasets(3)

    view = exp._subset(slice(0, 2), as_view=True)
    original_len = len(view._metadata)

    # Reload should be a no-op for in-memory datasets with no path; the key
    # invariant is that _metadata is not replaced with the parent's full table.
    view.reload()

    assert len(view._metadata) == original_len, "metadata length must not change after reload"


def test_r5_add_raises_on_view():
    """M2/R5: view.add() must raise NotImplementedError."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["u0"]})
    exp._data = _make_datasets(1)
    view = exp._subset(slice(0, 1), as_view=True)

    with pytest.raises(NotImplementedError, match="parent experiment"):
        view.add(InSituData())


def test_r7_copy_returns_independent_experiment():
    """M4/R7: view.copy() must return a standalone InSituExperiment (not a view)."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["u0", "u1", "u2"]})
    exp._data = _make_datasets(3)
    exp._colors = {"A": "#FF0000"}

    view = exp._subset(slice(0, 2), as_view=True)
    copied = view.copy()

    assert isinstance(copied, InSituExperiment)
    assert not isinstance(copied, InSituExperimentView), "copy must not be a view"
    assert copied._parent_indices is None
    assert copied._path is None
    assert len(copied._data) == 2

    # Mutating the copy must not touch the parent.
    original_uid = exp._data[0]._uid
    copied._data[0]._uid = "MUTATED"
    assert exp._data[0]._uid == original_uid, "parent dataset must be unaffected by copy mutation"


def test_r8_r11_view_saveas_restores_parent_paths(tmp_path):
    """M5+M6 / R8+R11: view.saveas() must not leave parent child _paths stale."""
    import numpy as np
    from anndata import AnnData
    from insitupy.containers.cell_data import CellData

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

    exp = InSituExperiment()
    for i in range(3):
        exp._data.append(_make_xd(i))
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(3)]})

    parent_dir = tmp_path / "parent"
    exp.saveas(parent_dir)

    original_paths = [xd._path for xd in exp._data]

    view = exp._subset(slice(0, 2), as_view=True)
    export_dir = tmp_path / "export"
    view.saveas(export_dir)

    # Parent child paths must be restored to their original locations.
    for i, (xd, orig) in enumerate(zip(exp._data, original_paths)):
        assert xd._path == orig, f"parent _data[{i}]._path was mutated by view.saveas()"


def test_r8_saveas_relative_path_then_save(tmp_path, monkeypatch):
    """M5/R8: saveas() with a *relative* path must leave exp.save() working.

    Regression: saveas() resolved each child ``_path`` via ``.resolve()`` but
    left ``self._path`` unresolved, so the save() parent-path guard compared an
    absolute child parent against a relative experiment path and raised
    ``ValueError`` ("save path ... does not lie inside ..."). Absolute-path
    tests do not catch this because ``Path(abs).resolve() == Path(abs)``.
    """
    import numpy as np
    from anndata import AnnData
    from insitupy.containers.cell_data import CellData

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

    exp = InSituExperiment()
    for i in range(2):
        exp._data.append(_make_xd(i))
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(2)]})

    # Use a RELATIVE path: chdir into tmp_path so "rel_exp" resolves there.
    monkeypatch.chdir(tmp_path)
    exp.saveas("rel_exp")

    # save() must not raise the parent-path guard error after a relative saveas.
    exp.save()


# ── persistence-minor-fixes m2 ────────────────────────────────────────────────


def test_m2_save_filters_merges_by_uid_after_parent_reorder(tmp_path):
    """view.save_filters() must write each mask to the row whose uid matches,
    even when the parent experiment was reordered on disk after the view was created."""
    n = 10
    exp = _make_metadata_experiment(tmp_path, n)

    # Create a view of the first 5 rows while the parent is still in original order.
    view = exp._subset(slice(0, 5), as_view=True)
    assert view._parent_indices == list(range(5))

    # Set a discriminating view mask (alternating T/F for uid00..uid04).
    view._filters["qc"]["mask"] = [True, False, True, False, True]

    # Reorder the parent ON DISK (reversed): uid09..uid00.
    order = list(range(n))[::-1]
    reordered = InSituExperiment()
    reordered._metadata = exp._metadata.iloc[order].reset_index(drop=True)
    reordered._data = [exp._data[i] for i in order]
    reordered._filters = {
        "qc": {"mask": [exp._filters["qc"]["mask"][i] for i in order], "note": None}
    }
    reordered._colors = exp._colors
    reordered._path = tmp_path
    reordered.save_metadata(path=tmp_path)
    reordered.save_filters(path=tmp_path)

    # Save the view filters — must land on correct uids despite stale _parent_indices.
    view.save_filters()

    # Load the on-disk state after save.
    import json
    on_disk_meta = pd.read_parquet(tmp_path / "metadata.parquet")
    with open(tmp_path / "filters.json") as f:
        payload = json.load(f)

    mask = payload["filters"]["qc"]["mask"]
    assert len(mask) == n

    # Build uid -> on-disk mask position map from the (reversed) on-disk metadata.
    uid_to_pos = {row["uid"]: i for i, row in on_disk_meta.iterrows()}

    # View rows (uid00..uid04) should carry the alternating pattern.
    assert mask[uid_to_pos["uid00"]] is True
    assert mask[uid_to_pos["uid01"]] is False
    assert mask[uid_to_pos["uid02"]] is True
    assert mask[uid_to_pos["uid03"]] is False
    assert mask[uid_to_pos["uid04"]] is True

    # Non-view rows (uid05..uid09) must keep the on-disk values from `reordered`.
    for uid in [f"uid{i:02d}" for i in range(5, n)]:
        expected = reordered._filters["qc"]["mask"][
            reordered._metadata.index[reordered._metadata["uid"] == uid].tolist()[0]
        ]
        assert mask[uid_to_pos[uid]] == expected, f"{uid} non-view value changed"


def test_m2_save_filters_legacy_metadata_without_uid_raises(tmp_path):
    """view.save_filters() must raise ValueError when on-disk metadata has no uid column."""
    n = 5
    exp = _make_metadata_experiment(tmp_path, n)
    view = exp._subset(slice(0, 3), as_view=True)

    # Overwrite the on-disk metadata with a uid-less parquet.
    uid_less = exp._metadata.drop(columns=["uid"])
    uid_less.to_parquet(tmp_path / "metadata.parquet", index=False)

    with pytest.raises(ValueError, match="uid"):
        view.save_filters()


def test_view_saveas_rejects_free_after_save(tmp_path):
    """view.saveas(free_after_save=True) must raise: it would empty the shared parent data.

    A view's datasets are the parent's objects, so the base saveas' end-of-save
    _release_data() would wipe the parent's in-memory modalities. The view
    override rejects the flag up front (before any materialise/write).
    """
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["u0", "u1"]})
    exp._data = _make_datasets(2)
    view = exp._subset(slice(0, 1), as_view=True)

    with pytest.raises(ValueError, match="free_after_save"):
        view.saveas(tmp_path / "export", free_after_save=True)


# ── __getitem__ / query() always return a linked view ─────────────────────────


def test_getitem_returns_view():
    """exp[:3] must return an InSituExperimentView sharing datasets with the parent."""
    n = 5
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(n)]})
    exp._data = _make_datasets(n)

    subset = exp[:3]

    assert isinstance(subset, InSituExperimentView)
    assert len(subset._data) == 3
    assert subset._data[0] is exp._data[0], "view must share dataset references, not copy them"


def test_getitem_on_view_returns_another_view():
    """Slicing an existing view must still return a view (propagates, doesn't collapse)."""
    n = 5
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(n)]})
    exp._data = _make_datasets(n)

    view = exp[:4]
    view_of_view = view[:2]

    assert isinstance(view_of_view, InSituExperimentView)
    assert len(view_of_view._data) == 2


def test_query_returns_view():
    """query() delegates to __getitem__, so it must also return an InSituExperimentView."""
    n = 4
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({
        "uid": [f"uid{i}" for i in range(n)],
        "group": ["A", "B", "A", "B"],
    })
    exp._data = _make_datasets(n)

    subset = exp.query({"group": ["A"]})

    assert isinstance(subset, InSituExperimentView)
    assert len(subset._data) == 2


# ── filters.apply() returns a fully independent copy ───────────────────────────


def test_apply_deep_copies_datasets():
    """filters.apply() must deep-copy datasets so the result is independent of the parent."""
    n = 4
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(n)]})
    exp._data = _make_datasets(n)
    exp._filters = {"qc": {"mask": [True, True, False, False], "note": None}}

    applied = exp.filters.apply("qc")

    assert isinstance(applied, InSituExperiment)
    assert not isinstance(applied, InSituExperimentView)
    assert len(applied._data) == 2
    assert applied._data[0] is not exp._data[0], "apply() must deep-copy datasets, not alias them"

    original_uid = exp._data[0]._uid
    applied._data[0]._uid = "MUTATED"
    assert exp._data[0]._uid == original_uid, "parent dataset must be unaffected by apply() mutation"

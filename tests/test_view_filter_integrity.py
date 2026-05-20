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
    exp._colors = {"A": "#FF0000", "B": "#0000FF"}
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
    exp._colors = {"A": "#FF0000", "B": "#0000FF"}
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
    assert "A" in reloaded._colors


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
    assert "B" in exp._colors

    view = exp._subset(slice(0, 5), as_view=True)
    # Simulate view missing "B" (e.g. sync_colors only saw "A" datasets).
    view._colors = {"A": "#FF0000"}
    view.save_colors()

    with open(tmp_path / "colors.json") as f:
        colors = json.load(f)

    assert "B" in colors, "parent's 'B' color key must survive view.save_colors()"
    assert "A" in colors


# ── M3 ────────────────────────────────────────────────────────────────────────


def test_m3_add_extends_filter_masks():
    """add() must keep filter masks in sync with metadata after adding a dataset."""
    n = 5
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": [f"uid{i}" for i in range(n)]})
    exp._data = _make_datasets(n)
    exp._filters = {"qc": {"mask": [True] * n, "note": None}}
    # No exp._path so save() is not triggered.

    view = exp._subset(slice(0, 3), as_view=True)
    initial_meta_len = len(view._metadata)

    view.add(InSituData())

    assert len(view._metadata) == initial_meta_len + 1
    assert len(view._filters["qc"]["mask"]) == len(view._metadata), (
        "filter mask must stay in sync with metadata after add()"
    )
    assert not view._filters["qc"]["mask"][-1], "new entry must default to False"


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

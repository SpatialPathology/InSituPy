import pytest
import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype

from insitupy.experiment.data import InSituExperiment, InSituExperimentView


def test_metadata_parquet_round_trip(tmp_path):
    """Saving and reloading via Parquet preserves string IDs and nullable dtypes."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "patient_id": pd.Series(["P0001", "P0456"], dtype="string"),
            "n_cells": pd.Series([10, 20], dtype="Int64"),
        }
    )

    exp.save_metadata(path=tmp_path)

    # Parquet is written; schema sidecar is not.
    assert (tmp_path / "metadata.parquet").exists()
    assert not (tmp_path / "metadata.schema.json").exists()

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    assert out["patient_id"].tolist() == ["P0001", "P0456"]
    assert is_string_dtype(out["patient_id"])
    assert is_integer_dtype(out["n_cells"])


def test_metadata_csv_only_remains_loadable(tmp_path):
    metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "n_cells": [10, 20],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv")

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    # Legacy CSV-only folders are still accepted and loaded via pandas inference.
    assert is_integer_dtype(out["n_cells"])


def test_legacy_metadata_slide_sample_id_discarded(tmp_path):
    """Legacy metadata.csv with slide_id/sample_id loads without error; those columns are dropped."""
    metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "slide_id": ["0000001", "0000456"],
            "sample_id": ["s1", "s2"],
            "n_cells": [10, 20],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv")

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    assert "slide_id" not in out.columns
    assert "sample_id" not in out.columns
    assert "n_cells" in out.columns
    assert "uid" in out.columns


# ── InSituExperimentView.save_metadata ────────────────────────────────────────

def _make_experiment_on_disk(tmp_path):
    """Helper: create a 3-dataset experiment on disk and return it."""
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame(
        {
            "uid": ["aaa", "bbb", "ccc"],
            "group": ["A", "B", "C"],
            "n_cells": pd.Series([10, 20, 30], dtype="Int64"),
        }
    )
    exp._path = tmp_path
    exp.save_metadata(path=tmp_path)
    return exp


def _make_view(exp, mask):
    """Helper: build an InSituExperimentView from a boolean mask."""
    view = InSituExperimentView()
    view._metadata = exp._metadata[mask].reset_index(drop=True)
    view._path = exp._path
    view._colors = {}
    view._filters = {}
    view._composites = {}
    view._applied_filters = []
    view._data = []
    return view


def test_view_save_metadata_preserves_non_view_rows(tmp_path):
    """Saving a view must not drop datasets that are not part of the view."""
    exp = _make_experiment_on_disk(tmp_path)
    mask = pd.Series([True, False, True])
    view = _make_view(exp, mask)

    # Modify something in the view.
    view._metadata["group"] = ["X", "X"]
    view.save_metadata()

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    assert len(out) == 3, "All three rows must be present after view save"
    assert out.set_index("uid").loc["aaa", "group"] == "X"
    assert out.set_index("uid").loc["bbb", "group"] == "B"  # untouched
    assert out.set_index("uid").loc["ccc", "group"] == "X"


def test_view_save_metadata_new_column(tmp_path):
    """A column added via add_metadata_column on a view fills pd.NA for non-view rows."""
    exp = _make_experiment_on_disk(tmp_path)
    mask = pd.Series([True, False, True])
    view = _make_view(exp, mask)

    view.add_metadata_column("cluster", ["C1", "C2"])
    view.save_metadata()

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata.set_index("uid")

    assert out.loc["aaa", "cluster"] == "C1"
    assert out.loc["ccc", "cluster"] == "C2"
    assert pd.isna(out.loc["bbb", "cluster"])


def test_view_save_metadata_no_path_raises():
    """A view without a parent path must raise ValueError."""
    view = InSituExperimentView()
    view._path = None
    with pytest.raises(ValueError, match="parent experiment path is not set"):
        view.save_metadata()


def test_view_save_metadata_no_uid_on_disk_raises(tmp_path):
    """If on-disk metadata has no uid column, save_metadata must raise."""
    # Write legacy metadata without uid.
    pd.DataFrame({"group": ["A", "B"]}).to_parquet(tmp_path / "metadata.parquet", index=False)

    view = InSituExperimentView()
    view._path = tmp_path
    view._metadata = pd.DataFrame({"uid": ["aaa"], "group": ["X"]})
    with pytest.raises(ValueError, match="no 'uid' column"):
        view.save_metadata()

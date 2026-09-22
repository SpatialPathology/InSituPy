"""Tests for the InSituExperiment.metadata read-only guard (API-3).

Exercises real read paths (column access, .loc, iteration, .value_counts(),
passing to a function) so a read that incorrectly trips the pandas-subclass
write guard is caught, and pins that writes through the public getter now
raise InSituPyError instead of silently no-op'ing.
"""

import logging

import pandas as pd
import pytest

from insitupy._exceptions import InSituPyError
from insitupy.experiment.data import InSituExperiment


def _make_experiment():
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2", "sample-3"],
            "group": ["A", "B", "A"],
            "n_cells": [10, 20, 30],
        }
    )
    return exp


def test_metadata_read_does_not_warn(caplog):
    """Reading exp.metadata (and common access patterns) emits no log record."""
    exp = _make_experiment()
    with caplog.at_level(logging.WARNING, logger="insitupy.experiment.data"):
        _ = exp.metadata
        _ = exp.metadata.shape
        _ = exp.metadata["group"]
        _ = exp.metadata.loc[0]
    assert caplog.records == []


def test_metadata_setitem_raises():
    exp = _make_experiment()
    with pytest.raises(InSituPyError, match="add_metadata_column"):
        exp.metadata["new_col"] = 1


def test_metadata_setattr_raises():
    exp = _make_experiment()
    with pytest.raises(InSituPyError, match="add_metadata_column"):
        exp.metadata.newattr = 1


def test_metadata_reads_work():
    """Pandas-subclass smoke test at real altitude: ordinary reads succeed and
    return correct values without tripping the write guard."""
    exp = _make_experiment()
    md = exp.metadata

    assert md.shape == (3, 3)
    assert md["group"].tolist() == ["A", "B", "A"]
    assert md.loc[0, "uid"] == "sample-1"

    rows = list(md.iterrows())
    assert len(rows) == 3
    assert rows[1][1]["uid"] == "sample-2"

    counts = md["group"].value_counts()
    assert counts["A"] == 2
    assert counts["B"] == 1

    def _first_uid(df):
        # Passing the guarded frame to an ordinary function must behave like a
        # normal DataFrame read.
        return df["uid"].iloc[0]

    assert _first_uid(md) == "sample-1"


def test_metadata_copy_returns_plain_editable_frame():
    exp = _make_experiment()
    md_copy = exp.metadata.copy()

    assert type(md_copy) is pd.DataFrame
    # Editable: assignment must not raise.
    md_copy["new_col"] = 1
    assert "new_col" in md_copy.columns
    # The internal metadata is untouched by mutating the copy.
    assert "new_col" not in exp._metadata.columns


def test_validated_mutators_still_work():
    exp = _make_experiment()
    exp.add_metadata_column("new_col", [1, 2, 3])
    assert exp.metadata["new_col"].tolist() == [1, 2, 3]

    exp.set_metadata_values(0, "group", "Z")
    assert exp.metadata.loc[0, "group"] == "Z"

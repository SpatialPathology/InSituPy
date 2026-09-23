"""Tests for composite filter functionality in FilterManager."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from insitupy.experiment.filters import CompositeFilterSpec, FilterManager


def make_experiment():
    exp = MagicMock()
    exp._filters = {}
    exp._composites = {}
    exp._metadata = pd.DataFrame({
        "uid":    ["a", "b", "c", "d", "e"],
        "region": ["R1", "R1", "R2", "R2", "R2"],
        "qc":     ["pass", "fail", "pass", "pass", "fail"],
    })
    return exp


def test_base_create():
    fm = FilterManager(make_experiment())
    msg = fm.create(by="qc", include="pass", key="qc_pass")
    assert "3/5" in msg
    assert fm.base_keys() == ["qc_pass"]
    assert fm.composite_keys() == []


def test_combine_and():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    msg = fm.combine(["qc_pass", "region_r2"], "and", key="combined")
    assert "composite" in msg.lower()
    mask = fm._resolve_mask("combined")
    # pass AND R2: c(T,T), d(T,T) → [F,F,T,T,F]
    assert mask.tolist() == [False, False, True, True, False]


def test_combine_or_with_negate():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "or", key="qc_or_not_r2", negate=["region_r2"])
    mask = fm._resolve_mask("qc_or_not_r2")
    # qc_pass=[T,F,T,T,F], NOT region_r2=[T,T,F,F,F]  → OR → [T,T,T,T,F]
    assert mask.tolist() == [True, True, True, True, False]


def test_summary_columns():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="combined")
    df = fm.summary()
    assert set(df.columns) >= {"filter_key", "type", "formula", "n_selected"}
    assert set(df["type"].tolist()) == {"base", "composite"}
    row = df[df["filter_key"] == "combined"].iloc[0]
    assert row["formula"] == "qc_pass AND region_r2"


def test_invert():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.invert("qc_pass", "qc_fail")
    mask = fm._resolve_mask("qc_fail")
    assert mask.tolist() == [False, True, False, False, True]
    assert "qc_fail" in fm.base_keys()


def test_materialize():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="combined")
    fm.materialize("combined", new_key="combined_frozen")
    assert "combined_frozen" in fm.base_keys()
    assert "combined" in fm.composite_keys()
    assert fm._resolve_mask("combined_frozen").tolist() == [False, False, True, True, False]


def test_remove_blocks_if_referenced():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="combined")
    try:
        fm.remove("qc_pass")
        assert False, "Should have raised"
    except ValueError as e:
        assert "combined" in str(e)
    fm.remove("combined")
    fm.remove("qc_pass")  # now allowed


def test_lazy_reevaluation():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="combined")
    before = fm._resolve_mask("combined").tolist()
    # Mutate the base filter directly and check composite re-evaluates
    from insitupy.experiment.filters import FilterSpec
    exp._filters["qc_pass"] = FilterSpec(key="qc_pass", mask=[True, True, True, True, True]).to_dict()
    after = fm._resolve_mask("combined").tolist()
    assert before != after
    assert after == [False, False, True, True, True]


# ── R-A2: overwrite=True must not leave a key in both stores ──────────────────


def test_create_overwrite_evicts_composite():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="x")

    fm.create(by="qc", include="fail", key="x", overwrite=True)

    assert "x" in fm.base_keys()
    assert "x" not in fm.composite_keys()
    assert fm.keys().count("x") == 1


def test_invert_overwrite_evicts_composite():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="x")

    fm.invert("qc_pass", "x", overwrite=True)

    assert "x" in fm.base_keys()
    assert "x" not in fm.composite_keys()
    assert fm.keys().count("x") == 1


def test_materialize_new_key_overwrite_evicts_composite():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="comp1")
    fm.combine(["qc_pass"], "or", key="x")  # second composite, used as the overwrite target

    fm.materialize("comp1", new_key="x", overwrite=True)

    assert "x" in fm.base_keys()
    assert "x" not in fm.composite_keys()
    assert fm.keys().count("x") == 1
    assert "comp1" in fm.composite_keys(), "the materialized source composite must be untouched"


def test_rename_base_onto_composite_evicts():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="x")
    fm.create(by="qc", include="fail", key="y")

    fm.rename("y", "x", overwrite=True)

    assert "x" in fm.base_keys()
    assert "x" not in fm.composite_keys()
    assert fm.keys().count("x") == 1


def test_rename_composite_onto_base_evicts():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="c2")
    fm.create(by="qc", include="fail", key="x")  # unreferenced base, safe to overwrite

    fm.rename("c2", "x", overwrite=True)

    assert "x" in fm.composite_keys()
    assert "x" not in fm.base_keys()
    assert fm.keys().count("x") == 1


def test_materialize_new_key_equal_key_overwrite():
    """materialize(key, new_key=key, overwrite=True) covers the read-before-evict ordering."""
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="c")

    fm.materialize("c", new_key="c", overwrite=True)

    assert "c" in fm.base_keys()
    assert "c" not in fm.composite_keys()
    assert fm.keys().count("c") == 1


def test_combine_overwrite_referenced_base_raises():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="x")
    fm.create(by="region", include="R2", key="y")
    fm.combine(["x", "y"], "and", key="c")
    filters_before = dict(exp._filters)
    composites_before = dict(exp._composites)

    with pytest.raises(ValueError, match="x"):
        fm.combine(["y"], "and", key="x", overwrite=True)

    assert exp._filters == filters_before
    assert exp._composites == composites_before


def test_rename_overwrite_referenced_base_raises():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="x")
    fm.create(by="region", include="R2", key="y")
    fm.combine(["x", "y"], "and", key="c")
    fm.combine(["y"], "or", key="c2")  # unrelated composite, used as the renamed-from key
    filters_before = dict(exp._filters)
    composites_before = dict(exp._composites)

    with pytest.raises(ValueError, match="x"):
        fm.rename("c2", "x", overwrite=True)

    assert exp._filters == filters_before
    assert exp._composites == composites_before


def test_combine_self_reference_raises():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="x")
    fm.create(by="region", include="R2", key="y")

    with pytest.raises(ValueError, match="x"):
        fm.combine(["x", "y"], "and", key="x", overwrite=True)

    assert "x" in fm.base_keys()
    assert fm.composite_keys() == []


def test_create_failed_validation_keeps_existing():
    """A writer that fails validation after _check_key_free must not have evicted first."""
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "and", key="x")

    with pytest.raises(KeyError):
        fm.create(by="missing_col", include="whatever", key="x", overwrite=True)

    assert "x" in fm.composite_keys()
    assert "x" not in fm.base_keys()

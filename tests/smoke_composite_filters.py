"""Smoke test for composite filter functionality in FilterManager."""
import pandas as pd
from unittest.mock import MagicMock

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
    print("PASS: base create")


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
    print("PASS: combine AND")


def test_combine_or_with_negate():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.create(by="region", include="R2", key="region_r2")
    fm.combine(["qc_pass", "region_r2"], "or", key="qc_or_not_r2", negate=["region_r2"])
    mask = fm._resolve_mask("qc_or_not_r2")
    # qc_pass=[T,F,T,T,F], NOT region_r2=[T,T,F,F,F]  → OR → [T,T,T,T,F]
    assert mask.tolist() == [True, True, True, True, False]
    print("PASS: combine OR with negate")


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
    print("PASS: summary columns and formula")


def test_invert():
    exp = make_experiment()
    fm = FilterManager(exp)
    fm.create(by="qc", include="pass", key="qc_pass")
    fm.invert("qc_pass", "qc_fail")
    mask = fm._resolve_mask("qc_fail")
    assert mask.tolist() == [False, True, False, False, True]
    assert "qc_fail" in fm.base_keys()
    print("PASS: invert")


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
    print("PASS: materialize")


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
    print("PASS: remove blocks if referenced")


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
    print("PASS: lazy re-evaluation")


if __name__ == "__main__":
    test_base_create()
    test_combine_and()
    test_combine_or_with_negate()
    test_summary_columns()
    test_invert()
    test_materialize()
    test_remove_blocks_if_referenced()
    test_lazy_reevaluation()
    print("\nAll smoke tests passed.")

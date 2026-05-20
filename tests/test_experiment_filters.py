"""Tests for insitupy.experiment.filters — FilterSpec and CompositeFilterSpec."""

import numpy as np
import pytest

from insitupy.experiment.filters import CompositeFilterSpec, FilterSpec


class TestFilterSpecToDict:
    def test_returns_dict_with_mask_and_note(self):
        # to_dict() must include both 'mask' and 'note' keys
        spec = FilterSpec(key="quality", mask=[True, False, True], note="QC filter")
        d = spec.to_dict()
        assert "mask" in d
        assert "note" in d

    def test_mask_is_list_of_booleans(self):
        # mask must be a plain Python list (not numpy array) for JSON compatibility
        spec = FilterSpec(key="q", mask=[True, False, True])
        d = spec.to_dict()
        assert isinstance(d["mask"], list)
        assert all(isinstance(v, (bool, np.bool_)) for v in d["mask"])

    def test_mask_values_are_correct(self):
        # The stored mask must exactly match the input values
        spec = FilterSpec(key="q", mask=[True, False, False, True])
        assert spec.to_dict()["mask"] == [True, False, False, True]

    def test_note_none_when_not_provided(self):
        # Default note is None and must be serialized as None
        spec = FilterSpec(key="q", mask=[True, True])
        assert spec.to_dict()["note"] is None

    def test_note_preserved(self):
        spec = FilterSpec(key="q", mask=[True], note="exclusion reason")
        assert spec.to_dict()["note"] == "exclusion reason"

    def test_numpy_boolean_array_converted_correctly(self):
        # Numpy boolean arrays must be serialized as plain Python booleans
        mask = np.array([True, False, True])
        spec = FilterSpec(key="q", mask=mask.tolist())
        d = spec.to_dict()
        assert d["mask"] == [True, False, True]


class TestFilterSpecFromEntry:
    def test_roundtrip_preserves_mask_and_note(self):
        # Serialise and deserialise: result must match original
        original = FilterSpec(key="sample_filter", mask=[True, False, True, True], note="manual")
        d = original.to_dict()
        restored = FilterSpec.from_entry(key="sample_filter", entry=d)
        assert restored.key == "sample_filter"
        assert restored.mask == original.mask
        assert restored.note == original.note

    def test_from_entry_without_note_defaults_to_none(self):
        # If entry has no 'note', the restored spec must have note=None
        entry = {"mask": [True, False]}
        spec = FilterSpec.from_entry("k", entry)
        assert spec.note is None

    def test_from_entry_raises_for_non_dict(self):
        # Passing a list instead of a dict must raise ValueError
        with pytest.raises(ValueError, match="dictionary"):
            FilterSpec.from_entry("k", [True, False])

    def test_from_entry_raises_for_missing_mask(self):
        # A dict without 'mask' must raise ValueError
        with pytest.raises(ValueError, match="mask"):
            FilterSpec.from_entry("k", {"note": "test"})

    def test_from_entry_with_integer_mask_coerces_to_bool(self):
        # Integer 0/1 mask values must be coerced to bool
        entry = {"mask": [1, 0, 1]}
        spec = FilterSpec.from_entry("k", entry)
        assert spec.mask == [True, False, True]

    def test_from_entry_note_as_non_string_coerces(self):
        # Non-string note values must be coerced to string without raising
        entry = {"mask": [True], "note": 42}
        spec = FilterSpec.from_entry("k", entry)
        assert isinstance(spec.note, str)

    def test_from_entry_key_stored_in_spec(self):
        # The key argument must be stored in the returned FilterSpec
        spec = FilterSpec.from_entry("my_key", {"mask": [True, False]})
        assert spec.key == "my_key"


class TestFilterSpecRoundtrip:
    def test_empty_mask_roundtrip(self):
        # Edge case: empty mask must survive serialization round-trip
        original = FilterSpec(key="empty", mask=[])
        restored = FilterSpec.from_entry("empty", original.to_dict())
        assert restored.mask == []

    def test_all_true_mask_roundtrip(self):
        original = FilterSpec(key="all_true", mask=[True, True, True])
        restored = FilterSpec.from_entry("all_true", original.to_dict())
        assert restored.mask == [True, True, True]

    def test_all_false_mask_roundtrip(self):
        original = FilterSpec(key="all_false", mask=[False, False])
        restored = FilterSpec.from_entry("all_false", original.to_dict())
        assert restored.mask == [False, False]


# ── CompositeFilterSpec tests ──────────────────────────────────────────────────

class TestCompositeFilterSpecEvaluate:
    def test_and_operation(self):
        comp = CompositeFilterSpec(
            key="c", operation="and",
            source_keys=["a", "b"], negated=[],
        )
        masks = {"a": [True, True, False, False], "b": [True, False, True, False]}
        assert comp.evaluate(masks) == [True, False, False, False]

    def test_or_operation(self):
        comp = CompositeFilterSpec(
            key="c", operation="or",
            source_keys=["a", "b"], negated=[],
        )
        masks = {"a": [True, True, False, False], "b": [True, False, True, False]}
        assert comp.evaluate(masks) == [True, True, True, False]

    def test_negate_single_key(self):
        comp = CompositeFilterSpec(
            key="c", operation="and",
            source_keys=["a"], negated=["a"],
        )
        masks = {"a": [True, False, True]}
        assert comp.evaluate(masks) == [False, True, False]

    def test_negate_one_of_two_and(self):
        comp = CompositeFilterSpec(
            key="c", operation="and",
            source_keys=["a", "b"], negated=["b"],
        )
        masks = {"a": [True, True, False], "b": [True, False, False]}
        # a AND NOT b  →  [T AND F, T AND T, F AND T]  →  [F, T, F]
        assert comp.evaluate(masks) == [False, True, False]

    def test_three_keys_and(self):
        comp = CompositeFilterSpec(
            key="c", operation="and",
            source_keys=["a", "b", "c2"], negated=[],
        )
        masks = {
            "a":  [True, True, True, False],
            "b":  [True, True, False, True],
            "c2": [True, False, True, True],
        }
        assert comp.evaluate(masks) == [True, False, False, False]


class TestCompositeFilterSpecFormulaStr:
    def test_and_no_negate(self):
        comp = CompositeFilterSpec(key="c", operation="and", source_keys=["a", "b"], negated=[])
        assert comp.formula_str() == "a AND b"

    def test_or_with_negate(self):
        comp = CompositeFilterSpec(key="c", operation="or", source_keys=["a", "b"], negated=["b"])
        assert comp.formula_str() == "a OR NOT b"

    def test_single_negated(self):
        comp = CompositeFilterSpec(key="c", operation="and", source_keys=["a"], negated=["a"])
        assert comp.formula_str() == "NOT a"


class TestCompositeFilterSpecSerialisation:
    def test_to_dict_keys(self):
        comp = CompositeFilterSpec(key="c", operation="and", source_keys=["a", "b"], negated=["b"], note="test")
        d = comp.to_dict()
        assert set(d.keys()) == {"operation", "source_keys", "negated", "note"}

    def test_roundtrip(self):
        original = CompositeFilterSpec(
            key="c", operation="or", source_keys=["x", "y"], negated=["y"], note="combined",
        )
        restored = CompositeFilterSpec.from_entry("c", original.to_dict())
        assert restored.key == "c"
        assert restored.operation == "or"
        assert restored.source_keys == ["x", "y"]
        assert restored.negated == ["y"]
        assert restored.note == "combined"

    def test_from_entry_defaults_negated_to_empty(self):
        entry = {"operation": "and", "source_keys": ["a", "b"]}
        comp = CompositeFilterSpec.from_entry("c", entry)
        assert comp.negated == []

    def test_from_entry_raises_for_non_dict(self):
        with pytest.raises(ValueError, match="dictionary"):
            CompositeFilterSpec.from_entry("c", ["and", "a"])

    def test_from_entry_raises_for_missing_operation(self):
        with pytest.raises(ValueError, match="operation"):
            CompositeFilterSpec.from_entry("c", {"source_keys": ["a"]})

    def test_from_entry_raises_for_missing_source_keys(self):
        with pytest.raises(ValueError, match="source_keys"):
            CompositeFilterSpec.from_entry("c", {"operation": "and"})

    def test_from_entry_raises_for_invalid_operation(self):
        with pytest.raises(ValueError, match="'xor'"):
            CompositeFilterSpec.from_entry("c", {"operation": "xor", "source_keys": ["a"]})

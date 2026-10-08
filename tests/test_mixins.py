"""Tests for insitupy._mixins — DeepCopyMixin, GetMixin, _UpdatablePlottingConfig."""

import dataclasses

import pytest

from insitupy._mixins import DeepCopyMixin, GetMixin, _UpdatablePlottingConfig


class _SimpleObj(DeepCopyMixin):
    def __init__(self, value):
        self.value = value
        self.nested = {"key": [1, 2, 3]}


class _GetObj(GetMixin):
    def __init__(self):
        self.x = 42
        self.name = "test"


@dataclasses.dataclass
class _Config(_UpdatablePlottingConfig):
    alpha: float = 0.5
    color: str = "red"
    size: int = 10


class TestDeepCopyMixin:
    def test_copy_returns_new_object(self):
        # copy() must return a different object identity
        obj = _SimpleObj(1)
        copied = obj.copy()
        assert copied is not obj

    def test_copy_preserves_value(self):
        # The copied object must have the same value as the original
        obj = _SimpleObj(99)
        assert obj.copy().value == 99

    def test_copy_is_deep(self):
        # Mutating nested structure on copy must not affect original
        obj = _SimpleObj(1)
        copied = obj.copy()
        copied.nested["key"].append(999)
        assert 999 not in obj.nested["key"]


class TestGetMixin:
    def test_get_returns_attribute(self):
        # get(key) must return the same value as direct attribute access
        obj = _GetObj()
        assert obj.get("x") == 42
        assert obj.get("name") == "test"

    def test_getitem_returns_attribute(self):
        # obj["key"] must work identically to obj.get("key")
        obj = _GetObj()
        assert obj["x"] == 42
        assert obj["name"] == "test"

    def test_get_missing_key_raises_attribute_error(self):
        # Accessing a non-existent key should raise AttributeError, not KeyError
        obj = _GetObj()
        with pytest.raises(AttributeError):
            obj.get("does_not_exist")


class TestUpdatablePlottingConfig:
    def test_update_values_changes_attribute(self):
        # update_values must actually change the attribute value
        cfg = _Config()
        cfg.update_values(alpha=0.9)
        assert cfg.alpha == 0.9

    def test_update_multiple_values(self):
        # Multiple keyword arguments must all be applied
        cfg = _Config()
        cfg.update_values(color="blue", size=20)
        assert cfg.color == "blue"
        assert cfg.size == 20

    def test_update_invalid_key_raises_attribute_error(self):
        # Unknown attribute must raise AttributeError to catch typos
        cfg = _Config()
        with pytest.raises(AttributeError):
            cfg.update_values(nonexistent_key=1)

    def test_show_all_does_not_raise(self):
        # show_all() logs output; just verify it runs without error
        cfg = _Config()
        cfg.show_all()  # should not raise

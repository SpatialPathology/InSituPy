"""Tests for insitupy._checks — try_import utility."""

import pytest

from insitupy._checks import try_import
from insitupy._exceptions import MissingPackageError


class TestTryImport:
    def test_imports_existing_package(self):
        # A package that is definitely installed should be returned as a module
        mod = try_import("os")
        import os
        assert mod is os

    def test_raises_missing_package_error_for_nonexistent(self):
        # Missing packages must raise MissingPackageError, not plain ImportError,
        # so callers can distinguish "not installed" from other import failures
        with pytest.raises(MissingPackageError):
            try_import("_this_package_does_not_exist_xyz")

    def test_missing_package_error_contains_package_name(self):
        # The error message should name the package so users know what to install
        with pytest.raises(MissingPackageError) as exc_info:
            try_import("_nonexistent_package_abc")
        assert "_nonexistent_package_abc" in str(exc_info.value)

    def test_custom_installation_command_in_error(self):
        # Custom install commands must propagate to the error message
        with pytest.raises(MissingPackageError) as exc_info:
            try_import(
                "_fake_pkg",
                installation_command="pip install fake-pkg-extra"
            )
        assert "pip install fake-pkg-extra" in str(exc_info.value)

    def test_returns_module_object(self):
        # Return value must be the module, not a string or None
        mod = try_import("sys")
        import sys
        assert mod is sys

"""Tests for insitupy._exceptions — custom exception classes."""

import pytest

from insitupy._exceptions import (
    InSituDataMissingObject,
    InSituDataRepeatedCropError,
    InvalidDataTypeError,
    InvalidFileTypeError,
    InvalidXeniumDirectory,
    MissingPackageError,
    ModalityNotFoundError,
    ModalityNotFoundWarning,
    ModuleNotFoundOnWindows,
    NotEnoughFeatureMatchesError,
    NotOneElementError,
    UnknownOptionError,
    WrongNapariLayerTypeError,
)


class TestModuleNotFoundOnWindows:
    def test_is_module_not_found_error(self):
        # Must subclass ModuleNotFoundError so callers catching that still work
        exc = ModuleNotFoundOnWindows(ModuleNotFoundError("numpy"))
        assert isinstance(exc, ModuleNotFoundError)

    def test_message_contains_package_name(self):
        # Package name must appear in the message so users know what to install
        inner = ModuleNotFoundError("rpy2")
        inner.name = "rpy2"
        exc = ModuleNotFoundOnWindows(inner)
        assert "rpy2" in str(exc)


class TestInSituDataRepeatedCropError:
    def test_stores_xlim_and_ylim(self):
        # xlim/ylim must be accessible after construction for programmatic inspection
        exc = InSituDataRepeatedCropError(xlim=(10, 20), ylim=(5, 15))
        assert exc.xlim == (10, 20)
        assert exc.ylim == (5, 15)

    def test_message_contains_limits(self):
        # Both limits should appear in the message for useful error output
        exc = InSituDataRepeatedCropError(xlim=(1, 2), ylim=(3, 4))
        assert "1" in exc.message and "2" in exc.message
        assert "3" in exc.message and "4" in exc.message

    def test_is_exception(self):
        # Must be raiseable and catchable as a plain Exception
        with pytest.raises(InSituDataRepeatedCropError):
            raise InSituDataRepeatedCropError(xlim=(0, 1), ylim=(0, 1))


class TestInSituDataMissingObject:
    def test_message_contains_name(self):
        # The missing object name must appear in the message
        exc = InSituDataMissingObject("images")
        assert "images" in exc.message

    def test_message_suggests_read(self):
        # The message must hint at calling read_<name>()
        exc = InSituDataMissingObject("cells")
        assert "read_cells" in exc.message


class TestWrongNapariLayerTypeError:
    def test_message_contains_found_and_wanted(self):
        # Both the found and expected types must be in the message
        exc = WrongNapariLayerTypeError(found="Labels", wanted="Points")
        assert "Labels" in exc.message
        assert "Points" in exc.message


class TestNotOneElementError:
    def test_message_contains_actual_length(self):
        # Must report how many elements were found, not just "wrong count"
        exc = NotOneElementError([1, 2, 3])
        assert "3" in exc.message

    def test_empty_list(self):
        exc = NotOneElementError([])
        assert "0" in exc.message


class TestUnknownOptionError:
    def test_message_contains_option_and_available(self):
        # Both the invalid option and the allowed ones must appear
        exc = UnknownOptionError("foo", ["bar", "baz"])
        assert "foo" in exc.message
        assert "bar" in exc.message
        assert "baz" in exc.message


class TestNotEnoughFeatureMatchesError:
    def test_message_contains_number_and_threshold(self):
        # Both the found count and the threshold must appear
        exc = NotEnoughFeatureMatchesError(number=5, threshold=10)
        assert "5" in exc.message
        assert "10" in exc.message

    def test_partial_result_stored(self):
        # partial_result must be accessible for downstream QC
        dummy = object()
        exc = NotEnoughFeatureMatchesError(number=3, threshold=10, partial_result=dummy)
        assert exc.partial_result is dummy

    def test_partial_result_defaults_to_none(self):
        exc = NotEnoughFeatureMatchesError(number=3, threshold=10)
        assert exc.partial_result is None


class TestModalityNotFoundError:
    def test_message_contains_modality(self):
        exc = ModalityNotFoundError("table")
        assert "table" in exc.message

    def test_is_exception(self):
        with pytest.raises(ModalityNotFoundError):
            raise ModalityNotFoundError("images")


class TestModalityNotFoundWarning:
    def test_is_user_warning(self):
        # Must subclass UserWarning so warnings.warn() works with it
        w = ModalityNotFoundWarning("transcripts")
        assert isinstance(w, UserWarning)

    def test_message_contains_modality(self):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn(ModalityNotFoundWarning("cells"))
        assert len(caught) == 1
        assert "cells" in str(caught[0].message)


class TestInvalidFileTypeError:
    def test_message_contains_allowed_and_received(self):
        # Both allowed and received types must be in the error message
        exc = InvalidFileTypeError(allowed_types=[".zarr", ".h5"], received_type=".csv")
        assert ".csv" in exc.message
        assert ".zarr" in exc.message

    def test_custom_message_overrides_default(self):
        exc = InvalidFileTypeError(
            allowed_types=[".zarr"], received_type=".csv", message="custom msg"
        )
        assert exc.message == "custom msg"

    def test_is_exception(self):
        with pytest.raises(InvalidFileTypeError):
            raise InvalidFileTypeError([".zarr"], ".csv")


class TestInvalidDataTypeError:
    def test_message_contains_allowed_and_received(self):
        exc = InvalidDataTypeError(allowed_types=[int, float], received_type=str)
        assert exc.message  # non-empty message
        assert "str" in exc.message

    def test_is_exception(self):
        with pytest.raises(InvalidDataTypeError):
            raise InvalidDataTypeError([int], str)


class TestInvalidXeniumDirectory:
    def test_message_contains_directory(self, tmp_path):
        # Directory path must appear so users can identify the problematic path
        exc = InvalidXeniumDirectory(tmp_path)
        assert str(tmp_path) in exc.message

    def test_non_xenium_dir_message(self, tmp_path):
        # Regular (non-.ispy) directory should produce "not a valid Xenium directory" message
        exc = InvalidXeniumDirectory(tmp_path)
        assert "not a valid Xenium directory" in exc.message or "experiment.xenium" in exc.message

    def test_ispy_dir_suggests_read(self, tmp_path):
        # A directory with .ispy should suggest InSituData.read() instead
        (tmp_path / ".ispy").mkdir()
        exc = InvalidXeniumDirectory(tmp_path)
        assert "InSituData.read()" in exc.message


class TestMissingPackageError:
    def test_is_import_error(self):
        # Must subclass ImportError so callers catching ImportError still work
        exc = MissingPackageError("mellon", None)
        assert isinstance(exc, ImportError)

    def test_message_contains_package_name(self):
        exc = MissingPackageError("mellon", None)
        assert "mellon" in str(exc)

    def test_default_install_command(self):
        # When no install command given, should default to `pip install <pkg>`
        exc = MissingPackageError("mypackage", None)
        assert "pip install mypackage" in str(exc)

    def test_custom_install_command(self):
        exc = MissingPackageError("mypackage", "conda install mypackage")
        assert "conda install mypackage" in str(exc)

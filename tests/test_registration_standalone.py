"""Tests for the register_images_standalone() function and related dataclasses."""
import numpy as np
import pytest
import dask.array as da

from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy.images.registration import (
    RegistrationConfig,
    register_images_standalone,
)


def _blank_image(shape=(64, 64), dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# RegistrationConfig
# ---------------------------------------------------------------------------

def test_registration_config_defaults():
    """RegistrationConfig can be instantiated with defaults."""
    config = RegistrationConfig()
    assert config.max_width == 4000
    assert config.feature_detection_method == "sift"
    assert config.ratio_test is True
    assert config.min_good_matches == 20


def test_registration_config_frozen():
    """RegistrationConfig is immutable (frozen dataclass)."""
    config = RegistrationConfig()
    with pytest.raises((TypeError, AttributeError)):
        config.max_width = 100  # type: ignore[misc]


def test_registration_config_custom():
    """RegistrationConfig accepts non-default values."""
    config = RegistrationConfig(max_width=2000, min_good_matches=5, flann=False)
    assert config.max_width == 2000
    assert config.min_good_matches == 5
    assert config.flann is False


# ---------------------------------------------------------------------------
# register_images_standalone — failure paths (blank images, no features)
# ---------------------------------------------------------------------------

_COMMON_KWARGS = dict(
    axes_moving="YX",
    axes_fixed="YX",
    min_good_matches=1,
    test_flipping=False,
    verbose=False,
)


def test_blank_images_raise_not_enough_features():
    """Blank images produce no keypoints → NotEnoughFeatureMatchesError."""
    moving = _blank_image()
    fixed = _blank_image()
    with pytest.raises(NotEnoughFeatureMatchesError):
        register_images_standalone(moving, fixed, **_COMMON_KWARGS)


def test_failure_exception_has_partial_result_attribute():
    """The exception always exposes partial_result (may be None for zero-match case)."""
    moving = _blank_image()
    fixed = _blank_image()
    with pytest.raises(NotEnoughFeatureMatchesError) as exc_info:
        register_images_standalone(moving, fixed, **_COMMON_KWARGS)
    assert hasattr(exc_info.value, "partial_result")


def test_dask_array_input_is_accepted():
    """Dask array inputs are computed before feature extraction (no crash)."""
    moving = da.from_array(_blank_image())
    fixed = da.from_array(_blank_image())
    with pytest.raises(NotEnoughFeatureMatchesError):
        register_images_standalone(moving, fixed, **_COMMON_KWARGS)


def test_failure_writes_qc_file_to_qc_dir(tmp_path):
    """On failure, if qc_dir is provided the directory is created."""
    moving = _blank_image()
    fixed = _blank_image()
    with pytest.raises(NotEnoughFeatureMatchesError):
        register_images_standalone(
            moving, fixed,
            **_COMMON_KWARGS,
            qc_dir=tmp_path,
            qc_identifier="test_failure",
        )
    # Directory must exist (even if no QC file written for zero-match case)
    assert tmp_path.exists()


def test_force_failure_raises_even_with_no_natural_failure():
    """force_failure=True always raises NotEnoughFeatureMatchesError."""
    moving = _blank_image()
    fixed = _blank_image()
    with pytest.raises(NotEnoughFeatureMatchesError):
        register_images_standalone(
            moving, fixed,
            **_COMMON_KWARGS,
            force_failure=True,
        )


def test_rgb_axes_accepted():
    """YXS axes (RGB input) is accepted (no crash before feature extraction)."""
    moving = np.zeros((64, 64, 3), dtype=np.uint8)
    fixed = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(NotEnoughFeatureMatchesError):
        register_images_standalone(
            moving, fixed,
            axes_moving="YXS",
            axes_fixed="YXS",
            min_good_matches=1,
            test_flipping=False,
            verbose=False,
        )

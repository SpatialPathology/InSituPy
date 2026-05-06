"""Tests for the register_images_standalone() function and related dataclasses."""
import dask.array as da
import numpy as np
import pytest

from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy.images.registration import (
    RegistrationConfig,
    register_images_standalone,
)

try:
    import cv2 as _cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


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


# ---------------------------------------------------------------------------
# register_images_standalone — real image integration test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV required")
def test_registration_recovers_known_transform():
    """register_images_standalone recovers a synthetic affine transform on real image data.

    Uses skimage.data.coins() as a fixed reference image, applies a known
    rotation + translation to create the moving image, then checks that the
    registered output is close to the fixed image (MAE on the interior region).
    """
    skimage = pytest.importorskip("skimage", reason="scikit-image required")
    from skimage.data import coins

    fixed = coins()  # (303, 384) uint8, rich texture for SIFT

    # Build a known affine transform: 5° rotation + (20, 15) px translation
    h, w = fixed.shape
    M_gt = _cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
    M_gt[0, 2] += 20
    M_gt[1, 2] += 15
    moving = _cv2.warpAffine(fixed, M_gt, (w, h), flags=_cv2.INTER_LINEAR)

    registered, _ = register_images_standalone(
        moving, fixed,
        axes_moving="YX",
        axes_fixed="YX",
        min_good_matches=10,
        test_flipping=False,
        verbose=False,
    )

    assert registered.shape == fixed.shape

    # Crop borders to exclude zero-padding introduced by the synthetic warp
    pad = 40
    mae = np.mean(np.abs(
        registered[pad:-pad, pad:-pad].astype(float)
        - fixed[pad:-pad, pad:-pad].astype(float)
    ))
    assert mae < 15.0, f"Registration quality too low: MAE = {mae:.2f} (threshold 15.0)"


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

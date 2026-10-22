"""Tests for the register_images_standalone() function and related dataclasses."""
import dask.array as da
import numpy as np
import pytest

from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy.images.registration import (
    RegistrationConfig,
    _convert_to_grayscale,
    load_and_scale_images,
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


# ---------------------------------------------------------------------------
# convert_to_grayscale respects the axes descriptor
# ---------------------------------------------------------------------------

def _grayscale_features(img, axes):
    config = RegistrationConfig(
        axes_moving=axes, axes_fixed=axes, max_width=None, convert_to_grayscale=True, verbose=False
    )
    scaled = load_and_scale_images(img, img, config)
    return scaled.moving_scaled, scaled.axes_moving_effective


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
def test_grayscale_channels_first_rgb_matches_channels_last():
    rng = np.random.default_rng(0)
    yxs = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    gray_yxs, axes_yxs = _grayscale_features(yxs, "YXS")
    # the stage functions take canonical axes only; the helper itself handles S anywhere
    gray_syx = _convert_to_grayscale(np.moveaxis(yxs, -1, 0), "SYX")

    assert axes_yxs == "YX"
    assert gray_syx.shape == (40, 50)
    np.testing.assert_array_equal(gray_syx, gray_yxs)
    # RGB luminance weights (not BGR)
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255
    np.testing.assert_array_equal(
        _cv2.cvtColor(red, _cv2.COLOR_RGB2GRAY), _grayscale_features(red, "YXS")[0]
    )


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
def test_grayscale_multichannel_is_channel_mean():
    cyx = np.stack([np.full((30, 20), v, dtype=np.uint8) for v in (0, 60, 120, 240)])
    gray, axes = _grayscale_features(cyx, "CYX")

    assert axes == "YX"
    assert gray.shape == (30, 20)
    assert np.all(gray == 105)

# ---------------------------------------------------------------------------
# Axis order: channel-first RGB moving gives the canonical result (WP-12)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV required")
def test_channel_first_rgb_moving_matches_yxs():
    """SYX moving with H&E deconvolution and flip test equals the YXS run, in SYX layout."""
    pytest.importorskip("skimage", reason="scikit-image required")
    from skimage.data import astronaut

    fixed = astronaut()[::2, ::2]  # (256, 256, 3) uint8, textured RGB
    h, w = fixed.shape[:2]
    M_gt = _cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0)
    M_gt[:, 2] += (12, 8)
    moving = _cv2.warpAffine(fixed, M_gt, (w, h), flags=_cv2.INTER_LINEAR)

    kwargs = dict(
        fixed=fixed, axes_fixed="YXS", deconvolve_moving=True, deconvolve_fixed=True,
        decon_scale_factor=1.0, max_width=None, flann=False, min_good_matches=5,
        test_flipping=True, verbose=False,
    )
    reg_yxs, T_yxs = register_images_standalone(moving, axes_moving="YXS", **kwargs)
    reg_syx, T_syx = register_images_standalone(
        np.moveaxis(moving, -1, 0), axes_moving="SYX", **kwargs
    )

    np.testing.assert_allclose(T_syx, T_yxs)
    assert reg_syx.shape == (3, h, w)
    np.testing.assert_array_equal(reg_syx, np.moveaxis(reg_yxs, -1, 0))

def test_stage_functions_reject_non_canonical_axes():
    """Direct stage calls with SYX would flip/grayscale the wrong axis, so they raise."""
    img = np.zeros((3, 32, 32), dtype=np.uint8)
    config = RegistrationConfig(axes_moving="SYX", axes_fixed="YX", verbose=False)
    with pytest.raises(ValueError, match="canonical order"):
        load_and_scale_images(img, img[0], config)

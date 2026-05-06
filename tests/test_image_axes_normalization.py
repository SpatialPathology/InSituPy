import dask.array as da
import numpy as np
import pytest

from insitupy.images.axes import _transpose_to_standard_axes, normalize_axes_and_shape


def test_normalize_legacy_cyx_for_2d_image():
    img = da.from_array(np.zeros((32, 48), dtype=np.uint8))

    with pytest.warns(UserWarning, match="CYX"):
        normalized_img, axes = normalize_axes_and_shape(img, "CYX")

    assert axes == "YX"
    assert normalized_img.shape == (32, 48)


def test_normalize_legacy_cyx_singleton_channel_squeezes_to_yx():
    img = da.from_array(np.zeros((1, 32, 48), dtype=np.uint8))

    with pytest.warns(UserWarning, match="Coercing"):
        normalized_img, axes = normalize_axes_and_shape(img, "CYX")

    assert axes == "YX"
    assert normalized_img.shape == (32, 48)


def test_normalize_raises_for_nonlegacy_axes_shape_mismatch():
    img = da.from_array(np.zeros((32, 48), dtype=np.uint8))

    with pytest.raises(ValueError, match="Axes and image dimensionality mismatch"):
        normalize_axes_and_shape(img, "YXS")


def test_transpose_raises_for_inconsistent_pyramid_levels():
    img = [
        da.from_array(np.zeros((32, 48), dtype=np.uint8)),
        da.from_array(np.zeros((2, 16, 24), dtype=np.uint8)),
    ]

    with pytest.raises(ValueError, match="Inconsistent image pyramid dimensions"):
        _transpose_to_standard_axes(img, "CYX")


def test_transpose_handles_legacy_cyx_2d_without_crashing():
    img = da.from_array(np.zeros((32, 48), dtype=np.uint8))

    with pytest.warns(UserWarning, match="CYX"):
        transposed_img, axes = _transpose_to_standard_axes(img, "CYX")

    assert axes == "YX"
    assert transposed_img.shape == (32, 48)


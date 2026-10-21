import dask.array as da
import numpy as np
import pytest

from insitupy.images.axes import (
    _canonicalize_axes,
    _transpose_to_standard_axes,
    normalize_axes_and_shape,
)


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


# ── canonical axes (WP-12) ────────────────────────────────────────────────────

def _arr(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


# (axes, input shape, rgb, expected axes, numpy op turning the input into the expected output)
CANONICAL_CASES = [
    ("YXS", (6, 5, 3), None, "YXS", lambda a: a),
    ("SYX", (3, 6, 5), None, "YXS", lambda a: np.moveaxis(a, 0, -1)),
    ("SYX", (4, 6, 5), None, "YXS", lambda a: np.moveaxis(a, 0, -1)),  # RGBA
    ("YXC", (6, 5, 2), None, "CYX", lambda a: np.moveaxis(a, -1, 0)),
    ("CYX", (2, 6, 5), None, "CYX", lambda a: a),
    ("XY", (5, 6), None, "YX", lambda a: a.T),
    ("YX", (6, 5), None, "YX", lambda a: a),
    ("TYXS", (1, 6, 5, 3), None, "YXS", lambda a: a[0]),
    ("ZYX", (1, 6, 5), None, "YX", lambda a: a[0]),
    ("CYXS", (1, 6, 5, 3), None, "YXS", lambda a: a[0]),
    ("YXS", (6, 5, 1), None, "YX", lambda a: a[..., 0]),
    ("CYX", (3, 6, 5), True, "YXS", lambda a: np.moveaxis(a, 0, -1)),
    ("YXS", (6, 5, 3), False, "CYX", lambda a: np.moveaxis(a, -1, 0)),
]


@pytest.mark.parametrize("as_dask", [False, True])
@pytest.mark.parametrize(("axes", "shape", "rgb", "expected_axes", "op"), CANONICAL_CASES)
def test_canonicalize_axes(axes, shape, rgb, expected_axes, op, as_dask):
    arr = _arr(shape)
    img = da.from_array(arr, chunks=2) if as_dask else arr

    out, out_axes = _canonicalize_axes(img, axes, rgb=rgb)

    assert out_axes == expected_axes
    if as_dask:
        assert isinstance(out, da.Array)  # stays lazy
    np.testing.assert_array_equal(np.asarray(out), op(arr))


@pytest.mark.parametrize(
    ("axes", "shape", "rgb", "match"),
    [
        ("ZYX", (3, 6, 5), None, "no time series or z-stacks"),
        ("YXS", (6, 5, 2), None, "3 \\(RGB\\) or 4 \\(RGBA\\)"),
        ("CYXS", (2, 6, 5, 3), None, "both a 'C' and an 'S'"),
        ("YX", (6, 5), True, "no channel axis"),
        ("YYX", (6, 6, 5), None, "repeated"),
    ],
)
def test_canonicalize_axes_rejects(axes, shape, rgb, match):
    with pytest.raises(ValueError, match=match):
        _canonicalize_axes(_arr(shape), axes, rgb=rgb)


def test_canonicalize_axes_transposes_every_pyramid_level():
    levels = [_arr((3, 8, 6), seed=1), _arr((3, 4, 3), seed=2)]
    pyramid = [da.from_array(lvl) for lvl in levels]

    out, axes = _canonicalize_axes(pyramid, "SYX")

    assert axes == "YXS"
    for got, want in zip(out, levels, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.moveaxis(want, 0, -1))

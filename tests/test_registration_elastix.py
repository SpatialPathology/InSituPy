"""Tests for the elastix-based consecutive-section registration (insitupy.images.registration_elastix)."""
import json

import cv2
import dask.array as da
import numpy as np
import pytest
from scipy import ndimage as ndi

from insitupy.images.registration_elastix import (
    DisplacementTransform,
    apply_displacement_transform,
    register_images_elastix,
)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

PX_FIXED = 4.0    # µm
PX_MOVING = 2.0   # µm


def _tissue_density(shape=(400, 400), seed=0):
    """Asymmetric 'tissue' in nuclear-density units [0, 1] plus its mask.

    Ellipse + lobe with a notch, regional density variation and a cluster of ring-shaped 'ducts'
    in one quadrant, so that rotations/flips are distinguishable.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx - 200) / 150.0) ** 2 + ((yy - 215) / 120.0) ** 2 <= 1
    mask |= (xx - 300) ** 2 + (yy - 115) ** 2 <= 55 ** 2
    mask &= ~((np.abs(xx - 200) < 10) & (yy > 290))  # notch at the bottom
    regional = ndi.gaussian_filter(rng.random(shape), 25)
    regional = (regional - regional.min()) / (np.ptp(regional) + 1e-9)
    regional += 0.6 * np.clip((xx - 250) / 100.0, 0, 1)  # dense band on the right
    # sparse bright nuclei (as in DAPI), denser where `regional` is high
    nuclei = ndi.gaussian_filter((rng.random(shape) < 0.08 + 0.15 * regional).astype(np.float32), 0.8)
    dens = 0.15 + 0.3 * regional + 1.5 * nuclei
    rings = np.zeros(shape, np.float32)
    for _ in range(14):  # duct cluster, upper-left
        cx, cy = rng.integers(90, 190), rng.integers(140, 230)
        cv2.circle(rings, (int(cx), int(cy)), int(rng.integers(6, 12)), 1.0, thickness=3)
    dens = np.clip(dens + rings, 0, 1.5) * mask
    return ndi.gaussian_filter(dens, 0.7).astype(np.float32), mask


def _true_forward(angle=90.0, flip=True, scale_phys=1.15, moving_shape=(560, 640),
                  fixed_shape=(400, 400), amp=3.0):
    """Return T_true(u): moving level-0 px -> fixed level-0 px (affine + sinusoid)."""
    hm, wm = moving_shape
    hf, wf = fixed_shape
    s = scale_phys * PX_MOVING / PX_FIXED
    cm = np.array([wm / 2.0, hm / 2.0])
    cf = np.array([wf / 2.0, hf / 2.0])
    th = np.deg2rad(angle)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) * s  # CCW in image coords
    F = np.diag([-1.0, 1.0]) if flip else np.eye(2)
    L = R @ F

    def T(ux, uy):
        v = np.stack([ux - cm[0], uy - cm[1]], axis=0)
        x = L[0, 0] * v[0] + L[0, 1] * v[1] + cf[0]
        y = L[1, 0] * v[0] + L[1, 1] * v[1] + cf[1]
        return x + amp * np.sin(2 * np.pi * y / 150.0), y + amp * np.sin(2 * np.pi * x / 170.0)

    return T


def _make_pair(rgb_moving=True, seed=0, **kw):
    dens, mask = _tissue_density(seed=seed)
    # fixed: DAPI-like uint16 with flat "stitched tile" background in two rectangles
    fixed = dens * 3000.0 + 60.0  # camera/background offset
    fixed[:, :120] += 40.0        # tile steps
    fixed[300:, :] += 25.0
    fixed += np.random.default_rng(seed + 1).normal(0, 5, fixed.shape)
    fixed = np.clip(fixed, 0, 65535).astype(np.uint16)

    T = _true_forward(**kw)
    hm, wm = kw.get("moving_shape", (560, 640))
    uy, ux = np.mgrid[0:hm, 0:wm].astype(np.float32)
    fx, fy = T(ux, uy)
    mdens = cv2.remap(dens, fx.astype(np.float32), fy.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mmask = cv2.remap(mask.astype(np.uint8), fx.astype(np.float32), fy.astype(np.float32),
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if rgb_moving:
        # H&E-like: white background, pink stroma, purple nuclei proportional to density
        white = np.array([245, 245, 245], np.float32)
        pink = np.array([230, 150, 190], np.float32)
        purple = np.array([90, 40, 140], np.float32)
        t = mmask[..., None].astype(np.float32)
        d = np.clip(mdens, 0, 1)[..., None]
        rgb = white * (1 - t) + t * (pink * (1 - d) + purple * d)
        moving = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        moving = np.clip(mdens * 3000.0, 0, 65535).astype(np.uint16)
    return moving, fixed, T, mmask


def _sample_map(transform: DisplacementTransform, x, y):
    fy, fx = transform.grid_f
    gx = ((x + 0.5) / fx - 0.5).astype(np.float32)
    gy = ((y + 0.5) / fy - 0.5).astype(np.float32)
    mx = ndi.map_coordinates(transform.map_xy[..., 0], [gy, gx], order=1)
    my = ndi.map_coordinates(transform.map_xy[..., 1], [gy, gx], order=1)
    return mx, my


def _landmark_errors(transform, T, mmask, n=40, seed=3):
    """Pick interior moving points u, map them forward with T_true, then pull back with the
    estimated map; return the error in moving level-0 px."""
    inner = ndi.binary_erosion(mmask, iterations=25)
    ys, xs = np.nonzero(inner)
    idx = np.random.default_rng(seed).choice(len(xs), size=n, replace=False)
    ux, uy = xs[idx].astype(np.float64), ys[idx].astype(np.float64)
    fx, fy = T(ux, uy)
    mx, my = _sample_map(transform, fx, fy)
    return np.hypot(mx - ux, my - uy)


_FAST = dict(
    pixel_size_moving=PX_MOVING, pixel_size_fixed=PX_FIXED,
    init_resolution=8.0, elastix_resolution=4.0,
    affine_iterations=300, bspline_iterations=300, bspline_grid_spacing=200.0,
    verbose=False,
)


# ---------------------------------------------------------------------------
# Tests needing itk-elastix
# ---------------------------------------------------------------------------

def test_recovers_rotation_flip_scale_and_warp():
    """Global search + elastix + map composition recover a known 90°/flip/1.15 + sinusoid."""
    pytest.importorskip("itk")
    moving, fixed, T, mmask = _make_pair(rgb_moving=True)
    registered, tf = register_images_elastix(
        moving, fixed, axes_moving="YXS", axes_fixed="YX", nonrigid=True, **_FAST,
    )
    assert registered.shape == fixed.shape + (3,)
    assert registered.dtype == np.uint8
    assert tf.metrics["chosen_flip"] is True
    err = _landmark_errors(tf, T, mmask)
    # moving level-0 px are 2 µm; elastix runs at 4 µm/px. Affine-only leaves ~4 px from the
    # injected sinusoid, so this threshold also checks that the B-spline stage is composed in.
    assert np.median(err) < 1.5, f"median landmark error {np.median(err):.2f} px"
    assert tf.metrics["tissue_dice"] > 0.9
    assert tf.metrics["min_relative_jacobian"] > 0
    json.dumps(tf.metrics)  # must be JSON-serialisable (persisted in image metadata)


def test_axis_order_of_pure_translation():
    """A pure shift must come back with the right sign and x/y order (silent swap guard)."""
    pytest.importorskip("itk")
    dens, _ = _tissue_density()
    fixed = (dens * 3000).astype(np.uint16)
    # moving(u) = fixed(u + (dx, dy))  ->  pull map: fixed x -> moving x - (dx, dy)
    dx, dy = 12.0, -6.0
    moving = ndi.shift(fixed.astype(np.float32), (-dy, -dx), order=1).astype(np.uint16)
    _, tf = register_images_elastix(
        moving, fixed, axes_moving="YX", axes_fixed="YX",
        pixel_size_moving=PX_FIXED, pixel_size_fixed=PX_FIXED,
        nonrigid=False, test_flipping=False, scale_range=(1.0, 1.0), angle_step=90.0,
        init_resolution=8.0, elastix_resolution=4.0, warp=False, verbose=False,
    )
    mx, my = _sample_map(tf, np.array([200.0]), np.array([215.0]))
    assert abs(mx[0] - (200.0 - dx)) < 1.0 and abs(my[0] - (215.0 - dy)) < 1.0, (mx, my)


def test_initial_transform_skips_search():
    pytest.importorskip("itk")
    dens, _ = _tissue_density()
    fixed = (dens * 3000).astype(np.uint16)
    moving = fixed.copy()
    _, tf = register_images_elastix(
        moving, fixed, axes_moving="YX", axes_fixed="YX",
        pixel_size_moving=PX_FIXED, pixel_size_fixed=PX_FIXED,
        initial_transform=np.array([[1.0, 0, 0], [0, 1.0, 0]]), nonrigid=False,
        elastix_resolution=4.0, warp=False, verbose=False,
    )
    assert len(tf.metrics["candidates"]) == 1
    mx, my = _sample_map(tf, np.array([200.0]), np.array([215.0]))
    assert abs(mx[0] - 200.0) < 1.0 and abs(my[0] - 215.0) < 1.0


# ---------------------------------------------------------------------------
# Tiled warp (no itk needed)
# ---------------------------------------------------------------------------

def _affine_transform(fixed_shape, moving_shape, B, grid_f=4.0):
    """DisplacementTransform whose pull map is the affine B (fixed px -> moving px)."""
    H, W = fixed_shape
    Hg, Wg = int(np.ceil(H / grid_f)) + 1, int(np.ceil(W / grid_f)) + 1
    jy, jx = np.mgrid[0:Hg, 0:Wg].astype(np.float64)
    x = (jx + 0.5) * grid_f - 0.5
    y = (jy + 0.5) * grid_f - 0.5
    mx = B[0, 0] * x + B[0, 1] * y + B[0, 2]
    my = B[1, 0] * x + B[1, 1] * y + B[1, 2]
    return DisplacementTransform(
        map_xy=np.stack([mx, my], -1).astype(np.float32), grid_f=(grid_f, grid_f),
        fixed_shape=fixed_shape, moving_shape=moving_shape,
        pixel_size_fixed=1.0, pixel_size_moving=1.0, affine=np.eye(3)[:2], metrics={},
    )


@pytest.mark.parametrize("axes", ["YX", "YXS", "CYX"])
def test_tiled_warp_matches_single_affine_warp(axes):
    rng = np.random.default_rng(0)
    mh, mw = 101, 133
    base = (ndi.gaussian_filter(rng.random((mh, mw)), 2) * 60000).astype(np.uint16)
    if axes == "YX":
        img = base
    elif axes == "YXS":
        img = np.dstack([(base // 257).astype(np.uint8)] * 3)
        img[..., 1] = 255 - img[..., 1]
    else:
        img = np.stack([base, 65535 - base], 0)
    B = np.array([[0.9, 0.2, 5.0], [-0.15, 1.05, 3.0]])  # fixed -> moving
    fixed_shape = (97, 120)
    tf = _affine_transform(fixed_shape, (mh, mw), B)

    out = apply_displacement_transform(da.from_array(img, chunks=40), tf, axes, tile_size=37)

    def ref(ch):
        return cv2.warpAffine(ch, B, (fixed_shape[1], fixed_shape[0]),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    expected = ref(img) if axes != "CYX" else np.stack([ref(c) for c in img], 0)
    assert out.shape == expected.shape and out.dtype == img.dtype
    sl = (slice(2, -2), slice(2, -2))
    o = out[sl] if axes != "CYX" else out[:, 2:-2, 2:-2]
    e = expected[sl] if axes != "CYX" else expected[:, 2:-2, 2:-2]
    assert np.abs(o.astype(np.int64) - e.astype(np.int64)).max() <= 1


def test_transform_save_load_roundtrip(tmp_path):
    tf = _affine_transform((60, 70), (80, 90), np.array([[1.1, 0.0, 2.0], [0.1, 0.9, 1.0]]))
    tf.metrics = {"tissue_dice": 0.9, "candidates": [{"flip": False}]}
    path = tf.save(tmp_path / "t.npz")
    tf2 = DisplacementTransform.load(path)
    img = (np.random.default_rng(1).random((80, 90)) * 255).astype(np.uint8)
    np.testing.assert_array_equal(
        apply_displacement_transform(img, tf, "YX"), apply_displacement_transform(img, tf2, "YX")
    )
    assert tf2.metrics == tf.metrics and tf2.grid_f == tf.grid_f


def test_shape_mismatch_raises():
    tf = _affine_transform((60, 70), (80, 90), np.eye(3)[:2])
    with pytest.raises(ValueError, match="moving shape"):
        apply_displacement_transform(np.zeros((81, 90), np.uint8), tf, "YX")

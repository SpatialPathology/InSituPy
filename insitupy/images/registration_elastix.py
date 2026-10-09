"""Elastix-based image registration for consecutive tissue sections.

The feature-based pipeline in :mod:`insitupy.images.registration` matches individual nuclei and
therefore only works when both images show the *same* section. Consecutive sections share tissue
structure (outline, ducts, vessels, dense regions) but not cells, and they deform non-rigidly
(folds, stretch, shrinkage). This module registers them at tissue scale:

1. nuclear-density representations and tissue masks at low resolution,
2. a global flip x rotation x scale search,
3. elastix affine refinement (mutual information) of the best candidates,
4. optional elastix B-spline refinement,
5. a tiled full-resolution warp through a dense pull map.

Expected precision is tissue-neighbourhood (tens of µm), not single-cell.

Requires the optional dependency ``itk-elastix`` (``pip install "insitupy-spatial[registration]"``).

Coordinate conventions:
    All transforms are pull maps (like ``cv2.remap``): for each fixed level-0 pixel ``(x, y)`` they
    give the moving level-0 pixel to sample. Pixel centres are at integer coordinates. A low-res
    pixel index ``j`` on a grid with factor ``f`` (level-0 pixels per low-res pixel) corresponds to
    the level-0 coordinate ``(j + 0.5) * f - 0.5``.

Public API:
    register_images_elastix       - orchestrator function
    apply_displacement_transform  - tiled warp of an image through a ``DisplacementTransform``
    DisplacementTransform         - dense transform with ``save``/``load``
    ElastixRegistrationConfig     - frozen config dataclass
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from insitupy._constants import SHRT_MAX

logger = logging.getLogger(__name__)

# Tree drawing characters (same style as insitupy.images.registration)
_TSIGN = "├"   # ├
_LSIGN = "└"   # └
_VLINE = "│"   # │
_HLINE = "─"   # ─

_SCORE_SMOOTHING_UM = 40.0  # smoothing of the maps used to rank candidates after elastix affine
_MASK_RES = 8.0             # µm/px at which tissue masks are detected
# elastix sampling masks: the fixed tissue mask is dilated so that the tissue boundary (and its
# mismatch) is part of the metric; a moving mask would silently drop exactly those samples.
_FIXED_MASK_DILATION_UM = 100.0
_USE_MOVING_MASK = False
# weight of the smoothed tissue mask blended into the elastix representations (outline term).
# Calibrated on SATURN3 DRUFU-M11 (2026-10-09): structure NCC 0.40 -> 0.47 at 0.3.
_MASK_BLEND = 0.3
_SUPPORTED_AXES = ("YX", "YXS", "SYX")
_PRECISION_NOTE = "tissue-scale (tens of µm), not single-cell"


def _import_itk():
    """Import ``itk`` lazily (slow import, optional dependency)."""
    from insitupy._checks import try_import
    return try_import(
        "itk",
        installation_command='pip install "insitupy-spatial[registration]"',
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElastixRegistrationConfig:
    """Immutable configuration for elastix-based registration.

    The fields mirror the keyword arguments of :func:`register_images_elastix`.
    """
    axes_moving: str = "YXS"
    axes_fixed: str = "YX"
    pixel_size_moving: float | None = None
    pixel_size_fixed: float | None = None
    # global initialisation
    init_resolution: float = 8.0
    init_smoothing: float = 120.0
    angle_step: float = 4.0
    scale_range: tuple[float, float] = (0.8, 1.3)
    scale_step: float = 0.05
    test_flipping: bool = True
    top_k: int = 3
    initial_transform: np.ndarray | None = None
    # elastix
    elastix_resolution: float = 4.0
    representation_smoothing: float = 8.0
    affine_resolutions: int = 4
    affine_iterations: int = 500
    nonrigid: bool = True
    bspline_grid_spacing: float = 300.0
    bspline_resolutions: int = 3
    bspline_iterations: int = 500
    bspline_bending_weight: float = 1.0
    random_seed: int = 0
    parameter_maps: tuple[dict, ...] | None = None
    # output
    tile_size: int = 4096
    verbose: bool = True


@dataclass
class DisplacementTransform:
    """Dense pull transform from fixed level-0 pixels to moving level-0 pixels.

    Attributes:
        map_xy: ``(Hg, Wg, 2)`` float32 array on a low-resolution grid over the fixed image.
            Channel 0 is the moving x coordinate, channel 1 the moving y coordinate, both in
            moving level-0 pixels.
        grid_f: ``(fy, fx)`` fixed level-0 pixels per grid step. Grid index ``j`` corresponds to
            the fixed level-0 coordinate ``(j + 0.5) * f - 0.5``.
        fixed_shape: ``(H, W)`` of the fixed image at level 0.
        moving_shape: ``(H, W)`` of the moving image at level 0.
        pixel_size_fixed: Fixed level-0 pixel size in µm.
        pixel_size_moving: Moving level-0 pixel size in µm.
        affine: 2x3 affine (moving level-0 px -> fixed level-0 px) of the affine stage only,
            for summaries and compatibility with matrix-based tools.
        metrics: JSON-serialisable QC metrics.
    """
    map_xy: np.ndarray
    grid_f: tuple[float, float]
    fixed_shape: tuple[int, int]
    moving_shape: tuple[int, int]
    pixel_size_fixed: float
    pixel_size_moving: float
    affine: np.ndarray
    metrics: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Save the transform as a compressed ``.npz`` file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "grid_f": [float(v) for v in self.grid_f],
            "fixed_shape": [int(v) for v in self.fixed_shape],
            "moving_shape": [int(v) for v in self.moving_shape],
            "pixel_size_fixed": float(self.pixel_size_fixed),
            "pixel_size_moving": float(self.pixel_size_moving),
            "metrics": self.metrics,
        }
        np.savez_compressed(
            path,
            map_xy=self.map_xy.astype(np.float32),
            affine=np.asarray(self.affine, dtype=np.float64),
            meta=np.array(json.dumps(meta)),
        )
        # np.savez appends ".npz" if missing
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> DisplacementTransform:
        """Load a transform written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return cls(
                map_xy=f["map_xy"].astype(np.float32),
                grid_f=tuple(meta["grid_f"]),
                fixed_shape=tuple(meta["fixed_shape"]),
                moving_shape=tuple(meta["moving_shape"]),
                pixel_size_fixed=meta["pixel_size_fixed"],
                pixel_size_moving=meta["pixel_size_moving"],
                affine=f["affine"],
                metrics=meta["metrics"],
            )


@dataclass
class _LowRes:
    """Low-resolution representation of one image."""
    signal: np.ndarray    # float32 nuclear signal (unsmoothed, unnormalised)
    mask: np.ndarray      # bool tissue mask
    fx: float             # level-0 px per low-res px (x)
    fy: float             # level-0 px per low-res px (y)
    res: float            # µm per low-res px


@dataclass
class _Candidate:
    A: np.ndarray         # 3x3, moving level-0 px -> fixed level-0 px
    flip: bool
    angle: float
    scale: float
    init_score: float
    affine_score: float | None = None


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _lr_to_l0_matrix(fx: float, fy: float) -> np.ndarray:
    """3x3 matrix mapping low-res pixel coords to level-0 pixel coords."""
    return np.array([
        [fx, 0.0, 0.5 * fx - 0.5],
        [0.0, fy, 0.5 * fy - 0.5],
        [0.0, 0.0, 1.0],
    ])


def _to3(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return M if M.shape == (3, 3) else np.vstack([M, [0.0, 0.0, 1.0]])


def _lr_matrix(A: np.ndarray, fix: _LowRes, mov: _LowRes) -> np.ndarray:
    """Convert a level-0 matrix (moving -> fixed) to low-res pixel coords of the given grids."""
    return np.linalg.inv(_lr_to_l0_matrix(fix.fx, fix.fy)) @ A @ _lr_to_l0_matrix(mov.fx, mov.fy)


def _l0_matrix(M_lr: np.ndarray, fix: _LowRes, mov: _LowRes) -> np.ndarray:
    """Convert a low-res matrix (moving -> fixed) to level-0 pixel coords."""
    return _lr_to_l0_matrix(fix.fx, fix.fy) @ M_lr @ np.linalg.inv(_lr_to_l0_matrix(mov.fx, mov.fy))


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _similarity(angle: float, flip: bool, scale: float, mc, fc) -> np.ndarray:
    """Flip about the moving centroid, rotate/scale about it, then move it onto the fixed centroid."""
    F = np.eye(3)
    if flip:
        F = np.array([[-1.0, 0.0, 2.0 * mc[0]], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = _to3(cv2.getRotationMatrix2D((float(mc[0]), float(mc[1])), float(angle), float(scale)))
    return _translation(fc[0] - mc[0], fc[1] - mc[1]) @ R @ F


# ---------------------------------------------------------------------------
# Image access and representations
# ---------------------------------------------------------------------------

def _as_level_list(img) -> list:
    """Return a list of pyramid levels (highest resolution first)."""
    if isinstance(img, (list, tuple)):
        if len(img) == 0:
            raise ValueError("Empty image pyramid.")
        if isinstance(img[0], (list, tuple)):
            return _as_level_list(img[0])
        return list(img)
    return [img]


def _hw(arr, axes: str) -> tuple[int, int]:
    if axes == "YX":
        return int(arr.shape[0]), int(arr.shape[1])
    if axes == "YXS":
        return int(arr.shape[0]), int(arr.shape[1])
    if axes == "SYX":
        return int(arr.shape[1]), int(arr.shape[2])
    raise ValueError(f"Unsupported axes '{axes}'. Supported: {_SUPPORTED_AXES}.")


def _hematoxylin(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2hed
    return np.clip(rgb2hed(rgb)[..., 0], 0, None).astype(np.float32)


def _saturation_255(rgb: np.ndarray, dtype) -> np.ndarray:
    """RGB saturation (max - min) scaled to 0-255."""
    sat = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        sat *= 255.0 / float(np.iinfo(dtype).max)
    elif sat.max() <= 1.0:
        sat *= 255.0
    return sat


def _keep_largest_filled(mask: np.ndarray, closing: int) -> np.ndarray:
    if closing > 0:
        mask = ndi.binary_closing(mask, iterations=closing)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    return mask


def _mask_fluorescence(signal: np.ndarray, res: float) -> np.ndarray:
    """Texture mask: tissue has nuclear texture, stitched-tile background is flat.

    The high-pass removes smooth background and tile steps before the local RMS is taken, so
    tile borders do not dominate the texture map.
    """
    from skimage.filters import threshold_otsu
    x = signal.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    hp = x - ndi.gaussian_filter(x, 15.0 / res)
    sd = np.sqrt(ndi.gaussian_filter(hp ** 2, 20.0 / res))
    if sd.max() <= 0:
        return np.zeros_like(sd, dtype=bool)
    mask = sd > threshold_otsu(sd)
    mask = ndi.binary_opening(mask, iterations=2)
    return _keep_largest_filled(mask, closing=5)


def _mask_brightfield(sat_255: np.ndarray, res: float) -> np.ndarray:
    """Saturation mask: white background has no saturation (pale eosin stroma still does).

    White background measured < 1 on the 0-255 scale on the SATURN3 H&E scans, pale or lacy
    tissue down to ~3-5, hence the low threshold. The closing over ~80 µm turns lacy tissue
    (fat, fragmented stroma) into one tissue envelope.
    """
    sat = ndi.gaussian_filter(sat_255, 20.0 / res)
    return _keep_largest_filled(sat > 4.0, closing=max(1, int(round(80.0 / res))))


def _prepare(img, axes: str, pixel_size: float, res: float, name: str) -> _LowRes:
    """Load an image at ``res`` µm/px and compute its nuclear signal and tissue mask.

    The mask is always detected at ``_MASK_RES`` µm/px and then resized to the signal grid, so
    that it does not depend on the working resolution (the texture the fluorescence mask relies
    on changes with resolution).
    """
    lr = _load(img, axes, pixel_size, res, name)
    if res == _MASK_RES:
        mask = lr.mask
    else:
        mask_lr = _load(img, axes, pixel_size, _MASK_RES, name).mask
        Ht, Wt = lr.signal.shape
        mask = cv2.resize(mask_lr.astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_LINEAR) > 0.5
    _check_mask(mask, name, res)
    return _LowRes(signal=lr.signal, mask=mask, fx=lr.fx, fy=lr.fy, res=res)


def _check_mask(mask: np.ndarray, name: str, res: float) -> None:
    frac = float(mask.mean())
    if frac == 0.0 or frac > 0.98:
        raise ValueError(
            f"{name}: tissue mask detection failed (mask covers {frac:.1%} of the image at "
            f"{res} µm/px). Check that the image contains tissue on a background."
        )


def _load(img, axes: str, pixel_size: float, res: float, name: str) -> _LowRes:
    """Load an image at ``res`` µm/px and compute its nuclear signal and tissue mask there.

    Only the pyramid level closest to (but not coarser than) ``res`` is computed, and it is
    block-averaged lazily before computing, so level 0 of a large image is never loaded whole.
    """
    levels = _as_level_list(img)
    H0, W0 = _hw(levels[0], axes)

    # coarsest level that is still at least as fine as the target resolution
    chosen = levels[0]
    for lev in levels:
        h, _ = _hw(lev, axes)
        if pixel_size * H0 / h <= res * 1.0001:
            chosen = lev
    H_L, W_L = _hw(chosen, axes)
    gy, gx = H0 / H_L, W0 / W_L

    arr = chosen if isinstance(chosen, da.Array) else da.from_array(np.asarray(chosen))
    if axes == "SYX":
        arr = da.moveaxis(arr, 0, -1)
    rgb = axes in ("YXS", "SYX")
    dtype = arr.dtype

    # lazy integer block averaging down to <= 2x the target size
    k = max(1, int((res / (pixel_size * gy)) // 2))
    if k > 1:
        arr = da.coarsen(np.mean, arr.astype(np.float32), {0: k, 1: k}, trim_excess=True)
    arr = np.asarray(arr.compute() if isinstance(arr, da.Array) else arr)
    Hc, Wc = arr.shape[:2]

    Ht = max(1, int(round(Hc * k * gy * pixel_size / res)))
    Wt = max(1, int(round(Wc * k * gx * pixel_size / res)))
    fy = gy * k * Hc / Ht
    fx = gx * k * Wc / Wt

    if rgb:
        small = cv2.resize(arr.astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_AREA)
        if np.issubdtype(dtype, np.integer):
            small_int = np.clip(np.rint(small), 0, np.iinfo(dtype).max).astype(dtype)
        else:
            small_int = small
        signal = _hematoxylin(small_int)
        mask = _mask_brightfield(_saturation_255(small, dtype), res)
    else:
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected a 2D image for axes 'YX', got shape {arr.shape}.")
        small = cv2.resize(arr.astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_AREA)
        signal = np.log1p(np.clip(small, 0, None)).astype(np.float32)
        mask = _mask_fluorescence(signal, res)
    return _LowRes(signal=signal, mask=mask, fx=fx, fy=fy, res=res)


def _smooth_norm(lr: _LowRes, sigma_um: float) -> np.ndarray:
    """Gaussian-smooth and normalise the signal to [0, 1] using percentiles inside the mask."""
    x = ndi.gaussian_filter(lr.signal, sigma_um / lr.res) if sigma_um > 0 else lr.signal
    vals = x[lr.mask]
    lo, hi = np.percentile(vals, [1, 99.5]) if vals.size else (0.0, 1.0)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else 0.0


def _ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    if m.sum() < 10:
        return 0.0
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def _score(fix_s, fmask, warped_s, wmask) -> tuple[float, float, float]:
    """Return (score, dice, ncc). NCC is only evaluated when the outlines overlap reasonably."""
    d = _dice(fmask, wmask)
    if d < 0.5:
        return d / 2.0, d, 0.0
    c = _ncc(fix_s, warped_s, fmask & wmask)
    return 0.5 * d + 0.5 * c, d, c


def _warp_affine_lr(img, mask, M_lr, shape):
    H, W = shape
    wm = cv2.warpAffine(img, M_lr[:2], (W, H), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wmask = cv2.warpAffine(mask.astype(np.uint8), M_lr[:2], (W, H), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return wm, wmask


def _phase_correct(fix_s, mov_s, M_lr, max_shift) -> np.ndarray:
    H, W = fix_s.shape
    wm = cv2.warpAffine(mov_s, M_lr[:2], (W, H), flags=cv2.INTER_LINEAR)
    (dx, dy), _ = cv2.phaseCorrelate(fix_s.astype(np.float64), wm.astype(np.float64))
    dx = float(np.clip(dx, -max_shift[0], max_shift[0]))
    dy = float(np.clip(dy, -max_shift[1], max_shift[1]))
    return _translation(-dx, -dy) @ M_lr


def _eval_candidate(fix_s, fix, mov_s, mov, M_lr):
    """Score a candidate with and without phase-correlation shift correction; keep the better."""
    H, W = fix_s.shape
    best = None
    max_shift = (0.1 * W, 0.1 * H)
    for M in (M_lr, _phase_correct(fix_s, mov_s, M_lr, max_shift)):
        wm, wmask = _warp_affine_lr(mov_s, mov.mask, M, (H, W))
        s = _score(fix_s, fix.mask, wm, wmask)[0]
        if best is None or s > best[0]:
            best = (s, M)
    return best


# ---------------------------------------------------------------------------
# Global initialisation
# ---------------------------------------------------------------------------

def _centroid_xy(mask: np.ndarray) -> np.ndarray:
    cy, cx = ndi.center_of_mass(mask)
    return np.array([cx, cy])


def _global_search(moving, fixed, cfg: ElastixRegistrationConfig) -> list[_Candidate]:
    """Exhaustive flip x rotation x scale search, then a local refinement of the best candidates."""
    # coarse grid at 2x the init resolution
    res_c = 2.0 * cfg.init_resolution
    fix_c = _prepare(fixed, cfg.axes_fixed, cfg.pixel_size_fixed, res_c, "fixed")
    mov_c = _prepare(moving, cfg.axes_moving, cfg.pixel_size_moving, res_c, "moving")
    fix_cs = _smooth_norm(fix_c, cfg.init_smoothing)
    mov_cs = _smooth_norm(mov_c, cfg.init_smoothing)
    fc = _centroid_xy(fix_c.mask)
    mc = _centroid_xy(mov_c.mask)

    flips = (False, True) if cfg.test_flipping else (False,)
    angles = np.arange(0.0, 360.0, cfg.angle_step)
    lo, hi = cfg.scale_range
    scales = np.arange(lo, hi + 1e-9, cfg.scale_step)

    coarse = []
    for flip in flips:
        for scale in scales:
            for angle in angles:
                M = _similarity(angle, flip, scale, mc, fc)
                s, M = _eval_candidate(fix_cs, fix_c, mov_cs, mov_c, M)
                coarse.append(_Candidate(_l0_matrix(M, fix_c, mov_c), flip, float(angle), float(scale), s))
    coarse.sort(key=lambda c: c.init_score, reverse=True)

    # local refinement at the init resolution around the best distinct coarse candidates
    fix_i = _prepare(fixed, cfg.axes_fixed, cfg.pixel_size_fixed, cfg.init_resolution, "fixed")
    mov_i = _prepare(moving, cfg.axes_moving, cfg.pixel_size_moving, cfg.init_resolution, "moving")
    fix_is = _smooth_norm(fix_i, cfg.init_smoothing)
    mov_is = _smooth_norm(mov_i, cfg.init_smoothing)
    fc_i = _centroid_xy(fix_i.mask)

    d_angles = np.arange(-cfg.angle_step, cfg.angle_step + 1e-9, 1.0)
    d_scales = 1.0 + np.arange(-cfg.scale_step, cfg.scale_step + 1e-9, cfg.scale_step / 2.0)
    refined = []
    for cand in _distinct(coarse, 3):
        M0 = _lr_matrix(cand.A, fix_i, mov_i)
        for da_ in d_angles:
            for ds in d_scales:
                Rd = _to3(cv2.getRotationMatrix2D((float(fc_i[0]), float(fc_i[1])), float(da_), float(ds)))
                s, M = _eval_candidate(fix_is, fix_i, mov_is, mov_i, Rd @ M0)
                refined.append(_Candidate(
                    _l0_matrix(M, fix_i, mov_i), cand.flip,
                    float((cand.angle + da_) % 360.0), float(cand.scale * ds), s,
                ))
    refined.sort(key=lambda c: c.init_score, reverse=True)
    return _distinct(refined, cfg.top_k)


def _distinct(cands: list[_Candidate], k: int, min_angle: float = 10.0) -> list[_Candidate]:
    """Pick the best ``k`` candidates that differ in flip or by at least ``min_angle`` degrees."""
    out: list[_Candidate] = []
    for c in cands:
        if all(
            c.flip != o.flip or min(abs(c.angle - o.angle), 360.0 - abs(c.angle - o.angle)) >= min_angle
            for o in out
        ):
            out.append(c)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# Elastix
# ---------------------------------------------------------------------------

def _parameter_object(itk, cfg: ElastixRegistrationConfig, nonrigid: bool):
    po = itk.ParameterObject.New()
    if cfg.parameter_maps is not None:
        for pm in cfg.parameter_maps:
            po.AddParameterMap({
                str(k): [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
                for k, v in pm.items()
            })
        return po

    seed = str(int(cfg.random_seed))
    aff = po.GetDefaultParameterMap("affine", int(cfg.affine_resolutions))
    aff["AutomaticTransformInitialization"] = ["false"]
    aff["MaximumNumberOfIterations"] = [str(int(cfg.affine_iterations))]
    aff["RandomSeed"] = [seed]
    aff["WriteResultImage"] = ["false"]
    po.AddParameterMap(aff)
    if nonrigid:
        bs = po.GetDefaultParameterMap(
            "bspline", int(cfg.bspline_resolutions), float(cfg.bspline_grid_spacing)
        )
        bs["Registration"] = ["MultiMetricMultiResolutionRegistration"]
        bs["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
        bs["Metric0Weight"] = ["1.0"]
        bs["Metric1Weight"] = [str(float(cfg.bspline_bending_weight))]
        bs["MaximumNumberOfIterations"] = [str(int(cfg.bspline_iterations))]
        bs["RandomSeed"] = [seed]
        bs["WriteResultImage"] = ["false"]
        po.AddParameterMap(bs)
    return po


def _itk_image(itk, arr: np.ndarray, res: float, origin: float = 0.0):
    img = itk.image_from_array(np.ascontiguousarray(arr))
    img.SetSpacing((float(res), float(res)))
    img.SetOrigin((float(origin), float(origin)))
    return img


def _run_elastix(itk, fix_e: _LowRes, fix_es: np.ndarray, mov_e: _LowRes, mov_es: np.ndarray,
                 M_lr: np.ndarray, cfg: ElastixRegistrationConfig, nonrigid: bool, log: bool):
    """Run elastix on the moving image pre-warped by ``M_lr``.

    Returns the pull map in **moving low-res pixel coordinates** on the fixed low-res grid
    (``(H, W, 2)``, x then y).
    """
    H, W = fix_es.shape
    res = fix_e.res
    pad = int(round(0.1 * max(H, W)))
    M_pad = _translation(pad, pad) @ M_lr
    canvas = (W + 2 * pad, H + 2 * pad)
    mov_w = cv2.warpAffine(mov_es, M_pad[:2], canvas, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mov_wm = cv2.warpAffine(mov_e.mask.astype(np.uint8), M_pad[:2], canvas, flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    f_img = _itk_image(itk, fix_es.astype(np.float32), res)
    m_img = _itk_image(itk, mov_w.astype(np.float32), res, origin=-pad * res)
    fmask = fix_e.mask
    if _FIXED_MASK_DILATION_UM > 0:
        fmask = ndi.binary_dilation(fmask, iterations=max(1, int(round(_FIXED_MASK_DILATION_UM / res))))
    f_mask = _itk_image(itk, fmask.astype(np.uint8), res)

    elx = itk.ElastixRegistrationMethod.New(f_img, m_img)
    elx.SetParameterObject(_parameter_object(itk, cfg, nonrigid))
    elx.SetFixedMask(f_mask)
    if _USE_MOVING_MASK:
        elx.SetMovingMask(_itk_image(itk, mov_wm.astype(np.uint8), res, origin=-pad * res))
    elx.SetLogToConsole(log)
    elx.UpdateLargestPossibleRegion()
    tpo = elx.GetTransformParameterObject()

    with tempfile.TemporaryDirectory() as tmp:
        tfx = itk.TransformixFilter.New(f_img)
        tfx.SetTransformParameterObject(tpo)
        tfx.SetComputeDeformationField(True)
        tfx.SetOutputDirectory(tmp)  # transformix writes deformationField.nii otherwise into cwd
        tfx.SetLogToConsole(log)
        tfx.UpdateLargestPossibleRegion()
        d = np.array(itk.array_from_image(tfx.GetOutputDeformationField()), dtype=np.float64)

    # fixed index j -> physical j*res -> + d -> canvas coords (frame of fixed lr) -> moving lr
    jy, jx = np.mgrid[0:H, 0:W].astype(np.float64)
    kx = jx + d[..., 0] / res
    ky = jy + d[..., 1] / res
    Minv = np.linalg.inv(M_lr)
    mx = Minv[0, 0] * kx + Minv[0, 1] * ky + Minv[0, 2]
    my = Minv[1, 0] * kx + Minv[1, 1] * ky + Minv[1, 2]
    return np.stack([mx, my], axis=-1).astype(np.float32)


def _remap_lr(img: np.ndarray, mask: np.ndarray, map_lr: np.ndarray):
    mx, my = map_lr[..., 0], map_lr[..., 1]
    w = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wm = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return w, wm


def _lr_map_to_l0(map_lr: np.ndarray, mov: _LowRes) -> np.ndarray:
    out = np.empty_like(map_lr)
    out[..., 0] = (map_lr[..., 0] + 0.5) * mov.fx - 0.5
    out[..., 1] = (map_lr[..., 1] + 0.5) * mov.fy - 0.5
    return out


def _fit_affine(map_l0: np.ndarray, fix: _LowRes) -> np.ndarray:
    """Least-squares 2x3 affine (moving level-0 -> fixed level-0) from a pull map inside the mask."""
    jy, jx = np.nonzero(fix.mask)
    P = np.stack([(jx + 0.5) * fix.fx - 0.5, (jy + 0.5) * fix.fy - 0.5, np.ones(jx.size)], axis=1)
    Q = map_l0[jy, jx].astype(np.float64)
    B, *_ = np.linalg.lstsq(P, Q, rcond=None)  # fixed -> moving: Q = P @ B
    B3 = np.vstack([B.T, [0.0, 0.0, 1.0]])
    return np.linalg.inv(B3)[:2]


def _min_relative_jacobian(map_final: np.ndarray, map_affine: np.ndarray, mask: np.ndarray) -> float:
    """Min of det(J_final) / det(J_affine) inside the (eroded) mask; <= 0 means local folding."""
    def det(m):
        dxdx = np.gradient(m[..., 0], axis=1)
        dxdy = np.gradient(m[..., 0], axis=0)
        dydx = np.gradient(m[..., 1], axis=1)
        dydy = np.gradient(m[..., 1], axis=0)
        return dxdx * dydy - dxdy * dydx
    inner = ndi.binary_erosion(mask, iterations=2)
    if not inner.any():
        return 1.0
    det_aff = float(np.median(det(map_affine)[inner]))
    if abs(det_aff) < 1e-12:
        return 0.0
    return float(np.min(det(map_final)[inner] / det_aff))


# ---------------------------------------------------------------------------
# Tiled warp
# ---------------------------------------------------------------------------

_CV2_REMAP_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def apply_displacement_transform(
    image,
    transform: DisplacementTransform,
    axes: str,
    tile_size: int = 4096,
) -> np.ndarray:
    """Warp a moving image into the fixed frame through a :class:`DisplacementTransform`.

    The output is assembled tile by tile. For each output tile only the corresponding window of
    the moving image is read (lazily from dask arrays), so the full-resolution map is never
    materialised.

    Args:
        image: Moving image at level 0 (numpy or dask array, or a list of pyramid levels of
            which level 0 is used). Its level-0 shape must match ``transform.moving_shape``.
        transform: Transform returned by :func:`register_images_elastix`.
        axes: ``'YX'``, ``'YXS'`` (RGB) or ``'CYX'`` (multichannel).
        tile_size: Output tile edge length in pixels.

    Returns:
        Numpy array with the fixed level-0 spatial shape and the moving dtype.
    """
    img = _as_level_list(image)[0]
    if axes not in ("YX", "YXS", "CYX"):
        raise ValueError(f"Unsupported axes '{axes}'. Supported: 'YX', 'YXS', 'CYX'.")
    hw = (img.shape[0], img.shape[1]) if axes in ("YX", "YXS") else (img.shape[1], img.shape[2])
    if tuple(int(v) for v in hw) != tuple(int(v) for v in transform.moving_shape):
        raise ValueError(
            f"Image shape {hw} does not match the transform's moving shape {transform.moving_shape}."
        )

    H, W = (int(v) for v in transform.fixed_shape)
    dtype = np.dtype(img.dtype)
    if axes == "YX":
        out = np.zeros((H, W), dtype=dtype)
    elif axes == "YXS":
        out = np.zeros((H, W, img.shape[2]), dtype=dtype)
    else:
        out = np.zeros((img.shape[0], H, W), dtype=dtype)

    fy, fx = (float(v) for v in transform.grid_f)
    map_x = np.ascontiguousarray(transform.map_xy[..., 0], dtype=np.float32)
    map_y = np.ascontiguousarray(transform.map_xy[..., 1], dtype=np.float32)
    mh, mw = hw
    work_dtype = dtype if dtype.type in _CV2_REMAP_DTYPES else np.dtype(np.float32)

    def _remap(window_2d_or_rgb, sx, sy):
        win = window_2d_or_rgb
        if win.dtype != work_dtype:
            win = win.astype(work_dtype)
        res = cv2.remap(win, sx, sy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if res.dtype != dtype:
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                res = np.clip(np.rint(res), info.min, info.max)
            res = res.astype(dtype)
        return res

    for y0 in range(0, H, tile_size):
        y1 = min(H, y0 + tile_size)
        for x0 in range(0, W, tile_size):
            x1 = min(W, x0 + tile_size)
            ys = np.arange(y0, y1, dtype=np.float32)
            xs = np.arange(x0, x1, dtype=np.float32)
            gx = np.broadcast_to((xs + 0.5) / fx - 0.5, (y1 - y0, x1 - x0))
            gy = np.broadcast_to(((ys + 0.5) / fy - 0.5)[:, None], (y1 - y0, x1 - x0))
            gx = np.ascontiguousarray(gx, dtype=np.float32)
            gy = np.ascontiguousarray(gy, dtype=np.float32)
            src_x = cv2.remap(map_x, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            src_y = cv2.remap(map_y, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            finite = np.isfinite(src_x) & np.isfinite(src_y)
            if not finite.any():
                continue
            wx0 = int(max(0, np.floor(src_x[finite].min()) - 2))
            wy0 = int(max(0, np.floor(src_y[finite].min()) - 2))
            wx1 = int(min(mw, np.ceil(src_x[finite].max()) + 3))
            wy1 = int(min(mh, np.ceil(src_y[finite].max()) + 3))
            if wx1 <= wx0 or wy1 <= wy0:
                continue
            if max(wx1 - wx0, wy1 - wy0) > SHRT_MAX:
                raise ValueError(
                    "Degenerate transform: one output tile maps to a source window larger than "
                    f"{SHRT_MAX} px. Reduce `tile_size` or check the transform."
                )
            sx = (src_x - wx0).astype(np.float32)
            sy = (src_y - wy0).astype(np.float32)

            if axes == "YX":
                win = np.asarray(img[wy0:wy1, wx0:wx1])
                out[y0:y1, x0:x1] = _remap(win, sx, sy)
            elif axes == "YXS":
                win = np.asarray(img[wy0:wy1, wx0:wx1, :])
                warped = _remap(win, sx, sy)
                out[y0:y1, x0:x1, :] = warped.reshape(y1 - y0, x1 - x0, -1)
            else:
                win = np.asarray(img[:, wy0:wy1, wx0:wx1])
                for c in range(win.shape[0]):
                    out[c, y0:y1, x0:x1] = _remap(win[c], sx, sy)
    return out


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def _to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _overlay_bgr(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Fixed in magenta, moving in green, overlap white."""
    return np.dstack([_to_u8(fixed), _to_u8(moving), _to_u8(fixed)])


def _checkerboard(a: np.ndarray, b: np.ndarray, n: int = 8) -> np.ndarray:
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    sel = ((yy * n // max(H, 1)) + (xx * n // max(W, 1))) % 2 == 0
    return _to_u8(np.where(sel, a, b))


def _write_qc(qc_dir: Path, ident: str, fix_es, mov_es, fix_e, mov_e, init_lr, map_aff, map_fin,
              disp_um, cands: list[_Candidate]):
    import pandas as pd
    qc_dir.mkdir(parents=True, exist_ok=True)
    p = lambda name: str(qc_dir / f"{ident}__elastix_{name}")  # noqa: E731
    cv2.imwrite(p("fixed_lr.png"), _to_u8(fix_es))
    cv2.imwrite(p("moving_lr.png"), _to_u8(mov_es))
    h = 400
    fm = cv2.resize(fix_e.mask.astype(np.uint8) * 255, (int(h * fix_e.mask.shape[1] / fix_e.mask.shape[0]), h))
    mm = cv2.resize(mov_e.mask.astype(np.uint8) * 255, (int(h * mov_e.mask.shape[1] / mov_e.mask.shape[0]), h))
    cv2.imwrite(p("masks.png"), np.hstack([fm, np.full((h, 10), 128, np.uint8), mm]))

    H, W = fix_es.shape
    w_init, _ = _warp_affine_lr(mov_es, mov_e.mask, init_lr, (H, W))
    w_aff, _ = _remap_lr(mov_es, mov_e.mask, map_aff)
    w_fin, _ = _remap_lr(mov_es, mov_e.mask, map_fin)
    cv2.imwrite(p("overlay_init.png"), _overlay_bgr(fix_es, w_init))
    cv2.imwrite(p("overlay_affine.png"), _overlay_bgr(fix_es, w_aff))
    cv2.imwrite(p("overlay_final.png"), _overlay_bgr(fix_es, w_fin))
    cv2.imwrite(p("checkerboard_final.png"), _checkerboard(fix_es, w_fin))

    vmax = max(float(np.percentile(disp_um[fix_e.mask], 99)) if fix_e.mask.any() else 0.0, 1e-6)
    heat = cv2.applyColorMap(_to_u8(disp_um / vmax), cv2.COLORMAP_VIRIDIS)
    heat[~fix_e.mask] = 0
    cv2.imwrite(p(f"displacement_0-{vmax:.0f}um.png"), heat)

    pd.DataFrame([
        {"flip": c.flip, "angle_deg": round(c.angle, 2), "scale": round(c.scale, 4),
         "init_score": round(c.init_score, 4),
         "affine_score": None if c.affine_score is None else round(c.affine_score, 4)}
        for c in cands
    ]).to_csv(p("candidates.csv"), index=False)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def _validate(cfg: ElastixRegistrationConfig) -> None:
    for name in ("axes_moving", "axes_fixed"):
        if getattr(cfg, name) not in _SUPPORTED_AXES:
            raise ValueError(
                f"`{name}` must be one of {_SUPPORTED_AXES} for elastix registration, "
                f"got '{getattr(cfg, name)}'. For multichannel images select the registration "
                "channel first."
            )
    for name in ("pixel_size_moving", "pixel_size_fixed"):
        v = getattr(cfg, name)
        if v is None or not v > 0:
            raise ValueError(f"`{name}` (µm per level-0 pixel) is required and must be > 0, got {v}.")
    for name in ("init_resolution", "elastix_resolution", "angle_step", "scale_step",
                 "bspline_grid_spacing"):
        if not getattr(cfg, name) > 0:
            raise ValueError(f"`{name}` must be > 0, got {getattr(cfg, name)}.")
    lo, hi = cfg.scale_range
    if not 0 < lo <= hi:
        raise ValueError(f"`scale_range` must satisfy 0 < min <= max, got {cfg.scale_range}.")
    if cfg.top_k < 1:
        raise ValueError(f"`top_k` must be >= 1, got {cfg.top_k}.")
    if cfg.initial_transform is not None and np.asarray(cfg.initial_transform).shape not in [(2, 3), (3, 3)]:
        raise ValueError("`initial_transform` must be a 2x3 or 3x3 affine matrix.")


def register_images_elastix(
    moving,
    fixed,
    *,
    axes_moving: str = "YXS",
    axes_fixed: str = "YX",
    pixel_size_moving: float | None = None,
    pixel_size_fixed: float | None = None,
    nonrigid: bool = True,
    test_flipping: bool = True,
    initial_transform: np.ndarray | None = None,
    init_resolution: float = 8.0,
    init_smoothing: float = 120.0,
    angle_step: float = 4.0,
    scale_range: tuple[float, float] = (0.8, 1.3),
    scale_step: float = 0.05,
    top_k: int = 3,
    elastix_resolution: float = 4.0,
    representation_smoothing: float = 8.0,
    affine_resolutions: int = 4,
    affine_iterations: int = 500,
    bspline_grid_spacing: float = 300.0,
    bspline_resolutions: int = 3,
    bspline_iterations: int = 500,
    bspline_bending_weight: float = 1.0,
    random_seed: int = 0,
    parameter_maps: tuple[dict, ...] | None = None,
    tile_size: int = 4096,
    warp: bool = True,
    debug: bool = False,
    qc_dir: str | Path | None = None,
    qc_identifier: str = "registration",
    verbose: bool = True,
) -> tuple[np.ndarray | None, DisplacementTransform]:
    """Register a moving image to a fixed image at tissue scale using elastix.

    Designed for **consecutive sections** (e.g. an H&E of the neighbouring cut registered onto the
    Xenium DAPI image), where the feature-based :func:`register_images_standalone` fails because
    nuclei do not correspond between sections. The pipeline is:

    1. Nuclear-density representations and tissue masks at low resolution (hematoxylin for RGB
       brightfield images, log intensity for fluorescence).
    2. Global flip x rotation x scale search on heavily smoothed maps.
    3. elastix affine refinement (mutual information) of the ``top_k`` candidates; the best one
       by tissue overlap and structure correlation is kept.
    4. Optional elastix B-spline refinement (``nonrigid=True``) for folds, stretch and shrinkage.
    5. Tiled full-resolution warp of the moving image (``warp=True``).

    Expected precision is tissue-neighbourhood (tens of µm), not single-cell.

    Requires ``itk-elastix`` (``pip install "insitupy-spatial[registration]"``).

    Args:
        moving: Image to be registered: numpy or dask array, or a list of pyramid levels
            (highest resolution first). Pyramids are used to avoid loading level 0 for the
            low-resolution stages.
        fixed: Fixed reference image (same accepted types). Its level 0 is never loaded; only its
            shape is used.
        axes_moving: Axes of the moving image: ``'YXS'``/``'SYX'`` (RGB brightfield) or ``'YX'``
            (fluorescence channel).
        axes_fixed: Axes of the fixed image (same options).
        pixel_size_moving: Moving level-0 pixel size in µm (required).
        pixel_size_fixed: Fixed level-0 pixel size in µm (required).
        nonrigid: If True, add a B-spline stage after the affine stage.
        test_flipping: If True, also test a horizontally flipped moving image in the global search.
        initial_transform: Optional 2x3/3x3 affine (moving level-0 px -> fixed level-0 px). If
            given, the global search is skipped and this is the only starting point. Use it when
            the automatic search fails (e.g. from manually picked landmarks).
        init_resolution: Resolution (µm/px) of the global search refinement. The coarse grid runs
            at twice this value.
        init_smoothing: Gaussian sigma (µm) of the maps scored in the global search.
        angle_step: Rotation step (degrees) of the coarse grid.
        scale_range: ``(min, max)`` scale (moving -> fixed) tested in the global search.
        scale_step: Scale step of the coarse grid.
        top_k: Number of global-search candidates refined by elastix affine.
        elastix_resolution: Resolution (µm/px) of the images given to elastix.
        representation_smoothing: Gaussian sigma (µm) applied before elastix.
        affine_resolutions: Number of elastix pyramid levels for the affine stage.
        affine_iterations: Maximum iterations per level for the affine stage.
        bspline_grid_spacing: Final B-spline control point spacing in µm.
        bspline_resolutions: Number of elastix pyramid levels for the B-spline stage.
        bspline_iterations: Maximum iterations per level for the B-spline stage.
        bspline_bending_weight: Weight of the bending-energy penalty (higher = smoother).
        random_seed: elastix random seed (makes the result reproducible).
        parameter_maps: Expert override: elastix parameter maps (dicts of str -> value or list)
            replacing the default affine/B-spline maps.
        tile_size: Output tile edge length for the full-resolution warp.
        warp: If False, skip the full-resolution warp and return ``None`` as the image (useful
            when the transform is applied to other channels only).
        debug: If True, write QC images, the candidate table and the transform to ``qc_dir``.
        qc_dir: Directory for QC output. Defaults to ``<cwd>/registration_qc`` when ``debug=True``.
        qc_identifier: Prefix for QC file names.
        verbose: If True, log progress messages.

    Returns:
        Tuple of:
            - registered (np.ndarray or None): The warped moving image with the fixed level-0
              spatial shape (``None`` if ``warp=False``).
            - transform (DisplacementTransform): Dense transform plus QC ``metrics`` (tissue Dice,
              structure correlation, non-rigid displacement statistics, minimum relative Jacobian
              (<= 0 means local folding), chosen flip/angle/scale, candidate scores).

    Raises:
        MissingPackageError: If ``itk-elastix`` is not installed.
        ValueError: On invalid parameters, missing pixel sizes or failed tissue detection.
        RuntimeError: If elastix fails for all candidates.
    """
    cfg = ElastixRegistrationConfig(
        axes_moving=axes_moving, axes_fixed=axes_fixed,
        pixel_size_moving=pixel_size_moving, pixel_size_fixed=pixel_size_fixed,
        init_resolution=init_resolution, init_smoothing=init_smoothing, angle_step=angle_step,
        scale_range=tuple(scale_range), scale_step=scale_step, test_flipping=test_flipping,
        top_k=top_k, initial_transform=initial_transform,
        elastix_resolution=elastix_resolution, representation_smoothing=representation_smoothing,
        affine_resolutions=affine_resolutions, affine_iterations=affine_iterations,
        nonrigid=nonrigid, bspline_grid_spacing=bspline_grid_spacing,
        bspline_resolutions=bspline_resolutions, bspline_iterations=bspline_iterations,
        bspline_bending_weight=bspline_bending_weight, random_seed=random_seed,
        parameter_maps=parameter_maps, tile_size=tile_size, verbose=verbose,
    )
    _validate(cfg)
    itk = _import_itk()
    t_start = time.time()

    def info(msg, *args):
        if verbose:
            logger.info(msg, *args)

    # --- Stage 1+2: global initialisation ------------------------------------
    if cfg.initial_transform is not None:
        A = _to3(cfg.initial_transform)
        lin = A[:2, :2]
        flip = bool(np.linalg.det(lin) < 0)
        candidates = [_Candidate(A, flip, float("nan"), float(np.sqrt(abs(np.linalg.det(lin)))), float("nan"))]
        info("%s%s%s Global search skipped (initial_transform given)", _TSIGN, _HLINE, _HLINE)
    else:
        info("%s%s%s Global search (flip x rotation x scale)", _TSIGN, _HLINE, _HLINE)
        t0 = time.time()
        candidates = _global_search(moving, fixed, cfg)
        for c in candidates:
            info("%s     flip=%s angle=%.1f scale=%.3f score=%.3f", _VLINE, c.flip, c.angle, c.scale, c.init_score)
        info("%s     (%.1f s)", _VLINE, time.time() - t0)

    # --- Stage 3: elastix affine on each candidate ---------------------------
    fix_e = _prepare(fixed, cfg.axes_fixed, cfg.pixel_size_fixed, cfg.elastix_resolution, "fixed")
    mov_e = _prepare(moving, cfg.axes_moving, cfg.pixel_size_moving, cfg.elastix_resolution, "moving")
    fix_es = _smooth_norm(fix_e, cfg.representation_smoothing)
    mov_es = _smooth_norm(mov_e, cfg.representation_smoothing)
    if _MASK_BLEND > 0:
        sig = 50.0 / cfg.elastix_resolution
        fix_es = (1 - _MASK_BLEND) * fix_es + _MASK_BLEND * ndi.gaussian_filter(fix_e.mask.astype(np.float32), sig)
        mov_es = (1 - _MASK_BLEND) * mov_es + _MASK_BLEND * ndi.gaussian_filter(mov_e.mask.astype(np.float32), sig)
    fix_ss = _smooth_norm(fix_e, _SCORE_SMOOTHING_UM)
    mov_ss = _smooth_norm(mov_e, _SCORE_SMOOTHING_UM)
    elx_log = False  # elastix' own console log is very verbose; QC files cover debugging

    info("%s%s%s elastix affine (%d candidate%s)", _TSIGN, _HLINE, _HLINE,
         len(candidates), "" if len(candidates) == 1 else "s")
    t0 = time.time()
    best = None
    for c in candidates:
        M_lr = _lr_matrix(c.A, fix_e, mov_e)
        try:
            map_lr = _run_elastix(itk, fix_e, fix_es, mov_e, mov_es, M_lr, cfg, nonrigid=False, log=elx_log)
        except RuntimeError as exc:
            logger.warning("%s     elastix failed for candidate (flip=%s, angle=%.1f): %s",
                           _VLINE, c.flip, c.angle, str(exc).strip().splitlines()[-1] if str(exc) else exc)
            continue
        w, wm = _remap_lr(mov_ss, mov_e.mask, map_lr)
        c.affine_score = _score(fix_ss, fix_e.mask, w, wm)[0]
        info("%s     flip=%s angle=%.1f -> score %.3f", _VLINE, c.flip, c.angle, c.affine_score)
        if best is None or c.affine_score > best[0].affine_score:
            best = (c, M_lr, map_lr)
    if best is None:
        raise RuntimeError("elastix registration failed for all candidates. Run with debug=True for details.")
    winner, M_lr, map_aff = best
    info("%s     (%.1f s)", _VLINE, time.time() - t0)

    # --- Stage 4: B-spline ----------------------------------------------------
    if cfg.nonrigid:
        info("%s%s%s elastix B-spline (grid %.0f µm)", _TSIGN, _HLINE, _HLINE, cfg.bspline_grid_spacing)
        t0 = time.time()
        map_fin = _run_elastix(itk, fix_e, fix_es, mov_e, mov_es, M_lr, cfg, nonrigid=True, log=elx_log)
        info("%s     (%.1f s)", _VLINE, time.time() - t0)
    else:
        map_fin = map_aff

    # --- Metrics ----------------------------------------------------------------
    w_aff, wm_aff = _remap_lr(mov_ss, mov_e.mask, map_aff)
    w_fin, wm_fin = _remap_lr(mov_ss, mov_e.mask, map_fin)
    dice_aff = _dice(fix_e.mask, wm_aff)
    dice_fin = _dice(fix_e.mask, wm_fin)
    ncc_aff_full = _ncc(fix_ss, w_aff, fix_e.mask & wm_aff)
    ncc_fin_full = _ncc(fix_ss, w_fin, fix_e.mask & wm_fin)

    map_aff_l0 = _lr_map_to_l0(map_aff, mov_e)
    map_fin_l0 = _lr_map_to_l0(map_fin, mov_e)
    disp_um = np.hypot(*np.moveaxis(map_fin_l0 - map_aff_l0, -1, 0)) * cfg.pixel_size_moving
    in_mask = disp_um[fix_e.mask]
    min_jac = _min_relative_jacobian(map_fin, map_aff, fix_e.mask) if cfg.nonrigid else 1.0

    metrics = {
        "method": "elastix",
        "nonrigid": bool(cfg.nonrigid),
        "chosen_flip": bool(winner.flip),
        "chosen_angle_deg": float(winner.angle),
        "chosen_scale": float(winner.scale),
        "tissue_dice_affine": float(dice_aff),
        "tissue_dice": float(dice_fin),
        "structure_ncc_affine": float(ncc_aff_full),
        "structure_ncc": float(ncc_fin_full),
        "nonrigid_displacement_median_um": float(np.median(in_mask)) if in_mask.size else 0.0,
        "nonrigid_displacement_p95_um": float(np.percentile(in_mask, 95)) if in_mask.size else 0.0,
        "min_relative_jacobian": float(min_jac),
        "candidates": [
            {"flip": bool(c.flip), "angle_deg": float(c.angle), "scale": float(c.scale),
             "init_score": float(c.init_score),
             "affine_score": None if c.affine_score is None else float(c.affine_score)}
            for c in candidates
        ],
        "elastix_resolution_um": float(cfg.elastix_resolution),
        "precision_note": _PRECISION_NOTE,
    }
    for k, v in list(metrics.items()):
        if isinstance(v, float) and not np.isfinite(v):
            metrics[k] = None

    transform = DisplacementTransform(
        map_xy=map_fin_l0.astype(np.float32),
        grid_f=(float(fix_e.fy), float(fix_e.fx)),
        fixed_shape=_hw(_as_level_list(fixed)[0], cfg.axes_fixed),
        moving_shape=_hw(_as_level_list(moving)[0], cfg.axes_moving),
        pixel_size_fixed=float(cfg.pixel_size_fixed),
        pixel_size_moving=float(cfg.pixel_size_moving),
        affine=_fit_affine(map_aff_l0, fix_e),
        metrics=metrics,
    )
    info("%s     tissue Dice %.3f (affine %.3f), structure NCC %.3f (affine %.3f)",
         _VLINE, dice_fin, dice_aff, ncc_fin_full, ncc_aff_full)
    if cfg.nonrigid:
        info("%s     non-rigid displacement median %.0f µm, p95 %.0f µm, min rel. Jacobian %.2f",
             _VLINE, metrics["nonrigid_displacement_median_um"], metrics["nonrigid_displacement_p95_um"], min_jac)

    # --- QC ---------------------------------------------------------------------
    if debug:
        resolved = Path.cwd() / "registration_qc" if qc_dir is None else Path(qc_dir)
        if qc_dir is None:
            logger.info("QC directory (auto-resolved): %s", resolved.resolve())
        _write_qc(resolved, qc_identifier, fix_es, mov_es, fix_e, mov_e, M_lr,
                  map_aff, map_fin, disp_um, candidates)
        transform.save(resolved / f"{qc_identifier}__elastix_transform.npz")

    # --- Stage 5: full-resolution warp ---------------------------------------
    registered = None
    if warp:
        info("%s%s%s Warping full-resolution image (tiles of %d px)", _TSIGN, _HLINE, _HLINE, cfg.tile_size)
        axes_warp = "YXS" if cfg.axes_moving == "SYX" else cfg.axes_moving
        mov0 = _as_level_list(moving)[0]
        if cfg.axes_moving == "SYX":
            mov0 = da.moveaxis(mov0, 0, -1) if isinstance(mov0, da.Array) else np.moveaxis(mov0, 0, -1)
        registered = apply_displacement_transform(mov0, transform, axes_warp, tile_size=cfg.tile_size)

    info("%s%s%s Done (%.1f s)", _LSIGN, _HLINE, _HLINE, time.time() - t_start)
    return registered, transform


def config_to_kwargs(config: ElastixRegistrationConfig | dict | None) -> dict:
    """Convert an :class:`ElastixRegistrationConfig` (or dict of its fields) to keyword arguments
    of :func:`register_images_elastix`. Unknown keys raise ``TypeError``."""
    if config is None:
        return {}
    if isinstance(config, ElastixRegistrationConfig):
        return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    if isinstance(config, dict):
        names = {f.name for f in dataclasses.fields(ElastixRegistrationConfig)}
        unknown = set(config) - names
        if unknown:
            raise TypeError(f"Unknown elastix_config keys: {sorted(unknown)}. Valid keys: {sorted(names)}.")
        return dict(config)
    raise TypeError(f"`elastix_config` must be an ElastixRegistrationConfig, dict or None, got {type(config)}.")

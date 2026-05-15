"""Standalone image registration using feature-based alignment.

Public API:
    register_images_standalone  — orchestrator function
    RegistrationConfig          — frozen config dataclass
    ScaledImages                — result of loading/scaling stage
    FeatureMatchResult          — result of feature extraction stage
    TransformResult             — result of transformation estimation stage
    save_registration_qc        — write QC files to disk
    save_registered_image_tiff  — write registered image as OME-TIFF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

import dask.array as da
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from insitupy._constants import SHRT_MAX
from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy.images.axes import ImageAxes, get_height_and_width
from insitupy.images.io import write_ome_tiff
from insitupy.images.utils import (
    clip_image_histogram,
    convert_to_8bit_func,
    deconvolve_he,
    fit_image_to_size_limit,
    otsu_thresholding,
    resize_image,
    scale_to_max_width,
)
from insitupy.images.warp import apply_warp
from insitupy.utils.utils import remove_last_line_from_csv

logger = logging.getLogger(__name__)

# Tree drawing characters
_TSIGN = "\u251c"   # ├
_LSIGN = "\u2514"   # └
_VLINE = "\u2502"   # │
_HLINE = "\u2500"   # ─
_TICK  = "\u2714"   # ✔


# ---------------------------------------------------------------------------
# Private helper: percentile scaling for visualization
# ---------------------------------------------------------------------------

def _percentile_scale_for_saving(img: np.ndarray, upper_percentile: float = 99.0) -> np.ndarray:
    """Clip to upper percentile and scale intensities to [0, 1] for visualization export."""
    arr = np.asarray(img)
    if arr.size == 0:
        return arr

    def _scale_channel(channel: np.ndarray) -> np.ndarray:
        c = channel.astype(np.float32, copy=False)
        c_min = np.nanmin(c)
        c_max = np.nanpercentile(c, upper_percentile)
        if not np.isfinite(c_min) or not np.isfinite(c_max) or c_max <= c_min:
            return np.zeros_like(c, dtype=np.float32)
        c = np.clip(c, c_min, c_max)
        return (c - c_min) / (c_max - c_min)

    if arr.ndim == 2:
        return _scale_channel(arr)

    if arr.ndim == 3:
        scaled = np.empty(arr.shape, dtype=np.float32)
        for channel_idx in range(arr.shape[2]):
            scaled[..., channel_idx] = _scale_channel(arr[..., channel_idx])
        return scaled

    return arr


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegistrationConfig:
    """Immutable configuration for image registration."""
    axes_moving: str = "YXS"
    axes_fixed: str = "YX"
    max_width: int | None = 4000
    convert_to_grayscale: bool = False
    deconvolve_moving: bool = False
    deconvolve_fixed: bool = False
    decon_scale_factor: float = 0.2
    perspective_transform: bool = False
    feature_detection_method: Literal["sift", "surf"] = "sift"
    flann: bool = True
    mutual_nearest_neighbor: bool = False
    ratio_test: bool = True
    keep_fraction: float = 0.2
    max_features: int = 500
    min_good_matches: int = 20
    test_flipping: bool = True
    adjust_contrast_method: Literal["otsu", "clip"] | None = "clip"
    verbose: bool = True
    # QC metadata (not algorithm parameters, but needed for detail figure scaling)
    pixel_size_moving: float | None = None
    pixel_size_fixed: float | None = None
    physical_size_unit: str = "µm"


@dataclass
class ScaledImages:
    """Result of the image loading and scaling stage."""
    # Scaled 8-bit images for feature detection
    moving_scaled: np.ndarray       # 8-bit downscaled moving image (for feature detection)
    fixed_scaled: np.ndarray        # 8-bit downscaled fixed image (for feature detection)

    # Scale factors (scaled → original space)
    x_sf_moving: float
    y_sf_moving: float
    x_sf_fixed: float
    y_sf_fixed: float

    # Original image dimensions
    moving_shape: tuple
    fixed_shape: tuple

    # Fixed image dimensions (height, width)
    fixed_h: int
    fixed_w: int

    # Full-resolution moving image for warping.
    # IMPORTANT: this is always the ORIGINAL (non-deconvolved, non-grayscale-converted)
    # moving image, regardless of deconvolve_moving / convert_to_grayscale settings.
    # Deconvolution and grayscale conversion only affect moving_scaled (feature detection).
    moving_for_warp: np.ndarray     # full-res or SHRT_MAX-resized original moving image
    resize_factor_moving: float     # 1.0 if no resize was needed

    # Effective axes of moving_scaled / fixed_scaled after deconvolution/grayscale conversion.
    # Used for feature detection only, NOT for warping.
    # For warping, use config.axes_moving (the original axes of moving_for_warp).
    axes_moving_effective: str
    axes_fixed_effective: str

    # Full-resolution deconvolved images for QC output (only set when deconvolve_moving/fixed=True).
    moving_deconvolved: np.ndarray | None = None
    fixed_deconvolved: np.ndarray | None = None


@dataclass
class FeatureMatchResult:
    """Result of the feature extraction stage."""
    kpsA: list                  # Keypoints in moving image (cv2.KeyPoint list)
    kpsB: list                  # Keypoints in fixed image (cv2.KeyPoint list)
    good_matches: list          # Filtered feature matches (cv2.DMatch list)
    ptsA: np.ndarray            # Matched point coordinates in moving image (N×2, original scale)
    ptsB: np.ndarray            # Matched point coordinates in fixed image (N×2, original scale)
    flip_axis: int | None    # Axis used for flipping (0=vertical, None=no flip)
    matchedVis: np.ndarray | None  # cv2.drawMatches visualization (or None)


@dataclass
class TransformResult:
    """Result of the transformation matrix estimation stage."""
    T: np.ndarray                           # Transformation matrix in original coordinate space (2×3 or 3×3)
    T_for_warp: np.ndarray                  # Matrix for actual warping (may differ if image was resized)
    inlier_mask: np.ndarray | None       # Boolean RANSAC inlier mask (aligned with good_matches)
    ptsA_for_warp: np.ndarray               # Points scaled for warp-space (used internally)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _deconvolve_he_image(
    img: np.ndarray,
    axes: str,
    decon_scale_factor: float,
    name: str = "image",
) -> np.ndarray:
    """Apply H&E color deconvolution to extract nuclei channel."""
    logger.info("%s%s%s Color deconvolution (%s, scale factor: %s)", _TSIGN, _HLINE, _HLINE, name, decon_scale_factor)
    nuclei_img, _, _ = deconvolve_he(
        img=resize_image(img, scale_factor=decon_scale_factor, axes=axes),
        return_type="grayscale",
        convert=True,
    )
    nuclei_img = resize_image(nuclei_img, scale_factor=1 / decon_scale_factor, axes="YX")
    return nuclei_img


def _enforce_unique_match_pairs(
    good_matches: list,
    ordered_match_indices: list[int],
) -> list[int]:
    """Keep one-to-one unique queryIdx/trainIdx pairs while preserving input order."""
    selected = []
    used_query = set()
    used_train = set()
    for idx in ordered_match_indices:
        m = good_matches[idx]
        if m.queryIdx in used_query or m.trainIdx in used_train:
            continue
        selected.append(idx)
        used_query.add(m.queryIdx)
        used_train.add(m.trainIdx)
    return selected


def _rank_match_indices_for_qc(
    features: FeatureMatchResult,
    scaled: ScaledImages,
    transform: TransformResult | None,
    config: RegistrationConfig,
    patch_half_size: int = 16,
) -> list[int]:
    """Rank good match indices by robust QC quality score."""
    good_matches = features.good_matches
    n_total = len(good_matches)

    if n_total == 0:
        return []

    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img.astype(np.float32)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return img.mean(axis=-1).astype(np.float32)

    def _extract_patch(img: np.ndarray, x: float, y: float, half: int) -> np.ndarray | None:
        cx = int(round(x))
        cy = int(round(y))
        x0, x1 = cx - half, cx + half + 1
        y0, y1 = cy - half, cy + half + 1
        h, w = img.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        return img[y0:y1, x0:x1]

    def _zncc(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return -2.0
        aa = a.astype(np.float32)
        bb = b.astype(np.float32)
        aa = aa - np.mean(aa)
        bb = bb - np.mean(bb)
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 1e-8:
            return -2.0
        return float(np.sum(aa * bb) / denom)

    has_transform = transform is not None and transform.T is not None
    has_inlier_mask = (
        has_transform
        and transform.inlier_mask is not None
        and len(transform.inlier_mask) == n_total
    )

    reproj_error_by_idx = {}
    if has_transform and len(features.ptsA) == n_total and len(features.ptsB) == n_total:
        ptsA = features.ptsA.astype(np.float32)
        if config.perspective_transform:
            projected = cv2.perspectiveTransform(ptsA.reshape(-1, 1, 2), transform.T).reshape(-1, 2)
        else:
            projected = cv2.transform(ptsA.reshape(-1, 1, 2), transform.T).reshape(-1, 2)
        errors = np.linalg.norm(projected - features.ptsB, axis=1)
        reproj_error_by_idx = {idx: float(err) for idx, err in enumerate(errors)}

    img_gray = _to_gray(scaled.moving_scaled)
    tmpl_gray = _to_gray(scaled.fixed_scaled)
    ncc_by_idx = {}
    for idx, match in enumerate(good_matches):
        ptA = features.kpsA[match.queryIdx].pt
        ptB = features.kpsB[match.trainIdx].pt
        patch_a = _extract_patch(img_gray, ptA[0], ptA[1], patch_half_size)
        patch_b = _extract_patch(tmpl_gray, ptB[0], ptB[1], patch_half_size)
        ncc_by_idx[idx] = _zncc(patch_a, patch_b)

    def _rank_key(idx: int) -> tuple:
        inlier_rank = 0 if (has_inlier_mask and bool(transform.inlier_mask[idx])) else (1 if has_inlier_mask else 0)
        reproj_rank = reproj_error_by_idx.get(idx, np.inf)
        ncc_rank = -ncc_by_idx.get(idx, -2.0)
        dist_rank = float(good_matches[idx].distance)
        if has_transform:
            return (inlier_rank, reproj_rank, ncc_rank, dist_rank)
        return (ncc_rank, dist_rank)

    ranked = sorted(range(n_total), key=_rank_key)
    selected = _enforce_unique_match_pairs(good_matches, ranked)

    # Fallback if uniqueness filtering became too strict.
    if len(selected) == 0:
        selected = ranked

    return selected


def _create_match_figure(
    scaled: ScaledImages,
    features: FeatureMatchResult,
    transform: TransformResult | None,
    config: RegistrationConfig,
    topn: int,
    rank_matches_for_qc: bool = True,
    ranked_idx: list[int] | None = None,
    figsize: tuple = (16, 8),
    create_detail_figure: bool = True,
    detail_window_um: float = 200.0,
) -> tuple[plt.Figure, plt.Figure | None]:
    """Create a figure showing top-N matched keypoints between moving and fixed images."""
    kpsA = features.kpsA
    kpsB = features.kpsB
    good_matches = features.good_matches
    n_total = len(good_matches)
    n_display = min(topn, n_total)

    if ranked_idx is None:
        if rank_matches_for_qc:
            ranked_idx = _rank_match_indices_for_qc(features, scaled, transform, config)
        else:
            ranked_idx = list(range(n_total))
    rank_by_match_idx = {match_idx: rank for rank, match_idx in enumerate(ranked_idx, start=1)}

    def _create_top5_detail_figure() -> plt.Figure:
        """Create 5x2 zoomed view for globally ranked matches 1-5."""
        n_rows = 5
        fig_detail, axes = plt.subplots(n_rows, 2, figsize=(12, 3.0 * n_rows))

        unit = str(getattr(config, "physical_size_unit", "µm")).strip().lower().replace("μ", "µ")
        is_um_unit = unit in {"µm", "um", "micrometer", "micrometre", "micron", "microns"}

        px_um_img = config.pixel_size_moving
        px_um_tmpl = config.pixel_size_fixed
        if px_um_img is None:
            px_um_img = px_um_tmpl
        if px_um_tmpl is None:
            px_um_tmpl = px_um_img

        def _half_window_scaled(sf: float, pixel_size_um: float | None) -> int:
            fallback_half_original = detail_window_um / 2.0
            if is_um_unit and pixel_size_um is not None and float(pixel_size_um) > 0:
                half_window_original = (detail_window_um / float(pixel_size_um)) / 2.0
            else:
                half_window_original = fallback_half_original
            return max(1, int(np.ceil(half_window_original * sf)))

        def _fixed_window_bounds(center: float, half_window: int, max_size: int) -> tuple[float, float]:
            """Return bounds with fixed width (2*half_window) whenever possible within image limits."""
            if max_size <= 1:
                return 0.0, 0.0

            desired_width = float(2 * half_window)
            available_width = float(max_size - 1)
            width = min(desired_width, available_width)

            x0 = float(center) - width / 2.0
            x1 = float(center) + width / 2.0

            if x0 < 0.0:
                x1 -= x0
                x0 = 0.0
            if x1 > available_width:
                x0 -= (x1 - available_width)
                x1 = available_width
            x0 = max(0.0, x0)
            x1 = min(available_width, x1)
            return x0, x1

        top5_idx = ranked_idx[:min(5, n_total)]
        img_h, img_w = scaled.moving_scaled.shape[:2]
        tmpl_h, tmpl_w = scaled.fixed_scaled.shape[:2]
        image_scaled_disp = _percentile_scale_for_saving(scaled.moving_scaled)
        template_scaled_disp = _percentile_scale_for_saving(scaled.fixed_scaled)

        for row in range(n_rows):
            ax_img_row, ax_tmpl_row = axes[row, 0], axes[row, 1]
            ax_img_row.imshow(image_scaled_disp, cmap="gray" if scaled.moving_scaled.ndim == 2 else None)
            ax_tmpl_row.imshow(template_scaled_disp, cmap="gray" if scaled.fixed_scaled.ndim == 2 else None)
            ax_img_row.axis("off")
            ax_tmpl_row.axis("off")

            if row >= len(top5_idx):
                ax_img_row.set_title(f"Target - rank {row + 1} (not available)")
                ax_tmpl_row.set_title(f"Template - rank {row + 1} (not available)")
                continue

            match_i = top5_idx[row]
            m = good_matches[match_i]
            ptA = kpsA[m.queryIdx].pt
            ptB = kpsB[m.trainIdx].pt
            rank_label = rank_by_match_idx[match_i]

            half_w_img = _half_window_scaled(scaled.x_sf_moving, px_um_img)
            half_h_img = _half_window_scaled(scaled.y_sf_moving, px_um_img)
            half_w_tmpl = _half_window_scaled(scaled.x_sf_fixed, px_um_tmpl)
            half_h_tmpl = _half_window_scaled(scaled.y_sf_fixed, px_um_tmpl)

            x0_img, x1_img = _fixed_window_bounds(ptA[0], half_w_img, img_w)
            y0_img, y1_img = _fixed_window_bounds(ptA[1], half_h_img, img_h)

            x0_tmpl, x1_tmpl = _fixed_window_bounds(ptB[0], half_w_tmpl, tmpl_w)
            y0_tmpl, y1_tmpl = _fixed_window_bounds(ptB[1], half_h_tmpl, tmpl_h)

            ax_img_row.set_xlim(x0_img, x1_img)
            ax_img_row.set_ylim(y1_img, y0_img)
            ax_tmpl_row.set_xlim(x0_tmpl, x1_tmpl)
            ax_tmpl_row.set_ylim(y1_tmpl, y0_tmpl)

            ax_img_row.plot(ptA[0], ptA[1], marker="o", markersize=10,
                            markerfacecolor="none", markeredgecolor="yellow", markeredgewidth=1.1)
            ax_tmpl_row.plot(ptB[0], ptB[1], marker="o", markersize=10,
                             markerfacecolor="none", markeredgecolor="yellow", markeredgewidth=1.1)

            ax_img_row.set_title(f"Target - rank {rank_label}")
            ax_tmpl_row.set_title(f"Template - rank {rank_label}")

        fig_detail.suptitle("Top-5 matches detailed view (200 µm window)", y=0.995)
        fig_detail.tight_layout(rect=[0, 0, 1, 0.985])
        return fig_detail

    if n_display == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No matches available", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        detail_fig = _create_top5_detail_figure() if create_detail_figure else None
        return fig, detail_fig

    subset_idx = ranked_idx[:n_display]

    fig, (ax_img, ax_tmpl) = plt.subplots(1, 2, figsize=figsize)

    image_scaled_disp = _percentile_scale_for_saving(scaled.moving_scaled)
    template_scaled_disp = _percentile_scale_for_saving(scaled.fixed_scaled)
    ax_img.imshow(image_scaled_disp, cmap="gray" if scaled.moving_scaled.ndim == 2 else None)
    ax_tmpl.imshow(template_scaled_disp, cmap="gray" if scaled.fixed_scaled.ndim == 2 else None)
    has_transform = transform is not None and transform.T is not None
    if not rank_matches_for_qc:
        ranking_text = "original order"
    elif has_transform:
        ranking_text = "ranked by inlier+reprojection"
    else:
        ranking_text = "ranked by NCC+distance"
    ax_img.set_title(f"Target \u2013 top {n_display} of {n_total} matches ({ranking_text})")
    ax_tmpl.set_title("Template")
    ax_img.axis("off")
    ax_tmpl.axis("off")

    cmap = plt.colormaps.get_cmap("tab20")
    n_colors = 20
    for plot_i, match_i in enumerate(subset_idx):
        m = good_matches[match_i]
        ptA = kpsA[m.queryIdx].pt
        ptB = kpsB[m.trainIdx].pt
        color = cmap(plot_i % n_colors)
        rank_label = rank_by_match_idx[match_i]

        ax_img.plot(*ptA, "o", color=color, markersize=5,
                    markeredgewidth=0.5, markeredgecolor="white")
        ax_img.annotate(str(rank_label), ptA, color=color, fontsize=5,
                        xytext=(3, 3), textcoords="offset points")
        ax_tmpl.plot(*ptB, "o", color=color, markersize=5,
                     markeredgewidth=0.5, markeredgecolor="white")
        ax_tmpl.annotate(str(rank_label), ptB, color=color, fontsize=5,
                         xytext=(3, 3), textcoords="offset points")

        con = ConnectionPatch(
            xyA=ptA, xyB=ptB,
            coordsA="data", coordsB="data",
            axesA=ax_img, axesB=ax_tmpl,
            color=color, linewidth=0.5, alpha=0.6
        )
        con.set_clip_on(True)
        fig.add_artist(con)

    plt.tight_layout()
    detail_fig = _create_top5_detail_figure() if create_detail_figure else None
    return fig, detail_fig


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def load_and_scale_images(
    moving: np.ndarray | da.Array,
    fixed: np.ndarray | da.Array,
    config: RegistrationConfig,
) -> ScaledImages:
    """Load images into memory, apply preprocessing, and scale to max_width.

    The original (non-deconvolved, non-grayscale-converted) moving image is
    preserved as ``ScaledImages.moving_for_warp``. Deconvolution and grayscale
    conversion only affect ``moving_scaled`` (used for feature detection).

    Args:
        moving: Moving image (numpy or dask array).
        fixed: Fixed reference image (numpy or dask array).
        config: Registration configuration.

    Returns:
        ScaledImages dataclass.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If deconvolution is requested for non-RGB images.
    """
    if not HAS_OPENCV:
        raise ImportError(
            "OpenCV (cv2) is required for image registration. "
            "Install it with: pip install opencv-python"
        )

    verbose = config.verbose

    # --- Compute dask arrays to numpy ---
    if isinstance(moving, da.Array):
        if verbose:
            logger.info("%s%s%s Loading images into memory", _TSIGN, _HLINE, _HLINE)
        moving_np = moving.compute()
    else:
        moving_np = np.asarray(moving)

    if isinstance(fixed, da.Array):
        fixed_np = fixed.compute()
    else:
        fixed_np = np.asarray(fixed)

    # --- Preprocessing for moving_scaled (feature detection copy) ---
    axes_moving_effective = config.axes_moving
    moving_deconvolved = None
    if config.deconvolve_moving:
        if config.axes_moving not in ["YXS", "SYX"]:
            raise ValueError(
                f"HE deconvolution requires RGB image with axes 'YXS' or 'SYX', "
                f"got '{config.axes_moving}'"
            )
        moving_for_feature = _deconvolve_he_image(
            moving_np, config.axes_moving, config.decon_scale_factor, name="moving"
        )
        moving_deconvolved = moving_for_feature
        axes_moving_effective = "YX"
    else:
        moving_for_feature = moving_np

    if config.convert_to_grayscale and len(moving_for_feature.shape) == 3:
        moving_for_feature = cv2.cvtColor(moving_for_feature, cv2.COLOR_BGR2GRAY)
        axes_moving_effective = "YX"

    # --- Preprocessing for fixed_scaled ---
    axes_fixed_effective = config.axes_fixed
    fixed_deconvolved = None
    if config.deconvolve_fixed:
        if config.axes_fixed not in ["YXS", "SYX"]:
            raise ValueError(
                f"HE deconvolution requires RGB template with axes 'YXS' or 'SYX', "
                f"got '{config.axes_fixed}'"
            )
        fixed_for_feature = _deconvolve_he_image(
            fixed_np, config.axes_fixed, config.decon_scale_factor, name="fixed"
        )
        fixed_deconvolved = fixed_for_feature
        axes_fixed_effective = "YX"
    else:
        fixed_for_feature = fixed_np

    if config.convert_to_grayscale and len(fixed_for_feature.shape) == 3:
        fixed_for_feature = cv2.cvtColor(fixed_for_feature, cv2.COLOR_BGR2GRAY)
        axes_fixed_effective = "YX"

    # --- Scale to max_width ---
    if config.max_width is not None:
        if verbose:
            logger.info("%s%s%s Scaling", _TSIGN, _HLINE, _HLINE)
        moving_scaled = scale_to_max_width(
            moving_for_feature,
            axes=axes_moving_effective,
            max_width=config.max_width,
            use_square_area=True,
            verbose=config.verbose,
            print_spacer=f"{_VLINE}     Moving:  ",
        )
        fixed_scaled = scale_to_max_width(
            fixed_for_feature,
            axes=axes_fixed_effective,
            max_width=config.max_width,
            use_square_area=True,
            verbose=config.verbose,
            print_spacer=f"{_VLINE}     Fixed:   ",
        )
    else:
        moving_scaled = moving_for_feature
        fixed_scaled = fixed_for_feature

    # --- Convert to 8-bit ---
    moving_scaled = convert_to_8bit_func(moving_scaled)
    fixed_scaled = convert_to_8bit_func(fixed_scaled)

    # --- Scale factors (scaled → original space) ---
    h_orig_moving, w_orig_moving = get_height_and_width(moving_np, ImageAxes(config.axes_moving))
    x_sf_moving = moving_scaled.shape[1] / w_orig_moving
    y_sf_moving = moving_scaled.shape[0] / h_orig_moving

    h_orig_fixed, w_orig_fixed = get_height_and_width(fixed_np, ImageAxes(config.axes_fixed))
    x_sf_fixed = fixed_scaled.shape[1] / w_orig_fixed
    y_sf_fixed = fixed_scaled.shape[0] / h_orig_fixed

    # --- Fixed image dimensions ---
    fixed_h, fixed_w = get_height_and_width(fixed_np, ImageAxes(config.axes_fixed))

    # --- SHRT_MAX resize for moving (apply to original, not deconvolved) ---
    h_moving, w_moving = get_height_and_width(moving_np, ImageAxes(config.axes_moving))
    if max(h_moving, w_moving) > SHRT_MAX:
        if verbose:
            logger.info(
                "%s%s%s Warning: dimensions %s exceed SHRT_MAX (%s). Resizing.",
                _TSIGN, _HLINE, _HLINE, moving_np.shape, SHRT_MAX,
            )
        moving_for_warp, resize_factor_moving = fit_image_to_size_limit(
            moving_np,
            size_limit=SHRT_MAX,
            return_scale_factor=True,
            axes=config.axes_moving,
        )
        if verbose:
            logger.info(
                "%s     Resized to %s (factor: %.3f)",
                _VLINE, moving_for_warp.shape, resize_factor_moving,
            )
    else:
        moving_for_warp = moving_np
        resize_factor_moving = 1.0

    return ScaledImages(
        moving_scaled=moving_scaled,
        fixed_scaled=fixed_scaled,
        x_sf_moving=x_sf_moving,
        y_sf_moving=y_sf_moving,
        x_sf_fixed=x_sf_fixed,
        y_sf_fixed=y_sf_fixed,
        moving_shape=moving_np.shape,
        fixed_shape=fixed_np.shape,
        fixed_h=fixed_h,
        fixed_w=fixed_w,
        moving_for_warp=moving_for_warp,
        resize_factor_moving=resize_factor_moving,
        axes_moving_effective=axes_moving_effective,
        axes_fixed_effective=axes_fixed_effective,
        moving_deconvolved=moving_deconvolved,
        fixed_deconvolved=fixed_deconvolved,
    )


def extract_features(
    scaled: ScaledImages,
    config: RegistrationConfig,
    *,
    save_matched_vis: bool = True,
    force_failure: bool = False,
) -> FeatureMatchResult:
    """Extract and match features between moving and fixed images.

    Args:
        scaled: ScaledImages from the loading stage.
        config: Registration configuration.
        save_matched_vis: If True, store a cv2.drawMatches visualization.
        force_failure: If True, simulate a failure even when sufficient matches are found.

    Returns:
        FeatureMatchResult dataclass.

    Raises:
        NotEnoughFeatureMatchesError: With partial_result attached if insufficient matches found.
    """
    if not HAS_OPENCV:
        raise ImportError(
            "OpenCV (cv2) is required for image registration. "
            "Install it with: pip install opencv-python"
        )

    verbose = config.verbose
    method_name = config.feature_detection_method.upper()
    contrast_info = f", contrast: {config.adjust_contrast_method}" if config.adjust_contrast_method else ""
    if verbose:
        logger.info("%s%s%s Feature extraction (%s%s)", _TSIGN, _HLINE, _HLINE, method_name, contrast_info)

    flip_axis_list = [None, 0] if config.test_flipping else [None]

    matches_list = []
    best_good_matches, best_kpsA, best_kpsB = [], None, None
    best_flip_axis = None
    # Work on a copy of moving_scaled so we never mutate the input ScaledImages
    moving_work = scaled.moving_scaled.copy()
    winning_kpsA = winning_kpsB = winning_good_matches = None
    winning_flip_axis = None

    for flip_axis in flip_axis_list:
        if flip_axis is not None:
            flip_dir = "vertical" if flip_axis == 0 else "horizontal"
            if verbose:
                logger.info("%s     Testing %s flip", _VLINE, flip_dir)
            moving_work = np.flip(scaled.moving_scaled, axis=flip_axis).copy()
        else:
            moving_work = scaled.moving_scaled.copy()

        # --- Contrast adjustment ---
        if config.adjust_contrast_method is not None:
            if config.adjust_contrast_method == "otsu":
                image_contrast_adj = otsu_thresholding(image=convert_to_8bit_func(moving_work))
                template_contrast_adj = otsu_thresholding(image=convert_to_8bit_func(scaled.fixed_scaled))
            elif config.adjust_contrast_method == "clip":
                image_contrast_adj = clip_image_histogram(image=moving_work, lower_perc=20, upper_perc=99)
                template_contrast_adj = clip_image_histogram(image=scaled.fixed_scaled, lower_perc=20, upper_perc=99)
            else:
                raise ValueError(
                    f"Invalid method '{config.adjust_contrast_method}' for `adjust_contrast_method`."
                )
        else:
            image_contrast_adj = moving_work
            template_contrast_adj = scaled.fixed_scaled

        # --- Feature detection ---
        if config.feature_detection_method == "sift":
            sift = cv2.SIFT_create()
            kpsA, descsA = sift.detectAndCompute(image_contrast_adj, None)
            kpsB, descsB = sift.detectAndCompute(template_contrast_adj, None)
        elif config.feature_detection_method == "surf":
            surf = cv2.xfeatures2d.SURF_create(400)
            kpsA, descsA = surf.detectAndCompute(image_contrast_adj, None)
            kpsB, descsB = surf.detectAndCompute(template_contrast_adj, None)
        else:
            raise ValueError(f"Unknown feature detection method '{config.feature_detection_method}'.")

        # --- Feature matching ---
        if config.flann:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            fl = cv2.FlannBasedMatcher(index_params, search_params)
            matches = fl.knnMatch(descsA, descsB, k=2)
            if config.mutual_nearest_neighbor:
                fl_rev = cv2.FlannBasedMatcher(index_params, search_params)
                reverse_matches = fl_rev.knnMatch(descsB, descsA, k=1)
        else:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descsA, descsB, k=2)
            if config.mutual_nearest_neighbor:
                reverse_matches = bf.knnMatch(descsB, descsA, k=1)

        # --- Filter matches ---
        if config.ratio_test:
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        else:
            matches = sorted(matches, key=lambda x: x[0].distance)
            keep = int(len(matches) * config.keep_fraction)
            good_matches = [m[0] for m in matches[:keep]][:config.max_features]

        if config.mutual_nearest_neighbor:
            reverse_best = {}
            for rev in reverse_matches:
                if len(rev) == 0:
                    continue
                m_rev = rev[0]
                reverse_best[m_rev.queryIdx] = m_rev.trainIdx
            n_before = len(good_matches)
            good_matches = [
                m for m in good_matches
                if reverse_best.get(m.trainIdx, None) == m.queryIdx
            ]
            if verbose:
                logger.info("%s     Mutual NN filter: %d / %d kept", _VLINE, len(good_matches), n_before)

        matches_list.append(len(good_matches))
        # Track best result for failure diagnostics
        if len(good_matches) > len(best_good_matches):
            best_good_matches = good_matches
            best_kpsA = kpsA
            best_kpsB = kpsB
            best_flip_axis = flip_axis
            # Keep a copy of moving_work that corresponds to the best flip state
            best_moving_work = moving_work.copy()

        if len(good_matches) >= config.min_good_matches and not force_failure:
            if verbose:
                logger.info(
                    "%s     Good matches: %d / %d required  %s",
                    _VLINE, len(good_matches), config.min_good_matches, _TICK,
                )
            winning_kpsA = kpsA
            winning_kpsB = kpsB
            winning_good_matches = good_matches
            winning_flip_axis = flip_axis
            winning_moving_work = moving_work
            break
        else:
            if force_failure and len(good_matches) >= config.min_good_matches:
                if verbose:
                    logger.info(
                        "%s     Good matches: %d / %d required  %s (force_failure=True, simulating failure)",
                        _VLINE, len(good_matches), config.min_good_matches, _TICK,
                    )
            else:
                if verbose:
                    logger.info(
                        "%s     Good matches: %d / %d required (insufficient, testing flip)",
                        _VLINE, len(good_matches), config.min_good_matches,
                    )

    if winning_good_matches is None:
        # Build partial result for failure diagnostics
        partial_result = None
        if best_kpsA is not None:
            matched_vis = None
            if save_matched_vis:
                matched_vis = cv2.drawMatches(
                    best_moving_work, best_kpsA,
                    scaled.fixed_scaled, best_kpsB,
                    best_good_matches, None,
                )
            # Build partial ptsA/ptsB in original space for consistency
            n_best = len(best_good_matches)
            ptsA_partial = np.zeros((n_best, 2), dtype="float")
            ptsB_partial = np.zeros((n_best, 2), dtype="float")
            for i, m in enumerate(best_good_matches):
                ptsA_partial[i] = best_kpsA[m.queryIdx].pt
                ptsB_partial[i] = best_kpsB[m.trainIdx].pt
            ptsA_partial[:, 0] /= scaled.x_sf_moving
            ptsA_partial[:, 1] /= scaled.y_sf_moving
            ptsB_partial[:, 0] /= scaled.x_sf_fixed
            ptsB_partial[:, 1] /= scaled.y_sf_fixed
            partial_result = FeatureMatchResult(
                kpsA=best_kpsA,
                kpsB=best_kpsB,
                good_matches=best_good_matches,
                ptsA=ptsA_partial,
                ptsB=ptsB_partial,
                flip_axis=best_flip_axis,
                matchedVis=matched_vis,
            )
        raise NotEnoughFeatureMatchesError(
            number=np.max(matches_list),
            threshold=config.min_good_matches,
            partial_result=partial_result,
        )

    # --- Build matched visualization ---
    matched_vis = None
    if save_matched_vis:
        matched_vis = cv2.drawMatches(
            winning_moving_work, winning_kpsA,
            scaled.fixed_scaled, winning_kpsB,
            winning_good_matches, None,
        )

    # --- Scale keypoint coordinates to original space ---
    n = len(winning_good_matches)
    ptsA = np.zeros((n, 2), dtype="float")
    ptsB = np.zeros((n, 2), dtype="float")
    for i, m in enumerate(winning_good_matches):
        ptsA[i] = winning_kpsA[m.queryIdx].pt
        ptsB[i] = winning_kpsB[m.trainIdx].pt

    ptsA[:, 0] /= scaled.x_sf_moving
    ptsA[:, 1] /= scaled.y_sf_moving
    ptsB[:, 0] /= scaled.x_sf_fixed
    ptsB[:, 1] /= scaled.y_sf_fixed

    return FeatureMatchResult(
        kpsA=winning_kpsA,
        kpsB=winning_kpsB,
        good_matches=winning_good_matches,
        ptsA=ptsA,
        ptsB=ptsB,
        flip_axis=winning_flip_axis,
        matchedVis=matched_vis,
    )


def calculate_transformation_matrix(
    features: FeatureMatchResult,
    scaled: ScaledImages,
    config: RegistrationConfig,
) -> TransformResult:
    """Estimate the transformation matrix from matched feature points.

    Args:
        features: FeatureMatchResult from the extraction stage.
        scaled: ScaledImages from the loading stage (for resize_factor).
        config: Registration configuration.

    Returns:
        TransformResult dataclass.
    """
    if not HAS_OPENCV:
        raise ImportError(
            "OpenCV (cv2) is required for calculating transformation matrix. "
            "Install it with: pip install opencv-python"
        )

    verbose = config.verbose
    transform_type = "perspective" if config.perspective_transform else "affine"
    if verbose:
        logger.info("%s%s%s Transformation matrix (%s)", _TSIGN, _HLINE, _HLINE, transform_type)

    if config.perspective_transform:
        T, mask = cv2.findHomography(features.ptsA, features.ptsB, method=cv2.RANSAC)
    else:
        T, mask = cv2.estimateAffine2D(features.ptsA, features.ptsB)

    inlier_mask = mask.ravel().astype(bool) if mask is not None else None

    # If moving image was resized before warping, re-estimate T in resized space
    if scaled.resize_factor_moving != 1:
        ptsA_for_warp = features.ptsA * scaled.resize_factor_moving
        if config.perspective_transform:
            T_for_warp, _ = cv2.findHomography(ptsA_for_warp, features.ptsB, method=cv2.RANSAC)
        else:
            T_for_warp, _ = cv2.estimateAffine2D(ptsA_for_warp, features.ptsB)
    else:
        ptsA_for_warp = features.ptsA
        T_for_warp = T

    return TransformResult(
        T=T,
        T_for_warp=T_for_warp,
        inlier_mask=inlier_mask,
        ptsA_for_warp=ptsA_for_warp,
    )


def perform_registration_warp(
    scaled: ScaledImages,
    transform: TransformResult,
    features: FeatureMatchResult,
    config: RegistrationConfig,
) -> np.ndarray:
    """Warp the moving image onto the fixed image using the estimated transform.

    Uses ``scaled.moving_for_warp`` (the original, non-deconvolved image) and
    ``config.axes_moving`` (the original axes) for the warp call.

    Args:
        scaled: ScaledImages from the loading stage.
        transform: TransformResult from the estimation stage.
        features: FeatureMatchResult (for flip_axis).
        config: Registration configuration.

    Returns:
        Warped numpy array with spatial dimensions matching the fixed image.
    """
    verbose = config.verbose
    if verbose:
        logger.info("%s%s%s Registration", _TSIGN, _HLINE, _HLINE)

    image_to_warp = scaled.moving_for_warp

    if features.flip_axis is not None:
        flip_dir = "vertical" if features.flip_axis == 0 else "horizontal"
        if verbose:
            logger.info("%s     Applying %s flip", _VLINE, flip_dir)
        image_to_warp = np.flip(image_to_warp, axis=features.flip_axis)

    registered = apply_warp(
        image_to_warp,
        transform.T_for_warp,
        (scaled.fixed_w, scaled.fixed_h),
        config.axes_moving,  # use original axes, NOT axes_moving_effective
    )
    return registered


# ---------------------------------------------------------------------------
# QC and I/O helpers
# ---------------------------------------------------------------------------

def save_registration_qc(
    qc_dir: str | Path,
    identifier: str,
    T: np.ndarray | None,
    scaled_images: ScaledImages,
    features: FeatureMatchResult | None,
    config: RegistrationConfig,
    *,
    rank_matches_for_qc: bool = True,
) -> None:
    """Save registration QC files to disk.

    Writes:
    - ``{identifier}__transform.csv``: 3×3 transformation matrix (if T is not None)
    - ``{identifier}__matches_overview.png``: Top-100 ranked feature matches
    - ``{identifier}__matches_detail.png``: Top-5 zoomed detail panels

    Args:
        qc_dir: Output directory (created if it does not exist).
        identifier: Prefix for output filenames.
        T: Transformation matrix to save. If None, only match visualization is saved.
        scaled_images: ScaledImages from the loading stage.
        features: FeatureMatchResult from the extraction stage. May be None on failure.
        config: RegistrationConfig for metadata.
        rank_matches_for_qc: If True, rank matches by composite quality score.
    """
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Save deconvolved images for visual inspection (only when available)
    if scaled_images.moving_deconvolved is not None:
        _decon_small = scale_to_max_width(scaled_images.moving_deconvolved, axes="YX", max_width=4000, verbose=False)
        _decon_png = (_percentile_scale_for_saving(_decon_small) * 255).astype(np.uint8)
        cv2.imwrite(str(qc_dir / f"{identifier}__deconvolved_target.png"), _decon_png)

    if scaled_images.fixed_deconvolved is not None:
        _decon_small = scale_to_max_width(scaled_images.fixed_deconvolved, axes="YX", max_width=4000, verbose=False)
        _decon_png = (_percentile_scale_for_saving(_decon_small) * 255).astype(np.uint8)
        cv2.imwrite(str(qc_dir / f"{identifier}__deconvolved_template.png"), _decon_png)

    if T is not None:
        # Save transformation matrix as 3×3 CSV
        if T.shape == (2, 3):
            T_csv = np.vstack([T, [0, 0, 1]])
        else:
            T_csv = T
        csv_path = qc_dir / f"{identifier}__transform.csv"
        np.savetxt(csv_path, T_csv, delimiter=",")
        remove_last_line_from_csv(csv_path)

    if features is not None and features.kpsA is not None and features.good_matches:
        # Build transform result stub for ranking (we have T but not full TransformResult here)
        transform_for_rank = None
        if T is not None:
            transform_for_rank = TransformResult(
                T=T,
                T_for_warp=T,
                inlier_mask=None,
                ptsA_for_warp=features.ptsA,
            )

        if rank_matches_for_qc:
            ranked_idx_cache = _rank_match_indices_for_qc(
                features, scaled_images, transform_for_rank, config
            )
        else:
            ranked_idx_cache = list(range(len(features.good_matches)))

        overview_fig, top5_fig = _create_match_figure(
            scaled=scaled_images,
            features=features,
            transform=transform_for_rank,
            config=config,
            topn=100,
            rank_matches_for_qc=rank_matches_for_qc,
            ranked_idx=ranked_idx_cache,
        )
        overview_fig.savefig(
            qc_dir / f"{identifier}__matches_overview.png", dpi=150, bbox_inches="tight"
        )
        plt.close(overview_fig)

        if top5_fig is not None:
            top5_fig.savefig(
                qc_dir / f"{identifier}__matches_detail.png", dpi=150, bbox_inches="tight"
            )
            plt.close(top5_fig)

    elif features is not None and features.matchedVis is not None:
        # Fallback: save the cv2.drawMatches visualisation
        plt.imshow(features.matchedVis)
        plt.savefig(qc_dir / f"{identifier}__matches_overview.png", dpi=400)
        plt.close()


def save_registered_image_tiff(
    output_dir: str | Path,
    identifier: str,
    registered: np.ndarray,
    axes: str,
    photometric: Literal["rgb", "minisblack", "maxisblack"] = "rgb",
    ome_metadata: dict | None = None,
) -> Path:
    """Save a registered image as OME-TIFF.

    Args:
        output_dir: Directory to save the file (created if needed).
        identifier: Identifier for the filename (``{identifier}__registered.ome.tif``).
        registered: The registered image array.
        axes: Axis descriptor (e.g. "YXS", "YX", "CYX").
        photometric: Photometric interpretation.
        ome_metadata: Optional OME metadata dictionary.

    Returns:
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{identifier}__registered.ome.tif"
    write_ome_tiff(
        file=outfile,
        image=registered,
        axes=axes,
        photometric=photometric,
        overwrite=True,
        metadata=ome_metadata or {},
    )
    return outfile


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def register_images_standalone(
    moving: np.ndarray | da.Array,
    fixed: np.ndarray | da.Array,
    *,
    axes_moving: str = "YXS",
    axes_fixed: str = "YX",
    max_width: int | None = 4000,
    convert_to_grayscale: bool = False,
    deconvolve_moving: bool = False,
    deconvolve_fixed: bool = False,
    decon_scale_factor: float = 0.2,
    perspective_transform: bool = False,
    feature_detection_method: Literal["sift", "surf"] = "sift",
    flann: bool = True,
    mutual_nearest_neighbor: bool = False,
    ratio_test: bool = True,
    keep_fraction: float = 0.2,
    min_good_matches: int = 20,
    test_flipping: bool = True,
    adjust_contrast_method: Literal["otsu", "clip"] | None = "clip",
    debug: bool = False,
    qc_dir: str | Path | None = None,
    qc_identifier: str = "registration",
    rank_matches_for_qc: bool = True,
    pixel_size_moving: float | None = None,
    pixel_size_fixed: float | None = None,
    physical_size_unit: str = "µm",
    verbose: bool = True,
    force_failure: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Register a moving image to a fixed image using feature-based alignment.

    The moving image is warped so that it spatially aligns with the fixed image.
    Internally this runs: image scaling → feature extraction → transformation
    matrix estimation → warping.

    Args:
        moving: Image to be registered (aligned/warped). Can be numpy or dask array.
        fixed: Fixed reference image that stays stationary. Can be numpy or dask array.
        axes_moving: Axis descriptor for moving image (e.g. "YXS" for RGB, "CYX" for
            multichannel, "YX" for grayscale).
        axes_fixed: Axis descriptor for fixed image.
        max_width: Maximum width (in pixels) for downscaling before feature detection.
            None disables scaling.
        convert_to_grayscale: If True, convert images to grayscale before feature detection.
        deconvolve_moving: If True, apply H&E colour deconvolution to extract nuclei channel
            from moving image.
        deconvolve_fixed: If True, apply H&E colour deconvolution to extract nuclei channel
            from fixed image.
        decon_scale_factor: Scale factor for deconvolution (resize before deconvolution,
            then scale back).
        perspective_transform: If True, estimate a perspective (homography) transform.
            If False, estimate affine.
        feature_detection_method: Feature detector to use ("sift" or "surf").
        flann: If True, use FLANN-based matcher. If False, use brute-force matcher.
        mutual_nearest_neighbor: If True, apply mutual nearest-neighbor filtering.
        ratio_test: If True, apply Lowe's ratio test for match filtering.
        keep_fraction: Fraction of matches to keep when ratio_test is False.
        min_good_matches: Minimum number of good matches required for registration.
        test_flipping: If True, test vertical flip of the moving image.
        adjust_contrast_method: Contrast adjustment before feature detection
            ("otsu", "clip", or None).
        debug: If True, save QC files (feature match visualizations, transform CSV).
        qc_dir: Directory for QC output files. If None and debug=True, defaults to
            ``<cwd>/registration_qc/`` and logs the resolved path.
        qc_identifier: Identifier prefix for QC output filenames.
        rank_matches_for_qc: If True, rank matches by composite quality score in QC figures.
        pixel_size_moving: Pixel size of moving image in physical units.
        pixel_size_fixed: Pixel size of fixed image in physical units.
        physical_size_unit: Unit of physical size (default "µm").
        verbose: If True, log progress messages.

    Returns:
        Tuple of:
            - registered (np.ndarray): The warped moving image with spatial dimensions
              matching the fixed image.
            - T (np.ndarray): The estimated transformation matrix (2×3 for affine,
              3×3 for perspective) in original (non-resized) coordinate space.

    Raises:
        ImportError: If OpenCV is not installed.
        NotEnoughFeatureMatchesError: If insufficient feature matches are found.
        ValueError: If deconvolution is requested for non-RGB images.
    """
    config = RegistrationConfig(
        axes_moving=axes_moving,
        axes_fixed=axes_fixed,
        max_width=max_width,
        convert_to_grayscale=convert_to_grayscale,
        deconvolve_moving=deconvolve_moving,
        deconvolve_fixed=deconvolve_fixed,
        decon_scale_factor=decon_scale_factor,
        perspective_transform=perspective_transform,
        feature_detection_method=feature_detection_method,
        flann=flann,
        mutual_nearest_neighbor=mutual_nearest_neighbor,
        ratio_test=ratio_test,
        keep_fraction=keep_fraction,
        min_good_matches=min_good_matches,
        test_flipping=test_flipping,
        adjust_contrast_method=adjust_contrast_method,
        verbose=verbose,
        pixel_size_moving=pixel_size_moving,
        pixel_size_fixed=pixel_size_fixed,
        physical_size_unit=physical_size_unit,
    )

    # Resolve QC directory
    if debug:
        if qc_dir is None:
            resolved_qc_dir = Path.cwd() / "registration_qc"
            logger.info("QC directory (auto-resolved): %s", resolved_qc_dir.resolve())
        else:
            resolved_qc_dir = Path(qc_dir)
    else:
        resolved_qc_dir = None

    # Stage 1: Load and scale
    scaled = load_and_scale_images(moving, fixed, config)

    # Stage 2: Feature extraction
    try:
        features = extract_features(scaled, config, force_failure=force_failure)
    except NotEnoughFeatureMatchesError as exc:
        if debug:
            save_registration_qc(
                qc_dir=resolved_qc_dir,
                identifier=qc_identifier + "__FAILED",
                T=None,
                scaled_images=scaled,
                features=exc.partial_result,
                config=config,
                rank_matches_for_qc=rank_matches_for_qc,
            )
        raise

    # Stage 3: Transformation matrix
    transform = calculate_transformation_matrix(features, scaled, config)

    # Stage 4: Warp
    registered = perform_registration_warp(scaled, transform, features, config)

    # Stage 5: QC output (if debug)
    if debug:
        save_registration_qc(
            qc_dir=resolved_qc_dir,
            identifier=qc_identifier,
            T=transform.T,
            scaled_images=scaled,
            features=features,
            config=config,
            rank_matches_for_qc=rank_matches_for_qc,
        )

    return registered, transform.T

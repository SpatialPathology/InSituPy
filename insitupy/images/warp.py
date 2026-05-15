"""Utility functions for applying geometric image warps (affine and perspective)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from insitupy._constants import SHRT_MAX
from insitupy.images.axes import ImageAxes
from insitupy.images.utils import fit_image_to_size_limit

logger = logging.getLogger(__name__)

def load_transformation_matrix(
    source: np.ndarray | str | Path,
    reference_pixel_size: float | None = None,
    source_pixel_size: float | None = None,
) -> np.ndarray:
    """Load and validate a transformation matrix, optionally converting from pixel to physical space.

    Args:
        source: A 2×3 or 3×3 numpy array, or a path to a CSV/Excel file containing the matrix.
        reference_pixel_size: Pixel size (µm/pixel) of the reference image used to produce the
            matrix. When provided, the matrix is converted from pixel space to physical (µm) space.
        source_pixel_size: Pixel size (µm/pixel) of the source image. Used to scale the linear
            block when ``reference_pixel_size`` is also provided.

    Returns:
        2×3 or 3×3 float64 numpy array. If ``reference_pixel_size`` is given, the returned matrix
        is in physical (µm) space; otherwise it is in pixel space.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If the file format is not supported or the matrix shape is invalid.
        NotImplementedError: If ``source`` is a 3×3 perspective matrix (third row ≠ [0, 0, 1])
            and ``reference_pixel_size`` is provided. Coordinate-space conversion for perspective
            matrices is not yet implemented.
    """
    import pandas as pd

    if isinstance(source, (str, Path)):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Transformation matrix file not found: {source}")
        if source.suffix.lower() in ['.csv', '.txt']:
            M = pd.read_csv(source, header=None).values
        elif source.suffix.lower() in ['.xlsx', '.xls']:
            M = pd.read_excel(source, header=None).values
        else:
            raise ValueError(
                f"Unsupported file format: {source.suffix}. Use .csv, .txt, .xlsx, or .xls"
            )
    else:
        M = np.array(source)

    if M.shape not in [(2, 3), (3, 3)]:
        raise ValueError(
            f"Transformation matrix must be 2×3 or 3×3, got shape {M.shape}. "
            f"Expected format:\n"
            f"[[a, b, xoff],\n"
            f" [d, e, yoff]] or with [0, 0, 1] as third row."
        )

    if M.shape == (3, 3) and reference_pixel_size is not None:
        if not np.allclose(M[2, :], [0, 0, 1]):
            raise NotImplementedError(
                "Coordinate-space conversion (reference_pixel_size) for 3×3 perspective matrices "
                "(third row ≠ [0, 0, 1]) is not yet implemented. "
                "Either pass the matrix already in physical space and omit reference_pixel_size, "
                "or use a 2×3 affine matrix."
            )

    M = M.astype(np.float64)

    if reference_pixel_size is not None:
        # Extract 2×3 view for the conversion (works for both 2×3 and affine-as-3×3)
        M2 = M[:2, :].copy()
        if source_pixel_size is not None:
            M2[:2, :2] *= (reference_pixel_size / source_pixel_size)
        M2[0, 2] *= reference_pixel_size  # x offset: pixels → µm
        M2[1, 2] *= reference_pixel_size  # y offset: pixels → µm
        M[:2, :] = M2
        logger.info(
            f"Converted transformation matrix from pixel coordinates "
            f"(reference: {reference_pixel_size} µm/pixel) to physical coordinates."
        )

    return M


def apply_warp(
    image: np.ndarray,
    M: np.ndarray,
    output_size: tuple,
    axes: str,
) -> np.ndarray:
    """Apply an affine or perspective warp to an image.

    Operates entirely in pixel space. The caller is responsible for any
    coordinate-space conversion before calling this function.

    Args:
        image: Input numpy array. Shape must be consistent with ``axes``.
        M: 2×3 affine matrix or 3×3 perspective matrix (already in pixel space).
        output_size: ``(width, height)`` of the output image in pixels.
        axes: Axis descriptor – one of ``'YX'``, ``'YXS'`` (RGB), or ``'CYX'``
            (multichannel).

    Returns:
        Warped numpy array with spatial dimensions ``(height, width)``.
    """
    import cv2

    img_axes = ImageAxes(axes)
    h_img, w_img = img_axes.Y, img_axes.X  # axis indices

    # --- SHRT_MAX size guard ---
    max_dim = max(image.shape[img_axes.Y], image.shape[img_axes.X])
    if max_dim > SHRT_MAX:
        image, sf = fit_image_to_size_limit(image, axes=axes, size_limit=SHRT_MAX, return_scale_factor=True)
        M = M.copy()
        M[0, 2] *= sf  # scale x translation
        M[1, 2] *= sf  # scale y translation
        logger.info(f"apply_warp: image resized by factor {sf:.4f} to satisfy SHRT_MAX limit.")

    # --- Determine warp type ---
    if M.shape == (3, 3) and not np.allclose(M[2, :], [0, 0, 1]):
        # True perspective matrix
        def _warp(img_2d):
            return cv2.warpPerspective(
                img_2d, M, output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    else:
        # Affine – use 2×3 slice
        M_affine = M[:2, :]
        def _warp(img_2d):
            return cv2.warpAffine(
                img_2d, M_affine, output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

    # --- Dispatch by axes ---
    if len(image.shape) == 2:
        # Grayscale YX
        return _warp(image)
    elif len(image.shape) == 3:
        if img_axes.is_rgb:
            # YXS — warp the whole array directly (cv2 handles HxWxC)
            return _warp(image)
        else:
            # CYX — per-channel loop
            n_channels = image.shape[img_axes.C]
            transformed_channels = []
            for c in range(n_channels):
                channel = np.take(image, c, axis=img_axes.C)
                transformed_channels.append(_warp(channel))
            return np.stack(transformed_channels, axis=img_axes.C)
    else:
        raise ValueError(f"apply_warp: unsupported image shape {image.shape} for axes '{axes}'")

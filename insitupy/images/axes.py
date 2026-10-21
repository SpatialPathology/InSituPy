"""Image axes handling: one canonical in-memory axis order.

- **Canonical in-memory axes:** ``"YX"`` grayscale; ``"YXS"`` RGB (3 samples) or RGBA (4 samples);
  ``"CYX"`` multi-channel. ``ImageData`` stores only ``YX`` and ``YXS`` (it splits ``CYX``).
- **Readers** that return an axes string (``read_image``, ``read_zarr``) return canonical axes and
  data in that order. Singleton extra axes are dropped.
- **RGB** is the letter ``S`` in canonical axes. ``ImageData.metadata[name]["rgb"] == (axes == "YXS")``
  always.
- **Array functions with an** ``axes`` **argument** (``apply_warp``, ``register_images_standalone``)
  accept any permutation of ``YX``, ``YX`` + ``S`` or ``YX`` + ``C`` and return the input layout.
- **Writers** decide the disk order (``write_zarr``: ``SYX``; ``write_ome_tiff``: interleaved ``YXS``).
"""
from warnings import warn


class ImageAxes:
    '''
    ImageAxes object simplyifing working with images of different axis configurations.

    Args:
        pattern:
            String describing the axis configuration, e.g. "TYXS", "YXS", or "CYX".
            C: Channels in multi-channel image.
            S: Channels in RGB image.
            Y: Y-coordinate in OpenCV or rows in numpy arrays.
            X: X-coordinate in OpenCV or columns in numpy array.
            T: Time dimension for time-series experiments.
    '''
    def __init__(self,
                 pattern: str  # description of axes, e.g. YXS for RGB, CYX for IF, TYXS for time-series RGB
                 ):
        self.pattern = pattern

        # find channel axis for RGB
        self.C = self.pattern.find("S")
        self.is_rgb = True
        if self.C == -1:
            # in this case there was no S in the pattern (no RGB image) and we need to search for a C for multi-channel images
            self.C = self.pattern.find("C")
            self.is_rgb = False
            if self.C == -1:
                # no channel axis found (is the case for grayscale image)
                self.C = None

        # find x and y and t
        self.X = self.pattern.find("X")
        self.Y = self.pattern.find("Y")
        if -1 in [self.X, self.Y]:
            raise ValueError(f"No X and Y given in image axis {self.pattern}")

        # find time series axis
        self.T = self.pattern.find("T")
        self.T = None if self.T == -1 else self.T

def get_height_and_width(image, axes_config: ImageAxes):
    """Return the height and width of an image using an axis configuration.

    Args:
        image: Image array (numpy or dask) with at least Y and X dimensions.
        axes_config: :class:`ImageAxes` instance describing the axis order of
            *image*.  The ``Y`` and ``X`` attributes are used to index into
            ``image.shape``.

    Returns:
        A tuple ``(height, width)`` where both values are integers.
    """
    h_image = image.shape[axes_config.Y] # height of image
    w_image = image.shape[axes_config.X] # width of image
    return (h_image, w_image)


def _shape_list(img):
    if isinstance(img, list):
        return [tuple(arr.shape) for arr in img]
    return [tuple(img.shape)]


def _coerce_legacy_cyx_single_channel_to_yx(img):
    if isinstance(img, list):
        coerced = []
        for arr in img:
            if arr.ndim == 3:
                if arr.shape[0] != 1:
                    raise ValueError(
                        "Cannot coerce legacy 'CYX' metadata to 'YX': found a 3D level with channel size != 1. "
                        f"Shape: {tuple(arr.shape)}"
                    )
                coerced.append(arr[0, ...])
            else:
                coerced.append(arr)
        return coerced

    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(
                "Cannot coerce legacy 'CYX' metadata to 'YX': found 3D data with channel size != 1. "
                f"Shape: {tuple(img.shape)}"
            )
        return img[0, ...]
    return img


def normalize_axes_and_shape(img, axes: str):
    """
    Normalize known legacy axes mismatches and validate axis/shape consistency.

    Supported compatibility behavior:
    - Legacy metadata with ``axes='CYX'`` but effectively grayscale data
      (2D data, or 3D with singleton channel) is coerced to ``'YX'``.

    For all other mismatches, a ValueError is raised with detailed shape info.
    """
    shapes_before = _shape_list(img)
    ndims_before = sorted({len(s) for s in shapes_before})

    if len(ndims_before) != 1:
        if axes == "CYX" and set(ndims_before).issubset({2, 3}):
            # Allow mixed 2D and singleton-3D levels from legacy exports.
            for shape in shapes_before:
                if len(shape) == 3 and shape[0] != 1:
                    raise ValueError(
                        "Inconsistent image pyramid dimensions for axes='CYX'. "
                        f"Expected singleton channel for 3D levels when coercing to 'YX', got shapes: {shapes_before}"
                    )
            img = _coerce_legacy_cyx_single_channel_to_yx(img)
            warn(
                "Axes metadata says 'CYX' but data is effectively single-channel grayscale. "
                "Coercing axes to 'YX' for compatibility with legacy exports."
            )
            return img, "YX"

        raise ValueError(
            f"Inconsistent image dimensions across pyramid levels: {shapes_before}. "
            f"Provided axes='{axes}'."
        )

    ndim = ndims_before[0]

    if axes == "CYX" and ndim == 2:
        warn(
            "Axes metadata says 'CYX' but image data is 2D. "
            "Assuming legacy metadata mismatch and coercing axes to 'YX'."
        )
        return img, "YX"

    if axes == "CYX" and ndim == 3:
        shapes = _shape_list(img)
        if all(shape[0] == 1 for shape in shapes):
            img = _coerce_legacy_cyx_single_channel_to_yx(img)
            warn(
                "Axes metadata says 'CYX' but channel dimension is singleton across all levels. "
                "Coercing to 'YX'."
            )
            return img, "YX"

    if len(axes) != ndim:
        raise ValueError(
            f"Axes and image dimensionality mismatch: axes='{axes}' (len={len(axes)}), "
            f"image ndim={ndim}, shapes={_shape_list(img)}"
        )

    return img, axes

def _canonical_permutation(axes: str) -> tuple[tuple[int, ...], str]:
    """Return the permutation that brings *axes* into canonical order, and the canonical string.

    *axes* must be a permutation of ``"YX"``, ``"YX"`` + ``"S"`` or ``"YX"`` + ``"C"`` (no other
    letters, no repeats). Applying the permutation with ``arr.transpose(perm)`` yields ``"YX"``,
    ``"YXS"`` or ``"CYX"``; ``np.argsort(perm)`` restores the input layout.

    Raises:
        ValueError: If *axes* is not a supported axes string.
    """
    letters = sorted(axes)
    if letters == sorted("YX"):
        canon = "YX"
    elif letters == sorted("YXS"):
        canon = "YXS"
    elif letters == sorted("YXC"):
        canon = "CYX"
    else:
        raise ValueError(
            f"Unsupported image axes '{axes}'. Expected an order of 'YX' (grayscale), "
            "'YX' plus 'S' (RGB/RGBA) or 'YX' plus 'C' (multi-channel)."
        )
    perm = tuple(axes.index(ax) for ax in canon)
    return perm, canon


def _canonicalize_axes(img, axes: str, *, rgb: bool | None = None):
    """Return ``(img, axes)`` in canonical order: ``"YX"``, ``"YXS"`` (3 or 4 samples) or ``"CYX"``.

    Legacy ``CYX`` metadata quirks are fixed first (:func:`normalize_axes_and_shape`). Extra axes
    (``T``, ``Z``, ...) and channel axes of length 1 are dropped.

    Args:
        img: numpy or dask array, or a pyramid list of them (every level is treated the same;
            dask stays lazy).
        axes: Axes string of *img*, e.g. ``"SYX"``, ``"TYXS"`` or ``"CYX"``.
        rgb: ``None`` keeps the letters. ``True`` treats the channel axis as colour samples
            (``S``), ``False`` as separate channels (``C``).

    Raises:
        ValueError: For extra axes longer than 1, both a ``C`` and an ``S`` axis, ``rgb=True``
            without a channel axis, an ``S`` axis with other than 3 or 4 samples, or unknown
            or repeated letters.
    """
    img, axes = normalize_axes_and_shape(img, axes)
    is_pyramid = isinstance(img, list)
    levels = img if is_pyramid else [img]
    shape = tuple(levels[0].shape)

    if len(set(axes)) != len(axes):
        raise ValueError(f"Image axes '{axes}' contain a repeated letter.")

    # drop singleton extra axes and singleton channel axes
    drop = []
    for k, ax in enumerate(axes):
        if ax in "YX":
            continue
        if ax not in "CS" and shape[k] != 1:
            raise ValueError(
                f"Axis '{ax}' of '{axes}' has length {shape[k]}. Only 2D images with an optional "
                "channel axis are supported (no time series or z-stacks)."
            )
        if shape[k] == 1:
            drop.append(k)
    if drop:
        levels = [lvl.squeeze(axis=tuple(drop)) for lvl in levels]
        axes = "".join(ax for k, ax in enumerate(axes) if k not in drop)
        shape = tuple(levels[0].shape)

    if "C" in axes and "S" in axes:
        raise ValueError(
            f"Image axes '{axes}' (shape {shape}) have both a 'C' and an 'S' axis. "
            "Multi-channel RGB images are not supported."
        )

    if rgb is True:
        if "C" not in axes and "S" not in axes:
            raise ValueError(
                f"is_rgb=True, but the image has no channel axis (axes '{axes}', shape {shape})."
            )
        axes = axes.replace("C", "S")
    elif rgb is False:
        axes = axes.replace("S", "C")

    if "S" in axes:
        n = shape[axes.index("S")]
        if n not in (3, 4):
            raise ValueError(
                f"An 'S' axis holds 3 (RGB) or 4 (RGBA) samples, got {n} (axes '{axes}', shape "
                f"{shape}). Label the axis 'C' for multi-channel data."
            )

    perm, canon = _canonical_permutation(axes)
    if perm != tuple(range(len(perm))):
        levels = [lvl.transpose(perm) for lvl in levels]

    return (levels if is_pyramid else levels[0]), canon


def _transpose_to_standard_axes(img, axes: str):
    """Transpose image to the canonical axis order (``YX``, ``CYX`` or ``YXS``).

    Thin wrapper around :func:`_canonicalize_axes`, kept for existing callers.
    """
    return _canonicalize_axes(img, axes)

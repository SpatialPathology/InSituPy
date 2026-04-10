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

def _transpose_to_standard_axes(img, axes: str):
    """
    Transpose image to standard axis order: YX, CYX, or YXS.

    Args:
        img: Dask array or list of dask arrays (image pyramid)
        axes: String describing current axis configuration (e.g., "CYX", "YXS", "TCYX")

    Returns:
        Tuple of (transposed_img, new_axes_string)
        - transposed_img: Image with axes in standard order
        - new_axes_string: Updated axes string ("YX", "CYX", or "YXS")
    """
    img, axes = normalize_axes_and_shape(img, axes)
    axes_obj = ImageAxes(axes)

    # Determine target axis order and transpose if necessary
    if axes_obj.C is not None:
        # Multi-channel (C) or RGB (S) image - target: CYX or YXS
        if axes_obj.is_rgb:
            target_axes = "YXS"
            target_order = [axes_obj.Y, axes_obj.X, axes_obj.C]
        else:
            target_axes = "CYX"
            target_order = [axes_obj.C, axes_obj.Y, axes_obj.X]
    else:
        # Grayscale image - target: YX
        target_axes = "YX"
        target_order = [axes_obj.Y, axes_obj.X]

    # Only transpose if current order differs from target
    current_order = list(range(len(img.shape) if not isinstance(img, list) else len(img[0].shape)))
    if target_order != current_order:
        if isinstance(img, list):
            # Handle image pyramid (list of dask arrays)
            img = [arr.transpose(target_order) for arr in img]
        else:
            # Handle single dask array
            img = img.transpose(target_order)

        return img, target_axes
    else:
        # No transposition needed
        return img, axes
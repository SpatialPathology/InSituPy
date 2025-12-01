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
    h_image = image.shape[axes_config.Y] # height of image
    w_image = image.shape[axes_config.X] # width of image
    return (h_image, w_image)

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
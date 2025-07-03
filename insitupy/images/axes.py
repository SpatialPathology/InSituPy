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
        if self.C == -1:
            # in this case there was no S in the pattern (no RGB image) and we need to search for a C for multi-channel images
            self.C = self.pattern.find("C")
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
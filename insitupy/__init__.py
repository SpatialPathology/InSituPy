__author__ = "Johannes Wirth"
__email__ = "j.wirth@tum.de"
__version__ = "1.4.0"

# check if napari is available
try:
    import napari
    WITH_NAPARI = True
except ImportError:
    WITH_NAPARI = False

from . import images as im
from . import io
from . import plotting as pl
from . import utils
from ._core._deprecated import XeniumData
from ._core.dataclasses import AnnotationsData, BoundariesData, ImageData
from ._core.insitudata import (InSituData, calc_distance_of_cells_from,
                               register_images)
from ._core.xenium import read_xenium
from .palettes import CustomPalettes

__all__ = [
    "InSituData",
    "AnnotationsData",
    "BoundariesData",
    "ImageData",
    "im",
    "utils"
]
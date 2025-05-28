__author__ = "Johannes Wirth"
__email__ = "j.wirth@tum.de"
__version__ = "0.8.8"

# check if napari is available
try:
    import napari
    WITH_NAPARI = True
except ImportError:
    WITH_NAPARI = False

from insitupy._constants import CACHE

from . import images as im
from . import io
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import utils
from ._core.dataclasses import (AnnotationsData, BoundariesData, CellData,
                                ImageData, MultiCellData, RegionsData)
from ._core.insitudata import InSituData
from ._core.insituexperiment import InSituExperiment
from ._core.readers import read_xenium
from .images.registration import register_images
from .palettes import CustomPalettes
from .tools.dge import differential_gene_expression
from .tools.distance import calc_distance_of_cells_from

__all__ = [
    "InSituData",
    "InSituExperiment",
    "CustomPalettes",
    "AnnotationsData",
    "BoundariesData",
    "CellData",
    "ImageData",
    "MultiCellData",
    "RegionsData",
    "read_xenium",
    "differential_gene_expression",
    "calc_distance_of_cells_from",
    "register_images",
    "im",
    "io",
    "pl",
    "pp",
    "tl",
    "utils"
]
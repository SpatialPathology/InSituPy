__author__ = "Johannes Wirth"
__email__ = "j.wirth@tum.de"
__version__ = "0.11.0"

# check if napari is available
try:
    import napari
    WITH_NAPARI = True
except ImportError:
    print((
        "Napari is not installed. Interactive visualization using `.show()` will not be possible. "
        "To enable these features, please install napari (for example, with `pip install \"napari[all]\"`)."
    ))
    WITH_NAPARI = False

from . import _core, dataclasses, datasets, experiment
from . import images as im
from . import io
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import utils
from ._constants import CACHE
from ._core.data import InSituData
from .experiment.data import InSituExperiment

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

# configure logging
import logging

from insitupy._logging import TqdmLoggingHandler

logger = logging.getLogger('insitupy')
logger.setLevel(logging.INFO)
logger.propagate = False

# Only add handler if one doesn't exist yet
# Use TqdmLoggingHandler to prevent progress bar disruption from log messages
if not logger.handlers:
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
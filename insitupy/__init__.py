from ._version import __author__, __email__, __version__
from ._constants import WITH_NAPARI

from . import _core, containers, datasets, experiment
from . import images as im
from . import io
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import utils
from ._constants import CACHE
from ._core.data import InSituData
from .experiment.data import InSituExperiment

try:
    from . import spatialdata
except ImportError:
    pass

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "InSituData",
    "InSituExperiment",
    "_core",
    "containers",
    "datasets",
    "experiment",
    "im",
    "io",
    "pl",
    "pp",
    "tl",
    "utils",
]

# configure logging
from ._logging import setup_logging as _setup_logging
_setup_logging()
del _setup_logging
from . import _core, containers, datasets, experiment, io, utils
from . import images as im
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from ._constants import CACHE, WITH_NAPARI
from ._core.data import InSituData
from ._version import __author__, __email__, __version__
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

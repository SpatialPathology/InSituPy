from . import _core, containers, datasets, experiment, experimental, io, utils
from . import images as im
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from ._constants import CACHE, WITH_NAPARI
from ._core.data import InSituData
from ._module_aliases import register_module_aliases as _register_module_aliases
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
    "experimental",
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

# Make the short aliases importable (`from insitupy.tl import dge`). Sphinx autodoc >= 9 imports the
# module path of `pp.x` / `tl.x` / `im.x` API entries instead of walking attributes, so without this
# those API pages render empty. `insitupy.pp` etc. as attributes stay the real submodules.
_register_module_aliases(__name__, {"im": im, "pl": pl, "pp": pp, "tl": tl})
del _register_module_aliases

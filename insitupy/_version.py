from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("insitupy-spatial")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Johannes Wirth"
__email__ = "j.wirth@tum.de"

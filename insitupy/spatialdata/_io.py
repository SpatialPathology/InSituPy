try:
    from spatialdata import read_zarr
except ImportError:
    raise ImportError("Please install spatialdata with `pip install spatialdata`.")

import logging
import warnings

# --------------------
# Wrapper dataclasses
# --------------------

def _silent_read_zarr(path):
    logging.info(f"Reading `SpatialData` zarr store from {path}...")
    logging.disable(logging.INFO)  # Disable INFO and below
    logging.disable(logging.WARNING)  # Disable INFO and below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zs = read_zarr(path)
    logging.disable(logging.NOTSET)  # Re-enable logging
    return zs
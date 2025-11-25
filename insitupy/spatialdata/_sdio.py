import os
from pathlib import Path
from typing import Dict, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData

try:
    from spatialdata import read_zarr
except ImportError:
    raise ImportError("Please install spatialdata with `pip install spatialdata`.")

import logging
import warnings

from insitupy._constants import MODALITIES_COLOR_DICT, SAMPLE_STR
from insitupy._textformat import textformat as tf

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
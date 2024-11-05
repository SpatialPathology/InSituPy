import string
from pathlib import Path

import matplotlib

from insitupy.palettes import CustomPalettes

# make sure that images do not exceed limits in c++ (required for cv2::remap function in cv2::warpAffine)
# see also https://www.geeksforgeeks.org/climits-limits-h-cc/
SHRT_MAX = 2**15-1 # 32767
SHRT_MIN = -(2**15-1) # -32767

# create cache dir
CACHE = Path.home() / ".cache/InSituPy/"

# modalities
MODALITIES = ["annotations", "cells", "images", "regions", "transcripts"]
LOAD_FUNCS = [
    'load_annotations',
    'load_cells',
    'load_images',
    'load_regions',
    'load_transcripts'
    ]

# naming
ISPY_METADATA_FILE = ".ispy"
XENIUM_HEX_RANGE = string.ascii_lowercase[:16]
NORMAL_HEX_RANGE = "".join([str(e) for e in range(10)]) + string.ascii_lowercase[:6]
XENIUM_INT_TO_HEX_CONV_DICT = {k:v for k,v in zip(NORMAL_HEX_RANGE, XENIUM_HEX_RANGE)}
XENIUM_HEX_TO_INT_CONV_DICT = {v:k for k,v in zip(NORMAL_HEX_RANGE, XENIUM_HEX_RANGE)}

# napari layer symbols
# SHAPES_SYMBOL = "\u2605" # Star: ★
# POINTS_SYMBOL = "\u2022" # Bullet: •
SHAPES_SYMBOL = "\U0001F52C" # 🔬
POINTS_SYMBOL = "\U0001F4CD" # 📍
REGIONS_SYMBOL = "\U0001F30D" # 🌍

# cmaps
palettes = CustomPalettes()
DEFAULT_CATEGORICAL_CMAP = palettes.tab20_mod
REGION_CMAP = matplotlib.colormaps["tab10"]

# annotations
FORBIDDEN_ANNOTATION_NAMES = ["rest"]
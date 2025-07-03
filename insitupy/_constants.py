import string
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from insitupy._textformat import textformat as tf
from insitupy.palettes import CustomPalettes

# make sure that images do not exceed limits in c++ (required for cv2::remap function in cv2::warpAffine)
# see also https://www.geeksforgeeks.org/climits-limits-h-cc/
SHRT_MAX = 2**15-1 # 32767
SHRT_MIN = -(2**15-1) # -32767

# create cache dir
CACHE = Path.home() / ".cache/InSituPy/"

# modalities
MODALITIES = ["cells", "images", "transcripts", "annotations", "regions"]
MODALITIES_ABBR = "CITAR"
LOAD_FUNCS = [
    'load_annotations',
    'load_cells',
    'load_images',
    'load_regions',
    'load_transcripts'
    ]
MODALITIES_COLOR_DICT = {
    "images": tf.Blue,
    "cells": tf.Green,
    "transcripts": tf.Purple,
    "annotations": tf.Cyan,
    "regions": tf.Yellow
}

# naming
ISPY_METADATA_FILE = ".ispy"
XENIUM_HEX_RANGE = string.ascii_lowercase[:16]
NORMAL_HEX_RANGE = "".join([str(e) for e in range(10)]) + string.ascii_lowercase[:6]
XENIUM_INT_TO_HEX_CONV_DICT = {k:v for k,v in zip(NORMAL_HEX_RANGE, XENIUM_HEX_RANGE)}
XENIUM_HEX_TO_INT_CONV_DICT = {v:k for k,v in zip(NORMAL_HEX_RANGE, XENIUM_HEX_RANGE)}

# napari layer symbols
# SHAPES_SYMBOL = "\u2605" # Star: ★
# POINTS_SYMBOL = "\u2022" # Bullet: •
ANNOTATIONS_SYMBOL = "\U0001F52C" # 🔬
POINTS_SYMBOL = "\U0001F4CD" # 📍
REGIONS_SYMBOL = "\U0001F30D" # 🌍

# annotations
FORBIDDEN_ANNOTATION_NAMES = ["rest"]

## Matplotlib settings
# cmaps
palettes = CustomPalettes()
DEFAULT_CATEGORICAL_CMAP = palettes.tab20_mod
REGION_CMAP = matplotlib.colormaps["tab10"]
DEFAULT_CONTINUOUS_CMAP = "viridis"

# font size
def _init_mpl_fontsize(scale_factor=1):
    '''
    https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
    '''
    SMALL_SIZE = 14*scale_factor
    MEDIUM_SIZE = 16*scale_factor
    BIGGER_SIZE = 18*scale_factor

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

_init_mpl_fontsize()

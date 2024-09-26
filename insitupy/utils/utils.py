import math
import os
from datetime import datetime
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pandas.api.types import is_numeric_dtype, is_string_dtype
from parse import datetime
from shapely import LineString, Point, Polygon, affinity

from insitupy._constants import (XENIUM_HEX_TO_INT_CONV_DICT,
                                 XENIUM_INT_TO_HEX_CONV_DICT)


class textformat:
    '''
    Helper class to format printed text.
    e.g. print(color.RED + color.BOLD + 'Hello, World!' + color.END)
    '''
    # colors and formats
    # PURPLE = '\033[95m'
    # CYAN = '\033[96m'
    # DARKCYAN = '\033[36m'
    # BLUE = '\033[94m'
    # GREEN = '\033[92m'
    # YELLOW = '\033[93m'
    # RED = '\033[91m'
    # BOLD = '\033[1m'
    # UNDERLINE = '\033[4m'
    # END = '\033[0m'

    ResetAll = "\033[0m"

    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    Purple       = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"

    # signs
    TSIGN = "\u251c"
    LSIGN = "\u2514"
    HLINE = "\u2500"
    RARROWHEAD = "\u27A4"
    TICK = "\u2714"
    CIRCLE_EMPTY = "\u25EF"
    CIRCLE_FILLED = "\u2B24"
    CIRCLE_HALF = "\u25D0"
    CIRCLE_THREEQUARTER = "\u25D5"
    CIRCLE_ONEQUARTER = "\u25D4"

    # spacer
    SPACER = "    "

def remove_last_line_from_csv(filename):
    with open(filename) as myFile:
        lines = myFile.readlines()
        last_line = lines[len(lines)-1]
        lines[len(lines)-1] = last_line.rstrip()
    with open(filename, 'w') as myFile:
        myFile.writelines(lines)

def decode_robust(s, encoding="utf-8"):
    try:
        return s.decode(encoding)
    except (UnicodeDecodeError, AttributeError):
        return s

def decode_robust_series(s, encoding="utf-8"):
    '''
    Function to decode a pandas series in a robust fashion with different checks.
    This circumvents the return of NaNs and makes a decision in case of different errors.
    '''
    if is_numeric_dtype(s):
        return s
    if is_string_dtype(s):
        return s
    try:
        decoded = s.str.decode(encoding)
        if decoded.isna().all():
            return decoded
        elif decoded.isna().any():
            namask = decoded.isna()
            decoded[namask] = s[namask]
        return decoded
    except (UnicodeDecodeError, AttributeError):
        return s

def convert_to_list(elem):
    '''
    Return element to list if it is not a list already.
    '''
    return [elem] if (isinstance(elem, str) or isinstance(elem, os.PathLike) or isinstance(elem, int)) else list(elem)

def nested_dict_numpy_to_list(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, ndarray):
            dictionary[key] = value.tolist()
        elif isinstance(value, dict):
            nested_dict_numpy_to_list(value)

def get_nrows_maxcols(keys, max_cols):
    '''
    Determine optimal number of rows and columns for `plt.subplot` based on
    list of keys [`keys`] and maximum number of columns [`max_cols`].

    Returns: `n_plots`, `n_rows`, `max_cols`
    '''

    n_plots = len(keys)
    if n_plots > max_cols:
        n_rows = math.ceil(n_plots / max_cols)
    else:
        n_rows = 1
        max_cols = n_plots

    return n_plots, n_rows, max_cols


def _generate_time_based_uid():
    time_str = datetime.now().strftime("%y%m%d-%H%M%S%f")
    short_uid = str(uuid4()).split("-")[0]
    uid = f"{time_str}-{short_uid}"
    return uid

def convert_int_to_xenium_hex(value, dataset_suffix=1, final_length=8):
    """Convert integers into Xenium-style hexadecimal representation.
    Described here: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cellID

    Args:
        value (_type_): _description_
        dataset_suffix (int, optional): _description_. Defaults to 1.
        final_length (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    # generate hexadecimal representation
    hex_repr = hex(value)[2:]

    # convert normal hex to xenium-modified hex
    hex_repr = "".join([str(XENIUM_INT_TO_HEX_CONV_DICT[elem]) for elem in hex_repr])

    # add a to the beginning to fill to final length
    hex_repr = hex_repr.rjust(final_length, 'a')

    # add dataset suffix
    hex_repr += f"-{dataset_suffix}"

    return hex_repr

def convert_xenium_hex_to_int(hex_repr):
    """Convert Xenium-style hexadecimal representation into integers.
    Described here: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cellID


    Args:
        hex_repr (_type_): _description_

    Returns:
        _type_: _description_
    """
    # remove dataset suffix
    hex_repr_split = hex_repr.split("-")

    # try to extract a dataset suffix
    try:
        dataset_suffix = int(hex_repr_split[1])
    except IndexError:
        dataset_suffix = None

    # extract the hex repr
    hex_repr = hex_repr_split[0]

    # remove leading a
    hex_repr = hex_repr.lstrip("a")

    # convert xenium-modified hex to normal hex
    hex_repr = "".join([str(XENIUM_HEX_TO_INT_CONV_DICT[elem]) for elem in hex_repr])

    # generate decimal representation
    dec = int(hex_repr, 16)

    return dec, dataset_suffix


def create_ellipse_from_bbox(corner_coords):
    """
    Create an ellipse from a bounding box defined by its corner coordinates.

    Parameters:
    corner_coords (list of tuples): A list containing the coordinates of the four corners
                                     of the bounding box in the format [(x1, y1), (x2, y2),
                                     (x3, y3), (x4, y4)].

    Returns:
    shapely.geometry.Polygon: A Shapely polygon representing the ellipse.
    """
    # Unpack the corner coordinates
    x_coords = [coord[0] for coord in corner_coords]
    y_coords = [coord[1] for coord in corner_coords]

    # Calculate the bounding box's min and max coordinates
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Calculate the center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculate the semi-major and semi-minor axes
    semi_major = (x_max - x_min) / 2
    semi_minor = (y_max - y_min) / 2

    # Create a unit circle centered at the origin
    unit_circle = Point(0, 0).buffer(1)  # Create a unit circle

    # Scale the unit circle to create the ellipse
    ellipse = affinity.scale(unit_circle, xfact=semi_major, yfact=semi_minor)

    # Translate the ellipse to the center of the bounding box
    ellipse = affinity.translate(ellipse, xoff=center_x, yoff=center_y)

    return ellipse

def convert_napari_shape_to_polygon_or_line(napari_shape_data, shape_type):
    """
    Convert shape data from Napari format to a Shapely polygon or line.

    This function takes shape data in the format used by Napari and converts it
    into a Shapely geometry object, which can be either a polygon, an ellipse,
    or a line string, depending on the specified shape type.

    Parameters:
    napari_shape_data (numpy.ndarray): An array of shape data, where each row
                                        represents a point (y, x) in the Napari
                                        coordinate system.
    shape_type (str): A string indicating the type of shape to create.
                      Accepted values are:
                      - "polygon": Converts the shape data to a Shapely Polygon.
                      - "ellipse": Converts the shape data to a Shapely Polygon
                                   representing an ellipse based on the bounding box.
                      - "path": Converts the shape data to a Shapely LineString.

    Returns:
    shapely.geometry.Polygon or shapely.geometry.LineString: A Shapely geometry
    object representing the converted shape.

    Raises:
    TypeError: If the provided shape_type is not one of the accepted values.
    """
    if shape_type == "polygon":
        result = Polygon(np.stack([napari_shape_data[:, 1], napari_shape_data[:, 0]], axis=1))
    elif shape_type == "ellipse":
        result = create_ellipse_from_bbox(np.flip(napari_shape_data, axis=1))
    elif shape_type == "path":
        result = LineString(np.flip(napari_shape_data, axis=1))
    else:
        TypeError(f"Shape has an unknown type: {shape_type}")

    return result

def exclude_index(array, exclude_index):
    """
    Exclude the element at the specified index from the array.

    Args:
        array (np.ndarray): The input NumPy array.
        exclude_index (int): The index of the element to exclude.

    Returns:
        np.ndarray: A new array with the element at exclude_index excluded.
    """
    return np.concatenate((array[:exclude_index], array[exclude_index+1:]))

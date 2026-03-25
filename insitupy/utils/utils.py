import logging
import math
import os
from datetime import datetime
from typing import Optional, Tuple, Union
from uuid import uuid4
from warnings import warn

logger = logging.getLogger(__name__)

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.api.types import is_numeric_dtype, is_string_dtype
import shapely as _shapely
from shapely import LineString, Point, Polygon, affinity

from insitupy._constants import (XENIUM_HEX_TO_INT_CONV_DICT,
                                 XENIUM_INT_TO_HEX_CONV_DICT)


def create_ansi_color_code_from_rgb(rgb_color):
    """Return an ANSI 24-bit foreground colour escape code for the given RGB tuple.

    Args:
        rgb_color: A 3-element sequence of integers ``(R, G, B)`` in 0–255.

    Returns:
        The ANSI escape string ``\\033[38;2;R;G;Bm``.
    """
    # Create the ANSI escape code
    ansi_escape_code = f'\033[38;2;{rgb_color[0]};{rgb_color[1]};{rgb_color[2]}m'
    return ansi_escape_code

def remove_last_line_from_csv(filename):
    """Strip the trailing newline from the last line of a CSV file in-place.

    Args:
        filename: Path to the CSV file to modify.
    """
    with open(filename) as myFile:
        lines = myFile.readlines()
        last_line = lines[len(lines)-1]
        lines[len(lines)-1] = last_line.rstrip()
    with open(filename, 'w') as myFile:
        myFile.writelines(lines)

def decode_robust(s, encoding="utf-8"):
    """Decode *s* with the given *encoding*, returning *s* unchanged on failure.

    Args:
        s: Bytes-like object to decode, or a value that will be returned as-is
            if it is already a string or decoding fails.
        encoding: Character encoding to use for decoding.

    Returns:
        The decoded string, or the original *s* if decoding raised
        :exc:`UnicodeDecodeError` or :exc:`AttributeError`.
    """
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
    """Recursively convert all NumPy arrays in a nested dict to plain Python lists.

    Modifies *dictionary* in-place.

    Args:
        dictionary: A possibly-nested dict whose ``ndarray`` values should be
            converted to lists.
    """
    for key, value in dictionary.items():
        if isinstance(value, ndarray):
            dictionary[key] = value.tolist()
        elif isinstance(value, dict):
            nested_dict_numpy_to_list(value)

def get_nrows_maxcols(n_keys, max_cols):
    '''
    Determine optimal number of rows and columns for `plt.subplot` based on
    number of keys ['n_keys'] and maximum number of columns [`max_cols`].

    Returns: `n_plots`, `n_rows`, `max_cols`
    '''

    #n_plots = len(keys)
    if n_keys > max_cols:
        n_rows = math.ceil(n_keys / max_cols)
    else:
        n_rows = 1
        max_cols = n_keys

    return n_keys, n_rows, max_cols

def remove_empty_subplots(axes, nplots, nrows, ncols):
    """Hide trailing empty axes in a flat subplot array.

    Args:
        axes: 1-D array of :class:`~matplotlib.axes.Axes` objects.
        nplots: Number of subplots that were actually populated.
        nrows: Total number of rows in the subplot grid.
        ncols: Total number of columns in the subplot grid.

    Raises:
        ValueError: If *axes* is not 1-D.
    """
    if len(axes.shape) != 1:
        raise ValueError("Axis object must have only one dimension.")
    if nplots > 1:
        # check if there are empty plots remaining
        i = nplots
        while i < nrows * ncols:
            # remove empty plots
            axes[i].set_axis_off()
            i+=1

def check_list(List, list_to_compare):
    '''
    Compare two lists and return the elements that are in both lists.
    If not all elements are in both lists give message telling which are not.
    '''

    not_in = []
    List = [l if l in list_to_compare else not_in.append(l) for l in List]

    # remove None values
    List = [elem for elem in List if elem is not None]

    if len(not_in) > 0:
        logger.warning("Following elements not found: {}".format(", ".join(not_in)))

    return List

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
    if (shape_type == "polygon") or (shape_type == "rectangle"):
        result = Polygon(np.stack([napari_shape_data[:, 1], napari_shape_data[:, 0]], axis=1))
    elif shape_type == "ellipse":
        result = create_ellipse_from_bbox(np.flip(napari_shape_data, axis=1))
    elif shape_type in ["path", "line"]:
        result = LineString(np.flip(napari_shape_data, axis=1))
    else:
        raise TypeError(f"Shape has an unknown type: {shape_type}")

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

def _crop_transcripts(
    transcript_df: Union[pd.DataFrame, dd.DataFrame],
    shape: Optional[Polygon] = None,
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    materialize: bool = True
    ):

    # Compute Dask → pandas BEFORE filtering to avoid task-graph overhead.
    # On an in-memory-backed Dask DF this is nearly free; on disk-backed it
    # loads the data once, which is the same total work as the old approach.
    if materialize and isinstance(transcript_df, dd.DataFrame):
        transcript_df = transcript_df.compute()

    if shape is not None:
        if xlim is not None and ylim is not None:
            if verbose:
                warn("Both xlim/ylim and shape are provided. Shape will be used for cropping.")

        # Detect column layout
        try:
            x_vals = transcript_df.loc[:, ("coordinates", "x")]
            y_vals = transcript_df.loc[:, ("coordinates", "y")]
            grouped_df = True
        except KeyError:
            x_vals = transcript_df["x_location"]
            y_vals = transcript_df["y_location"]
            grouped_df = False

        minx, miny, maxx, maxy = shape.bounds

        if isinstance(transcript_df, pd.DataFrame):
            # Step 1: bounding-box pre-filter — reduces 41M rows to ~region-size candidates
            bbox_mask = (
                (x_vals >= minx) & (x_vals <= maxx) &
                (y_vals >= miny) & (y_vals <= maxy)
            ).to_numpy()
            # Step 2: shapely 2.0 vectorized containment on the candidate subset only
            pts = _shapely.points(x_vals.to_numpy()[bbox_mask], y_vals.to_numpy()[bbox_mask])
            within = _shapely.contains(shape, pts)
            # Merge back into a full-length boolean mask
            mask = bbox_mask.copy()
            mask[mask] = within
        else:
            # Dask path (only reached when materialize=False).
            # dask_geopandas only supports flat column names, so the grouped
            # multi-level column case always falls through to compute-then-filter.
            dask_mask_set = False
            if not grouped_df:
                try:
                    import dask_geopandas as dask_gpd
                    mask = dask_gpd.points_from_xy(
                        df=transcript_df, x="x_location", y="y_location"
                    ).within(shape)
                    dask_mask_set = True
                except ImportError:
                    warn("dask-geopandas not installed; loading transcripts into memory.")

            if not dask_mask_set:
                # Compute to pandas and use the vectorized fast path
                transcript_df = transcript_df.compute()
                x_vals = transcript_df.loc[:, ("coordinates", "x")] if grouped_df else transcript_df["x_location"]
                y_vals = transcript_df.loc[:, ("coordinates", "y")] if grouped_df else transcript_df["y_location"]
                bbox_mask = (
                    (x_vals >= minx) & (x_vals <= maxx) &
                    (y_vals >= miny) & (y_vals <= maxy)
                ).to_numpy()
                pts = _shapely.points(x_vals.to_numpy()[bbox_mask], y_vals.to_numpy()[bbox_mask])
                within = _shapely.contains(shape, pts)
                mask = bbox_mask.copy()
                mask[mask] = within

    else:
        if xlim is None or ylim is None:
            raise ValueError("Either both xlim and ylim must be provided, or shape must be provided.")

        try:
            # infer mask for selection
            xmask = (transcript_df["coordinates", "x"] >= xlim[0]) & (transcript_df["coordinates", "x"] <= xlim[1])
            ymask = (transcript_df["coordinates", "y"] >= ylim[0]) & (transcript_df["coordinates", "y"] <= ylim[1])
            grouped_df = True
        except KeyError:
            xmask = (transcript_df["x_location"] >= xlim[0]) & (transcript_df["x_location"] <= xlim[1])
            ymask = (transcript_df["y_location"] >= ylim[0]) & (transcript_df["y_location"] <= ylim[1])
            grouped_df = False

        # create filtering mask
        mask = xmask & ymask

        # get minimum x and y for shifting the coordinates after cropping
        minx = xlim[0]
        miny = ylim[0]

    # select
    transcript_df = transcript_df.loc[mask, :].copy()

    if grouped_df:
        # move origin again to 0 by subtracting the lower limits from the coordinates
        transcript_df["coordinates", "x"] -= minx
        transcript_df["coordinates", "y"] -= miny
    else:
        # move origin again to 0 by subtracting the lower limits from the coordinates
        transcript_df["x_location"] -= minx
        transcript_df["y_location"] -= miny

    # Re-wrap pandas result as Dask to satisfy the transcripts setter contract.
    if isinstance(transcript_df, pd.DataFrame):
        n_partitions = max(1, min(8, len(transcript_df) // 2_000_000))
        transcript_df = dd.from_pandas(transcript_df, npartitions=n_partitions)

    return transcript_df



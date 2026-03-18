import ast
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Union

import geopandas
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame

from ..utils.utils import convert_to_list

logger = logging.getLogger(__name__)

# force geopandas to use shapely. Default in future versions of geopandas.
os.environ['USE_PYGEOS'] = '0'


def parse_geopandas(
    data: Union[GeoDataFrame, pd.DataFrame, dict,
                str, os.PathLike, Path],
    uid_col: str = "id"
    ):
    """Parse geometry data from various input types into a GeoDataFrame.

    Accepts a GeoDataFrame, DataFrame, dict, or a GeoJSON file path and
    returns a normalised GeoDataFrame with CRS set to EPSG:4326 and
    *uid_col* as the index.  Returns None if the data is empty.

    Args:
        data: Geometry data as a GeoDataFrame, pandas DataFrame, dict
            with a ``"geometry"`` key, or a path to a ``.geojson`` file.
        uid_col: Column name used as the index.  Defaults to ``"id"``.

    Returns:
        A :class:`~geopandas.GeoDataFrame` or None if the data is empty.

    Raises:
        ValueError: If *data* is a file path with an unsupported extension.
    """
    # check if the input is a path or a GeoDataFrame
    if isinstance(data, GeoDataFrame):
        df = data
        df["origin"] = "manual"
    elif isinstance(data, pd.DataFrame) or isinstance(data, dict):
        df = GeoDataFrame(data, geometry=data["geometry"])
        df["origin"] = "manual"
    else:
        # read annotations as GeoDataFrame
        data = Path(data)
        if data.suffix == ".geojson":
            df = read_qupath_geojson(file=data)
            df["origin"] = "file"
        else:
            raise ValueError(f"Unknown file extension: {data.suffix}. File is expected to be `.geojson` or `.parquet`.")

    if len(df) > 0:
        # set the crs to EPSG:4326 (does not matter for us but to circumvent errors it is better to set it)
        df = df.set_crs(4326)

        if df.index.name != uid_col:
            # set uid column as index
            df = df.set_index(uid_col)

        return df
    else:
        # empty data object
        return None

def _read_file_helper(file, engine):
    dataframe = geopandas.read_file(file, engine=engine)
    if engine == "pyogrio" and "classification" in dataframe.columns:
        # convert string representations of dicts to actual dicts
        # only if they are strings (pyogrio may already parse them as dicts)
        def safe_literal_eval(val):
            """Parse val as JSON or Python literal; return as-is if parsing fails."""
            if isinstance(val, dict):
                return val

            logger.debug(val)
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    try:
                        return ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        return val
            return val
        dataframe['classification'] = dataframe['classification'].apply(safe_literal_eval)
    return dataframe

def read_qupath_geojson(file: Union[str, os.PathLike, Path]) -> pd.DataFrame:
    """
    Reads a QuPath-compatible GeoJSON file and transforms it into a flat DataFrame.

    Parameters:
    - file (Union[str, os.PathLike, Path]): The file path (as a string or pathlib.Path) of the QuPath GeoJSON file.

    Returns:
    pandas.DataFrame: A DataFrame with flattened columns including "name" and "color" extracted from the "classification" column.
    """
    # Read the GeoJSON file into a GeoDataFrame
    dataframe = _read_file_helper(file=file, engine="pyogrio")

    # annotation geojsons contain a classification column where each entry is a dict with name and color of the annotation
    if "classification" in dataframe.columns:
        # print(dataframe)
        # print(dataframe["classification"])
        # Flatten the "classification" column into separate "name" and "color" columns
        if "name" in dataframe.columns:
            warnings.warn(
                "The geometries contain both a 'name' (set e.g. by 'Set properties' in QuPath) and a 'classification name'.\n"
                "Currently, the `read_qupath_geojson` function overwrites the name with the classification name and saves it in a column named just 'name'.\n"
                "This behavior might change in the future.",
                UserWarning,
                stacklevel=2,
            )
        def _extract_classification_value(entry, key, default):
            if isinstance(entry, dict):
                return entry.get(key, default)
            return default

        dataframe["name"] = [_extract_classification_value(elem, "name", "unclassified") for elem in dataframe["classification"]]
        dataframe["color"] = [_extract_classification_value(elem, "color", [0,0,0]) for elem in dataframe["classification"]]
        dataframe["scale"] = [_extract_classification_value(elem, "scale", (1,1)) for elem in dataframe["classification"]]

        # Remove the redundant "classification" column
        dataframe = dataframe.drop(["classification"], axis=1)

    # Exported TMA cores instead contain the columns 'name' and 'isMissing'. These we just leave.

    # Return the transformed DataFrame
    return dataframe

def write_qupath_geojson(dataframe: GeoDataFrame,
                         file: Union[str, os.PathLike, Path]
                         ):
    """
    Converts a GeoDataFrame with "name" and "color" columns into a QuPath-compatible GeoJSON-like format,
    adding a new "classification" column containing dictionaries with "name" and "color" entries.
    The modified GeoDataFrame is then saved to the specified GeoJSON file.

    Parameters:
    - dataframe (geopandas.GeoDataFrame): The input GeoDataFrame containing "name" and "color" columns.
    - file (Union[str, os.PathLike, Path]): The file path (as a string or pathlib.Path) where the GeoJSON data will be saved.
    """
    columns_to_move = ["name", "color", "scale"]
    if np.any([elem in dataframe.columns for elem in columns_to_move]):
        existing_columns_to_move = [elem for elem in columns_to_move if elem in dataframe.columns]

        # Initialize an empty list to store dictionaries for each row
        classification_list = []

        # Iterate over rows in the GeoDataFrame
        for _, row in dataframe.iterrows():
            # Create a dictionary with "name" and "color" entries for each row
            classification_dict = {}

            for column in existing_columns_to_move:
                entry = row[column]

                # convert numpy arrays to lists
                if isinstance(entry, np.ndarray):
                    entry = convert_to_list(entry)
                elif isinstance(entry, tuple):
                    entry = convert_to_list(entry)

                classification_dict[column] = entry
            # Append the dictionary to the list
            classification_list.append(classification_dict)

        # Add a new "classification" column to the GeoDataFrame
        dataframe["classification"] = classification_list

        # Remove the original "name" and "color" columns
        dataframe = dataframe.drop(existing_columns_to_move, axis=1)

    # Write the GeoDataFrame to a GeoJSON file
    dataframe.to_file(file, driver="GeoJSON")


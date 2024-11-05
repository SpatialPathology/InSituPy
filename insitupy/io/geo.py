import os
import warnings
from pathlib import Path
from typing import Union

import geopandas
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame

from ..utils.utils import convert_to_list

# force geopandas to use shapely. Default in future versions of geopandas.
os.environ['USE_PYGEOS'] = '0'


def parse_geopandas(
    data: Union[GeoDataFrame, pd.DataFrame, dict,
                str, os.PathLike, Path],
    uid_col: str = "id"
    ):
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

    # set the crs to EPSG:4326 (does not matter for us but to circumvent errors it is better to set it)
    df = df.set_crs(4326)

    if df.index.name != uid_col:
        # set uid column as index
        df = df.set_index(uid_col)

    return df

def read_qupath_geojson(file: Union[str, os.PathLike, Path]) -> pd.DataFrame:
    """
    Reads a QuPath-compatible GeoJSON file and transforms it into a flat DataFrame.

    Parameters:
    - file (Union[str, os.PathLike, Path]): The file path (as a string or pathlib.Path) of the QuPath GeoJSON file.

    Returns:
    pandas.DataFrame: A DataFrame with flattened columns including "name" and "color" extracted from the "classification" column.
    """
    # Read the GeoJSON file into a GeoDataFrame
    dataframe = geopandas.read_file(file)

    # annotation geojsons contain a classification column where each entry is a dict with name and color of the annotation
    if "classification" in dataframe.columns:
        # Flatten the "classification" column into separate "name" and "color" columns
        if "name" in dataframe.columns:
            warnings.warn(
                (
                    f"The geometries contain both a 'name' (set e.g. by 'Set properties' in QuPath) and a 'classification name'.\n"
                    f"Currently, the `read_qupath_geojson` function overwrites the name with the classification name and saves it in a column named just 'name'.\n"
                    f"This behavior might change in the future."
                )
                )
        try:
            dataframe["name"] = [elem["name"] if pd.notnull(elem) else "unclassified" for elem in dataframe["classification"]]
        except KeyError:
            pass
        try:
            dataframe["color"] = [elem["color"] if pd.notnull(elem) else [0,0,0] for elem in dataframe["classification"]]
        except KeyError:
            pass

        try:
            dataframe["scale"] = [elem["scale"] if pd.notnull(elem) else (1,1) for elem in dataframe["classification"]]
        except KeyError:
            pass

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

                classification_dict[column] = entry
            # Append the dictionary to the list
            classification_list.append(classification_dict)

        # Add a new "classification" column to the GeoDataFrame
        dataframe["classification"] = classification_list

        # Remove the original "name" and "color" columns
        dataframe = dataframe.drop(existing_columns_to_move, axis=1)

    # try:
    #     # remove scale column
    #     dataframe = dataframe.drop("scale", axis=1)
    # except KeyError:
    #     pass

    # Write the GeoDataFrame to a GeoJSON file
    dataframe.to_file(file, driver="GeoJSON")


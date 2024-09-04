import os
from pathlib import Path
from typing import Union

import geopandas as gpd
import shapely

from insitupy.io.files import read_json


def read_baysor_polygons(
    file: Union[str, os.PathLike, Path]
    ) -> gpd.GeoDataFrame:

    d = read_json(file)

    # prepare output dictionary
    df = {
    "geometry": [],
    "cell": [],
    "type": [],
    "minx": [],
    "miny": [],
    "maxx": [],
    "maxy": []
    }

    for elem in d["geometries"]:
        coords = elem["coordinates"][0]

        # check if there are enough coordinates for a Polygon (some segmented cells are very small in Baysor)
        if len(coords) > 3:
            p = shapely.Polygon(coords)
            df["geometry"].append(p)
            df["type"].append("polygon")

        else:
            p = shapely.LineString(coords)
            df["geometry"].append(p)
            df["type"].append("line")
        df["cell"].append(elem["cell"])

        # extract bounding box
        bounds = p.bounds
        df["minx"].append(bounds[0])
        df["miny"].append(bounds[1])
        df["maxx"].append(bounds[2])
        df["maxy"].append(bounds[3])

    # create geopandas dataframe
    df = gpd.GeoDataFrame(df)

    return df
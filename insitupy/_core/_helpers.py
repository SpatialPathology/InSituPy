from warnings import warn

import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon

try:
    from rasterio.features import rasterize
except ImportError:
    raise ImportError("This function requires the rasterio package, please install with `pip install rasterio`.")
from datetime import datetime

import dask.array as da


def _get_expression_values(adata, X, key_type, key):
    # get expression values
    if key_type == "genes":
        gene_loc = adata.var_names.get_loc(key)
        color_value = X[:, gene_loc]
    elif key_type == "obs":
        color_value = adata.obs[key]
    elif key_type == "obsm":
        #TODO: Implement it for obsm
        obsm_key = key.split("#", maxsplit=1)[0]
        obsm_col = key.split("#", maxsplit=1)[1]
        data = adata.obsm[obsm_key]

        if isinstance(data, pd.DataFrame):
            color_value = data[obsm_col].values
        elif isinstance(data, np.ndarray):
            color_value = data[:, int(obsm_col)-1]
        else:
            warn("Data in `obsm` needs to be either pandas DataFrame or numpy array to be parsed.")
        pass
    else:
        print("Unknown key selected.", flush=True)

    return color_value

def _convert_to_float_coords(coords, mode):
    # Convert Decimal to float

    # if len(coords) == 1:
    if mode == "Polygon":
        float_coords = [[(float(x), float(y)) for x, y in ring] for ring in coords]
        poly = Polygon(float_coords[0])
    elif mode == "MultiPolygon":
        float_coords = [[[(float(x), float(y)) for x, y in ring] for ring in poly] for poly in coords]
        poly = MultiPolygon(float_coords)
    return poly

def _generate_mask(values, xmax, ymax, seg_mask_value):
    # rasterize polygons
    boundaries_mask = rasterize(
        list(zip(values, seg_mask_value)),
        out_shape=(ymax,xmax))
    boundaries_mask = da.from_array(boundaries_mask)

    return boundaries_mask


def sort_paths_by_datetime(paths):
    def extract_datetime(path):
        # Assumes ID format: "250805-115555000343-2c58ca86"
        parts = path.name.split("-")
        date_part = parts[0]  # "250805"
        time_part = parts[1]  # "115555000343"

        # Combine into full datetime string: "250805115555000343"
        full_dt_str = date_part + time_part

        # Parse as datetime: YYMMDDHHMMSSffffff
        return datetime.strptime(full_dt_str, "%y%m%d%H%M%S%f")

    return sorted(paths, key=extract_datetime, reverse=True)

import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon

from insitupy._core._helpers import _convert_to_float_coords, _generate_mask
from insitupy._core.dataclasses import BoundariesData

try:
    from rasterio.features import rasterize
except ImportError:
    raise ImportError("This function requires the rasterio package, please install with `pip install rasterio`.")

from math import ceil

import dask.array as da

from insitupy.io.geo import parse_geopandas
from insitupy.utils._shapely import scale_polygon
from insitupy.utils.utils import convert_int_to_xenium_hex


def _read_measurements(
    measurements_dict,
    coordinates_path,
    metadata_path,
    xshift, yshift
    ) -> anndata.AnnData:

    if not isinstance(measurements_dict, dict):
        raise ValueError(f"`measurements_dict` must be a dictionary not '{type(measurements_dict)}'.")

    main_key = list(measurements_dict.keys())[0]

    measurements = {}
    for n, path in measurements_dict.items():
        measurements[n] = pd.read_csv(path, index_col=0)

    # Extract metadata
    metadata = pd.read_csv(metadata_path, index_col=0)

    # # Extract measurements into a dictionary
    # measurement_types = ["Nucleus", "Cytoplasm", "Membrane", "Cell"]
    # measurements = {
    #     mtype.lower(): df.loc[:, df.columns.str.contains("Mean") & df.columns.str.contains(f"{mtype}:")].copy()
    #     for mtype in measurement_types
    # }

    # # Format column names
    # for m in measurements.values():
    #     m.columns = [col.split(":")[1].strip() for col in m.columns]

    # # Move DAPI mean to metadata and drop it from nucleus measurements
    # metadata["DAPI_mean"] = measurements["nucleus"]["DAPI-01"]
    # for m in measurements.values():
    #     m.drop(columns=["DAPI-01"], inplace=True, errors="ignore")

    # Extract and format coordinates
    coordinates = pd.read_csv(coordinates_path, index_col=0)
    # coordinates = df.loc[:, df.columns.str.contains("Centroid")].copy()
    # coordinates.columns = ["x", "y"]

    # shift coordinates to annotation origin
    coordinates["x"] -= xshift
    coordinates["y"] -= yshift

    # # Set index
    # cell_names = [convert_int_to_xenium_hex(i) for i in range(len(metadata))]
    # metadata.index = coordinates.index = cell_names
    # for m in measurements.values():
    #     m.index = cell_names

    # # filter out cells without nucleus measurements
    # ids_wo_na = ~measurements["nucleus"].isna().any(axis=1)
    # metadata = metadata.loc[ids_wo_na, :]
    # coordinates = coordinates.loc[ids_wo_na, :]

    # for n, m in measurements.items():
    #     measurements[n] = m.loc[ids_wo_na, :]

    adata = anndata.AnnData(measurements[main_key])

    for n, m in measurements.items():
        if n != main_key:
            adata.layers[n] = m.values

    # add metadata and coordinates
    adata.obs = pd.merge(left=adata.obs, right=metadata, left_index=True, right_index=True)
    adata.obsm["spatial"] = coordinates.values

    return adata

def _read_boundaries(
    cells_path,
    nuclei_path,
    xshift, yshift,
    pixel_size
    ) -> BoundariesData:
    cells_path = Path(cells_path)
    nuclei_path = Path(nuclei_path)

    # --- Read the nuclear and cellular geometries ---
    cells = parse_geopandas(cells_path).rename(columns={"geometry": "cells_geometry"})
    nuclei = parse_geopandas(nuclei_path).rename(columns={"geometry": "nuclei_geometry"})

    bounds = pd.merge(left=nuclei, right=cells,
                      left_index=True, right_index=True)

    # move the polygons to the annotation origin
    bounds["cells_geometry"] = bounds["cells_geometry"].translate(
        xoff=-xshift/pixel_size, yoff=-yshift/pixel_size
        )
    bounds["nuclei_geometry"] = bounds["nuclei_geometry"].translate(
        xoff=-xshift/pixel_size, yoff=-yshift/pixel_size
        )

    # get segmentation mask values for rasterization
    seg_mask_value = range(1, len(bounds)+1)

    # Calculate bounds for rasterization
    polygon_bounds = bounds["cells_geometry"].bounds
    xmax = ceil(polygon_bounds.loc[:, "maxx"].max())
    ymax = ceil(polygon_bounds.loc[:, "maxy"].max())

    # Convert data into segmentation masks
    cellbounds_mask = _generate_mask(
        bounds["cells_geometry"],
        xmax=xmax, ymax=ymax,
        seg_mask_value=seg_mask_value)
    nucbounds_mask = _generate_mask(
        bounds["nuclei_geometry"],
        xmax=xmax, ymax=ymax,
        seg_mask_value=seg_mask_value)

    # --- Create BoundariesData object ---
    boundaries = BoundariesData(
        cell_names=bounds.index.values,
        seg_mask_value=seg_mask_value
    )

    boundaries.add_boundaries(
        cell_boundaries=cellbounds_mask,
        pixel_size=pixel_size,
        nuclei_boundaries=nucbounds_mask
    )

    return boundaries
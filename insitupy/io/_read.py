import json
from math import ceil
from pathlib import Path

import anndata
import dask.array as da
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon

from insitupy.dataclasses.dataclasses import BoundariesData
from insitupy.io.geo import parse_geopandas
from insitupy.utils._helpers import _convert_to_float_coords, _generate_mask
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

    if metadata_path is not None:
        # Extract metadata
        metadata = pd.read_csv(metadata_path, index_col=0)
    else:
        metadata = None

    # Extract and format coordinates
    coordinates = pd.read_csv(coordinates_path, index_col=0)

    # shift coordinates to annotation origin
    coordinates["x"] -= xshift
    coordinates["y"] -= yshift

    # create an AnnData object
    adata = anndata.AnnData(measurements[main_key])

    for n, m in measurements.items():
        if n != main_key:
            adata.layers[n] = m.values

    if metadata is not None:
        # add metadata
        adata.obs = pd.merge(left=adata.obs, right=metadata, left_index=True, right_index=True)

    # add coordinates
    adata.obsm["spatial"] = coordinates.values

    return adata

def _read_boundaries(
    cells_path,
    nuclei_path,
    xshift, yshift,
    pixel_size
    ) -> BoundariesData:
    # --- Read the nuclear and cellular geometries ---
    cells = parse_geopandas(cells_path).rename(columns={"geometry": "cells_geometry"})

    if nuclei_path is not None:
        add_nuclei = True
        nuclei = parse_geopandas(nuclei_path).rename(columns={"geometry": "nuclei_geometry"})

        bounds = pd.merge(left=nuclei, right=cells,
                        left_index=True, right_index=True)
    else:
        add_nuclei = False
        bounds = cells

    # move the polygons to the annotation origin
    bounds["cells_geometry"] = bounds["cells_geometry"].translate(
        xoff=-xshift/pixel_size, yoff=-yshift/pixel_size
        )

    if add_nuclei:
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

    if add_nuclei:
        nucbounds_mask = _generate_mask(
            bounds["nuclei_geometry"],
            xmax=xmax, ymax=ymax,
            seg_mask_value=seg_mask_value)
    else:
        nucbounds_mask = None

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
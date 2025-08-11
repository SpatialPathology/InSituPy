import json
from math import ceil
from pathlib import Path

import anndata
import pandas as pd
from parse import *

from insitupy import __version__
from insitupy._io.geo import parse_geopandas
from insitupy.utils._helpers import _convert_to_float_coords, _generate_mask
from insitupy.utils.utils import convert_int_to_xenium_hex


def _get_pixel_size_from_qupath_metadata(metadata, name):
    for elem in metadata["images"]:
        if elem["serverBuilder"]["metadata"]["name"] == name:
            return elem["serverBuilder"]["metadata"]["pixelCalibration"]["pixelWidth"]["value"]

def _list_insitupy_data_folders(
    data_path):
    # Define the path to the 'insitupy' export folder
    data_path = Path(data_path)

    # Initialize a dictionary to store dataset subdirectory paths
    dataset_paths = {}

    # Check if the 'insitupy' folder exists
    if not data_path.exists():
        print(f"No 'insitupy' folder found at {data_path}")
        return dataset_paths

    # Iterate through the contents of the 'insitupy' folder
    for item in data_path.iterdir():
        if item.is_dir():
            # Collect full paths to subdirectories (datasets) within each data folder
            subdirs = [d for d in item.iterdir() if d.is_dir()]

            # if subdirectories were found, add it to the dictionary
            if len(subdirs) > 1:
                dataset_paths[item.name] = subdirs

    # Print summary of folders and datasets found
    print(f"Data folders found:")
    for name, paths in dataset_paths.items():
        print(f"\t- '{name}': {len(paths)} dataset(s)")
    return dataset_paths

def _read_measurements_qupath(
    path, xshift, yshift
    ) -> anndata.AnnData:
    path = Path(path)
    df = pd.read_csv(path, sep="\t")

    # Extract metadata
    metadata = df[[
        "Object ID", 'Centroid X µm', 'Centroid Y µm',
        'Nucleus: Area µm^2', 'Nucleus: Length µm', 'Nucleus: Circularity', 'Nucleus: Solidity', 'Nucleus: Max diameter µm', 'Nucleus: Min diameter µm',
        'Cell: Area µm^2', 'Cell: Length µm', 'Cell: Circularity', 'Cell: Solidity', 'Cell: Max diameter µm', 'Cell: Min diameter µm'
    ]].copy()

    # Extract measurements into a dictionary
    measurement_types = ["Nucleus", "Cytoplasm", "Membrane", "Cell"]
    measurements = {
        mtype.lower(): df.loc[:, df.columns.str.contains("Mean") & df.columns.str.contains(f"{mtype}:")].copy()
        for mtype in measurement_types
    }

    # Format column names
    for m in measurements.values():
        m.columns = [col.split(":")[1].strip() for col in m.columns]

    # # Move DAPI mean to metadata and drop it from measurements
    # metadata["DAPI_mean"] = measurements["nucleus"]["DAPI-01"]
    # for m in measurements.values():
    #     m.drop(columns=["DAPI-01"], inplace=True, errors="ignore")

    # Extract and format coordinates
    coordinates = df.loc[:, df.columns.str.contains("Centroid")].copy()
    coordinates.columns = ["x", "y"]

    # shift coordinates to annotation origin
    coordinates["x"] -= xshift
    coordinates["y"] -= yshift

    # Set index
    cell_names = [convert_int_to_xenium_hex(i) for i in range(len(metadata))]
    metadata.index = coordinates.index = cell_names
    for m in measurements.values():
        m.index = cell_names

    # filter out cells without nucleus measurements
    ids_wo_na = ~measurements["nucleus"].isna().any(axis=1)
    metadata = metadata.loc[ids_wo_na, :]
    coordinates = coordinates.loc[ids_wo_na, :]

    for n, m in measurements.items():
        measurements[n] = m.loc[ids_wo_na, :]

    adata = anndata.AnnData(measurements["cell"])

    for n, m in measurements.items():
        if n != "cell":
            adata.layers[n] = m.values

    # add metadata and coordinates
    adata.obs = pd.merge(left=adata.obs, right=metadata, left_index=True, right_index=True)
    adata.obsm["spatial"] = coordinates.values

    return adata

def _read_boundaries_qupath(
    bound_path,
    object_ids,
    cell_names,
    xshift, yshift,
    pixel_size
    ):
    from insitupy.dataclasses.dataclasses import BoundariesData

    bound_path = Path(bound_path)

    # --- Read the cellular geometries ---
    bounds = parse_geopandas(bound_path)

    # --- Read the nuclear geometries ---
    # Load the GeoJSON file
    with open(bound_path, 'r') as f:
        data = json.load(f)

    nucleus_geom = []
    for feature in data['features']:
        try:
            geom = feature['nucleusGeometry']
        except KeyError:
            nucleus_geom.append(None)
        else:
            coords = geom['coordinates']
            mode = geom['type']

            poly = _convert_to_float_coords(coords, mode)

            nucleus_geom.append(poly)

    # --- Format the boundaries data ---
    # add the nucleus geometry to the dataframe
    bounds["nucleus_geometry"] = nucleus_geom

    # convert nucleus_geometry to geoseries
    bounds["nucleus_geometry"] = bounds["nucleus_geometry"].astype("geometry")

    # select only cells that were not filtered out yet
    bounds = bounds.loc[object_ids]

    # add names from metadata
    bounds["name"] = cell_names

    # move the polygons to the annotation origin
    bounds["geometry"] = bounds["geometry"].translate(
        xoff=-xshift/pixel_size, yoff=-yshift/pixel_size
        )
    bounds["nucleus_geometry"] = bounds["nucleus_geometry"].translate(
        xoff=-xshift/pixel_size, yoff=-yshift/pixel_size
        )

    seg_mask_value = range(1, len(bounds)+1)

    # Calculate bounds for rasterization
    polygon_bounds = bounds["geometry"].bounds
    xmax = ceil(polygon_bounds.loc[:, "maxx"].max())
    ymax = ceil(polygon_bounds.loc[:, "maxy"].max())

    # Convert data into segmentation masks
    cellbounds_mask = _generate_mask(
        bounds["geometry"],
        xmax=xmax, ymax=ymax,
        seg_mask_value=seg_mask_value)
    nucbounds_mask = _generate_mask(
        bounds["nucleus_geometry"],
        xmax=xmax, ymax=ymax,
        seg_mask_value=seg_mask_value)

    # --- Create BoundariesData object ---
    boundaries = BoundariesData(
        cell_names=bounds["name"].values,
        seg_mask_value=seg_mask_value
    )

    boundaries.add_boundaries(
        cell_boundaries=cellbounds_mask,
        pixel_size=pixel_size,
        nuclei_boundaries=nucbounds_mask
    )

    return boundaries

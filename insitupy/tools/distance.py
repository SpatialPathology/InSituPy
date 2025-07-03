from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from insitupy._core._utils import _get_cell_layer
from insitupy._core.insitudata import InSituData


def calc_distance_of_cells_from(
    data: InSituData,
    annotation_key: str,
    annotation_class: str,
    cells_layer: Optional[str] = None,
    region_key: Optional[str] = None,
    region_name: Optional[str] = None,
    key_to_save: Optional[str] = None
    ):

    """
    Calculate the distance of cells from a specified annotation class within a given region and save the results.

    This function calculates the distance of each cell in the spatial data to the closest point
    of a specified annotation class. The distances are then saved in the cell data matrix.

    Args:
        data (InSituData): The input data containing cell and annotation information.
        annotation_key (str): The key to retrieve the annotation information.
        annotation_class (Optional[str]): The specific annotation class to calculate distances from.
        region_key: (Optional[str]): If not None, `region_key` is used together with `region_name` to determine the region in which cells are considered
                                     for the analysis.
        region_name: (Optional[str]): If not None, `region_name` is used together with `region_key` to determine the region in which cells are considered
                                     for the analysis.
        key_to_save (Optional[str]): The key under which to save the calculated distances in the cell data matrix.
                                     If None, a default key is generated based on the annotation class.

    Returns:
        None
    """
    # extract anndata object
    celldata, cells_layer_name = _get_cell_layer(
        cells=data.cells, cells_layer=cells_layer, verbose=True, return_layer_name=True
        )
    adata = celldata.matrix

    if region_name is None:
        print(f'Calculate the distance of cells from the annotation "{annotation_class}"')
        region_mask = [True] * len(adata)
    else:
        assert region_key is not None, "`region_key` must not be None if `region_name` is not None."
        print(f'Calculate the distance of cells from the annotation "{annotation_class}" within region "{region_name}"')

        try:
            region_df = adata.obsm["regions"]
        except KeyError:
            data.assign_regions(keys=region_key)
            region_df = adata.obsm["regions"]
        else:
            if region_key not in region_df.columns:
                data.assign_regions(keys=region_key)

        # generate mask for selected region
        region_mask = region_df[region_key] == region_name

    # create geopandas points from cells
    x = adata.obsm["spatial"][:, 0][region_mask]
    y = adata.obsm["spatial"][:, 1][region_mask]
    indices = adata.obs_names[region_mask]
    cells = gpd.points_from_xy(x, y)

    # retrieve annotation information
    annot_df = data.annotations[annotation_key]
    class_df = annot_df[annot_df["name"] == annotation_class]

    # calculate distance of cells to their closest point
    # scaled_geometries = [
    #     scale_func(geometry, xfact=scale[0], yfact=scale[1], origin=(0,0))
    #     for geometry, scale in zip(class_df["geometry"], class_df["scale"])
    #     ]
    scaled_geometries = class_df["geometry"].tolist()
    dists = np.array([cells.distance(geometry) for geometry in scaled_geometries])
    min_dists = dists.min(axis=0)

    # add indices to minimum distances
    min_dists = pd.Series(min_dists, index=indices)

    # add results to CellData
    if key_to_save is None:
        #key_to_save = f"dist_from_{annotation_class}"
        key_to_save = annotation_class
    #adata.obs[key_to_save] = min_dists

    obsm_keys = adata.obsm.keys()
    if "distance_from" not in obsm_keys:
        # add empty pandas dataframe with obs_names as index
        adata.obsm["distance_from"] = pd.DataFrame(index=adata.obs_names)

    adata.obsm["distance_from"][key_to_save] = min_dists
    print(f'Saved distances to `.cells[{cells_layer_name}].matrix.obsm["distance_from"]["{key_to_save}"]`')
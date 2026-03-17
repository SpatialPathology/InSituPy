import logging
import warnings
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from insitupy._core.data import InSituData
from insitupy.dataclasses._utils import _get_cell_layer

logger = logging.getLogger(__name__)


def calc_distance_of_cells_from(
    data: InSituData,
    annotation_tuple: Tuple[str, str] = None,
    annotation_key: Optional[str] = None,
    annotation_class: Optional[str] = None,
    cells_layer: Optional[str] = None,
    region_tuple: Optional[Tuple[str, str]] = None,
    region_key: Optional[str] = None,
    region_name: Optional[str] = None,
    key_to_save: Optional[str] = None
    ):

    """
    Calculate the distance of cells from a specified annotation class within a given region.

    Computes, for each cell, the distance to the closest point of the target annotation
    geometry and stores the result in
    ``data.cells[cells_layer].table.obsm["distance_from"][key_to_save]``.

    Args:
        data (InSituData): The input data containing cell and annotation information.
        annotation_tuple (Tuple[str, str]): Annotation specifier as ``(key, name)`` where
            ``key`` is the annotation category and ``name`` is the specific annotation class
            to calculate distances from.
        annotation_key (str, optional): Deprecated. Use ``annotation_tuple`` instead.
        annotation_class (str, optional): Deprecated. Use ``annotation_tuple`` instead.
        cells_layer (str, optional): Cell segmentation layer to use. Defaults to None (main layer).
        region_tuple (Tuple[str, str], optional): Region specifier as ``(key, name)`` used to
            restrict which cells are included in the analysis. Defaults to None (all cells).
        region_key (str, optional): Deprecated. Use ``region_tuple`` instead.
        region_name (str, optional): Deprecated. Use ``region_tuple`` instead.
        key_to_save (str, optional): Key under which to save the calculated distances in
            ``obsm["distance_from"]``. Defaults to None (uses the annotation name).

    Returns:
        None
    """
    if annotation_key is not None or annotation_class is not None:
        warnings.warn(
            "'annotation_key' and 'annotation_class' are deprecated. "
            "Use 'annotation_tuple=(key, name)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        annotation_tuple = (annotation_key, annotation_class)

    if annotation_tuple is None:
        raise ValueError("'annotation_tuple' must be provided.")

    if region_key is not None or region_name is not None:
        warnings.warn(
            "'region_key' and 'region_name' are deprecated. "
            "Use 'region_tuple=(key, name)' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        region_tuple = (region_key, region_name)

    annotation_key_resolved, annotation_name = annotation_tuple

    # extract anndata object
    celldata, cells_layer_name = _get_cell_layer(
        cells=data.cells, cells_layer=cells_layer, verbose=True, return_layer_name=True
        )
    adata = celldata.table

    if region_tuple is None:
        logger.info('Calculate the distance of cells from the annotation "%s"', annotation_name)
        region_mask = [True] * len(adata)
    else:
        region_key_resolved, region_name_resolved = region_tuple
        logger.info('Calculate the distance of cells from the annotation "%s" within region "%s"', annotation_name, region_name_resolved)

        try:
            region_df = adata.obsm["regions"]
        except KeyError:
            data.assign_regions(keys=region_key_resolved)
            region_df = adata.obsm["regions"]
        else:
            if region_key_resolved not in region_df.columns:
                data.assign_regions(keys=region_key_resolved)

        # generate mask for selected region
        region_mask = region_df[region_key_resolved] == region_name_resolved

    # create geopandas points from cells
    x = adata.obsm["spatial"][:, 0][region_mask]
    y = adata.obsm["spatial"][:, 1][region_mask]
    indices = adata.obs_names[region_mask]
    cells = gpd.points_from_xy(x, y)

    # retrieve annotation information
    annot_df = data.annotations[annotation_key_resolved]
    class_df = annot_df[annot_df["name"] == annotation_name]

    # calculate distance of cells to their closest point
    scaled_geometries = class_df["geometry"].tolist()
    dists = np.array([cells.distance(geometry) for geometry in scaled_geometries])
    min_dists = dists.min(axis=0)

    # add indices to minimum distances
    min_dists = pd.Series(min_dists, index=indices)

    # add results to CellData
    if key_to_save is None:
        key_to_save = annotation_name
    #adata.obs[key_to_save] = min_dists

    obsm_keys = adata.obsm.keys()
    if "distance_from" not in obsm_keys:
        # add empty pandas dataframe with obs_names as index
        adata.obsm["distance_from"] = pd.DataFrame(index=adata.obs_names)

    adata.obsm["distance_from"][key_to_save] = min_dists
    logger.info('Saved distances to `.cells[%s].table.obsm["distance_from"]["%s"]`', cells_layer_name, key_to_save)
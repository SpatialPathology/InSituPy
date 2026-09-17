import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely

from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy.containers.spatial_units_data import SpatialUnitsData
from insitupy.experiment.data import InSituExperiment


def pseudobulk_annotation(
    data: InSituExperiment | InSituData, # type: ignore
    cell_layer: str = "proseg",
    annotation_key: str = "vessel",
    counts_layer: str | None = None,
    mode: str = "mean",
    min_cells: int = 1,
    units_key: str = "vessel",
    object_name: str = "vessel",
):
    """
    Compute a pseudobulk expression profile per annotated object.

    For each polygon in ``.annotations[annotation_key]``, determines which
    cells lie inside it via a point-in-polygon test on cell centroids,
    aggregates their counts into one pseudobulk profile, and stores the
    polygon together with the pseudobulk as a new SpatialUnitsData layer in
    ``.units[units_key]``.
    Args:
        data (Union[InSituExperiment, InSituData]): Experiment or
            sample-level data object containing cells and annotations.
        cell_layer (str, optional): Name of the CellData layer. Defaults
            to "proseg".
            
        annotation_key (str, optional): Key under which the object
            polygons are stored in ``.annotations``. Defaults to "vessel".
            
        counts_layer (Optional[str], optional): Name of the AnnData layer
            with raw counts to aggregate. Defaults to None (uses ``.X``).
            
        mode (str, optional): Aggregation method over the cells inside
            each polygon -- "sum", "mean", or "median". "mean" is usually
            more comparable across objects of different size than "sum".
            Defaults to "mean".
            
        min_cells (int, optional): Objects with fewer than `min_cells`
            cells inside their polygon are excluded. Defaults to 1.
            
        units_key (str, optional): Key under which the pseudobulk layer is
            stored in ``.units``. Defaults to "vessel".
            
        object_name (str, optional): Label for the objects, used in log
            messages and as ``SpatialUnitsData.unit_type``. Defaults to
            "vessel".

    Returns:
        Union[InSituExperiment, InSituData]: The same object, with
        ``.units[units_key]`` added for each sample that has at least one
        object with both a pseudobulk and a polygon.
    """
    
    if mode not in ("sum", "mean", "median"):
        raise ValueError(f"Unknown mode '{mode}', expected 'sum', 'mean' or 'median'.")

    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = data.iterdata()
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        sample_id = xd.sample_id
        table = xd.cells[cell_layer].table

        try:
            annotation_gdf = xd.annotations[annotation_key]
        except KeyError:
            print(f"[{sample_id}] skipped: no annotation '{annotation_key}' found")
            continue

        coords = table.obsm["spatial"]
        cell_points = shapely.points(coords[:, 0], coords[:, 1])

        X = table.layers[counts_layer] if counts_layer is not None else table.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        pb_rows = []
        obs_records = []
        kept_ids = []

        for object_id, poly in zip(annotation_gdf.index, annotation_gdf["geometry"]):
            if poly is None or poly.is_empty:
                continue

            # point-in-polygon: all cell centroids inside this polygon
            sel = shapely.covers(poly, cell_points)
            n_cells = int(sel.sum())
            if n_cells < min_cells:
                continue

            sub = X[sel]
            if mode == "sum":
                agg = np.asarray(sub.sum(axis=0)).ravel()
            elif mode == "mean":
                agg = np.asarray(sub.mean(axis=0)).ravel()
            else:  # median
                agg = np.median(sub.toarray(), axis=0)

            pb_rows.append(agg)
            obs_records.append({"n_cells": n_cells})
            kept_ids.append(object_id)

        if len(pb_rows) == 0:
            print(f"[{sample_id}] skipped: no {object_name} with >= {min_cells} cells inside its polygon")
            continue

        pb_obs = pd.DataFrame(obs_records, index=pd.Index(kept_ids))
        pb_adata = ad.AnnData(
            X=np.vstack(pb_rows),
            obs=pb_obs,
            var=pd.DataFrame(index=table.var_names),
        )

        shapes = gpd.GeoDataFrame(
            {
                "name": kept_ids,
                "geometry": [annotation_gdf.loc[i, "geometry"] for i in kept_ids],
            },
            geometry="geometry",
        )

        su = SpatialUnitsData(shapes=shapes, data=pb_adata, unit_type=object_name)
        xd.add_units(su, key=units_key, overwrite=True)
        print(f"[{sample_id}] added {pb_adata.n_obs} {object_name} pseudobulk(s) to xd.units['{units_key}']")

    return data

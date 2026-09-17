import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy.experiment.data import InSituExperiment


def spatial_clustering(
    data: InSituExperiment | InSituData, # type: ignore # ohne InSituExperiment!!!
    cell_layer: str = "proseg",
    celltype_col: str = "cell_type_final",
    celltype_labels: list = ["Endothelial", "Pericyte"],
    cluster_eps: float = 20.0,
    cluster_min_samples: int = 3,
    column_name: str = "vessel_id",
    margin: float = 100.0,
    cells_compartment: str = "cells",
    annotation_key: str = "vessel",
    object_name: str = "vessel",
    sample_id: str | None = None,
):
    """
    Cluster cells into spatial objects and store their true contour as an annotation.

    Groups spatially neighboring cells of the given `celltype_labels` into
    discrete objects via DBSCAN, then builds each object's actual contour
    from the cell segmentation mask (rather than e.g. a circle around a
    centroid). Not specific to any particular structure -- use it for
    vessels, immune aggregates, glands, or any other
    spatial grouping of cells; 

    For each sample:
      1. Cells with `celltype_col` in `celltype_labels` are clustered with
         DBSCAN (spatially neighboring cells -> one object;
         Membership is written to ``obs[column_name]`` (e.g.
         ``"N1_vessel_3"``); DBSCAN noise (label -1) stays NaN.
      2. For each object, a polygon is built from the actual cell
         segmentation -- directly from ``.cells[cell_layer].boundaries``,
         The segmentation mask is cropped to a window around the involved
         cells' centroids (+ `margin`), reduced to the matching
         `seg_mask_value`s, and vectorized with
         ``rasterio.features.shapes``.
      3. The polygons are stored as an annotation in
         ``.annotations[annotation_key]`` (class = `object_name`, id = the
         ``obs[column_name]`` value) so they are available directly on the
         InSituData object 

    `margin` is a heuristic: it must exceed the largest centroid-to-edge
    distance of the involved cells, otherwise the contour gets truncated at
    the crop window. Two distinct warnings distinguish whether a larger
    `margin` would help ("mask touches crop edge") or not, because the
    window is already bounded by the image edge ("mask touches the actual
    tissue/image edge" -- the object genuinely sits at the section
    boundary, or the DBSCAN cluster became unusually large/elongated
    through chaining, e.g. from a large `cluster_eps` or additional cell
    types spanning a long structure).

    Args:
        data (Union[InSituExperiment, InSituData]): Experiment or
            sample-level data object containing cells.
        cell_layer (str, optional): Name of the CellData layer. Defaults
            to "proseg".
        celltype_col (str, optional): obs column holding cell type labels.
            Defaults to "cell_type_final".
        celltype_labels (list, optional): Cell types to cluster into
            objects (e.g. Endothelial/Pericyte for vessels). Defaults to
            ``["Endothelial", "Pericyte"]``.
        cluster_eps (float, optional): DBSCAN `eps` for clustering cells
            into objects. Defaults to 20.0.
        cluster_min_samples (int, optional): DBSCAN `min_samples`. Defaults
            to 3.
        column_name (str, optional): obs column to write object membership
            to. Freely choosable; required by `pseudobulk_annotation`.
            Defaults to "vessel_id".
        margin (float, optional): Buffer in µm around cell centroids when
            cropping the segmentation mask for polygon generation. Defaults
            to 100.0.
        cells_compartment (str, optional): "cells" or "nuclei" -- which
            segmentation mask to use. Defaults to "cells".
        annotation_key (str, optional): Key under which object polygons are
            stored in ``.annotations``. Freely choosable; required by
            `pseudobulk_annotation`. Defaults to "vessel".
        object_name (str, optional): Label for the objects -- used in ID
            strings (``f"{sample_id}_{object_name}_{n}"``), the annotation
            class, and log messages (e.g. "vessel", "niche", "gland").
            Defaults to "vessel".
        sample_id (str, optional): Override for the sample label used in ID
            strings and log messages, instead of ``xd.sample_id``. Only
            valid when `data` is a single InSituData -- raises ValueError
            if combined with an InSituExperiment, since every sample must
            keep its own sample_id (a shared override would collide object
            IDs across samples). Defaults to None (use ``xd.sample_id``).

    Returns:
        Tuple[Union[InSituExperiment, InSituData], gpd.GeoDataFrame]: The
        same `data` object, annotated in place (``obs[column_name]`` and
        ``.annotations[annotation_key]`` for each sample with at least one
        valid object), and `object_gdf`: a GeoDataFrame across all samples
        with one row per object (`column_name`, "name" [sample_id],
        "geometry", "area_um2") 
    """
    from affine import Affine
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape as shapely_shape

    is_experiment = _is_experiment(data)

    if sample_id is not None and is_experiment:
        raise ValueError(
            "`sample_id` overrides the label for a single InSituData and cannot be "
            "combined with an InSituExperiment -- each sample must keep its own "
            "sample_id, otherwise object IDs would collide across samples."
        )

    if is_experiment:
        iterator = data.iterdata()
    else:
        iterator = zip([None], [data])

    gap = 20.0  # µm, bridges gaps up to ~2*gap
    all_rows = []

    for _, xd in iterator:
        current_sample_id = sample_id if sample_id is not None else xd.sample_id
        table = xd.cells[cell_layer].table

        # --- always create column_name, even if the sample gets skipped ----
        # --- below (avoids a downstream KeyError) ---------------------------
        table.obs[column_name] = pd.array([np.nan] * table.n_obs, dtype="object")

        coords = table.obsm["spatial"]
        object_mask = table.obs[celltype_col].isin(celltype_labels).to_numpy()
        object_coords = coords[object_mask]

        if object_coords.shape[0] == 0:
            print(f"[{current_sample_id}] skipped: 0 {object_name} cells")
            continue

        # --- cluster cells into objects --------------------------------------
        clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(object_coords)
        labels = clustering.labels_
        valid = labels != -1
        n_noise = (~valid).sum()

        if valid.sum() == 0:
            print(f"[{current_sample_id}] skipped: no {object_name} clusters found (eps/min_samples too strict?)")
            continue

        # --- write column_name to obs (the object-forming cells themselves) -
        object_indices = np.where(object_mask)[0]
        object_id_labels = np.full(len(object_indices), np.nan, dtype=object)
        object_id_labels[valid] = [f"{current_sample_id}_{object_name}_{int(l)}" for l in labels[valid]]
        table.obs.iloc[object_indices, table.obs.columns.get_loc(column_name)] = object_id_labels

        n_clusters = len(pd.unique(labels[valid]))
        print(f"[{current_sample_id}] {object_coords.shape[0]} {object_name} cells -> {n_clusters} {object_name} objects "
              f"({n_noise} noise cells excluded)")

        # --- build object contours from the actual segmentation -------------
        boundaries = xd.cells[cell_layer].boundaries
        mask = boundaries[cells_compartment]
        if isinstance(mask, list):
            mask = mask[0]  # pyramid: use full resolution level
        pixel_size = boundaries.metadata[cells_compartment]["pixel_size"]

        value_by_name = dict(zip(
            boundaries.cell_names.compute(),
            boundaries.seg_mask_value.compute()
        ))

        vid = table.obs[column_name]
        obs_names = table.obs_names.to_numpy()

        sample_rows = []

        for v in pd.unique(vid.dropna()):
            sel = (vid == v).to_numpy()
            xy = coords[sel]
            cell_ids = obs_names[sel]

            values = {value_by_name.get(cid) for cid in cell_ids}
            values.discard(None)
            if not values:
                print(f"[{current_sample_id}] {v}: no matching segmentation values found, skipped")
                continue

            xmin, ymin = xy.min(axis=0) - margin
            xmax, ymax = xy.max(axis=0) + margin
            col_min = max(int(xmin / pixel_size), 0)
            row_min = max(int(ymin / pixel_size), 0)
            col_max = min(int(np.ceil(xmax / pixel_size)), mask.shape[1])
            row_max = min(int(np.ceil(ymax / pixel_size)), mask.shape[0])
            if col_max <= col_min or row_max <= row_min:
                print(f"[{current_sample_id}] {v}: empty crop window, skipped")
                continue

            crop = mask[row_min:row_max, col_min:col_max].compute()
            bool_mask = np.isin(crop, list(values))
            if not bool_mask.any():
                print(f"[{current_sample_id}] {v}: empty mask after crop, skipped")
                continue

            touches_top = bool_mask[0, :].any()
            touches_bottom = bool_mask[-1, :].any()
            touches_left = bool_mask[:, 0].any()
            touches_right = bool_mask[:, -1].any()

            if touches_top or touches_bottom or touches_left or touches_right:
                # distinguish: window bounded by `margin` (a larger margin
                # would help) vs. window already bounded by the image edge
                # (a larger margin will NOT help)
                at_image_edge = (
                    (touches_top and row_min == 0)
                    or (touches_bottom and row_max == mask.shape[0])
                    or (touches_left and col_min == 0)
                    or (touches_right and col_max == mask.shape[1])
                )
                if at_image_edge:
                    print(f"[{current_sample_id}] {v}: mask touches the actual tissue/image edge -- "
                          f"this {object_name} is genuinely cut off at the section boundary "
                          "(or the DBSCAN cluster is unusually large/elongated); "
                          "increasing `margin` will NOT help here")
                else:
                    print(f"[{current_sample_id}] {v}: mask touches crop edge "
                          f"(window rows {row_min}:{row_max} of {mask.shape[0]}, "
                          f"cols {col_min}:{col_max} of {mask.shape[1]}), "
                          "consider increasing `margin`")

            transform = Affine(pixel_size, 0, col_min * pixel_size, 0, pixel_size, row_min * pixel_size)
            geoms = [
                shapely_shape(geom)
                for geom, _ in rasterio_shapes(bool_mask.astype(np.uint8), mask=bool_mask, transform=transform)
            ]
            closed_geom = unary_union(geoms).buffer(gap).buffer(-gap)
            row = {column_name: v, "name": current_sample_id, "geometry": closed_geom}
            all_rows.append(row)
            sample_rows.append(row)

        if len(sample_rows) == 0:
            print(f"[{current_sample_id}] no {object_name} polygons could be built")
            continue

        # --- store this sample's polygons as an annotation --------------------
        sample_gdf = gpd.GeoDataFrame(
            {
                column_name: [r[column_name] for r in sample_rows],
                "class_label": [object_name] * len(sample_rows),
                "geometry": [r["geometry"] for r in sample_rows],
            },
            geometry="geometry",
        )
        xd.annotations.add_data(
            data=sample_gdf,
            key=annotation_key,
            scale_factor=1.0,
            uid_col=column_name,
            name_col="class_label",
        )
        print(f"[{current_sample_id}] saved {len(sample_rows)} {object_name} polygon(s) to .annotations['{annotation_key}']")

    object_gdf = gpd.GeoDataFrame(all_rows, geometry="geometry")
    object_gdf["area_um2"] = object_gdf.geometry.area
    return data, object_gdf

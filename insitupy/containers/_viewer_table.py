"""Viewer sidecar of a cells layer: ``X_csc.zarr`` (AnnData zarr v2, consolidated, CSC ``X``)."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)

_ZARR_V2_WARNING = "Writing zarr v2 data will no longer be the default"


def _label_ids(table: AnnData, boundaries) -> np.ndarray:
    """Return the raster label (``seg_mask_value``) of every table cell, 0 where unknown."""
    names = np.asarray(boundaries.cell_names.compute()).astype(str)
    values = np.asarray(boundaries.seg_mask_value.compute()).astype(np.uint32)
    ids = pd.Series(values, index=names)
    if ids.index.has_duplicates:
        logger.warning("Duplicate cell names in the boundaries; keeping the first label id of each.")
        ids = ids[~ids.index.duplicated(keep="first")]
    return ids.reindex(table.obs_names.astype(str)).fillna(0).astype(np.uint32).to_numpy()


def write_viewer_table(table: AnnData, boundaries, path: str | Path) -> None:
    """Write the viewer sidecar of a cells layer to *path*.

    Holds the full ``obs`` plus ``obs["_label_id"]`` (the mask label of each cell, 0 if the cell
    has no mask), ``obsm["spatial"]``, ``var`` and ``X`` in CSC layout. InSituPy never reads it.
    """
    path = Path(path)
    obs = table.obs.copy()
    if boundaries is not None and boundaries.seg_mask_value is not None:
        if "_label_id" in obs.columns:
            logger.warning("Column '_label_id' of obs is replaced in the viewer table.")
        obs["_label_id"] = _label_ids(table, boundaries)

    # always CSC, also for a dense X (e.g. after scaling): the viewer reads X by column
    X = table.X
    if X is not None:
        X = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)

    obsm = {"spatial": table.obsm["spatial"]} if "spatial" in table.obsm else {}
    sidecar = AnnData(
        X=X,
        obs=obs,
        var=table.var,
        obsm=obsm,
        uns={"_insitupy": {"role": "viewer_table", "x_on_disk": "csc", "label_id": "seg_mask_value"}},
    )

    # the warning fires even under the zarr_write_format=2 override
    with warnings.catch_warnings(), anndata.settings.override(zarr_write_format=2):
        warnings.filterwarnings("ignore", message=_ZARR_V2_WARNING)
        sidecar.write_zarr(path)

    if not (path / ".zmetadata").exists():
        raise RuntimeError(f"Viewer table at {path} was written without consolidated metadata.")

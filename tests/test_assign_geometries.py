"""Tests for InSituData.assign_geometries / assign_annotations / assign_regions (U-B2/U-H16).

Covers the corrected default write target (obsm, not obs), the add_to_obs
opt-in path, and the previously dead `add_masks`/`overwrite` flags on the
obsm path.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Polygon

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_insitudata_with_geometries():
    """4 cells at known coordinates; two named polygons under key 'zones'.

    Cells 0 and 1 fall inside polygon 'A'; cell 2 falls inside polygon 'B';
    cell 3 falls inside neither ('unassigned').
    """
    coords = np.array([
        [2.0, 2.0],
        [3.0, 3.0],
        [12.0, 12.0],
        [50.0, 50.0],
    ])
    n = len(coords)
    X = np.zeros((n, 2))
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n)]))
    var = pd.DataFrame(index=pd.Index(["g0", "g1"]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = coords

    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="s", sample_id="x",
        method_name="t", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)

    poly_a = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    poly_b = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    gdf = gpd.GeoDataFrame({
        "id": ["A_0", "B_0"],
        "name": ["A", "B"],
        "geometry": [poly_a, poly_b],
        "color": ["#ff0000", "#0000ff"],
    })
    xd._annotations.add_data(data=gdf, key="zones", scale_factor=1.0)
    xd._regions.add_data(data=gdf, key="zones", scale_factor=1.0)
    return xd


EXPECTED_LABELS = ["A", "A", "B", "unassigned"]


# ── obsm is the default target ──────────────────────────────────────────────

def test_assign_annotations_writes_to_obsm_by_default():
    xd = _make_insitudata_with_geometries()
    xd.assign_annotations(keys="zones")

    table = xd.cells["main"].table
    assert "annotations" in table.obsm
    assert list(table.obsm["annotations"]["zones"]) == EXPECTED_LABELS
    assert "zones" not in table.obs
    assert "annotations-zones" not in table.obs


def test_assign_regions_writes_to_obsm_by_default():
    xd = _make_insitudata_with_geometries()
    xd.assign_regions(keys="zones")

    table = xd.cells["main"].table
    assert "regions" in table.obsm
    assert list(table.obsm["regions"]["zones"]) == EXPECTED_LABELS
    assert "regions-zones" not in table.obs


# ── add_to_obs opt-in ────────────────────────────────────────────────────────

def test_assign_annotations_add_to_obs_writes_obs_column():
    xd = _make_insitudata_with_geometries()
    xd.assign_annotations(keys="zones", add_to_obs=True)

    table = xd.cells["main"].table
    assert "annotations-zones" in table.obs
    assert list(table.obs["annotations-zones"]) == EXPECTED_LABELS
    assert "annotations" not in table.obsm


# ── add_masks only valid with add_to_obs=True ───────────────────────────────

def test_add_masks_on_obsm_path_raises():
    xd = _make_insitudata_with_geometries()
    with pytest.raises(ValueError, match="add_to_obs"):
        xd.assign_annotations(keys="zones", add_masks=True)


def test_add_masks_with_add_to_obs_true_does_not_raise():
    xd = _make_insitudata_with_geometries()
    xd.assign_annotations(keys="zones", add_masks=True, add_to_obs=True)
    table = xd.cells["main"].table
    assert "annotations-zones" in table.obs
    # per-name mask columns are also merged in when add_masks=True
    assert "A" in table.obs.columns
    assert "B" in table.obs.columns


# ── overwrite is honored on the obsm path ───────────────────────────────────

def test_overwrite_false_skips_existing_obsm_key():
    xd = _make_insitudata_with_geometries()
    xd.assign_annotations(keys="zones")
    table = xd.cells["main"].table
    assert list(table.obsm["annotations"]["zones"]) == EXPECTED_LABELS

    # mark the column so we can detect whether a re-run touches it
    table.obsm["annotations"]["zones"] = "sentinel"

    with pytest.warns(UserWarning, match="overwrite"):
        xd.assign_annotations(keys="zones", overwrite=False)
    assert list(table.obsm["annotations"]["zones"]) == ["sentinel"] * 4

    xd.assign_annotations(keys="zones", overwrite=True)
    assert list(table.obsm["annotations"]["zones"]) == EXPECTED_LABELS


# ── zero-assignment guard (AC-A4 1) ─────────────────────────────────────────

def test_assign_annotations_warns_when_zero_cells_assigned():
    """A key whose polygons are far from every cell (e.g. a wrong
    scale_factor at import) assigns "unassigned" to every cell. This should
    warn instead of failing silently.
    """
    xd = _make_insitudata_with_geometries()

    poly_far = Polygon([(1000, 1000), (1010, 1000), (1010, 1010), (1000, 1010)])
    gdf = gpd.GeoDataFrame({
        "id": ["far_0"],
        "name": ["far"],
        "geometry": [poly_far],
        "color": ["#00ff00"],
    })
    xd._annotations.add_data(data=gdf, key="far", scale_factor=1.0)

    with pytest.warns(UserWarning, match="assigned zero cells"):
        xd.assign_annotations(keys="far")

    table = xd.cells["main"].table
    assert list(table.obsm["annotations"]["far"]) == ["unassigned"] * 4

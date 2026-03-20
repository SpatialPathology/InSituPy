"""Tests for CellData.crop(inplace=True) — verifies in-place modification.

The inplace=False path is already covered by test_celldata_sync_safeguards.py.
Note: crop() shifts coordinates by -xlim[0] / -ylim[0] after filtering, and
returns None when inplace=True.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData


def _create_table(obs_names):
    """AnnData with spatial coordinates spread along the diagonal: cell_i → (i*2, i*2)."""
    n = len(obs_names)
    table = AnnData(
        X=np.ones((n, 2)),
        obs=pd.DataFrame(index=pd.Index(obs_names, dtype=str)),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.array(
        [[float(i * 2), float(i * 2)] for i in range(n)]
    )
    return table


def _create_celldata_no_boundaries():
    """3 cells at (0,0), (2,2), (4,4) with no segmentation boundaries."""
    return CellData(table=_create_table(["c1", "c2", "c3"]), boundaries=None)


def _create_celldata_with_boundaries():
    """3 cells at (0,0), (2,2), (4,4) with a segmentation mask."""
    boundaries = BoundariesData(
        cell_names=["c1", "c2", "c3"],
        seg_mask_value=[1, 2, 3],
    )
    mask = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32)
    boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=None, pixel_size=1)
    return CellData(table=_create_table(["c1", "c2", "c3"]), boundaries=boundaries)


# ── inplace=True without boundaries ──────────────────────────────────────────

class TestCropInplaceNoBoundaries:
    def test_returns_none(self):
        # crop(inplace=True) returns None; the original is modified instead
        celldata = _create_celldata_no_boundaries()
        result = celldata.crop(xlim=(0, 2), ylim=(0, 2), inplace=True)
        assert result is None

    def test_obs_names_updated_in_place(self):
        # xlim=(0,2), ylim=(0,2): c1 at (0,0) and c2 at (2,2) are inside (inclusive)
        celldata = _create_celldata_no_boundaries()
        celldata.crop(xlim=(0, 2), ylim=(0, 2), inplace=True)
        assert list(celldata.table.obs_names) == ["c1", "c2"]

    def test_spatial_coordinates_shifted_in_place(self):
        # After crop with xlim=(2,10), ylim=(2,10) → keep c2 (2,2) and c3 (4,4)
        # Coordinates are shifted by -xlim[0]=−2 and -ylim[0]=−2
        # → c2 becomes (0,0), c3 becomes (2,2)
        celldata = _create_celldata_no_boundaries()
        celldata.crop(xlim=(2, 10), ylim=(2, 10), inplace=True)
        assert list(celldata.table.obs_names) == ["c2", "c3"]
        np.testing.assert_allclose(
            celldata.table.obsm["spatial"],
            np.array([[0.0, 0.0], [2.0, 2.0]]),
        )

    def test_boundaries_remain_none(self):
        celldata = _create_celldata_no_boundaries()
        celldata.crop(xlim=(0, 2), ylim=(0, 2), inplace=True)
        assert celldata.boundaries is None

    def test_crop_all_cells_included(self):
        celldata = _create_celldata_no_boundaries()
        celldata.crop(xlim=(0, 10), ylim=(0, 10), inplace=True)
        assert list(celldata.table.obs_names) == ["c1", "c2", "c3"]

    def test_crop_no_cells_gives_empty(self):
        celldata = _create_celldata_no_boundaries()
        celldata.crop(xlim=(10, 20), ylim=(10, 20), inplace=True)
        assert celldata.table.n_obs == 0


# ── inplace=True with boundaries ─────────────────────────────────────────────

class TestCropInplaceWithBoundaries:
    def test_obs_names_updated_in_place(self):
        celldata = _create_celldata_with_boundaries()
        celldata.crop(xlim=(0, 2), ylim=(0, 2), inplace=True)
        assert list(celldata.table.obs_names) == ["c1", "c2"]

    def test_boundary_cell_names_synced(self):
        celldata = _create_celldata_with_boundaries()
        celldata.crop(xlim=(0, 2), ylim=(0, 2), inplace=True)
        assert list(celldata.boundaries.cell_names.compute()) == ["c1", "c2"]


# ── inplace=False vs inplace=True consistency ─────────────────────────────────

class TestCropInplaceConsistency:
    def test_inplace_and_copy_give_same_obs_names(self):
        celldata_a = _create_celldata_no_boundaries()
        celldata_b = _create_celldata_no_boundaries()

        result_copy = celldata_b.crop(xlim=(2, 10), ylim=(2, 10), inplace=False)
        celldata_a.crop(xlim=(2, 10), ylim=(2, 10), inplace=True)

        assert list(celldata_a.table.obs_names) == list(result_copy.table.obs_names)

    def test_inplace_and_copy_give_same_coordinates(self):
        celldata_a = _create_celldata_no_boundaries()
        celldata_b = _create_celldata_no_boundaries()

        result_copy = celldata_b.crop(xlim=(2, 10), ylim=(2, 10), inplace=False)
        celldata_a.crop(xlim=(2, 10), ylim=(2, 10), inplace=True)

        np.testing.assert_allclose(
            celldata_a.table.obsm["spatial"],
            result_copy.table.obsm["spatial"],
        )

    def test_inplace_false_does_not_modify_original(self):
        celldata = _create_celldata_no_boundaries()
        original_obs = list(celldata.table.obs_names)
        celldata.crop(xlim=(2, 10), ylim=(2, 10), inplace=False)
        assert list(celldata.table.obs_names) == original_obs

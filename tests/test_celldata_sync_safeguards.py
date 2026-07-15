from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core._napari import _sync_cells_for_viewer_if_needed
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData
from insitupy.containers.multi_cell_data import MultiCellData


def _create_table(obs_names):
    table = AnnData(
        X=np.ones((len(obs_names), 2)),
        obs=pd.DataFrame(index=pd.Index(obs_names, dtype=str)),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.arange(len(obs_names) * 2, dtype=float).reshape(len(obs_names), 2)
    return table


def _create_boundaries(array_as_pyramid=False):
    boundaries = BoundariesData(
        cell_names=["c1", "c2", "c3"],
        seg_mask_value=[1, 2, 3],
    )

    mask = np.array(
        [
            [0, 1, 2],
            [3, 2, 1],
        ],
        dtype=np.uint32,
    )
    nuclei = np.array(
        [
            [0, 10, 20],
            [30, 20, 10],
        ],
        dtype=np.uint32,
    )

    if array_as_pyramid:
        boundaries.add_boundaries(
            cell_boundaries=[mask.copy()],
            nuclei_boundaries=[nuclei.copy()],
            pixel_size=1,
        )
    else:
        boundaries.add_boundaries(
            cell_boundaries=mask.copy(),
            nuclei_boundaries=nuclei.copy(),
            pixel_size=1,
        )

    return boundaries


def _to_numpy(arr):
    return arr.compute() if hasattr(arr, "compute") else np.asarray(arr)


def _create_sparse_boundaries(nucleus_to_cell_map=None, nucleus_count=None):
    """Boundaries with non-contiguous seg_mask_value, mimicking a real (filtered) Xenium
    read where cell labels no longer coincide numerically with their row position."""
    boundaries = BoundariesData(
        cell_names=["c1", "c2", "c3"],
        seg_mask_value=[2, 5, 9],
        nucleus_to_cell_map=nucleus_to_cell_map,
        nucleus_count=nucleus_count,
    )
    mask = np.array([[0, 2], [5, 9]], dtype=np.uint32)
    nuclei = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=nuclei, pixel_size=1)
    return boundaries


def test_sync_rejects_duplicate_obs_names():
    boundaries = _create_boundaries()
    table = _create_table(["c1", "c1", "c2"])
    celldata = CellData(table=table, boundaries=boundaries)

    with pytest.raises(ValueError, match="must be unique"):
        celldata.sync()


def test_sync_rejects_when_no_overlap():
    boundaries = _create_boundaries()
    table = _create_table(["x1", "x2", "x3"])
    celldata = CellData(table=table, boundaries=boundaries)

    with pytest.raises(ValueError, match="No matching values"):
        celldata.sync()


def test_sync_zeros_removed_cells_in_array_masks_and_filters_table():
    boundaries = _create_boundaries(array_as_pyramid=False)
    table = _create_table(["c1", "c3", "x1"])
    celldata = CellData(table=table, boundaries=boundaries)

    with patch("insitupy.containers.cell_data.logger.info") as mock_info:
        summary = celldata.sync(verbose=True, return_summary=True)

    assert list(celldata.table.obs_names) == ["c1", "c3"]
    assert list(celldata.boundaries.cell_names.compute()) == ["c1", "c3"]
    assert list(celldata.boundaries.seg_mask_value.compute()) == [1, 3]
    assert summary == {
        "had_boundaries": True,
        "changed": True,
        "removed_table": 1,
        "removed_boundaries": 1,
        "reordered_boundaries": False,
    }
    mock_info.assert_called_with(
        "CellData.sync(): synchronized table and boundaries (removed 1 table entries, removed 1 boundary entries)."
    )

    cells = _to_numpy(celldata.boundaries["cells"])
    nuclei = _to_numpy(celldata.boundaries["nuclei"])

    assert 2 not in np.unique(cells)
    assert np.all(nuclei[cells == 0] == 0)


def test_sync_zeros_removed_cells_in_pyramid_masks_and_filters_table():
    boundaries = _create_boundaries(array_as_pyramid=True)
    table = _create_table(["c1", "c3", "x1"])
    celldata = CellData(table=table, boundaries=boundaries)

    celldata.sync()

    assert list(celldata.table.obs_names) == ["c1", "c3"]
    assert list(celldata.boundaries.cell_names.compute()) == ["c1", "c3"]
    assert list(celldata.boundaries.seg_mask_value.compute()) == [1, 3]

    cells = _to_numpy(celldata.boundaries["cells"][0])
    nuclei = _to_numpy(celldata.boundaries["nuclei"][0])

    assert 2 not in np.unique(cells)
    assert np.all(nuclei[cells == 0] == 0)


def test_sync_is_noop_without_boundaries_even_with_duplicate_obs_names():
    table = _create_table(["c1", "c1", "c2"])
    celldata = CellData(table=table, boundaries=None)

    with patch("insitupy.containers.cell_data.logger.info") as mock_info:
        summary = celldata.sync(verbose=True, return_summary=True)

    assert list(celldata.table.obs_names) == ["c1", "c1", "c2"]
    assert summary == {
        "had_boundaries": False,
        "changed": False,
        "removed_table": 0,
        "removed_boundaries": 0,
        "reordered_boundaries": False,
    }
    mock_info.assert_called_with("CellData.sync(): no boundaries present; nothing to synchronize.")


def test_sync_reports_noop_when_already_aligned():
    boundaries = _create_boundaries()
    table = _create_table(["c1", "c2", "c3"])
    celldata = CellData(table=table, boundaries=boundaries)

    with patch("insitupy.containers.cell_data.logger.info") as mock_info:
        summary = celldata.sync(verbose=True, return_summary=True)

    assert summary == {
        "had_boundaries": True,
        "changed": False,
        "removed_table": 0,
        "removed_boundaries": 0,
        "reordered_boundaries": False,
    }
    mock_info.assert_called_with("CellData.sync(): no synchronization needed; table and boundaries are already aligned.")


def test_is_synced_true_when_table_and_boundaries_match():
    boundaries = _create_boundaries()
    table = _create_table(["c1", "c2", "c3"])
    celldata = CellData(table=table, boundaries=boundaries)

    assert celldata.is_synced is True


def test_is_synced_false_when_order_differs():
    boundaries = _create_boundaries()
    table = _create_table(["c3", "c2", "c1"])
    celldata = CellData(table=table, boundaries=boundaries)

    assert celldata.is_synced is False


def test_multicelldata_is_synced_reflects_all_layers():
    synced = CellData(table=_create_table(["c1", "c2", "c3"]), boundaries=_create_boundaries())
    unsynced = CellData(table=_create_table(["c3", "c2", "c1"]), boundaries=_create_boundaries())

    cells = MultiCellData()
    cells.add_celldata(synced, key="main", is_main=True)
    cells.add_celldata(unsynced, key="other")

    assert cells.is_synced is False


def test_viewer_preflight_warns_without_syncing_unsynced_cells():
    unsynced = CellData(table=_create_table(["c3", "c2", "c1"]), boundaries=_create_boundaries())
    cells = MultiCellData()
    cells.add_celldata(unsynced, key="main", is_main=True)

    data = type("DataStub", (), {"cells": cells})()

    synced = _sync_cells_for_viewer_if_needed(data)

    assert synced is False
    assert data.cells.is_synced is False
    assert list(data.cells.table.obs_names) == ["c3", "c2", "c1"]
    assert list(data.cells.boundaries.cell_names.compute()) == ["c1", "c2", "c3"]


def test_sync_remaps_nucleus_to_cell_map_after_sparse_filter():
    # nucleus 0,1 -> c1 (row 0, multinucleated); nucleus 2 -> c2 (row 1); nucleus 3 -> c3 (row 2)
    nucleus_to_cell_map = {0: 0, 1: 0, 2: 1, 3: 2}
    boundaries = _create_sparse_boundaries(nucleus_to_cell_map=nucleus_to_cell_map)
    table = _create_table(["c1", "c2", "c3"])
    celldata = CellData(table=table, boundaries=boundaries)

    # filter out the middle cell (c2)
    filtered = celldata[[True, False, True]]

    new_cell_names = list(filtered.boundaries.cell_names.compute())
    assert new_cell_names == ["c1", "c3"]

    new_map = filtered.boundaries.nucleus_to_cell_map
    n_new = len(new_cell_names)

    # every surviving map value is a valid row index into the new table
    assert all(0 <= v < n_new for v in new_map.values())
    # the nucleus that belonged to the removed cell (c2) is gone
    assert 2 not in new_map
    # surviving nuclei still resolve, by name, to the same cell they did before the filter
    assert new_cell_names[new_map[0]] == "c1"
    assert new_cell_names[new_map[1]] == "c1"
    assert new_cell_names[new_map[3]] == "c3"


def test_sync_reindexes_nucleus_count_after_sparse_filter():
    nucleus_count = np.array([2, 1, 1])
    boundaries = _create_sparse_boundaries(nucleus_count=nucleus_count)
    table = _create_table(["c1", "c2", "c3"])
    celldata = CellData(table=table, boundaries=boundaries)

    filtered = celldata[[True, False, True]]

    assert len(filtered.boundaries.nucleus_count) == len(filtered.table)
    assert list(filtered.boundaries.nucleus_count) == [2, 1]


def test_sync_invalidates_unrepairable_nucleus_to_cell_map():
    # value 5 does not index any of the 3 boundary rows even before filtering -
    # the demo's stale-identity-map shape in miniature.
    nucleus_to_cell_map = {0: 0, 1: 5}
    boundaries = _create_sparse_boundaries(nucleus_to_cell_map=nucleus_to_cell_map)
    table = _create_table(["c1", "c2", "c3"])
    celldata = CellData(table=table, boundaries=boundaries)

    with pytest.warns(UserWarning, match="nucleus_to_cell_map is inconsistent"):
        celldata.sync()

    assert celldata.boundaries.nucleus_to_cell_map is None


def test_crop_works_without_boundaries():
    table = _create_table(["c1", "c2", "c3"])
    celldata = CellData(table=table, boundaries=None)

    cropped = celldata.crop(xlim=(1, 4), ylim=(2, 5), inplace=False)

    assert cropped is not None
    assert cropped.boundaries is None
    assert list(cropped.table.obs_names) == ["c2", "c3"]
    assert np.allclose(cropped.table.obsm["spatial"], np.array([[1.0, 1.0], [3.0, 3.0]]))

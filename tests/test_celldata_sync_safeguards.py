import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy.dataclasses.dataclasses import BoundariesData, CellData


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

    celldata.sync()

    assert list(celldata.table.obs_names) == ["c1", "c3"]
    assert list(celldata.boundaries.cell_names.compute()) == ["c1", "c3"]
    assert list(celldata.boundaries.seg_mask_value.compute()) == [1, 3]

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

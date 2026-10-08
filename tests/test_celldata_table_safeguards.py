import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData


def _create_table(obs_names):
    table = AnnData(
        X=np.ones((len(obs_names), 2)),
        obs=pd.DataFrame(index=pd.Index(obs_names, dtype=str)),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.arange(len(obs_names) * 2, dtype=float).reshape(len(obs_names), 2)
    return table


def _create_celldata():
    boundaries = BoundariesData(
        cell_names=["c1", "c2", "c3"],
        seg_mask_value=[1, 2, 3],
    )
    boundaries.add_boundaries(
        cell_boundaries=np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32),
        nuclei_boundaries=None,
        pixel_size=1,
    )

    return CellData(table=_create_table(["c1", "c2", "c3"]), boundaries=boundaries)


def test_table_setter_rejects_ids_not_in_boundaries():
    celldata = _create_celldata()
    new_table = _create_table(["c1", "c2", "x1"])

    with pytest.raises(ValueError, match="not present in boundaries"):
        celldata.table = new_table


def test_matrix_setter_rejects_ids_not_in_boundaries():
    celldata = _create_celldata()
    new_table = _create_table(["x1", "x2", "x3"])

    with pytest.raises(ValueError, match="not present in boundaries"):
        celldata.matrix = new_table


def test_set_table_allows_partial_overlap_and_syncs():
    celldata = _create_celldata()
    new_table = _create_table(["c1", "c3", "x1"])

    celldata.set_table(new_table, allow_partial_overlap=True)

    assert list(celldata.table.obs_names) == ["c1", "c3"]
    assert list(celldata.boundaries.cell_names.compute()) == ["c1", "c3"]
    assert list(celldata.boundaries.seg_mask_value.compute()) == [1, 3]

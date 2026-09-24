"""Regression tests for container fixes from ultrareview Run C (R-C1, R-C2, R-C5, R-C7)."""

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from shapely.geometry import Point

from insitupy.containers.cell_data import CellData
from insitupy.containers.multi_cell_data import MultiCellData
from insitupy.containers.multi_spatial_units_data import MultiSpatialUnitsData
from insitupy.containers.spatial_units_data import SpatialUnitsData


def _geometry_only_units(names=("u1", "u2")):
    gdf = gpd.GeoDataFrame({"name": list(names),
                            "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))]})
    return SpatialUnitsData(shapes=gdf, data=None, unit_type="niche")


def _celldata(n=3):
    table = AnnData(
        X=np.ones((n, 2)),
        obs=pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n)])),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.arange(n * 2, dtype=float).reshape(n, 2)
    return CellData(table=table, boundaries=None)


def test_geometry_only_units_layer_round_trips(tmp_path):
    # R-C1: save() skips data.h5ad when there is no table; read() must load that layer back
    musd = MultiSpatialUnitsData()
    musd.add_units(_geometry_only_units(), key="niches", is_main=True)
    musd.save(tmp_path / "units")

    back = MultiSpatialUnitsData.read(tmp_path / "units")

    layer = back["niches"]
    assert layer.table is None
    assert len(layer) == 2
    assert "2 geometries" in repr(layer)


def test_units_repr_without_main_layer():
    # R-C2: add_units() defaults to is_main=False, so no main layer is set
    musd = MultiSpatialUnitsData()
    musd.add_units(_geometry_only_units(), key="niches")

    text = repr(musd)

    assert "without main layer" in text
    assert "'niches'" in text


def test_cells_repr_without_main_layer():
    # R-C7: same for add_celldata(), whose is_main also defaults to False
    mcd = MultiCellData()
    mcd.add_celldata(cd=_celldata(), key="proseg")

    text = repr(mcd)

    assert "without main layer" in text
    assert "'proseg'" in text


def test_multicelldata_read_accepts_str_path(tmp_path):
    # R-C5: the path was joined with `/` before being converted to Path
    mcd = MultiCellData()
    mcd.add_celldata(cd=_celldata(), key="main", is_main=True)
    mcd.save(tmp_path / "cells")

    back = MultiCellData.read(str(tmp_path / "cells"))

    assert back.main_key == "main"
    assert back["main"].table.n_obs == 3

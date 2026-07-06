"""Tests for MultiSpatialUnitsData: key collisions, main-layer routing,
and main-key switching — the real failure modes around storing multiple
named SpatialUnitsData layers per InSituData."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point

from insitupy.containers.multi_spatial_units_data import MultiSpatialUnitsData
from insitupy.containers.spatial_units_data import SpatialUnitsData


def _make_units(names, unit_type="unit", n_vars=2, seed=0):
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame(
        {
            "name": names,
            "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))],
        }
    )
    table = AnnData(
        X=rng.random((len(names), n_vars)),
        obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
        var=pd.DataFrame(index=[f"v{i}" for i in range(n_vars)]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def test_is_empty_true_for_fresh_container():
    musd = MultiSpatialUnitsData()
    assert musd.is_empty is True


def test_is_empty_false_after_add():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["u1", "u2"]), key="main", is_main=True)
    assert musd.is_empty is False


def test_add_units_raises_on_duplicate_key_without_overwrite():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["u1"]), key="niche", is_main=True)

    with pytest.raises(KeyError):
        musd.add_units(_make_units(["u2"]), key="niche")


def test_overwrite_true_replaces_existing_layer():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["u1"], unit_type="first"), key="niche", is_main=True)
    musd.add_units(_make_units(["u2", "u3"], unit_type="second"), key="niche", overwrite=True)

    assert list(musd["niche"].shapes["name"]) == ["u2", "u3"]
    assert musd["niche"].unit_type == "second"


def test_shapes_and_table_resolve_to_main_layer():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["main1", "main2"], unit_type="main_type"), key="main", is_main=True)
    musd.add_units(_make_units(["alt1"], unit_type="alt_type"), key="alt")

    assert list(musd.shapes["name"]) == ["main1", "main2"]
    assert list(musd.table.obs_names) == ["main1", "main2"]
    assert musd.unit_type == "main_type"


def test_set_main_switches_shapes_and_table():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["main1", "main2"], unit_type="main_type"), key="main", is_main=True)
    musd.add_units(_make_units(["alt1"], unit_type="alt_type"), key="alt")

    musd.set_main("alt")

    assert musd.main_key == "alt"
    assert list(musd.shapes["name"]) == ["alt1"]
    assert musd.unit_type == "alt_type"


def test_deleting_main_key_raises():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["u1"]), key="main", is_main=True)
    musd.add_units(_make_units(["u2"]), key="alt")

    with pytest.raises(KeyError):
        del musd["main"]


def test_deleting_non_main_key_succeeds():
    musd = MultiSpatialUnitsData()
    musd.add_units(_make_units(["u1"]), key="main", is_main=True)
    musd.add_units(_make_units(["u2"]), key="alt")

    del musd["alt"]

    assert "alt" not in musd
    assert musd.main_key == "main"

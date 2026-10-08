"""Integration tests for InSituData.add_units()/.units/.crop()/.align_units()
against the new MultiSpatialUnitsData container — the bug this fixes is that
InSituData.add_units() used to silently overwrite any existing units layer."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point, Polygon

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.spatial_units_data import SpatialUnitsData

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_units(names, unit_type="unit", n_vars=2, coords=None, seed=0):
    """SpatialUnitsData with circular shapes at `coords` (default: diagonal)."""
    if coords is None:
        coords = [(i * 2, i * 2) for i in range(len(names))]
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame(
        {
            "name": names,
            "geometry": [Point(x, y).buffer(0.4) for x, y in coords],
        }
    )
    table = AnnData(
        X=rng.random((len(names), n_vars)),
        obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
        var=pd.DataFrame(index=[f"v{i}" for i in range(n_vars)]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def _make_cells_table(obs_names):
    n = len(obs_names)
    table = AnnData(
        X=np.ones((n, 2)),
        obs=pd.DataFrame(index=pd.Index(obs_names, dtype=str)),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.array([[float(i * 2), float(i * 2)] for i in range(n)])
    return table


def _make_insitudata(with_cells=True):
    """Minimal InSituData with no project path, optionally with 3 diagonal cells."""
    xd = InSituData(
        path=None, metadata=None,
        slide_id="slide1", sample_id="s1",
        method_name="test", method_params={},
    )
    if with_cells:
        celldata = CellData(table=_make_cells_table(["c1", "c2", "c3"]), boundaries=None)
        xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


# ── InSituData.units is always present, never None ───────────────────────────


class TestUnitsAlwaysPresent:
    def test_fresh_object_units_is_empty_not_none(self):
        xd = _make_insitudata()
        assert xd.units is not None
        assert xd.units.is_empty is True

    def test_del_units_resets_to_empty_container(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["u1", "u2"], unit_type="niche"))
        assert xd.units.is_empty is False

        del xd.units

        assert xd.units is not None
        assert xd.units.is_empty is True


# ── InSituData.add_units() ────────────────────────────────────────────────────


class TestAddUnits:
    def test_first_call_with_no_key_auto_becomes_main(self):
        xd = _make_insitudata()
        su = _make_units(["u1", "u2"], unit_type="niche")

        xd.add_units(su)

        assert xd.units.main_key == "niche"
        assert list(xd.units.table.obs_names) == ["u1", "u2"]

    def test_second_call_same_default_key_raises_without_overwrite(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["u1"], unit_type="niche"))

        with pytest.raises(KeyError):
            xd.add_units(_make_units(["u2"], unit_type="niche"))

    def test_second_call_same_default_key_succeeds_with_overwrite(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["u1"], unit_type="niche"))
        xd.add_units(_make_units(["u2", "u3"], unit_type="niche"), overwrite=True)

        assert list(xd.units["niche"].table.obs_names) == ["u2", "u3"]

    def test_explicit_distinct_keys_both_retrievable_main_key_unchanged(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["v1"], unit_type="visium"))
        xd.add_units(_make_units(["n1", "n2"], unit_type="niche"), key="niche")

        assert set(xd.units.keys()) == {"visium", "niche"}
        assert list(xd.units["visium"].table.obs_names) == ["v1"]
        assert list(xd.units["niche"].table.obs_names) == ["n1", "n2"]
        # first-added layer remains main
        assert xd.units.main_key == "visium"


# ── InSituData.crop() crops units ─────────────────────────────────────────────


class TestCropUnits:
    def test_crop_drops_out_of_region_units_and_shifts_coordinates(self):
        xd = _make_insitudata(with_cells=True)
        xd.add_units(_make_units(["u1", "u2", "u3"], unit_type="niche"))

        xd.crop(xlim=(2, 10), ylim=(2, 10), inplace=True, verbose=False)

        assert list(xd.units.shapes["name"]) == ["u2", "u3"]
        bounds = xd.units.shapes.geometry.bounds
        # u2 was at (2,2) -> shifted to (0,0); u3 was at (4,4) -> shifted to (2,2)
        np.testing.assert_allclose(bounds.iloc[0][["minx", "miny"]].values, [-0.4, -0.4], atol=1e-6)
        np.testing.assert_allclose(bounds.iloc[1][["minx", "miny"]].values, [1.6, 1.6], atol=1e-6)


# ── shape-based crop with a bounding box extending below zero ────────────────
#
# Regression test: mirrors TestCropShapeNegativeBoundsRegression in
# test_celldata_crop.py. When cropping units by a `shape` whose bounding box
# dips below x=0 or y=0 (e.g. a region annotated in napari that slightly
# overshoots the tissue), the coordinate shift must clip to 0 rather than
# using the raw shape.bounds minx/miny, otherwise units end up shifted
# relative to where the image was actually cropped.

class TestCropUnitsShapeNegativeBoundsRegression:
    def test_shape_bounds_below_zero_does_not_overshift_coordinates(self):
        su = _make_units(["u1", "u2", "u3"], unit_type="niche")
        # shape covers x,y in [-5, 10]; the image crop origin still clips to
        # 0, so unit coordinates must not be shifted at all here.
        shape = Polygon([(-5, -5), (10, -5), (10, 10), (-5, 10)])

        su.crop(shape=shape, inplace=True)

        assert list(su.shapes["name"]) == ["u1", "u2", "u3"]
        bounds = su.shapes.geometry.bounds
        np.testing.assert_allclose(bounds.iloc[0][["minx", "miny"]].values, [-0.4, -0.4], atol=1e-6)
        np.testing.assert_allclose(bounds.iloc[1][["minx", "miny"]].values, [1.6, 1.6], atol=1e-6)
        np.testing.assert_allclose(bounds.iloc[2][["minx", "miny"]].values, [3.6, 3.6], atol=1e-6)


# ── InSituData.align_units() per-key relaxation ───────────────────────────────


class TestAlignUnitsPerKeyRelaxation:
    IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_succeeds_when_self_has_differently_keyed_layer(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["n1"], unit_type="niche"), key="niche")

        other = _make_insitudata(with_cells=False)
        other.add_units(_make_units(["v1"], unit_type="visium"), key="visium")

        xd.align_units(other=other, transformation_matrix=self.IDENTITY, verbose=False)

        assert set(xd.units.keys()) == {"niche", "visium"}

    def test_raises_on_same_key_collision(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["v1"], unit_type="visium"), key="visium")

        other = _make_insitudata(with_cells=False)
        other.add_units(_make_units(["v2"], unit_type="visium"), key="visium")

        with pytest.raises(KeyError):
            xd.align_units(other=other, transformation_matrix=self.IDENTITY, verbose=False)

    def test_same_key_collision_succeeds_with_overwrite(self):
        xd = _make_insitudata()
        xd.add_units(_make_units(["v1"], unit_type="visium"), key="visium")

        other = _make_insitudata(with_cells=False)
        other.add_units(_make_units(["v2", "v3"], unit_type="visium"), key="visium")

        xd.align_units(other=other, transformation_matrix=self.IDENTITY, overwrite=True, verbose=False)

        assert list(xd.units["visium"].table.obs_names) == ["v2", "v3"]

    def test_raises_when_other_has_no_units(self):
        xd = _make_insitudata()
        other = _make_insitudata(with_cells=False)

        with pytest.raises(ValueError, match="no spatial units"):
            xd.align_units(other=other, transformation_matrix=self.IDENTITY, verbose=False)

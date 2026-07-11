"""Tests for spatialdata.convert_to_spatialdata.

All tests are skipped when the spatialdata package is not installed.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point, Polygon

# Skip the entire module if spatialdata is not installed
pytest.importorskip("spatialdata")

from insitupy._constants import SPATIALDATA_DIALECT_VERSION  # noqa: E402
from insitupy._core.data import InSituData  # noqa: E402
from insitupy.containers import AnnotationsData, RegionsData  # noqa: E402
from insitupy.containers.cell_data import CellData  # noqa: E402
from insitupy.containers.spatial_units_data import SpatialUnitsData  # noqa: E402
from insitupy.experiment.data import InSituExperiment  # noqa: E402
from insitupy.spatialdata._convert import _transform_regions_for_spatialdata  # noqa: E402
from insitupy.spatialdata.convert import convert_to_spatialdata  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_insitudata(n_cells=10, n_genes=5, seed=0, sample_id="s1", cell_prefix="cell"):
    """Minimal InSituData with expression table and spatial coordinates."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(float)

    obs = pd.DataFrame(index=pd.Index([f"{cell_prefix}_{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100

    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="test", sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _poly(x=0, y=0, size=10):
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def _poly_gdf(*names):
    """One Polygon row per name, with unique ids."""
    return gpd.GeoDataFrame({
        "id": [f"{n}_{i}" for i, n in enumerate(names)],
        "name": list(names),
        "geometry": [_poly(i * 20) for i in range(len(names))],
        "color": ["#ff0000"] * len(names),
    })


def _make_units(names, unit_type="unit", n_vars=2, seed=0):
    """Minimal SpatialUnitsData with polygon shapes and an AnnData table."""
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame({
        "name": names,
        "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))],
    })
    table = AnnData(
        X=rng.random((len(names), n_vars)),
        obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
        var=pd.DataFrame(index=[f"v{i}" for i in range(n_vars)]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def _make_transcripts_df(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x_location": rng.random(n) * 100,
        "y_location": rng.random(n) * 100,
        "z_location": np.zeros(n),
        "feature_name": [f"gene_{i % 3}" for i in range(n)],
        "cell_id": [f"cell_{i}" for i in range(n)],
    })


def _make_experiment(n_samples=2, n_cells=6, n_genes=3):
    """Multi-sample InSituExperiment; each sample has cells, units, annotations, regions."""
    exp = InSituExperiment()

    for i in range(n_samples):
        xd = _make_insitudata(
            n_cells=n_cells, n_genes=n_genes, seed=i,
            sample_id=f"s{i}", cell_prefix=f"s{i}cell",
        )
        xd.add_units(_make_units([f"s{i}u0", f"s{i}u1"], unit_type="unit", seed=i))
        xd.annotations.add_data(data=_poly_gdf(f"ann{i}"), key="roi", scale_factor=1.0)
        xd.regions.add_data(data=_poly_gdf(f"reg{i}"), key="roi", scale_factor=1.0)
        exp._data.append(xd)

    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(n_samples)],
        "slide_id": ["slide1"] * n_samples,
        "sample_id": [f"s{i}" for i in range(n_samples)],
    })
    return exp


# ── convert_to_spatialdata ────────────────────────────────────────────────────

class TestConvertToSpatialdata:
    def test_returns_spatialdata_object(self):
        from spatialdata import SpatialData
        xd = _make_insitudata()
        sdata = convert_to_spatialdata(xd)
        assert isinstance(sdata, SpatialData)

    def test_tables_element_present(self):
        xd = _make_insitudata()
        sdata = convert_to_spatialdata(xd)
        assert len(sdata.tables) > 0

    def test_table_obs_shape_matches(self):
        n_cells = 8
        xd = _make_insitudata(n_cells=n_cells)
        sdata = convert_to_spatialdata(xd)
        table = next(iter(sdata.tables.values()))
        assert table.n_obs == n_cells

    def test_table_var_shape_matches(self):
        n_genes = 6
        xd = _make_insitudata(n_genes=n_genes)
        sdata = convert_to_spatialdata(xd)
        table = next(iter(sdata.tables.values()))
        assert table.n_vars == n_genes

    def test_no_images_element_when_none_loaded(self):
        xd = _make_insitudata()
        sdata = convert_to_spatialdata(xd)
        # Without images loaded, images dict should be empty
        assert len(sdata.images) == 0


# ── Multi-sample export: every populated modality present (item 4) ────────────

class TestMultiSampleExport:
    def test_all_modalities_present_for_each_sample(self):
        exp = _make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)

        for i in range(2):
            prefix = f"SAMPLE.sample_{i}.."
            assert f"{prefix}CELLS.main.table" in sdata.tables
            assert f"{prefix}CELLS.main.circles" in sdata.shapes
            assert f"{prefix}UNITS.unit.table" in sdata.tables
            assert f"{prefix}UNITS.unit.shapes" in sdata.shapes
            assert f"{prefix}ANNOTATIONS.roi" in sdata.shapes
            assert f"{prefix}REGIONS.roi" in sdata.shapes


# ── Units export correctness (item 1) ──────────────────────────────────────────

class TestUnitsExport:
    def test_units_table_and_shapes_present_and_match(self):
        xd = _make_insitudata(n_cells=4)
        su = _make_units(["u0", "u1", "u2"], unit_type="niche", n_vars=3, seed=7)
        xd.add_units(su)

        sdata = convert_to_spatialdata(xd)

        assert "UNITS.niche.table" in sdata.tables
        assert "UNITS.niche.shapes" in sdata.shapes

        table = sdata.tables["UNITS.niche.table"]
        assert table.n_obs == 3
        assert table.n_vars == 3

        shapes = sdata.shapes["UNITS.niche.shapes"]
        assert len(shapes) == 3
        assert shapes.geometry.iloc[0].area == pytest.approx(su.shapes.geometry.iloc[0].area)


# ── Regions/annotations guard-bug regression (item 2) ──────────────────────────

class TestRegionsAnnotationsGuardFix:
    def test_regions_exported_without_annotations(self):
        """Pre-fix: the guard checked `xd.annotations`, so regions were silently dropped."""
        xd = _make_insitudata(n_cells=2)
        xd._annotations = None
        xd.regions.add_data(data=_poly_gdf("r1"), key="roi", scale_factor=1.0)

        shapes = _transform_regions_for_spatialdata(xd)
        assert len(shapes) == 1

    def test_no_crash_with_annotations_and_no_regions(self):
        """Pre-fix: the guard passed on `xd.annotations` but then iterated `xd.regions`,
        raising AttributeError when regions was None."""
        xd = _make_insitudata(n_cells=2)
        xd.annotations.add_data(data=_poly_gdf("a1"), key="roi", scale_factor=1.0)
        xd._regions = None

        shapes = _transform_regions_for_spatialdata(xd)
        assert shapes == {}


# ── Case-insensitive conflict resolution, end to end (item 3) ──────────────────

class TestCaseInsensitiveConflictResolution:
    def test_conflicting_annotation_keys_are_renamed(self):
        xd = _make_insitudata(n_cells=2)
        xd.annotations.add_data(data=_poly_gdf("x"), key="Demo", scale_factor=1.0)
        xd.annotations.add_data(data=_poly_gdf("y"), key="demo", scale_factor=1.0)

        sdata = convert_to_spatialdata(xd)  # must not raise

        assert "ANNOTATIONS.Demo" in sdata.shapes
        assert "ANNOTATIONS.demo_v2" in sdata.shapes
        assert "ANNOTATIONS.demo" not in sdata.shapes


# ── include_transcripts flag (transcripts-optional item) ───────────────────────

class TestIncludeTranscripts:
    def test_transcripts_included_by_default(self):
        xd = _make_insitudata(n_cells=3)
        xd.transcripts = _make_transcripts_df()
        sdata = convert_to_spatialdata(xd)
        assert len(sdata.points) == 1

    def test_transcripts_skipped_when_disabled(self):
        xd = _make_insitudata(n_cells=3)
        xd.transcripts = _make_transcripts_df()
        sdata = convert_to_spatialdata(xd, include_transcripts=False)
        assert len(sdata.points) == 0


# ── Dialect descriptor stamped into sdata.attrs (item 5) ───────────────────────

class TestDialectAttrs:
    def test_dialect_descriptor_present(self):
        xd = _make_insitudata(n_cells=2)
        sdata = convert_to_spatialdata(xd)
        descriptor = sdata.attrs["insitupy_spatialdata_dialect"]
        assert descriptor["version"] == SPATIALDATA_DIALECT_VERSION

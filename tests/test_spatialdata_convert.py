"""Tests for spatialdata.convert_to_spatialdata.

All tests are skipped when the spatialdata package is not installed.
"""

import pytest

# Skip the entire module if spatialdata is not installed
pytest.importorskip("spatialdata")

from insitupy._constants import SPATIALDATA_DIALECT_VERSION  # noqa: E402
from insitupy.spatialdata._convert import _transform_regions_for_spatialdata  # noqa: E402
from insitupy.spatialdata.convert import convert_to_spatialdata  # noqa: E402
from tests.spatialdata_fixtures import (  # noqa: E402
    make_experiment,
    make_insitudata,
    make_transcripts_df,
    make_units,
    poly_gdf,
    roundtrip_through_zarr,
)

# ── convert_to_spatialdata ────────────────────────────────────────────────────

class TestConvertToSpatialdata:
    def test_returns_spatialdata_object(self):
        from spatialdata import SpatialData
        xd = make_insitudata()
        sdata = convert_to_spatialdata(xd)
        assert isinstance(sdata, SpatialData)

    def test_tables_element_present(self):
        xd = make_insitudata()
        sdata = convert_to_spatialdata(xd)
        assert len(sdata.tables) > 0

    def test_table_obs_shape_matches(self):
        n_cells = 8
        xd = make_insitudata(n_cells=n_cells)
        sdata = convert_to_spatialdata(xd)
        table = next(iter(sdata.tables.values()))
        assert table.n_obs == n_cells

    def test_table_var_shape_matches(self):
        n_genes = 6
        xd = make_insitudata(n_genes=n_genes)
        sdata = convert_to_spatialdata(xd)
        table = next(iter(sdata.tables.values()))
        assert table.n_vars == n_genes

    def test_no_images_element_when_none_loaded(self):
        xd = make_insitudata()
        sdata = convert_to_spatialdata(xd)
        # Without images loaded, images dict should be empty
        assert len(sdata.images) == 0


# ── Multi-sample export: every populated modality present (item 4) ────────────

class TestMultiSampleExport:
    def test_all_modalities_present_for_each_sample(self):
        exp = make_experiment(n_samples=2)
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
        xd = make_insitudata(n_cells=4)
        su = make_units(["u0", "u1", "u2"], unit_type="niche", n_vars=3, seed=7)
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


# ── seg_mask_value obs column (WP2) ────────────────────────────────────────────

class TestSegMaskValueColumn:
    def test_column_present_and_correct_when_boundaries_exist(self):
        xd = make_insitudata(n_cells=4, with_boundaries=True, contiguous_seg_mask_value=False)
        sdata = convert_to_spatialdata(xd)

        table = sdata.tables["CELLS.main.table"]
        assert "_insitupy_seg_mask_value" in table.obs.columns
        expected = xd.cells["main"].boundaries.seg_mask_value.compute()
        assert list(table.obs["_insitupy_seg_mask_value"]) == list(expected)

    def test_column_absent_without_boundaries(self):
        xd = make_insitudata(n_cells=4)
        sdata = convert_to_spatialdata(xd)

        table = sdata.tables["CELLS.main.table"]
        assert "_insitupy_seg_mask_value" not in table.obs.columns


# ── Nucleus-to-cell mapping export (WP2) ───────────────────────────────────────

class TestNucleusMapExport:
    def test_nucleus_map_present_for_multinucleated_cells(self):
        xd = make_insitudata(n_cells=4, with_boundaries=True, with_multinucleated_cells=True)
        sdata = convert_to_spatialdata(xd)

        assert "CELLS.main.nucleus_map" in sdata.tables
        nmap = sdata.tables["CELLS.main.nucleus_map"]
        assert nmap.n_obs == 3  # 3 nuclei total (2 on cell 0, 1 on cell 1)
        assert set(nmap.obs.columns) >= {"nucleus_label", "cell_id"}
        assert sorted(nmap.obs["nucleus_label"].astype(int)) == [1, 2, 3]

    def test_nucleus_map_absent_for_ordinary_boundaries(self):
        xd = make_insitudata(n_cells=4, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)

        assert "CELLS.main.nucleus_map" not in sdata.tables

    def test_orphan_nucleus_excluded_instead_of_raising(self):
        """Real failure-mode test: Xenium marks a nucleus with no assigned cell
        using a cell_index past the valid range (see the IndexError this used
        to raise, insitupy/spatialdata/_convert.py). Such entries must be
        skipped on export rather than crash or produce a bogus cell_id."""
        xd = make_insitudata(n_cells=4, with_boundaries=True, with_multinucleated_cells=True)
        boundaries = xd.cells["main"].boundaries
        n_cells = len(xd.cells["main"].table)
        boundaries._nucleus_to_cell_map[3] = n_cells  # orphan: one past the valid range

        sdata = convert_to_spatialdata(xd)  # must not raise IndexError

        nmap = sdata.tables["CELLS.main.nucleus_map"]
        assert nmap.n_obs == 3  # the 3 valid nuclei only; the orphan is dropped
        assert 4 not in nmap.obs["nucleus_label"].astype(int).to_numpy()


# ── Cell-only segmentation export regression ────────────────────────────────────

class TestCellOnlySegmentationExport:
    def test_boundaries_export_without_nucleus_layer(self):
        """Pre-fix: add_boundaries() always creates a 'nuclei' metadata entry (data=None)
        even when no nucleus mask was added; the exporter crashed trying to wrap that None
        in a DataArray. This is the ordinary case (segmentation without a nucleus channel)."""
        xd = make_insitudata(n_cells=4, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)  # must not raise
        assert "CELLS.main.boundaries.cells" in sdata.labels
        assert "CELLS.main.boundaries.nuclei" not in sdata.labels

    def test_boundaries_export_with_nucleus_layer(self):
        """Happy-path lock-in: when nucleus boundaries ARE present, they must still
        export alongside cell boundaries - the fix must not skip real data too."""
        xd = make_insitudata(n_cells=4, with_boundaries=True, with_nucleus_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        assert "CELLS.main.boundaries.cells" in sdata.labels
        assert "CELLS.main.boundaries.nuclei" in sdata.labels


# ── Raw-array image export regression ───────────────────────────────────────────

class TestRawArrayImageExport:
    def test_image_added_via_raw_array_exports_without_pyramid_wrapping(self):
        """Pre-fix: add_image() with a raw array (not a file path) stores a bare array;
        the exporter assumed a pyramid list and crashed with IndexError."""
        xd = make_insitudata(n_cells=2, with_image=True)
        sdata = convert_to_spatialdata(xd)  # must not raise
        assert "IMAGES.dapi" in sdata.images


# ── Regions/annotations guard-bug regression (item 2) ──────────────────────────

class TestRegionsAnnotationsGuardFix:
    def test_regions_exported_without_annotations(self):
        """Pre-fix: the guard checked `xd.annotations`, so regions were silently dropped."""
        xd = make_insitudata(n_cells=2)
        xd._annotations = None
        xd.regions.add_data(data=poly_gdf("r1"), key="roi", scale_factor=1.0)

        shapes = _transform_regions_for_spatialdata(xd)
        assert len(shapes) == 1

    def test_no_crash_with_annotations_and_no_regions(self):
        """Pre-fix: the guard passed on `xd.annotations` but then iterated `xd.regions`,
        raising AttributeError when regions was None."""
        xd = make_insitudata(n_cells=2)
        xd.annotations.add_data(data=poly_gdf("a1"), key="roi", scale_factor=1.0)
        xd._regions = None

        shapes = _transform_regions_for_spatialdata(xd)
        assert shapes == {}


# ── Case-insensitive conflict resolution, end to end (item 3) ──────────────────

class TestCaseInsensitiveConflictResolution:
    def test_conflicting_annotation_keys_are_renamed(self):
        xd = make_insitudata(n_cells=2)
        xd.annotations.add_data(data=poly_gdf("x"), key="Demo", scale_factor=1.0)
        xd.annotations.add_data(data=poly_gdf("y"), key="demo", scale_factor=1.0)

        sdata = convert_to_spatialdata(xd)  # must not raise

        assert "ANNOTATIONS.Demo" in sdata.shapes
        assert "ANNOTATIONS.demo_v2" in sdata.shapes
        assert "ANNOTATIONS.demo" not in sdata.shapes


# ── include_transcripts flag (transcripts-optional item) ───────────────────────

class TestIncludeTranscripts:
    def test_transcripts_included_by_default(self):
        xd = make_insitudata(n_cells=3)
        xd.transcripts = make_transcripts_df()
        sdata = convert_to_spatialdata(xd)
        assert len(sdata.points) == 1

    def test_transcripts_skipped_when_disabled(self):
        xd = make_insitudata(n_cells=3)
        xd.transcripts = make_transcripts_df()
        sdata = convert_to_spatialdata(xd, include_transcripts=False)
        assert len(sdata.points) == 0


# ── Dialect descriptor stamped into sdata.attrs (item 5) ───────────────────────

class TestDialectAttrs:
    def test_dialect_descriptor_present(self):
        xd = make_insitudata(n_cells=2)
        sdata = convert_to_spatialdata(xd)
        descriptor = sdata.attrs["insitupy_spatialdata_dialect"]
        assert descriptor["version"] == SPATIALDATA_DIALECT_VERSION

    def test_bare_insitudata_gets_flat_slide_and_sample_id(self):
        xd = make_insitudata(n_cells=2, sample_id="s1")
        sdata = convert_to_spatialdata(xd)
        descriptor = sdata.attrs["insitupy_spatialdata_dialect"]
        assert descriptor["slide_id"] == xd.slide_id
        assert descriptor["sample_id"] == xd.sample_id

    def test_experiment_gets_samples_keyed_by_uid(self):
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        descriptor = sdata.attrs["insitupy_spatialdata_dialect"]

        samples = descriptor["samples"]
        for uid, xd in ((meta["uid"], d) for meta, d in exp.iterdata()):
            assert samples[uid] == {"slide_id": xd.slide_id, "sample_id": xd.sample_id}


# ── Real disk round trip (WP5: nothing previously called sdata.write()) ───────

class TestExportRoundTripsThroughDisk:
    """Every test above asserts against the in-memory SpatialData object that
    `convert_to_spatialdata` returns directly - none of them ever write to a zarr
    store. These tests route the same object through a real
    `sdata.write()` -> `spatialdata.read_zarr()` round trip via
    `roundtrip_through_zarr`, so the actual point of an exchange format - surviving
    a disk write/read - is exercised at least once for the single- and
    multi-sample export paths.
    """

    def test_single_sample_survives_disk_roundtrip(self, tmp_path):
        n_cells = 8
        xd = make_insitudata(n_cells=n_cells)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="single.zarr")

        assert len(sdata2.tables) > 0
        table = next(iter(sdata2.tables.values()))
        assert table.n_obs == n_cells

    def test_multi_sample_modalities_survive_disk_roundtrip(self, tmp_path):
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="experiment.zarr")

        for i in range(2):
            prefix = f"SAMPLE.sample_{i}.."
            assert f"{prefix}CELLS.main.table" in sdata2.tables
            assert f"{prefix}CELLS.main.circles" in sdata2.shapes
            assert f"{prefix}UNITS.unit.table" in sdata2.tables
            assert f"{prefix}UNITS.unit.shapes" in sdata2.shapes
            assert f"{prefix}ANNOTATIONS.roi" in sdata2.shapes
            assert f"{prefix}REGIONS.roi" in sdata2.shapes

    def test_units_table_and_shapes_survive_disk_roundtrip(self, tmp_path):
        xd = make_insitudata(n_cells=4)
        su = make_units(["u0", "u1", "u2"], unit_type="niche", n_vars=3, seed=7)
        xd.add_units(su)

        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="units.zarr")

        table = sdata2.tables["UNITS.niche.table"]
        assert table.n_obs == 3
        assert table.n_vars == 3

        shapes = sdata2.shapes["UNITS.niche.shapes"]
        assert len(shapes) == 3
        assert shapes.geometry.iloc[0].area == pytest.approx(su.shapes.geometry.iloc[0].area)


# ── InSituExperimentView accepted by convert_to_spatialdata ────────────────────

class TestExportSampleSubsetView:
    """The flagship case that motivated widening `_is_experiment()`: exporting a
    sample subset (e.g. for faster manual testing) via plain subscripting, which
    always returns an InSituExperimentView. Before the fix, `_is_experiment()`
    used an exact-class check and raised ValueError for any view.
    """

    def test_convert_to_spatialdata_accepts_a_view(self):
        exp = make_experiment(n_samples=3)
        view = exp[:2]

        sdata = convert_to_spatialdata(view)  # must not raise

        for i in range(2):
            prefix = f"SAMPLE.sample_{i}.."
            assert f"{prefix}CELLS.main.table" in sdata.tables
        assert "SAMPLE.sample_2..CELLS.main.table" not in sdata.tables

    def test_view_export_scoped_to_selected_samples_in_dialect_attrs(self):
        exp = make_experiment(n_samples=3)
        view = exp[:2]

        sdata = convert_to_spatialdata(view)

        samples = sdata.attrs["insitupy_spatialdata_dialect"]["samples"]
        assert set(samples) == {"sample_0", "sample_1"}

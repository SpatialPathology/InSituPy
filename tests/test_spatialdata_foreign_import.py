"""Labels-native foreign-store import tests (WP4).

Uses `make_foreign_labels_native_sdata` - a hand-built SpatialData object with no
InSituPy dialect descriptor, mimicking spatialdata-io Xenium output - and exercises
`convert_from_foreign_spatialdata` (the public, hardened successor to
`_convert_from_spatialdata_manual`). See
.log/reports/260713/wp4-labels-native-import/report-wp4-labels-native-import.md.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point

pytest.importorskip("spatialdata")

from spatialdata import SpatialData  # noqa: E402
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel  # noqa: E402
from spatialdata.transformations import Identity, Scale  # noqa: E402
from xarray import DataArray  # noqa: E402

from insitupy._core.data import InSituData  # noqa: E402
from insitupy.io.data import _warn_ignored_spatialdata_params  # noqa: E402
from insitupy.spatialdata import convert_from_foreign_spatialdata  # noqa: E402
from tests.spatialdata_fixtures import make_foreign_labels_native_sdata  # noqa: E402


class TestSegMaskValueTakenFromInstanceKey:
    def test_seg_mask_value_taken_from_instance_key(self):
        """Real failure-mode: the fixture's ids are deliberately non-contiguous,
        so this only passes if seg_mask_value is read from the table's own
        instance_key column, not fabricated as arange(1, n+1)."""
        sdata = make_foreign_labels_native_sdata(contiguous_ids=False, n_cells=6)

        data = convert_from_foreign_spatialdata(
            sdata, cells={"main": {"table_key": "table", "cell_boundaries_data": ("cell_labels", 1.0)}},
        )

        expected = list(sdata.tables["table"].obs["cell_id"])
        assert list(data.cells["main"].boundaries.seg_mask_value.compute()) == expected


class TestCellOnlySegmentationImportsWithoutNucleus:
    def test_cell_only_segmentation_imports_without_nucleus(self):
        """Pre-fix: comparing cell/nucleus pixel sizes before a None check raised
        TypeError whenever only cell boundaries were given - the ordinary case for
        a segmentation with no nucleus channel."""
        sdata = make_foreign_labels_native_sdata(with_nucleus=False, n_cells=4)

        data = convert_from_foreign_spatialdata(
            sdata, cells={"main": {"table_key": "table", "cell_boundaries_data": ("cell_labels", 1.0)}},
        )  # must not raise

        assert data.cells["main"].boundaries is not None


class TestIndependentPixelSizesForCellAndNucleus:
    def test_independent_pixel_sizes_are_honoured(self):
        """Real failure-mode: a foreign store's cell/nucleus label rasters are not
        guaranteed to share a resolution, unlike InSituPy's own exporter.

        Passes label_map (SD-A2): nucleus_boundaries_data without one now raises -
        see TestNucleusBoundariesWithoutLabelMapRaises below for that contract."""
        sdata = make_foreign_labels_native_sdata(n_cells=4, nucleus_pixel_size=0.5, with_nucleus_map=True)

        data = convert_from_foreign_spatialdata(
            sdata, cells={"main": {
                "table_key": "table",
                "cell_boundaries_data": ("cell_labels", 1.0),
                "nucleus_boundaries_data": ("nucleus_labels", 0.5),
                "label_map": "nucleus_boundaries",
            }},
        )

        boundaries = data.cells["main"].boundaries
        assert boundaries.metadata["cells"]["pixel_size"] == 1.0
        assert boundaries.metadata["nuclei"]["pixel_size"] == 0.5


class TestIdentityTransformImageImports:
    def test_identity_transform_image_imports(self):
        sdata = make_foreign_labels_native_sdata()

        data = convert_from_foreign_spatialdata(
            sdata,
            images={"morphology": {"key": "morphology", "pixel_size": 1.0}},
            cells={"main": {"table_key": "table", "cell_boundaries_data": ("cell_labels", 1.0)}},
        )

        assert data.images["morphology"] is not None
        assert data.images.metadata["morphology"]["pixel_size"] == 1.0


class TestSingleScaleImageImports:
    def test_single_scale_image_imports(self):
        """Exercises the non-pyramidal (img_data.scale0 AttributeError, now
        _get_base_resolution_array's plain-DataArray branch) path against a real
        single-scale foreign raster, not just the code path's presence."""
        n_cells = 6
        sdata = make_foreign_labels_native_sdata(n_cells=n_cells)

        data = convert_from_foreign_spatialdata(
            sdata,
            images={"morphology": {"key": "morphology", "pixel_size": 1.0}},
            cells={"main": {"table_key": "table", "cell_boundaries_data": ("cell_labels", 1.0)}},
        )

        img = data.images["morphology"]
        arr = img[0] if isinstance(img, list) else img
        assert arr.shape == (n_cells * 4, n_cells * 4)


class TestRGBFlagPreservedOnImport:
    def test_rgb_flag_preserved_on_import(self):
        """Regression: `is_rgb` was computed in `_add_images_to_insitudata`
        but never forwarded to `add_image`, so a channel-first (3, H, W)
        RGB array fell back to shape-based auto-detection - which inspects
        the width, not the channel count, and mis-classifies RGB images as
        non-RGB multi-channel images."""
        rgb_arr = DataArray(
            np.zeros((3, 8, 8), dtype=np.uint8), dims=("c", "y", "x"),
        )
        sdata = SpatialData(
            images={"img": Image2DModel.parse(rgb_arr, transformations={"global": Identity()})}
        )

        data = convert_from_foreign_spatialdata(
            sdata, images={"img": {"key": "img", "pixel_size": 1.0, "is_rgb": True}},
        )

        assert data.images.metadata["img"]["rgb"] is True


class TestShortDocumentedCallProducesValidInsitudata:
    def test_short_documented_call_produces_valid_insitudata(self):
        """The full auto-detection chain end to end: cells_key/cell_boundaries_data,
        seg_mask_value, and centroids are all derived from the table's own
        spatialdata_attrs, with only a table_key naming the source table.

        Under SD-A2, the minimal call now needs a resolvable pixel size (an Identity
        labels transform is no longer trusted as 1 um/pixel) - the extended fixture's
        vendor_pixel_size supplies both a scaled sibling and a kept obsm['spatial']."""
        sdata = make_foreign_labels_native_sdata(vendor_pixel_size=0.2125)

        data = convert_from_foreign_spatialdata(sdata, cells={"main": {"table_key": "table"}})

        assert isinstance(data, InSituData)
        assert data.cells["main"].is_synced
        # SD-B1: kept the vendor micrometre centroids, not label-derived pixel centroids.
        np.testing.assert_allclose(
            data.cells["main"].table.obsm["spatial"],
            sdata.tables["table"].obsm["spatial"],
        )


class TestCoordinateSystemSelection:
    def test_explicit_coordinate_system_overrides_global(self):
        """Real failure-mode: a silent wrong pick among multiple coordinate systems
        is a correctness bug. Hand-built (not the shared fixture) - a labels
        element (always Identity, per SD-A2 - its own transform is never trusted
        for pixel size) plus a shapes sibling carrying two coordinate systems with
        different Scale factors, so picking the wrong one is observable."""
        n_cells = 3
        size = n_cells * 4
        instance_ids = np.arange(1, n_cells + 1)

        cell_mask = np.zeros((size, size), dtype=np.uint32)
        for i, value in enumerate(instance_ids):
            cell_mask[i * 4, i * 4] = value
        cell_mask_arr = DataArray(cell_mask, dims=("y", "x"))
        labels = {
            "cell_labels": Labels2DModel.parse(
                cell_mask_arr, transformations={"global": Identity(), "aligned": Identity()},
            )
        }

        # sibling shapes element: Scale(1/0.3) in "aligned" -> pixel_size 0.3; a different
        # Scale in "global" -> a different pixel_size, so the coordinate_system choice
        # is what determines which one _resolve_foreign_pixel_size returns.
        circles_gdf = gpd.GeoDataFrame({
            "geometry": [Point(i * 4, i * 4).buffer(1.0) for i in range(n_cells)],
        })
        shapes = {
            "cell_circles": ShapesModel.parse(
                circles_gdf,
                transformations={
                    "global": Scale([1 / 0.6, 1 / 0.6], axes=("x", "y")),
                    "aligned": Scale([1 / 0.3, 1 / 0.3], axes=("x", "y")),
                },
            )
        }

        obs = pd.DataFrame({
            "cell_id": instance_ids,
            "region": pd.Categorical(["cell_labels"] * n_cells),
        })
        adata = AnnData(
            X=np.random.default_rng(0).random((n_cells, 2)),
            obs=obs,
            var=pd.DataFrame(index=["gene_0", "gene_1"]),
        )
        table = TableModel.parse(adata, region="cell_labels", region_key="region", instance_key="cell_id")
        sdata = SpatialData(labels=labels, shapes=shapes, tables={"table": table})

        data = convert_from_foreign_spatialdata(
            sdata, cells={"main": {"table_key": "table"}}, coordinate_system="aligned",
        )

        assert data.cells["main"].boundaries.metadata["cells"]["pixel_size"] == pytest.approx(0.3)


class TestMultiLayerImport:
    def test_two_cell_layers_from_distinct_tables_import_correctly(self):
        """The exact capability the old shared table_key could not express: two
        cell layers sourced from two distinct tables must land under their own
        given names, main/aux/etc. resolved from the *dict's* order. Layer
        'secondary' relies on region-driven auto-detection of its labels key
        (cell_boundaries_data) with its own, distinct pixel size (2.0, passed
        explicitly per SD-A2 - a labels element's own transform, Scale or not,
        is no longer trusted for pixel size) - if the per-iteration cells_key /
        cell_boundaries_data locals leaked from layer 'main' (explicit,
        pixel_size 1.0) instead of being reset per entry, auto-detection for
        'secondary' would be skipped and this test would fail."""
        n_cells = 3
        size = n_cells * 4

        ids_a = np.arange(1, n_cells + 1)
        mask_a = np.zeros((size, size), dtype=np.uint32)
        for i, value in enumerate(ids_a):
            mask_a[i * 4, i * 4] = value
        labels_a = Labels2DModel.parse(
            DataArray(mask_a, dims=("y", "x")), transformations={"global": Identity()},
        )
        obs_a = pd.DataFrame({
            "cell_id": ids_a, "region": pd.Categorical(["labels_a"] * n_cells),
        })
        adata_a = AnnData(
            X=np.random.default_rng(0).random((n_cells, 2)),
            obs=obs_a, var=pd.DataFrame(index=["gene_0", "gene_1"]),
        )
        table_a = TableModel.parse(adata_a, region="labels_a", region_key="region", instance_key="cell_id")

        ids_b = np.arange(101, 101 + n_cells)
        mask_b = np.zeros((size, size), dtype=np.uint32)
        for i, value in enumerate(ids_b):
            mask_b[i * 4, i * 4] = value
        labels_b = Labels2DModel.parse(
            DataArray(mask_b, dims=("y", "x")), transformations={"global": Identity()},
        )
        obs_b = pd.DataFrame({
            "cell_id": ids_b, "region": pd.Categorical(["labels_b"] * n_cells),
        })
        adata_b = AnnData(
            X=np.random.default_rng(1).random((n_cells, 2)),
            obs=obs_b, var=pd.DataFrame(index=["gene_0", "gene_1"]),
        )
        table_b = TableModel.parse(adata_b, region="labels_b", region_key="region", instance_key="cell_id")

        sdata = SpatialData(
            labels={"labels_a": labels_a, "labels_b": labels_b},
            tables={"table_a": table_a, "table_b": table_b},
        )

        data = convert_from_foreign_spatialdata(
            sdata,
            cells={
                "main": {"table_key": "table_a", "cell_boundaries_data": ("labels_a", 1.0)},
                "secondary": {"table_key": "table_b", "pixel_size": 2.0},
            },
        )

        assert set(data.cells.keys()) == {"main", "secondary"}
        assert data.cells.main_key == "main"
        assert data.cells["main"].boundaries.metadata["cells"]["pixel_size"] == 1.0
        assert data.cells["secondary"].boundaries.metadata["cells"]["pixel_size"] == pytest.approx(2.0)
        assert list(data.cells["secondary"].boundaries.seg_mask_value.compute()) == list(ids_b)


class TestForeignSpecValidation:
    def test_empty_cells_spec_raises_missing_table_key(self):
        """{} is uniformly invalid: table_key has no default anymore."""
        sdata = make_foreign_labels_native_sdata()

        with pytest.raises(ValueError, match="table_key"):
            convert_from_foreign_spatialdata(sdata, cells={"main": {}})

    def test_unknown_cells_spec_key_raises(self):
        """Catches the stringly-typed-spec footgun, e.g. a typo'd key name."""
        sdata = make_foreign_labels_native_sdata()

        with pytest.raises(ValueError, match="unknown"):
            convert_from_foreign_spatialdata(
                sdata, cells={"main": {"table_key": "table", "tabel_key": "table"}},
            )

    def test_units_spec_missing_units_key_raises(self):
        sdata = make_foreign_labels_native_sdata()

        with pytest.raises(ValueError, match="units_key"):
            convert_from_foreign_spatialdata(sdata, units={"visium": {"table_key": "table"}})


class TestForeignTranscriptsImport:
    def test_transcripts_branch_renames_coordinates_and_assigns(self):
        """The foreign reader's transcripts branch (`_assign_sdata_transcripts`,
        shared with the dialect reader) had zero coverage before this test - it is
        pure glue, but a plain SpatialData Points element (unlike the dialect
        reader's own dialect-produced frame) is the real shape this branch has to
        handle."""
        points_df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "feature_name": ["gene_0", "gene_1", "gene_0"],
        })
        sdata = SpatialData(points={"transcripts": PointsModel.parse(points_df)})

        data = convert_from_foreign_spatialdata(sdata, transcripts="transcripts")

        assert "x_location" in data.transcripts.columns
        assert "y_location" in data.transcripts.columns
        assert list(data.transcripts["feature_name"]) == ["gene_0", "gene_1", "gene_0"]


# ── SD-P1: foreign-import correctness (pixel size, nucleus parenting, mutation, channels) ──


class TestMinimalImportKeepsMicrometreCentroids:
    def test_minimal_import_keeps_micrometre_centroids(self):
        """SD-B1: the minimal/auto-detect path must keep the vendor micrometre centroids
        rather than silently overwriting them with label-derived pixel centroids - an
        Identity transform on a foreign labels raster does not imply 1 micrometer/pixel."""
        vendor_pixel_size = 0.2125
        sdata = make_foreign_labels_native_sdata(n_cells=5, vendor_pixel_size=vendor_pixel_size)

        data = convert_from_foreign_spatialdata(sdata, cells={"main": {"table_key": "table"}})

        imported = data.cells["main"].table.obsm["spatial"]
        vendor = sdata.tables["table"].obsm["spatial"]
        np.testing.assert_allclose(imported, vendor)

        # Not pixel-scaled: guards against a silent regression back to label-derived
        # pixel centroids (which would be ~1/vendor_pixel_size, ~4.7x off, larger) under
        # a different code path.
        vendor_span = vendor.max(axis=0) - vendor.min(axis=0)
        imported_span = imported.max(axis=0) - imported.min(axis=0)
        np.testing.assert_allclose(imported_span / vendor_span, 1.0)


class TestImportRaisesWhenPixelSizeUnresolvable:
    def test_import_raises_when_pixel_size_unresolvable(self):
        """SD-B1: no explicit pixel_size, no scaled sibling, and no existing obsm - raise
        loudly instead of silently treating an Identity transform as 1 micrometer/pixel."""
        sdata = make_foreign_labels_native_sdata(with_nucleus=False, n_cells=4)

        with pytest.raises(ValueError, match="pixel_size"):
            convert_from_foreign_spatialdata(sdata, cells={"main": {"table_key": "table"}})


class TestNucleusParentingCorrectWithLabelMap:
    def test_nucleus_parenting_correct_with_label_map(self):
        """Primary SD-B2 regression: real spatialdata-io output has no reliable 1:1
        nucleus->cell relationship (XOA v2-v4 nucleus labels are not cell-ordered).
        Passing label_map must parent every nucleus to its real (shifted-by-one, per
        the fixture) cell, not the same-index cell the old 1:1 fallback assumed."""
        n_cells = 6
        sdata = make_foreign_labels_native_sdata(n_cells=n_cells, with_nucleus_map=True, contiguous_ids=False)

        data = convert_from_foreign_spatialdata(
            sdata, cells={"main": {
                "table_key": "table",
                "cell_boundaries_data": ("cell_labels", 1.0),
                "nucleus_boundaries_data": ("nucleus_labels", 1.0),
                "label_map": "nucleus_boundaries",
            }},
        )

        boundaries = data.cells["main"].boundaries
        instance_ids = list(sdata.tables["table"].obs["cell_id"])
        expected_map = {
            int(instance_id) - 1: str((i + 1) % n_cells)
            for i, instance_id in enumerate(instance_ids)
        }
        assert boundaries.nucleus_to_cell_map == expected_map

        # 0 of N nuclei mis-parented: each nucleus raster label resolves, through the LUT,
        # to the seg_mask_value of its real (shifted) parent cell.
        lut = boundaries.label_lut("nuclei")
        seg_mask_value = boundaries.seg_mask_value.compute()
        n_checked = 0
        for i, instance_id in enumerate(instance_ids):
            expected_parent_seg_value = seg_mask_value[(i + 1) % n_cells]
            assert lut[int(instance_id)] == expected_parent_seg_value
            n_checked += 1
        assert n_checked == n_cells


class TestNucleusBoundariesWithoutLabelMapRaises:
    def test_nucleus_boundaries_without_label_map_raises(self):
        """SD decision 3: a nucleus raster with no reliable 1:1 relationship to cells must
        raise rather than silently mis-parenting under the old fallback."""
        sdata = make_foreign_labels_native_sdata(n_cells=4, with_nucleus_map=True)

        with pytest.raises(ValueError, match="label_map"):
            convert_from_foreign_spatialdata(
                sdata, cells={"main": {
                    "table_key": "table",
                    "cell_boundaries_data": ("cell_labels", 1.0),
                    "nucleus_boundaries_data": ("nucleus_labels", 1.0),
                }},
            )


class TestCallerSdataNotMutated:
    def test_caller_sdata_not_mutated(self):
        """SD-B8: the caller's SpatialData object must not be mutated by import. Forces
        spatial_source='labels' (an obsm overwrite) - the strongest exercise of the
        table-copy fix, since the pre-fix code mutated the table in place here."""
        sdata = make_foreign_labels_native_sdata(n_cells=4, vendor_pixel_size=0.2125)
        original_spatial = sdata.tables["table"].obsm["spatial"].copy()

        convert_from_foreign_spatialdata(
            sdata, cells={"main": {
                "table_key": "table",
                "cell_boundaries_data": ("cell_labels", 1.0),
                "spatial_source": "labels",
            }},
        )

        np.testing.assert_array_equal(sdata.tables["table"].obsm["spatial"], original_spatial)


class TestChannelNamesFromCCoord:
    def test_channel_names_from_c_coord(self):
        """SD-B11: real channel names (from the DataArray's own `c` coordinate) are used
        instead of the generic f'{name}_{i}' fallback - sanitized for use as image keys."""
        sdata = make_foreign_labels_native_sdata(n_cells=4, multichannel_morphology=True)

        data = convert_from_foreign_spatialdata(
            sdata, images={"morphology_focus": {"key": "morphology_focus", "pixel_size": 0.2125}},
        )

        assert set(data.images.keys()) == {
            "DAPI", "ATP1A1_CD45_E-Cadherin", "18S", "AlphaSMA_Vimentin",
        }


class TestWarnIgnoredSpatialdataParams:
    def test_warn_ignored_spatialdata_params_names_each_changed_value(self):
        """SD-B18: a single consolidated warning names every read_xenium parameter that
        differs from its default and is not honored by the spatialdata backend."""
        with pytest.warns(UserWarning) as record:
            _warn_ignored_spatialdata_params(
                nuclei_type="mip",
                load_cell_segmentation_images=True,
                load_background_images=True,
                transcript_mode="pandas",
                restructure_transcripts=True,
            )

        assert len(record) == 1
        message = str(record[0].message)
        for name in (
            "nuclei_type", "load_cell_segmentation_images", "load_background_images",
            "transcript_mode", "restructure_transcripts",
        ):
            assert name in message

    def test_warn_ignored_spatialdata_params_silent_on_defaults(self):
        """No warning at all when every parameter is left at its default - a real user
        calling read_xenium(backend='spatialdata') with no extra args must not be warned
        about parameters they never touched."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_ignored_spatialdata_params(
                nuclei_type="focus",
                load_cell_segmentation_images=False,
                load_background_images=False,
                transcript_mode="dask",
                restructure_transcripts=False,
            )

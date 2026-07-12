"""Labels-native foreign-store import tests (WP4).

Uses `make_foreign_labels_native_sdata` - a hand-built SpatialData object with no
InSituPy dialect descriptor, mimicking spatialdata-io Xenium output. Skipped until
WP4 fixes the documented `convert_from_spatialdata` defects. See
.log/reports/260711/spatialdata-work-packages/report-wp4-labels-native-import.md.

Each test wires in the foreign-store fixture; the skip reason and docstring
document the exact assertion to add once WP4 lands.
"""

import pytest

pytest.importorskip("spatialdata")

from tests.spatialdata_fixtures import make_foreign_labels_native_sdata  # noqa: E402


class TestSegMaskValueTakenFromInstanceKey:
    @pytest.mark.skip(
        reason="blocked on WP4: _create_boundaries_from_spatialdata "
               "(insitupy/spatialdata/_convert.py) fabricates "
               "seg_mask_value = arange(1, n+1) unconditionally."
    )
    def test_seg_mask_value_taken_from_instance_key(self):
        """Target assertion once WP4 lands::

            data = convert_from_spatialdata(
                sdata, cells_key="cell_labels", table_key="table",
                cell_boundaries_data=("cell_labels", 1.0),
            )
            expected = list(sdata.tables["table"].obs["cell_id"])
            assert list(data.cells["main"].boundaries.seg_mask_value.compute()) == expected
            # i.e. NOT arange(1, n+1) - the fixture's ids are deliberately non-contiguous.
        """
        make_foreign_labels_native_sdata(contiguous_ids=False, n_cells=6)


class TestCellOnlySegmentationImportsWithoutNucleus:
    @pytest.mark.skip(
        reason="blocked on WP4: _convert.py compares "
               "cell_boundaries_data[1] != nucleus_boundaries_data[1] before a "
               "None check, so passing only cell boundaries raises TypeError."
    )
    def test_cell_only_segmentation_imports_without_nucleus(self):
        """Target assertion once WP4 lands::

            data = convert_from_spatialdata(
                sdata, cells_key="cell_labels", table_key="table",
                cell_boundaries_data=("cell_labels", 1.0), nucleus_boundaries_data=None,
            )  # must not raise
            assert data.cells["main"].boundaries is not None
        """
        make_foreign_labels_native_sdata(with_nucleus=False, n_cells=4)


class TestIdentityTransformImageImports:
    @pytest.mark.skip(
        reason="blocked on WP4: the general image-loading path in "
               "_add_images_to_insitudata is unverified against a real "
               "Identity-transformed, single-scale array."
    )
    def test_identity_transform_image_imports(self):
        """Target assertion once WP4 lands::

            data = convert_from_spatialdata(sdata, image_data=("morphology", 1.0), ...)
            assert data.images["morphology"] is not None
            assert data.images.metadata["morphology"]["pixel_size"] == 1.0
        """
        make_foreign_labels_native_sdata()


class TestSingleScaleImageImports:
    @pytest.mark.skip(
        reason="blocked on WP4: exercises the non-pyramidal "
               "(img_data.scale0 AttributeError) branch of _add_images_to_insitudata "
               "against a real single-scale foreign raster, not just the code "
               "path's presence."
    )
    def test_single_scale_image_imports(self):
        """Target assertion once WP4 lands: same call as
        `test_identity_transform_image_imports`; additionally assert the imported
        array's shape matches the fixture's single-scale raster (no pyramid levels
        to pick from).
        """
        make_foreign_labels_native_sdata()


class TestShortDocumentedCallProducesValidInsitudata:
    @pytest.mark.skip(
        reason="blocked on WP4: the short, well-documented entry point for "
               "spatialdata-io input does not exist yet - this test also pins down "
               "the call shape WP4 settles on."
    )
    def test_short_documented_call_produces_valid_insitudata(self):
        """Target assertion once WP4 lands (exact call shape TBD by WP4's own
        planning; sketched here per its report's ergonomics goal)::

            data = convert_from_spatialdata(sdata)  # minimal args, dialect-less store
            assert isinstance(data, InSituData)
            assert data.cells["main"].is_synced
        """
        make_foreign_labels_native_sdata()

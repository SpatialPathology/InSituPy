"""Tests for the serialized concatenated build_table() union element (WP3).

See .log/reports/260713/spatialdata-concat-table-element/report-spatialdata-concat-table-element.md.
"""

import numpy as np
import pytest

pytest.importorskip("spatialdata")

from insitupy.spatialdata.convert import (  # noqa: E402
    convert_from_spatialdata,
    convert_table_from_spatialdata,
    convert_to_spatialdata,
)
from tests.spatialdata_fixtures import make_experiment, roundtrip_through_zarr  # noqa: E402


def _built_experiment(tmp_path, n_samples=2):
    """`make_experiment()` plus a table build, giving build_table() a path to write to.

    Mirrors tests/test_experiment_table.py's `_make_experiment_with_path` convention of
    setting `._path` directly rather than calling the full `.saveas()` - build_table()'s
    `in_memory` method only needs `self.path` to know where to write, not a fully
    persisted-to-disk experiment.
    """
    exp = make_experiment(n_samples=n_samples)
    exp._path = tmp_path
    exp.build_table()
    return exp


class TestPresenceMatrixSurvivesZarrRoundtrip:
    def test_presence_matrix_survives_zarr_roundtrip(self, tmp_path):
        """Uses `make_experiment()`'s heterogeneous, partially-overlapping gene
        panels - exactly why that fixture needed real panel heterogeneity: a fully
        disjoint or identical panel would not exercise the presence record.
        """
        exp = _built_experiment(tmp_path)
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")
        reconstructed = sdata2.tables["TABLES.main"]
        np.testing.assert_array_equal(
            reconstructed.uns["_insitupy_gene_presence"],
            exp.table["main"].uns["_insitupy_gene_presence"],
        )


class TestRegionListSurvivesZarrRoundtrip:
    def test_region_list_survives_zarr_roundtrip(self, tmp_path):
        exp = _built_experiment(tmp_path)
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")
        reconstructed = sdata2.tables["TABLES.main"]
        expected_regions = {f"SAMPLE.sample_{i}..CELLS.main.circles" for i in range(2)}
        assert set(reconstructed.uns["spatialdata_attrs"]["region"]) == expected_regions
        assert expected_regions <= set(sdata2.shapes)  # region list names real elements


class TestTableReconstructedFromStoreEqualsOriginal:
    def test_table_reconstructed_from_store_equals_original(self, tmp_path):
        """Full-experiment case."""
        exp = _built_experiment(tmp_path)
        original = exp.table["main"]
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")
        reconstructed = convert_table_from_spatialdata(sdata2, "main")
        assert reconstructed.obs_names.equals(original.obs_names)
        assert reconstructed.var_names.equals(original.var_names)
        assert reconstructed.X is not None
        assert original.X is not None
        np.testing.assert_allclose(np.asarray(reconstructed.X), np.asarray(original.X))


class TestTableReconstructedForSampleSubsetView:
    def test_table_reconstructed_for_sample_subset_view(self, tmp_path):
        """Uses `InSituExperiment._subset(..., as_view=True)` (already-working real
        API, see tests/test_experiment_table.py::TestViewTable) to build the
        in-memory expectation.
        """
        exp = _built_experiment(tmp_path)
        view = exp._subset(slice(0, 1), as_view=True)
        expected = view.table["main"]  # inner-over-covered set recomputed for the subset
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")
        reconstructed = convert_table_from_spatialdata(sdata2, "main", covered_labels=["sample_0"])
        assert reconstructed.var_names.equals(expected.var_names)
        assert reconstructed.X is not None
        assert expected.X is not None
        np.testing.assert_allclose(np.asarray(reconstructed.X), np.asarray(expected.X))


class TestConcatTableRowsLinkToCorrectCircles:
    def test_concat_table_rows_link_to_correct_circles(self, tmp_path):
        """Real failure mode: the suffix-stripping logic that recovers each row's
        original (pre-concatenation) cell name could silently mislink cells to the
        wrong sample or wrong circles instance. Neither the presence-matrix test nor
        the region-*list* test above would catch this - they don't inspect per-row
        correctness - but it's exactly what makes the element meaningful to an
        external SpatialData-aware viewer rather than merely round-trippable by
        InSituPy's own reader.
        """
        exp = _built_experiment(tmp_path)
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")
        reconstructed = sdata2.tables["TABLES.main"]

        for meta, xd in exp.iterdata():
            uid = meta["uid"]
            circles_key = f"SAMPLE.{uid}..CELLS.main.circles"
            sample_rows = reconstructed.obs[reconstructed.obs["region"] == circles_key]
            expected_names = set(xd.cells["main"].table.obs_names)
            assert set(sample_rows["cell_id"]) == expected_names


class TestConcatTableDoesNotPolluteSampleReconstruction:
    def test_concat_table_does_not_pollute_sample_reconstruction(self, tmp_path):
        """Real failure mode: without skipping the None-keyed (non-per-sample) group
        in convert_from_spatialdata's per-sample reconstruction loop, the
        un-prefixed TABLES.main element would be misread as a bogus extra sample
        with uid=None.
        """
        exp = _built_experiment(tmp_path)
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")

        reconstructed_exp = convert_from_spatialdata(sdata2)
        assert len(reconstructed_exp) == 2
        assert None not in set(reconstructed_exp._metadata["uid"])


class TestConcatTableExportScopedToView:
    """Real failure mode: before the row-scoping fix, exporting an
    InSituExperimentView whose parent had already called build_table() silently
    dropped the entire TABLES.<layer> element - every sample outside the view was
    "unresolved" against exported_keys, and any unresolved label skipped the whole
    layer rather than just that sample's rows.
    """

    def test_concat_table_present_and_scoped_for_view_export(self, tmp_path):
        exp = _built_experiment(tmp_path, n_samples=3)
        view = exp._subset(slice(0, 2), as_view=True)

        sdata = convert_to_spatialdata(view)

        assert "TABLES.main" in sdata.tables
        reconstructed = sdata.tables["TABLES.main"]
        expected_regions = {f"SAMPLE.sample_{i}..CELLS.main.circles" for i in range(2)}
        assert set(reconstructed.obs["region"]) == expected_regions

    def test_concat_table_empty_view_does_not_raise(self, tmp_path):
        """An empty view covers zero samples of the built table; the concat-table
        element is skipped entirely (there is nothing to link region-wise), rather
        than producing a degenerate 0-row table that spatialdata's TableModel would
        reject anyway (an empty ``cell_id`` column defaults to float64, not a valid
        instance-key dtype).
        """
        exp = _built_experiment(tmp_path, n_samples=3)
        view = exp._subset(slice(0, 0), as_view=True)

        sdata = convert_to_spatialdata(view)  # must not raise

        assert "TABLES.main" not in sdata.tables

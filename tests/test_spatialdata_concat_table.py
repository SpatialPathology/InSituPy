"""Tests for the serialized concatenated build_table() union element (WP3).

Skipped until WP3 adds a TABLES.<layer> element to the exporter and a reader path
that reconstructs `.table` from it. See
.log/reports/260711/spatialdata-work-packages/report-wp3-concat-table-element.md.

Each test wires in the shared multi-panel experiment fixture and the parts of the
export/round-trip that already work today; the skip reason and docstring document
the exact assertion to add once WP3 lands.
"""

import pytest

pytest.importorskip("spatialdata")

from insitupy.spatialdata.convert import convert_to_spatialdata  # noqa: E402
from tests.spatialdata_fixtures import make_experiment, roundtrip_through_zarr  # noqa: E402


class TestPresenceMatrixSurvivesZarrRoundtrip:
    @pytest.mark.skip(reason="blocked on WP3: no TABLES.<layer> element is written by the exporter yet.")
    def test_presence_matrix_survives_zarr_roundtrip(self, tmp_path):
        """Uses `make_experiment()`'s heterogeneous, partially-overlapping gene
        panels - exactly why that fixture needed real panel heterogeneity: a fully
        disjoint or identical panel would not exercise the presence record.
        Target assertion once WP3 lands::

            exp.build_table()  # requires exp.saveas(path) first
            sdata2 = roundtrip_through_zarr(convert_to_spatialdata(exp), tmp_path)
            reconstructed = sdata2.tables["TABLES.main"]
            np.testing.assert_array_equal(
                reconstructed.uns["_insitupy_gene_presence"],
                exp.table["main"].uns["_insitupy_gene_presence"],
            )
        """
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")


class TestRegionListSurvivesZarrRoundtrip:
    @pytest.mark.skip(reason="blocked on WP3: no TABLES.<layer> element is written by the exporter yet.")
    def test_region_list_survives_zarr_roundtrip(self, tmp_path):
        """Target assertion once WP3 lands::

            reconstructed = sdata2.tables["TABLES.main"]
            expected_regions = {f"SAMPLE.sample_{i}..CELLS.main.circles" for i in range(2)}
            assert set(reconstructed.uns["spatialdata_attrs"]["region"]) == expected_regions
            assert expected_regions <= set(sdata2.shapes)  # region list names real elements
        """
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")


class TestTableReconstructedFromStoreEqualsOriginal:
    @pytest.mark.skip(reason="blocked on WP3: no reader path reconstructs `.table` from a store element yet.")
    def test_table_reconstructed_from_store_equals_original(self, tmp_path):
        """Full-experiment case. Target assertion once WP3 lands::

            exp.build_table()
            original = exp.table["main"]
            sdata2 = roundtrip_through_zarr(convert_to_spatialdata(exp), tmp_path)
            reconstructed = <WP3's reader-side reconstruction of `.table` from sdata2>
            assert reconstructed.obs_names.equals(original.obs_names)
            assert reconstructed.var_names.equals(original.var_names)
            np.testing.assert_allclose(reconstructed.X, original.X)
        """
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")


class TestTableReconstructedForSampleSubsetView:
    @pytest.mark.skip(reason="blocked on WP3: subset reconstruction from the stored element is unimplemented.")
    def test_table_reconstructed_for_sample_subset_view(self, tmp_path):
        """Uses `InSituExperiment._subset(..., as_view=True)` (already-working real
        API, see tests/test_experiment_table.py::TestViewTable) to build the
        in-memory expectation. Target assertion once WP3 lands::

            exp.build_table()
            view = exp._subset(slice(0, 1), as_view=True)
            expected = view.table["main"]  # inner-over-covered set recomputed for the subset
            sdata2 = roundtrip_through_zarr(convert_to_spatialdata(exp), tmp_path)
            reconstructed = <WP3's reader-side reconstruction of the subset's `.table`>
            assert reconstructed.var_names.equals(expected.var_names)
            np.testing.assert_allclose(reconstructed.X, expected.X)
        """
        exp = make_experiment(n_samples=2)
        exp._subset(slice(0, 1), as_view=True)
        sdata = convert_to_spatialdata(exp)
        roundtrip_through_zarr(sdata, tmp_path, name="concat.zarr")

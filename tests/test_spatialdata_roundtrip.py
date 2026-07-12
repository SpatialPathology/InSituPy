"""Round-trip identity tests for the SpatialData reader (WP2).

Skipped until WP2 implements `convert_from_spatialdata` reconstructing a real
InSituData/InSituExperiment from the SAMPLE.<uid>.. dialect. See
.log/reports/260711/spatialdata-work-packages/report-wp2-reader-parity-consolidation.md.

Each test wires in the shared fixtures and performs the parts of the round trip
that already work today (build -> export -> write/read through disk); the skip
reason and docstring document the exact assertion to add once WP2 lands.
"""

import pytest

pytest.importorskip("spatialdata")

from insitupy.spatialdata.convert import convert_to_spatialdata  # noqa: E402
from tests.spatialdata_fixtures import (  # noqa: E402
    make_experiment,
    make_insitudata,
    make_units,
    roundtrip_through_zarr,
)


class TestSingleSampleRoundtripIdentity:
    @pytest.mark.skip(
        reason="blocked on WP2: convert_from_spatialdata requires hand-named keys "
               "and does not auto-detect the dialect."
    )
    def test_single_sample_roundtrip_identity(self, tmp_path):
        """Target assertion once WP2 lands::

            reconstructed = convert_from_spatialdata(sdata2)  # dialect auto-detected
            assert reconstructed.cells.table.obs_names.equals(xd.cells.table.obs_names)
            np.testing.assert_allclose(reconstructed.cells.table.X, xd.cells.table.X)
            assert reconstructed.cells["main"].is_synced
        """
        xd = make_insitudata(n_cells=6, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        roundtrip_through_zarr(sdata, tmp_path, name="single.zarr")


class TestMultiSampleRoundtripReconstructsExperiment:
    @pytest.mark.skip(
        reason="blocked on WP2: no SAMPLE. prefix reconstruction or "
               "InSituExperiment return path exists yet."
    )
    def test_multi_sample_roundtrip_reconstructs_experiment(self, tmp_path):
        """Target assertion once WP2 lands::

            reconstructed = convert_from_spatialdata(sdata2)  # -> InSituExperiment
            assert isinstance(reconstructed, InSituExperiment)
            assert set(reconstructed.metadata["uid"]) == {"sample_0", "sample_1"}
            # heterogeneous panels preserved per sample, not unioned or truncated:
            assert list(reconstructed.data[0].cells.table.var_names) == panel_0
            assert list(reconstructed.data[1].cells.table.var_names) == panel_1
        """
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        roundtrip_through_zarr(sdata, tmp_path, name="experiment.zarr")


class TestUnitsRoundtripUsesCorrectTableKey:
    @pytest.mark.skip(
        reason="blocked on WP2: convert.py's units_key handling builds "
               "SpatialUnitsData from `table_key` (the cells key) instead of a "
               "units-specific table key - see WP2 report background."
    )
    def test_units_roundtrip_uses_correct_table_key(self, tmp_path):
        """Targeted regression test for the named bug (not a generic round-trip
        check). Target assertion once WP2 lands::

            reconstructed = convert_from_spatialdata(sdata2, ...)
            assert reconstructed.units["niche"].table.obs_names.equals(su.table.obs_names)
            # i.e. NOT the cells table - what today's units_key/table_key mixup
            # would silently attach instead.
        """
        xd = make_insitudata(n_cells=4)
        su = make_units(["u0", "u1", "u2"], unit_type="niche", seed=3)
        xd.add_units(su)
        sdata = convert_to_spatialdata(xd)
        roundtrip_through_zarr(sdata, tmp_path, name="units.zarr")


class TestIsSyncedPreservedAfterRoundtrip:
    @pytest.mark.skip(
        reason="blocked on WP2: convert_from_spatialdata has no dialect-aware "
               "reconstruction path to preserve is_synced against yet."
    )
    def test_is_synced_preserved_after_roundtrip(self, tmp_path):
        """Target assertion once WP2 lands::

            reconstructed = convert_from_spatialdata(sdata2)
            assert reconstructed.cells["main"].is_synced
            np.testing.assert_array_equal(
                reconstructed.cells["main"].boundaries.seg_mask_value.compute(),
                xd.cells["main"].boundaries.seg_mask_value.compute(),
            )
        """
        xd = make_insitudata(n_cells=6, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        roundtrip_through_zarr(sdata, tmp_path, name="synced.zarr")

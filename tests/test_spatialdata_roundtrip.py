"""Round-trip identity tests for the SpatialData reader (WP2).

Exercises `convert_from_spatialdata` reconstructing a real InSituData/InSituExperiment
from the SAMPLE.<uid>.. dialect - a true inverse of `convert_to_spatialdata`, verified
through a real disk write/read via `roundtrip_through_zarr`.
"""

import shutil

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("spatialdata")

from insitupy.experiment.data import InSituExperiment  # noqa: E402
from insitupy.spatialdata.convert import convert_from_spatialdata, convert_to_spatialdata  # noqa: E402
from tests.spatialdata_fixtures import (  # noqa: E402
    make_experiment,
    make_insitudata,
    make_transcripts_df,
    make_units,
    roundtrip_through_zarr,
)


class TestSingleSampleRoundtripIdentity:
    def test_single_sample_roundtrip_identity(self, tmp_path):
        xd = make_insitudata(n_cells=6, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="single.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        assert reconstructed.cells["main"].table.obs_names.equals(xd.cells["main"].table.obs_names)
        np.testing.assert_allclose(reconstructed.cells["main"].table.X, xd.cells["main"].table.X)
        assert reconstructed.cells["main"].is_synced


class TestMultiSampleRoundtripReconstructsExperiment:
    def test_multi_sample_roundtrip_reconstructs_experiment(self, tmp_path):
        exp = make_experiment(n_samples=2)
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="experiment.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        assert isinstance(reconstructed, InSituExperiment)
        assert set(reconstructed.metadata["uid"]) == {"sample_0", "sample_1"}

        # Heterogeneous panels preserved per sample, not unioned or truncated -
        # match by uid since sample order is not guaranteed to be preserved.
        by_uid = {meta["uid"]: d for meta, d in exp.iterdata()}
        rec_by_uid = {meta["uid"]: d for meta, d in reconstructed.iterdata()}
        for uid, orig in by_uid.items():
            rec = rec_by_uid[uid]
            assert list(rec.cells["main"].table.var_names) == list(orig.cells["main"].table.var_names)


class TestUnitsRoundtripUsesCorrectTableKey:
    def test_units_roundtrip_uses_correct_table_key(self, tmp_path):
        """Targeted regression test for the named bug (not a generic round-trip
        check): convert.py's units_key handling used to build SpatialUnitsData
        from `table_key` (the cells key) instead of a units-specific table key -
        structurally impossible in the new dialect-driven reader, since each
        unit's own table key is derived per-key, never reused across modalities.
        """
        xd = make_insitudata(n_cells=4)
        su = make_units(["u0", "u1", "u2"], unit_type="niche", seed=3)
        xd.add_units(su)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="units.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        # i.e. NOT the cells table - what the units_key/table_key mixup would
        # silently attach instead.
        assert reconstructed.units["niche"].table.obs_names.equals(su.table.obs_names)


class TestIsSyncedPreservedAfterRoundtrip:
    def test_is_synced_preserved_after_roundtrip(self, tmp_path):
        """Uses non-contiguous seg_mask_value - the real-data case - so this
        only passes because of the _insitupy_seg_mask_value obs column, not by
        coincidence of identity ordering."""
        xd = make_insitudata(n_cells=6, with_boundaries=True, contiguous_seg_mask_value=False)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="synced.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        assert reconstructed.cells["main"].is_synced
        np.testing.assert_array_equal(
            reconstructed.cells["main"].boundaries.seg_mask_value.compute(),
            xd.cells["main"].boundaries.seg_mask_value.compute(),
        )


# ── Nucleus-to-cell mapping round trip (WP2 decision 5) ────────────────────────

class TestNucleusMappingRoundtrip:
    def test_multinucleated_mapping_survives_roundtrip(self, tmp_path):
        """Real failure-mode test: a cell with 2 nuclei and a cell with 0 nuclei -
        the case the deleted arange(1, n+1)-style fabrication would get wrong."""
        xd = make_insitudata(n_cells=6, with_boundaries=True, with_multinucleated_cells=True)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="nucleus_map.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        boundaries = reconstructed.cells["main"].boundaries
        assert boundaries.nucleus_to_cell_map == xd.cells["main"].boundaries.nucleus_to_cell_map
        np.testing.assert_array_equal(boundaries.nucleus_count, xd.cells["main"].boundaries.nucleus_count)

    def test_no_nucleus_map_for_ordinary_boundaries(self, tmp_path):
        """Lock-in: the ordinary (non-multinucleated) case must not regress -
        no CELLS.<key>.nucleus_map element, and nucleus_to_cell_map/nucleus_count
        reconstruct as None, matching BoundariesData's own "not available" default."""
        xd = make_insitudata(n_cells=4, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="no_nucleus_map.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        boundaries = reconstructed.cells["main"].boundaries
        assert boundaries.nucleus_to_cell_map is None
        assert boundaries.nucleus_count is None


# ── Pixel-size fidelity (caught the inverted-formula bug during planning) ─────

class TestPixelSizeFidelityAcrossRoundtrip:
    def test_image_pixel_size_survives_roundtrip(self, tmp_path):
        xd = make_insitudata(n_cells=2, with_image=True)  # pixel_size=0.5 in the fixture
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="pixel_size.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        assert reconstructed.images.metadata["dapi"]["pixel_size"] == pytest.approx(0.5)


# ── Transcripts feature_name dtype (Bug 1: MemoryError on .show()) ─────────────

class TestTranscriptsFeatureNameStaysCategorical:
    def test_feature_name_stays_categorical_after_roundtrip(self, tmp_path):
        """The exporter deliberately stores feature_name as a known categorical
        so `.unique()` stays O(#categories). An importer-side `.astype(str)`
        used to discard that dtype, which only detonated later when the lazy
        transcript viewer computed `.unique()` over full partitions -
        allocating one Python str per row instead of one per gene."""
        xd = make_insitudata(n_cells=4)
        xd.transcripts = make_transcripts_df(seed=0)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="transcripts_dtype.zarr")

        reconstructed = convert_from_spatialdata(sdata2, verbose=False)
        dtype = reconstructed.transcripts["feature_name"].dtype
        assert isinstance(dtype, pd.CategoricalDtype), f"expected categorical, got {dtype!r}"

    def test_saveas_survives_mismatched_dictionary_widths_across_partitions(self, tmp_path):
        """Bug 2, sibling of Bug 1 above: keeping feature_name categorical (as Bug 1's fix
        requires) means a transcripts frame read back from a partitioned parquet store - like
        a SpatialData Points element - has each partition's dictionary index width inferred
        independently (pyarrow picks int8 for <=127 local categories, int16 otherwise, ...).
        `.saveas()` -> `_save_transcripts` used to infer its target parquet schema from a
        single partition, so a differently-sized partition later on failed to convert
        ('Integer value ... not in range: -128 to 127'). Reproduces that condition directly
        with two independently-written parquet files (mirroring two partitions of a real,
        large transcript table) rather than depending on partition sizing internals."""
        parts_dir = tmp_path / "mismatched_parts"
        parts_dir.mkdir()

        few_genes = [f"gene_{i}" for i in range(5)]
        many_genes = [f"gene_{i}" for i in range(200)]  # >127 -> forces int16 indices

        # cell_id is also categorical here (a second categorical column, distinct
        # from feature_name) with the same mismatched-cardinality-per-partition
        # shape: 5 categories in df_small -> int8 index, 200 in df_large -> int16.
        # This is the case the old name-based guard (hardcoded to "feature_name")
        # would miss entirely.
        few_cell_ids = [f"cell_{i}" for i in range(len(few_genes))]
        many_cell_ids = [f"cell_{i}" for i in range(len(many_genes))]

        df_small = pd.DataFrame({
            "x_location": np.zeros(len(few_genes)),
            "y_location": np.zeros(len(few_genes)),
            "z_location": np.zeros(len(few_genes)),
            "feature_name": pd.Categorical(few_genes),
            "cell_id": pd.Categorical(few_cell_ids),
        })
        df_large = pd.DataFrame({
            "x_location": np.zeros(len(many_genes)),
            "y_location": np.zeros(len(many_genes)),
            "z_location": np.zeros(len(many_genes)),
            "feature_name": pd.Categorical(many_genes),
            "cell_id": pd.Categorical(many_cell_ids),
        })
        pq.write_table(pa.Table.from_pandas(df_small, preserve_index=False), parts_dir / "part.0.parquet")
        pq.write_table(pa.Table.from_pandas(df_large, preserve_index=False), parts_dir / "part.1.parquet")

        xd = make_insitudata(n_cells=4)
        xd.transcripts = dd.read_parquet(parts_dir)

        out_path = tmp_path / "saved_project"
        xd.saveas(out_path, overwrite=True)  # must not raise

        saved = dd.read_parquet(out_path / "transcripts" / "transcripts.parquet").compute()
        assert sorted(saved["feature_name"].astype(str)) == sorted(few_genes + many_genes)
        assert saved["cell_id"].nunique() == len(many_cell_ids)

        shutil.rmtree(parts_dir)

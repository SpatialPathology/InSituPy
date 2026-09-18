"""SD-P2: SpatialData-dialect writer/reader hardening.

Value-level and failure-mode tests for the six silent data-loss / corruption
paths hardened in SD-P2 (SD-B3, SD-B5, SD-B6, SD-B7, SD-B9, SD-B10) plus the
dialect round-trip half of SD-B17 (assert values, not just keys).

All tests are skipped when the spatialdata package is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("spatialdata")

from scipy.sparse import csr_matrix, issparse  # noqa: E402

from insitupy.spatialdata._convert import (  # noqa: E402
    _generate_spatialdata_key,
    _merge_dicts_with_warning,
    _parse_dialect_key,
)
from insitupy.spatialdata.convert import (  # noqa: E402
    convert_from_spatialdata,
    convert_to_spatialdata,
)
from tests.spatialdata_fixtures import (  # noqa: E402
    make_experiment,
    make_insitudata,
    make_transcripts_df,
    make_units,
    poly_gdf,
    roundtrip_through_zarr,
)


# ── SD-B3: reserved obs bookkeeping names ──────────────────────────────────────

class TestReservedObsNames:
    @pytest.mark.parametrize("colname", ["region", "cell_id", "_insitupy_seg_mask_value"])
    def test_reserved_cell_obs_name_raises(self, colname):
        """A user obs column named like the writer's own bookkeeping columns would
        be overwritten on write and dropped on read - refuse instead of losing it."""
        xd = make_insitudata(n_cells=4)
        xd.cells["main"].table.obs[colname] = "foo"
        with pytest.raises(ValueError, match="reserved"):
            convert_to_spatialdata(xd)

    @pytest.mark.parametrize("colname", ["region", "unit_id"])
    def test_reserved_unit_obs_name_raises(self, colname):
        xd = make_insitudata(n_cells=4)
        su = make_units(["u0", "u1"], seed=1)
        su.table.obs[colname] = "foo"
        xd.add_units(su)
        with pytest.raises(ValueError, match="reserved"):
            convert_to_spatialdata(xd)

    def test_non_reserved_obs_column_survives_roundtrip(self, tmp_path):
        """Guard against over-firing: an ordinary user obs column exports and
        round-trips intact."""
        xd = make_insitudata(n_cells=4)
        xd.cells["main"].table.obs["celltype"] = pd.Categorical(["A", "B", "A", "B"])
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="obs_survives.zarr")
        rec = convert_from_spatialdata(sdata2, verbose=False)
        assert list(rec.cells["main"].table.obs["celltype"]) == ["A", "B", "A", "B"]

    def test_double_roundtrip_no_spurious_collision(self, tmp_path):
        """A table read back from a store has had the writer's bookkeeping columns
        stripped, so re-exporting it must not spuriously trip the SD-B3 guard."""
        xd = make_insitudata(n_cells=4, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="double.zarr")
        rec = convert_from_spatialdata(sdata2, verbose=False)
        convert_to_spatialdata(rec)  # must not raise


# ── SD-B5: two source names sanitising to the same dialect key ─────────────────

class TestSanitisedKeyCollision:
    def test_annotation_names_colliding_by_dot_dash_raise(self):
        """'my-roi' and 'my.roi' both sanitise to 'ANNOTATIONS.my_roi'; without the
        guard the second silently overwrites the first inside the builder."""
        xd = make_insitudata(n_cells=2)
        xd.annotations.add_data(data=poly_gdf("a"), key="my-roi", scale_factor=1.0)
        xd.annotations.add_data(data=poly_gdf("b"), key="my.roi", scale_factor=1.0)
        with pytest.raises(ValueError, match="same SpatialData dialect key"):
            convert_to_spatialdata(xd)

    def test_merge_dicts_raises_on_duplicate(self):
        """The cross-modality / cross-sample backstop: merging two dicts that share
        a key (e.g. two samples with the same uid) refuses rather than overwriting."""
        with pytest.raises(ValueError, match="Duplicate SpatialData dialect key"):
            _merge_dicts_with_warning({"CELLS.main.table": 1}, {"CELLS.main.table": 2})


# ── SD-B7: partially-written store ─────────────────────────────────────────────

class TestPartialStore:
    def _store_missing_cell_table(self, tmp_path, name):
        xd = make_insitudata(n_cells=4, with_boundaries=True)
        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name=name)
        # Simulate a write that died after the descriptor but before the table element.
        del sdata2.tables["CELLS.main.table"]
        return sdata2

    def test_partial_store_warns_and_skips(self, tmp_path):
        sdata2 = self._store_missing_cell_table(tmp_path, "partial_warn.zarr")
        with pytest.warns(UserWarning, match="partially written"):
            rec = convert_from_spatialdata(sdata2, verbose=False)
        assert "main" not in rec.cells.keys()

    def test_partial_store_strict_raises(self, tmp_path):
        sdata2 = self._store_missing_cell_table(tmp_path, "partial_strict.zarr")
        with pytest.raises(ValueError, match="partially written"):
            convert_from_spatialdata(sdata2, verbose=False, strict=True)


# ── SD-B9: 2D / unassigned transcript frames ───────────────────────────────────

class TestTranscriptShapes:
    def test_transcripts_2d_export_and_roundtrip(self, tmp_path):
        xd = make_insitudata(n_cells=4)
        xd.transcripts = make_transcripts_df(n=6).drop(columns=["z_location"])
        sdata = convert_to_spatialdata(xd)  # must not raise (no z_location)
        assert len(sdata.points) == 1
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="tx_2d.zarr")
        rec = convert_from_spatialdata(sdata2, verbose=False)
        cols = set(rec.transcripts.columns)
        assert {"x_location", "y_location"} <= cols
        assert "z_location" not in cols

    def test_transcripts_unassigned_export(self, tmp_path):
        xd = make_insitudata(n_cells=4)
        xd.transcripts = make_transcripts_df(n=6).drop(columns=["cell_id"])
        sdata = convert_to_spatialdata(xd)  # must not raise (no cell_id)
        assert len(sdata.points) == 1

    def test_transcripts_missing_xy_raises(self):
        xd = make_insitudata(n_cells=4)
        xd.transcripts = make_transcripts_df(n=6).drop(columns=["y_location"])
        with pytest.raises(ValueError, match="coordinate column"):
            convert_to_spatialdata(xd)


# ── SD-B10: uid containing '.' / '..' ──────────────────────────────────────────

class TestUidWithDot:
    def test_generate_parse_roundtrip_uid_with_single_dot(self):
        key = _generate_spatialdata_key(sample_id="s.1", modality="cells", locator=["main", "table"])
        uid, modality, locator = _parse_dialect_key(key)
        assert uid == "s.1"
        assert modality == "CELLS"
        assert locator == ["main", "table"]

    def test_generate_rejects_double_dot_uid(self):
        with pytest.raises(ValueError, match=r"'\.\.'"):
            _generate_spatialdata_key(sample_id="s..1", modality="cells", locator=["main", "table"])

    def test_experiment_uid_with_dot_roundtrips(self, tmp_path):
        exp = make_experiment(n_samples=1)
        exp._metadata.loc[0, "uid"] = "s.1"
        sdata = convert_to_spatialdata(exp)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="uid_dot.zarr")
        rec = convert_from_spatialdata(sdata2, verbose=False)
        assert set(rec.metadata["uid"]) == {"s.1"}
        # the sample's cells actually reconstructed (not silently emptied)
        _meta, rec_xd = next(iter(rec.iterdata()))
        assert rec_xd.cells["main"].table.n_obs > 0


# ── SD-B17 (dialect half): value-level round-trip fidelity ──────────────────────

class TestValueLevelRoundtrip:
    def test_sparse_x_layers_obs_varm_uns_survive(self, tmp_path):
        """Assert values, not just element keys: sparse X, a sparse layer, a
        categorical obs column, varm, and a uns payload all round-trip."""
        xd = make_insitudata(n_cells=6, n_genes=4)
        table = xd.cells["main"].table
        dense = np.asarray(table.X, dtype=float)

        table.X = csr_matrix(dense)
        table.layers["counts"] = csr_matrix(dense * 2.0)
        table.obs["celltype"] = pd.Categorical(["A", "B"] * 3)
        table.varm["loadings"] = np.arange(table.n_vars * 2).reshape(table.n_vars, 2).astype(float)
        table.uns["note"] = "hello"

        sdata = convert_to_spatialdata(xd)
        sdata2 = roundtrip_through_zarr(sdata, tmp_path, name="values.zarr")
        rt = convert_from_spatialdata(sdata2, verbose=False).cells["main"].table

        x_rt = rt.X.toarray() if issparse(rt.X) else np.asarray(rt.X)
        np.testing.assert_allclose(x_rt, dense)

        assert "counts" in rt.layers
        layer_rt = rt.layers["counts"]
        layer_rt = layer_rt.toarray() if issparse(layer_rt) else np.asarray(layer_rt)
        np.testing.assert_allclose(layer_rt, dense * 2.0)

        assert list(rt.obs["celltype"]) == ["A", "B"] * 3
        np.testing.assert_allclose(
            rt.varm["loadings"],
            np.arange(rt.n_vars * 2).reshape(rt.n_vars, 2),
        )
        assert rt.uns.get("note") == "hello"

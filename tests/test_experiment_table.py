"""Tests for InSituExperiment build_table(), .table[], import_from_table(), and view.table[]."""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment, TableAccessor

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_insitudata(n_cells=10, n_genes=5, seed=0, gene_names=None, cell_prefix="cell"):
    """Create a minimal InSituData with a small AnnData table."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(
        index=pd.Index([f"{cell_prefix}_{i}" for i in range(n_cells)])
    )
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_genes)]
    var = pd.DataFrame(index=pd.Index(gene_names))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100

    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="slide1", sample_id="s1",
        method_name="t", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5,
                                shared_genes=True):
    """Build a small InSituExperiment with a tmp_path set."""
    exp = InSituExperiment()

    for i in range(n_samples):
        if shared_genes:
            gene_names = [f"gene_{j}" for j in range(n_genes)]
        else:
            # Different gene sets: first half shared, second half unique per sample
            shared = [f"gene_{j}" for j in range(n_genes // 2)]
            unique = [f"sample{i}_gene_{j}" for j in range(n_genes - n_genes // 2)]
            gene_names = shared + unique

        xd = _make_insitudata(
            n_cells=n_cells, n_genes=n_genes,
            gene_names=gene_names,
            seed=i, cell_prefix=f"s{i}cell",
        )
        exp._data.append(xd)

    # Build metadata with known, predictable uids
    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(n_samples)],
        "slide_id": ["slide1"] * n_samples,
        "sample_id": [f"s{i}" for i in range(n_samples)],
    })

    exp._path = tmp_path
    return exp


# ── Phase 1 regression: to_anndata / import_from_anndata unchanged ─────────────

class TestToAnnDataUnchanged:
    def test_returns_anndata(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        adata = exp.to_anndata()
        assert isinstance(adata, AnnData)

    def test_shape(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        adata = exp.to_anndata()
        # 2 samples × 10 cells = 20 obs, 5 shared genes
        assert adata.n_obs == 20
        assert adata.n_vars == 5

    def test_label_col_in_obs(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        adata = exp.to_anndata()
        assert "uid" in adata.obs.columns

    def test_label_col_first(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        adata = exp.to_anndata()
        assert adata.obs.columns[0] == "uid"

    def test_obs_names_unique(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        adata = exp.to_anndata(make_obs_names_unique=True)
        assert adata.obs_names.is_unique

    def test_invalid_label_col_raises(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        with pytest.raises(ValueError, match="not found in metadata"):
            exp.to_anndata(label_col="nonexistent_col")


class TestImportFromAnnDataUnchanged:
    def test_roundtrip_obs_column(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        adata = exp.to_anndata()
        adata.obs["cluster"] = "A"

        exp.import_from_anndata(
            adata=adata,
            uid_column="uid",
            uid_column_adata="uid",
            obs_columns_to_transfer=["cluster"],
        )

        for _, xd in exp.iterdata():
            assert "cluster" in xd.cells.table.obs.columns

    def test_raises_when_both_none(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        adata = exp.to_anndata()
        with pytest.raises(ValueError, match="(?i)at least one"):
            exp.import_from_anndata(
                adata=adata,
                uid_column="uid",
                uid_column_adata="uid",
            )


class TestFullTableDefaults:
    """Default build_table()/to_anndata() retain obs + experiment metadata."""

    def test_metadata_columns_added_by_default(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        adata = exp.to_anndata()
        assert "slide_id" in adata.obs.columns
        assert "sample_id" in adata.obs.columns
        # label_col present exactly once (added by the concat, not duplicated)
        assert list(adata.obs.columns).count("uid") == 1

    def test_per_cell_obs_retained_by_default(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        for xd in exp._data:
            xd.cells.table.obs["qc_flag"] = "ok"
        adata = exp.to_anndata()
        assert "qc_flag" in adata.obs.columns

    def test_metadata_obs_collision_gets_meta_suffix(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        # per-cell obs column name collides with metadata column "sample_id"
        for i, xd in enumerate(exp._data):
            xd.cells.table.obs["sample_id"] = f"percell_{i}"
        adata = exp.to_anndata()
        # per-cell column preserved with its values; metadata stored under _meta
        assert "sample_id" in adata.obs.columns
        assert "sample_id_meta" in adata.obs.columns
        assert adata.obs["sample_id"].astype(str).str.startswith("percell_").all()

    def test_build_table_persists_metadata_columns(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        tbl = exp.table["main"]
        assert "slide_id" in tbl.obs.columns
        assert "sample_id" in tbl.obs.columns


# ── Phase 2: build_table() + .table[] ─────────────────────────────────────────

class TestBuildTableBasic:
    def test_zarr_created(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        assert (tmp_path / "tables" / "main.zarr").exists()

    def test_table_returns_anndata(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        assert isinstance(exp.table, TableAccessor)
        tbl = exp.table["main"]
        assert isinstance(tbl, AnnData)

    def test_shape(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        exp.build_table()
        tbl = exp.table["main"]
        assert tbl.n_obs == 20
        assert tbl.n_vars == 5

    def test_obs_names_unique(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table(make_obs_names_unique=True)
        tbl = exp.table["main"]
        assert tbl.obs_names.is_unique

    def test_obs_names_pattern(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=3)
        exp.build_table(make_obs_names_unique=True)
        tbl = exp.table["main"]
        # Each obs name should contain a "-" separator from the prefix
        for name in tbl.obs_names:
            assert "-" in str(name)

    def test_label_col_in_obs(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        tbl = exp.table["main"]
        assert "uid" in tbl.obs.columns

    def test_dataset_name_column_values(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2)
        exp.build_table()
        tbl = exp.table["main"]
        sample_ids = set(np.unique(np.asarray(tbl.obs["uid"])))
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids


class TestBuildTableJoin:
    def test_inner_result_from_asymmetric_panels(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        exp.build_table()
        tbl = exp.table["main"]
        # inner-over-all = 3 shared genes (n_genes // 2 = 3)
        assert tbl.n_vars == 3

    def test_min_shared_genes_warning(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        with pytest.warns(UserWarning, match="shared"):
            exp.build_table(min_shared_genes=10)


class TestBuildTableOverwrite:
    def test_overwrite_false_raises(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        with pytest.raises(FileExistsError):
            exp.build_table(overwrite=False)

    def test_overwrite_true_succeeds(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        exp.build_table(overwrite=True)  # Should not raise
        assert (tmp_path / "tables" / "main.zarr").exists()


class TestBuildTableNoPath:
    def test_raises_without_path(self):
        exp = InSituExperiment()
        xd = _make_insitudata()
        exp._data.append(xd)
        exp._metadata = pd.DataFrame({
            "uid": ["s0"],
            "slide_id": ["slide1"],
            "sample_id": ["s0"],
        })
        with pytest.raises(ValueError, match="no save path"):
            exp.build_table()


class TestTableAccessorNoBuild:
    def test_accessor_returned_before_build(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        assert isinstance(exp.table, TableAccessor)
        assert exp.table.keys() == []

    def test_getitem_explicit_layer_not_built_warns_and_returns_none(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        with pytest.warns(UserWarning, match="build_table"):
            result = exp.table["main"]
        assert result is None

    def test_getitem_none_raises_keyerror(self, tmp_path):
        # Before any build: KeyError points at build_table.
        exp = _make_experiment_with_path(tmp_path)
        with pytest.raises(KeyError, match="build_table"):
            _ = exp.table[None]
        # After build: KeyError lists the available layer(s).
        exp.build_table()
        with pytest.raises(KeyError, match="main"):
            _ = exp.table[None]


# ── Multi-layer tables ─────────────────────────────────────────────────────────

def _add_proseg_layer(xd, n_cells=8, seed=99):
    """Add a 'proseg' CellData layer to an InSituData."""
    rng = np.random.default_rng(seed)
    n_genes = xd.cells.table.n_vars
    gene_names = list(xd.cells.table.var_names)
    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"proseg_{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index(gene_names))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    cd = CellData(table=table, boundaries=None)
    xd.cells.add_celldata(cd=cd, key="proseg", is_main=False)


def _make_multilayer_experiment(tmp_path, n_samples=2, n_cells_main=10,
                                 n_cells_proseg=8, n_genes=5):
    """Experiment where each sample has both 'main' and 'proseg' cell layers."""
    exp = _make_experiment_with_path(
        tmp_path, n_samples=n_samples, n_cells=n_cells_main, n_genes=n_genes
    )
    for i, xd in enumerate(exp._data):
        _add_proseg_layer(xd, n_cells=n_cells_proseg, seed=100 + i)
    return exp


class TestBuildTableMultiLayer:
    def test_two_zarrs_created(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        assert (tmp_path / "tables" / "main.zarr").exists()
        assert (tmp_path / "tables" / "proseg.zarr").exists()

    def test_table_explicit_layer(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path, n_samples=2,
                                          n_cells_main=10, n_cells_proseg=8)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        tbl_main = exp.table["main"]
        tbl_proseg = exp.table["proseg"]
        assert tbl_main.n_obs == 20   # 2 samples × 10 cells
        assert tbl_proseg.n_obs == 16  # 2 samples × 8 cells

    def test_table_none_raises_keyerror_single_layer(self, tmp_path):
        """table[None] raises KeyError even when exactly one layer is built."""
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        with pytest.raises(KeyError, match="explicit layer"):
            _ = exp.table[None]

    def test_table_none_raises_keyerror_multi_layer(self, tmp_path):
        """table[None] raises KeyError listing all built layers."""
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        with pytest.raises(KeyError) as exc:
            _ = exp.table[None]
        assert "main" in str(exc.value) and "proseg" in str(exc.value)

    def test_table_keys(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        assert set(exp.table.keys()) == {"main", "proseg"}

    def test_build_params_embedded_per_layer(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        assert not (tmp_path / "tables" / "main.json").exists()
        assert not (tmp_path / "tables" / "proseg.json").exists()
        assert exp._read_build_params("main")["cells_layer"] == "main"
        assert exp._read_build_params("proseg")["cells_layer"] == "proseg"

    def test_overwrite_one_layer_leaves_other(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        exp.build_table(cells_layer="main", overwrite=True)
        assert (tmp_path / "tables" / "proseg.zarr").exists()

    def test_import_from_table_layer_specific(self, tmp_path):
        """import_from_table reads from the correct layer zarr."""
        import anndata as ad
        exp = _make_multilayer_experiment(tmp_path, n_cells_main=10)
        exp.build_table(cells_layer="main")

        zarr_path = tmp_path / "tables" / "main.zarr"
        tbl = ad.read_zarr(zarr_path)
        tbl.obs["cluster"] = "X"
        tbl.write_zarr(zarr_path)

        exp.import_from_table(obs_columns=["cluster"], cells_layer="main")
        for _, xd in exp.iterdata():
            assert "cluster" in xd.cells["main"].table.obs.columns


# ── Phase 3: import_from_table() ───────────────────────────────────────────────

class TestImportFromTable:
    def test_roundtrip_obs_column(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        exp.build_table()

        # Load the table, add a column, write it back
        import anndata as ad
        zarr_path = tmp_path / "tables" / "main.zarr"
        tbl = ad.read_zarr(zarr_path)
        tbl.obs["cluster"] = "X"
        tbl.write_zarr(zarr_path)

        exp.import_from_table(obs_columns=["cluster"])

        for _, xd in exp.iterdata():
            assert "cluster" in xd.cells.table.obs.columns
            assert (xd.cells.table.obs["cluster"] == "X").all()

    def test_raises_without_table(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        with pytest.raises(ValueError, match="build_table"):
            exp.import_from_table(obs_columns=["cluster"])

    def test_raises_when_both_none(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        with pytest.raises(ValueError, match="(?i)at least one"):
            exp.import_from_table()


# ── Phase 3: InSituExperimentView.table[] ─────────────────────────────────────

class TestViewTable:
    def test_view_table_filters_samples(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=3, n_cells=10, n_genes=5)
        exp.build_table()

        # Create a view with only the first 2 samples
        view = exp._subset(slice(0, 2), as_view=True)
        tbl = view.table["main"]

        assert isinstance(tbl, AnnData)
        # Only 2 samples × 10 cells = 20 rows
        assert tbl.n_obs == 20
        sample_ids = set(np.unique(np.asarray(tbl.obs["uid"])))
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids
        assert "sample_2" not in sample_ids

    def test_view_table_no_parent_build_warns(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)

        view = exp._subset(slice(0, 1), as_view=True)
        with pytest.warns(UserWarning, match="build_table"):
            result = view.table["main"]
        assert result is None

    def test_view_inherits_path(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)
        view = exp._subset(slice(0, 1), as_view=True)
        assert view.path == exp.path


# ── concat_on_disk ─────────────────────────────────────────────────────────────

def _make_saved_experiment(tmp_path, n_samples=2, n_cells=10, n_genes=5):
    """Build an experiment whose datasets are physically saved to disk.

    Creates the minimal directory structure expected by
    ``_resolve_per_sample_h5ad_paths``:

        {sample_dir}/cells/{timestamp_uid}/
            .multicelldata
            main/
                table.h5ad
    """
    # Fixed timestamp dir name that parses correctly by sort_paths_by_datetime
    TIMESTAMP_DIR = "260101-000000000000-a1b2c3d4"

    exp = InSituExperiment()
    exp._path = tmp_path

    for i in range(n_samples):
        xd = _make_insitudata(
            n_cells=n_cells, n_genes=n_genes,
            seed=i, cell_prefix=f"s{i}cell",
        )
        sample_dir = tmp_path / f"sample_{i}"
        sample_dir.mkdir()

        # Write h5ad inside the expected cells subdirectory
        cells_ts_dir = sample_dir / "cells" / TIMESTAMP_DIR
        main_dir = cells_ts_dir / "main"
        main_dir.mkdir(parents=True)

        # Write the h5ad
        xd.cells.table.write_h5ad(main_dir / "table.h5ad")

        # Write .multicelldata JSON
        multicelldata_meta = {
            "key_main": "main",
            "all_keys": ["main"],
            "version": "test",
        }
        (cells_ts_dir / ".multicelldata").write_text(json.dumps(multicelldata_meta))

        xd._path = sample_dir
        exp._data.append(xd)

    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(n_samples)],
        "slide_id": ["slide1"] * n_samples,
        "sample_id": [f"s{i}" for i in range(n_samples)],
    })

    return exp


class TestConcatOnDisk:
    def test_zarr_created(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")
        assert (tmp_path / "tables" / "main.zarr").exists()

    def test_shape(self, tmp_path):
        exp = _make_saved_experiment(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        exp.build_table(method="concat_on_disk")
        tbl = exp.table["main"]
        assert tbl.n_obs == 20
        assert tbl.n_vars == 5

    def test_label_col_in_obs(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")
        tbl = exp.table["main"]
        assert "uid" in tbl.obs.columns

    def test_build_params_embedded_in_zarr(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")
        assert not (tmp_path / "tables" / "main.json").exists()
        params = exp._read_build_params("main")
        assert params["label_col"] == "uid"
        assert params["method"] == "concat_on_disk"
        assert params["cells_layer"] == "main"

    def test_in_memory_params_embedded_in_zarr(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table(method="in_memory")
        assert not (tmp_path / "tables" / "main.json").exists()
        params = exp._read_build_params("main")
        assert params["method"] == "in_memory"
        assert params["cells_layer"] == "main"

    def test_unsupported_filter_raises(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        with pytest.raises(ValueError, match="does not support"):
            exp.build_table(method="concat_on_disk", obs_keys=["some_col"])

    def test_unsupported_metadata_keys_raises(self, tmp_path):
        # "all" is the accepted default (concat_on_disk keeps everything it can),
        # but an explicit metadata request cannot be honored by the streaming
        # path and must still raise.
        exp = _make_saved_experiment(tmp_path)
        with pytest.raises(ValueError, match="does not support"):
            exp.build_table(method="concat_on_disk", metadata_keys=["slide_id"])

    def test_no_saved_path_raises(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        # Unset xd paths so resolve fails
        for xd in exp._data:
            xd._path = None
        with pytest.raises(ValueError, match="no save path"):
            exp.build_table(method="concat_on_disk")

    def test_invalid_method_raises(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        with pytest.raises(ValueError, match="Unknown method"):
            exp.build_table(method="unknown_method")

    def test_overwrite(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")
        exp.build_table(method="concat_on_disk", overwrite=True)  # Should not raise
        assert (tmp_path / "tables" / "main.zarr").exists()

    def test_asymmetric_panels_inner_result(self, tmp_path):
        """With asymmetric gene panels, exp.table returns the inner-over-all gene set."""
        import anndata as ad
        exp = _make_saved_experiment(tmp_path, n_samples=2, n_genes=6, n_cells=5)
        # Patch sample 1 to have different genes
        TIMESTAMP_DIR = "260101-000000000000-a1b2c3d4"
        rng = np.random.default_rng(99)
        shared = [f"gene_{j}" for j in range(3)]
        unique = [f"other_gene_{j}" for j in range(3)]
        h5ad_path = tmp_path / "sample_1" / "cells" / TIMESTAMP_DIR / "main" / "table.h5ad"
        X = rng.integers(0, 20, size=(5, 6)).astype(float)
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=pd.Index([f"s1cell_{i}" for i in range(5)])),
            var=pd.DataFrame(index=pd.Index(shared + unique)),
        )
        adata.write_h5ad(h5ad_path)

        exp.build_table(method="concat_on_disk")
        tbl = exp.table["main"]
        # inner-over-all = only 3 shared genes
        assert tbl.n_vars == 3


# ── Presence-record reconstruction helpers ────────────────────────────────────

def _make_asymmetric_experiment(tmp_path):
    """3-sample experiment with asymmetric gene panels.

    s0: {gA, gB, gC}
    s1: {gA, gB, gD}
    s2: {gA, gE, gF}

    full inner = {gA} (1 gene)
    view[0,1] inner = {gA, gB} (2 genes) — the key regression case
    """
    gene_sets = [
        ["gA", "gB", "gC"],
        ["gA", "gB", "gD"],
        ["gA", "gE", "gF"],
    ]
    exp = InSituExperiment()
    for i, genes in enumerate(gene_sets):
        xd = _make_insitudata(
            n_cells=5, n_genes=3, gene_names=genes,
            seed=i, cell_prefix=f"s{i}cell",
        )
        exp._data.append(xd)
    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(3)],
        "slide_id": ["slide1"] * 3,
        "sample_id": [f"s{i}" for i in range(3)],
    })
    exp._path = tmp_path
    return exp


def _make_saved_asymmetric_experiment(tmp_path):
    """3-sample on-disk experiment with asymmetric gene panels (for concat_on_disk)."""
    TIMESTAMP_DIR = "260101-000000000000-a1b2c3d4"
    gene_sets = [
        ["gA", "gB", "gC"],
        ["gA", "gB", "gD"],
        ["gA", "gE", "gF"],
    ]
    exp = InSituExperiment()
    exp._path = tmp_path
    for i, genes in enumerate(gene_sets):
        xd = _make_insitudata(
            n_cells=5, n_genes=3, gene_names=genes,
            seed=i, cell_prefix=f"s{i}cell",
        )
        sample_dir = tmp_path / f"sample_{i}"
        sample_dir.mkdir()
        cells_ts_dir = sample_dir / "cells" / TIMESTAMP_DIR
        main_dir = cells_ts_dir / "main"
        main_dir.mkdir(parents=True)
        xd.cells.table.write_h5ad(main_dir / "table.h5ad")
        multicelldata_meta = {
            "key_main": "main",
            "all_keys": ["main"],
            "version": "test",
        }
        (cells_ts_dir / ".multicelldata").write_text(json.dumps(multicelldata_meta))
        xd._path = sample_dir
        exp._data.append(xd)
    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(3)],
        "slide_id": ["slide1"] * 3,
        "sample_id": [f"s{i}" for i in range(3)],
    })
    return exp


# ── Presence-record reconstruction tests ──────────────────────────────────────

class TestPresenceReconstruction:
    def test_view_table_recovers_subset_shared_genes(self, tmp_path):
        """Core regression: view.table returns inner-over-view, not inner-over-all."""
        exp = _make_asymmetric_experiment(tmp_path)
        exp.build_table()

        # Full inner = only gA (shared by all 3)
        full_tbl = exp.table["main"]
        assert set(full_tbl.var_names) == {"gA"}

        # View over s0+s1: inner = {gA, gB} — previously returned {gA}
        view = exp._subset([0, 1], as_view=True)
        view_tbl = view.table["main"]
        assert set(view_tbl.var_names) == {"gA", "gB"}

    def test_view_table_recovers_correct_values(self, tmp_path):
        """Values in view.table match the per-sample sources (no 0-fills surfaced)."""
        exp = _make_asymmetric_experiment(tmp_path)
        exp.build_table()

        view = exp._subset([0, 1], as_view=True)
        view_tbl = view.table["main"]

        # Trusted reference via in-memory re-concat
        ref = view.to_anndata()

        # var_names should match
        assert set(view_tbl.var_names) == set(ref.var_names)

        # Values should match per uid label (align by uid)
        X_view = np.asarray(view_tbl.X)
        uid_view = np.asarray(view_tbl.obs["uid"], dtype=str)
        X_ref = ref[:, list(view_tbl.var_names)].X
        uid_ref = np.asarray(ref.obs["uid"], dtype=str)

        for gene_idx, gene in enumerate(view_tbl.var_names):
            ref_gene_idx = list(ref.var_names).index(gene)
            for uid in ["sample_0", "sample_1"]:
                view_vals = sorted(X_view[uid_view == uid, gene_idx].tolist())
                ref_vals = sorted(X_ref[uid_ref == uid, ref_gene_idx].tolist())
                assert view_vals == ref_vals, f"Mismatch for gene {gene}, uid {uid}"

    def test_table_inner_has_no_nan_no_fill(self, tmp_path):
        """Neither base nor view table contains NaN or fabricated fill values."""
        exp = _make_asymmetric_experiment(tmp_path)
        exp.build_table()

        full_tbl = exp.table["main"]
        X_full = np.asarray(full_tbl.X)
        assert not np.isnan(X_full).any(), "NaN in base table"

        view = exp._subset([0, 1], as_view=True)
        view_tbl = view.table["main"]
        X_view = np.asarray(view_tbl.X)
        assert not np.isnan(X_view).any(), "NaN in view table"

        # All values should be > 0 (the test data is rng integers 0..20, and
        # every returned gene is genuinely measured in every returned sample)
        assert (X_view >= 0).all(), "Negative values in view table"

    def test_uns_records_presence_and_format_version(self, tmp_path):
        """The built zarr stores format version, presence labels, and presence matrix."""
        import zarr
        exp = _make_asymmetric_experiment(tmp_path)
        exp.build_table()

        z = zarr.open_group(str(tmp_path / "tables" / "main.zarr"), mode="r")
        uns = z["uns"]

        assert int(uns["_insitupy_table_format_version"][()]) == 2

        labels = np.array([str(l) for l in uns["_insitupy_presence_labels"][:]])
        assert set(labels) == {"sample_0", "sample_1", "sample_2"}

        presence = np.asarray(uns["_insitupy_gene_presence"][:], dtype=bool)
        n_datasets, n_vars = presence.shape
        assert n_datasets == 3
        assert n_vars == 6  # union of {gA, gB, gC, gD, gE, gF}

    def test_table_status_membership(self, tmp_path):
        """_table_status returns the correct membership string."""
        exp = _make_experiment_with_path(tmp_path, n_samples=2)
        exp.build_table()

        assert exp._table_status("main") == "matches current samples"
        assert "matches current samples" in repr(exp)

        # Drop a dataset to make metadata diverge from built set
        exp._data = exp._data[:1]
        exp._metadata = exp._metadata.iloc[:1].reset_index(drop=True)

        assert exp._table_status("main") == "samples changed — rebuild"
        assert "samples changed" in repr(exp)

    def test_view_build_table_raises_with_pointer(self, tmp_path):
        """view.build_table() raises NotImplementedError pointing to view.table."""
        exp = _make_experiment_with_path(tmp_path)
        view = exp._subset(slice(0, 1), as_view=True)
        with pytest.raises(NotImplementedError, match=r"view\.table"):
            view.build_table()

    def test_legacy_table_without_presence_still_loads(self, tmp_path):
        """Tables built by old code (no presence uns) still load via base and view."""
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)

        # Write a legacy zarr manually: inner join result, no presence entries
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir(parents=True)
        zarr_path = tables_dir / "main.zarr"

        # Build an inner-join AnnData (what old code would have written)
        legacy_adata = exp.to_anndata()
        legacy_adata.write_zarr(zarr_path)
        (tables_dir / "main.json").write_text(json.dumps({
            "label_col": "uid", "method": "in_memory", "cells_layer": "main"
        }))

        # Base access should return the stored table as-is
        result = exp.table["main"]
        assert result is not None
        assert result.n_vars == 3

        # View access should use the legacy row-filter-only path
        view = exp._subset(slice(0, 1), as_view=True)
        view_result = view.table["main"]
        assert view_result is not None
        assert view_result.n_obs == 5  # 1 sample × 5 cells

    def test_concat_on_disk_view_recovers_subset_genes(self, tmp_path):
        """concat_on_disk path also writes presence uns and view recovers subset genes."""
        import zarr
        exp = _make_saved_asymmetric_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")

        # Verify presence uns was written
        z = zarr.open_group(str(tmp_path / "tables" / "main.zarr"), mode="r")
        assert "_insitupy_table_format_version" in z["uns"]

        # Full inner = 1 gene
        full_tbl = exp.table["main"]
        assert set(full_tbl.var_names) == {"gA"}

        # View over s0+s1: inner = {gA, gB}
        view = exp._subset([0, 1], as_view=True)
        view_tbl = view.table["main"]
        assert set(view_tbl.var_names) == {"gA", "gB"}

    def test_versioned_table_missing_presence_raises(self, tmp_path):
        """A versioned store missing its presence arrays must raise — not silently
        fall back to returning the raw union table (which holds fill values)."""
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir(parents=True)
        zarr_path = tables_dir / "main.zarr"

        # Write a store that carries the format-version marker but NO presence
        # arrays, simulating a partial/interrupted or version-incompatible write.
        adata = exp.to_anndata()
        adata.uns["_insitupy_table_format_version"] = 2
        adata.write_zarr(zarr_path)
        (tables_dir / "main.json").write_text(json.dumps({
            "label_col": "uid", "method": "in_memory", "cells_layer": "main"
        }))

        with pytest.raises(RuntimeError, match="presence"):
            _ = exp.table["main"]

    def test_failed_build_preserves_previous_table(self, tmp_path, monkeypatch):
        """An interrupted rebuild must not destroy the previously built table."""
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)
        exp.build_table()
        n_vars_before = exp.table["main"].n_vars

        # Make the next build fail partway through (before the atomic swap).
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated build failure")
        monkeypatch.setattr(type(exp), "_collect_gene_presence", _boom)

        with pytest.raises(RuntimeError, match="simulated"):
            exp.build_table(overwrite=True)

        # The previous table survives intact, with no leftover staging/backup dirs.
        tables_dir = tmp_path / "tables"
        assert (tables_dir / "main.zarr").exists()
        assert not (tables_dir / "main.zarr.__ispy_tmp__").exists()
        assert not (tables_dir / "main.zarr.__ispy_bak__").exists()
        assert exp.table["main"].n_vars == n_vars_before

    def test_empty_view_table_returns_zero_rows(self, tmp_path):
        """An empty view (0 samples) returns a 0-row AnnData, NOT the full union, and
        does not emit the 'No view samples' warning."""
        import warnings as _warnings
        exp = _make_asymmetric_experiment(tmp_path)   # union = {gA..gF} = 6 vars
        exp.build_table()
        full = exp.table["main"]

        empty_view = exp._subset(slice(0, 0), as_view=True)
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            tbl = empty_view.table["main"]
        assert tbl.n_obs == 0
        assert tbl.n_obs != full.n_obs
        assert not any("No view samples" in str(w.message) for w in rec)
        # _reconstruct([]) keeps the union var axis (vacuous-truth all-True var mask).
        assert tbl.n_vars == 6

    def test_view_table_samples_not_in_built_table_warns_and_empty(self, tmp_path):
        """A view selecting only samples absent from the built table warns AND returns 0 rows."""
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)
        exp.build_table()  # built labels = {sample_0, sample_1}

        # Append a 3rd sample WITHOUT rebuilding, then view only that sample.
        extra = _make_insitudata(n_cells=5, n_genes=3, seed=42, cell_prefix="s2cell")
        exp._data.append(extra)
        exp._metadata = pd.concat(
            [exp._metadata,
             pd.DataFrame({"uid": ["sample_2"], "slide_id": ["slide1"], "sample_id": ["s2"]})],
            ignore_index=True,
        )
        view = exp._subset([2], as_view=True)   # uid == sample_2, not in built labels
        with pytest.warns(UserWarning, match="No view samples"):
            tbl = view.table["main"]
        assert tbl.n_obs == 0

    def test_empty_view_legacy_table_returns_zero_rows(self, tmp_path):
        """Legacy (no-presence) stores also return 0 rows for an empty view."""
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir(parents=True)
        exp.to_anndata().write_zarr(tables_dir / "main.zarr")
        (tables_dir / "main.json").write_text(json.dumps(
            {"label_col": "uid", "method": "in_memory", "cells_layer": "main"}
        ))

        empty_view = exp._subset(slice(0, 0), as_view=True)
        tbl = empty_view.table["main"]
        assert tbl is not None
        assert tbl.n_obs == 0

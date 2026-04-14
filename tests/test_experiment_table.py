"""Tests for InSituExperiment build_table(), .table[], import_from_table(), and view.table[]."""

import json

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
        tbl = exp.table[None]
        assert isinstance(tbl, AnnData)

    def test_shape(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        exp.build_table()
        tbl = exp.table[None]
        assert tbl.n_obs == 20
        assert tbl.n_vars == 5

    def test_obs_names_unique(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table(make_obs_names_unique=True)
        tbl = exp.table[None]
        assert tbl.obs_names.is_unique

    def test_obs_names_pattern(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=3)
        exp.build_table(make_obs_names_unique=True)
        tbl = exp.table[None]
        # Each obs name should contain a "-" separator from the prefix
        for name in tbl.obs_names:
            assert "-" in str(name)

    def test_label_col_in_obs(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        tbl = exp.table[None]
        assert "uid" in tbl.obs.columns

    def test_dataset_name_column_values(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2)
        exp.build_table()
        tbl = exp.table[None]
        sample_ids = set(np.unique(np.asarray(tbl.obs["uid"])))
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids


class TestBuildTableJoin:
    def test_join_inner(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        exp.build_table(join="inner")
        tbl = exp.table[None]
        # Only the 3 shared genes (n_genes // 2 = 3)
        assert tbl.n_vars == 3

    def test_join_outer(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        exp.build_table(join="outer")
        tbl = exp.table[None]
        # All genes: 3 shared + 3 unique per sample = 9
        assert tbl.n_vars == 9

    def test_min_shared_genes_warning(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        with pytest.warns(UserWarning, match="shared genes"):
            exp.build_table(join="inner", min_shared_genes=10)


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

    def test_getitem_none_warns_and_returns_none(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        with pytest.warns(UserWarning, match="build_table"):
            result = exp.table[None]
        assert result is None


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

    def test_table_auto_resolve_single(self, tmp_path):
        """When only one layer is built, table[None] auto-selects it."""
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        tbl = exp.table[None]
        assert isinstance(tbl, AnnData)

    def test_table_ambiguous_raises_warning(self, tmp_path):
        """When two layers exist, table[None] emits a warning and returns None."""
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        with pytest.warns(UserWarning, match="cells_layer="):
            result = exp.table[None]
        assert result is None

    def test_table_keys(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        assert set(exp.table.keys()) == {"main", "proseg"}

    def test_sidecar_per_layer(self, tmp_path):
        exp = _make_multilayer_experiment(tmp_path)
        exp.build_table(cells_layer="main")
        exp.build_table(cells_layer="proseg")
        main_params = json.loads((tmp_path / "tables" / "main.json").read_text())
        proseg_params = json.loads((tmp_path / "tables" / "proseg.json").read_text())
        assert main_params["cells_layer"] == "main"
        assert proseg_params["cells_layer"] == "proseg"

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
        tbl = view.table[None]

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
            result = view.table[None]
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
        tbl = exp.table[None]
        assert tbl.n_obs == 20
        assert tbl.n_vars == 5

    def test_label_col_in_obs(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")
        tbl = exp.table[None]
        assert "uid" in tbl.obs.columns

    def test_build_params_sidecar_written(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        exp.build_table(method="concat_on_disk")
        params_path = tmp_path / "tables" / "main.json"
        assert params_path.exists()
        params = json.loads(params_path.read_text())
        assert params["label_col"] == "uid"
        assert params["method"] == "concat_on_disk"
        assert params["cells_layer"] == "main"

    def test_in_memory_also_writes_sidecar(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table(method="in_memory")
        params_path = tmp_path / "tables" / "main.json"
        assert params_path.exists()
        params = json.loads(params_path.read_text())
        assert params["method"] == "in_memory"
        assert params["cells_layer"] == "main"

    def test_unsupported_filter_raises(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        with pytest.raises(ValueError, match="does not support"):
            exp.build_table(method="concat_on_disk", obs_keys=["some_col"])

    def test_unsupported_metadata_keys_raises(self, tmp_path):
        exp = _make_saved_experiment(tmp_path)
        with pytest.raises(ValueError, match="does not support"):
            exp.build_table(method="concat_on_disk", metadata_keys="all")

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

    def test_join_inner(self, tmp_path):
        """Inner join retains only shared genes."""
        exp = _make_saved_experiment(tmp_path, n_samples=2, n_genes=6, n_cells=5)
        # Give sample 1 different genes by patching its h5ad with different var names
        TIMESTAMP_DIR = "260101-000000000000-a1b2c3d4"
        rng = np.random.default_rng(99)
        import anndata as ad
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

        exp.build_table(method="concat_on_disk", join="inner")
        tbl = exp.table[None]
        assert tbl.n_vars == 3  # only 3 shared genes

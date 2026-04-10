"""Tests for InSituExperiment build_table(), .table, import_from_table(), and view.table."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment


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


# ── Phase 2: build_table() + .table ────────────────────────────────────────────

class TestBuildTableBasic:
    def test_zarr_created(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        assert (tmp_path / "tables" / "concat.zarr").exists()

    def test_table_returns_anndata(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        tbl = exp.table
        assert isinstance(tbl, AnnData)

    def test_shape(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        exp.build_table()
        tbl = exp.table
        assert tbl.n_obs == 20
        assert tbl.n_vars == 5

    def test_obs_names_unique(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table(make_obs_names_unique=True)
        tbl = exp.table
        assert tbl.obs_names.is_unique

    def test_obs_names_pattern(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=3)
        exp.build_table(make_obs_names_unique=True)
        tbl = exp.table
        # Each obs name should contain a "-" separator from the prefix
        for name in tbl.obs_names:
            assert "-" in str(name)

    def test_label_col_in_obs(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        exp.build_table()
        tbl = exp.table
        assert "uid" in tbl.obs.columns

    def test_dataset_name_column_values(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2)
        exp.build_table()
        tbl = exp.table
        sample_ids = set(tbl.obs["uid"].unique())
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids


class TestBuildTableJoin:
    def test_join_inner(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        exp.build_table(join="inner")
        tbl = exp.table
        # Only the 3 shared genes (n_genes // 2 = 3)
        assert tbl.n_vars == 3

    def test_join_outer(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_genes=6,
                                         shared_genes=False)
        exp.build_table(join="outer")
        tbl = exp.table
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
        assert (tmp_path / "tables" / "concat.zarr").exists()


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


class TestTablePropertyNoBuild:
    def test_returns_none_and_warns(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path)
        with pytest.warns(UserWarning, match="build_table"):
            result = exp.table
        assert result is None


# ── Phase 3: import_from_table() ───────────────────────────────────────────────

class TestImportFromTable:
    def test_roundtrip_obs_column(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=10, n_genes=5)
        exp.build_table()

        # Load the table, add a column, write it back
        import anndata as ad
        zarr_path = tmp_path / "tables" / "concat.zarr"
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


# ── Phase 3: InSituExperimentView.table ────────────────────────────────────────

class TestViewTable:
    def test_view_table_filters_samples(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=3, n_cells=10, n_genes=5)
        exp.build_table()

        # Create a view with only the first 2 samples
        view = exp._subset(slice(0, 2), as_view=True)
        tbl = view.table

        assert isinstance(tbl, AnnData)
        # Only 2 samples × 10 cells = 20 rows
        assert tbl.n_obs == 20
        sample_ids = set(tbl.obs["uid"].unique())
        assert "sample_0" in sample_ids
        assert "sample_1" in sample_ids
        assert "sample_2" not in sample_ids

    def test_view_table_no_parent_build_warns(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)

        view = exp._subset(slice(0, 1), as_view=True)
        with pytest.warns(UserWarning, match="build_table"):
            result = view.table
        assert result is None

    def test_view_inherits_path(self, tmp_path):
        exp = _make_experiment_with_path(tmp_path, n_samples=2, n_cells=5, n_genes=3)
        view = exp._subset(slice(0, 1), as_view=True)
        assert view.path == exp.path

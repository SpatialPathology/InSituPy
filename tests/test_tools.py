"""Tests for tools: tl.dge, calc_distance_of_cells_from,
calculate_gex_diff_to_neighbors, pseudobulk_dge."""

import contextlib
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import issparse
from shapely.geometry import Polygon

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.results import DiffExprResults
from insitupy.preprocessing import normalize_and_transform
from insitupy.tools.dge import dge
from insitupy.tools.distance import calc_distance_of_cells_from
from insitupy.tools.neighbors import calculate_gex_diff_to_neighbors
from insitupy.tools.pseudobulk import pseudobulk_dge
from insitupy.utils.dge import create_deg_dataframe
from insitupy.utils.go import get_up_down_genes

# ── Helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _capture_insitupy_logs(caplog, level, logger_name="insitupy.tools.dge"):
    """caplog's handler is attached at the root logger, but `insitupy` sets
    `propagate=False` (see insitupy/_logging.py), so records from `insitupy.*`
    loggers never reach it. Attach the handler directly to the source logger.
    """
    logger = logging.getLogger(logger_name)
    caplog.set_level(level, logger=logger_name)
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)

def _make_insitudata_with_celltypes(n_per_type=15, n_genes=10, seed=0):
    """InSituData with two cell types ('A' and 'B') with distinct expression."""
    rng = np.random.default_rng(seed)
    n = n_per_type * 2

    # Cell type A: high expression of first 5 genes; B: high expression of last 5
    X_A = np.zeros((n_per_type, n_genes))
    X_A[:, :5] = rng.integers(20, 50, size=(n_per_type, 5))
    X_A[:, 5:] = rng.integers(0, 5, size=(n_per_type, 5))

    X_B = np.zeros((n_per_type, n_genes))
    X_B[:, :5] = rng.integers(0, 5, size=(n_per_type, 5))
    X_B[:, 5:] = rng.integers(20, 50, size=(n_per_type, 5))

    X = np.vstack([X_A, X_B]).astype(float)

    obs = pd.DataFrame(
        {"celltype": ["A"] * n_per_type + ["B"] * n_per_type},
        index=pd.Index([f"cell_{i}" for i in range(n)]),
    )
    var = pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n, 2)) * 100

    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="t", sample_id="s",
        method_name="t", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_adata_with_spatial(n_cells=20, n_genes=8, seed=0):
    """Minimal AnnData with spatial coordinates and integer count matrix."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 30, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    return table


# ── tl.dge ────────────────────────────────────────────────────────────────────

class TestDge:
    # NOTE: dge() raises by default (assert_log1p=True) on marker-less raw integer counts
    # (AC-B2/AC-2) - the fixture builds raw counts, so every test here normalizes (log1p)
    # first. This is the correct altitude: DGE on raw counts is exactly the wrong-science
    # case the guard targets. See TestDgeLog1pGuard for the guard's own regression tests.

    def test_returns_diffexprresults(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            method="t-test",
            verbose=False,
        )
        assert isinstance(result, DiffExprResults)

    def test_main_is_dataframe_with_expected_columns(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            method="t-test",
            verbose=False,
        )
        for col in ("log2foldchange", "pvalue", "padj", "scores"):
            assert col in result.main.columns, f"Missing column: {col}"

    def test_main_index_is_gene_names(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            verbose=False,
        )
        gene_names = list(xd.cells.table.var_names)
        assert set(result.main.index) == set(gene_names)

    def test_explicit_ref_cell_type_tuple(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple=("celltype", "B"),
            verbose=False,
        )
        assert isinstance(result, DiffExprResults)
        assert result.main is not None


# ── tl.dge: create_deg_dataframe column correctness (AC-B1) ────────────────────

class TestCreateDegDataframe:
    def test_padj_is_bh_adjusted_and_excludes_raw_only_significant_gene(self):
        """Regression for AC-B1: `padj` must hold scanpy's `pvals_adj` (BH-corrected),
        not the raw `pvals`, and a gene that is only raw-significant (not BH-significant)
        must not be plotted/counted as significant downstream (get_up_down_genes)."""
        genes = ["g0", "g1", "g2"]
        logfc = np.array([2.0, -2.0, 0.1])
        pvals = np.array([0.01, 0.02, 0.5])
        # g0: raw p < 0.05 but BH-adjusted >= 0.05 - the "plotted-significant-when-it-isn't" case
        pvals_adj = np.array([0.08, 0.03, 0.6])
        scores = np.array([3.0, -3.0, 0.2])

        adata = AnnData(
            X=np.zeros((2, len(genes))),
            var=pd.DataFrame(index=pd.Index(genes)),
        )
        adata.uns["rank_genes_groups"] = {
            "names": np.array(genes, dtype=[("DATA", object)]),
            "logfoldchanges": np.array(logfc, dtype=[("DATA", float)]),
            "pvals": np.array(pvals, dtype=[("DATA", float)]),
            "pvals_adj": np.array(pvals_adj, dtype=[("DATA", float)]),
            "scores": np.array(scores, dtype=[("DATA", float)]),
        }

        res_dict = create_deg_dataframe(adata, groups="DATA")
        df = res_dict["DATA"]

        np.testing.assert_array_equal(df["pvalue"].to_numpy(), pvals)
        np.testing.assert_array_equal(df["padj"].to_numpy(), pvals_adj)
        np.testing.assert_allclose(df["neg_log10_pvals"].to_numpy(), -np.log10(pvals_adj))

        # concrete downstream regression: g0 must not appear as "up" under the default
        # pval_col="padj", even though it was raw-significant under the old (buggy) meaning.
        genes_up, genes_down = get_up_down_genes(df.set_index("gene"))
        assert "g0" not in genes_up
        assert "g1" in genes_down


# ── tl.dge: log1p input guard (AC-B2 / AC-2) ────────────────────────────────────

class TestDgeLog1pGuard:
    def test_markerless_raw_integer_counts_raises(self):
        xd = _make_insitudata_with_celltypes()
        with pytest.raises(ValueError, match="raw integer counts"):
            dge(
                target=xd,
                target_cell_type_tuple=("celltype", "A"),
                ref_cell_type_tuple="rest",
                verbose=False,
            )

    def test_markerless_raw_integer_counts_assert_log1p_false_proceeds(self):
        xd = _make_insitudata_with_celltypes()
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            assert_log1p=False,
            verbose=False,
        )
        assert isinstance(result, DiffExprResults)

    def test_sqrt_transformed_raises(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="sqrt")
        with pytest.raises(ValueError, match="sqrt"):
            dge(
                target=xd,
                target_cell_type_tuple=("celltype", "A"),
                ref_cell_type_tuple="rest",
                verbose=False,
            )

    def test_scaled_raises(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p", scale=True)
        with pytest.raises(ValueError, match="scaled"):
            dge(
                target=xd,
                target_cell_type_tuple=("celltype", "A"),
                ref_cell_type_tuple="rest",
                verbose=False,
            )

    def test_sqrt_transformed_assert_log1p_false_proceeds(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="sqrt")
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            assert_log1p=False,
            verbose=False,
        )
        assert isinstance(result, DiffExprResults)

    def test_log1p_normalized_does_not_raise(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            verbose=False,
        )
        assert isinstance(result, DiffExprResults)

    def test_markerless_non_integer_warns_but_proceeds(self):
        xd = _make_insitudata_with_celltypes()
        # hand-transform floats without going through normalize_and_transform, so no
        # marker is written - a legacy-store / externally-built-AnnData stand-in.
        table = xd.cells.table
        X = table.X.toarray() if issparse(table.X) else np.asarray(table.X)
        table.X = X / 3.14159

        with pytest.warns(UserWarning, match="cannot verify"):
            result = dge(
                target=xd,
                target_cell_type_tuple=("celltype", "A"),
                ref_cell_type_tuple="rest",
                verbose=False,
            )
        assert isinstance(result, DiffExprResults)


# ── tl.dge: ambiguity check restricted to a single object (AC-B3) ──────────────

class TestDgeCrossObjectAmbiguity:
    def test_exclude_ambiguous_does_not_drop_cross_object_id_collisions(self):
        """Two distinct InSituData objects whose cell tables coincidentally share
        obs_names (both built with the same default naming) are different physical
        cells. exclude_ambiguous_assignments=True must not drop them (AC-B3) - unlike
        the same-object case, which still deduplicates genuine duplicates."""
        xd_a = _make_insitudata_with_celltypes(seed=1)
        xd_b = _make_insitudata_with_celltypes(seed=2)
        normalize_and_transform(xd_a, transformation_method="log1p")
        normalize_and_transform(xd_b, transformation_method="log1p")

        # both objects use the same default obs_names ("cell_0".."cell_14" for celltype A) -
        # a coincidental hex-ID-style collision across two distinct slides/objects.
        assert set(xd_a.cells.table.obs_names[:15]) == set(xd_b.cells.table.obs_names[:15])

        result = dge(
            target=xd_a,
            target_cell_type_tuple=("celltype", "A"),
            ref=xd_b,
            ref_cell_type_tuple=("celltype", "A"),
            exclude_ambiguous_assignments=True,
            verbose=False,
        )

        # no cells were dropped despite the fully-overlapping obs_names
        assert result.config.target_cell_number == 15
        assert result.config.ref_cell_number == 15


# ── tl.dge: identical target/reference selection raises (AC-B19) ───────────────

class TestDgeIdenticalSelectionRaises:
    def test_default_tuples_raise_without_exclude_ambiguous(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        with pytest.raises(ValueError, match="identical selection"):
            dge(target=xd, exclude_ambiguous_assignments=False, verbose=False)

    def test_default_tuples_raise_with_exclude_ambiguous(self):
        # previously crashed on an empty object once ambiguous cells were dropped
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")
        with pytest.raises(ValueError, match="identical selection"):
            dge(target=xd, exclude_ambiguous_assignments=True, verbose=False)


# ── tl.dge: NaN rows filtered before rank_genes_groups (section 4f) ────────────

class TestDgeNanFilter:
    def test_nan_rows_removed_and_logged(self, caplog):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")

        table = xd.cells.table
        X = table.X.toarray() if issparse(table.X) else np.array(table.X)
        X[0, 0] = np.nan
        X[1, 1] = np.nan
        table.X = X

        with _capture_insitupy_logs(caplog, logging.WARNING):
            result = dge(
                target=xd,
                target_cell_type_tuple=("celltype", "A"),
                ref_cell_type_tuple="rest",
                verbose=False,
            )

        assert not result.main["log2foldchange"].isna().any()
        assert any("NaN" in rec.getMessage() for rec in caplog.records)

    def test_nan_wiping_out_a_group_raises(self):
        xd = _make_insitudata_with_celltypes()
        normalize_and_transform(xd, transformation_method="log1p")

        table = xd.cells.table
        X = table.X.toarray() if issparse(table.X) else np.array(table.X)
        # celltype A occupies the first 15 rows (see _make_insitudata_with_celltypes)
        X[:15, :] = np.nan
        table.X = X

        with pytest.raises(ValueError, match="empty"):
            dge(
                target=xd,
                target_cell_type_tuple=("celltype", "A"),
                ref_cell_type_tuple="rest",
                verbose=False,
            )


# ── tl.calc_distance_of_cells_from ───────────────────────────────────────────

class TestCalcDistanceOfCellsFrom:
    def test_missing_annotation_tuple_raises(self):
        xd = _make_insitudata_with_celltypes()
        with pytest.raises(ValueError, match="annotation_tuple"):
            calc_distance_of_cells_from(xd, annotation_tuple=None)

    def test_distance_stored_in_obsm(self):
        xd = _make_insitudata_with_celltypes()

        # Add a rectangle annotation to the InSituData
        # parse_geopandas expects an "id" column (used as index)
        rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        gdf = gpd.GeoDataFrame(
            {"id": ["boundary_0"], "name": ["boundary"], "geometry": [rect], "color": ["#ff0000"]}
        )
        xd._annotations.add_data(data=gdf, key="my_annot", scale_factor=1.0)

        calc_distance_of_cells_from(
            xd,
            annotation_tuple=("my_annot", "boundary"),
            key_to_save="boundary_dist",
        )

        table = xd.cells.table
        assert "distance_from" in table.obsm
        assert "boundary_dist" in table.obsm["distance_from"].columns

    def test_distance_values_are_nonnegative(self):
        xd = _make_insitudata_with_celltypes()
        rect = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
        gdf = gpd.GeoDataFrame(
            {"id": ["roi_0"], "name": ["roi"], "geometry": [rect], "color": ["#ff0000"]}
        )
        xd._annotations.add_data(data=gdf, key="rois", scale_factor=1.0)

        calc_distance_of_cells_from(xd, annotation_tuple=("rois", "roi"))

        distances = xd.cells.table.obsm["distance_from"]["roi"]
        assert (distances >= 0).all()


# ── tl.calculate_gex_diff_to_neighbors ───────────────────────────────────────

class TestCalculateGexDiffToNeighbors:
    # NOTE: _make_adata_with_spatial builds raw integer counts with no transformation
    # marker; these tests exercise the function's I/O shape, not DGE scientific
    # correctness, so they opt out of the log1p guard (AC-B2/AC-2) via assert_log1p=False.
    def test_returns_four_tuple(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        result = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", assert_log1p=False, verbose=False
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_first_element_is_dataframe(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        df, A, diffs, qc = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", assert_log1p=False, verbose=False
        )
        assert isinstance(df, pd.DataFrame)

    def test_adjacency_matrix_shape(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        df, A, diffs, qc = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", assert_log1p=False, verbose=False
        )
        n = adata.n_obs
        assert A.shape == (n, n)

    def test_qc_stats_is_dict(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        df, A, diffs, qc = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", assert_log1p=False, verbose=False
        )
        assert isinstance(qc, dict)

    def test_log1p_guard_fires_on_raw_counts(self):
        # Direct coverage of the AC-B2/AC-2 guard on this function (not only via dge):
        # a marker-less raw integer-count matrix must raise unless assert_log1p=False.
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        with pytest.raises(ValueError, match="raw integer counts"):
            calculate_gex_diff_to_neighbors(
                adata, radius=200.0, strategy="mean", verbose=False
            )


# ── tl.pseudobulk_dge ────────────────────────────────────────────────────────

class TestPseudobulkDge:
    def test_missing_condition_column_raises(self):
        pytest.importorskip("pydeseq2")
        pdata = AnnData(
            X=np.ones((4, 5)),
            obs=pd.DataFrame(
                {"celltype": ["A", "A", "B", "B"]},
                index=[f"s{i}" for i in range(4)],
            ),
            var=pd.DataFrame(index=[f"g{j}" for j in range(5)]),
        )
        with pytest.raises(ValueError, match="Condition column"):
            pseudobulk_dge(
                pdata=pdata,
                dge_setup=("condition", "treated", "control"),
                celltype_col="celltype",
                celltype="A",
                plot_qc=False,
                verbose=False,
            )

    def test_missing_target_condition_raises(self):
        pytest.importorskip("pydeseq2")
        pdata = AnnData(
            X=np.ones((4, 5)),
            obs=pd.DataFrame(
                {"condition": ["treated", "treated", "control", "control"],
                 "celltype": ["A", "A", "A", "A"]},
                index=[f"s{i}" for i in range(4)],
            ),
            var=pd.DataFrame(index=[f"g{j}" for j in range(5)]),
        )
        with pytest.raises(ValueError, match="Target condition"):
            pseudobulk_dge(
                pdata=pdata,
                dge_setup=("condition", "missing_cond", "control"),
                celltype_col="celltype",
                celltype="A",
                plot_qc=False,
                verbose=False,
            )

    def test_full_analysis(self):
        pytest.skip("requires real pseudobulk dataset with PyDESeq2")


# ── ShapesData.add_data / InSituData convenience conversions ──────────────────

def _make_polygon_gdf(name: str, coords=None) -> gpd.GeoDataFrame:
    if coords is None:
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly = Polygon(coords)
    return gpd.GeoDataFrame(
        {"id": [f"{name}_0"], "name": [name], "geometry": [poly], "color": ["#ff0000"]}
    )


class TestAddDataScaleFactor:
    def test_raises_when_scale_factor_omitted(self):
        from insitupy.containers import AnnotationsData
        ann = AnnotationsData()
        gdf = _make_polygon_gdf("roi")
        with pytest.raises(ValueError, match="scale_factor is required"):
            ann.add_data(data=gdf, key="test")


class TestAnnotationsToRegions:
    def test_round_trip_geometry_preserved(self):
        xd = _make_insitudata_with_celltypes()
        gdf = _make_polygon_gdf("Tumor")
        xd._annotations.add_data(data=gdf, key="pathology", scale_factor=1.0)

        xd.annotations_to_regions(key="pathology")

        assert "pathology" in xd.regions.keys()
        region_geom = xd.regions["pathology"].geometry.iloc[0]
        annot_geom = xd.annotations["pathology"].geometry.iloc[0]
        assert region_geom.equals(annot_geom)

    def test_custom_region_key(self):
        xd = _make_insitudata_with_celltypes()
        gdf = _make_polygon_gdf("Tumor")
        xd._annotations.add_data(data=gdf, key="pathology", scale_factor=1.0)

        xd.annotations_to_regions(key="pathology", region_key="tumor_region")

        assert "tumor_region" in xd.regions.keys()
        assert "pathology" not in xd.regions.keys()

    def test_missing_key_raises(self):
        xd = _make_insitudata_with_celltypes()
        with pytest.raises(KeyError, match="not found"):
            xd.annotations_to_regions(key="nonexistent")

    def test_name_filter_applied(self):
        xd = _make_insitudata_with_celltypes()
        poly_a = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly_b = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        gdf = gpd.GeoDataFrame({
            "id": ["a_0", "b_0"],
            "name": ["Tumor", "Stroma"],
            "geometry": [poly_a, poly_b],
            "color": ["#ff0000", "#0000ff"],
        })
        xd._annotations.add_data(data=gdf, key="pathology", scale_factor=1.0)

        xd.annotations_to_regions(key="pathology", name_filter="Tumor")

        region_df = xd.regions["pathology"]
        assert list(region_df["name"]) == ["Tumor"]


class TestRegionsToAnnotations:
    def test_round_trip_geometry_preserved(self):
        xd = _make_insitudata_with_celltypes()
        gdf = _make_polygon_gdf("Tumor")
        xd._regions.add_data(data=gdf, key="pathology", scale_factor=1.0)

        xd.regions_to_annotations(key="pathology")

        assert "pathology" in xd.annotations.keys()
        annot_geom = xd.annotations["pathology"].geometry.iloc[0]
        region_geom = xd.regions["pathology"].geometry.iloc[0]
        assert annot_geom.equals(region_geom)

    def test_custom_annotation_key(self):
        xd = _make_insitudata_with_celltypes()
        gdf = _make_polygon_gdf("Tumor")
        xd._regions.add_data(data=gdf, key="pathology", scale_factor=1.0)

        xd.regions_to_annotations(key="pathology", annotation_key="pathology_annot")

        assert "pathology_annot" in xd.annotations.keys()
        assert "pathology" not in xd.annotations.keys()

    def test_missing_key_raises(self):
        xd = _make_insitudata_with_celltypes()
        with pytest.raises(KeyError, match="not found"):
            xd.regions_to_annotations(key="nonexistent")

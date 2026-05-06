"""Tests for tools: tl.dge, calc_distance_of_cells_from,
calculate_gex_diff_to_neighbors, pseudobulk_dge."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Polygon

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.results import DiffExprResults
from insitupy.tools.dge import dge
from insitupy.tools.distance import calc_distance_of_cells_from
from insitupy.tools.neighbors import calculate_gex_diff_to_neighbors
from insitupy.tools.pseudobulk import pseudobulk_dge

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    def test_returns_diffexprresults(self):
        xd = _make_insitudata_with_celltypes()
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
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple="rest",
            method="t-test",
            verbose=False,
        )
        for col in ("log2foldchange", "padj", "scores"):
            assert col in result.main.columns, f"Missing column: {col}"

    def test_main_index_is_gene_names(self):
        xd = _make_insitudata_with_celltypes()
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
        result = dge(
            target=xd,
            target_cell_type_tuple=("celltype", "A"),
            ref_cell_type_tuple=("celltype", "B"),
            verbose=False,
        )
        assert isinstance(result, DiffExprResults)
        assert result.main is not None


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
    def test_returns_four_tuple(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        result = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", verbose=False
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_first_element_is_dataframe(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        df, A, diffs, qc = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", verbose=False
        )
        assert isinstance(df, pd.DataFrame)

    def test_adjacency_matrix_shape(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        df, A, diffs, qc = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", verbose=False
        )
        n = adata.n_obs
        assert A.shape == (n, n)

    def test_qc_stats_is_dict(self):
        adata = _make_adata_with_spatial(n_cells=20, n_genes=8)
        df, A, diffs, qc = calculate_gex_diff_to_neighbors(
            adata, radius=200.0, strategy="mean", verbose=False
        )
        assert isinstance(qc, dict)


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

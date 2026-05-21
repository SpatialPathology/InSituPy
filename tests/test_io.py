"""Tests for InSituData and InSituExperiment partial save methods:
save_geometries(), save_cells(), save_images()."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Polygon

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_insitudata(n_cells=10, n_genes=5, seed=0):
    """Minimal InSituData with cells but no project path."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100

    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="slide1", sample_id="s1",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_polygon_gdf(name: str) -> gpd.GeoDataFrame:
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return gpd.GeoDataFrame(
        {"id": [f"{name}_0"], "name": [name], "geometry": [poly], "color": ["#ff0000"]}
    )


def _make_experiment(n_samples=2, **kwargs):
    """InSituExperiment with no path set (simulates pre-saveas state)."""
    exp = InSituExperiment()
    for i in range(n_samples):
        xd = _make_insitudata(seed=i, **kwargs)
        exp._data.append(xd)
    exp._metadata = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(n_samples)],
    })
    return exp


# ── InSituData.save_geometries ────────────────────────────────────────────────

class TestInSituDataSaveGeometries:
    def test_annotations_written_to_disk(self, tmp_path):
        xd = _make_insitudata()
        xd._annotations.add_data(data=_make_polygon_gdf("Tumor"), key="pathology", scale_factor=1.0)

        xd.save_geometries(path=tmp_path)

        assert (tmp_path / "annotations").exists()

    def test_regions_written_to_disk(self, tmp_path):
        xd = _make_insitudata()
        xd._regions.add_data(data=_make_polygon_gdf("Tumor"), key="rois", scale_factor=1.0)

        xd.save_geometries(path=tmp_path)

        assert (tmp_path / "regions").exists()

    def test_cells_not_written(self, tmp_path):
        xd = _make_insitudata()
        xd._annotations.add_data(data=_make_polygon_gdf("Tumor"), key="pathology", scale_factor=1.0)

        xd.save_geometries(path=tmp_path)

        assert not (tmp_path / "cells").exists()

    def test_metadata_json_written(self, tmp_path):
        xd = _make_insitudata()
        xd._annotations.add_data(data=_make_polygon_gdf("Tumor"), key="pathology", scale_factor=1.0)

        xd.save_geometries(path=tmp_path)

        assert (tmp_path / ".ispy").exists()

    def test_raises_when_unlinked_and_no_path(self):
        xd = _make_insitudata()
        with pytest.raises(RuntimeError, match="no project is linked"):
            xd.save_geometries()

    def test_empty_geometries_writes_only_metadata(self, tmp_path):
        xd = _make_insitudata()

        xd.save_geometries(path=tmp_path)

        assert (tmp_path / ".ispy").exists()
        assert not (tmp_path / "annotations").exists()
        assert not (tmp_path / "regions").exists()


# ── InSituData.save_cells ─────────────────────────────────────────────────────

class TestInSituDataSaveCells:
    def test_cells_written_to_disk(self, tmp_path):
        xd = _make_insitudata()

        xd.save_cells(path=tmp_path)

        assert (tmp_path / "cells").exists()

    def test_annotations_not_written(self, tmp_path):
        xd = _make_insitudata()
        xd._annotations.add_data(data=_make_polygon_gdf("Tumor"), key="pathology", scale_factor=1.0)

        xd.save_cells(path=tmp_path)

        assert not (tmp_path / "annotations").exists()

    def test_metadata_json_written(self, tmp_path):
        xd = _make_insitudata()

        xd.save_cells(path=tmp_path)

        assert (tmp_path / ".ispy").exists()

    def test_raises_when_unlinked_and_no_path(self):
        xd = _make_insitudata()
        with pytest.raises(RuntimeError, match="no project is linked"):
            xd.save_cells()


# ── InSituData.save_images ────────────────────────────────────────────────────

class TestInSituDataSaveImages:
    def test_raises_when_unlinked_and_no_path(self):
        xd = _make_insitudata()
        with pytest.raises(RuntimeError, match="no project is linked"):
            xd.save_images()


# ── InSituExperiment.save_geometries ─────────────────────────────────────────

class TestInSituExperimentSaveGeometries:
    def test_raises_when_no_experiment_path(self):
        exp = _make_experiment()
        with pytest.raises(ValueError, match="No save path"):
            exp.save_geometries()

    def test_raises_on_path_inconsistency(self, tmp_path):
        exp = _make_experiment()
        exp._path = tmp_path
        # Datasets have no path set → parent check fails
        with pytest.raises(ValueError, match="Saving geometries failed"):
            exp.save_geometries()

    def test_calls_per_dataset_save(self, tmp_path, monkeypatch):
        exp = _make_experiment(n_samples=2)
        exp._path = tmp_path
        calls = []
        # Patch dataset paths so path-consistency check passes
        for xd in exp._data:
            xd._path = tmp_path / "data-000"
        monkeypatch.setattr(
            "insitupy._core.data.InSituData.save_geometries",
            lambda self, **kw: calls.append(self),
        )
        # Override path check by setting consistent paths
        for i, xd in enumerate(exp._data):
            xd._path = tmp_path / f"data-{str(i).zfill(3)}"
        exp.save_geometries()
        assert len(calls) == 2


# ── InSituExperiment.save_cells ───────────────────────────────────────────────

class TestInSituExperimentSaveCells:
    def test_raises_when_no_experiment_path(self):
        exp = _make_experiment()
        with pytest.raises(ValueError, match="No save path"):
            exp.save_cells()

    def test_raises_on_path_inconsistency(self, tmp_path):
        exp = _make_experiment()
        exp._path = tmp_path
        with pytest.raises(ValueError, match="Saving cells failed"):
            exp.save_cells()

    def test_calls_per_dataset_save(self, tmp_path, monkeypatch):
        exp = _make_experiment(n_samples=2)
        exp._path = tmp_path
        calls = []
        monkeypatch.setattr(
            "insitupy._core.data.InSituData.save_cells",
            lambda self, **kw: calls.append(self),
        )
        for i, xd in enumerate(exp._data):
            xd._path = tmp_path / f"data-{str(i).zfill(3)}"
        exp.save_cells()
        assert len(calls) == 2


# ── InSituExperiment.save collect-then-raise (M1) ────────────────────────────

def _make_experiment_with_paths(tmp_path, n_samples=3):
    """Experiment with consistent paths set (simulates post-saveas state)."""
    exp = InSituExperiment()
    exp._path = tmp_path
    for i in range(n_samples):
        xd = _make_insitudata(seed=i)
        xd._uid = f"sample_{i}"
        xd._path = tmp_path / f"sample_{i}"
        exp._data.append(xd)
    exp._metadata = pd.DataFrame({"uid": [f"sample_{i}" for i in range(n_samples)]})
    return exp


class TestInSituExperimentSaveCollectFailures:
    def test_all_datasets_attempted_when_some_fail(self, tmp_path, monkeypatch):
        exp = _make_experiment_with_paths(tmp_path, n_samples=3)
        calls = []

        def patched_save(self_xd, **kwargs):
            calls.append(self_xd.uid)
            if self_xd.uid in ("sample_1", "sample_2"):
                raise RuntimeError("injected failure")

        monkeypatch.setattr("insitupy._core.data.InSituData.save", patched_save)

        with pytest.raises(RuntimeError):
            exp.save(collect_warnings_mode=False)

        assert set(calls) == {"sample_0", "sample_1", "sample_2"}

    def test_error_names_failing_uids_and_count(self, tmp_path, monkeypatch):
        exp = _make_experiment_with_paths(tmp_path, n_samples=3)

        def patched_save(self_xd, **kwargs):
            if self_xd.uid in ("sample_1", "sample_2"):
                raise RuntimeError("injected failure")

        monkeypatch.setattr("insitupy._core.data.InSituData.save", patched_save)

        with pytest.raises(RuntimeError, match=r"save\(\) failed for 2/3") as exc_info:
            exp.save(collect_warnings_mode=False)

        msg = str(exc_info.value)
        assert "sample_1" in msg
        assert "sample_2" in msg

    def test_experiment_level_files_not_written_on_failure(self, tmp_path, monkeypatch):
        exp = _make_experiment_with_paths(tmp_path, n_samples=2)
        written = []

        def always_fails(self_xd, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr("insitupy._core.data.InSituData.save", always_fails)
        monkeypatch.setattr(exp, "save_metadata", lambda **kw: written.append("metadata"))
        monkeypatch.setattr(exp, "save_colors", lambda **kw: written.append("colors"))
        monkeypatch.setattr(exp, "save_filters", lambda **kw: written.append("filters"))

        with pytest.raises(RuntimeError):
            exp.save(collect_warnings_mode=False)

        assert written == []

    def test_collect_warnings_mode_also_collects_failures(self, tmp_path, monkeypatch):
        exp = _make_experiment_with_paths(tmp_path, n_samples=2)

        def patched_save(self_xd, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr("insitupy._core.data.InSituData.save", patched_save)

        with pytest.raises(RuntimeError, match=r"save\(\) failed for 2/2"):
            exp.save(collect_warnings_mode=True)

"""Tests for IO reader functions: read_any, read_visium, read_qupath, read_qupath_project."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from insitupy.io import read_any, read_qupath, read_visium
from insitupy.io.experiment import read_qupath_project
from insitupy._core.data import InSituData


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_measurements_csv(path: Path, n_cells=3, n_genes=4):
    """Write a minimal gene-expression CSV: index=cell IDs, columns=gene names."""
    index = [f"cell_{i}" for i in range(n_cells)]
    data = np.arange(n_cells * n_genes).reshape(n_cells, n_genes)
    df = pd.DataFrame(data, index=index, columns=[f"gene{j}" for j in range(n_genes)])
    df.to_csv(path)
    return index


def _write_coordinates_csv(path: Path, cell_ids):
    """Write a minimal cell-coordinates CSV: index=cell IDs, columns x and y."""
    n = len(cell_ids)
    df = pd.DataFrame(
        {"x": np.arange(n, dtype=float) * 10, "y": np.arange(n, dtype=float) * 5},
        index=cell_ids,
    )
    df.to_csv(path)


# ── read_any ──────────────────────────────────────────────────────────────────

class TestReadAny:
    def _make_files(self, tmp_path, n_cells=3, n_genes=4):
        m = tmp_path / "measurements.csv"
        c = tmp_path / "coordinates.csv"
        ids = _write_measurements_csv(m, n_cells=n_cells, n_genes=n_genes)
        _write_coordinates_csv(c, ids)
        return m, c, ids

    def test_returns_insitudata(self, tmp_path):
        m, c, _ = self._make_files(tmp_path)
        result = read_any(
            cellular_measurements={"main": m},
            cellular_coordinates=c,
        )
        assert isinstance(result, InSituData)

    def test_obs_var_shape(self, tmp_path):
        m, c, ids = self._make_files(tmp_path, n_cells=5, n_genes=6)
        result = read_any(
            cellular_measurements={"main": m},
            cellular_coordinates=c,
        )
        table = result.cells.table
        assert table.n_obs == 5
        assert table.n_vars == 6

    def test_spatial_coordinates_loaded(self, tmp_path):
        m, c, ids = self._make_files(tmp_path, n_cells=3)
        result = read_any(
            cellular_measurements={"main": m},
            cellular_coordinates=c,
        )
        assert "spatial" in result.cells.table.obsm
        assert result.cells.table.obsm["spatial"].shape == (3, 2)

    def test_dataset_and_sample_name_stored(self, tmp_path):
        m, c, _ = self._make_files(tmp_path)
        result = read_any(
            cellular_measurements={"main": m},
            cellular_coordinates=c,
            dataset_name="MySlide",
            sample_name="MySample",
        )
        assert result.slide_id == "MySlide"
        assert result.sample_id == "MySample"

    def test_missing_measurements_raises(self, tmp_path):
        _, c, ids = self._make_files(tmp_path)
        with pytest.raises(FileNotFoundError):
            read_any(
                cellular_measurements={"main": tmp_path / "nonexistent.csv"},
                cellular_coordinates=c,
            )

    def test_nucleus_boundaries_without_cell_boundaries_raises(self, tmp_path):
        m, c, _ = self._make_files(tmp_path)
        with pytest.raises(ValueError, match="cell_boundaries"):
            read_any(
                cellular_measurements={"main": m},
                cellular_coordinates=c,
                nucleus_boundaries=tmp_path / "nuclei.geojson",
            )

    def test_images_without_pixel_size_raises(self, tmp_path):
        m, c, _ = self._make_files(tmp_path)
        dummy_image = tmp_path / "img.zarr"
        dummy_image.mkdir()
        with pytest.raises(ValueError, match="pixel_size"):
            read_any(
                cellular_measurements={"main": m},
                cellular_coordinates=c,
                images={"nuclei": dummy_image},
            )

    def test_coordinate_shift_applied(self, tmp_path):
        m, c, _ = self._make_files(tmp_path, n_cells=3)
        result_no_shift = read_any(
            cellular_measurements={"main": m},
            cellular_coordinates=c,
            xshift=0, yshift=0,
        )
        result_shifted = read_any(
            cellular_measurements={"main": m},
            cellular_coordinates=c,
            xshift=5.0, yshift=2.5,
        )
        coords_no_shift = result_no_shift.cells.table.obsm["spatial"]
        coords_shifted = result_shifted.cells.table.obsm["spatial"]
        np.testing.assert_allclose(
            coords_no_shift[:, 0] - 5.0, coords_shifted[:, 0]
        )


# ── read_visium ───────────────────────────────────────────────────────────────

class TestReadVisium:
    def test_missing_directory_raises(self, tmp_path):
        pytest.importorskip("spatialdata_io")
        with pytest.raises(FileNotFoundError):
            read_visium(
                path=tmp_path / "nonexistent_visium",
                dataset_name="test",
            )

    def test_basic_load(self):
        pytest.skip("requires real Visium dataset")


# ── read_qupath ───────────────────────────────────────────────────────────────

class TestReadQupath:
    def test_missing_annotation_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="annotation"):
            read_qupath(
                path=tmp_path,
                pixel_size=0.65,
                dataset_name="slide1",
                sample_name="sample1",
            )

    def test_missing_measurements_raises(self, tmp_path):
        (tmp_path / "annotation.geojson").write_text("{}")
        with pytest.raises(FileNotFoundError, match="measurements"):
            read_qupath(
                path=tmp_path,
                pixel_size=0.65,
                dataset_name="slide1",
                sample_name="sample1",
            )

    def test_missing_boundaries_raises(self, tmp_path):
        (tmp_path / "annotation.geojson").write_text("{}")
        (tmp_path / "measurements.tsv").write_text("id\tgene1\ncell1\t1\n")
        with pytest.raises(FileNotFoundError, match="boundaries"):
            read_qupath(
                path=tmp_path,
                pixel_size=0.65,
                dataset_name="slide1",
                sample_name="sample1",
            )

    def test_missing_image_raises(self, tmp_path):
        (tmp_path / "annotation.geojson").write_text("{}")
        (tmp_path / "measurements.tsv").write_text("id\tgene1\ncell1\t1\n")
        (tmp_path / "cells.geojson").write_text("{}")
        with pytest.raises(FileNotFoundError, match="image"):
            read_qupath(
                path=tmp_path,
                pixel_size=0.65,
                dataset_name="slide1",
                sample_name="sample1",
            )

    def test_full_load(self):
        pytest.skip("requires real QuPath export dataset")


# ── read_qupath_project ───────────────────────────────────────────────────────

class TestReadQupathProject:
    def test_missing_pixel_size_without_project_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="pixel_size"):
            read_qupath_project(path=tmp_path, pixel_size=None)

    def test_full_load(self):
        pytest.skip("requires real QuPath project directory")

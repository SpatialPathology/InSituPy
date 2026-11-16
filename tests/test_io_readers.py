"""Tests for IO reader functions: read_any, read_visium, read_qupath, read_qupath_project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insitupy._core.data import InSituData
from insitupy.io import read_any, read_qupath, read_visium, read_visium_hd
from insitupy.io.experiment import read_qupath_project

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

    def test_missing_pixel_size_raises(self, tmp_path):
        """fullres_pixel_size is required; omitting it must raise before any
        spatialdata-io parsing happens (no real Visium bundle needed - the
        raise precedes the directory-content checks)."""
        pytest.importorskip("spatialdata_io")
        existing_dir = tmp_path / "visium_bundle"
        existing_dir.mkdir()
        with pytest.raises(ValueError, match="fullres_pixel_size"):
            read_visium(
                path=existing_dir,
                dataset_name="test",
                fullres_pixel_size=None,
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


class TestReadVisiumHD:
    @staticmethod
    def _bundle(root: Path, bins=(8, 16), segmentation=True, mpp=0.5):
        import json
        for b in bins:
            d = root / "binned_outputs" / f"square_{b:03d}um" / "spatial"
            d.mkdir(parents=True)
        seg = root / "segmented_outputs"
        (seg / "spatial").mkdir(parents=True)
        sf = {"microns_per_pixel": mpp, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1}
        (seg / "spatial" / "scalefactors_json.json").write_text(json.dumps(sf))
        if segmentation:
            (seg / "cell_segmentations.geojson").write_text("{}")
            (seg / "filtered_feature_cell_matrix.h5").write_text("")
        return root

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_visium_hd(tmp_path / "nope")

    def test_unknown_bin_size_raises(self, tmp_path):
        self._bundle(tmp_path, bins=(8,))
        with pytest.raises(ValueError, match=r"Available: \[8\]"):
            read_visium_hd(tmp_path, bin_size=2)

    def test_controlled_kwargs_rejected(self, tmp_path):
        self._bundle(tmp_path)
        with pytest.raises(TypeError):
            read_visium_hd(tmp_path, load_segmentations_only=True)

    def test_nuclei_without_segmentation_raises(self, tmp_path):
        self._bundle(tmp_path)
        with pytest.raises(ValueError, match="load_segmentation"):
            read_visium_hd(tmp_path, load_segmentation=False, load_nuclei=True)

    def test_reader_logic_on_stubbed_visium_hd(self, tmp_path, monkeypatch):
        pytest.importorskip("spatialdata")
        pytest.importorskip("spatialdata_io")
        import spatialdata_io

        from tests.spatialdata_fixtures import make_visium_hd_like_sdata

        self._bundle(tmp_path, mpp=0.5)
        (tmp_path / "segmented_outputs" / "nucleus_segmentations.geojson").write_text("{}")
        calls = {}

        def fake_visium_hd(path, **kwargs):
            calls.update(kwargs)
            return make_visium_hd_like_sdata(prefix="pre")

        monkeypatch.setattr(spatialdata_io, "visium_hd", fake_visium_hd)
        xd = read_visium_hd(tmp_path, load_nuclei=True, verbose=False)

        assert calls["bin_size"] == [8, 16]
        assert calls["load_segmentations_only"] is False
        assert calls["load_nucleus_segmentations"] is True
        assert calls["bins_as_squares"] is True

        assert list(xd.units.keys()) == ["square_008um", "square_016um"]
        assert xd.units.main_key == "square_008um"
        su = xd.units["square_008um"]
        assert su._unit_type == "visium_hd_bin"
        assert su._data.uns["bin_size_um"] == 8
        # geometries and centroids scaled by microns_per_pixel (0.5): 8 px bins -> 4 um
        minx, _, maxx, _ = su.shapes.geometry.iloc[0].bounds
        assert maxx - minx == pytest.approx(4.0)
        # first row is location 2 -> bin origin (0, 8) px, centre (4, 12) px -> (2, 6) um
        np.testing.assert_allclose(su._data.obsm["spatial"][0], (2.0, 6.0))
        cd = xd.cells["main"]
        assert cd.matrix.n_obs == 3
        assert cd.boundaries["nuclei"] is not None
        assert xd.images["hires"].metadata["pixel_size"] == pytest.approx(0.5 / 0.5)
        assert xd.images["lowres"].metadata["pixel_size"] == pytest.approx(0.5 / 0.1)

    def test_real_dataset_smoke(self):
        import os
        pytest.importorskip("spatialdata_io")
        root = Path(os.environ.get(
            "INSITUPY_VISIUM_HD_TEST_DIR",
            Path.home() / ".cache/InSituPy/demo_datasets/visium_test_dataset_v4"))
        if not root.is_dir():
            pytest.skip("Visium HD test dataset not available")
        xd = read_visium_hd(root, load_nuclei=True, verbose=False)
        assert list(xd.units.keys()) == ["square_008um", "square_016um"]
        assert len(xd.units["square_008um"]) == 20830
        assert len(xd.units["square_016um"]) == 5262
        cd = xd.cells["main"]
        assert cd.matrix.n_obs == 612
        mask = np.asarray(cd.boundaries["cells"].compute())
        nuc = np.asarray(cd.boundaries["nuclei"].compute())
        xy = cd.matrix.obsm["spatial"]
        values = np.asarray(cd.boundaries.seg_mask_value)
        checked = 0
        for i in np.random.default_rng(0).choice(612, 40, replace=False):
            x, y = int(xy[i, 0]), int(xy[i, 1])
            if mask[y, x] != 0:
                assert mask[y, x] == values[i]
                if nuc[y, x] != 0:
                    assert nuc[y, x] == values[i]
                checked += 1
        assert checked > 10
        assert set(np.unique(nuc)) <= set(np.unique(mask))


class TestReadVisiumHD:
    @staticmethod
    def _bundle(root: Path, bins=(8, 16), segmentation=True, mpp=0.5):
        import json
        for b in bins:
            d = root / "binned_outputs" / f"square_{b:03d}um" / "spatial"
            d.mkdir(parents=True)
        seg = root / "segmented_outputs"
        (seg / "spatial").mkdir(parents=True)
        sf = {"microns_per_pixel": mpp, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1}
        (seg / "spatial" / "scalefactors_json.json").write_text(json.dumps(sf))
        if segmentation:
            (seg / "cell_segmentations.geojson").write_text("{}")
            (seg / "filtered_feature_cell_matrix.h5").write_text("")
        return root

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_visium_hd(tmp_path / "nope")

    def test_unknown_bin_size_raises(self, tmp_path):
        self._bundle(tmp_path, bins=(8,))
        with pytest.raises(ValueError, match=r"Available: \[8\]"):
            read_visium_hd(tmp_path, bin_size=2)

    def test_controlled_kwargs_rejected(self, tmp_path):
        self._bundle(tmp_path)
        with pytest.raises(TypeError):
            read_visium_hd(tmp_path, load_segmentations_only=True)

    def test_nuclei_without_segmentation_raises(self, tmp_path):
        self._bundle(tmp_path)
        with pytest.raises(ValueError, match="load_segmentation"):
            read_visium_hd(tmp_path, load_segmentation=False, load_nuclei=True)

    def test_reader_logic_on_stubbed_visium_hd(self, tmp_path, monkeypatch):
        pytest.importorskip("spatialdata")
        pytest.importorskip("spatialdata_io")
        import spatialdata_io

        from tests.spatialdata_fixtures import make_visium_hd_like_sdata

        self._bundle(tmp_path, mpp=0.5)
        (tmp_path / "segmented_outputs" / "nucleus_segmentations.geojson").write_text("{}")
        calls = {}

        def fake_visium_hd(path, **kwargs):
            calls.update(kwargs)
            return make_visium_hd_like_sdata(prefix="pre")

        monkeypatch.setattr(spatialdata_io, "visium_hd", fake_visium_hd)
        xd = read_visium_hd(tmp_path, load_nuclei=True, verbose=False)

        assert calls["bin_size"] == [8, 16]
        assert calls["load_segmentations_only"] is False
        assert calls["load_nucleus_segmentations"] is True
        assert calls["bins_as_squares"] is True

        assert list(xd.units.keys()) == ["square_008um", "square_016um"]
        assert xd.units.main_key == "square_008um"
        su = xd.units["square_008um"]
        assert su._unit_type == "visium_hd_bin"
        assert su._data.uns["bin_size_um"] == 8
        # geometries and centroids scaled by microns_per_pixel (0.5): 8 px bins -> 4 um
        minx, _, maxx, _ = su.shapes.geometry.iloc[0].bounds
        assert maxx - minx == pytest.approx(4.0)
        # first row is location 2 -> bin origin (0, 8) px, centre (4, 12) px -> (2, 6) um
        np.testing.assert_allclose(su._data.obsm["spatial"][0], (2.0, 6.0))
        cd = xd.cells["main"]
        assert cd.matrix.n_obs == 3
        assert cd.boundaries["nuclei"] is not None
        assert xd.images.metadata["hires"]["pixel_size"] == pytest.approx(0.5 / 0.5)
        assert xd.images.metadata["lowres"]["pixel_size"] == pytest.approx(0.5 / 0.1)

    def test_real_dataset_smoke(self):
        import os
        pytest.importorskip("spatialdata_io")
        root = Path(os.environ.get(
            "INSITUPY_VISIUM_HD_TEST_DIR",
            Path.home() / ".cache/InSituPy/demo_datasets/visium_test_dataset_v4"))
        if not root.is_dir():
            pytest.skip("Visium HD test dataset not available")
        xd = read_visium_hd(root, load_nuclei=True, verbose=False)
        assert list(xd.units.keys()) == ["square_008um", "square_016um"]
        assert len(xd.units["square_008um"]) == 20830
        assert len(xd.units["square_016um"]) == 5262
        cd = xd.cells["main"]
        assert cd.matrix.n_obs == 612
        mask = np.asarray(cd.boundaries["cells"].compute())
        nuc = np.asarray(cd.boundaries["nuclei"].compute())
        xy = cd.matrix.obsm["spatial"]
        values = np.asarray(cd.boundaries.seg_mask_value)
        checked = 0
        for i in np.random.default_rng(0).choice(612, 40, replace=False):
            x, y = int(xy[i, 0]), int(xy[i, 1])
            if mask[y, x] != 0:
                assert mask[y, x] == values[i]
                if nuc[y, x] != 0:
                    assert nuc[y, x] == values[i]
                checked += 1
        assert checked > 10
        assert set(np.unique(nuc)) <= set(np.unique(mask))

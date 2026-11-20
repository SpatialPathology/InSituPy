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

    def test_units_coordinates_are_in_um_like_the_shapes(self):
        """The units table's obsm['spatial'] must be scaled like the spot geometries."""
        import os
        pytest.importorskip("spatialdata_io")
        root = Path(os.environ.get(
            "INSITUPY_VISIUM_TEST_DIR",
            Path.home() / ".cache/InSituPy/demo_datasets/visium_hbreastcancer"
            / "CytAssist_FFPE_Human_Breast_Cancer"))
        if not (root / "spatial").is_dir():
            pytest.skip("Visium test dataset not available")
        xd = read_visium(root, dataset_name="v", fullres_pixel_size=0.5, verbose=False)
        su = xd.units["visium"]
        centroids = np.c_[su.shapes.centroid.x, su.shapes.centroid.y]
        np.testing.assert_allclose(su.table.obsm["spatial"], centroids, atol=1.0)

    def test_default_sample_name_from_real_data(self):
        """Without sample_name, sample_id comes from the Space Ranger library_ids."""
        import os
        pytest.importorskip("spatialdata_io")
        root = Path(os.environ.get(
            "INSITUPY_VISIUM_TEST_DIR",
            Path.home() / ".cache/InSituPy/demo_datasets/visium_hbreastcancer"
            / "CytAssist_FFPE_Human_Breast_Cancer"))
        if not (root / "spatial").is_dir():
            pytest.skip("Visium test dataset not available")
        xd = read_visium(root, fullres_pixel_size=0.5, verbose=False)
        assert xd.sample_id == "CytAssist_FFPE_Human_Breast_Cancer"


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
        assert xd.images.metadata["hires"]["pixel_size"] == pytest.approx(0.5 / 0.5)
        assert xd.images.metadata["lowres"]["pixel_size"] == pytest.approx(0.5 / 0.1)

    @staticmethod
    def _write_mapping(root: Path, barcodes, cell_ids):
        pytest.importorskip("pyarrow")
        pd.DataFrame({
            "square_002um": barcodes,
            "square_008um": ["s_008um_00000_00000-1"] * len(barcodes),
            "square_016um": ["s_016um_00000_00000-1"] * len(barcodes),
            "cell_id": cell_ids,
            "in_nucleus": [True] * len(barcodes),
            "in_cell": [cid is not None for cid in cell_ids],
        }).to_parquet(root / "barcode_mappings.parquet")

    def _read_stubbed(self, tmp_path, monkeypatch, **kwargs):
        pytest.importorskip("spatialdata")
        pytest.importorskip("spatialdata_io")
        import spatialdata_io

        from tests.spatialdata_fixtures import make_visium_hd_like_sdata

        def fake_visium_hd(path, **kw):
            sdata = make_visium_hd_like_sdata(prefix="pre")
            # as in real spatialdata-io output, cell tables are indexed by cell id
            cell_table = sdata.tables["cell_segmentations"]
            cell_table.obs_names = cell_table.obs["cell_id"].astype(str).to_numpy()
            return sdata

        monkeypatch.setattr(spatialdata_io, "visium_hd", fake_visium_hd)
        return read_visium_hd(tmp_path, verbose=False, **kwargs)

    @staticmethod
    def _write_library_ids(h5_path: Path, ids):
        h5py = pytest.importorskip("h5py")
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as h:
            h.attrs["library_ids"] = np.array(ids, dtype="S")

    def test_default_sample_name_from_library_ids(self, tmp_path, monkeypatch):
        self._bundle(tmp_path)
        self._write_library_ids(
            tmp_path / "binned_outputs" / "square_008um" / "filtered_feature_bc_matrix.h5",
            [b"my_sample"],
        )
        assert self._read_stubbed(tmp_path, monkeypatch).sample_id == "my_sample"

    def test_explicit_sample_name_wins(self, tmp_path, monkeypatch):
        self._bundle(tmp_path)
        self._write_library_ids(
            tmp_path / "binned_outputs" / "square_008um" / "filtered_feature_bc_matrix.h5",
            [b"my_sample"],
        )
        xd = self._read_stubbed(tmp_path, monkeypatch, sample_name="override")
        assert xd.sample_id == "override"

    @pytest.mark.parametrize("sub, expected", [("slideA", "slideA"), ("run42/outs", "run42")])
    def test_default_sample_name_falls_back_to_folder(self, tmp_path, monkeypatch, sub, expected):
        root = tmp_path / sub
        root.mkdir(parents=True)
        self._bundle(root)  # seg h5 is empty (not HDF5), no bin h5
        assert self._read_stubbed(root, monkeypatch).sample_id == expected

    def test_default_10x_sample_name_multi_library_falls_back(self, tmp_path):
        from insitupy.io.data import _default_10x_sample_name

        root = tmp_path / "aggr_run"
        h5 = root / "filtered_feature_bc_matrix.h5"
        self._write_library_ids(h5, [b"a", b"b"])
        assert _default_10x_sample_name(root, [h5]) == "aggr_run"

    def test_default_10x_sample_name_relative_dot(self, tmp_path, monkeypatch):
        from insitupy.io.data import _default_10x_sample_name

        root = tmp_path / "slideB"
        root.mkdir()
        monkeypatch.chdir(root)
        assert _default_10x_sample_name(Path("."), []) == "slideB"

    def test_bin_mapping_keeps_only_layer_cells_and_parses_grid(self, tmp_path, monkeypatch):
        self._bundle(tmp_path)
        self._write_mapping(
            tmp_path,
            barcodes=["s_002um_00000_00004-1", "s_002um_00012_00007-1",
                      "s_002um_00001_00001-1", "s_002um_00002_00002-1"],
            cell_ids=["cellid_000000001-1", "cellid_000000002-1", "cellid_000000099-1", None],
        )
        xd = self._read_stubbed(tmp_path, monkeypatch)
        mapping = xd.cells["main"].table.uns["visium_hd_bin_mapping"]
        assert list(mapping.columns) == ["cell_id", "array_row", "array_col", "in_nucleus"]
        # the cell not in the layer and the unassigned bin are dropped
        assert list(mapping["cell_id"].astype(str)) == ["cellid_000000001-1", "cellid_000000002-1"]
        assert list(mapping["array_row"]) == [0, 12]
        assert list(mapping["array_col"]) == [4, 7]
        assert set(mapping["cell_id"].astype(str)) <= set(xd.cells["main"].table.obs_names)

        xd = self._read_stubbed(tmp_path, monkeypatch, load_bin_mapping=False)
        assert "visium_hd_bin_mapping" not in xd.cells["main"].table.uns

    def test_bin_mapping_unparsable_barcode_raises(self, tmp_path, monkeypatch):
        self._bundle(tmp_path)
        self._write_mapping(tmp_path, ["s_002um_bad-1"], ["cellid_000000001-1"])
        with pytest.raises(ValueError, match="array_row"):
            self._read_stubbed(tmp_path, monkeypatch)

    def test_bin_mapping_no_overlap_warns(self, tmp_path, monkeypatch):
        self._bundle(tmp_path)
        self._write_mapping(tmp_path, ["s_002um_00000_00004-1"], ["cellid_000000099-1"])
        with pytest.warns(UserWarning, match="matches the cell layer"):
            xd = self._read_stubbed(tmp_path, monkeypatch)
        assert len(xd.cells["main"].table.uns["visium_hd_bin_mapping"]) == 0

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

    def test_real_dataset_bin_nesting_mapping_and_transfer(self):
        import os
        pytest.importorskip("spatialdata_io")
        pq = pytest.importorskip("pyarrow.parquet")
        import pyarrow.compute as pc

        import insitupy as isp
        root = Path(os.environ.get(
            "INSITUPY_VISIUM_HD_TEST_DIR",
            Path.home() / ".cache/InSituPy/demo_datasets/visium_test_dataset_v4"))
        if not root.is_dir():
            pytest.skip("Visium HD test dataset not available")

        # nesting 2 -> 8 -> 16 um is plain integer division of the grid indices
        df = pq.read_table(
            root / "barcode_mappings.parquet",
            columns=["square_002um", "square_008um", "square_016um"],
            filters=pc.field("cell_id").is_valid(),
        ).to_pandas()
        df = df.sample(1000, random_state=0)
        grid = {
            s: df[f"square_{s:03d}um"].str.extract(rf"^s_{s:03d}um_(\d+)_(\d+)-\d+$").astype(int).to_numpy()
            for s in (2, 8, 16)
        }
        np.testing.assert_array_equal(grid[8], grid[2] // 4)
        np.testing.assert_array_equal(grid[16], grid[2] // 8)

        xd = read_visium_hd(root, verbose=False)
        cells = xd.cells["main"].table
        mapping = cells.uns["visium_hd_bin_mapping"]
        assert len(mapping) > 0
        assert set(mapping["cell_id"].astype(str)) <= set(cells.obs_names)

        # bins -> cells transfer on a numeric column
        isp.tl.transfer_obs(xd, "array_row", source=("units", "square_008um"),
                            target=("cells", "main"))
        mapped = cells.obs_names.isin(mapping["cell_id"].astype(str))
        assert mapped.any()
        assert cells.obs.loc[mapped, "array_row"].notna().mean() > 0.9

        # mapping survives a save / read round trip
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hd"
            xd.saveas(out)
            xd2 = isp.InSituData.read(out)
            xd2.load_cells()
            m2 = xd2.cells["main"].table.uns["visium_hd_bin_mapping"]
            assert len(m2) == len(mapping)

"""Image IO roundtrip tests and utility tests.

Covers:
- Priority 1: write_ome_tiff, write_zarr roundtrips (data corruption risk)
- Priority 5: im.create_img_pyramid, crop_dask_array_or_pyramid, read_image
"""

import zipfile

import dask.array as da
import numpy as np
import pytest

from insitupy._exceptions import NoImageOverlapError
from insitupy.images import (
    read_image,
    read_ome_tiff,
    read_zarr,
    write_ome_tiff,
    write_zarr,
)
from insitupy.images.utils import create_img_pyramid, crop_dask_array_or_pyramid


def _gray_array(height=64, width=64, dtype=np.uint16):
    arr = np.arange(height * width, dtype=dtype).reshape(height, width)
    return da.from_array(arr, chunks=(height, width))


# ── Priority 1: write_ome_tiff roundtrip ──────────────────────────────────────

class TestWriteOmeTiffRoundtrip:
    def test_pixel_values_preserved_at_full_resolution(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.ome.tiff"
        write_ome_tiff(
            image=img,
            file=out,
            axes="YX",
            photometric="minisblack",
            subresolutions=2,
            pixelsize=0.2125,
        )
        pyramid = read_ome_tiff(out)
        # Full-resolution level should match original pixel values
        result = np.asarray(pyramid[0]) if isinstance(pyramid, list) else np.asarray(pyramid)
        np.testing.assert_array_equal(result, np.asarray(img))

    def test_pyramid_has_multiple_levels(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.ome.tiff"
        write_ome_tiff(
            image=img,
            file=out,
            axes="YX",
            photometric="minisblack",
            subresolutions=2,
        )
        pyramid = read_ome_tiff(out)
        assert isinstance(pyramid, list)
        assert len(pyramid) == 3  # original + 2 subresolutions

    def test_overwrite_guard_raises(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.ome.tiff"
        write_ome_tiff(image=img, file=out, axes="YX", photometric="minisblack", subresolutions=0)

        with pytest.raises(FileExistsError):
            write_ome_tiff(image=img, file=out, axes="YX", photometric="minisblack", subresolutions=0)

    def test_overwrite_flag_replaces_file(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.ome.tiff"
        write_ome_tiff(image=img, file=out, axes="YX", photometric="minisblack", subresolutions=0)
        write_ome_tiff(
            image=img, file=out, axes="YX", photometric="minisblack", subresolutions=0, overwrite=True
        )
        assert out.exists()


# ── Priority 1: write_zarr roundtrip ──────────────────────────────────────────

class TestWriteZarrRoundtrip:
    def _metadata(self, axes="YX", pixel_size=0.5):
        return {"OME": {}, "axes": axes, "pixel_size": pixel_size}

    def test_pixel_values_preserved(self, tmp_path):
        arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        img = da.from_array(arr, chunks=(64, 64))
        out = tmp_path / "test.zarr"
        meta = self._metadata()
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=True)

        result_img, _, _, _ = read_zarr(out)
        # read_zarr returns a list for pyramids; check full-resolution level
        level0 = result_img[0] if isinstance(result_img, list) else result_img
        result = np.asarray(level0.compute() if hasattr(level0, "compute") else level0)
        np.testing.assert_array_equal(result, arr)

    def test_axes_and_pixel_size_preserved(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.zarr"
        meta = self._metadata(axes="YX", pixel_size=0.2125)
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=True)

        _, _, axes_out, pixel_size_out = read_zarr(out)
        assert axes_out == "YX"
        assert pixel_size_out == pytest.approx(0.2125)

    def test_pyramid_preserved(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.zarr"
        meta = self._metadata()
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=True)

        result_img, _, _, _ = read_zarr(out)
        assert isinstance(result_img, list)
        assert len(result_img) > 1

    def test_overwrite_guard_raises(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.zarr"
        meta = self._metadata()
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=False)

        with pytest.raises(FileExistsError):
            write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=False)

    def test_overwrite_flag_replaces_store(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.zarr"
        meta = self._metadata()
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=False)
        write_zarr(
            image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=False, overwrite=True
        )
        assert out.exists()

    def test_single_array_store_roundtrip(self, tmp_path):
        arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        img = da.from_array(arr, chunks=(64, 64))
        out = tmp_path / "test.zarr"
        write_zarr(image=img, file=out, img_metadata=self._metadata(), axes="YX", save_pyramid=False)

        result_img, _, axes_out, _ = read_zarr(out)
        assert not isinstance(result_img, list)
        np.testing.assert_array_equal(np.asarray(result_img), arr)
        assert axes_out == "YX"

    @pytest.mark.parametrize("save_pyramid", [True, False])
    def test_zarr_zip_suffix_writes_zip_archive(self, tmp_path, save_pyramid):
        arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        img = da.from_array(arr, chunks=(64, 64))
        out = tmp_path / "test.zarr.zip"
        meta = self._metadata(pixel_size=0.2125)
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=save_pyramid)

        assert out.is_file()
        assert zipfile.is_zipfile(out)

        result_img, _, axes_out, pixel_size_out = read_zarr(out)
        level0 = result_img[0] if isinstance(result_img, list) else result_img
        np.testing.assert_array_equal(np.asarray(level0), arr)
        assert axes_out == "YX"
        assert pixel_size_out == pytest.approx(0.2125)


    def test_pyramid_with_more_than_ten_levels_keeps_resolution_order(self, tmp_path):
        # level keys "10", "11" must not sort between "1" and "2"
        arr = np.zeros((2048, 2048), dtype=np.uint8)
        pyramid = create_img_pyramid(da.from_array(arr, chunks=(512, 512)), axes="YX", nsubres=11)
        assert len(pyramid) == 12
        out = tmp_path / "test.zarr"
        write_zarr(image=pyramid, file=out, img_metadata=self._metadata(), axes="YX")

        result_img, _, _, _ = read_zarr(out)
        assert [lvl.shape for lvl in result_img] == [lvl.shape for lvl in pyramid]

# ── Priority 5: read_image dispatch ───────────────────────────────────────────

class TestReadImageDispatch:
    def test_zarr_path_dispatches_to_zarr_reader(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.zarr"
        meta = {"OME": {}, "axes": "YX", "pixel_size": 1.0}
        write_zarr(image=img, file=out, img_metadata=meta, axes="YX", save_pyramid=True)

        result_img, ome_meta, axes, pixel_size = read_image(out)
        assert axes == "YX"
        assert pixel_size == pytest.approx(1.0)

    def test_ome_tiff_path_dispatches_to_tiff_reader(self, tmp_path):
        img = _gray_array()
        out = tmp_path / "test.ome.tiff"
        write_ome_tiff(
            image=img, file=out, axes="YX", photometric="minisblack",
            subresolutions=0, pixelsize=0.5,
        )
        result_img, ome_meta, axes, pixel_size = read_image(out)
        result = np.asarray(result_img.compute() if hasattr(result_img, "compute") else result_img)
        assert result.shape == (64, 64)
        assert pixel_size == pytest.approx(0.5)

    def test_planar_rgb_ome_tiff_reads_as_yxs(self, tmp_path):
        # tifffile reports planar RGB as 'SYX'; read_image must return canonical YXS
        import tifffile

        rgb = np.random.default_rng(5).integers(0, 256, (40, 30, 3), dtype=np.uint8)
        out = tmp_path / "planar.ome.tif"
        tifffile.imwrite(
            out, np.moveaxis(rgb, -1, 0), photometric="rgb", planarconfig="separate",
            ome=True, metadata={"axes": "SYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5},
        )

        result_img, _, axes, _ = read_image(out)
        level0 = result_img[0] if isinstance(result_img, list) else result_img
        assert axes == "YXS"
        np.testing.assert_array_equal(np.asarray(level0), rgb)


# ── Priority 5: create_img_pyramid ────────────────────────────────────────────

class TestCreateImgPyramid:
    def test_level_count(self):
        img = da.from_array(np.zeros((64, 64), dtype=np.uint8), chunks=(64, 64))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=3, scale_steps=2)
        assert len(pyramid) == 4  # original + 3 subresolutions

    def test_downsampling_factor(self):
        img = da.from_array(np.zeros((64, 64), dtype=np.uint8), chunks=(64, 64))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=2, scale_steps=2)
        assert pyramid[0].shape == (64, 64)
        assert pyramid[1].shape == (32, 32)
        assert pyramid[2].shape == (16, 16)

    def test_rgb_downsampling_preserves_channel_dim(self):
        img = da.from_array(np.zeros((64, 64, 3), dtype=np.uint8), chunks=(64, 64, 3))
        pyramid = create_img_pyramid(img, axes="YXS", nsubres=2, scale_steps=2)
        assert pyramid[0].shape == (64, 64, 3)
        assert pyramid[1].shape == (32, 32, 3)

    def test_mismatched_axes_raises(self):
        img = da.from_array(np.zeros((64, 64), dtype=np.uint8), chunks=(64, 64))
        with pytest.raises(ValueError, match="axes"):
            create_img_pyramid(img, axes="YXS", nsubres=1)


# ── Priority 5: crop_dask_array_or_pyramid ────────────────────────────────────

class TestCropDaskArrayOrPyramid:
    def test_single_array_crop_bounds(self):
        arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        img = da.from_array(arr, chunks=(64, 64))
        cropped = crop_dask_array_or_pyramid(img, xlim=(0, 32), ylim=(0, 32), pixel_size=1.0)
        result = np.asarray(cropped)
        assert result.shape == (32, 32)
        np.testing.assert_array_equal(result, arr[0:32, 0:32])

    def test_pixel_size_scaling(self):
        arr = np.zeros((100, 100), dtype=np.uint8)
        img = da.from_array(arr, chunks=(100, 100))
        # pixel_size=2.0: xlim/ylim in physical units → halved in pixels
        # xlim=(0, 50) with pixel_size=2.0 → pixel range [0:25]
        cropped = crop_dask_array_or_pyramid(img, xlim=(0, 50), ylim=(0, 50), pixel_size=2.0)
        result = np.asarray(cropped)
        assert result.shape == (25, 25)

    def test_pyramid_crop_preserves_level_count(self):
        arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        img = da.from_array(arr, chunks=(64, 64))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=2, scale_steps=2)
        cropped = crop_dask_array_or_pyramid(pyramid, xlim=(0, 32), ylim=(0, 32), pixel_size=1.0)
        assert isinstance(cropped, list)
        assert len(cropped) == len(pyramid)

    def test_out_of_bounds_pyramid_crop_raises(self):
        size = 64
        arr = np.zeros((size, size), dtype=np.uint8)
        img = da.from_array(arr, chunks=(size, size))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=2, scale_steps=2)
        with pytest.raises(NoImageOverlapError):
            crop_dask_array_or_pyramid(
                pyramid, xlim=(1000, 2000), ylim=(1000, 2000), pixel_size=1.0
            )

    def test_small_region_crops_every_pyramid_level(self):
        # A 20x20 px region truncates to 0 px on the coarsest levels of a
        # 7-level pyramid (64x downscale); it overlaps the image, so it must
        # not raise and every level must stay non-empty.
        size = 512
        img = da.from_array(np.zeros((size, size), dtype=np.uint8), chunks=(size, size))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=6, scale_steps=2)
        assert len(pyramid) == 7

        cropped = crop_dask_array_or_pyramid(pyramid, xlim=(100, 120), ylim=(100, 120), pixel_size=1.0)

        assert len(cropped) == len(pyramid)
        assert cropped[0].shape == (20, 20)
        assert all(level.shape[0] >= 1 and level.shape[1] >= 1 for level in cropped)

    def test_small_region_at_far_edge_stays_in_bounds(self):
        size = 512
        img = da.from_array(np.zeros((size, size), dtype=np.uint8), chunks=(size, size))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=6, scale_steps=2)

        cropped = crop_dask_array_or_pyramid(pyramid, xlim=(505, 512), ylim=(505, 512), pixel_size=1.0)

        assert cropped[0].shape == (7, 7)
        assert all(level.shape[0] >= 1 and level.shape[1] >= 1 for level in cropped)

    def test_pyramid_crop_multilevel_alignment_fractional_bounds(self):
        # Regression test for compounded truncation across pyramid levels (F2).
        # size=201 is not evenly divisible by scale_steps=2, so each level's
        # step scale factor is non-integer (e.g. 201/101 ~= 1.99) - exactly the
        # condition that exposes the compounding bug, since nested int()
        # truncation at each step then no longer equals a single truncation
        # applied once to the original physical limits.
        size = 201
        arr = np.zeros((size, size), dtype=np.uint8)
        img = da.from_array(arr, chunks=(size, size))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=3, scale_steps=2)
        level_sizes = [p.shape[0] for p in pyramid]
        assert level_sizes == [201, 101, 51, 26]

        pixel_size = 1.0
        xlim = (30.93, 121.4)
        ylim = (30.93, 121.4)

        cropped = crop_dask_array_or_pyramid(pyramid, xlim=xlim, ylim=ylim, pixel_size=pixel_size)

        x0_full, x1_full = xlim
        y0_full, y1_full = ylim
        for level, (orig, crop) in enumerate(zip(pyramid, cropped, strict=True)):
            sf_x = pyramid[0].shape[1] / orig.shape[1]
            sf_y = pyramid[0].shape[0] / orig.shape[0]
            expected_x0 = max(0, int(x0_full / sf_x))
            expected_x1 = min(orig.shape[1], int(x1_full / sf_x))
            expected_y0 = max(0, int(y0_full / sf_y))
            expected_y1 = min(orig.shape[0], int(y1_full / sf_y))
            assert crop.shape == (expected_y1 - expected_y0, expected_x1 - expected_x0), (
                f"level {level} misaligned: shape not derived fresh from physical limits"
            )

    def test_pyramid_crop_matches_fixed_formula_not_compounded_truncation(self):
        # Explicit divergence check: for these fractional bounds, the old
        # cumulative-reassignment ("buggy") formula and the fixed
        # from-physical-limits-each-time formula give different pixel bounds
        # at levels 1 and 3. Pins the implementation to the fixed formula and
        # would fail if the compounding bug were reintroduced.
        size = 201
        arr = np.zeros((size, size), dtype=np.uint8)
        img = da.from_array(arr, chunks=(size, size))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=3, scale_steps=2)

        pixel_size = 1.0
        xlim = (30.93, 121.4)
        ylim = (30.93, 121.4)

        cropped = crop_dask_array_or_pyramid(pyramid, xlim=xlim, ylim=ylim, pixel_size=pixel_size)

        # reproduce the pre-fix cumulative-reassignment computation verbatim
        scale_factors = [1] + [
            pyramid[i].shape[0] / pyramid[i + 1].shape[0] for i in range(len(pyramid) - 1)
        ]
        xlim_scaled = (xlim[0] / pixel_size, xlim[1] / pixel_size)
        buggy_bounds = []
        for sf in scale_factors:
            xlim_scaled = (int(xlim_scaled[0] / sf), int(xlim_scaled[1] / sf))
            buggy_bounds.append(xlim_scaled)

        # fixed formula, computed independently of the implementation
        fixed_bounds = []
        for img_level in pyramid:
            sf_x = pyramid[0].shape[1] / img_level.shape[1]
            fixed_bounds.append((int(xlim[0] / sf_x), int(xlim[1] / sf_x)))

        # sanity: these parameters actually exercise the bug (levels 1 and 3
        # differ between the two formulas) - otherwise this test would not be
        # exercising the regression it claims to guard
        assert buggy_bounds[1] != fixed_bounds[1]
        assert buggy_bounds[3] != fixed_bounds[3]

        # the implementation's crop width matches the fixed formula at every
        # level, not the buggy compounded one
        for level, (crop, (fx0, fx1)) in enumerate(zip(cropped, fixed_bounds, strict=True)):
            assert crop.shape[1] == fx1 - fx0, f"level {level} used compounded truncation"


# ── Format v2: sharded Zarr v3 + NGFF metadata ────────────────────────────────

@pytest.fixture
def small_layout(monkeypatch):
    """Shrink the layout constants so multi-shard behaviour is exercised on tiny arrays."""
    from insitupy.images import _zarr_layout

    monkeypatch.setattr(_zarr_layout, "ZARR_SHARD_SIZE", 64)
    monkeypatch.setattr(_zarr_layout, "ZARR_CHUNK_SIZE", 16)
    monkeypatch.setattr(_zarr_layout, "DEFAULT_CHUNK_SIZE_X", 64)
    monkeypatch.setattr(_zarr_layout, "DEFAULT_CHUNK_SIZE_Y", 64)


def _rgb_array(height=300, width=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _rgb_metadata(pixel_size=0.5, shape=(300, 200, 3)):
    return {"OME": {}, "axes": "YXS", "pixel_size": pixel_size, "shape": shape, "rgb": True}


class TestShardedNgffLayout:
    def test_rgb_roundtrip_layout_and_metadata(self, tmp_path, small_layout):
        import zarr

        from insitupy.containers.image_data import ImageData

        arr = _rgb_array()
        img = da.from_array(arr, chunks=(300, 200, 3))
        out = tmp_path / "he.zarr"
        write_zarr(image=img, file=out, img_metadata=_rgb_metadata(0.5), axes="YXS")

        # in memory: YXS with identical values on every level
        data = ImageData()
        data.add_image(out, channel_names="he", verbose=False)
        assert data.metadata["he"]["axes"] == "YXS"
        expected = create_img_pyramid(img, axes="YXS", nsubres=6, scale_steps=2)
        assert len(data["he"]) == len(expected)
        for got, want in zip(data["he"], expected, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

        # on disk: channel-first, sharded
        root = zarr.open_group(out, mode="r")
        level0 = root["0"]
        assert level0.shape == (3, 300, 200)
        assert level0.chunks == (3, 16, 16)
        assert level0.shards == (3, 64, 64)
        assert root.attrs["axes"] == "SYX"
        assert root.attrs["rgb"] is True
        assert tuple(root.attrs["shape"]) == (3, 300, 200)

        ome = root.attrs["ome"]
        ms = ome["multiscales"][0]
        assert [a["name"] for a in ms["axes"]] == ["c", "y", "x"]
        assert [a.get("unit") for a in ms["axes"]] == [None, "micrometer", "micrometer"]
        for ds, lvl in zip(ms["datasets"], expected, strict=True):
            h, w = lvl.shape[0], lvl.shape[1]
            scale = ds["coordinateTransformations"][0]["scale"]
            assert scale == pytest.approx([1.0, 0.5 * 300 / h, 0.5 * 200 / w])
        channels = ome["omero"]["channels"]
        assert [c["color"] for c in channels] == ["FF0000", "00FF00", "0000FF"]
        assert all(c["window"]["start"] == 0 and c["window"]["end"] == 255 for c in channels)

    def test_misaligned_input_chunks_are_stored_exactly(self, tmp_path, small_layout):
        # input chunks straddle shard borders: two tasks writing one shard would corrupt it
        arr = _rgb_array(seed=1)
        img = da.from_array(arr, chunks=(37, 53, 3))
        out = tmp_path / "he.zarr"
        write_zarr(image=img, file=out, img_metadata=_rgb_metadata(), axes="YXS")

        result, _, axes_out, _ = read_zarr(out)
        assert axes_out == "YXS"
        np.testing.assert_array_equal(np.asarray(result[0]), arr)

    @pytest.mark.parametrize("name", ["he.zarr", "he.zarr.zip"])
    def test_single_array_rgb_roundtrip(self, tmp_path, small_layout, name):
        # no pyramid: the root array is stored channel-first and must come back as YXS
        arr = _rgb_array(seed=4)
        out = tmp_path / name
        write_zarr(image=da.from_array(arr), file=out, img_metadata=_rgb_metadata(), axes="YXS",
                   save_pyramid=False)

        result, _, axes_out, _ = read_zarr(out)
        assert not isinstance(result, list)
        assert axes_out == "YXS"
        np.testing.assert_array_equal(np.asarray(result), arr)

    def test_reads_use_shard_sized_dask_chunks(self, tmp_path, small_layout, monkeypatch):
        from insitupy.images import _zarr_layout

        out = tmp_path / "he.zarr"
        write_zarr(image=da.from_array(_rgb_array()), file=out, img_metadata=_rgb_metadata(), axes="YXS")
        result, _, _, _ = read_zarr(out)
        assert result[0].chunksize == (64, 64, 3)

        # plain-chunk store: read in the working block, not in the 16 pixel stored chunks
        monkeypatch.setattr(_zarr_layout, "ZARR_SHARD_SIZE", None)
        plain = tmp_path / "plain.zarr"
        write_zarr(image=da.from_array(_rgb_array()), file=plain, img_metadata=_rgb_metadata(), axes="YXS")
        result, _, _, _ = read_zarr(plain)
        assert result[0].chunksize == (64, 64, 3)

    def test_grayscale_window_ignores_zeros_and_outliers(self, tmp_path, small_layout):
        import zarr

        rng = np.random.default_rng(2)
        arr = np.zeros((300, 200), dtype=np.uint16)
        arr[50:250, 50:150] = rng.integers(1, 1000, (200, 100))
        out = tmp_path / "g.zarr"
        write_zarr(image=da.from_array(arr), file=out,
                   img_metadata={"OME": {}, "axes": "YX", "pixel_size": 1.0}, axes="YX")

        window = zarr.open_group(out, mode="r").attrs["ome"]["omero"]["channels"][0]["window"]
        assert 0 < window["end"] <= 1000
        assert window["start"] == 0
        assert window["min"] == 0
        assert window["max"] == 65535

    def test_all_zero_grayscale_writes_valid_window(self, tmp_path, small_layout):
        import zarr

        out = tmp_path / "g.zarr"
        write_zarr(image=da.zeros((300, 200), dtype=np.uint16, chunks=(300, 200)), file=out,
                   img_metadata={"OME": {}, "axes": "YX", "pixel_size": 1.0}, axes="YX")
        window = zarr.open_group(out, mode="r").attrs["ome"]["omero"]["channels"][0]["window"]
        assert window["start"] < window["end"] <= window["max"]

    def test_caller_metadata_is_not_mutated(self, tmp_path, small_layout):
        from insitupy.containers.image_data import ImageData

        data = ImageData()
        data.add_image(da.from_array(_rgb_array()), channel_names="he", axes="YXS", pixel_size=0.5,
                       verbose=False)
        data.save(tmp_path / "imgs")
        assert data.metadata["he"]["axes"] == "YXS"
        assert tuple(data.metadata["he"]["shape"]) == (300, 200, 3)

    def test_ome_zarr_reader_opens_rgb_and_grayscale(self, tmp_path, small_layout):
        pytest.importorskip("ome_zarr")
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader

        rgb = _rgb_array()
        gray = np.arange(300 * 200, dtype=np.uint16).reshape(300, 200)
        write_zarr(image=da.from_array(rgb), file=tmp_path / "he.zarr", img_metadata=_rgb_metadata(), axes="YXS")
        write_zarr(image=da.from_array(gray), file=tmp_path / "g.zarr",
                   img_metadata={"OME": {}, "axes": "YX", "pixel_size": 1.0}, axes="YX")

        for name, want in (("he.zarr", np.moveaxis(rgb, -1, 0)), ("g.zarr", gray)):
            nodes = list(Reader(parse_url(tmp_path / name))())
            assert len(nodes) == 1
            assert {"Multiscales", "OMERO"} <= {type(s).__name__ for s in nodes[0].specs}
            assert len(nodes[0].data) == 7
            np.testing.assert_array_equal(np.asarray(nodes[0].data[0]), want)

    def test_shipped_defaults_shard_4096_chunk_512(self, tmp_path):
        import zarr

        out = tmp_path / "he.zarr"
        write_zarr(image=da.from_array(_rgb_array(600, 500)), file=out,
                   img_metadata=_rgb_metadata(shape=(600, 500, 3)), axes="YXS")
        level0 = zarr.open_group(out, mode="r")["0"]
        assert level0.shards == (3, 4096, 4096)
        assert level0.chunks == (3, 512, 512)

    def test_zip_store_has_plain_chunks(self, tmp_path, small_layout):
        import zarr

        arr = _rgb_array()
        out = tmp_path / "he.zarr.zip"
        write_zarr(image=da.from_array(arr), file=out, img_metadata=_rgb_metadata(), axes="YXS")

        with zarr.storage.ZipStore(out, mode="r") as store:
            level0 = zarr.open_group(store, mode="r")["0"]
            assert level0.shards is None
            assert level0.chunks == (3, 16, 16)
        result, _, axes_out, _ = read_zarr(out)
        assert axes_out == "YXS"
        np.testing.assert_array_equal(np.asarray(result[0]), arr)

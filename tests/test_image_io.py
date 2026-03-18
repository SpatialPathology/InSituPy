"""Image IO roundtrip tests and utility tests.

Covers:
- Priority 1: write_ome_tiff, write_zarr roundtrips (data corruption risk)
- Priority 5: im.create_img_pyramid, crop_dask_array_or_pyramid, read_image
"""

import numpy as np
import dask.array as da
import pytest

from insitupy.images import read_image, read_ome_tiff, read_zarr, write_ome_tiff, write_zarr
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

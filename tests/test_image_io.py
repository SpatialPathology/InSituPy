"""Image IO roundtrip tests and utility tests.

Covers:
- Priority 1: write_ome_tiff, write_zarr roundtrips (data corruption risk)
- Priority 5: im.create_img_pyramid, crop_dask_array_or_pyramid, read_image
"""

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

    def test_out_of_bounds_pyramid_crop_raises(self):
        size = 64
        arr = np.zeros((size, size), dtype=np.uint8)
        img = da.from_array(arr, chunks=(size, size))
        pyramid = create_img_pyramid(img, axes="YX", nsubres=2, scale_steps=2)
        with pytest.raises(NoImageOverlapError):
            crop_dask_array_or_pyramid(
                pyramid, xlim=(1000, 2000), ylim=(1000, 2000), pixel_size=1.0
            )

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

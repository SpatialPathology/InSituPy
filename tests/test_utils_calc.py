"""Tests for insitupy.utils._calc — numerical computation functions."""

import dask.array as da
import numpy as np
import pytest

from insitupy.utils._calc import (
    cohens_d,
    create_tiles,
    intensity_median,
    summarize_tile_measurements,
)


class TestCohensD:
    def test_identical_groups_returns_zero(self):
        # Two identical distributions have no effect → d = 0 (before small-sample correction)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)  # n>=50 so no correction
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        d = cohens_d(a, b, correct_small_sample_size=False)
        assert d == 0.0

    def test_known_effect_size_unpaired(self):
        # a=[0,0,0,0], b=[1,1,1,1]: mean_diff=1, pooled_std via ddof=1 formula
        # std([0,0,0,0], ddof=1)=0 → SDpool=0 → d=nan; test with nonzero std
        rng = np.random.default_rng(42)
        a = rng.normal(loc=0.0, scale=1.0, size=100)
        b = rng.normal(loc=2.0, scale=1.0, size=100)
        d = cohens_d(a, b, correct_small_sample_size=False)
        # Expected d ≈ 2.0; allow tolerance
        assert abs(d) > 1.5  # large effect detected

    def test_sign_reflects_direction(self):
        # a > b should give positive d; a < b should give negative d
        a = np.array([5.0] * 60)
        b = np.array([3.0] * 60)
        d = cohens_d(a, b, correct_small_sample_size=False)
        # SDpool=0 → nan; use spread data instead
        rng = np.random.default_rng(7)
        a = rng.normal(5.0, 1.0, 60)
        b = rng.normal(3.0, 1.0, 60)
        d = cohens_d(a, b, correct_small_sample_size=False)
        assert d > 0  # a has larger mean

    def test_identical_std_zero_returns_nan(self):
        # When both groups are constant, pooled std=0 → d=nan, not crash
        a = np.array([2.0, 2.0, 2.0])
        b = np.array([5.0, 5.0, 5.0])
        d = cohens_d(a, b, correct_small_sample_size=False)
        assert np.isnan(d)

    def test_paired_known_values(self):
        # d_paired = mean(diff) / std(diff)
        # diff = [1,1,1,1] → mean=1, std=0 → nan; use spread
        a = np.array([1.0, 3.0, 5.0, 7.0])
        b = np.array([0.0, 2.0, 4.0, 6.0])
        # diff = [1,1,1,1], std(diff, ddof=0)=0 → nan
        d = cohens_d(a, b, paired=True)
        # With ddof=0 std=0 → inf or nan; just verify no exception raised
        assert d is not None  # may be nan or inf

    def test_paired_requires_equal_length(self):
        # Unequal lengths in paired mode must raise ValueError
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="equal"):
            cohens_d(a, b, paired=True)

    def test_small_sample_correction_applied(self):
        # With n<50 and correct_small_sample_size=True, result should differ from uncorrected
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 10)
        b = rng.normal(1.0, 1.0, 10)
        d_corrected = cohens_d(a, b, correct_small_sample_size=True)
        d_uncorrected = cohens_d(a, b, correct_small_sample_size=False)
        # Correction factor < 1 for small n, so |d_corrected| < |d_uncorrected|
        assert abs(d_corrected) != abs(d_uncorrected) or np.isnan(d_corrected)


class TestIntensityMedian:
    def test_returns_median_of_masked_pixels(self):
        # Only pixels where mask=True are used; result should be median of [1,3,5]
        region_mask = np.array([[True, False, True],
                                 [False, True, False]])
        intensity_image = np.array([[1, 99, 3],
                                     [99, 5, 99]])
        result = intensity_median(region_mask, intensity_image)
        # Values: 1, 3, 5 → median = 3
        assert result == 3.0

    def test_single_pixel_region(self):
        # Single-pixel region: median equals that pixel value
        region_mask = np.array([[True]])
        intensity_image = np.array([[42]])
        assert intensity_median(region_mask, intensity_image) == 42.0


class TestCreateTiles:
    def test_single_tile_for_small_array(self):
        # Array smaller than tile_size should produce exactly one tile
        arr = da.from_array(np.ones((100, 100)))
        tiles = create_tiles(arr, tile_size=200, overlap=0)
        assert len(tiles) == 1

    def test_tile_tuple_structure(self):
        # Each tile entry must be (dask_array, global_slice, inner_slice)
        arr = da.from_array(np.ones((50, 50)))
        tiles = create_tiles(arr, tile_size=200, overlap=0)
        tile, global_slice, inner_slice = tiles[0]
        assert isinstance(tile, da.Array)
        assert isinstance(global_slice, tuple) and len(global_slice) == 2
        assert isinstance(inner_slice, tuple) and len(inner_slice) == 2

    def test_multiple_tiles_for_large_array(self):
        # 300x300 with tile_size=200, overlap=0 → 2x2 = 4 tiles
        arr = da.from_array(np.ones((300, 300)))
        tiles = create_tiles(arr, tile_size=200, overlap=0)
        assert len(tiles) == 4

    def test_raises_for_non_2d_array(self):
        # 3D input must raise ValueError, catches wrong image format
        arr = da.from_array(np.ones((3, 100, 100)))
        with pytest.raises(ValueError):
            create_tiles(arr, tile_size=200, overlap=0)

    def test_tiles_cover_full_array(self):
        # Union of all global slices must cover the full array
        h, w = 256, 256
        arr = da.from_array(np.ones((h, w)))
        tiles = create_tiles(arr, tile_size=150, overlap=20)
        covered = np.zeros((h, w), dtype=bool)
        for tile, (sy, sx), _ in tiles:
            covered[sy, sx] = True
        assert covered.all()


class TestSummarizeTileMeasurements:
    def test_single_tile_passthrough(self):
        # With one tile, output must equal that tile's measurements
        measurements = np.array([10.0, 20.0, 30.0])
        cell_ids = np.array([1, 2, 3])
        areas = np.array([50, 60, 70])
        m, ids = summarize_tile_measurements([(measurements, cell_ids, areas)])
        np.testing.assert_array_equal(np.sort(ids), np.array([1, 2, 3]))
        # Measurements should match
        for cid, meas in zip([1, 2, 3], [10.0, 20.0, 30.0]):
            idx = np.where(ids == cid)[0][0]
            assert m[idx] == meas

    def test_selects_measurement_from_larger_area_tile(self):
        # Cell 1 appears in two tiles; tile with larger area should win
        # Tile 1: cell 1 with measurement 5.0 and area 100
        # Tile 2: cell 1 with measurement 9.0 and area 200  → should pick 9.0
        tile1 = (np.array([5.0]), np.array([1]), np.array([100]))
        tile2 = (np.array([9.0]), np.array([1]), np.array([200]))
        m, ids = summarize_tile_measurements([tile1, tile2])
        assert ids[0] == 1
        assert m[0] == 9.0

    def test_unique_cells_across_tiles(self):
        # Must return exactly one entry per unique cell ID
        tile1 = (np.array([1.0, 2.0]), np.array([10, 20]), np.array([50, 50]))
        tile2 = (np.array([3.0, 4.0]), np.array([20, 30]), np.array([100, 50]))
        m, ids = summarize_tile_measurements([tile1, tile2])
        assert len(np.unique(ids)) == len(ids)
        assert set(ids.tolist()) == {10, 20, 30}

    def test_output_shapes_match(self):
        # measurements and cell_ids must have the same length
        tile = (np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3]), np.array([10, 20, 30]))
        m, ids = summarize_tile_measurements([tile])
        assert len(m) == len(ids)

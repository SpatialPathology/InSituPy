"""Tests for `_write_dask_array_to_zarr`: values survive the write and no dask warning is raised."""

import warnings

import dask.array as da
import numpy as np
import pytest
import zarr
from dask.array.core import PerformanceWarning

from insitupy.containers._zarr_compat import _get_zarr_store, _write_dask_array_to_zarr


def _roundtrip(tmp_path, arr):
    store = _get_zarr_store(tmp_path / "t.zarr", mode="w")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _write_dask_array_to_zarr(store, "m/0", arr)
    return zarr.open(store, mode="r")["m/0"][:]


@pytest.mark.parametrize("shape", [(50, 150), (64, 150), (100, 64), (200, 300)])
def test_2d_array_wider_than_one_chunk_writes_without_warning(tmp_path, shape):
    # Mimics a small pyramid level (e.g. 3223 x 4427 with 4096 chunks): the whole array is below
    # dask's auto chunk size but its width is not a multiple of the zarr chunk. The shapes use a
    # 64-pixel chunk so the test stays tiny; the warning depends on the ratio, not on the size.
    data = np.random.default_rng(0).integers(0, 2**31, size=shape, dtype=np.uint32)
    arr = da.from_array(data, chunks=shape).rechunk((64, 64))

    np.testing.assert_array_equal(_roundtrip(tmp_path, arr), data)


@pytest.mark.parametrize("length", [0, 1, 1000])
def test_1d_array_roundtrip(tmp_path, length):
    # zero-length arrays are the reason the zarr chunk edge is clamped to >= 1
    data = np.arange(length, dtype=np.uint32)
    arr = da.from_array(data, chunks=(max(1, length),))

    np.testing.assert_array_equal(_roundtrip(tmp_path, arr), data)


def test_irregular_chunks_are_aligned_before_writing(tmp_path):
    # the zarr chunk is the first dask chunk (30), the later ones (7, 63) are not multiples of it:
    # the write must still put every value in the right place
    data = np.arange(100, dtype=np.uint32)
    arr = da.from_array(data, chunks=((30, 7, 63),))

    np.testing.assert_array_equal(_roundtrip(tmp_path, arr), data)

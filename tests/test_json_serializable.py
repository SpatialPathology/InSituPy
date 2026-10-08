"""
Regression tests for the JSON-serialization fix in images/io.write_zarr and
the make_json_serializable helper in utils/utils.
"""
import json
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import zarr

from insitupy.images.io import write_zarr
from insitupy.utils.utils import make_json_serializable

# ---------------------------------------------------------------------------
# Tests for make_json_serializable
# ---------------------------------------------------------------------------

def test_make_json_serializable_numpy_scalar():
    result = make_json_serializable(np.int64(42))
    assert result == 42
    assert isinstance(result, int)
    # Verify json.dumps does not raise
    json.dumps(result)


def test_make_json_serializable_numpy_float():
    result = make_json_serializable(np.float32(3.14))
    assert isinstance(result, float)
    json.dumps(result)


def test_make_json_serializable_numpy_array():
    arr = np.array([1, 2, 3], dtype=np.int64)
    result = make_json_serializable(arr)
    assert result == [1, 2, 3]
    json.dumps(result)


def test_make_json_serializable_nested_dict():
    data = {"a": np.int64(1), "b": {"c": np.float32(2.5)}}
    result = make_json_serializable(data)
    json.dumps(result)
    assert result["a"] == 1
    assert abs(result["b"]["c"] - 2.5) < 1e-3


def test_make_json_serializable_passthrough():
    for v in [None, True, False, 1, 2.0, "hello", [1, 2], {"k": "v"}]:
        result = make_json_serializable(v)
        json.dumps(result)


def test_make_json_serializable_path(tmp_path):
    result = make_json_serializable(tmp_path)
    assert isinstance(result, str)
    json.dumps(result)


# ---------------------------------------------------------------------------
# Tests for write_zarr with numpy-typed metadata
# ---------------------------------------------------------------------------

def test_write_zarr_with_numpy_scalar_metadata():
    """write_zarr must succeed and produce valid Zarr attrs when metadata
    contains numpy scalar values (pre-fix this would raise TypeError)."""
    img = da.from_array(np.zeros((3, 64, 64), dtype=np.uint8))
    metadata = {
        "pixel_size": np.float64(0.2125),
        "axes": "CYX",
        "channel_names": ["DAPI", "GFP", "RFP"],
        "shape": np.array([3, 64, 64]),
        "rgb": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test.zarr"
        write_zarr(
            image=img,
            file=outpath,
            img_metadata=metadata,
            save_pyramid=False,
            axes="CYX",
            verbose=False,
            overwrite=False,
        )

        # Verify the zarr store was written
        assert outpath.exists()

        # Load and verify attrs are JSON-serializable
        store = zarr.open(str(outpath), mode="r")
        attrs = dict(store.attrs)
        serialized = json.dumps(attrs)   # must not raise
        assert len(serialized) > 0

        # Values must have been converted to native Python types
        assert isinstance(attrs["pixel_size"], float)


def test_write_zarr_attrs_contain_only_json_serializable_values():
    """All values in written Zarr attrs must be natively JSON-serializable."""
    img = da.from_array(np.zeros((1, 32, 32), dtype=np.uint16))
    metadata = {
        "pixel_size": np.float32(0.5),
        "axes": "CYX",
        "channel_names": np.array(["DAPI"]),
        "shape": np.array([1, 32, 32]),
        "rgb": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "test2.zarr"
        write_zarr(
            image=img,
            file=outpath,
            img_metadata=metadata,
            save_pyramid=False,
            axes="CYX",
            verbose=False,
            overwrite=False,
        )

        store = zarr.open(str(outpath), mode="r")
        attrs = dict(store.attrs)

        # Every value must be serializable with the standard json library
        for key, val in attrs.items():
            try:
                json.dumps(val)
            except TypeError as exc:
                raise AssertionError(
                    f"Zarr attr '{key}' is not JSON-serializable: {exc}"
                ) from exc

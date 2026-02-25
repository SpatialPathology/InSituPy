import numpy as np
import zarr


def test_can_open_v2_store_with_zarr_v3(tmp_path):
    """Smoke test: zarr v3 runtime can still read a zarr v2 store."""
    assert int(zarr.__version__.split(".")[0]) >= 3

    store_path = tmp_path / "smoke_v2.zarr"
    root = zarr.open_group(store_path, mode="w", zarr_format=2)
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    root.create_array("0", data=values, chunks=(2, 2))

    reopened = zarr.open_group(store_path, mode="r")
    arr = reopened["0"][:]

    np.testing.assert_array_equal(arr, values)

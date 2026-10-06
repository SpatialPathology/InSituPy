from __future__ import annotations

import dask.array as da
import zarr

# Detect Zarr version for compatibility
ZARR_V3 = hasattr(zarr.storage, 'LocalStore')


def _get_zarr_store(path, mode: str = "r", zipped: bool = False):
    """
    Get a Zarr store compatible with both Zarr v2 and v3.

    Args:
        path: Path to the zarr store
        mode: Mode to open the store ('r', 'w', 'a')
        zipped: Whether the store is a ZipStore

    Returns:
        For Zarr v3: store object (no context manager needed)
        For Zarr v2: store object (should be used as context manager)
    """
    if ZARR_V3:
        # Zarr v3 API
        if zipped:
            return zarr.storage.ZipStore(path, mode=mode)
        else:
            return zarr.storage.LocalStore(path)
    else:
        # Zarr v2 API
        if zipped:
            return zarr.ZipStore(path, mode=mode)
        else:
            return zarr.DirectoryStore(path)


def _write_dask_array_to_zarr(store, name: str, arr) -> None:
    """
    Write a dask array to `name` within `store`, creating (or overwriting) the
    destination zarr array explicitly via the stable `zarr`-level API rather
    than dask's `to_zarr(..., zarr_array_kwargs=...)` kwarg-forwarding path.

    Background: dask's `Array.to_zarr()` only had `zarr_array_kwargs` as a
    real, explicitly-named parameter in dask 2025.12.0-2026.1.1. Outside that
    window it is just the *name* dask happens to forward through its trailing
    `**kwargs` catch-all straight into `zarr.create_array()`/`zarr.create()`,
    which raises `TypeError: create_array() got an unexpected keyword argument
    'zarr_array_kwargs'` because no such parameter exists there.

    Creating the zarr array ourselves and writing into it with `da.store`
    sidesteps this entirely: no `zarr_array_kwargs`/`mode`/`**kwargs` handling
    is involved, so it is stable across dask versions before, during, and
    after the broken window. The array is rechunked to the zarr chunk grid
    first, which also avoids the false "risk of data loss" PerformanceWarning
    that `to_zarr(<zarr.Array>)` raises in dask >= 2025.11 (see the comment at
    the write below).
    """
    # clamp each chunk edge to >= 1: zarr rejects a zero-length chunk edge, so a
    # zero-length array (an empty nucleus_to_cell_map, or a zero-cell CellData)
    # would otherwise raise "integer chunk edge length must be >= 1, got 0".
    chunks = tuple(max(1, c[0]) for c in arr.chunks)
    if ZARR_V3:
        z = zarr.create_array(
            store=store,
            name=name,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chunks,
            overwrite=True,
        )
    else:
        z = zarr.create(
            store=store,
            path=name,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chunks,
            overwrite=True,
        )
    # One dask chunk per zarr chunk, so no two tasks ever write the same zarr chunk and
    # `lock=False` is safe. `arr.to_zarr(z)` would instead re-chunk to "auto" sizes: for an
    # array below `array.chunk-size` that is wider than one zarr chunk (e.g. 3223 x 4427 with
    # 3223 x 4096 chunks) that is a single chunk not divisible by the zarr chunk, and dask
    # (>= 2025.11) emits a PerformanceWarning about "risk of data loss" although one chunk
    # spanning the whole axis cannot race.
    da.store(arr.rechunk(chunks), z, lock=False)

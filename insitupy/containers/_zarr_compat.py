from __future__ import annotations

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

    Passing an existing `zarr.Array` as `to_zarr`'s `url` argument instead
    sidesteps this entirely: dask detects `isinstance(url, zarr.Array)` before
    any `zarr_array_kwargs`/`mode`/`**kwargs` handling and writes directly
    into the given array. That branch is unchanged since 2024, so it is
    stable across dask versions before, during, and after the broken window.
    """
    chunks = tuple(c[0] for c in arr.chunks)
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
    arr.to_zarr(z)

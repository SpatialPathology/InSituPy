"""
Test script to explore and validate `get_zarr_source_path` — a function that
extracts the source Zarr store path from a lazily-loaded dask array.

This helps us understand how dask internally stores references to Zarr stores
so we can detect when source == target and prevent data corruption on overwrite.
"""
import shutil
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import zarr

print(f"Zarr version: {zarr.__version__}")


# ---------------------------------------------------------------------------
# 1. Candidate implementation of get_zarr_source_path
# ---------------------------------------------------------------------------
def get_zarr_source_path(arr) -> Path | None:
    """
    Attempt to extract the source Zarr store path from a dask array
    that was created via `da.from_zarr()`.

    For image pyramids (list of dask arrays), checks the first level.

    Returns:
        Path: The resolved path to the source Zarr store, or None if
              the source cannot be determined (e.g., array was created
              from in-memory data or was .persist()-ed).
    """
    # Handle list of dask arrays (pyramids)
    if isinstance(arr, list):
        for a in arr:
            result = get_zarr_source_path(a)
            if result is not None:
                return result
        return None

    if not isinstance(arr, da.Array):
        return None

    graph = arr.__dask_graph__()

    if not hasattr(graph, 'layers'):
        return None

    # Look for the 'original-from-zarr-*' MaterializedLayer that dask creates
    # when loading via da.from_zarr(). Its value is a zarr.core.array.Array.
    for layer_name, layer in graph.layers.items():
        mapping = getattr(layer, 'mapping', None)
        if mapping is None:
            # Try the layer itself as a mapping
            if hasattr(layer, 'items'):
                mapping = layer
            else:
                continue

        try:
            items = list(mapping.items())
        except Exception:
            continue

        for key, val in items:
            # The zarr array is stored directly as the value (not in a tuple)
            # in the 'original-from-zarr-*' layer
            path = _extract_path_from_zarr_array(val)
            if path is not None:
                return path

            # Also check inside tuples (some dask versions)
            if isinstance(val, tuple):
                for v in val:
                    path = _extract_path_from_zarr_array(v)
                    if path is not None:
                        return path

    return None


def _extract_path_from_zarr_array(v) -> Path | None:
    """Try to extract a Zarr store path from a zarr.core.array.Array or store object."""

    # Primary case: zarr.core.array.Array (both v2 and v3)
    if isinstance(v, zarr.Array):
        store = v.store
        return _extract_path_from_store(store)

    return None


def _extract_path_from_store(store) -> Path | None:
    """Extract a filesystem path from a zarr store object."""

    # zarr v3: LocalStore
    if isinstance(store, zarr.storage.LocalStore):
        root = getattr(store, 'root', None)
        if root is not None:
            return Path(str(root)).resolve()
        return Path(str(store)).resolve()

    # Generic fallback: .path attribute
    if hasattr(store, 'path'):
        try:
            return Path(store.path).resolve()
        except Exception:
            pass

    # Generic fallback: .root attribute
    if hasattr(store, 'root'):
        try:
            return Path(str(store.root)).resolve()
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# 2. Test cases
# ---------------------------------------------------------------------------
def create_test_zarr(path: Path, shape=(100, 100), chunks=(50, 50)):
    """Create a simple test zarr store with random data."""
    data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    z = zarr.open(str(path), mode='w', shape=shape, chunks=chunks, dtype=np.uint8)
    z[:] = data
    z.attrs['OME'] = {}
    z.attrs['axes'] = 'YX'
    z.attrs['pixel_size'] = 0.5
    return data


def create_test_zarr_pyramid(path: Path, shape=(100, 100), chunks=(50, 50)):
    """Create a test zarr store with pyramid (group with sub-arrays)."""
    data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    root = zarr.open_group(str(path), mode='w')
    root.create_array('0', data=data, chunks=chunks)
    half = data[::2, ::2]
    root.create_array('1', data=half, chunks=chunks)
    root.attrs['OME'] = {}
    root.attrs['axes'] = 'YX'
    root.attrs['pixel_size'] = 0.5
    return data


def test_flat_zarr_array():
    """Test: dask array from a flat (non-pyramid) zarr store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.zarr"
        create_test_zarr(zarr_path)

        arr = da.from_zarr(str(zarr_path))

        result = get_zarr_source_path(arr)
        print(f"  get_zarr_source_path -> {result}")
        print(f"  Expected             -> {zarr_path.resolve()}")
        if result is not None:
            assert result == zarr_path.resolve(), f"MISMATCH: {result} != {zarr_path.resolve()}"
            print("  PASS")
        else:
            print("  FAIL (returned None)")


def test_pyramid_zarr_array():
    """Test: dask array from a pyramid zarr store (component='0')."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "pyramid.zarr"
        create_test_zarr_pyramid(zarr_path)

        # Load as InSituPy does: da.from_zarr with LocalStore
        store = zarr.storage.LocalStore(zarr_path)
        arr = da.from_zarr(store, component='0')

        result = get_zarr_source_path(arr)
        expected = zarr_path.resolve()
        print(f"  get_zarr_source_path -> {result}")
        print(f"  Expected (root)      -> {expected}")
        if result is not None:
            # The path may include the component subdir, so check containment
            if result == expected or str(expected) in str(result) or str(result) in str(expected):
                print("  PASS")
            else:
                print(f"  PARTIAL (got: {result})")
        else:
            print("  FAIL (returned None)")


def test_pyramid_as_list():
    """Test: list of dask arrays from a pyramid zarr store (like InSituPy loads)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "pyramid.zarr"
        create_test_zarr_pyramid(zarr_path)

        store = zarr.storage.LocalStore(zarr_path)
        img_list = [
            da.from_zarr(store, component='0'),
            da.from_zarr(store, component='1'),
        ]

        result = get_zarr_source_path(img_list)
        expected = zarr_path.resolve()
        print(f"  get_zarr_source_path -> {result}")
        print(f"  Expected (root)      -> {expected}")
        if result is not None:
            if result == expected or str(expected) in str(result) or str(result) in str(expected):
                print("  PASS")
            else:
                print(f"  PARTIAL (got: {result})")
        else:
            print("  FAIL (returned None)")


def test_in_memory_array():
    """Test: dask array from in-memory numpy data (no zarr source)."""
    data = np.random.rand(100, 100)
    arr = da.from_array(data, chunks=(50, 50))

    result = get_zarr_source_path(arr)
    print(f"  get_zarr_source_path -> {result}")
    print(f"  Expected             -> None")
    assert result is None, f"Expected None but got {result}"
    print("  PASS")


def test_computed_array():
    """Test: dask array derived from a zarr-backed array via operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.zarr"
        create_test_zarr(zarr_path)

        arr = da.from_zarr(str(zarr_path))
        transformed = arr * 2 + 1

        result = get_zarr_source_path(transformed)
        print(f"  get_zarr_source_path -> {result}")
        print(f"  Expected             -> {zarr_path.resolve()} (or None)")
        if result is not None:
            print(f"  PASS (source found through transformations)")
        else:
            print(f"  INFO: source NOT found through transformations (conservative = safe)")


def test_persisted_array():
    """Test: dask array that was .persist()-ed (cached in memory)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.zarr"
        create_test_zarr(zarr_path)

        arr = da.from_zarr(str(zarr_path))
        persisted = arr.persist()

        result = get_zarr_source_path(persisted)
        print(f"  get_zarr_source_path -> {result}")
        print(f"  Expected             -> None (data is in memory, safe to overwrite)")
        if result is None:
            print("  PASS (persist removed zarr reference)")
        else:
            print(f"  INFO: zarr reference survived persist (conservative = blocks overwrite)")


def test_rechunked_array():
    """Test: dask array that was rechunked (as happens before write_zarr)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.zarr"
        create_test_zarr(zarr_path)

        arr = da.from_zarr(str(zarr_path))
        rechunked = arr.rechunk((25, 25))

        result = get_zarr_source_path(rechunked)
        print(f"  get_zarr_source_path -> {result}")
        print(f"  Expected             -> {zarr_path.resolve()} (should still detect source)")
        if result is not None:
            print(f"  PASS (source found through rechunk)")
        else:
            print(f"  INFO: source NOT found through rechunk")


# ---------------------------------------------------------------------------
# 4. Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing get_zarr_source_path")
    print("=" * 60)

    tests = [
        test_flat_zarr_array,
        test_pyramid_zarr_array,
        test_pyramid_as_list,
        test_in_memory_array,
        test_computed_array,
        test_persisted_array,
        test_rechunked_array,
    ]

    for test in tests:
        print(f"\n\n{'#'*60}")
        print(f"# {test.__doc__.strip()}")
        print(f"{'#'*60}")
        try:
            test()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'='*60}")
    print("All tests completed.")
    print("=" * 60)

"""On-disk layout of image and mask levels: Zarr v3 shards made of small inner chunks."""
import dask.array as da
import zarr

from insitupy._constants import (
    DEFAULT_CHUNK_SIZE_X,
    DEFAULT_CHUNK_SIZE_Y,
    ZARR_CHUNK_SIZE,
    ZARR_SHARD_SIZE,
)


def zarr_layout(shape, axes: str, *, sharded: bool) -> tuple[tuple, tuple | None]:
    """Return ``(chunks, shards)`` for one image or mask level.

    Y and X get ``ZARR_CHUNK_SIZE`` inner chunks inside ``ZARR_SHARD_SIZE`` shards; every other
    axis (S, C, T) is one block of full extent. Plain chunks (``shards=None``) are returned when
    not *sharded*, when ``ZARR_SHARD_SIZE`` is ``None`` or when any axis has length 0.
    """
    if len(axes) != len(shape):
        raise ValueError(f"Axes '{axes}' do not match array shape {tuple(shape)}.")

    chunks = tuple(
        ZARR_CHUNK_SIZE if ax in "YX" else max(int(n), 1)
        for ax, n in zip(axes, shape, strict=True)
    )

    if not sharded or ZARR_SHARD_SIZE is None or 0 in shape:
        return chunks, None

    if ZARR_SHARD_SIZE % ZARR_CHUNK_SIZE != 0:
        raise ValueError(
            f"ZARR_SHARD_SIZE ({ZARR_SHARD_SIZE}) must be a multiple of ZARR_CHUNK_SIZE ({ZARR_CHUNK_SIZE})."
        )
    shards = tuple(ZARR_SHARD_SIZE if ax in "YX" else c for ax, c in zip(axes, chunks, strict=True))
    return chunks, shards


def working_chunks(chunks, shards, axes: str | None) -> tuple:
    """Return the dask chunks used for reading or writing a level.

    The shard shape when sharded. Otherwise, per Y/X axis, the smallest multiple of the stored
    chunk that is at least ``DEFAULT_CHUNK_SIZE_Y/X``, and the stored chunk for other axes. With
    ``axes=None`` (unknown or inconsistent) the shards or chunks are returned unchanged.
    """
    if shards is not None:
        return tuple(shards)
    if axes is None or len(axes) != len(chunks):
        return tuple(chunks)

    target = {"Y": DEFAULT_CHUNK_SIZE_Y, "X": DEFAULT_CHUNK_SIZE_X}
    out = []
    for ax, c in zip(axes, chunks, strict=True):
        if ax in target:
            c = max(int(c), 1)
            out.append(-(-target[ax] // c) * c)
        else:
            out.append(int(c))
    return tuple(out)


def write_level(store, name: str | None, arr, axes: str, *, sharded: bool) -> None:
    """Write one image or mask level into *store* in the layout of :func:`zarr_layout`.

    *name* is the array path inside the store, or ``None`` for the store root. The array is
    rechunked to whole shards (or aligned blocks of plain chunks) first, so no two dask tasks
    write the same shard and ``lock=False`` is safe.
    """
    arr = da.asarray(arr)
    chunks, shards = zarr_layout(arr.shape, axes, sharded=sharded)
    z = zarr.create_array(
        store=store,
        name=name,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=chunks,
        shards=shards,
        overwrite=True,
    )
    if arr.size == 0:
        return
    da.store(arr.rechunk(working_chunks(chunks, shards, axes)), z, lock=False)

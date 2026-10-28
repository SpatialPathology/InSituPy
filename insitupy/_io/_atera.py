import os
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.io import read_elem
from dask import delayed

from insitupy.images.io import _get_zarr_store

ATERA_METADATA_FILE = "experiment.spatial"
ATERA_CELL_FEATURE_MATRIX_FILE = "cell_feature_matrix.zarr.zip"
ATERA_CELLS_FILE = "cells.zarr.zip"
ATERA_TRANSCRIPTS_FILE = "transcripts.zarr.zip"
ATERA_MORPHOLOGY_DIR = "morphology_2d"

# upper bound of transcript rows per dask partition; one spatial tile of a WTA bundle can hold
# far more rows than fit comfortably into memory at once
MAX_TRANSCRIPT_ROWS_PER_PARTITION = 5_000_000

# Xenium-style cell ids encode each hex digit of the id prefix as a letter a-p
# see: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cellID
_XENIUM_HEX_LETTERS = np.frombuffer(b"abcdefghijklmnop", dtype="S1")
_UNASSIGNED_ID = np.iinfo(np.uint32).max


def _encode_xenium_cell_ids(cell_id: np.ndarray) -> np.ndarray:
    """Vectorized conversion of ``(prefix, dataset_suffix)`` uint32 pairs to Xenium-style cell ids.

    Equivalent to :func:`insitupy.utils.utils.convert_int_to_xenium_hex` applied row-wise
    (e.g. ``[7790, 1] -> "aaaabogo-1"``). The "unassigned" sentinel
    ``(0xFFFFFFFF, 0xFFFFFFFF)`` used for transcripts outside of cells becomes ``"UNASSIGNED"``,
    the value Xenium uses in its transcripts file.

    Args:
        cell_id: Array of shape ``(n, 2)``.

    Returns:
        Object array of length ``n`` with string cell ids.
    """
    cell_id = np.asarray(cell_id)
    prefix = cell_id[:, 0].astype(np.uint64)
    digits = np.stack(
        [(prefix >> np.uint64(4 * (7 - i))) & np.uint64(0xF) for i in range(8)], axis=1
    ).astype(np.int64)
    letters = _XENIUM_HEX_LETTERS[digits].view("S8").ravel().astype(str)
    ids = np.char.add(np.char.add(letters, "-"), cell_id[:, 1].astype(str)).astype(object)
    unassigned = (cell_id[:, 0] == _UNASSIGNED_ID) & (cell_id[:, 1] == _UNASSIGNED_ID)
    ids[unassigned] = "UNASSIGNED"
    return ids


def _read_table_from_atera(path: str | os.PathLike | Path) -> AnnData:
    """Read the cell-by-gene table of an Atera bundle from ``cell_feature_matrix.zarr.zip``.

    The store is an AnnData zarr store inside a zip archive. Only "Gene Expression" features are
    kept (as in ``read_xenium``); 10x's flags ``highly_variable`` and ``filtered`` are renamed with
    an ``atera_`` prefix so that scanpy does not pick them up silently.
    """
    path = Path(path)
    store = _get_zarr_store(path / ATERA_CELL_FEATURE_MATRIX_FILE, mode="r", zipped=True)
    try:
        group = zarr.open_group(store, mode="r")
        obs = read_elem(group["obs"])
        var = read_elem(group["var"])
        X = read_elem(group["X"])
        obsm = read_elem(group["obsm"]) if "obsm" in group else {}
    finally:
        store.close()

    # keep only gene expression features (drops control probes/codewords and unassigned codewords)
    gex_mask = (var["feature_type"] == "Gene Expression").to_numpy()
    X = X[:, gex_mask]
    X.indices = X.indices.astype(np.int32)  # halves memory; number of genes is far below 2**31

    var = var.loc[gex_mask, ["feature_name", "feature_type", "genome", "highly_variable", "filtered"]]
    var = var.rename(columns={
        "feature_type": "feature_types",
        "highly_variable": "atera_highly_variable",
        "filtered": "atera_filtered",
    })
    var["gene_ids"] = var.index.astype(str)
    var.index = pd.Index(var.pop("feature_name").astype(str).to_numpy())

    # the index (barcode) is also stored as a column
    obs = obs.drop(columns=["barcode"], errors="ignore")
    obs = obs.rename(columns={"cell_id": "atera_cell_id", "filtered": "atera_filtered"})
    obs.index = pd.Index(obs.index.astype(str).to_numpy())

    adata = AnnData(X=X, obs=obs, var=var, obsm=dict(obsm))
    adata.var_names_make_unique()

    # cell centroids are given in microns
    coord_cols = ["centroid_column", "centroid_row"]
    adata.obsm["spatial"] = adata.obs[coord_cols].to_numpy()
    adata.obs = adata.obs.drop(columns=coord_cols)

    return adata


def _read_transcript_tile(
    transcripts_file: Path,
    tile_key: str,
    start: int,
    stop: int,
    feature_dtype: pd.CategoricalDtype,
) -> pd.DataFrame:
    """Read rows ``[start, stop)`` of one spatial tile of ``transcripts.zarr.zip``.

    Opens its own store so that it can run in independent dask tasks.
    """
    store = _get_zarr_store(transcripts_file, mode="r", zipped=True)
    try:
        tile = zarr.open_group(store, mode="r")["grid"][tile_key]
        location = tile["location"][start:stop]
        gene_offset = tile["gene_offset"][:]
        quality_score = tile["quality_score"][start:stop]
        overlaps_nucleus = tile["overlaps_nucleus"][start:stop]
        cell_id = tile["cell_id"][start:stop]
        codeword_index = tile["codeword_identity"][start:stop]
    finally:
        store.close()

    # rows of a tile are sorted by gene and gene_offset[g] = [start, end) gives the rows of gene g
    gene_codes = np.searchsorted(gene_offset[:, 1].astype(np.int64), np.arange(start, stop), side="right")

    return pd.DataFrame({
        "x_location": location[:, 0].astype(np.float32),
        "y_location": location[:, 1].astype(np.float32),
        "z_location": location[:, 2].astype(np.float32),
        "feature_name": pd.Categorical.from_codes(gene_codes, dtype=feature_dtype),
        "qv": quality_score.ravel().astype(np.float32),
        "overlaps_nucleus": overlaps_nucleus.ravel().astype(np.uint8),
        "cell_id": _encode_xenium_cell_ids(cell_id),
        "codeword_index": codeword_index.ravel().astype(np.uint32),
    })


def _read_transcripts_from_atera(path: str | os.PathLike | Path) -> dd.DataFrame:
    """Lazily read ``transcripts.zarr.zip`` into a dask DataFrame with Xenium column names.

    One partition per row range of a spatial tile (at most ``MAX_TRANSCRIPT_ROWS_PER_PARTITION``
    rows). Coordinates are in microns.
    """
    transcripts_file = Path(path) / ATERA_TRANSCRIPTS_FILE
    store = _get_zarr_store(transcripts_file, mode="r", zipped=True)
    try:
        root = zarr.open_group(store, mode="r")
        gene_names = [str(g) for g in root.attrs["gene_names"]]
        grid = root["grid"]
        tile_keys = list(grid.attrs.get("grid_keys") or sorted(grid.group_keys()))
        tile_sizes = {key: grid[key]["location"].shape[0] for key in tile_keys}
    finally:
        store.close()

    feature_dtype = pd.CategoricalDtype(categories=gene_names)
    meta = _read_transcript_tile(transcripts_file, tile_keys[0], 0, 0, feature_dtype)

    parts = [
        delayed(_read_transcript_tile)(transcripts_file, key, start,
                                       min(start + MAX_TRANSCRIPT_ROWS_PER_PARTITION, tile_sizes[key]),
                                       feature_dtype)
        for key in tile_keys
        for start in range(0, tile_sizes[key], MAX_TRANSCRIPT_ROWS_PER_PARTITION)
    ]
    if not parts:
        return dd.from_pandas(meta, npartitions=1)
    return dd.from_delayed(parts, meta=meta)

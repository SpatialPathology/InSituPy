import os
import warnings
from numbers import Number
from pathlib import Path
from typing import List, Literal, Union

import dask.array as da
import numpy as np
import pandas as pd
import scanpy as sc
import zarr
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from scipy.sparse import csr_matrix
from zarr.errors import ArrayNotFoundError

from insitupy._exceptions import InvalidFileTypeError
from insitupy.dataclasses.dataclasses import BoundariesData
from insitupy.images.io import _get_zarr_store
from insitupy.images.utils import _efficiently_resize_array
from insitupy.utils.utils import (convert_int_to_xenium_hex,
                                  decode_robust_series)


def _read_nucleus_to_cell_map_from_store(
    store,
    cell_names: np.ndarray
) -> dict:
    """
    Read nucleus-to-cell mapping from Xenium zarr store.

    Parameters
    ----------
    store
        Opened Xenium cells.zarr.zip store (zarr v2 or v3 compatible)
    cell_names : np.ndarray
        Array of cell names for fallback 1:1 mapping

    Returns
    -------
    dict
        Mapping from nucleus index (0-indexed) to cell index (0-indexed).
        For v2.0+ data with multinucleated cells, reads from polygon_sets.
        For v1.x data, creates 1:1 mapping.
    """
    try:
        nucleus_cell_index = da.from_zarr(store, component="polygon_sets/0/cell_index").compute()
        # v2.0+: nucleus_cell_index[i] gives the cell index for nucleus polygon i
        # To look up a mask value N, use: nucleus_to_cell_map[N - 1]
        nucleus_to_cell_map = {
            i: int(nucleus_cell_index[i]) for i in range(len(nucleus_cell_index))
        }
    except (ArrayNotFoundError, TypeError):
        # v1.x fallback: assume 1:1 mapping (nucleus index = cell index)
        nucleus_to_cell_map = {i: i for i in range(len(cell_names))}

    return nucleus_to_cell_map


def _read_nucleus_count_from_store(
    store
) -> Union[np.ndarray, None]:
    """
    Read nucleus count per cell from Xenium zarr store.

    Parameters
    ----------
    store
        Opened Xenium cells.zarr.zip store (zarr v2 or v3 compatible)

    Returns
    -------
    Union[np.ndarray, None]
        Array with number of nuclei per cell, or None if not available
    """
    try:
        cell_summary = da.from_zarr(store, component="cell_summary").compute()
        nucleus_count = cell_summary[:, 7]  # column index 7 = nucleus_count
    except (ArrayNotFoundError, TypeError, IndexError):
        nucleus_count = None

    return nucleus_count


def _read_table_from_xenium(path) -> AnnData:
    # extract parameters from metadata
    path = Path(path)
    cf_h5_path = path / "cell_feature_matrix.h5"

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # read matrix data
        adata = sc.read_10x_h5(cf_h5_path)

    # read cell information
    cells_parquet_path = path / "cells.parquet"
    cells = pd.read_parquet(cells_parquet_path)

    # transform cell ids from bytes to str
    cells = cells.set_index("cell_id")

    # make sure that the indices are decoded strings
    if is_numeric_dtype(cells.index):
        cells.index = cells.index.astype(str)
    else:
        cells.index = decode_robust_series(cells.index)

    # add information to anndata observations
    adata.obs = pd.merge(left=adata.obs, right=cells, left_index=True, right_index=True)

    # transfer coordinates to .obsm
    coord_cols = ["x_centroid", "y_centroid"]
    adata.obsm["spatial"] = adata.obs[coord_cols].values
    adata.obsm["spatial"]
    adata.obs.drop(coord_cols, axis=1, inplace=True)

    return adata


def _read_boundaries_from_xenium(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number,
    downscale: bool = False
    # mode: Literal["dataframe", "mask"] = "mask"
    ) -> BoundariesData:
    # # read boundaries data
    path = Path(path)

    # else:
    cells_zarr_file = path / "cells.zarr.zip"

    store = _get_zarr_store(cells_zarr_file, mode='r', zipped=True)

    # open zarr directory using dask
    cell_boundaries = da.from_zarr(store, component="masks/1")
    nuclei_boundaries = da.from_zarr(store, component="masks/0")

    if pixel_size != 1 and downscale:
        cell_boundaries = _efficiently_resize_array(array=cell_boundaries, scale_factor=1/pixel_size)
        nuclei_boundaries = _efficiently_resize_array(array=nuclei_boundaries, scale_factor=1/pixel_size)
        pixel_size_for_meta = 1
    else:
        pixel_size_for_meta = pixel_size

    # read cell ids and seg mask value
    # for info see: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cells
    cell_ids = da.from_zarr(store, component="cell_id").compute()
    if len(cell_ids.shape) == 2:
        cell_names = np.array([convert_int_to_xenium_hex(elem[0], elem[1]) for elem in cell_ids])
    elif len(cell_ids.shape) == 1:
        cell_names = cell_ids.astype(str)
    else:
        raise ValueError(f"Unexpected shape for `cell_ids` array: {cell_ids.shape} instead of 1 or 2.")

    try:
        seg_mask_value = da.from_zarr(store, component="seg_mask_value").compute()
    except (ArrayNotFoundError, TypeError):
        seg_mask_value = np.array(range(1, len(cell_names)+1))

    # Read nucleus metadata using helper functions
    nucleus_to_cell_map = _read_nucleus_to_cell_map_from_store(store, cell_names)
    nucleus_count = _read_nucleus_count_from_store(store)

    # create boundariesdata object
    boundaries = BoundariesData(
        cell_names=cell_names,
        seg_mask_value=seg_mask_value,
        nucleus_to_cell_map=nucleus_to_cell_map,
        nucleus_count=nucleus_count
        )

    boundaries.add_boundaries(
        cell_boundaries=cell_boundaries,
        nuclei_boundaries=nuclei_boundaries,
        pixel_size=pixel_size_for_meta
        )

    # Attach the zarr store so it stays open while dask arrays are alive
    boundaries._store = store

    return boundaries


def _read_binned_expression(
    path: Union[str, os.PathLike, Path],
    gene_names_to_select = List
):
    path = Path(path)
    # add binned expression data to .varm
    trans_file = path / "transcripts.zarr.zip"

    # read zarr store
    t = zarr.open(trans_file, mode="r")

    # extract sparse array
    data_gene = t["density/gene"]
    data = data_gene["data"][:]
    indices = data_gene["indices"][:]
    indptr = data_gene["indptr"][:]

    # get dimensions of the array
    cols = data_gene.attrs["cols"]
    rows = data_gene.attrs["rows"]

    # get info on gene names
    gene_names = data_gene.attrs["gene_names"]
    n_genes = len(gene_names)

    sarr = csr_matrix((data, indices, indptr))

    # reshape to get binned data
    arr = sarr.toarray()
    arr = arr.reshape((n_genes, rows, cols))

    # select only genes that are available in the adata object
    gene_mask = [elem in gene_names_to_select for elem in gene_names]
    arr = arr[gene_mask]
    return arr


def _restructure_transcripts_dataframe(dataframe: pd.DataFrame, decode:bool = False):

    if decode:
        # decode columns
        dataframe = dataframe.apply(lambda x: decode_robust_series(x), axis=0)

    # set index and rename columns
    dataframe = dataframe.set_index("transcript_id")
    dataframe = dataframe.rename({
        "cell_id": "cell_id",
        "x_location": "x",
        "y_location": "y",
        "z_location": "z",
        "feature_name": "gene"
    }, axis=1)

    # reorder dataframe
    column_names_ordered = ["x", "y", "z", "gene", "qv", "overlaps_nucleus", "fov_name", "nucleus_distance", "cell_id"]
    in_df = [elem in dataframe.columns for elem in column_names_ordered]
    column_names_ordered = [elem for i, elem in zip(in_df, column_names_ordered) if i]
    dataframe = dataframe.loc[:, column_names_ordered]

    # group column names into MultiIndices
    grouped_column_names = [
        ("coordinates", "x"),
        ("coordinates", "y"),
        ("coordinates", "z"),
        ("properties", "gene"),
        ("properties", "qv"),
        ("properties", "overlaps_nucleus"),
        ("properties", "fov_name"),
        ("properties", "nucleus_distance"),
        ("cell_id", "xenium")
    ]
    grouped_column_names = [elem for i, elem in zip(in_df, grouped_column_names) if i]
    dataframe.columns = pd.MultiIndex.from_tuples(grouped_column_names)
    return dataframe
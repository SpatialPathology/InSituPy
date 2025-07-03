from __future__ import \
    annotations  # this prevents circular imports of type hints such as InSituExperiment in this case

from numbers import Number
from typing import Collection, Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from parse import *
from tqdm import tqdm

from insitupy import __version__
from insitupy._core._checks import _is_experiment
from insitupy._core._utils import _get_cell_layer
from insitupy.preprocessing.anndata import (clustering_anndata,
                                            normalize_and_transform_anndata,
                                            reduce_dimensions_anndata)

from .._exceptions import ModalityNotFoundError


def calculate_qc_metrics(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    percent_top: Collection[int] = None,
    log1p: bool = False,
    **kwargs
):
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, d in iterator:
        celldata = _get_cell_layer(cells=d.cells, cells_layer=cells_layer)
        sc.pp.calculate_qc_metrics(
            celldata.matrix, percent_top=percent_top, log1p=log1p, inplace=True, **kwargs
            )

def filter_cells(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    mask: Optional[Union[np.ndarray, list, pd.Series]] = None,
    **kwargs
):
    """
    Filters cells in the given data based on specified criteria.

    Args:
        data (Union[InSituExperiment, InSituData]): The data containing cells to be filtered.
        cells_layer (Optional[str]): The layer of cells to be used for filtering.
        min_counts (Optional[int]): Minimum number of counts for filtering cells.
        min_genes (Optional[int]): Minimum number of genes for filtering cells.
        max_counts (Optional[int]): Maximum number of counts for filtering cells.
        max_genes (Optional[int]): Maximum number of genes for filtering cells.
        mask (Optional[np.ndarray]): Boolean array for filtering cells.
        **kwargs: Additional arguments passed to the filtering function.

    Raises:
        ValueError: If more than one filtering argument is provided or if the mask is not a boolean array.

    """
    # Ensure only one of the filtering arguments is not None
    filter_args = [min_counts, min_genes, max_counts, max_genes, mask]
    if sum(arg is not None for arg in filter_args) > 1:
        raise ValueError("Only one of min_counts, min_genes, max_counts, max_genes, or mask can be provided.")

    # Check if mask is a boolean array
    if mask is not None and not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("Mask must be a boolean array.")

    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

        if mask is not None:
            celldata.matrix = celldata.matrix[mask]
        else:
            sc.pp.filter_cells(
                celldata.matrix,
                min_counts=min_counts,
                min_genes=min_genes,
                max_counts=max_counts,
                max_genes=max_genes,
                inplace=True,
                **kwargs
            )

        # sync cell names between boundaries and matrix
        celldata.sync()

def filter_genes(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None,
    **kwargs
):
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
        sc.pp.filter_genes(
            celldata.matrix,
            min_counts=min_counts,
            min_cells=min_cells,
            max_counts=max_counts,
            max_cells=max_cells,
            inplace=True,
            **kwargs
            )

def normalize_and_transform(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    adata_layer: Optional[str] = None,
    transformation_method: Literal["log1p", "sqrt"] = "log1p",
    target_sum: int = 250,
    scale: bool = False,
    assert_integer_counts: bool = True,
    verbose: bool = False
    ) -> None:
    """
    Normalize the data using either log1p or square root transformation.

    Args:
        transformation_method (Literal["log1p", "sqrt"], optional):
            The method used for data transformation. Choose between "log1p" for logarithmic transformation
            and "sqrt" for square root transformation. Default is "log1p".
        verbose (bool, optional):
            If True, print progress messages during normalization. Default is True.

    Raises:
        ValueError: If `transformation_method` is not one of ["log1p", "sqrt"].

    Returns:
        None: This method modifies the input matrix in place, normalizing the data based on the specified method.
            It does not return any value.
    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if xd.cells is not None:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            normalize_and_transform_anndata(
                adata=celldata.matrix,
                layer=adata_layer,
                transformation_method=transformation_method,
                target_sum=target_sum,
                scale=scale,
                verbose=verbose,
                assert_integer_counts=assert_integer_counts
                )
        else:
            raise ModalityNotFoundError(modality="cells")

def reduce_dimensions(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    method: Literal["umap", "tsne"] = "umap",
    n_neighbors: int = 16,
    n_pcs: int = 0,
    ):
    """
    Performs dimensionality reduction of the data using either UMAP or TSNE.

    Args:
        data (Union[InSituExperiment, InSituData]): The experiment or sample-level data object containing cell information.
        method (Literal["umap", "tsne"], optional): The dimensionality reduction method to use. Defaults to "umap".
        cells_layer (Optional[str]): The specific layer of cells to use for reduction.
        n_neighbors (int, optional): The number of neighbors to use in the reduction method. Defaults to 16.
        n_pcs (int, optional): The number of principal components to use. Defaults to 0.

    Raises:
        ModalityNotFoundError: If the 'cells' modality is not found in the individual samples.

    """

    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if xd.cells is not None:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

            reduce_dimensions_anndata(
                adata=celldata.matrix,
                method=method,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs
                )
        else:
            raise ModalityNotFoundError(modality="cells")

def cluster_cells(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    method: Literal["leiden", "louvain"] = "leiden"
    ):
    """
    Performs clustering on the data using the specified method.

    Args:
        data (Union[InSituExperiment, InSituData]): The experiment or sample-level data object containing cell information.
        cells_layer (Optional[str]): The specific layer of cells to use for clustering.
        method (Literal["leiden", "louvain"], optional): The clustering method to use. Defaults to "leiden".
        verbose (bool, optional): If True, enables verbose output. Defaults to True.

    Raises:
        ModalityNotFoundError: If the 'cells' modality is not found in the individual samples.

    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if xd.cells is not None:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

            clustering_anndata(
                adata=celldata.matrix,
                method=method,
                verbose=False
                )
        else:
            raise ModalityNotFoundError(modality="cells")
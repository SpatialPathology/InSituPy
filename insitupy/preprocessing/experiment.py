from numbers import Number
from typing import Collection, Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from insitupy._version import __version__
from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy._exceptions import ModalityNotFoundError
from insitupy.containers._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment
from insitupy.preprocessing.anndata import (cluster_anndata,
                                            normalize_and_transform_anndata,
                                            reduce_dimensions_anndata)


def calculate_qc_metrics(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    percent_top: Collection[int] = None,
    log1p: bool = False,
    **kwargs
):
    """
    Calculate quality control metrics for cells using ``sc.pp.calculate_qc_metrics``.

    Computed metrics (e.g. ``n_genes_by_counts``, ``total_counts``) are added
    directly to ``adata.obs`` of each sample's cell table in place.

    Args:
        data (Union[InSituExperiment, InSituData]): Experiment or sample-level
            data object containing cell information.
        cells_layer (Optional[str], optional): Name of the cell segmentation
            layer to use. Defaults to None (main layer).
        percent_top (Collection[int], optional): Which proportions of top genes
            make up the total counts, computed for each cell. Forwarded to
            ``sc.pp.calculate_qc_metrics``. Defaults to None.
        log1p (bool, optional): If True, compute log1p of all QC metrics.
            Defaults to False.
        **kwargs: Additional keyword arguments forwarded to
            ``sc.pp.calculate_qc_metrics``.

    Returns:
        None: Modifies the cell table of each sample in place.
    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, d in iterator:
        celldata = _get_cell_layer(cells=d.cells, cells_layer=cells_layer)
        sc.pp.calculate_qc_metrics(
            celldata.table, percent_top=percent_top, log1p=log1p, inplace=True, **kwargs
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
        **kwargs: Additional keyword arguments forwarded to ``sc.pp.filter_cells()``.

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
            celldata.table = celldata.table[mask]
        else:
            sc.pp.filter_cells(
                celldata.table,
                min_counts=min_counts,
                min_genes=min_genes,
                max_counts=max_counts,
                max_genes=max_genes,
                inplace=True,
                **kwargs
            )

        # sync cell names between boundaries and table
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
    """
    Filter genes from the cell count matrix based on count and cell thresholds.

    Wraps ``sc.pp.filter_genes`` and applies it to each sample in ``data``.
    Genes not passing the filters are removed from the count matrix in place.

    Args:
        data (Union[InSituExperiment, InSituData]): Experiment or sample-level
            data object containing cell information.
        cells_layer (Optional[str], optional): Name of the cell segmentation
            layer to use. Defaults to None (main layer).
        min_counts (Optional[int], optional): Minimum total counts required for
            a gene to pass the filter. Defaults to None.
        min_cells (Optional[int], optional): Minimum number of cells in which a
            gene must be expressed to pass the filter. Defaults to None.
        max_counts (Optional[int], optional): Maximum total counts allowed for a
            gene to pass the filter. Defaults to None.
        max_cells (Optional[int], optional): Maximum number of cells in which a
            gene may be expressed to pass the filter. Defaults to None.
        **kwargs: Additional keyword arguments forwarded to
            ``sc.pp.filter_genes()``.

    Returns:
        None: Modifies the cell table of each sample in place.
    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
        sc.pp.filter_genes(
            celldata.table,
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
    Normalize and transform the cell count data for an experiment or sample.

    Iterates over all samples in ``data``, normalizes each cell to
    ``target_sum`` total counts, stores intermediate layers, and applies the
    chosen transformation. Delegates to
    :func:`~insitupy.preprocessing.anndata.normalize_and_transform_anndata`
    for each sample.

    Args:
        data (Union[InSituExperiment, InSituData]): Experiment or sample-level
            data object containing cell information.
        cells_layer (Optional[str], optional): Name of the cell segmentation
            layer to use. Defaults to None (main layer).
        adata_layer (Optional[str], optional): Name of the AnnData layer
            containing raw integer counts. If None, ``adata.X`` is used.
            Defaults to None.
        transformation_method (Literal['log1p', 'sqrt'], optional):
            Transformation applied after normalization. Defaults to ``'log1p'``.
        target_sum (int, optional): Total counts each cell is normalized to.
            Defaults to 250.
        scale (bool, optional): If True, scale each gene to zero mean and unit
            variance after transformation. Defaults to False.
        assert_integer_counts (bool, optional): If True, raise an error when the
            count matrix does not contain integer values. Defaults to True.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Raises:
        ValueError: If ``transformation_method`` is not one of ``['log1p', 'sqrt']``.
        ModalityNotFoundError: If a sample has no cells modality.

    Returns:
        None: Modifies the cell table of each sample in place.
    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if not xd.cells.is_empty:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            normalize_and_transform_anndata(
                adata=celldata.table,
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
    Perform dimensionality reduction on cell data using UMAP or t-SNE.

    Computes PCA, builds a nearest-neighbor graph, and runs the chosen
    embedding for each sample. Results are stored in each sample's cell
    AnnData in place.

    Args:
        data (Union[InSituExperiment, InSituData]): Experiment or sample-level
            data object containing cell information.
        cells_layer (Optional[str], optional): Name of the cell segmentation
            layer to use. Defaults to None (main layer).
        method (Literal['umap', 'tsne'], optional): Dimensionality reduction
            method. Defaults to ``'umap'``.
        n_neighbors (int, optional): Number of neighbors for the neighborhood
            graph. Defaults to 16.
        n_pcs (int, optional): Number of principal components to use for the
            neighborhood graph. Set to 0 to use all PCs. Defaults to 0.

    Raises:
        ModalityNotFoundError: If a sample has no cells modality.

    Returns:
        None: Modifies the cell table of each sample in place.
    """

    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if not xd.cells.is_empty:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

            reduce_dimensions_anndata(
                adata=celldata.table,
                method=method,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs
                )
        else:
            raise ModalityNotFoundError(modality="cells")

def cluster_cells(
    data: Union[InSituExperiment, InSituData], # type: ignore
    cells_layer: Optional[str] = None,
    method: Literal["leiden", "louvain"] = "leiden",
    verbose: bool = False
    ):
    """
    Performs clustering on the data using the specified method.

    Args:
        data (Union[InSituExperiment, InSituData]): The experiment or sample-level data object containing cell information.
        cells_layer (Optional[str]): The specific layer of cells to use for clustering.
        method (Literal["leiden", "louvain"], optional): The clustering method to use. Defaults to "leiden".
        verbose (bool, optional): If True, enables verbose output. Defaults to False.

    Raises:
        ModalityNotFoundError: If the 'cells' modality is not found in the individual samples.

    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if not xd.cells.is_empty:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

            cluster_anndata(
                adata=celldata.table,
                method=method,
                verbose=False
                )
        else:
            raise ModalityNotFoundError(modality="cells")
from typing import Literal, Optional

import numpy as np
import scanpy as sc
from parse import *
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from insitupy import __version__
from insitupy._core._checks import check_integer_counts
from insitupy._textformat import textformat as tf


def normalize_and_transform_anndata(
    adata,
    layer: Optional[str] = None,
    transformation_method: Literal["log1p", "sqrt"] = "log1p",
    target_sum: int = None, # defaults to median of total counts of cells
    scale: bool = False,
    assert_integer_counts: bool = True,
    verbose: bool = True) -> None:

    if layer is None:
        if assert_integer_counts:
            # check if the matrix consists of raw integer counts
            check_integer_counts(adata.X)

        # store raw counts in layer
        print("Store raw counts in .layers['counts'].") if verbose else None
        counts = adata.X.copy()
        adata.layers['counts'] = counts
    else:
        print(f"Retrieve raw counts from .layers['{layer}'].") if verbose else None
        if assert_integer_counts:
            # check if the matrix consists of raw integer counts
            check_integer_counts(adata.layers[layer])

        # move layer into .X
        adata.X = adata.layers[layer].copy()

    # preprocessing according to napari tutorial in squidpy
    print(f"Normalization with target sum {target_sum}.") if verbose else None
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # make sure the matrix is saved as sparse array
    adata.X = csr_matrix(adata.X)

    # save before log transformation
    adata.layers['norm_counts'] = adata.X.copy()

    # transform either using log transformation or square root transformation
    print(f"Perform {transformation_method}-transformation.") if verbose else None
    if transformation_method == "log1p":
        sc.pp.log1p(adata)

        # make sure the matrix is saved as sparse array
        adata.X = csr_matrix(adata.X)
    elif transformation_method == "sqrt":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
        norm_counts = adata.layers['norm_counts'].copy()
        try:
            X = norm_counts.toarray()
        except AttributeError:
            X = norm_counts
        adata.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
    else:
        raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt"]')


    if scale:
        print(f"Scale data.") if verbose else None
        adata.layers[f'{transformation_method}'] = adata.X.copy()
        sc.pp.scale(adata)

        # make sure the matrix is saved as sparse array
        adata.X = csr_matrix(adata.X)

def reduce_dimensions_anndata(
    adata,
    method: Literal["umap", "tsne"] = "umap",
    n_neighbors: int = 16,
    n_pcs: int = 0,
    verbose: bool = True,
    **kwargs
    ) -> None:
    """
    Reduce the dimensionality of the data using PCA, UMAP, and t-SNE techniques, optionally performing batch correction.

    Args:
        umap (bool, optional):
            If True, perform UMAP dimensionality reduction. Default is True.
        tsne (bool, optional):
            If True, perform t-SNE dimensionality reduction. Default is True.
        verbose (bool, optional):
            If True, print progress messages during dimensionality reduction. Default is True.
        tsne_lr (int, optional):
            Learning rate for t-SNE. Default is 1000.
        tsne_jobs (int, optional):
            Number of CPU cores to use for t-SNE computation. Default is 8.
        **kwargs:
            Additional keyword arguments to be passed to scanorama function if batch correction is performed.

    Raises:
        ValueError: If an invalid `batch_correction_key` is provided.

    Returns:
        None: This method modifies the input matrix in place, reducing its dimensionality using specified techniques and
            batch correction if applicable. It does not return any value.
    """
    # dimensionality reduction
    print("Dimensionality reduction...") if verbose else None
    sc.pp.pca(adata)

    # calculate neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # dimensionality reduction
    if method.lower() == "umap":
        sc.tl.umap(adata, **kwargs)
    elif method.lower() == "tsne":
        sc.tl.tsne(adata, **kwargs)

def clustering_anndata(
    adata,
    method: Literal["leiden", "louvain"] = "leiden",
    verbose: bool = True
):
    # clustering
    if method.lower() == "leiden":
        print("Leiden clustering...") if verbose else None
        sc.tl.leiden(adata)
    elif method.lower() == "louvain":
        print("Louvain clustering...") if verbose else None
        sc.tl.louvain(adata)
    else:
        raise ValueError(f'`type` is not one of ["leiden", "louvain"]')

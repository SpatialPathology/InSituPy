import logging
from typing import Literal, Optional

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from insitupy._textformat import textformat as tf
from insitupy._version import __version__
from insitupy.utils._checks import check_integer_counts

logger = logging.getLogger(__name__)


def normalize_and_transform_anndata(
    adata,
    layer: Optional[str] = None,
    transformation_method: Literal["log1p", "sqrt"] = "log1p",
    target_sum: int = None, # defaults to median of total counts of cells
    scale: bool = False,
    assert_integer_counts: bool = True,
    verbose: bool = False) -> None:
    """
    Normalize and transform a single AnnData object in place.

    Stores raw counts in ``adata.layers['counts']``, normalizes to ``target_sum``,
    saves normalized counts in ``adata.layers['norm_counts']``, then applies the
    requested transformation. Optionally scales the data to unit variance.

    Args:
        adata: AnnData object to process. Modified in place.
        layer (Optional[str], optional): Name of the layer containing raw integer counts.
            If None, ``adata.X`` is used. Defaults to None.
        transformation_method (Literal['log1p', 'sqrt'], optional):
            Transformation applied after normalization. ``'log1p'`` applies
            ``sc.pp.log1p``; ``'sqrt'`` applies the Freeman-Tukey transform
            (``sqrt(x) + sqrt(x+1)``). Defaults to ``'log1p'``.
        target_sum (int, optional): Total counts to normalize each cell to.
            Defaults to None (normalizes to the median total count across cells).
        scale (bool, optional): If True, scale each gene to zero mean and unit
            variance after transformation. Defaults to False.
        assert_integer_counts (bool, optional): If True, raise an error when the
            count matrix does not contain integer values. Defaults to True.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Raises:
        ValueError: If ``transformation_method`` is not one of ``['log1p', 'sqrt']``.

    Returns:
        None: Modifies ``adata`` in place.
    """

    if layer is None:
        if assert_integer_counts:
            # check if the matrix consists of raw integer counts
            check_integer_counts(adata.X)

        # store raw counts in layer
        logger.info("Store raw counts in .layers['counts'].") if verbose else None
        counts = adata.X.copy()
        adata.layers['counts'] = counts
    else:
        logger.info("Retrieve raw counts from .layers['%s'].", layer) if verbose else None
        if assert_integer_counts:
            # check if the matrix consists of raw integer counts
            check_integer_counts(adata.layers[layer])

        # move layer into .X
        adata.X = adata.layers[layer].copy()

    # preprocessing according to napari tutorial in squidpy
    logger.info("Normalization with target sum %s.", target_sum) if verbose else None
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # make sure the matrix is saved as sparse array
    adata.X = csr_matrix(adata.X)

    # save before log transformation
    adata.layers['norm_counts'] = adata.X.copy()

    # transform either using log transformation or square root transformation
    logger.info("Perform %s-transformation.", transformation_method) if verbose else None
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
        logger.info("Scale data.") if verbose else None
        adata.layers[f'{transformation_method}'] = adata.X.copy()
        sc.pp.scale(adata)

        # make sure the matrix is saved as sparse array
        adata.X = csr_matrix(adata.X)

def reduce_dimensions_anndata(
    adata,
    method: Literal["umap", "tsne"] = "umap",
    n_neighbors: int = 16,
    n_pcs: int = 0,
    verbose: bool = False,
    **kwargs
    ) -> None:
    """
    Reduce the dimensionality of data using PCA followed by UMAP or t-SNE.

    Computes a PCA, builds a nearest-neighbor graph, then runs the chosen
    embedding method. All results are stored in ``adata`` in place.

    Args:
        adata: AnnData object to reduce. Modified in place.
        method (Literal['umap', 'tsne'], optional):
            Dimensionality reduction method to apply after PCA and neighbor
            graph computation. Defaults to ``'umap'``.
        n_neighbors (int, optional):
            Number of neighbors for ``sc.pp.neighbors``. Defaults to 16.
        n_pcs (int, optional):
            Number of principal components to use when computing the neighbor
            graph. 0 means use all PCs. Defaults to 0.
        verbose (bool, optional):
            If True, print progress messages. Defaults to False.
        **kwargs:
            Additional keyword arguments forwarded to ``sc.tl.umap()`` or
            ``sc.tl.tsne()``.

    Returns:
        None: Modifies ``adata`` in place.
    """
    # dimensionality reduction
    logger.info("Calculate PCA...") if verbose else None
    sc.pp.pca(adata)

    # calculate neighbors
    logger.info("Calculate neighbors...") if verbose else None
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # dimensionality reduction
    logger.info("Calculate %s...", method) if verbose else None
    if method.lower() == "umap":
        sc.tl.umap(adata, **kwargs)
    elif method.lower() == "tsne":
        sc.tl.tsne(adata, **kwargs)

def cluster_anndata(
    adata,
    method: Literal["leiden", "louvain"] = "leiden",
    verbose: bool = False
):
    """
    Cluster cells in an AnnData object using Leiden or Louvain community detection.

    Requires a precomputed neighbor graph (e.g. from ``reduce_dimensions_anndata``).
    Cluster labels are stored in ``adata.obs['leiden']`` or ``adata.obs['louvain']``.

    Args:
        adata: AnnData object with a precomputed neighbor graph. Modified in place.
        method (Literal['leiden', 'louvain'], optional):
            Clustering algorithm. Defaults to ``'leiden'``.
        verbose (bool, optional):
            If True, print progress messages. Defaults to False.

    Raises:
        ValueError: If ``method`` is not one of ``['leiden', 'louvain']``.

    Returns:
        None: Modifies ``adata`` in place.
    """
    # clustering
    if method.lower() == "leiden":
        logger.info("Leiden clustering...") if verbose else None
        sc.tl.leiden(adata, flavor='igraph')
    elif method.lower() == "louvain":
        logger.info("Louvain clustering...") if verbose else None
        sc.tl.louvain(adata)
    else:
        raise ValueError(f'`type` is not one of ["leiden", "louvain"]')

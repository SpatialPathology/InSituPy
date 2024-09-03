from typing import Literal, Optional

import numpy as np
import scanpy as sc
from parse import *
from scipy.sparse import csr_matrix

from insitupy import __version__
from insitupy.utils._scanorama import scanorama

from .._core._checks import check_integer_counts


def normalize_anndata(adata,
              transformation_method: Literal["log1p", "sqrt"] = "log1p",
              verbose: bool = True
              ) -> None:
    # check if the matrix consists of raw integer counts
    check_integer_counts(adata.X)

    # store raw counts in layer
    print("Store raw counts in anndata.layers['counts']...") if verbose else None
    adata.layers['counts'] = adata.X.copy()

    # preprocessing according to napari tutorial in squidpy
    print(f"Normalization, {transformation_method}-transformation...") if verbose else None
    sc.pp.normalize_total(adata)
    adata.layers['norm_counts'] = adata.X.copy()

    # transform either using log transformation or square root transformation
    if transformation_method == "log1p":
        sc.pp.log1p(adata)
    elif transformation_method == "sqrt":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
        X = adata.X.toarray()
        adata.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
    else:
        raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt"]')


def reduce_dimensions_anndata(adata,
                              umap: bool = True,
                              tsne: bool = False,
                              layer: Optional[str] = None,
                              batch_correction_key: Optional[str] = None,
                              perform_clustering: bool = True,
                              verbose: bool = True,
                              tsne_lr: int = 1000,
                              tsne_jobs: int = 8,
                              **kwargs
                              ) -> None:
    """
    Reduce the dimensionality of the data using PCA, UMAP, and t-SNE techniques, optionally performing batch correction.

    Args:
        umap (bool, optional):
            If True, perform UMAP dimensionality reduction. Default is True.
        tsne (bool, optional):
            If True, perform t-SNE dimensionality reduction. Default is True.
        layer (str, optional): 
            Specifies the layer of the AnnData object to operate on. Default is None (uses adata.X).
        batch_correction_key (str, optional):
            Batch key for performing batch correction using scanorama. Default is None, indicating no batch correction.
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
    
    # Determine the prefix for the data
    data_prefix = layer if layer else "raw"

    if batch_correction_key is None:
        # dimensionality reduction
        print("Dimensionality reduction...") if verbose else None

        # perform PCA with the specified layer
        sc.pp.pca(adata, layer=layer)

        # Manually rename the PCA results with the prefix. Future scanpy version will include an argument
        # key_added to do this automatically
        adata.obsm[f'{data_prefix}_pca'] = adata.obsm['X_pca']
        del adata.obsm['X_pca']

        adata.varm[f'{data_prefix}_PCs'] = adata.varm['PCs']
        del adata.varm['PCs']

        adata.uns[f'{data_prefix}_pca'] = adata.uns['pca']
        del adata.uns['pca']
        
        if umap:
            # Perform neighbors analysis with the specified prefix
            sc.pp.neighbors(adata, use_rep=f'{data_prefix}_pca', key_added=f'{data_prefix}_neighbors')

            # Perform UMAP using the custom neighbors key
            sc.tl.umap(adata, neighbors_key=f'{data_prefix}_neighbors')
            
            # Rename and store UMAP results with the appropriate prefix
            adata.obsm[f'{data_prefix}_umap'] = adata.obsm['X_umap']
            del adata.obsm['X_umap']

            adata.uns[f'{data_prefix}_umap'] = adata.uns['umap']
            del adata.uns['umap']
        
        if tsne:
            # Perform t-SNE using the PCA results with the specified prefix
            sc.tl.tsne(adata, n_jobs=tsne_jobs, learning_rate=tsne_lr, use_rep=f'{data_prefix}_pca', key_added=f'{data_prefix}_tsne')

    else:
        # PCA for batch correction
        sc.pp.pca(adata, layer=layer)

        neigh_uncorr_key = f'{data_prefix}_neighbors_uncorrected'
        sc.pp.neighbors(adata, use_rep=f'{data_prefix}_pca', key_added=neigh_uncorr_key)

        if perform_clustering:
            # Clustering
            sc.tl.leiden(adata, neighbors_key=neigh_uncorr_key, key_added=f'{data_prefix}_leiden_uncorrected')

        # Batch correction
        print(f"Batch correction using scanorama for {batch_correction_key}...") if verbose else None
        hvgs = list(adata.var_names[adata.var['highly_variable']])
        adata = scanorama(adata, batch_key=batch_correction_key, hvg=hvgs, verbose=False, **kwargs)

        # Find neighbors and reduce dimensions
        sc.pp.neighbors(adata, use_rep="X_scanorama", key_added=f'{data_prefix}_scanorama_neighbors')
        sc.tl.umap(adata, neighbors_key=f'{data_prefix}_scanorama_neighbors', key_added=f'{data_prefix}_scanorama_umap')
        sc.tl.tsne(adata, use_rep="X_scanorama", key_added=f'{data_prefix}_scanorama_tsne')

    if perform_clustering:
        # Clustering
        print("Leiden clustering...") if verbose else None
        sc.tl.leiden(adata, neighbors_key=f'{data_prefix}_neighbors', key_added=f'{data_prefix}_leiden')
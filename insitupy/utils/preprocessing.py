from typing import Literal, Optional

import numpy as np
import scanpy as sc
from parse import *
from scipy.sparse import csr_matrix

from insitupy import __version__
from insitupy.utils._scanorama import scanorama

from .._core._checks import check_integer_counts
from copy import deepcopy
from typing import List, Literal, Dict
from anndata import AnnData



def normalize_and_transform_anndata(adata: AnnData,
                      transformation_methods: List[Literal["log1p", "sqrt", "pearson_residuals"]],
                      verbose: bool = True
                      ) -> Dict[str, AnnData]:
    """
    Normalize and transform the data in an AnnData object based on specified methods.

    Args:
        adata (AnnData): The AnnData object to be normalized and transformed.
        transformation_methods (List[str]): List of transformation methods to apply.
            Options are ["log1p", "sqrt", "pearson_residuals"].
        verbose (bool, optional): If True, prints progress messages. Default is True.

    Returns:
        Dict[str, AnnData]: A dictionary where keys are transformation methods and values are the 
                            transformed AnnData objects.
    """

    # Check if the matrix consists of raw integer counts
    #check_integer_counts(adata.X)

    # Store raw counts in the original adata for comparison
    if verbose:
        print("Store raw counts in anndata.layers['counts']...")
    adata.layers['counts'] = adata.X.copy()

    # Normalize total counts for each transformation
    sc.pp.normalize_total(adata)
    adata.layers['norm_counts'] = adata.X.copy()

    # Dictionary to store different transformations
    transformed_data = {}

    for method in transformation_methods:
        if verbose:
            print(f"Applying transformation: {method}")

        # Copy the original AnnData object for each transformation
        adata_copy = adata.copy()

        # Apply the selected transformation method
        if method == "log1p":
            sc.pp.log1p(adata_copy)

        elif method == "sqrt_1":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
            X = adata_copy.X.toarray()
            adata_copy.X = csr_matrix(X + np.sqrt(X + 1))

        elif method == "sqrt_2":
            X = adata_copy.X.toarray()
            adata_copy.X = csr_matrix(np.sqrt(X))

        elif method == "pearson_residuals":
            # Applying the Pearson residuals transformation
            analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(adata_copy, layer="counts", inplace=False)
            adata_copy.X = csr_matrix(analytic_pearson["X"])

        else:
            raise ValueError(f'`transformation_method` {method} is not one of ["log1p", "sqrt_1", "sqrt_2", "pearson_residuals"]')

        # Store the transformed AnnData object in the results dictionary
        transformed_data[method] = adata_copy

    return transformed_data

def reduce_dimensions_anndata(data,
                              umap: bool = True,
                              tsne: bool = False,
                              batch_correction_key: Optional[str] = None,
                              perform_clustering: bool = True,
                              verbose: bool = True,
                              tsne_lr: int = 1000,
                              tsne_jobs: int = 8,
                              **kwargs
                              ) -> None:
    """
    Reduce the dimensionality of the data using PCA, UMAP, and t-SNE techniques,
    optionally performing batch correction.

    Args:
        data (Union[AnnData, Dict[str, AnnData]]):
            Either a single AnnData object or a dictionary where keys are labels
            (e.g., transformation methods) and values are AnnData objects to process.
        umap (bool, optional):
            If True, perform UMAP dimensionality reduction. Default is True.
        tsne (bool, optional):
            If True, perform t-SNE dimensionality reduction. Default is False.
        batch_correction_key (str, optional):
            Batch key for performing batch correction using scanorama.
            Default is None, indicating no batch correction.
        perform_clustering (bool, optional):
            If True, perform Leiden clustering. Default is True.
        verbose (bool, optional):
            If True, print progress messages during dimensionality reduction.
            Default is True.
        tsne_lr (int, optional):
            Learning rate for t-SNE. Default is 1000.
        tsne_jobs (int, optional):
            Number of CPU cores to use for t-SNE computation. Default is 8.
        **kwargs:
            Additional keyword arguments to be passed to the scanorama function
            if batch correction is performed.

    Raises:
        ValueError: If an invalid `batch_correction_key` is provided.

    Returns:
        None: This method modifies the input AnnData object(s) in place, reducing
        their dimensionality using specified techniques and batch correction if
        applicable. It does not return any value.
    """

    # Helper function to perform dimensionality reduction on a single AnnData object
    def process_anndata(adata, label):
        if verbose:
            print(f"Processing {label}...")

        if batch_correction_key is None:
            # Dimensionality reduction without batch correction
            if verbose:
                print("Performing PCA...")
            sc.pp.pca(adata)

            if umap:
                if verbose:
                    print("Computing neighbors and UMAP...")
                sc.pp.neighbors(adata)
                sc.tl.umap(adata)

            if tsne:
                if verbose:
                    print("Performing t-SNE...")
                sc.tl.tsne(adata, n_jobs=tsne_jobs, learning_rate=tsne_lr)

        else:
            # Dimensionality reduction with batch correction
            if verbose:
                print("Performing PCA for batch correction...")
            sc.pp.pca(adata)

            neigh_uncorr_key = 'neighbors_uncorrected'
            sc.pp.neighbors(adata, key_added=neigh_uncorr_key)

            if perform_clustering:
                if verbose:
                    print("Performing initial clustering...")
                sc.tl.leiden(adata, neighbors_key=neigh_uncorr_key, key_added='leiden_uncorrected')

            # Batch correction using scanorama
            if verbose:
                print(f"Batch correction using scanorama for {batch_correction_key}...")
            hvgs = list(adata.var_names[adata.var['highly_variable']])
            adata = scanorama(adata, batch_key=batch_correction_key, hvg=hvgs, verbose=False, **kwargs)

            # Dimensionality reduction after batch correction
            if verbose:
                print("Computing neighbors and embeddings after batch correction...")
            sc.pp.neighbors(adata, use_rep="X_scanorama")
            sc.tl.umap(adata)
            if tsne:
                sc.tl.tsne(adata, use_rep="X_scanorama", n_jobs=tsne_jobs, learning_rate=tsne_lr)

        if perform_clustering:
            if verbose:
                print("Performing Leiden clustering...")
            sc.tl.leiden(adata)

    # Check if data is a dictionary of AnnData objects
    if isinstance(data, dict):
        # Iterate over each AnnData object in the dictionary
        for label, adata in data.items():
            process_anndata(adata, label)
    else:
        # Data is a single AnnData object
        process_anndata(data, 'adata')
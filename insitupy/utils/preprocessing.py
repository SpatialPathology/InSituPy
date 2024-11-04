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
from scipy.stats import skew, kurtosis, norm, probplot, shapiro, anderson, kstest
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
import warnings
import anndata2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import numpy as np

from scipy.sparse import csr_matrix

import tempfile




anndata2ri.activate()
pandas2ri.activate()

def sctransform_anndata(adata, layer=None, **kwargs):
    if layer:
        mat = adata.layers[layer]
    else:
        mat = adata.X

    # Set names for the input matrix
    cell_names = adata.obs_names
    gene_names = adata.var_names
    r.assign('mat', mat.T)
    r.assign('cell_names', cell_names)
    r.assign('gene_names', gene_names)
    r('colnames(mat) <- cell_names')
    r('rownames(mat) <- gene_names')

    seurat = importr('Seurat')
    r('seurat_obj <- CreateSeuratObject(mat)')

    # Run
    for k, v in kwargs.items():
        r.assign(k, v)
    kwargs_str = ', '.join([f'{k}={k}' for k in kwargs.keys()])
    r(f'seurat_obj <- SCTransform(seurat_obj,vst.flavor="v2", {kwargs_str})')

   

    # Prevent partial SCT output because of default min.genes messing up layer addition
    r('diffDash <- setdiff(rownames(seurat_obj), rownames(mat))')
    r('diffDash <- gsub("-", "_", diffDash)')
    r('diffScore <- setdiff(rownames(mat), rownames(seurat_obj))')
    filtout_genes = np.array(r('setdiff(diffScore, diffDash)'))

    filtout_indicator = np.in1d(adata.var_names, filtout_genes)
    adata = adata[:, ~filtout_indicator]

    # Extract the SCT data and add it as a new layer in the original anndata object
    sct_data = np.asarray(r['as.matrix'](r('seurat_obj@assays$SCT@data')))
    adata.layers["SCT_adata"] = sct_data.T
    sct_data = np.asarray(r['as.matrix'](r('seurat_obj@assays$SCT@counts')))
    adata.layers['counts'] = sct_data.T
    return adata


def normalize_and_transform_anndata(adata,
                                    transformation_method: Literal["log1p", "sqrt_1", "sqrt_2", "pearson_residuals", "sctransform"] = "log1p",
                                    verbose: bool = True
                                    ) -> None:
    """
    Normalize and transform the data in an AnnData object based on the selected transformation method.
    
    Args:
        adata (AnnData): The AnnData object to be normalized and transformed.
        transformation_method (str): The transformation method to apply. Options are ["log1p", "sqrt_1", "sqrt_2", "pearson_residuals", "sctransform"].
        verbose (bool): If True, prints progress messages.
    """
    
    # Check if the matrix consists of raw integer counts (optional: you may need to implement the check_integer_counts function)
    # check_integer_counts(adata.X)

    # Store raw counts in layer
    print("Store raw counts in anndata.layers['counts']...") if verbose else None
    adata.layers['counts'] = adata.X.copy()

    # Preprocessing according to napari tutorial in squidpy
    print(f"Normalization, {transformation_method}-transformation...") if verbose else None
    sc.pp.normalize_total(adata)
    adata.layers['norm_counts'] = adata.X.copy()

    # Apply the selected transformation method
    if transformation_method == "log1p":
        sc.pp.log1p(adata)

    elif transformation_method == "sqrt_1":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
        X = adata.X.toarray()
        adata.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))

    elif transformation_method == "sqrt_2":
        X = adata.X.toarray()
        adata.X = csr_matrix(np.sqrt(X))
    
    elif transformation_method == "pearson_residuals":
        # Applying the Pearson residuals transformation
        analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(adata, layer="counts", inplace=False)
        adata.X = csr_matrix(analytic_pearson["X"])

    elif transformation_method == "sctransform":
        # Apply SCTransform by calling the sctransform_anndata function
        print("Applying SCTransform...") if verbose else None
        adata = sctransform_anndata(adata, verbose=verbose)
    
    else:
        raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt_1", "sqrt_2", "pearson_residuals", "sctransform"]')

    return adata



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
    data_prefix = layer if layer else "X"

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

warnings.filterwarnings("ignore", category=FutureWarning, message=".*incompatible with float64.*")


def compare_transformations_anndata(adata: AnnData,
                                    transformation_methods: List[Literal["log1p", "sqrt_1", "sqrt_2", "pearson_residuals", "sctransform"]],
                                    verbose: bool = True,
                                    output_path: str = "normalization_results.html") -> pd.DataFrame:
    """
    Normalize and transform the data in an AnnData object based on specified methods, 
    and then compare the transformed results, including SCTransform.

    Args:
        adata (AnnData): The AnnData object to be normalized and transformed.
        transformation_methods (List[str]): List of transformation methods to apply.
            Options are ["log1p", "sqrt_1", "sqrt_2", "pearson_residuals", "sctransform"].
        verbose (bool, optional): If True, prints progress messages. Default is True.
        output_path (str, optional): The path where the HTML report will be saved. The default is normalization_results.html

    Returns:
        pd.DataFrame: A DataFrame with comparison metrics for each transformation method.
    """

    # Step 1: Normalize and transform the data using the specified methods
    if verbose:
        print("Store raw counts in anndata.layers['counts']...")
    
    # Store raw counts for comparison
    adata.layers['counts'] = adata.X.copy()

    # Normalize total counts
    sc.pp.normalize_total(adata, target_sum=250)
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
            X = adata_copy.X.toarray()
            adata_copy.X = csr_matrix(X + np.sqrt(X + 1))

        elif method == "sqrt_2":
            X = adata_copy.X.toarray()
            adata_copy.X = csr_matrix(np.sqrt(X))

        elif method == "pearson_residuals":
            # Applying the Pearson residuals transformation
            analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(adata_copy, layer="counts", inplace=False)
            adata_copy.X = csr_matrix(analytic_pearson["X"])

        elif method == "sctransform":
            # Applying SCTransform using the custom function
            adata_copy = sctransform_anndata(adata_copy)

        else:
            raise ValueError(f'`transformation_method` {method} is not one of ["log1p", "sqrt_1", "sqrt_2", "pearson_residuals", "sctransform"]')

        # Store the transformed AnnData object in the results dictionary
        transformed_data[method] = adata_copy

    # Step 2: Compare the transformations and generate the plots
    results = {}
    plots = []

    # Store raw counts for comparison
    raw_counts = adata.layers["counts"].toarray().sum(axis=1)

    for method, transformed_adata in transformed_data.items():
        if verbose:
            print(f"Processing {method}...")

        # Extract the transformed counts
        transformed_counts = transformed_adata.X.toarray().sum(axis=1)

        # Check for NaNs and handle them
        if np.isnan(transformed_counts).any():
            print(f"Warning: NaN values found in {method} transformation. Replacing NaNs with 0.")
            transformed_counts = np.nan_to_num(transformed_counts)

        # Plot Histogram of both raw and transformed counts overlaid with a normal distribution
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Histogram of the data
        axs[0].hist(transformed_counts, bins=50, density=True, alpha=0.6, color='g', label=f'{method} counts')

        # Overlay normal distribution
        mean, std_dev = np.mean(transformed_counts), np.std(transformed_counts)
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
        y = norm.pdf(x, mean, std_dev)
        axs[0].plot(x, y, label='Normal Distribution', color='blue')

        axs[0].set_title(f'{method} transformation - Histogram with Normal Overlay')
        axs[0].legend()

        # Q-Q plot
        probplot(transformed_counts, dist="norm", plot=axs[1])
        axs[1].set_title(f'Q-Q Plot - {method}')

        # Save the plot to a string buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plots.append(f'<img src="data:image/png;base64,{image_base64}" alt="{method} plot"/>')

        # Skewness
        skewness = skew(transformed_counts)

        # Kurtosis
        kurt = kurtosis(transformed_counts)  # Excess kurtosis

        # Mean Absolute Deviation (MAD)
        mad = np.mean(np.abs(transformed_counts - np.mean(transformed_counts)))

        # Coefficient of Variation (CV)
        cv = np.std(transformed_counts) / np.mean(transformed_counts)

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = shapiro(transformed_counts)

        # Anderson-Darling test
        anderson_stat = anderson(transformed_counts)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = kstest(transformed_counts, 'norm', args=(mean, std_dev))

        # Store results for comparison
        results[method] = {
            "skewness": skewness,
            "kurtosis": kurt,
            "mad": mad,
            "cv": cv,
            "shapiro_stat": shapiro_stat,
            "shapiro_p": shapiro_p,
            "anderson_stat": anderson_stat.statistic,
            "ks_stat": ks_stat,
            "ks_p": ks_p
        }

    # Convert results dictionary to a DataFrame for easier comparison
    results_df = pd.DataFrame(results).T  # Transpose to have methods as rows

    # Highlight the best methods and create HTML table
    def highlight_best_method(results_df):
        # Copy the DataFrame to avoid modifying the original
        highlighted_df = results_df.copy()

        # Create temporary columns for absolute skewness and kurtosis
        highlighted_df['skewness_abs'] = np.abs(highlighted_df['skewness'])
        highlighted_df['kurtosis_abs'] = np.abs(highlighted_df['kurtosis'])

        # Define the "better" criteria for each metric:
        metrics_to_minimize = ['mad', 'cv', 'skewness_abs', 'kurtosis_abs', 'shapiro_stat', 'anderson_stat', 'ks_stat']

        # Highlight the best values by setting the background color
        for metric in metrics_to_minimize:
            best_value_index = highlighted_df[metric].idxmin()
            highlighted_df.loc[best_value_index, metric.replace('_abs', '')] = (
                f'<div style="background-color:lightgreen">{results_df.loc[best_value_index, metric.replace("_abs", "")]}</div>'
            )

        # Drop temporary columns
        highlighted_df = highlighted_df.drop(columns=['skewness_abs', 'kurtosis_abs'])

        # Convert the entire DataFrame to HTML with escape=False to allow HTML tags
        return highlighted_df.to_html(escape=False)

    results_html = highlight_best_method(results_df)

    # Generate the final HTML report
    full_html = f"""
    <html>
    <head>
        <title>Transformation Results</title>
    </head>
    <body>
        <h1>Transformation Comparison Results</h1>
        <h2>Summary Table</h2>
        {results_html}
        <h2>Transformation Method Plots</h2>
        {"<br>".join(plots)}
    </body>
    </html>
    """

    # Save the HTML file to the specified output path
    with open(output_path, "w") as file:
        file.write(full_html)

    print(f"HTML report created and saved as '{output_path}'")
    return results_df

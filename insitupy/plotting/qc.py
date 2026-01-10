import warnings
from numbers import Number
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from insitupy._core.data import InSituData
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.utils._checks import check_integer_counts


def plot_qc_metrics(
    data: Union[InSituData, AnnData], # type: ignore #TODO: Expand this to InSituExperiment
    cells_layer: Optional[str] = None,
    show_inset: bool = True,
    inset_fraction: Number = 0.2,
    plot_obs: bool = True,
    plot_var: bool = True,
    additional_obs: Optional[Union[str, list]] = None,
    additional_var: Optional[Union[str, list]] = None
    ):
    """
    Plots the QC metrics calculated by sc.pp.calculate_qc_metrics.

    Parameters:
    adata : AnnData
        Annotated data matrix with QC metrics calculated.
    plot_obs : bool, optional
        Whether to plot .obs metrics. Default is True.
    plot_var : bool, optional
        Whether to plot .var metrics. Default is True.
    additional_obs : str or list of str, optional
        Additional column(s) from .obs to plot as histograms. Must contain numeric data.
    additional_var : str or list of str, optional
        Additional column(s) from .var to plot as histograms. Must contain numeric data.
    """
    if isinstance(data, AnnData):
        adata = data.copy()
    else:
        # retrieve AnnData from cell layer
        celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        adata = celldata.table.copy()

    # QC metrics in .obs
    obs_metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']
    # QC metrics in .var
    var_metrics = ['n_cells_by_counts', 'mean_counts', 'pct_dropout_by_counts', 'total_counts']

    # Check if all metrics exist in .obs
    if plot_obs:
        obs_metrics = [metric for metric in obs_metrics if metric in adata.obs]
        if len(obs_metrics) == 0:
            print("Warning: No .obs metrics found in adata.obs")
    else:
        obs_metrics = []

    # Check if all metrics exist in .var
    if plot_var:
        var_metrics = [metric for metric in var_metrics if metric in adata.var]
        if len(var_metrics) == 0:
            print("Warning: No .var metrics found in adata.var")
    else:
        var_metrics = []

    # Process additional_obs
    if additional_obs is not None and plot_obs:
        if isinstance(additional_obs, str):
            additional_obs = [additional_obs]

        # Filter to only existing numeric columns
        additional_obs_metrics = []
        for metric in additional_obs:
            if metric not in adata.obs:
                print(f"Warning: '{metric}' not found in adata.obs")
            elif not pd.api.types.is_numeric_dtype(adata.obs[metric]):
                print(f"Warning: '{metric}' is not numeric and will be skipped")
            else:
                additional_obs_metrics.append(metric)
    else:
        additional_obs_metrics = []

    # Process additional_var
    if additional_var is not None and plot_var:
        if isinstance(additional_var, str):
            additional_var = [additional_var]

        # Filter to only existing numeric columns
        additional_var_metrics = []
        for metric in additional_var:
            if metric not in adata.var:
                print(f"Warning: '{metric}' not found in adata.var")
            elif not pd.api.types.is_numeric_dtype(adata.var[metric]):
                print(f"Warning: '{metric}' is not numeric and will be skipped")
            else:
                additional_var_metrics.append(metric)
    else:
        additional_var_metrics = []

    # Calculate number of plots and layout
    num_scatter = 1 if ('n_genes_by_counts' in obs_metrics and 'total_counts' in obs_metrics) else 0
    num_obs_plots = len(obs_metrics) + len(additional_obs_metrics) + num_scatter
    num_var_plots = len(var_metrics) + len(additional_var_metrics)

    # Determine if we need two rows
    if num_obs_plots > 0 and num_var_plots > 0:
        nrows = 2
        ncols = max(num_obs_plots, num_var_plots)
    else:
        nrows = 1
        ncols = num_obs_plots + num_var_plots

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))

    # Ensure axes is always 2D for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Track current column index for each row
    current_col = 0

    # Add obs metrics annotation
    if len(obs_metrics) > 0:
        axes[0, current_col].annotate('.obs Metrics', xy=(0, 0.5), xytext=(-axes[0, current_col].yaxis.labelpad - 5, 0),
                            xycoords=axes[0, current_col].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90, weight='bold')

    # Plot obs metrics
    for i, metric in enumerate(obs_metrics):
        sns.histplot(adata.obs[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=axes[0, current_col])
        axes[0, current_col].set_title(metric)
        axes[0, current_col].set_xlabel('Value')
        axes[0, current_col].set_ylabel('Frequency')

        if show_inset:
            # Inset histogram
            ax_inset = inset_axes(axes[0, current_col], width="40%", height="40%", loc='upper right')
            sns.histplot(adata.obs[metric], bins=100, color='skyblue', edgecolor='black', kde=False, ax=ax_inset)
            ax_inset.set_xlim(adata.obs[metric].min(), adata.obs[metric].max() * inset_fraction)  # Adjust the x-limits as needed
            ax_inset.set_xlabel('')
            ax_inset.set_ylabel('')
            ax_inset.set_yticklabels([])
        current_col += 1

    # Add scatter plot for n_genes_by_counts vs total_counts
    if 'n_genes_by_counts' in obs_metrics and 'total_counts' in obs_metrics:
        scatter_ax = axes[0, current_col]
        scatter_ax.scatter(adata.obs['n_genes_by_counts'], adata.obs['total_counts'], alpha=0.5, color='skyblue', s=8, edgecolor='black', linewidth=0.2)
        scatter_ax.set_title('n_genes_by_counts vs total_counts')
        scatter_ax.set_xlabel('n_genes_by_counts')
        scatter_ax.set_ylabel('total_counts')
        current_col += 1

    # Plot additional obs metrics
    for i, metric in enumerate(additional_obs_metrics):
        sns.histplot(adata.obs[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=axes[0, current_col])
        axes[0, current_col].set_title(metric)
        axes[0, current_col].set_xlabel('Value')
        axes[0, current_col].set_ylabel('Frequency')

        if show_inset:
            # Inset histogram
            ax_inset = inset_axes(axes[0, current_col], width="40%", height="40%", loc='upper right')
            sns.histplot(adata.obs[metric], bins=100, color='skyblue', edgecolor='black', kde=False, ax=ax_inset)
            ax_inset.set_xlim(adata.obs[metric].min(), adata.obs[metric].max() * inset_fraction)
            ax_inset.set_xlabel('')
            ax_inset.set_ylabel('')
            ax_inset.set_yticklabels([])
        current_col += 1

    # Hide unused axes in first row
    if nrows == 2:
        for col in range(current_col, ncols):
            axes[0, col].set_visible(False)

    # Plot var metrics in second row if two rows, otherwise continue in first row
    var_row = 1 if nrows == 2 else 0
    var_col = 0 if nrows == 2 else current_col

    # Add var metrics annotation
    if len(var_metrics) > 0:
        axes[var_row, var_col].annotate('.var Metrics', xy=(0, 0.5), xytext=(-axes[var_row, var_col].yaxis.labelpad - 5, 0),
                            xycoords=axes[var_row, var_col].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90, weight='bold')

    # Plot var metrics
    for i, metric in enumerate(var_metrics):
        sns.histplot(adata.var[metric], bins=50, color='coral', edgecolor='black', kde=False, ax=axes[var_row, var_col])
        axes[var_row, var_col].set_title(metric)
        axes[var_row, var_col].set_xlabel('Value')
        axes[var_row, var_col].set_ylabel('Frequency')
        var_col += 1

    # Plot additional var metrics
    for i, metric in enumerate(additional_var_metrics):
        sns.histplot(adata.var[metric], bins=50, color='coral', edgecolor='black', kde=False, ax=axes[var_row, var_col])
        axes[var_row, var_col].set_title(metric)
        axes[var_row, var_col].set_xlabel('Value')
        axes[var_row, var_col].set_ylabel('Frequency')
        var_col += 1

    # Hide unused axes in second row
    if nrows == 2:
        for col in range(var_col, ncols):
            axes[1, col].set_visible(False)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
        plt.tight_layout()
    plt.show()

def test_transformations(
    data: InSituData, # type: ignore #TODO: Expand this to InSituExperiment
    cells_layer: Optional[str] = None,
    target_sum: Number = 250,
    layer: Optional[str] = None,
    scale: bool = False,
    assert_integer_counts: bool = True
        ):
    """
    Test normalization and transformation methods by plotting histograms of raw,
    log1p-transformed, and sqrt-transformed counts.

    Args:
        adata (AnnData): Omics data as AnnData object.
        target_sum (int, optional): Target sum for normalization. Defaults to 1e4.
        layer (str, optional): Layer to use for transformation. Defaults to None.
    """
    # retrieve AnnData from cell layer
    celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
    adata = celldata.table.copy() # copy it to not affect it during the plotting
    # Check if the matrix consists of raw integer counts
    if layer is None:
        if assert_integer_counts:
            check_integer_counts(adata.X)
    else:
        adata.X = adata.layers[layer].copy()
        if assert_integer_counts:
            check_integer_counts(adata.X)

    # get raw counts
    raw_counts = adata.X.copy()

    # Preprocessing according to napari tutorial in squidpy
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Create a copy of the anndata object for log1p transformation
    adata_log1p = adata.copy()
    sc.pp.log1p(adata_log1p)

    # Create a copy of the anndata object for sqrt transformation
    adata_sqrt = adata.copy()
    try:
        X = adata_sqrt.X.toarray()
    except AttributeError:
        X = adata_sqrt.X
    adata_sqrt.X = np.sqrt(X) + np.sqrt(X + 1)

    if scale:
        sc.pp.scale(adata_log1p)
        sc.pp.scale(adata_sqrt)
        titles = ['Log1p-transformed and scaled counts','Sqrt-transformed and scaled counts']
    else:
        titles = ['Log1p-transformed counts','Sqrt-transformed ounts']

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(raw_counts.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Raw Counts', fontsize=14)
    axes[0].set_xlabel('Counts', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    axes[1].hist(adata_log1p.X.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[1].set_title(titles[0], fontsize=14)
    axes[1].set_xlabel('Counts', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    axes[2].hist(adata_sqrt.X.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[2].set_title(titles[1], fontsize=14)
    axes[2].set_xlabel('Counts', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)


    plt.tight_layout()
    plt.show()
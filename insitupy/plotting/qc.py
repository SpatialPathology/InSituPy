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
    data: Union[InSituData, AnnData],
    cells_layer: Optional[str] = None,
    show_inset: bool = True,
    inset_fraction: Number = 0.2,
    plot_obs: bool = True,
    plot_var: bool = True,
    additional_obs: Optional[Union[str, list]] = None,
    additional_var: Optional[Union[str, list]] = None,
    batch: Optional[str] = None,
    counts_thresh: Optional[Number] = None,
    genes_thresh: Optional[Number] = None,
):
    """
    Plots the QC metrics calculated by sc.pp.calculate_qc_metrics.

    Parameters:
    data : InSituData or AnnData
        Annotated data matrix with QC metrics calculated.
    cells_layer : str, optional
        Cell layer to use if data is InSituData.
    show_inset : bool, optional
        Whether to show inset histograms. Default is True.
    inset_fraction : Number, optional
        Fraction of x-axis range to show in inset. Default is 0.2.
    plot_obs : bool, optional
        Whether to plot .obs metrics. Default is True.
    plot_var : bool, optional
        Whether to plot .var metrics. Default is True.
    additional_obs : str or list of str, optional
        Additional column(s) from .obs to plot as histograms. Must contain numeric data.
    additional_var : str or list of str, optional
        Additional column(s) from .var to plot as histograms. Must contain numeric data.
    batch : str, optional
        Column in .obs to use for batch separation. Only allowed if not both plot_obs
        and plot_var are True. Default is None.
    counts_thresh : Number, optional
        Threshold to display as vertical line on total_counts plots. Default is None.
    genes_thresh : Number, optional
        Threshold to display as vertical line on n_genes_by_counts plots. Default is None.
    """
    # Validate batch argument
    if batch is not None and plot_obs and plot_var:
        raise ValueError(
            "batch can only be used when either plot_obs or plot_var is False, "
            "not when both are True."
        )

    if isinstance(data, AnnData):
        adata = data.copy()
    else:
        celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        adata = celldata.table.copy()

    # Validate batch column exists
    if batch is not None:
        if batch not in adata.obs:
            raise ValueError(f"batch column '{batch}' not found in adata.obs")
        batch_values = adata.obs[batch].unique()
        n_batches = len(batch_values)
    else:
        batch_values = [None]
        n_batches = 1

    # QC metrics in .obs
    obs_metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']
    # QC metrics in .var
    var_metrics = ['n_cells_by_counts', 'mean_counts', 'pct_dropout_by_counts', 'total_counts']

    if plot_obs:
        obs_metrics = [metric for metric in obs_metrics if metric in adata.obs]
        if len(obs_metrics) == 0:
            print("Warning: No .obs metrics found in adata.obs")
    else:
        obs_metrics = []

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

    # Helper function to add threshold lines
    def add_threshold_line(ax, metric, thresh_value):
        if thresh_value is not None:
            ax.axvline(x=thresh_value, color='red', linestyle='--', linewidth=1.5, label=f'thresh={thresh_value}')

    def get_threshold_for_metric(metric):
        if metric == 'total_counts' and counts_thresh is not None:
            return counts_thresh
        elif metric == 'n_genes_by_counts' and genes_thresh is not None:
            return genes_thresh
        return None

    # Calculate number of plots and layout
    num_scatter = 1 if ('n_genes_by_counts' in obs_metrics and 'total_counts' in obs_metrics) else 0
    num_obs_plots = len(obs_metrics) + len(additional_obs_metrics) + num_scatter
    num_var_plots = len(var_metrics) + len(additional_var_metrics)

    # Determine layout
    if batch is not None:
        # Batches as rows
        if num_obs_plots > 0 and num_var_plots > 0:
            raise ValueError("This should not happen due to earlier validation")
        nrows = n_batches
        ncols = num_obs_plots + num_var_plots
    else:
        if num_obs_plots > 0 and num_var_plots > 0:
            nrows = 2
            ncols = max(num_obs_plots, num_var_plots)
        else:
            nrows = 1
            ncols = num_obs_plots + num_var_plots

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))

    # Ensure axes is always 2D
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot function for a single batch/row
    def plot_obs_row(row_idx, adata_subset, row_label=None):
        current_col = 0

        # Add row label annotation
        if row_label is not None and len(obs_metrics) > 0:
            axes[row_idx, 0].annotate(
                str(row_label), xy=(0, 0.5),
                xytext=(-axes[row_idx, 0].yaxis.labelpad - 5, 0),
                xycoords=axes[row_idx, 0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, weight='bold'
            )
        elif row_label is None and len(obs_metrics) > 0 and row_idx == 0:
            axes[0, current_col].annotate(
                '.obs Metrics', xy=(0, 0.5),
                xytext=(-axes[0, current_col].yaxis.labelpad - 5, 0),
                xycoords=axes[0, current_col].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, weight='bold'
            )

        # Plot obs metrics
        for metric in obs_metrics:
            ax = axes[row_idx, current_col]
            sns.histplot(adata_subset.obs[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=ax)
            ax.set_title(metric if row_label is None else f"{metric}")
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

            thresh = get_threshold_for_metric(metric)
            if thresh is not None:
                add_threshold_line(ax, metric, thresh)

            if show_inset:
                ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
                sns.histplot(adata_subset.obs[metric], bins=100, color='skyblue', edgecolor='black', kde=False, ax=ax_inset)
                ax_inset.set_xlim(adata_subset.obs[metric].min(), adata_subset.obs[metric].max() * inset_fraction)
                ax_inset.set_xlabel('')
                ax_inset.set_ylabel('')
                ax_inset.set_yticklabels([])
                if thresh is not None:
                    add_threshold_line(ax_inset, metric, thresh)
            current_col += 1

        # Add scatter plot
        if 'n_genes_by_counts' in obs_metrics and 'total_counts' in obs_metrics:
            scatter_ax = axes[row_idx, current_col]
            scatter_ax.scatter(
                adata_subset.obs['n_genes_by_counts'], adata_subset.obs['total_counts'],
                alpha=0.5, color='skyblue', s=8, edgecolor='black', linewidth=0.2
            )
            scatter_ax.set_title('n_genes_by_counts vs total_counts')
            scatter_ax.set_xlabel('n_genes_by_counts')
            scatter_ax.set_ylabel('total_counts')

            if genes_thresh is not None:
                scatter_ax.axvline(x=genes_thresh, color='red', linestyle='--', linewidth=1.5)
            if counts_thresh is not None:
                scatter_ax.axhline(y=counts_thresh, color='red', linestyle='--', linewidth=1.5)
            current_col += 1

        # Plot additional obs metrics
        for metric in additional_obs_metrics:
            ax = axes[row_idx, current_col]
            sns.histplot(adata_subset.obs[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

            if show_inset:
                ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
                sns.histplot(adata_subset.obs[metric], bins=100, color='skyblue', edgecolor='black', kde=False, ax=ax_inset)
                ax_inset.set_xlim(adata_subset.obs[metric].min(), adata_subset.obs[metric].max() * inset_fraction)
                ax_inset.set_xlabel('')
                ax_inset.set_ylabel('')
                ax_inset.set_yticklabels([])
            current_col += 1

        return current_col

    def plot_var_row(row_idx, start_col=0, row_label=None):
        var_col = start_col

        # Add row/section label
        if row_label is not None and len(var_metrics) > 0:
            axes[row_idx, var_col].annotate(
                str(row_label), xy=(0, 0.5),
                xytext=(-axes[row_idx, var_col].yaxis.labelpad - 5, 0),
                xycoords=axes[row_idx, var_col].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, weight='bold'
            )
        elif row_label is None and len(var_metrics) > 0:
            axes[row_idx, var_col].annotate(
                '.var Metrics', xy=(0, 0.5),
                xytext=(-axes[row_idx, var_col].yaxis.labelpad - 5, 0),
                xycoords=axes[row_idx, var_col].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90, weight='bold'
            )

        for metric in var_metrics:
            ax = axes[row_idx, var_col]
            sns.histplot(adata.var[metric], bins=50, color='coral', edgecolor='black', kde=False, ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            var_col += 1

        for metric in additional_var_metrics:
            ax = axes[row_idx, var_col]
            sns.histplot(adata.var[metric], bins=50, color='coral', edgecolor='black', kde=False, ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            var_col += 1

        return var_col

    # Main plotting logic
    if batch is not None:
        # Plot each batch as a separate row
        for row_idx, batch_val in enumerate(batch_values):
            adata_batch = adata[adata.obs[batch] == batch_val]
            if plot_obs:
                end_col = plot_obs_row(row_idx, adata_batch, row_label=batch_val)
                for col in range(end_col, ncols):
                    axes[row_idx, col].set_visible(False)
            elif plot_var:
                # var metrics don't subset by batch (they're gene-level)
                end_col = plot_var_row(row_idx, start_col=0, row_label=batch_val)
                for col in range(end_col, ncols):
                    axes[row_idx, col].set_visible(False)
    else:
        # Original behavior without batching
        current_col = plot_obs_row(0, adata)

        if nrows == 2:
            for col in range(current_col, ncols):
                axes[0, col].set_visible(False)

        var_row = 1 if nrows == 2 else 0
        var_start = 0 if nrows == 2 else current_col

        if num_var_plots > 0:
            var_col = plot_var_row(var_row, start_col=var_start)
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
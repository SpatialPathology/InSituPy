from numbers import Number

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import median_abs_deviation

from insitupy._core.data import InSituData
from insitupy.containers._utils import _get_cell_layer


def _compute_mad_threshold(values, n_mads, log1p_transform=True):
    """
    Core MAD threshold computation.

    Args:
        values (array-like): Count values.
        n_mads (Number): Number of MADs from median for threshold.
        log1p_transform (bool): If True, compute MAD on log1p-transformed values.
            Default is True.

    Returns:
        tuple: (threshold_log1p, threshold_raw) - threshold in both scales.
    """
    if log1p_transform:
        log_values = np.log1p(values)
        median = np.median(log_values)
        mad = median_abs_deviation(log_values)
        thresh_log = median - n_mads * mad
        thresh_raw = np.expm1(thresh_log)
    else:
        median = np.median(values)
        mad = median_abs_deviation(values)
        thresh_raw = median - n_mads * mad
        thresh_log = np.log1p(thresh_raw)

    return thresh_log, thresh_raw



def calculate_mad_thresholds(
    data: InSituData | AnnData,
    cells_layer: str | None = None,
    batch: str | None = None,
    n_mads: Number = 5,
) -> pd.DataFrame:
    """
    Calculate MAD-based QC thresholds for total_counts and n_genes_by_counts.

    Thresholds are computed on log1p-transformed values for statistical validity
    (as recommended by sc-best-practices), then back-transformed to raw scale.

    Args:
        data (InSituData or AnnData): Annotated data matrix with QC metrics calculated.
        cells_layer (str, optional): Cell layer to use if data is InSituData.
        batch (str, optional): Column in .obs to use for batch separation. If None,
            computes global thresholds.
        n_mads (Number, optional): Number of MADs from median for threshold calculation.
            Default is 5.

    Returns:
        pd.DataFrame: DataFrame with columns: 'batch' (if batch is not None),
            'total_counts_thresh', 'log1p_total_counts_thresh',
            'n_genes_by_counts_thresh', 'log1p_n_genes_by_counts_thresh'.
    """
    if isinstance(data, AnnData):
        adata = data
    else:
        celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        adata = celldata.table

    # Check required columns exist
    required_cols = ['total_counts', 'n_genes_by_counts']
    for col in required_cols:
        if col not in adata.obs:
            raise ValueError(f"Required column '{col}' not found in adata.obs. "
                           "Run sc.pp.calculate_qc_metrics first.")

    # Determine batches
    if batch is not None:
        if batch not in adata.obs:
            raise ValueError(f"Batch column '{batch}' not found in adata.obs")
        batch_values = adata.obs[batch].unique()
    else:
        batch_values = [None]

    # Compute thresholds per batch
    results = []
    for batch_val in batch_values:
        if batch_val is not None:
            mask = adata.obs[batch] == batch_val
            subset = adata.obs.loc[mask]
        else:
            subset = adata.obs

        counts_log, counts_raw = _compute_mad_threshold(
            subset['total_counts'].values, n_mads
        )
        genes_log, genes_raw = _compute_mad_threshold(
            subset['n_genes_by_counts'].values, n_mads
        )

        row = {
            'total_counts_thresh': counts_raw,
            'log1p_total_counts_thresh': counts_log,
            'n_genes_by_counts_thresh': genes_raw,
            'log1p_n_genes_by_counts_thresh': genes_log,
        }
        if batch is not None:
            row['batch'] = batch_val

        results.append(row)

    df = pd.DataFrame(results)

    # Reorder columns to put batch first if present
    if batch is not None:
        cols = ['batch'] + [c for c in df.columns if c != 'batch']
        df = df[cols]

    return df

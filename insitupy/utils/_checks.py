import logging

import anndata
import dask.array as da
import numpy as np
from scipy.sparse import issparse

from insitupy._constants import WITH_NAPARI

logger = logging.getLogger(__name__)

if WITH_NAPARI:
    pass


# checker functions for data sanity
def check_adata(adata):
    """Raise TypeError if *adata* is not an :class:`~anndata.AnnData` instance."""
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')


def check_batch(batch, obs, verbose=False):
    """Raise ValueError if the *batch* column is absent from *obs*.

    Args:
        batch: Column name to check for in *obs*.
        obs: A DataFrame (typically ``adata.obs``).
        verbose: If True, log the number of unique batches found.
    """
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        logger.info(f'Object contains {obs[batch].nunique()} batches.')


def check_hvg(hvg, hvg_key, adata_var):
    """Validate that the HVG list and HVG key are present in ``adata.var``.

    Args:
        hvg: List of highly variable gene names.
        hvg_key: Column in *adata_var* that flags highly variable genes.
        adata_var: The ``var`` DataFrame of an AnnData object.

    Raises:
        TypeError: If *hvg* is not a list.
        ValueError: If any gene in *hvg* is absent from *adata_var*.
        KeyError: If *hvg_key* is not a column in *adata_var*.
    """
    if type(hvg) is not list:
        raise TypeError('HVG list is not a list')
    else:
        if not all(i in adata_var.index for i in hvg):
            raise ValueError('Not all HVGs are in the adata object')
    if hvg_key not in adata_var:
        raise KeyError('`hvg_key` not found in `adata.var`')

def check_sanity(adata, batch, hvg, hvg_key):
    """Run a suite of sanity checks on an AnnData object before processing.

    Delegates to :func:`check_adata`, :func:`check_batch`, and (when *hvg*
    is truthy) :func:`check_hvg`.

    Args:
        adata: AnnData object to validate.
        batch: Batch column name expected in ``adata.obs``.
        hvg: HVG gene list or falsy value to skip HVG check.
        hvg_key: Column in ``adata.var`` that marks highly variable genes.
    """
    check_adata(adata)
    check_batch(batch, adata.obs)
    if hvg:
        check_hvg(hvg, hvg_key, adata.var)


def check_integer_counts(X):
    '''
    Check if a matrix consists of raw integer counts or if it is processed already.
    '''

    # convert sparse matrix to numpy array
    if issparse(X):
        X = X.toarray()

    # check if the matrix contains raw counts
    if not np.all(np.modf(X)[0] == 0):
        raise ValueError("Anndata object does not contain raw counts. Preprocessing aborted.")


def is_integer_counts(X):
    '''
    Check if a matrix consists of raw integer counts or if it is processed already.
    '''

    # convert sparse matrix to numpy array
    if issparse(X):
        X = X.toarray()

    # check if the matrix contains raw counts
    return np.all(np.modf(X)[0] == 0)

def check_raw(adata, use_raw, layer=None):
    """Return the expression matrix, var DataFrame, and var names for either raw or processed data.

    Args:
        adata: AnnData object.
        use_raw: If True, return data from ``adata.raw``; otherwise from
            ``adata.X`` or ``adata.layers[layer]``.
        layer: Layer key to use when *use_raw* is False.  Ignored when
            *use_raw* is True.

    Returns:
        A tuple ``(X, var, var_names)`` where *X* is a dense numpy array
        and *var* / *var_names* are the corresponding variable metadata.
    """
    # check if plotting raw data
    if use_raw:
        adata_X = adata.raw.X
        adata_var = adata.raw.var
        adata_var_names = adata.raw.var_names
    else:
        if layer is None:
            adata_X = adata.X
        else:
            #adata_X = adata.layers[layer].toarray()
            adata_X = adata.layers[layer]

        if issparse(adata_X):
            adata_X = adata_X.toarray()

        adata_var = adata.var
        adata_var_names = adata.var_names

    return adata_X, adata_var, adata_var_names

def check_zip(path):
    """Determine whether *path* refers to a zip output and return the base path.

    Args:
        path: A :class:`~pathlib.Path` whose suffix is either ``".zip"`` or
            ``""`` (no extension for a directory output).

    Returns:
        A tuple ``(zip_output, base_path)`` where *zip_output* is a bool
        indicating zip mode and *base_path* is the path without the ``.zip``
        suffix (if applicable).

    Raises:
        ValueError: If the suffix is neither ``".zip"`` nor empty.
    """
    # check if the output directory is going to be zipped or not
    if path.suffix == ".zip":
        zip_output = True
        path = path.with_suffix("")
    elif path.suffix == "":
        zip_output = False
    else:
        raise ValueError(f"The specified output path ({path}) must be a valid directory or a zip file. It does not need to exist yet.")

    return zip_output

# Function to check if there are any valid labels in matplotlib figure
def has_valid_labels(ax):
    """Return True if *ax* has at least one legend handle with a non-underscore label.

    Args:
        ax: A matplotlib :class:`~matplotlib.axes.Axes` to inspect.

    Returns:
        True if a labelled artist exists, False otherwise.
    """
    for artist in ax.get_legend_handles_labels()[0]:  # Get the handles (artists)
        if artist.get_label() and not artist.get_label().startswith('_'):
            return True
    return False

def is_valid_rgb_tuple(value):
    """
    Check if a value is a valid RGB sequence.

    A valid RGB sequence is defined as a sequence (list, tuple, numpy array, etc.)
    containing three numeric values, each in the range of 0 to 255.

    Parameters:
    value: The value to check (list, tuple, numpy array, or sequence).

    Returns:
    bool: True if the value is a valid RGB sequence, False otherwise.
    """
    try:
        # Check if it's a sequence with length 3
        if len(value) != 3:
            return False

        # Check if all values are numeric and in range [0, 255]
        return all(isinstance(v, (int, np.integer, float, np.floating)) and 0 <= v <= 255 for v in value)
    except (TypeError, AttributeError):
        # Not a sequence or doesn't have len()
        return False

def check_rgb_column(df, column_name):
    """
    Check if a specified column in a DataFrame contains only valid RGB tuples.

    This function checks if the specified column exists in the DataFrame and
    verifies that all entries in the column are valid RGB tuples.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    column_name (str): The name of the column to validate.

    Returns:
    bool: True if all values in the column are valid RGB tuples, False otherwise.

    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Check if all values in the specified column are valid RGB tuples
    return df[column_name].apply(is_valid_rgb_tuple).all()

def _is_list_unique(lst):
    return len(lst) == len(set(lst))

def _is_list_of_dask_arrays(variable):
    # Check if the variable is a list
    if not isinstance(variable, list):
        return False

    # Check if all elements in the list are dask arrays
    for element in variable:
        if not isinstance(element, da.Array):
            return False

    return True


def _calculate_single_metrics(adata, layer=None, force_layer=False):
    """
    Calculate median genes by counts and median total counts for a single AnnData.

    Args:
        adata: AnnData object to calculate metrics for.
        layer: The layer to use for calculations. If None, uses adata.X or 'counts' layer.
        force_layer: Whether to use specified layer even if not integer counts.

    Returns:
        Tuple of (median_genes_per_cell, median_transcripts_per_cell).
    """
    import warnings

    import scanpy as sc

    # Determine which data to use
    use_layer = layer
    if layer is None and not is_integer_counts(adata.X) and not force_layer:
        use_layer = "counts"

    # Validate counts
    data = adata.layers.get(use_layer) if use_layer else adata.X
    if data is None or (not is_integer_counts(data) and not force_layer):
        warnings.warn(
            f"No raw counts provided{f' in layer {use_layer!r}' if use_layer else ''}, metrics are set to 0.",
            UserWarning,
            stacklevel=2,
        )
        return 0, 0

    df_cells, _ = sc.pp.calculate_qc_metrics(adata, percent_top=None, layer=use_layer)
    return df_cells["n_genes_by_counts"].median(), df_cells["total_counts"].median()

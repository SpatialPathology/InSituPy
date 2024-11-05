from typing import Literal

import anndata
import numpy as np
from scipy.sparse import issparse


# checker functions for data sanity
def check_adata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')


def check_batch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        print(f'Object contains {obs[batch].nunique()} batches.')


def check_hvg(hvg, hvg_key, adata_var):
    if type(hvg) is not list:
        raise TypeError('HVG list is not a list')
    else:
        if not all(i in adata_var.index for i in hvg):
            raise ValueError('Not all HVGs are in the adata object')
    if not hvg_key in adata_var:
        raise KeyError('`hvg_key` not found in `adata.var`')

def check_sanity(adata, batch, hvg, hvg_key):
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

def check_raw(adata, use_raw, layer=None):
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
    for artist in ax.get_legend_handles_labels()[0]:  # Get the handles (artists)
        if artist.get_label() and not artist.get_label().startswith('_'):
            return True
    return False

def is_valid_rgb_tuple(value):
    """
    Check if a value is a valid RGB tuple.

    A valid RGB tuple is defined as a list or tuple containing three integers,
    each in the range of 0 to 255.

    Parameters:
    value (list or tuple): The value to check.

    Returns:
    bool: True if the value is a valid RGB tuple, False otherwise.
    """
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return all(isinstance(v, int) and 0 <= v <= 255 for v in value)
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


def _check_assignment(
    data, key, force_assignment, modality: Literal["annotations", "regions"]
):
    try:
        column = data.cells.matrix.obsm[modality].columns
    except KeyError:
        do_assignment = True
    else:
        if key in column:
            do_assignment = False
        else:
            do_assignment = True

    if do_assignment or force_assignment:
        if modality == "annotations":
            # assign annotations
            data.assign_annotations(keys=key)
        elif modality == "regions":
            # assign regions
            data.assign_regions(keys=key)
    else:
        print(f"{modality.capitalize()} with key '{key} have already been assigned to `data`.")


def _substitution_func(
    row,
    annotation_key,
    annotation_name,
    check_reference,
    reference_name=None,
    ignore_duplicate_assignments=False
    ):
    elem = row[annotation_key]
    # check_reference = False
    # if reference_name is not None:
    #     check_reference = True
    try:
        split_name = elem.split(" & ")
        if annotation_name in split_name:
            if check_reference:
                if reference_name in split_name:
                    if not ignore_duplicate_assignments:
                        raise ValueError(f"Cell '{row.name}' was found to belong to both the annotation and the reference. To ignore this and use only the annotation assignment, use `ignore_duplicate_assignments=True`. Assignment that was found is: {elem}")
                    else:
                        return annotation_name
                else:
                    return annotation_name
            else:
                return annotation_name
        else:
            # TODO: Here also the reference name must be kept - does it really? Only if both annotation and reference are in the same dataset
            return elem
    except AttributeError:
        return elem
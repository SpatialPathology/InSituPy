import warnings
from typing import List, Literal, Tuple, Union

import anndata
import numpy as np
import pandas as pd

from insitupy.utils.utils import convert_to_list


def _extract_groups(
    adata,
    groupby,
    groups=None,
    extract_uns=False,
    uns_key='spatial',
    uns_exclusion_pattern=None,
    return_mask=False,
    strip=False
    ):

    '''
    Function to extract a group from a dataframe or anndata object.
    If `anndata_object` is True it expects the dataframe in `adata.obs`
    '''

    # convert groups into list
    if groups is not None:
        groups = [groups] if isinstance(groups, str) else list(groups)

    if type(adata) == anndata.AnnData:
        anndata_object = True
    elif type(adata) == pd.DataFrame:
        anndata_object = False
        extract_uns = False
        strip = False
    else:
        raise ValueError("Unknown type of input object.")

    # select dataframe on which to filter on
    if anndata_object:
        obs = adata.obs
    else:
        obs = adata

    # check if filtering is wanted
    filtering = True
    if groupby is None:
        filtering = False

    # # check if we want to filter `.uns`
    # if uns_exclusion_pattern is not None:
    #     extract_uns = True

    if groupby in obs.columns or not filtering:

        # create filtering mask for groups
        if groupby is None:
            mask = np.full(len(adata), True) # no filtering
        else:
            mask = obs[groupby].isin(groups).values

        # filter dataframe or anndata object
        if anndata_object:
            adata = adata[mask, :].copy()
        else:
            adata = adata.loc[mask, :].copy()

        if len(adata) == 0:
            print("Subset variables '{}' not in groupby '{}'. Object not returned.".format(groups, groupby))
            return
        elif filtering:
            # check if all groups are in groupby category
            groups_found = [group for group in groups if group in obs[groupby].unique()]
            groups_notfound = [group for group in groups if group not in groups_found]

            if len(groups_found) != len(groups):
                print("Following groups were not found in column {}: {}".format(groupby, groups_notfound))

            if extract_uns or uns_exclusion_pattern is not None:
                new_uns = {key:value for (key,value) in adata.uns[uns_key].items() if np.any([group in key for group in groups])}

                if uns_exclusion_pattern is not None:
                    new_uns = {key:value for (key,value) in new_uns.items() if uns_exclusion_pattern not in key}
                adata.uns[uns_key] = new_uns

        if strip:
            # remove annotations in .uns and obsp
            stores = [adata.uns, adata.obsp]
            for s in stores:
                keys = list(s.keys())
                for k in keys:
                    del s[k]

        if return_mask:
            return adata, mask
        else:
            return adata

    else:
        print("Subset category '{}' not found".format(groupby))
        return


def _select_anndata_elements(
    adata,
    obs_keys=None,
    var_keys=None,
    obsm_keys=None,
    varm_keys=None,
    uns_keys=None,
    layer_keys=None,
    inplace=False
    ):
    """
    Select specific columns or keys from obs, var, obsm, uns, and layers in an AnnData object and remove all others.

    Args:
        adata (AnnData): The AnnData object to be modified.
        obs_keys (List[str], optional): List of keys to keep in adata.obs. Defaults to None.
        var_keys (List[str], optional): List of keys to keep in adata.var. Defaults to None.
        obsm_keys (List[str], optional): List of keys to keep in adata.obsm. Defaults to None.
        varm_keys (List[str], optional): List of keys to keep in adata.varm. Defaults to None.
        uns_keys (List[str], optional): List of keys to keep in adata.uns. Defaults to None.
        layer_keys (List[str], optional): List of keys to keep in adata.layers. Defaults to None.
        inplace (bool, optional): Whether to modify the AnnData object in place or return a new modified object. Defaults to False.

    Returns:
        AnnData: Modified AnnData object with only the specified keys/columns if inplace is False, otherwise None.
    """
    if not inplace:
        adata = adata.copy()

    # .obs
    if obs_keys is None:
        adata.obs = adata.obs[[]]
    elif obs_keys == 'all':
        pass  # Keep all keys
    else:
        obs_keys = convert_to_list(obs_keys)
        missing_keys = [key for key in obs_keys if key not in adata.obs.columns]
        if missing_keys:
            warnings.warn(f"Keys not found in adata.obs: {missing_keys}")
        obs_keys = [key for key in obs_keys if key in adata.obs.columns]
        adata.obs = adata.obs[obs_keys]

    # .var
    if var_keys is None:
        adata.var = adata.var[[]]
    elif var_keys == 'all':
        pass  # Keep all keys
    else:
        var_keys = convert_to_list(var_keys)
        missing_keys = [key for key in var_keys if key not in adata.var.columns]
        if missing_keys:
            warnings.warn(f"Keys not found in adata.var: {missing_keys}")
        var_keys = [key for key in var_keys if key in adata.var.columns]
        adata.var = adata.var[var_keys]

    # .obsm
    if obsm_keys is None:
        keys_to_remove = list(adata.obsm.keys())
        for key in keys_to_remove:
            del adata.obsm[key]
    elif obsm_keys == 'all':
        pass  # Keep all keys
    else:
        obsm_keys = convert_to_list(obsm_keys)
        missing_keys = [key for key in obsm_keys if key not in adata.obsm.keys()]
        if missing_keys:
            warnings.warn(f"Keys not found in adata.obsm: {missing_keys}")
        keys_to_remove = set(adata.obsm.keys()) - set(obsm_keys)
        for key in keys_to_remove:
            del adata.obsm[key]

    # .varm
    if varm_keys is None:
        keys_to_remove = list(adata.varm.keys())
        for key in keys_to_remove:
            del adata.varm[key]
    elif varm_keys == 'all':
        pass  # Keep all keys
    else:
        varm_keys = convert_to_list(varm_keys)
        missing_keys = [key for key in varm_keys if key not in adata.varm.keys()]
        if missing_keys:
            warnings.warn(f"Keys not found in adata.varm: {missing_keys}")
        keys_to_remove = set(adata.varm.keys()) - set(varm_keys)
        for key in keys_to_remove:
            del adata.varm[key]

    # .uns
    if uns_keys is None:
        keys_to_remove = list(adata.uns.keys())
        for key in keys_to_remove:
            del adata.uns[key]
    elif uns_keys == 'all':
        pass  # Keep all keys
    else:
        uns_keys = convert_to_list(uns_keys)
        missing_keys = [key for key in uns_keys if key not in adata.uns.keys()]
        if missing_keys:
            warnings.warn(f"Keys not found in adata.uns: {missing_keys}")
        keys_to_remove = set(adata.uns.keys()) - set(uns_keys)
        for key in keys_to_remove:
            del adata.uns[key]

    # .layers
    if layer_keys is None:
        keys_to_remove = list(adata.layers.keys())
        for key in keys_to_remove:
            del adata.layers[key]
    elif layer_keys == 'all':
        pass  # Keep all keys
    else:
        layer_keys = convert_to_list(layer_keys)
        missing_keys = [key for key in layer_keys if key not in adata.layers.keys()]
        if missing_keys:
            warnings.warn(f"Keys not found in adata.layers: {missing_keys}")
        keys_to_remove = set(adata.layers.keys()) - set(layer_keys)
        for key in keys_to_remove:
            del adata.layers[key]

    if not inplace:
        return adata

FilterMode = Literal[
    "contains", "not contains", "starts with", "ends with",
    "is equal", "is not", "in", "not in",
    "greater than", "less than", "greater or equal", "less or equal"
    ]

def filter_anndata(
    adata: anndata.AnnData,
    filter_mode: FilterMode,
    filter_tuple: Tuple[str, Union[str, int, float, List[Union[str, int, float]]]]
) -> anndata.AnnData:
    """
    Filters an AnnData object based on a specified filter mode and condition.

    Args:
        adata (AnnData): The AnnData object to be filtered.
        filter_mode (FilterMode): The filtering mode to apply.
        filter_tuple (tuple): A tuple of the form (column_name, value) specifying the column
            in `adata.obs` and the value to filter by.

    Returns:
        AnnData: A new AnnData object containing only the filtered observations.

    Raises:
        ValueError: If the value type is incompatible with the filter mode.
    """
    column, value = filter_tuple

    if column not in adata.obs:
        raise ValueError(f"Column '{column}' does not exist in the AnnData object.")

    string_based_modes = ["contains", "not contains", "starts with", "ends with", "is equal", "is not", "in", "not in"]
    numeric_based_modes = ["greater than", "less than", "greater or equal", "less or equal"]

    if filter_mode in string_based_modes and not (
        isinstance(value, str) or
        (filter_mode in ["in", "not in"] and isinstance(value, list) and all(isinstance(v, str) for v in value))
    ):
        raise ValueError(f"Filter mode '{filter_mode}' expects a string or list of strings as value.")

    if filter_mode in numeric_based_modes and not isinstance(value, (int, float)):
        raise ValueError(f"Filter mode '{filter_mode}' expects a numeric value (int or float).")

    if filter_mode == "contains":
        mask = adata.obs[column].str.contains(value)
    elif filter_mode == "not contains":
        mask = ~adata.obs[column].str.contains(value)
    elif filter_mode == "starts with":
        mask = adata.obs[column].str.startswith(value)
    elif filter_mode == "ends with":
        mask = adata.obs[column].str.endswith(value)
    elif filter_mode == "is equal":
        mask = adata.obs[column] == value
    elif filter_mode == "is not":
        mask = adata.obs[column] != value
    elif filter_mode == "in":
        mask = adata.obs[column].isin(value)
    elif filter_mode == "not in":
        mask = ~adata.obs[column].isin(value)
    elif filter_mode == "greater than":
        mask = adata.obs[column] > value
    elif filter_mode == "less than":
        mask = adata.obs[column] < value
    elif filter_mode == "greater or equal":
        mask = adata.obs[column] >= value
    elif filter_mode == "less or equal":
        mask = adata.obs[column] <= value
    else:
        raise ValueError(f"Unsupported filter mode: {filter_mode}")

    return adata[mask].copy()


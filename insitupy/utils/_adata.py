import anndata
import numpy as np

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
        uns_keys (List[str], optional): List of keys to keep in adata.uns. Defaults to None.
        layer_keys (List[str], optional): List of keys to keep in adata.layers. Defaults to None.
        inplace (bool, optional): Whether to modify the AnnData object in place or return a new modified object. Defaults to False.

    Returns:
        AnnData: Modified AnnData object with only the specified keys/columns if inplace is False, otherwise None.
    """
    if not inplace:
        adata = adata.copy()

    if obs_keys is None:
        adata.obs = adata.obs[[]]
    elif obs_keys == 'all':
        pass  # Keep all keys
    else:
        obs_keys = convert_to_list(obs_keys)
        adata.obs = adata.obs[obs_keys]

    if var_keys is None:
        adata.var = adata.var[[]]
    elif var_keys == 'all':
        pass  # Keep all keys
    else:
        var_keys = convert_to_list(var_keys)
        adata.var = adata.var[var_keys]

    if obsm_keys is None:
        keys_to_remove = list(adata.obsm.keys())
        for key in keys_to_remove:
            del adata.obsm[key]
    elif obsm_keys == 'all':
        pass  # Keep all keys
    else:
        obsm_keys = convert_to_list(obsm_keys)
        keys_to_remove = set(adata.obsm.keys()) - set(obsm_keys)
        for key in keys_to_remove:
            del adata.obsm[key]

    if uns_keys is None:
        keys_to_remove = list(adata.uns.keys())
        for key in keys_to_remove:
            del adata.uns[key]
    elif uns_keys == 'all':
        pass  # Keep all keys
    else:
        uns_keys = convert_to_list(uns_keys)
        keys_to_remove = set(adata.uns.keys()) - set(uns_keys)
        for key in keys_to_remove:
            del adata.uns[key]

    if layer_keys is None:
        keys_to_remove = list(adata.layers.keys())
        for key in keys_to_remove:
            del adata.layers[key]
    elif layer_keys == 'all':
        pass  # Keep all keys
    else:
        layer_keys = convert_to_list(layer_keys)
        keys_to_remove = set(adata.layers.keys()) - set(layer_keys)
        for key in keys_to_remove:
            del adata.layers[key]

    if not inplace:
        return adata
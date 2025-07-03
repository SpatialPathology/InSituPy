from warnings import warn

import numpy as np
import pandas as pd


def _get_expression_values(adata, X, key_type, key):
    # get expression values
    if key_type == "genes":
        gene_loc = adata.var_names.get_loc(key)
        color_value = X[:, gene_loc]
    elif key_type == "obs":
        color_value = adata.obs[key]
    elif key_type == "obsm":
        #TODO: Implement it for obsm
        obsm_key = key.split("#", maxsplit=1)[0]
        obsm_col = key.split("#", maxsplit=1)[1]
        data = adata.obsm[obsm_key]

        if isinstance(data, pd.DataFrame):
            color_value = data[obsm_col].values
        elif isinstance(data, np.ndarray):
            color_value = data[:, int(obsm_col)-1]
        else:
            warn("Data in `obsm` needs to be either pandas DataFrame or numpy array to be parsed.")
        pass
    else:
        print("Unknown key selected.", flush=True)

    return color_value
"""
Functions in this module were adapted from the decoupler package (https://github.com/scverse/decoupler):
Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D., Müller-Dott S.,
Taus P., Dugourd A., Holland C.H., Ramirez Flores R.O. and Saez-Rodriguez J. 2022.
decoupleR: Ensemble of computational methods to infer biological activities from omics data.
Bioinformatics Advances. https://doi.org/10.1093/bioadv/vbac016

"""


from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack

from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment


def psbulk_profile(profile, mode='sum'):
    if mode == 'sum':
        profile = np.sum(profile, axis=0)
    elif mode == 'mean':
        profile = np.mean(profile, axis=0)
    elif mode == 'median':
        profile = np.median(profile, axis=0)
    elif callable(mode):
        profile = np.apply_along_axis(mode, 0, profile)
    else:
        raise ValueError("""mode={0} can be 'sum', 'mean', 'median' or a callable function.""".format(mode))
    return profile

def extract_psbulk_inputs(
    exp: InSituExperiment,
    cells_layer: str,
    layer: str
    ):

    obs_columns_list = []
    for _, data in exp.iterdata():
        celldata = _get_cell_layer(
            cells=data.cells,
            cells_layer=cells_layer
        )
        obs_columns_list.append(set(celldata.matrix.obs.columns))

    common_obs_columns = sorted(set.intersection(*obs_columns_list))

    obs = pd.DataFrame()
    #obs = {}

    # for c in common_obs_columns:
    #     column_dict={}
    #     for metadata, data in exp.iterdata():
    #         celldata = _get_cell_layer(
    #             cells=data.cells,
    #             cells_layer=cells_layer
    #     )
    #         column_dict[metadata["uid"]] = celldata.matrix.obs[c].copy()

    #     column=pd.concat([
    #         s.reset_index(drop=True) for s in column_dict.values()],
    #                      axis=0)
    #     obs[c]=column

    obs_per_data = {}
    for metadata, data in exp.iterdata():
        celldata = _get_cell_layer(
            cells=data.cells,
            cells_layer=cells_layer
        )

        obs_per_data[metadata["uid"]] = celldata.matrix.obs[common_obs_columns]

    obs = pd.concat(obs_per_data).reset_index(level=0, names="batch")

    var_names = []
    for _, data in exp.iterdata():
        celldata = _get_cell_layer(
                    cells=data.cells,
                    cells_layer=cells_layer
                )
        var_names.append(set(celldata.matrix.var_names))

    common_var_names = sorted(set.intersection(*var_names))

    var = pd.DataFrame({"genes": common_var_names})

        # Extract count matrix X
    X_list=[]
    for metadata, data in exp.iterdata():
            celldata = _get_cell_layer(
                        cells=data.cells,
                        cells_layer=cells_layer
                    )
            ad = celldata.matrix
            ad = ad[:, ad.var_names.isin(common_var_names)]
            if layer is not None:
                X_list.append(ad.layers[layer])
            else:
                X_list.append(ad.X)
    X = vstack(X_list)

    # Sort genes
    msk = np.argsort(var["genes"])

    X = X[:, msk]
    var = var.iloc[msk]
    var.set_index('genes', inplace=True)

    if issparse(X) and not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    return X, obs, var
def format_psbulk_inputs(sample_col, groups_col, obs):
    # Use one column if the same
    if sample_col == groups_col:
        groups_col = None

    if groups_col is None:
        # Filter extra columns in obs
        cols = obs.groupby(sample_col, observed=True).nunique(dropna=False).eq(1).all(0)
        cols = np.hstack([sample_col, cols[cols].index])
        obs = obs.loc[:, cols]

        # Get unique samples
        smples = np.unique(obs[sample_col].values)
        groups = None

        # Get number of samples and features
        n_rows = len(smples)
    else:
        # Check if extra grouping is needed
        if type(groups_col) is list:
            obs = obs.copy()
            joined_cols = '_'.join(groups_col)
            obs[joined_cols] = obs[groups_col[0]].str.cat(obs[groups_col[1:]].astype('U'), sep='_')
            groups_col = joined_cols

        # Filter extra columns in obs
        cols = obs.groupby([sample_col, groups_col], observed=True).nunique(dropna=False).eq(1).all(0)
        cols = np.hstack([sample_col, groups_col, cols[cols].index])
        obs = obs.loc[:, cols]

        # Get unique samples and groups
        smples = np.unique(obs[sample_col].values)
        groups = np.unique(obs[groups_col].values)

        # Get number of samples and features
        n_rows = len(smples) * len(groups)

    return obs, groups_col, smples, groups, n_rows
def psbulk_profile(profile, mode='sum'):
    if mode == 'sum':
        profile = np.sum(profile, axis=0)
    elif mode == 'mean':
        profile = np.mean(profile, axis=0)
    elif mode == 'median':
        profile = np.median(profile, axis=0)
    elif callable(mode):
        profile = np.apply_along_axis(mode, 0, profile)
    else:
        raise ValueError("""mode={0} can be 'sum', 'mean', 'median' or a callable function.""".format(mode))
    return profile
def compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                   new_obs, min_cells, min_counts, mode, dtype):

    # Init empty variables
    psbulk = np.zeros((n_rows, n_cols))
    props = np.zeros((n_rows, n_cols))
    ncells = np.zeros(n_rows)
    counts = np.zeros(n_rows)

    # Iterate for each group and sample
    i = 0
    if groups_col is None:
        for smp in smples:
            # Write new meta-data
            tmp = obs[obs[sample_col] == smp].drop_duplicates().values
            new_obs.loc[smp, :] = tmp

            # Get cells from specific sample
            profile = X[(obs[sample_col] == smp).values]
            if isinstance(X, csr_matrix):
                profile = profile.toarray()

            # Skip if few cells or not enough counts
            ncell = profile.shape[0]
            count = np.sum(profile)
            ncells[i] = ncell
            counts[i] = count
            if ncell < min_cells or np.abs(count) < min_counts:
                i += 1
                continue

            # Get prop of non zeros
            prop = np.sum(profile != 0, axis=0) / profile.shape[0]

            # Pseudo-bulk
            profile = psbulk_profile(profile, mode=mode)

            # Append
            props[i] = prop
            psbulk[i] = profile
            i += 1
    else:
        for grp in groups:
            for smp in smples:
                # Write new meta-data
                index = smp + '-' + grp
                tmp = obs[(obs[sample_col] == smp) & (obs[groups_col] == grp)].drop_duplicates().values
                if tmp.shape[0] == 0:
                    tmp = np.full(tmp.shape[1], np.nan)
                new_obs.loc[index, :] = tmp

                # Get cells from specific sample and group
                profile = X[((obs[sample_col] == smp) & (obs[groups_col] == grp)).values]
                if isinstance(X, csr_matrix):
                    profile = profile.toarray()

                # Skip if few cells or not enough counts
                ncell = profile.shape[0]
                count = np.sum(profile)
                ncells[i] = ncell
                counts[i] = count
                if ncell < min_cells or np.abs(count) < min_counts:
                    i += 1
                    continue

                # Get prop of non zeros
                prop = np.sum(profile != 0, axis=0) / profile.shape[0]

                # Pseudo-bulk
                profile = psbulk_profile(profile, mode=mode)

                # Append
                props[i] = prop
                psbulk[i] = profile
                i += 1

    return psbulk, ncells, counts, props

def filter_by_prop(adata, min_prop=0.2, min_smpls=2):
    """
    Determine which genes are expressed in a sufficient proportion of cells across samples.

    This function selects genes that are sufficiently expressed across cells in each sample and that this condition is
    met across a minimum number of samples.

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_pseudobulk``. It requires ``.layer['psbulk_props']``.
    min_prop : float
        Minimum proportion of cells that express a gene in a sample.
    min_smpls : int
        Minimum number of samples with bigger or equal proportion of cells with expression than ``min_prop``.

    Returns
    -------
    genes : ndarray
        List of genes to be kept.
    """

    # Define limits
    min_prop = np.clip(min_prop, 0, 1)
    min_smpls = np.clip(min_smpls, 0, adata.shape[0])

    if isinstance(adata, AnnData):
        layer_keys = adata.layers.keys()
        if 'psbulk_props' in list(layer_keys):
            var_names = adata.var_names.values.astype('U')
            props = adata.layers['psbulk_props']
            if isinstance(props, pd.DataFrame):
                props = props.values

            # Compute n_smpl
            nsmpls = np.sum(props >= min_prop, axis=0)

            # Set features to 0
            msk = nsmpls >= min_smpls
            genes = var_names[msk]
            return genes
    raise ValueError("""adata must be an AnnData object that contains the layer 'psbulk_props'. Please check the function
                     decoupler.get_pseudobulk.""")

def swap_layer(adata, layer_key, X_layer_key='X', inplace=False):
    """
    Swaps an ``adata.X`` for a given layer.

    Swaps an AnnData ``X`` matrix with a given layer. Generates a new object by default.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_key : str
        ``.layers`` key to place in ``.X``.
    X_layer_key : str, None
        ``.layers`` key where to move and store the original ``.X``. If None, the original ``.X`` is discarded.
    inplace : bool
        If ``False``, return a copy. Otherwise, do operation inplace and return ``None``.

    Returns
    -------
    layer : AnnData, None
        If ``inplace=False``, new AnnData object.
    """

    cdata = None
    if inplace:
        if X_layer_key is not None:
            adata.layers[X_layer_key] = adata.X
        adata.X = adata.layers[layer_key]
    else:
        cdata = adata.copy()
        if X_layer_key is not None:
            cdata.layers[X_layer_key] = cdata.X
        cdata.X = cdata.layers[layer_key]

    return cdata
def check_X(X, mode='sum', skip_checks=False):
    if isinstance(X, csr_matrix):
        is_finite = np.all(np.isfinite(X.data))
    else:
        is_finite = np.all(np.isfinite(X))
    if not is_finite:
        raise ValueError('Data contains non finite values (nan or inf), please set them to 0 or remove them.')
    skip_checks = type(mode) is dict or callable(mode) or skip_checks
    if not skip_checks:
        if isinstance(X, csr_matrix):
            is_positive = np.all(X.data >= 0)
        else:
            is_positive = np.all(X >= 0)
        if not is_positive:
            raise ValueError("""Data contains negative values. Check the parameters use_raw and layers to
            determine if you are selecting the correct matrix. To override this, set skip_checks=True.
            """)
        if mode == 'sum':
            if isinstance(X, csr_matrix):
                is_integer = float(np.sum(X.data)).is_integer()
            else:
                is_integer = float(np.sum(X)).is_integer()
            if not is_integer:
                raise ValueError("""Data contains float (decimal) values. Check the parameters use_raw and layers to
                determine if you are selecting the correct data, which should be positive integer counts when mode='sum'.
                To override this, set skip_checks=True.
                """)
def get_pseudobulk(
    exp,
    sample_col: str,
    groups_col: str,
    cells_layer: Optional[str] = None,
    counts_layer: str = "counts",
    # use_raw: bool = False,
    mode: Literal["sum", "mean", "median"] = "sum",
    min_cells: int = 10,
    min_counts: int = 1000,
    dtype = np.float32,
    skip_checks: bool = False,
    min_prop=None,
    min_smpls=None,
    remove_empty=True
    ):

    min_cells, min_counts = np.clip(min_cells, 1, None), np.clip(min_counts, 1, None)

    X, obs, var = extract_psbulk_inputs(
        exp=exp,
        cells_layer=cells_layer,
        layer=counts_layer
        )

    check_X(X, mode=mode, skip_checks=skip_checks)

    obs, groups_col, smples, groups, n_rows = format_psbulk_inputs(
        sample_col, groups_col, obs
        )

    n_cols = X.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)

    if type(mode) is dict:
        psbulks = []
        for l_name in mode:
            func = mode[l_name]
            if not callable(func):
                raise ValueError("""mode requieres a dictionary of layer names and callable functions. The layer {0} does not
                contain one.""".format(l_name))
            else:
                # Compute psbulk
                psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                                                               new_obs, min_cells, min_counts, func, dtype)
                psbulks.append(psbulk)
        layers = {k: v for k, v in zip(mode.keys(), psbulks)}
        layers['psbulk_props'] = props
    elif type(mode) is str or callable(mode):
        # Compute psbulk
        psbulk, ncells, counts, props = compute_psbulk(n_rows, n_cols, X, sample_col, groups_col, smples, groups, obs,
                                                       new_obs, min_cells, min_counts, mode, dtype)
        layers = {'psbulk_props': props}

    # Add QC metrics
    new_obs['psbulk_cells'] = ncells
    new_obs['psbulk_counts'] = counts

    # Create new AnnData
    psbulk = AnnData(psbulk.astype(dtype), obs=new_obs, var=var, layers=layers)

    # Remove empty samples and features
    if remove_empty:
        msk = psbulk.X == 0
        psbulk = psbulk[~np.all(msk, axis=1), ~np.all(msk, axis=0)].copy()

    # Place first element of mode dict as X
    if type(mode) is dict:
        swap_layer(psbulk, layer_key=list(mode.keys())[0], X_layer_key=None, inplace=True)

    # Filter by genes if not None.
    if min_prop is not None and min_smpls is not None:
        if groups_col is None:
            genes = filter_by_prop(psbulk, min_prop=min_prop, min_smpls=min_smpls)
        else:
            genes = []
            for group in groups:
                g = filter_by_prop(psbulk[psbulk.obs[groups_col] == group], min_prop=min_prop, min_smpls=min_smpls)
                genes.extend(g)
            genes = np.unique(genes)
        psbulk = psbulk[:, genes]

    return psbulk


def plot_psbulk_samples(adata, groupby, figsize=(5, 5), dpi=100, ax=None, return_fig=False, save=None, **kwargs):
    """
    Quality Control plot to assess the quality of the obtained pseudobulk samples.

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_pseudobulk``.
    groupby : str, list
        Name of the ``.obs`` column to group by. Can also be a list of columns.
    figsize : tuple
        Figure size.
    dpi : int
        DPI resolution of figure.
    ax : Axes, None
        A matplotlib axes object. If None returns new figure.
    return_fig : bool
        Whether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending on {``.pdf``, ``.png``, ``.svg``}.
    kwargs : dict
        Other keyword arguments are passed through to seaborn.scatterplot().

    Returns
    -------
    fig : Figure, None
        If return_fig, returns Figure object.
    """

    # Extract obs
    df = adata.obs.copy()

    # Transform to log10
    df['psbulk_cells'] = np.log10(df['psbulk_cells'])
    df['psbulk_counts'] = np.log10(df['psbulk_counts'])

    if type(groupby) is list and ax is not None:
        raise ValueError("""If a groupby is a list of columns ax must be None.""")
    elif type(groupby) is list:
        fig, axes = plt.subplots(1, len(groupby), figsize=figsize, dpi=dpi, tight_layout=True)
        axes = axes.ravel()
        for ax, grp in zip(axes, groupby):
            ax.grid(zorder=0)
            ax.set_axisbelow(True)
            sns.scatterplot(x='psbulk_cells', y='psbulk_counts', hue=grp, ax=ax, data=df, zorder=1, **kwargs)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=grp)
            ax.set_xlabel('Log10 number of cells')
            ax.set_ylabel('Log10 total sum of counts')
    else:
        # Plot
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.grid(zorder=0)
        ax.set_axisbelow(True)
        sns.scatterplot(x='psbulk_cells', y='psbulk_counts', hue=groupby, ax=ax, data=df, zorder=1, **kwargs)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=groupby)
        ax.set_xlabel('Log10 number of cells')
        ax.set_ylabel('Log10 total sum of counts')

    if return_fig:
        return fig


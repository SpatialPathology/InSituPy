import os
from numbers import Number
from typing import Literal, Optional, Tuple

import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy.stats import zscore
from tqdm import tqdm

from insitupy._constants import DEFAULT_CATEGORICAL_CMAP

from .._core._checks import check_raw, has_valid_labels
from ..io.plots import save_and_show_figure
from ..utils._regression import smooth_fit
from ..utils.utils import get_nrows_maxcols


def expr_along_obs_val(
    adata: AnnData,
    keys: str,
    obs_val: str,
    groupby: Optional[str] = None,
    splitby: str = None,
    hue: str = None,
    method: Literal["lowess", "loess"] = 'loess',
    stderr: bool = False,
    loess_bootstrap: bool = True,
    n_bootstraps_iterations: int = 100,
    xmin=None,
    xmax=None,
    cmap="tab10",
    linewidth=8,
    extra_cats=None,
    normalize=False,
    nsteps=100,
    show_progress=False,
    use_raw=False,
    max_cols=4,
    xlabel=None,ylabel=None,
    vline=None, hline=None, vlinewidth=4,
    #values_into_title=None, title_suffix='',
    custom_titles=None,
    legend_fontsize=24,
    plot_legend=True,
    xlabel_fontsize=28, ylabel_fontsize=28, title_fontsize=20, tick_fontsize=24,
    figsize=(8,6),
    savepath: Optional[os.PathLike] = None,
    save_only: bool = False,
    show: bool = True,
    return_data: bool = False,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    dpi_save: int = 300,
    smooth=True,
    **kwargs
    ):
    """
    Plot gene expression values along a specified observation category.

    Args:
        adata (AnnData): Annotated data matrix.
        keys (str): Keys for the gene expression values to be plotted.
        obs_val (str): Observation category to be plotted on the x-axis.
        groupby (Optional[str]): Observation category to group by.
        splitby (str, optional): Observation category to split by.
        hue (str, optional): Observation category to color by.
        method (Literal["lowess", "loess"], optional): Smoothing method to use. Defaults to 'loess'.
        stderr (bool, optional): Whether to plot standard error. Defaults to False.
        loess_bootstrap (bool, optional): Whether to use bootstrap for loess smoothing. Defaults to True.
        n_bootstraps_iterations (int, optional): Number of bootstrap iterations for loess smoothing. Defaults to 100.
        xmin (optional): Minimum x value for plotting.
        xmax (optional): Maximum x value for plotting.
        cmap (str, optional): Colormap to use for plotting. Defaults to "tab10".
        linewidth (int, optional): Line width for plotting. Defaults to 8.
        extra_cats (optional): Additional observation categories to include in the plot.
        normalize (bool, optional): Whether to normalize the expression values. Defaults to False.
        nsteps (int, optional): Number of steps for smoothing. Defaults to 100.
        show_progress (bool, optional): Whether to show progress bar. Defaults to False.
        use_raw (bool, optional): Whether to use raw data. Defaults to False.
        max_cols (int, optional): Maximum number of columns for subplots. Defaults to 4.
        xlabel (optional): Label for the x-axis.
        ylabel (optional): Label for the y-axis.
        vline (optional): Vertical lines to add to the plot.
        hline (optional): Horizontal lines to add to the plot.
        vlinewidth (int, optional): Line width for vertical lines. Defaults to 4.
        custom_titles (optional): Custom titles for the plots.
        legend_fontsize (int, optional): Font size for the legend. Defaults to 24.
        plot_legend (bool, optional): Whether to plot the legend. Defaults to True.
        xlabel_fontsize (int, optional): Font size for the x-axis label. Defaults to 28.
        ylabel_fontsize (int, optional): Font size for the y-axis label. Defaults to 28.
        title_fontsize (int, optional): Font size for the plot titles. Defaults to 20.
        tick_fontsize (int, optional): Font size for the axis ticks. Defaults to 24.
        figsize (tuple, optional): Figure size. Defaults to (8, 6).
        savepath (optional): Path to save the plot.
        save_only (bool, optional): Whether to only save the plot without showing. Defaults to False.
        show (bool, optional): Whether to show the plot. Defaults to True.
        axis (optional): Axis to plot on.
        return_data (bool, optional): Whether to return the data instead of plotting. Defaults to False.
        fig (optional): Figure to plot on.
        dpi_save (int, optional): DPI for saving the plot. Defaults to 300.
        smooth (bool, optional): Whether to apply smoothing. Defaults to True.
        **kwargs: Additional arguments for smoothing.

    Returns:
        Union[DataFrame, Tuple[Figure, Axes]]:
            If return_data is True, returns a DataFrame with the smoothed data.
            Otherwise, returns the figure and axes of the plot.
    """

    # check type of input
    if isinstance(keys, dict):
        if custom_titles is not None:
            print("Attention: `custom_titles` was not None and `keys` was dictionary. Titles were retrieved from dictionary.")
        custom_titles = list(keys.keys())
        keys = list(keys.values())

    # make inputs to lists
    keys = [keys] if isinstance(keys, str) else list(keys)

    # remove NaNs `obs_val` column
    adata_obs = adata.obs.copy()
    not_na_and_not_zero_mask = adata_obs[obs_val].notna() & adata_obs[obs_val] > 0
    adata_obs = adata_obs[not_na_and_not_zero_mask]

    # check whether to plot raw data
    X, var, var_names = check_raw(adata, use_raw=use_raw)

    # remove rows from X which were NaN above
    X = X[not_na_and_not_zero_mask]

    if hue is not None:
        hue_cats = list(adata_obs[hue].unique())
        cmap_colors = plt.get_cmap(cmap)
        color_dict = {a: cmap_colors(i) for i, a in enumerate(hue_cats)}

        if extra_cats is None:
            extra_cats = [hue]
        else:
            extra_cats.append(hue)

    #if show:
    if not return_data:
        # prepare plotting
        if axis is None:
            n_plots, n_rows, max_cols = get_nrows_maxcols(keys, max_cols)
            fig, axs = plt.subplots(n_rows,max_cols, figsize=(figsize[0]*max_cols, figsize[1]*n_rows))

        else:
            axs = axis
            #fig = None
            n_plots = 1
            show = False # otherwise plotting into given axes wouldn't work

        if n_plots > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

    data_collection = {}
    for i, key in (enumerate(tqdm(keys)) if show_progress else enumerate(keys)):
        # check if the keys are also grouped
        keys_grouped = isinstance(key, list)

        if groupby is not None:
            # select data per group
            groups = adata_obs[groupby].unique()
        else:
            groups = [None]

        added_to_legend = []

        group_collection = {}
        for group in groups:
            #partial = extract_groups(adata, groupby=groupby, groups=group)

            if group is not None:
                group_mask = adata_obs[groupby] == group
                group_obs = adata_obs.loc[group_mask, :].copy()
            else:
                group_mask = [True] * len(adata_obs)
                group_obs = adata_obs

            if hue is not None:
                _hue = adata_obs.loc[group_mask, hue][0]

            # hue_data = adata_obs.loc[group_mask, hue].copy()
            # print(hue_data)

            # select only group values from matrix
            group_X = X[group_mask, :]

            if splitby is None:
                # select x value
                x = group_obs.loc[:, obs_val].values
                # if xmin is None:
                #     xmin = x[x>0].min()
                #     print(xmin, flush=True)

                if keys_grouped:
                    # extract expression values of all keys in the group
                    idx = var.index.get_indexer(key)
                    dd = pd.DataFrame(group_X[:, idx], index=x)

                    if normalize:
                        #dd = dd.apply(minmax_scale, axis=0)
                        dd = dd.apply(zscore, axis=0)

                    dd = dd.reset_index().melt(id_vars="index") # reshape to get long list of x values
                    x = dd["index"].values
                    y = dd["value"].values

                elif key in var_names:
                    # extract expression values as y
                    idx = var.index.get_loc(key)
                    y = group_X[:, idx].copy()

                    if normalize:
                        #y = minmax_scale(y)
                        y = zscore(y)

                elif key in group_obs.columns:
                    y = group_obs.loc[:, key].values.copy()
                else:
                    print("Key '{}' not found.".format(key))
                    break

                if smooth:
                    # do smooth fitting
                    df = smooth_fit(x, y,
                                    xmax=xmin, xmin=xmax,
                                    nsteps=nsteps, method=method,
                                    stderr=stderr, loess_bootstrap=loess_bootstrap,
                                    K=n_bootstraps_iterations,
                                    **kwargs)
                else:
                    # set up dataframe without smooth fitting
                    df = pd.DataFrame({"x": x, "y_pred": y})

                if extra_cats is not None:
                    df = df.join(adata_obs.loc[group_mask, extra_cats].reset_index(drop=True))

            else:
                splits = group_obs[splitby].unique()
                df_collection = {}

                # get min and max values for x values
                x = group_obs[obs_val].values
                xmin = x.min()
                xmax = x.max()

                for split in splits:
                    # extract x values
                    split_mask = group_obs[splitby] == split
                    x = group_obs.loc[split_mask, obs_val].values

                    # extract expression values as y
                    idx = var.index.get_loc(key)
                    y = group_X[split_mask, idx].copy()

                    # do smooth fitting
                    if smooth:
                        df_split = smooth_fit(x, y,
                                xmax=xmin, xmin=xmax,
                                nsteps=nsteps, method=method, stderr=stderr, **kwargs)
                    else:
                        # set up dataframe without smooth fitting
                        df_split = pd.DataFrame({"x": x, "y_pred": y})

                    # collect data
                    df_collection[split] = df_split

                df_collection = pd.concat(df_collection)

                # calculate mean and std
                df = df_collection[['x', 'y_pred']].groupby('x').mean()
                df['std'] = df_collection[['x', 'y_pred']].groupby('x').std()
                df['conf_lower'] = [a-b for a,b in zip(df['y_pred'], df['std'])]
                df['conf_upper'] = [a+b for a,b in zip(df['y_pred'], df['std'])]
                df.reset_index(inplace=True)

            # remove NaNs
            df = df.dropna(how="all", axis=1)

            if return_data:
                group_collection[group] = df
            else:
                # sort by x-value
                df.sort_values('x', inplace=True)

                # plotting
                cols = df.columns
                if 'conf_lower' in cols and 'conf_upper' in cols:
                    axs[i].fill_between(df['x'],
                                    df['conf_lower'],
                                    df['conf_upper'],
                                    alpha = 0.2,
                                    color = 'grey')

                # determine label variable
                if hue is not None:
                    label = _hue if _hue not in added_to_legend else ""
                    color = color_dict[_hue]
                else:
                    label = group
                    color = None

                axs[i].plot(df['x'],
                    df['y_pred'],
                    label=label,
                    color=color,
                    linewidth=linewidth)

                if hue is not None and _hue not in added_to_legend:
                    added_to_legend.append(_hue)

        # optionally add vertical or horizontal lines to plot
        if vline is not None:
            if isinstance(vline, dict):
                linecolors = list(vline.keys())
                vline = list(vline.values())
            else:
                vline = [vline] if isinstance(vline, int) or isinstance(vline, float) else list(vline)
                linecolors = ['k'] * len(vline)

            for c, v in zip(linecolors, vline):
                axs[i].axvline(x=v, ymin=0, ymax=1, c=c, linewidth=vlinewidth, linestyle='dashed')

        if hline is not None:
            if isinstance(hline, dict):
                linecolors = list(hline.keys())
                hline = list(hline.values())
            else:
                hline = [hline] if isinstance(hline, int) or isinstance(hline, float) else list(hline)
                linecolors = ['k'] * len(hline)

            for c, h in zip(linecolors, hline):
                axs[i].axhline(y=h, xmin=0, xmax=1, c=c, linewidth=4, linestyle='dashed')

        if not return_data:
            if xlabel is None:
                xlabel = obs_val
            if ylabel is None:
                ylabel = "Gene expression"

            axs[i].set_xlabel(xlabel, fontsize=xlabel_fontsize)
            axs[i].set_ylabel(ylabel, fontsize=ylabel_fontsize)
            axs[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            #axs[i].set_xlim(0, 1)
            #axs[i].xaxis.set_major_locator(ticker.FixedLocator([0.1, 0.9]))

            if custom_titles is None:
                axs[i].set_title(key, fontsize=title_fontsize)
            else:
                assert len(custom_titles) == len(keys), "List of title values has not the same length as list of keys."
                axs[i].set_title(str(custom_titles[i]), fontsize=title_fontsize)

            if plot_legend:
                if has_valid_labels(axs[i]):
                    axs[i].legend(fontsize=legend_fontsize,
                    loc='best'
                    )
            else:
                # first check if there are valid labels in the axis to circumvent warning
                if has_valid_labels(axs[i]):
                    axs[i].legend().remove()

        if return_data:
            if len(group_collection) > 0:
                # collect data
                group_collection = pd.concat(group_collection)
                data_collection[key] = group_collection
            else:
                pass



    if return_data:
        # close plot
        plt.close()

        # return data
        data_collection = pd.concat(data_collection)
        data_collection.index.names = ['key', groupby, None]
        return data_collection

    else:
        if n_plots > 1:

            # check if there are empty plots remaining
            while i < n_rows * max_cols - 1:
                i+=1
                # remove empty plots
                axs[i].set_axis_off()
        if show:
            #fig.tight_layout()
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)
        else:
            return fig, axs

def cell_abundance_along_obs_val(
    adata: AnnData,
    obs_val: str,
    groupby: Optional[str] = None,
    xmin: Number = 0,
    savepath: Optional[os.PathLike] = None,
    figsize: Tuple = (8,6),
    save_only: bool = False,
    dpi_save: int = 300,
    histplot_multiple: str = "stack",
    histplot_element: str = "bars"
    ):
    # get data for plotting
    data = adata.obs[[obs_val, groupby]].dropna()

    # remove zeros
    data = data[data[obs_val] > xmin].copy()

    # Create the histogram
    fig, ax = plt.subplots(1,1, figsize=(figsize[0], figsize[1]))
    h = sns.histplot(data=data, x=obs_val,
                hue=groupby, palette=DEFAULT_CATEGORICAL_CMAP.colors,
                multiple=histplot_multiple, element=histplot_element,
                alpha=1, ax=ax
                )

    # Move the legend outside of the plot
    sns.move_legend(h, "upper left", bbox_to_anchor=(1, 1))

    # save or show figure
    save_and_show_figure(savepath=savepath,
                         fig=h.get_figure(),
                         save_only=save_only,
                         dpi_save=dpi_save,
                         tight=True
                         )
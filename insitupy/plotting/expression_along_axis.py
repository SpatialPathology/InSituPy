import os
import textwrap
import warnings
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from insitupy._constants import DEFAULT_CATEGORICAL_CMAP, _init_mpl_fontsize
from insitupy._core._checks import check_raw, has_valid_labels
from insitupy.io.plots import save_and_show_figure
from insitupy.utils._regression import smooth_fit
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)


def cell_expression_along_axis(
    adata,
    axis,
    genes: List[str],
    cell_type_column: str,
    cell_type: Union[str, List[str]],
    xlim: Tuple[Union[int, float], Union[int, float]] = (0, np.inf),
    min_expression: Union[int, float] = 0,
    xlabel: Optional[str] = None,
    fit_reg: bool = False,
    kde: bool = False,
    max_cols: bool = 4,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    fig_height: Number = 4,
    fig_marginal_ratio: Number = 0.15,
    scatter_size: Number = 1,
    wspace: Number = 0.15,
    hspace: Number = 0.25,
    font_scale_factor: Number = 1
):

    """
    Plot gene expression along a specified axis for a given cell type.

    This function visualizes gene expression for a selected cell type along a continuous axis
    (e.g., pseudotime, spatial coordinate, etc.). It creates a grid of subplots with scatter
    plots, optional regression fits, and marginal KDE plots for both the axis and gene expression.

    Args:
        adata: AnnData object containing the single-cell data.
        axis: Observation key in `adata.obs` to plot along the x-axis (e.g., pseudotime) or a tuple of a key in `adata.obsm` and a column name key in the respective table: (obsm key, column key).
        genes (List[str]): List of gene names to plot.
        cell_type_column: Column name in `adata.obs` that contains cell type annotations.
        cell_type Union[str, List[str]]: Specific cell type or a list of cell types to filter the data for plotting.
        xlim (Tuple[Union[int, float], Union[int, float]], optional): Limits for the x-axis. Defaults to (0, np.inf).
        min_expression (Union[int, float], optional): Minimum expression threshold to include a cell. Defaults to 0.
        xlabel (Optional[str], optional): Custom label for the x-axis. If None, uses the `axis` name. Defaults to None.
        fit_reg (bool, optional): Whether to fit a LOESS regression line to the scatter plot. Defaults to False.
        kde (bool, optional): Whether to overlay a 2D KDE plot on the scatter plot. Defaults to False.
        max_cols (int, optional): Maximum number of columns in the subplot grid. The actual number of columns will be the
            minimum of this value and the number of genes. Defaults to 4.
        savepath (Union[str, os.PathLike, Path], optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
        save_only (bool, optional): If True, the figure is saved but not shown. Defaults to False.
        dpi_save (int, optional): DPI for saving the figure. Defaults to 300.
        fig_height (Number, optional): Height of each subplot row. Defaults to 4.
        fig_marginal_ratio (Number, optional): Height ratio of the marginal KDE plot to the main plot. Defaults to 0.15.
        scatter_size (Number, optional): Size of scatter plot points. Defaults to 1.
        wspace (Number, optional): Width space between subplots. Defaults to 0.15.
        hspace (Number, optional): Height space between subplot rows. Defaults to 0.25.
        font_scale_factor (Number, optional): Scaling factor for font sizes. Defaults to 1.

    Returns:
        None: Displays and/or saves the generated plot.
    """

    # reset matplotlib settings
    _init_mpl_fontsize(scale_factor=font_scale_factor)

    # make sure genes is a list
    genes = convert_to_list(genes)

    # select the data for plotting
    data_for_one_celltype = _select_data(
        adata=adata,
        obs_val=axis,
        cell_type_column=cell_type_column,
        cell_type=cell_type,
        genes=genes,
        min_expression=min_expression,
        xlim=xlim,
    )
    
    # create xlabel string
    if xlabel is None:
        xlabel_str = " ".join(convert_to_list(axis))
        xlabel_str = textwrap.fill(xlabel_str, width=15)
    else:
        xlabel_str = xlabel

    # Prepare a figure with subplots
    num_genes = len(genes)
    num_rows = (num_genes + max_cols - 1) // max_cols
    num_cols = min(max_cols, num_genes)
    marg_height = fig_height * fig_marginal_ratio
    fig, axes = plt.subplots(num_rows + 1, num_cols * 2,
                             figsize=(fig_height * (1-hspace) * num_cols + marg_height,
                                      fig_height * (1-wspace) * num_rows + marg_height),
                             sharey='row', sharex='col',
                             gridspec_kw={'height_ratios': [marg_height] + [fig_height]*num_rows,
                                          'width_ratios': [fig_height, marg_height]*num_cols
                                          }
                             )

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for i, gene in enumerate(genes):
        row = i // num_cols + 1
        col = i % num_cols * 2

        if row == 1:
            # Histogram for the x-axis density
            sns.kdeplot(data=data_for_one_celltype, x="axis", ax=axes[0, col],
                        color='darkgray', fill=True)

            # remove the empty axes in between the plots
            axes[0, col+1].remove()

            # remove values axis from histogram
            axes[0, col].get_yaxis().set_visible(False)
            axes[0, col].spines['left'].set_visible(False)
            axes[0, col].spines['right'].set_visible(False)
            axes[0, col].spines['top'].set_visible(False)

        # set xlabel in the last row
        if row == num_rows:
            axes[row, col].set_xlabel(xlabel_str)

        # set ylabel in the first column
        if col == 0 and row != 0:
            axes[row, col].set_ylabel("Gene expression")

        # select data for current gene
        data_for_one_gene = data_for_one_celltype[[gene]].copy()

        if kde:
            # KDE plot
            sns.kdeplot(data=data_for_one_gene.reset_index("axis"),
                        x="axis", y=gene,
                        ax=axes[row, col],
                        fill=True, cmap="Reds", levels=8)

        # Scatter plot
        axis_values = data_for_one_gene.index.get_level_values('axis').values
        expr_values = data_for_one_gene[gene].values

        # drop NaNs
        not_nan = ~np.isnan(expr_values)
        axis_values = axis_values[not_nan]
        expr_values = expr_values[not_nan]

        axes[row, col].scatter(
            x=axis_values,
            y=expr_values,
            alpha=0.5, color='k', s=2
            )

        if fit_reg:
            if len(axis_values) > 1:
                try:
                    # perform loess regression for the second half of the plot
                    res = smooth_fit(
                    xs=axis_values,
                    ys=expr_values, # make sure there are no NaN in the data
                    loess_bootstrap=False, nsteps=100
                    )
                except ValueError as e:
                    print(f"A ValueError occurred during loess regression: {e}")
                    res = None
            else:
                print(f"Only one datapoint left for gene {gene} after filtering. Skipped LOESS regression.")
                res = None

            if res is not None:
                axes[row, col].plot(res["x"], res["y_pred"],
                            color='royalblue', linewidth=3,
                            #label=reg_label
                            )
                axes[row, col].fill_between(res["x"],
                                    res["conf_lower"],
                                    res["conf_upper"],
                                    color='royalblue',
                                    alpha=0.2,
                                    #label='95% CI of Loess Regression'
                                    )


        # Histogram for the gene expression
        sns.kdeplot(
            data=data_for_one_gene, y=gene, ax=axes[row, col+1], color='darkgray', fill=True
        )

        # remove values axis from histogram
        axes[row, col+1].get_xaxis().set_visible(False)
        axes[row, col+1].spines['bottom'].set_visible(False)
        axes[row, col+1].spines['right'].set_visible(False)
        axes[row, col+1].spines['top'].set_visible(False)

        # Set labels
        #axes[row, col].set_ylabel(f"{gene} in '{cell_type}'")
        #axes[row, col].set_ylabel(f"{gene}")
        axes[row, col].set_title(f"{gene}", y=1.0#pad=-2
                                 )

    plt.suptitle(f"Gene expression in '{cell_type}'")

    # Turn off empty subplots
    # total_plots = (num_rows + 1) * num_cols * 2
    for i in range(len(genes), num_cols * num_rows):
        row = i // num_cols + 1
        col = i % num_cols * 2
        axes[row, col].set_axis_off()
        axes[row, col + 1].set_axis_off()

        if row == 1:
            axes[0, col].set_axis_off()
            axes[0, col+1].set_axis_off()

        # add the xlabel to the plot above
        axes[row-1, col].set_xlabel(xlabel_str)

    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=False)


def _select_data(
    adata,
    obs_val,
    genes: List[str],
    cell_type_column,
    cell_type,
    xlim: Tuple[Union[int, float], Union[int, float]] = (0, np.inf),
    min_expression: Number = None,
    sort: bool = True,
    minmax_scale: bool = True,
    verbose: bool = True
):
    # make sure genes is a list
    genes = convert_to_list(genes)

    # Check type of obs_val
    adata_obs = adata.obs.copy()
    if isinstance(obs_val, tuple):
        print("Retrieve `obs_val` from .obsm.") if verbose else None
        obsm_key = obs_val[0]
        obsm_col = obs_val[1]
        #obs_val = f"distance_from_{obsm_col}"
        adata_obs["axis"] = adata.obsm[obsm_key][obsm_col]
    else:
        adata_obs["axis"] = adata.obs[obs_val]

    # Get data for plotting
    data = adata_obs[["axis", cell_type_column]].dropna()

    # Filter data for the specified cell type
    if isinstance(cell_type, str):
        selected_data = data[data[cell_type_column] == cell_type].copy()
    else:
        selected_data = data[data[cell_type_column].isin(cell_type)].copy()

    # Apply limits
    selected_data = selected_data[
        (selected_data["axis"] >= xlim[0]) &
        (selected_data["axis"] <= xlim[1])
        ]

    for i, gene in enumerate(genes):
        # Add gene expression information
        gene_loc = adata.var_names.get_loc(gene)
        expr = adata.X[:, gene_loc]
        expr = pd.Series(expr.toarray().flatten(), index=adata.obs_names)

        if min_expression is not None:
            # mask values below the threshold with NaN
            expr = expr.mask(expr < min_expression)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            # add gene expression to dataframe
            selected_data[gene] = expr

        # if min_expression is not None:
        #     # Apply limits
        #     #data_for_one_celltype = data_for_one_celltype[data_for_one_celltype[gene] >= min_expression]

    # drop the cell type column
    #selected_data = selected_data.drop(cell_type_column, axis=1)

    # add axis column to index
    selected_data = selected_data.set_index(["axis", cell_type_column], append=True)
    selected_data.index.names = ["cell_id", "axis", cell_type_column]

    if sort:
        #selected_data = selected_data.sort_values("axis")
        selected_data = selected_data.sort_index(level='axis', ascending=True)

    if minmax_scale:
        scaler = MinMaxScaler()
        selected_data = pd.DataFrame(scaler.fit_transform(selected_data),
                    index=selected_data.index, columns=selected_data.columns
                    )

    return selected_data

# binning function
def _bin_data(data,
              #expr, axis,
              axis_name: str = "axis",
              resolution=5,
              #minmax_scale: bool = True,
              plot: bool = False
              ):

    # make a copy of the data
    data = data.copy()

    # get values
    axis_data = data.index.get_level_values(axis_name).values

    # get gene names
    genes = [elem for elem in data.columns if elem != axis_name]

    if resolution > 0:
        # calculate number of bins from resolution
        nbins = int((axis_data.max() - axis_data.min()) / resolution)
        data["bin"] = pd.cut(axis_data, bins=nbins)

        # calculate mean per bin
        binned_mean = data.groupby("bin")[genes].mean()

        # extract the center of each bin
        binned_mean.index = [elem.mid for elem in binned_mean.index]
        binned_mean.index.name = "axis"
    else:
        # no actual binning is performed - data is just formated in the same way
        binned_mean = data.droplevel(["cell_id", "cell_type"])
        binned_mean.index.name = "axis"

    if plot:
        _bin_qc_plot(binned_data=binned_mean, raw_data=data)

    return binned_mean

def _bin_qc_plot(
    binned_data, raw_data, xlabel='x'
):
    # extract values from binned data
    bin_center = binned_data.index.values
    genes = binned_data.columns
    n_genes = len(genes)

    # extract values from raw data
    raw_axis = raw_data.index.get_level_values("axis").values

    fig, axs = plt.subplots(n_genes, 2, figsize=(8, 4*n_genes),
                            sharex='col', sharey='row'
                            )

    if n_genes == 1:
        # reshape the axis array so that 2d indexing works later
        axs = axs.reshape(1,2)

    for i, gene in enumerate(genes):
        expr = binned_data[gene].values
        raw_expr = raw_data[gene].values

        not_nan = ~np.isnan(expr)

        try:
            # perform loess regression for the second half of the plot
            res = smooth_fit(
            xs=bin_center[not_nan],
            ys=expr[not_nan], # make sure there are no NaN in the data
            loess_bootstrap=False, nsteps=100
            )
        except ValueError as e:
            print(f"A ValueError occurred during loess regression: {e}")
            res = None

        # Plot the original data
        axs[i, 0].scatter(
            raw_axis, raw_expr, label='Original Data', alpha=0.5, color='k', s=1
            )

        # Plot the binned values
        axs[i, 0].plot(
            bin_center, expr,
            color='firebrick',
            alpha=0.5,
            linestyle='-', label='Binned Mean')

        # Add labels and legend
        axs[i, 0].set_xlabel(xlabel)
        axs[i, 0].set_ylabel(f"Expression of '{gene}'")
        axs[i, 0].legend()

        #axs[1].plot(binned_data["bin_center"], binned_data["minmax"])
        axs[i, 1].scatter(
            x=bin_center, y=expr,s=1, color="k", label="Binned Mean")
        if res is not None:
            axs[i, 1].plot(res["x"], res["y_pred"], color='royalblue',
                           label="Loess Regression")
            axs[i, 1].fill_between(res["x"],
                                   res["conf_lower"],
                                   res["conf_upper"],
                                   color='royalblue',
                                   alpha=0.2, label='Confidence Interval')

        axs[i, 1].legend()
        axs[i, 1].set_xlabel(xlabel)
        axs[i, 1].set_ylabel(f"Scaled expression of '{gene}'")

    # Show plot
    plt.tight_layout()
    plt.show()

def cell_abundance_along_axis(
    adata: AnnData,
    axis: Union[str, Tuple[str, str]],
    groupby: Optional[str] = None,
    xlim: Tuple = (0, np.inf),
    savepath: Optional[os.PathLike] = None,
    figsize: Tuple = (8,6),
    save_only: bool = False,
    dpi_save: int = 300,
    multiple: Literal["layer", "dodge", "stack", "fill"] = "stack",
    histplot_element: Literal["bars", "step", "poly"] = "bars",
    kde: bool = False
    ):

    """
    Plot cell abundance along a specified observation value.

    Args:
        adata (AnnData): Annotated data matrix.
        axis (Union[str, Tuple[str, str]]): Observation category to be plotted on the x-axis.
            Can be a string representing a column in `adata.obs` or a tuple (obsm_key, obsm_col)
            where `obsm_key` is a key in `adata.obsm` and `obsm_col` is a column in the corresponding DataFrame.
        groupby (Optional[str], optional): Column in `adata.obs` to group by. Defaults to None.
        xmin (Number, optional): Minimum value of `obs_val` to include in the plot. Defaults to 0.
        savepath (Optional[os.PathLike], optional): Path to save the figure. Defaults to None.
        figsize (Tuple, optional): Size of the figure. Defaults to (8, 6).
        save_only (bool, optional): If True, only save the figure without displaying it. Defaults to False.
        dpi_save (int, optional): Dots per inch for saving the figure. Defaults to 300.
        histplot_multiple (str, optional): How to plot multiple histograms. Options are "layer", "dodge", "stack", "fill". Defaults to "stack".
        histplot_element (str, optional): Plotting element. Options are "bars", "step", "poly". Defaults to "bars".

    Returns:
        None
    """

    # check type of obs_val
    adata_obs = adata.obs.copy()
    if isinstance(axis, tuple):
        print("Retrieve `obs_val` from .obsm.")
        obsm_key = axis[0]
        obsm_col = axis[1]
        axis = f"distance_from_{obsm_col}"
        adata_obs[axis] = adata.obsm[obsm_key][obsm_col]

    # get data for plotting
    data = adata_obs[[axis, groupby]].dropna()

    # remove zeros
    xlim_mask = (data[axis] > xlim[0]) & (data[axis] <= xlim[1])
    data = data[xlim_mask].copy()

    # Create the histogram
    fig, ax = plt.subplots(1,1, figsize=(figsize[0], figsize[1]))

    if not kde:
        h = sns.histplot(data=data, x=axis,
                    hue=groupby, palette=DEFAULT_CATEGORICAL_CMAP.colors,
                    multiple=multiple, element=histplot_element,
                    alpha=1, ax=ax
                    )
    else:
        h = sns.kdeplot(data=data, x=axis,
                    hue=groupby, palette=DEFAULT_CATEGORICAL_CMAP.colors,
                    alpha=1, ax=ax, multiple=multiple
                    )
        plt.xlim(0, data[axis].max())

    # Move the legend outside of the plot
    sns.move_legend(h, "upper left", bbox_to_anchor=(1, 1))

    # save or show figure
    save_and_show_figure(savepath=savepath,
                         fig=h.get_figure(),
                         save_only=save_only,
                         dpi_save=dpi_save,
                         tight=True
                         )

# def cell_expression_along_axis(
#     adata,
#     obs_val,
#     genes,
#     cell_type_column,
#     cell_type,
#     xlim: Tuple[Number, Number] = (0, np.inf),
#     min_expression: Number = 0,
#     xlabel: Optional[str] = None,
#     fit_reg: bool = False,
#     lowess: bool = False,
#     robust: bool = False
#     ):
#     genes = convert_to_list(genes)

#     # if len(genes) == 1:
#     #     _single_cell_expression_along_axis(
#     #         adata=adata, obs_val=obs_val, gene=genes[0],
#     #         cell_type_column=cell_type_column, cell_type=cell_type,
#     #         xlim=xlim, min_expression=min_expression, xlabel=xlabel,
#     #         fit_reg=fit_reg, lowess=lowess, robust=robust
#     #     )
#     # elif len(genes) > 1:
#     _multi_cell_expression_along_axis(
#         adata=adata, obs_val=obs_val, genes=genes,
#         cell_type_column=cell_type_column, cell_type=cell_type,
#         xlim=xlim, min_expression=min_expression, xlabel=xlabel,
#         fit_reg=fit_reg, lowess=lowess, robust=robust
#     )
#     # else:
#     #     raise ValueError("`genes` must have length > 0.")

# def _single_cell_expression_along_axis(
#     adata,
#     obs_val,
#     gene,
#     cell_type_column,
#     cell_type,
#     xlim: Tuple[Number, Number] = (0, np.inf),
#     min_expression: Number = 0,
#     xlabel: Optional[str] = None,
#     fit_reg: bool = False,
#     lowess: bool = False,
#     robust: bool = False
#     ):

#     data_of_one_celltype = _select_data(
#         adata=adata,
#         obs_val=obs_val,
#         cell_type_column=cell_type_column,
#         cell_type=cell_type,
#         genes=gene,
#         xlim=xlim,
#     )

#     # Filter for minimum gene expression
#     data_of_one_celltype = data_of_one_celltype[data_of_one_celltype[gene] >= min_expression]

#     # Plot
#     g = sns.jointplot(data=data_of_one_celltype,
#                     x="axis", y=gene,
#                     height=4,
#                     color="firebrick", kind="kde", levels=8,
#                     marginal_kws={"fill": True},
#                     )
#     #g.plot_joint(sns.scatterplot, color="k", s=12)
#     g.plot_joint(sns.regplot, color="k",
#                  #lowess=True,
#                  fit_reg=fit_reg,
#                  lowess=lowess,
#                  robust=robust,
#                  scatter_kws={"s": 1},
#                  line_kws={"color": "orange"}
#                  )
#     g.ax_joint.set_ylabel(f"{gene} in '{cell_type}'")

#     # Set common x-label
#     if xlabel is None:
#         g.ax_joint.set_xlabel("_".join(convert_to_list(obs_val)))
#     else:
#         g.ax_joint.set_xlabel(xlabel)

#     # g = sns.jointplot(data=data_of_one_celltype,
#     #             x=axis_label, y=gene,
#     #             color="k", kind="reg", #levels=8,
#     #             marginal_kws={"fill": True},
#     #             )
#     # g.plot_marginals(sns.kdeplot, color="firebrick", #s=12
#     #                  )
#     plt.show()
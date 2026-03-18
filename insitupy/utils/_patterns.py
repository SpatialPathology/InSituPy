import logging
from numbers import Number
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import fdrcorrection
from tqdm.autonotebook import tqdm

from insitupy._constants import _init_mpl_fontsize
from insitupy.plotting.expression_along_axis import _bin_data, _select_data
from insitupy.utils._regression import smooth_fit

logger = logging.getLogger(__name__)
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)


# Functions
def total_variation(values):
    """Compute the total variation (sum of absolute first differences) of *values*."""
    return np.sum(np.abs(np.diff(values)))

def random_permutation_tv(expr):
    """Return the total variation of *expr* after a random permutation of its elements."""
    random_order = np.random.permutation(np.arange(len(expr)))
    expr_random = expr[random_order]

    return total_variation(expr_random)

def filter_outliers(data, threshold=1.5):
    """
    Remove values that lie significantly outside the IQR.

    Args:
        data (numpy.ndarray): The input array.
        threshold (float, optional): The multiplier for the IQR to define outliers. Default is 1.5.

    Returns:
        numpy.ndarray: The filtered array with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Filter the array to remove outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class EvaluateExpressionObject:
    """Container for spatial expression evaluation results, including raw data, binned data, and regression output."""

    def __init__(self, raw_data: Optional[pd.DataFrame] = None, binned_data: Optional[pd.DataFrame] = None, result: Optional[pd.DataFrame] = None):
        """
        EvaluateExpressionObject holds the results of the expression evaluation.

        Args:
            raw_data (Optional[pd.DataFrame]): The raw data used for evaluation.
            binned_data (Optional[pd.DataFrame]): The binned data used for evaluation.
            result (Optional[pd.DataFrame]): The result of the evaluation.
        """
        self.raw_data = raw_data
        self.binned_data = binned_data
        self.result = result

    def __repr__(self) -> str:
        """
        Return a string representation of the EvaluateExpressionObject.

        Returns:
            str: String representation of the object.

        """

        attributes = []
        if self.raw_data is not None:
            attributes.append("raw_data")
        if self.binned_data is not None:
            attributes.append("binned_data")
        if self.result is not None:
            attributes.append("result")

        return f"{self.__class__.__name__} with following attributes: {', '.join(attributes)}"

def evaluate_expression_along_axis(
    adata: pd.DataFrame,
    obs_val: str,
    genes: Union[str, List[str]],
    cell_type_column: str,
    cell_type: str,
    xlim: List[float],
    parallel: bool,
    bin_data: bool = False,
    resolution: Union[int, float] = 5,
    n_sim: int = 10000,
    min_expression: Optional[Union[int, float]] = None,
    n_jobs: int = 8,
    # plot_qc: bool = False
) -> EvaluateExpressionObject:
    """
    Evaluate gene expression along a specified axis.

    Args:
        adata (pd.DataFrame): Annotated data matrix.
        obs_val (str): Observation value to evaluate.
        genes (Union[str, List[str]]): Gene or list of genes to evaluate.
        cell_type_column (str): Column name for cell type.
        cell_type (str): Specific cell type to evaluate.
        xlim (List[float]): Limits for the x-axis.
        parallel (bool): Whether to run simulations in parallel.
        bin_data (bool, optional): Whether to bin the data. Defaults to False.
        resolution (Union[int, float], optional): Resolution for binning. Defaults to 5.
        n_sim (int, optional): Number of simulations. Defaults to 10000.
        min_expression (Optional[Union[int, float]], optional): Minimum expression threshold. Defaults to None.
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to 8.
        plot_qc (bool, optional): Whether to plot quality control. Defaults to False.

    Returns:
        EvaluateExpressionObject: Object containing raw data, binned data, and results.
    """
    genes = convert_to_list(genes)

    # get data
    data = _select_data(
        adata=adata,
        obs_val=obs_val,
        genes=genes,
        cell_type_column=cell_type_column, cell_type=cell_type,
        xlim=xlim,
        sort=True, minmax_scale=True,
        min_expression=min_expression, verbose=False
    )

    raw_data = data.copy() if bin_data else None
    if bin_data:
        data = _bin_data(data=data, resolution=resolution, plot=False)

    res = {
        "tv": [],
        "tv_pval": [],
        "spearman_r": [],
        "spearman_pval": [],
        "pearson_r": [],
        "pearson_pval": []
    }
    for gene in tqdm(genes):
        # extract values from data
        scaled_expr = data[gene].values
        not_nan = ~np.isnan(scaled_expr)
        axis = data.index.get_level_values('axis')

        try:
            # calculate correlation coefficients
            spearman_r, spearman_p = spearmanr(axis[not_nan], scaled_expr[not_nan])
            pearson_r, pearson_p = pearsonr(axis[not_nan], scaled_expr[not_nan])
        except ValueError:
            spearman_r = spearman_p = pearson_r = pearson_p = np.nan

        # drop NaNs before calculation of total variation
        scaled_expr = scaled_expr[~np.isnan(scaled_expr)]

        tv_gene = total_variation(values=scaled_expr)
        # simulation of random total variations
        # speed up computation with joblib
        if parallel:
            random_tvs = np.array(Parallel(n_jobs=n_jobs)(delayed(random_permutation_tv)(scaled_expr)
                                                    for _ in range(n_sim)))
        else:
            random_tvs = np.array([
                random_permutation_tv(
                    scaled_expr
                    )
                for _ in range(n_sim)
                ])
        random_tvs_filtered = filter_outliers(random_tvs)
        n = len(random_tvs_filtered)

        # collect results
        res["tv"].append(tv_gene)

        if n > 0:
            tv_pval = np.sum(random_tvs_filtered <= tv_gene) / n
        else:
            tv_pval = np.nan



        res["tv_pval"].append(tv_pval)

        res["spearman_r"].append(spearman_r)
        res["spearman_pval"].append(spearman_p)
        res["pearson_r"].append(pearson_r)
        res["pearson_pval"].append(pearson_p)

    # multiple testing correction
    fdrcorr = fdrcorrection(res["tv_pval"], is_sorted=False)
    res["tv_fdr"] = fdrcorr[1]

    # create pandas dataframe from results
    result_df = pd.DataFrame(res)
    result_df.index = genes

    # sort columns before returning results
    result_df = result_df.loc[:, ['tv', 'tv_pval', 'tv_fdr', 'spearman_r',
                      'spearman_pval', 'pearson_r', 'pearson_pval']]

    # if plot_qc:
    #     if bin_data:
    #         plot_evaluation(
    #             raw_data=raw_data, binned_data=data
    #         )
    #     else:
    #         plot_evaluation(
    #             raw_data=data, binned_data=None
    #         )

    result = EvaluateExpressionObject(
        raw_data=raw_data if bin_data else data,
        binned_data=data if bin_data else None,
        result=result_df)

    return result

def plot_evaluation(
    eval_object: EvaluateExpressionObject,
    genes: Optional[List[str]] = None,
    # raw_data: pd.DataFrame,
    # binned_data: Optional[pd.DataFrame],
    xlabel='x',
    maxcols=4,
    font_scale_factor: Optional[Number] = None
):
    """Plot gene expression along a spatial axis with optional LOESS regression overlays.

    For each gene in *genes*, draws raw or binned expression values and
    overlays a LOESS regression curve with a shaded confidence band.

    Args:
        eval_object: :class:`EvaluateExpressionObject` holding raw and/or
            binned expression data and pre-computed regression results.
        genes: List of gene names to plot.  Defaults to all columns in the
            raw data.
        xlabel: Label for the spatial x-axis.
        maxcols: Maximum number of subplot columns.
        font_scale_factor: Scaling factor applied to global matplotlib font
            sizes.  ``None`` leaves fonts unchanged.
    """
    # extract data from object
    binned_data = eval_object.binned_data
    raw_data = eval_object.raw_data

    if binned_data is not None:
        # extract values from binned data
        binned_axis = binned_data.index.values
        plot_binned = True
    else:
        plot_binned = False

    if genes is None:
        genes = raw_data.columns
    n_genes = len(genes)

    # extract values from raw data
    raw_axis = raw_data.index.get_level_values("axis").values

    # initialize figure
    if font_scale_factor is not None:
        _init_mpl_fontsize(scale_factor=font_scale_factor)
    nplots, nrows, ncols = get_nrows_maxcols(n_genes + 1, max_cols=maxcols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows),
                            sharex='all'#, sharey='row'
                            )

    if nplots > 1:
        axs = axs.ravel()

    # if n_genes == 1:
    #     # reshape the axis array so that 2d indexing works later
    #     axs = axs.reshape(1,2)

    for i, gene in enumerate(genes):
        raw_expr = raw_data[gene].values

        # prepare regression
        if plot_binned:
            binned_expr = binned_data[gene].values
            not_nan = ~np.isnan(binned_expr)
            xs = binned_axis[not_nan]
            ys = binned_expr[not_nan]
            reg_label = "Loess Regression of Binned Data"
        else:
            not_nan = ~np.isnan(raw_expr)
            xs = raw_axis[not_nan]
            ys = raw_expr[not_nan]
            reg_label = "Loess Regression of Raw Data"

        if len(xs) > 1:
            try:
                # perform loess regression for the second half of the plot
                res = smooth_fit(
                xs=xs,
                ys=ys, # make sure there are no NaN in the data
                loess_bootstrap=False, nsteps=100
                )
            except ValueError as e:
                logger.warning(f"A ValueError occurred during loess regression: {e}")
                res = None
        else:
            logger.warning(f"Only one datapoint left for gene {gene} after filtering. Skipped LOESS regression.")
            res = None

        # Plot the original data
        axs[i].scatter(
            raw_axis, raw_expr,
            label='Raw Data', alpha=0.5, color='k', s=2
            )

        if plot_binned:
            # Plot the binned values
            axs[i].scatter(
                binned_axis, binned_expr,
                color='darkorange',
                s=32, alpha=1,
                linestyle='-', label='Binned Mean')


        # axs[i, 1].scatter(
        #     x=binned_axis, y=binned_expr,s=1, color="k", label="Binned Mean")
        if res is not None:
            axs[i].plot(res["x"], res["y_pred"],
                        color='royalblue', linewidth=3,
                        label=reg_label)
            axs[i].fill_between(res["x"],
                                   res["conf_lower"],
                                   res["conf_upper"],
                                   color='royalblue',
                                   alpha=0.2, label='95% CI of Loess Regression')

        # Add labels and legend
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(f"Gene expression'")
        axs[i].set_title(f"{gene}")


        # axs[i, 1].legend()
        # axs[i, 1].set_xlabel(xlabel)
        # axs[i, 1].set_ylabel(f"Scaled expression of '{gene}'")

    handles, labels = axs[0].get_legend_handles_labels()
    axs[n_genes].legend(handles, labels, loc='upper center')

    remove_empty_subplots(axs, n_genes, nrows, ncols)

    #axs[nplots].legend()

    # Show plot
    plt.tight_layout()
    plt.show()

    # reset matplotlib settings
    _init_mpl_fontsize(scale_factor=1)

def loess_regress(
    eval_object: EvaluateExpressionObject,
    genes: Optional[List[str]] = None,
):
    """Run LOESS regression for each gene in an :class:`EvaluateExpressionObject`.

    Fits a LOESS curve to binned data (if available) or raw data for each
    gene and returns the regression results keyed by gene name.

    Args:
        eval_object: :class:`EvaluateExpressionObject` with raw and/or binned
            expression data.
        genes: Gene names to regress.  Defaults to all columns in raw data.

    Returns:
        A dict mapping gene name to a :class:`~insitupy.utils._regression.lowess_prediction`
        result object.
    """
    # extract data from object
    binned_data = eval_object.binned_data
    raw_data = eval_object.raw_data

    if binned_data is not None:
        # extract values from binned data
        binned_axis = binned_data.index.values
        regress_on_binned = True
    else:
        regress_on_binned = False

    if genes is None:
        genes = raw_data.columns

    # extract values from raw data
    raw_axis = raw_data.index.get_level_values("axis").values

    result = {}
    for gene in tqdm(genes):
        raw_expr = raw_data[gene].values

        # prepare regression
        if regress_on_binned:
            binned_expr = binned_data[gene].values
            not_nan = ~np.isnan(binned_expr)
            xs = binned_axis[not_nan]
            ys = binned_expr[not_nan]
        else:
            not_nan = ~np.isnan(raw_expr)
            xs = raw_axis[not_nan]
            ys = raw_expr[not_nan]

        if len(xs) > 1:
            try:
                # perform loess regression for the second half of the plot
                res = smooth_fit(
                xs=xs,
                ys=ys, # make sure there are no NaN in the data
                loess_bootstrap=False, nsteps=100
                )
            except ValueError as e:
                logger.warning(f"A ValueError occurred during loess regression: {e}")
                res = None
        else:
            logger.warning(f"Only one datapoint left for gene {gene} after filtering. Skipped LOESS regression.")
            res = None

        # collect results
        result[gene] = res

    return pd.concat(result)



# def evaluate_expression_along_axis(
#     adata,
#     obs_val,
#     genes,
#     cell_type_column,
#     cell_type,
#     xlim,
#     parallel,
#     bin_data: bool = False,
#     resolution: Number = 5,
#     n_sim: int = 10000,
#     min_expression: Optional[Number] = None,
#     n_jobs: int = 8,
#     plot_qc: bool = False
#     ):

#     genes = convert_to_list(genes)

#     # get data
#     data = _select_data(
#         adata=adata,
#         obs_val=obs_val,
#         genes=genes,
#         cell_type_column=cell_type_column, cell_type=cell_type,
#         xlim=xlim,
#         sort=True, minmax_scale=True,
#         min_expression=min_expression, verbose=False
#         )

#     if bin_data:
#         raw_data = data.copy()
#         data = _bin_data(data=data, resolution=resolution, plot=False)

#     #pvals = []
#     res = {
#         "tv": [],
#         "tv_pval": [],
#         "spearman_r": [],
#         "spearman_pval": [],
#         "pearson_r": [],
#         "pearson_pval": []
#     }
#     for gene in tqdm(genes):
#         # extract values from data
#         scaled_expr = data[gene].values
#         not_nan = ~np.isnan(scaled_expr)
#         axis = data.index.get_level_values('axis')

#         try:
#             # calculate correlation coefficients
#             spearman_r, spearman_p = spearmanr(axis[not_nan], scaled_expr[not_nan])
#             pearson_r, pearson_p = pearsonr(axis[not_nan], scaled_expr[not_nan])
#         except ValueError:
#             spearman_r = spearman_p = pearson_r = pearson_p = np.nan

#         # drop NaNs before calculateion of total variation
#         scaled_expr = scaled_expr[~np.isnan(scaled_expr)]

#         tv_gene = total_variation(values=scaled_expr)
#         # simulation of random total variations
#         # speed up computation with joblib
#         if parallel:
#             #random_tvs = np.array(Parallel(n_jobs=8)(delayed(random_permutation_tv)(expr_sorted) for _ in range(n_sim)))
#             random_tvs = np.array(Parallel(n_jobs=n_jobs)(delayed(random_permutation_tv)(scaled_expr)
#                                                     for _ in range(n_sim)))
#         else:
#             random_tvs = np.array([
#                 random_permutation_tv(
#                     scaled_expr
#                     )
#                 for _ in range(n_sim)
#                 ])
#         random_tvs_filtered = filter_outliers(random_tvs)
#         n = len(random_tvs_filtered)

#         # collect results
#         res["tv"].append(tv_gene)

#         if n > 0:
#             res["tv_pval"].append(np.sum(random_tvs_filtered <= tv_gene) / n)
#         else:
#             res["tv_pval"].append(np.nan)

#         res["spearman_r"].append(spearman_r)
#         res["spearman_pval"].append(spearman_p)
#         res["pearson_r"].append(pearson_r)
#         res["pearson_pval"].append(pearson_p)

#     res = pd.DataFrame(res)
#     res.index = genes
#     #res = pd.Series(pvals, index=genes)

#     if plot_qc:
#         if bin_data:
#             _qc_plot(
#                 raw_data=raw_data, binned_data=data
#             )
#         else:
#             _qc_plot(
#                 raw_data=data, binned_data=None
#             )

#     return res
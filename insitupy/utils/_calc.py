import warnings
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from tqdm import tqdm


def _calc_kernel_density(
    data: Union[np.ndarray, List],
    mode: Literal["gauss", "mellon"] = "gauss",
    verbose: bool = False
    ):
    """
    Calculate the kernel density estimation for the given data.

    Args:
        data (Union[np.ndarray, List]): Input data for density estimation.
        mode (Literal["gauss", "mellon"], optional): The mode of density estimation.
            "gauss" for Gaussian KDE using scipy, "mellon" for Mellon density estimator.
            Defaults to "gauss".
        verbose (bool, optional): If True, print statements will be used to indicate the mode.
            Defaults to False.

    Returns:
        np.ndarray: The estimated density values.

    Raises:
        UserWarning: If an invalid mode is provided.
    """
    # Make sure the data is a numpy array
    data = np.array(data)

    if mode == "mellon":
        try:
            import mellon
        except:
            raise ImportError("To calculate densities with the mellon package, please install it with `pip install mellon`.")
        if verbose:
            print("Using Mellon density estimator.")
        # Fit and predict log density
        model = mellon.DensityEstimator()
        density = model.fit_predict(data)
    elif mode == "gauss":
        if verbose:
            print("Using Gaussian KDE.")
        try:
            kde = gaussian_kde(data.T, bw_method="scott")
            density = kde(data.T)
        except LinAlgError:
            # return only NaN values - this happens if the data is not big enough
            density = np.empty(len(data))
            density[:] = np.nan

    else:
        warnings.warn(f"Invalid mode '{mode}' provided. Please use 'gauss' or 'mellon'.")
        return None

    return density

def calc_density(
    adata,
    groupby: str,
    mode: Literal["gauss", "mellon"] = "gauss",
    clip: bool = True,
    inplace: bool = False
):
    """
    Calculate the spatial density for groups in the AnnData object. Groups could be e.g. cell types in the sample.
    Spatial coordinates are expected to be saved in `adata.obsm["spatial"]`.

    Args:
        adata (AnnData): The annotated data matrix.
        groupby (str): The column in `adata.obs` to group by.
        mode (Literal["gauss", "mellon"], optional): The mode of density estimation.
            "gauss" for Gaussian KDE using scipy, "mellon" for Mellon density estimator.
            Defaults to "gauss".
        clip (bool, optional): If True, clip the density values to the 5th and 95th percentile.
        inplace (bool, optional): If True, modify `adata` in place. If False, return a copy of `adata` with the modifications.
            Defaults to False.

    Returns:
        AnnData: The modified AnnData object with added density values.
    """
    if inplace:
        _adata = adata
    else:
        _adata = adata.copy()

    # Initialize lists to store results
    density_df = pd.DataFrame(index=_adata.obs_names)

    # Iterate over unique values in the groupby column
    for group in tqdm(_adata.obs[groupby].unique()):
        # Select the respective values in adata.obsm["spatial"]
        group_mask = _adata.obs[groupby] == group
        spatial_data = _adata.obsm["spatial"][group_mask]

        # Fit and predict density
        density = _calc_kernel_density(spatial_data, mode=mode)

        # create pandas series from results
        density_series = pd.Series(
            data=density,
            index=_adata.obs_names[group_mask],
            name=group
            )

        # Store results in dataframes
        density_df[group] = density_series

    if clip:
        # clip the data
        quantiles_df = density_df.quantile([0.05, 1])
        density_df_clipped = density_df.clip(
            lower=quantiles_df.iloc[0],
            upper=quantiles_df.iloc[1],
            axis=1
            )

        _adata.obsm[f"density-{mode}"] = density_df_clipped

    else:
        _adata.obsm[f"density-{mode}"] = density_df

    if not inplace:
        return _adata

def cohens_d(a, b, paired=False, correct_small_sample_size=True):
    '''
    Function to calculate the Cohen's D measure of effect sizes.

    Function with correction was adapted from:
    https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/cohens-d/

    To allow measurement for different sample sizes following sources were used:
    https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
    https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

    For paired samples following websites were used:
    https://www.datanovia.com/en/lessons/t-test-effect-size-using-cohens-d-measure/
    The results were tested here: https://statistikguru.de/rechner/cohens-d-gepaarter-t-test.html
    '''
    if not paired:
        # calculate parameters
        mean1 = np.mean(a)
        mean2 = np.mean(b)
        std1 = np.std(a, ddof=1)
        std2 = np.std(b, ddof=1)
        n1 = len(a)
        n2 = len(b)
        dof = n1 + n2 - 2 # degrees of freedom
        SDpool = np.sqrt(((n1-1) * std1**2 + (n2 - 1) * std2**2) / dof) # pooled standard deviations

        if SDpool == 0:
            d = np.nan
        else:
            d = (mean1 - mean2) / SDpool

        n = np.min([n1, n2])
        if correct_small_sample_size and n < 50:
            # correct for small sample size
            corr_factor = ((n - 3) / (n-2.25)) * np.sqrt((n - 2) / n)
            d *= corr_factor

    else:
        assert len(a) == len(b), "For paired testing the size of both samples needs to be equal."
        diff = np.array(a) - np.array(b)
        d = np.mean(diff) / np.std(diff)

    return d
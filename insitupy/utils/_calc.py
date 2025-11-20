import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from skimage.measure import regionprops_table
from tqdm import tqdm
from tqdm.auto import tqdm


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




def intensity_median(region_mask, intensity_image):
    """Calculate median intensity for a region."""
    return np.median(intensity_image[region_mask])


def quantify_fluorescence(
    image_dask: da.Array,
    mask_dask: da.Array,
    method: Union[Literal["mean", "median"], str, Callable] = "median",
    downsample_factor: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient quantification for greyscale images.

    Uses lazy loading with dask to minimize memory usage. The image is only
    loaded into memory once after downsampling (if specified).

    Parameters
    ----------
    image_dask : dask array
        Greyscale fluorescence image, shape (Y, X)
    mask_dask : dask array
        Segmentation mask with cell IDs, shape (Y, X)
    method : {"mean", "median"} or str or callable, optional
        Quantification method. Built-in options: "mean", "median" (default).
        Other strings are passed to regionprops_table (e.g., "intensity_max", "intensity_min").
        For custom functions, provide a callable with signature
        func(region_mask, intensity_image) -> float.
    downsample_factor : int, optional
        Factor by which to downsample image and mask before quantification.
        For example, downsample_factor=2 will reduce dimensions by half.
        Uses mean for image downsampling and nearest neighbor for mask.
        This reduces memory usage proportionally to the square of the factor
        (e.g., factor=2 uses ~4x less RAM, factor=4 uses ~16x less RAM).
        Default is None (no downsampling).

    Returns
    -------
    measurements : np.ndarray
        Measurements array, shape (n_cells,)
    cell_idx : np.ndarray
        Array of cell IDs

    Notes
    -----
    Memory usage: Loads full mask and full image once after downsampling.
    For multi-channel images, process each channel separately by calling this
    function multiple times with image_dask[channel_idx].

    Examples
    --------
    >>> # Default usage with median intensity
    >>> measurements, cell_ids = quantify_fluorescence(image_zarr, mask_zarr)
    >>>
    >>> # Use mean instead of median
    >>> measurements, cell_ids = quantify_fluorescence(
    ...     image_zarr, mask_zarr, method="mean"
    ... )
    >>>
    >>> # With 4x downsampling for very large images
    >>> measurements, cell_ids = quantify_fluorescence(
    ...     image_zarr, mask_zarr, downsample_factor=4
    ... )
    >>>
    >>> # Custom function example
    >>> def intensity_p90(region_mask, intensity_image):
    ...     return np.percentile(intensity_image[region_mask], 90)
    >>>
    >>> measurements, cell_ids = quantify_fluorescence(
    ...     image_zarr, mask_zarr, method=intensity_p90
    ... )
    >>>
    >>> # For multi-channel images, process each channel separately
    >>> for c in range(n_channels):
    ...     measurements, cell_ids = quantify_fluorescence(
    ...         image_zarr[c], mask_zarr
    ...     )
    """
    if method == "median":
        method = intensity_median
    elif method == "mean":
        method = "intensity_mean"
    # Check image dimensions
    if image_dask.ndim != 2:
        raise ValueError(
            f"Image must be 2D greyscale with shape (Y, X), got shape {image_dask.shape} "
            f"with {image_dask.ndim} dimensions. For multi-channel images, select a single "
            f"channel first: image_dask[channel_idx]"
        )

    # Check mask dimensions
    if mask_dask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D with shape (Y, X), got shape {mask_dask.shape} "
            f"with {mask_dask.ndim} dimensions"
        )

    # Apply downsampling if requested (lazy operations on dask arrays)
    if downsample_factor is not None and downsample_factor > 1:
        # Downsample mask using nearest neighbor (to preserve cell IDs)
        mask_dask = mask_dask[::downsample_factor, ::downsample_factor]

        # Downsample image using coarsen with mean
        image_dask = da.coarsen(
            np.mean,
            image_dask,
            {0: downsample_factor, 1: downsample_factor},
            trim_excess=True
        )

    # Load mask and image
    mask = mask_dask.compute()
    image = image_dask.compute()

    # Ensure mask and image have matching shapes
    min_y = min(mask.shape[0], image.shape[0])
    min_x = min(mask.shape[1], image.shape[1])
    mask_cropped = mask[:min_y, :min_x]
    image_cropped = image[:min_y, :min_x]

    image_3d = image_cropped[:, :, np.newaxis]

    # Compute regionprops
    if isinstance(method, str):
        props = regionprops_table(
            mask_cropped,
            intensity_image=image_3d,
            properties=["label", method]
        )
    elif callable(method):
        props = regionprops_table(
            mask_cropped,
            intensity_image=image_3d,
            extra_properties=(method,)
        )
    else:
        raise ValueError("method must be 'mean', 'median', or a callable")

    # Extract cell IDs
    cell_idx = props.pop("label")

    # Get measurement
    func_name = method if isinstance(method, str) else method.__name__
    measurement_key = [k for k in props.keys() if k.startswith(func_name)][0]
    measurements = props[measurement_key]

    return np.array(measurements), np.array(cell_idx)
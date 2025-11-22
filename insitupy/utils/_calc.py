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
    downsample_factor: Optional[int] = None,
    return_area: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    return_area : bool, optional
        If True, also return the area (number of pixels) for each cell.
        Default is False.

    Returns
    -------
    measurements : np.ndarray
        Measurements array, shape (n_cells,)
    cell_idx : np.ndarray
        Array of cell IDs
    areas : np.ndarray, optional
        Array of cell areas in pixels, shape (n_cells,).
        Only returned if return_area=True.

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
    >>> # With area information
    >>> measurements, cell_ids, areas = quantify_fluorescence(
    ...     image_zarr, mask_zarr, return_area=True
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
    properties = ["label"]
    if return_area:
        properties.append("area")

    if isinstance(method, str):
        properties.append(method)
        props = regionprops_table(
            mask_cropped,
            intensity_image=image_3d,
            properties=properties
        )
    elif callable(method):
        props = regionprops_table(
            mask_cropped,
            intensity_image=image_3d,
            properties=properties,
            extra_properties=(method,)
        )
    else:
        raise ValueError("method must be 'mean', 'median', or a callable")

    # Extract cell IDs
    cell_idx = props.pop("label")

    # Extract areas if requested
    if return_area:
        areas = props.pop("area")

    # Get measurement
    func_name = method if isinstance(method, str) else method.__name__
    measurement_key = [k for k in props.keys() if k.startswith(func_name)][0]
    measurements = props[measurement_key]

    if return_area:
        return np.array(measurements), np.array(cell_idx), np.array(areas)
    else:
        return np.array(measurements), np.array(cell_idx)




def create_tiles(
    dask_array: da.Array,
    tile_size: int = 2000,
    overlap: int = 100
) -> List[Tuple[da.Array, Tuple[slice, slice], Tuple[slice, slice]]]:
    """
    Split a 2D dask array into overlapping tiles.

    Parameters
    ----------
    dask_array : dask array
        2D array to split, shape (Y, X)
    tile_size : int, optional
        Maximum size of each tile dimension in pixels (default: 2000)
    overlap : int, optional
        Overlap between adjacent tiles in pixels (default: 100)

    Returns
    -------
    tiles : list of tuples
        Each tuple contains:
        - tile: dask array slice
        - global_slice: (slice_y, slice_x) position in original array
        - inner_slice: (slice_y, slice_x) position excluding overlap for stitching

    Examples
    --------
    >>> tiles = create_tiles(image_dask, tile_size=2000, overlap=100)
    >>> for tile, global_pos, inner_pos in tiles:
    ...     # Process tile
    ...     result = process(tile.compute())
    """
    if dask_array.ndim != 2:
        raise ValueError(
            f"Array must be 2D with shape (Y, X), got shape {dask_array.shape}"
        )

    height, width = dask_array.shape
    step = tile_size - overlap

    tiles = []

    for y_start in range(0, height, step):
        for x_start in range(0, width, step):
            # Calculate tile boundaries with overlap
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Create slices for extracting tile
            global_slice = (slice(y_start, y_end), slice(x_start, x_end))

            # Calculate inner region (excluding overlap) for stitching
            inner_y_start = overlap if y_start > 0 else 0
            inner_x_start = overlap if x_start > 0 else 0
            inner_y_end = y_end - y_start
            inner_x_end = x_end - x_start

            # Adjust inner boundaries for last tiles
            if y_end < height:
                inner_y_end -= overlap
            if x_end < width:
                inner_x_end -= overlap

            inner_slice = (
                slice(inner_y_start, inner_y_end),
                slice(inner_x_start, inner_x_end)
            )

            # Extract tile (lazy operation)
            tile = dask_array[global_slice]

            tiles.append((tile, global_slice, inner_slice))

    return tiles

from typing import List, Tuple

import numpy as np


def summarize_tile_measurements(
    quant_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consolidate measurements from overlapping tiles.

    For each cell ID, selects the measurement from the tile where that cell
    has maximum area. This ensures measurements are taken from tiles where
    cells are complete rather than split at tile boundaries.

    Parameters
    ----------
    quant_results : list of tuples
        List of (measurements, cell_ids, areas) tuples from quantify_fluorescence
        with return_area=True

    Returns
    -------
    measurements : np.ndarray
        Consolidated measurements, one per unique cell
    cell_ids : np.ndarray
        Corresponding cell IDs

    Examples
    --------
    >>> quant_results = []
    >>> for img_tile, mask_tile in zip(img_tiles, mask_tiles):
    ...     result = quantify_fluorescence(
    ...         img_tile[0], mask_tile[0], return_area=True
    ...     )
    ...     quant_results.append(result)
    >>> measurements, cell_ids = summarize_tile_measurements(quant_results)
    """
    # Collect all data
    all_measurements = []
    all_cell_ids = []
    all_areas = []

    for measurements, cell_ids, areas in quant_results:
        all_measurements.append(measurements)
        all_cell_ids.append(cell_ids)
        all_areas.append(areas)

    # Concatenate
    all_measurements = np.concatenate(all_measurements)
    all_cell_ids = np.concatenate(all_cell_ids)
    all_areas = np.concatenate(all_areas)

    # Find unique cell IDs
    unique_cell_ids = np.unique(all_cell_ids)

    # For each cell, find the measurement with maximum area
    final_measurements = np.zeros(len(unique_cell_ids))

    for i, cell_id in enumerate(unique_cell_ids):
        # Find all occurrences of this cell
        mask = all_cell_ids == cell_id
        cell_measurements = all_measurements[mask]
        cell_areas = all_areas[mask]

        # Select measurement from tile with maximum area
        max_area_idx = np.argmax(cell_areas)
        final_measurements[i] = cell_measurements[max_area_idx]

    return final_measurements, unique_cell_ids
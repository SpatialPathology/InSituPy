from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pandas.api.types import is_numeric_dtype

from insitupy._constants import DEFAULT_CATEGORICAL_CMAP
from insitupy.palettes import CustomPalettes


def create_cmap_mapping(data, cmap: Union[str, ListedColormap] = None):

    if cmap is None:
        pal = CustomPalettes()
        cmap = pal.tab20_mod

    try:
        unique_categories = data.cat.categories # in case of categorical pandas series
    except AttributeError:
        try:
            unique_categories = data.categories # in case of numpy categories
        except AttributeError:
            try:
                unique_categories = np.sort(data[~data.isna()].unique())
            except AttributeError:
                try:
                    unique_categories = np.sort(np.unique(data[~np.isnan(data)]))
                except TypeError:
                    unique_categories = np.sort(np.unique(data))

    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    len_colormap = cmap.N
    category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}
    return category_to_rgba


def categorical_data_to_rgba(data,
                             cmap: Union[str, ListedColormap],
                             return_mapping: bool = False,
                             nan_val: tuple = (1,1,1,0)
                             ):

    # len_colormap = cmap.N
    # category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}

    if not isinstance(cmap, dict):
        category_to_rgba = create_cmap_mapping(data, cmap)
    else:
        category_to_rgba = cmap

    if nan_val is not None:
        # add key for nan
        category_to_rgba[np.nan] = nan_val

    res = np.array([category_to_rgba[category] for category in data])

    if return_mapping:
        return res, category_to_rgba
    else:
        return res


def _determine_climits(
    color_values,
    upper_climit_pct,
    lower_climit = None
    ) -> list:

    if lower_climit is None:
        lower_climit = color_values.min()

    color_values_above_zero = color_values[color_values > 0]
    try:
        upper_climit = np.percentile(color_values_above_zero, upper_climit_pct)
    except IndexError:
        # if there were not color_values_above_zero, a IndexError appears
        upper_climit = 0

    climits = [lower_climit, upper_climit]

    return climits


def continuous_data_to_rgba(
    data,
    cmap: Union[str, ListedColormap],
    upper_climit_pct: int = 99,
    lower_climit: Optional[int] = 0,
    clip = False
    ):
    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    climits = _determine_climits(color_values=data, upper_climit_pct=upper_climit_pct, lower_climit=lower_climit)

    norm = mpl.colors.Normalize(vmin=climits[0], vmax=climits[1], clip=clip)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return scalarMap.to_rgba(data)


def _data_to_rgba(
    data,
    continuous_cmap: Union[str, ListedColormap] = "viridis",
    categorical_cmap: Union[str, ListedColormap] = None,
    upper_climit_pct: int = 99,
    return_mapping: bool = False,
    nan_val: tuple = (1,1,1,0)
    ):
    if is_numeric_dtype(data):
        return continuous_data_to_rgba(data=data, cmap=continuous_cmap, upper_climit_pct=upper_climit_pct)
    else:
        if categorical_cmap is None:
            # pal = CustomPalettes()
            # categorical_cmap = pal.tab20_mod
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP
        return categorical_data_to_rgba(data=data, cmap=categorical_cmap,
                                        return_mapping=return_mapping,
                                        nan_val=nan_val)
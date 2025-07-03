import math
from typing import Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from insitupy._constants import (DEFAULT_CATEGORICAL_CMAP,
                                 DEFAULT_CONTINUOUS_CMAP)
from insitupy._core._checks import check_raw
from insitupy.palettes import CustomPalettes


def _extract_color_values(adata, key, raw, layer):
    ## Extract expression data
    # check if plotting raw data
    adata_X, adata_var, adata_var_names = check_raw(
        adata,
        use_raw=raw,
        layer=layer
        )

    # locate gene in matrix and extract values
    if key in adata_var_names:
        idx = adata_var.index.get_loc(key)
        color_values = adata_X[:, idx].copy()
        categorical = False

    elif key in adata.obs.columns:
        color_values = adata.obs[key]
        if is_numeric_dtype(adata.obs[key]):
            categorical = False
        else:
            categorical = True
    else:
        color_values = None
        categorical = None

    return color_values, categorical

def _create_handle_function(mode):
    if mode == "circle":
        def handle_function(color):
            return plt.Line2D([0], [0],
                               marker='o',
                               color='w',
                               markerfacecolor=color,
                               markersize=15,
                               markeredgecolor='black',
                               markeredgewidth=1.5
                               )
    elif mode == "rectangle":
        def handle_function(color):
            return plt.Line2D([0], [0],
                               marker='_',
                               color='w',
                               markerfacecolor=None,
                               markersize=15,
                               markeredgecolor=color,
                               markeredgewidth=8
                               )
    else:
        raise ValueError("Invalid mode. Choose 'circle' or 'rectangle'.")
    return handle_function

def _add_colorlegend_to_axis(
    color_dict: dict,
    ax: plt.Axes,
    max_per_col: int = 10,
    loc: str = 'center',
    bbox_to_anchor: tuple = (0.5, 0.5),
    title: Optional[str] = None,
    #marker: Optional[str] = 'o',
    mode: Literal["circle", "rectangle"] = "circle",
    remove_axis: bool = True
):
    # create function to create handles
    handle_function = _create_handle_function(mode)

    # Create legend manually
    handles = []
    labels = []

    for label, color in color_dict.items():
        handle = handle_function(color)
        handles.append(handle)
        labels.append(label)

    n_col = max(1, math.ceil(len(labels) / max_per_col))

    legend = ax.legend(
        handles, labels,
        loc=loc, ncol=n_col,
        frameon=True,
        title=title,
        bbox_to_anchor=bbox_to_anchor
        )

    if remove_axis:
        # Hide the axis
        ax.set_axis_off()

def _parse_unique_categories(data):
    # retrieve data
    try:
        unique_categories = data.cat.categories # in case of categorical pandas series
    except AttributeError:
        try:
            unique_categories = data.categories # in case of numpy categories
        except AttributeError:
            data = np.array(data)
            try:
                unique_categories = np.sort(data[~data.isna()].unique())
            except AttributeError:
                try:
                    unique_categories = np.sort(np.unique(data[~np.isnan(data)]))
                except TypeError:
                    #unique_categories = np.sort(np.unique(data))
                    # Convert all elements to strings before sorting
                    unique_categories = np.sort(np.unique(data.astype(str)))

    return unique_categories


def create_cmap_mapping(
    data,
    cmap: Optional[Union[str, ListedColormap]] = None,
    rgba_values: Optional[np.ndarray] = None
    ):
    unique_categories = _parse_unique_categories(data)

    if rgba_values is None:
        if cmap is None:
            pal = CustomPalettes()
            cmap = pal.tab20_mod

        # get colormap if necessary
        if not isinstance(cmap, ListedColormap):
            cmap = plt.get_cmap(cmap)

        len_colormap = cmap.N
        category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}
    else:
        category_to_rgba = {}
        for v in unique_categories:
            idx = np.argwhere(data == v)[0][0].item()
            category_to_rgba[v] = rgba_values[idx]

    return category_to_rgba


def categorical_data_to_rgba(data,
                             cmap: Union[str, ListedColormap],
                             return_mapping: bool = False,
                             nan_val: tuple = (1,1,1,0),
                             rgba_values: Optional[np.ndarray] = None
                             ):

    # len_colormap = cmap.N
    # category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}

    if not isinstance(cmap, dict):
        category_to_rgba = create_cmap_mapping(data, cmap, rgba_values)
    else:
        category_to_rgba = cmap

    if nan_val is not None:
        # add key for nan
        category_to_rgba[str(np.nan)] = nan_val

    res = np.array([category_to_rgba[str(category)] for category in data])

    if return_mapping:
        return res, category_to_rgba
    else:
        return res


def _determine_climits(
    color_values,
    upper_climit_pct,
    lower_climit = None,
    force_above_zero = False
    ) -> list:

    if lower_climit is None:
        lower_climit = color_values.min()

    if force_above_zero:
        color_values_for_calc = color_values[color_values > 0]
    else:
        color_values_for_calc = color_values
    try:
        upper_climit = np.percentile(color_values_for_calc, upper_climit_pct)
    except IndexError:
        # if there were no color values above zero, a IndexError appears
        upper_climit = 0

    climits = [lower_climit, upper_climit]

    return climits


def continuous_data_to_rgba(
    data,
    cmap: Union[str, ListedColormap],
    upper_climit_pct: int = 99,
    lower_climit: Optional[int] = None,
    clip = False,
    nan_val: tuple = (1,1,1,0),
    return_mapping: bool = False
    ):
    if np.any(pd.isna(data)):
        contains_nans = True
        # Convert the numpy array to a pandas Series
        value_series = pd.Series(data)

        # check where there are nas
        isna_mask = value_series.isna()
        notna_mask = ~isna_mask

        # get only values without NaNs
        notna_values = value_series[notna_mask].values
    else:
        notna_values = data
        contains_nans = False

    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    if lower_climit is None:
        lower_climit = np.min(notna_values)

    climits = _determine_climits(color_values=notna_values, upper_climit_pct=upper_climit_pct, lower_climit=lower_climit)

    norm = mpl.colors.Normalize(vmin=climits[0], vmax=climits[1], clip=clip)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_data = scalarMap.to_rgba(notna_values)

    if contains_nans:
        result_values = pd.Series(index=value_series.index, dtype="object")
        tuple_list = [tuple(elem) for elem in rgba_data]
        result_values.loc[notna_mask] = tuple_list
        result_values.loc[isna_mask] = [nan_val] * isna_mask.sum()
        result_rgba = np.array(result_values.tolist())
    else:
        result_rgba = rgba_data

    if return_mapping:
        return result_rgba, scalarMap
    else:
        return result_rgba


def _data_to_rgba(
    data: np.ndarray,
    continuous_cmap: Union[str, ListedColormap] = DEFAULT_CONTINUOUS_CMAP,
    categorical_cmap: Union[str, ListedColormap] = None,
    upper_climit_pct: int = 99,
    #return_all: bool = False,
    nan_val: tuple = (1,1,1,0),
    rgba_values: Optional[np.ndarray] = None
    ):
    if not is_numeric_dtype(data) or is_bool_dtype(data):
        if is_bool_dtype(data):
            # in case of boolean data it is important to convert it to object type
            data = data.astype('object')
            # also pandas NAs have to be substituted with numpy NaNs
            data = np.array([elem if pd.notna(elem) else np.nan for elem in data], dtype='object')
        if categorical_cmap is None:
            # pal = CustomPalettes()
            # categorical_cmap = pal.tab20_mod
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP
        rgba_list, mapping = categorical_data_to_rgba(data=data, cmap=categorical_cmap,
                                        return_mapping=True,
                                        nan_val=nan_val,
                                        rgba_values=rgba_values)
        cmap = categorical_cmap
    else:
        rgba_list, mapping = continuous_data_to_rgba(data=data, cmap=continuous_cmap,
                                       upper_climit_pct=upper_climit_pct,
                                       return_mapping=True)
        cmap = continuous_cmap

    return rgba_list, mapping, cmap

# def get_crange(adata, key, use_raw,
#     layer=None,
#     ctype='minmax', cmin_at_zero=True
#     ):
#     obs = adata.obs
#     adata_X, adata_var, adata_var_names = check_raw(adata, use_raw=use_raw, layer=layer)

#     if key in adata_var_names:
#         #c = adata_X[[elem in groups for elem in adata.obs[groupby]], adata_var_names == key]
#         c = adata_X[:, adata_var_names == key]
#         if cmin_at_zero:
#             cmin = 0
#         else:
#             cmin = c.min()

#         if ctype == 'percentile':
#             cmax = np.percentile(c, 95)
#         else:
#             cmax = c.max()
#         crange = [cmin, cmax]
#     elif key in obs.columns:
#         #if obs[key].dtype.name.startswith('float') or obs[key].dtype.name.startswith('int'):
#         if is_numeric_dtype(obs[key]):
#             #c = obs[key][[elem in groups for elem in obs[groupby]]]
#             c = obs[key]
#             if cmin_at_zero:
#                 cmin = 0
#             else:
#                 cmin = c.min()
#             cmax = np.percentile(c, 95)
#             crange = [cmin, cmax]
#         else:
#             return None
#     else:
#         print(f"Key {key} not in var_names of adata object. Use raw?")
#         return

#     return crange
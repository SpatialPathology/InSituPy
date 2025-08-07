import math
import os
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from napari.utils.notifications import show_info, show_warning
from napari.viewer import Viewer

from insitupy._constants import DEFAULT_CATEGORICAL_CMAP
from insitupy._core._checks import _check_assignment, _is_experiment
from insitupy._core.data import InSituData
from insitupy.dataclasses._utils import _get_cell_layer
from insitupy.experiment.data import InSituExperiment
from insitupy.io.plots import save_and_show_figure
from insitupy.palettes import map_to_colors
from insitupy.utils._colors import _add_colorlegend_to_axis, _data_to_rgba
from insitupy.utils.utils import (convert_to_list, get_nrows_maxcols,
                                  remove_empty_subplots)


def _generate_subplots(
    n_plots: int,
    n_keys: int,
    max_cols: int = 4,
    dpi_display: int = 80,
    header: Optional[str] = None,
    subplot_height: Number = 8,
    subplot_width: Number = 8
    ) -> tuple[plt.Figure, list[plt.Axes]]:

    if n_plots > 1:
        if n_keys > 1:
            # determine the layout of the subplots
            n_rows = n_plots
            max_cols = n_keys
            n_plots = n_rows * max_cols

            # create subplots
            fig, axs = plt.subplots(
                n_rows, max_cols,
                figsize=(subplot_width * max_cols, subplot_height * n_rows),
                dpi=dpi_display)
            fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.
        elif n_keys == 1:
            # determine the layout of the subplots
            n_plots, n_rows, max_cols = get_nrows_maxcols(n_keys=n_plots, max_cols=max_cols)
            fig, axs = plt.subplots(n_rows, max_cols,
                                    figsize=(subplot_width * max_cols, subplot_height * n_rows),
                                    dpi=dpi_display)
            fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.

            if n_plots > 1:
                axs = axs.ravel()
            else:
                axs = np.array([axs])

            remove_empty_subplots(
                axes=axs,
                nplots=n_plots,
                nrows=n_rows,
                ncols=max_cols
                )
        else:
            raise ValueError(f"n_keys < 1: {n_keys}")

    else:
        n_plots = n_keys
        if max_cols is None:
            max_cols = n_plots
            n_rows = 1
        else:
            if n_plots > max_cols:
                n_rows = math.ceil(n_plots / max_cols)
            else:
                n_rows = 1
                max_cols = n_plots

        fig, axs = plt.subplots(
            n_rows, max_cols,
            figsize=(subplot_width * max_cols, subplot_height * n_rows),
            dpi=dpi_display)

        if n_plots > 1:
            axs = axs.ravel()
        else:
            axs = np.array([axs])

        # remove axes from empty plots
        remove_empty_subplots(
            axes=axs,
            nplots=n_plots,
            nrows=n_rows,
            ncols=max_cols,
            )

    if header is not None:
        plt.suptitle(header, fontsize=24, x=0.5, y=1.02)

    return fig, axs

def _generate_experiment_subplots(
    data,
    n_keys: int,
    max_cols: int = 4,
    dpi_display: int = 80,
    header: Optional[str] = None
    ) -> tuple[plt.Figure, list[plt.Axes]]:
    try:
        n_data = len(data)
    except TypeError:
        # if the data is an InSituData, it raises a TypeError
        n_data = 1

    fig, axs = _generate_subplots(
        n_plots=n_data,
        n_keys=n_keys,
        max_cols=max_cols,
        dpi_display=dpi_display,
        header=header
    )

    return fig, axs


def plot_colorlegend(
    viewer: Optional[Viewer] = None,
    mapping: Optional[None] = None,
    layer_name: Optional[str] = None,
    max_per_col: int = 10,
    title: Optional[str] = None,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    verbose: bool = True,
    return_status: bool = False
    ):
    do_plotting = True
    if viewer is None and mapping is None:
        raise ValueError("Both `viewer` and `mapping` are None. One of both must not be None.")

    if viewer is not None and mapping is not None:
        raise ValueError("Both `viewer` and `mapping` are not None. At least one of both must be None.")

    if viewer is None:
        if title is None:
            title = "Color legend"
    else:
        # automatically get layer
        if layer_name is None:
            candidate_layers = [l for l in viewer.layers if l.name.startswith(f"{_config.current_data_name}")]
            try:
                layer_name = candidate_layers[0].name
            except IndexError:
                raise ValueError("No layer with cellular transcriptomic data found. First add a layer using the 'Show Data' widget.")

        # extract layer
        layer = viewer.layers[layer_name]

        # get values
        try:
            values = layer.properties["value"]
        except AttributeError:
            show_warning("The selected layer does not contain any plottable data.")
            do_plotting = False
        else:
            # create color mapping
            rgba_list, mapping, cmap = _data_to_rgba(values, rgba_values=layer.face_color, nan_val=None)

            # set title
            title = layer_name

    if do_plotting:
        if isinstance(mapping, dict):
            # categorical colorbar
            # create a figure for the colorbar
            fig, ax = plt.subplots(
                #figsize=(5, 3)
                )
            fig.subplots_adjust(bottom=0.5)

            # add color legend to axis
            _add_colorlegend_to_axis(
                color_dict=mapping,
                ax=ax,
                max_per_col=max_per_col,
                title=title
                )

        else:
            # continuous colorlegend
            # create a figure for the colorbar
            fig, ax = plt.subplots(
                figsize=(6, 1)
                )
            fig.subplots_adjust(bottom=0.5)

            # Add the colorbar to the figure
            cbar = fig.colorbar(mapping, orientation='horizontal', cax=ax)
            cbar.ax.set_title(title)

        save_and_show_figure(
            savepath=savepath,
            fig=fig,
            save_only=save_only,
            dpi_save=dpi_save,
            tight=False,
            verbose=verbose
            )
        plt.show()

    if return_status:
        return do_plotting

def _get_adata(d, cells_layer, mask_col):
    # get data
    celldata = _get_cell_layer(
        cells=d.cells,
        cells_layer=cells_layer,
        verbose=False
        )
    adata = celldata.matrix.copy()

    if mask_col is not None:
        adata = adata[adata.obs[mask_col]].copy()

    return adata

def calc_cellular_composition(
    data: Union[InSituData, InSituExperiment],
    cell_type_col: str,
    cell_type_values: Optional[Union[str, List[str]]] = None,
    cells_layer: Optional[str] = None,
    mask_col: Optional[str] = None,
    geom_key: Optional[str] = None,
    geom_values: Optional[Union[str, List[str]]] = None,
    modality: Literal["regions", "annotations"] = "regions",
    groupby: str = "uid",
    normalize: bool = True,
    force_assignment: bool = False,
    fill_missing_categories: bool = True
    ) -> pd.DataFrame:

    if geom_values is not None:
        geom_values = convert_to_list(geom_values)

    if cell_type_values is not None:
        cell_type_values = convert_to_list(cell_type_values)

    # check data
    is_experiment = _is_experiment(data)
    if is_experiment:
        exp = data
    else:
        exp = InSituExperiment()
        exp.add(data, metadata={groupby: data.sample_id})

    all_data_names = exp.metadata[groupby].values

    if not len(all_data_names) == len(np.unique(all_data_names)):
        raise ValueError(f"Values in {groupby} were found to be not unique. Please choose a column with unique values in `.metadata`.")

    # retrieve cell type compositions
    compositions_dict = {}
    for m, d in exp.iterdata():

        data_name = m[groupby]

        if geom_key is not None:
            # check whether the key exists in the selected geometry
            mod = d.get_modality(modality)
            if mod is None:
                raise ValueError(f"Modality '{modality}' not available.")
            if geom_key in mod.keys():
                # check whether the cells were already assigned to the requested geometry
                _check_assignment(
                    data=d,
                    cells_layer=cells_layer,
                    key=geom_key,
                    force_assignment=force_assignment,
                    modality=modality)

                # get data
                adata = _get_adata(d=d, cells_layer=cells_layer, mask_col=mask_col)

                assignment_series = adata.obsm[modality][geom_key]
                #cats = sorted([elem for elem in assignment_series.unique() if (elem != "unassigned") & ("&" not in elem)])
                cats = mod.metadata[geom_key]['classes']

                # calculate compositions
                compositions = {}
                for cat in cats:
                    if geom_values is not None:
                        if cat not in geom_values:
                            # skip this category
                            continue

                    # calculate percentage
                    #idx = assignment_series[assignment_series == cat].index
                    idx = assignment_series[assignment_series.str.contains(cat)].index
                    value_counts = adata.obs[cell_type_col].loc[idx].value_counts(normalize=normalize) * 100

                    if cell_type_values is not None:
                        # filter for cell type values
                        in_cats = value_counts.index.isin(cell_type_values)
                        count_results = value_counts[in_cats].copy()
                        count_results['Others'] = value_counts[~in_cats].sum()
                    else:
                        count_results = value_counts

                    compositions[cat] = count_results.sort_index()

                compositions = pd.DataFrame(compositions)
                collect = True

            else:
                collect = False
            #     unique_cats = np.unique(adata.obs["majority_voting_simple"])
            #     compositions = pd.DataFrame(
            #         data = {None: [np.nan] * len(unique_cats)},
            #         index = unique_cats
            #     )

        else:
            # get data
            adata = _get_adata(d=d, cells_layer=cells_layer, mask_col=mask_col)

            compositions = pd.DataFrame(
                {
                    "total": adata.obs[cell_type_col].value_counts(normalize=normalize) * 100
                    }
                )
            collect = True

        if collect:
            # collect data
            compositions_dict[data_name] = compositions


    # concatenate results
    compositions_df = pd.concat(compositions_dict, axis=1)

    if fill_missing_categories:
        # fill dataframe with missing values to get same width in all plots
        all_categories = compositions_df.columns.levels[1]

        # Create a complete MultiIndex with all combinations
        full_columns = pd.MultiIndex.from_product(
            [all_data_names, all_categories],
            names=compositions_df.columns.names
            )

        # Reindex the columns to include all combinations
        compositions_df = compositions_df.reindex(columns=full_columns)

    # swap multi index levels to have annotations/regions on top of samples
    compositions_df = compositions_df.swaplevel(0, 1, axis=1)

    compositions_df.columns.names = [geom_key, groupby]

    # sort by index
    compositions_df = compositions_df.sort_index()

    return compositions_df


def plot_cellular_composition(
    data: Union[InSituData, InSituExperiment],
    cell_type_col: str,
    cell_type_values: Optional[Union[str, List[str]]] = None,
    cells_layer: Optional[str] = None,
    mask_col: Optional[str] = None,
    geom_key: Optional[str] = None,
    geom_values: Optional[Union[str, List[str]]] = None,
    modality: Optional[Literal["regions", "annotations"]] = None,
    plot_type: Literal["bar", "barh"] = "barh",
    groupby: str = "uid",
    normalize: bool = True,
    force_assignment: bool = False,
    max_cols: int = 4,
    aspect_factor: Number = 1,
    legend_max_per_col: int = 10,
    savepath: Union[str, os.PathLike, Path] = None,
    palette: Optional[Union[ListedColormap, List[str]]] = DEFAULT_CATEGORICAL_CMAP,
    return_data: bool = False,
    save_only: bool = False,
    dpi_save: int = 300,
    ):

    """
    Plots the composition of cell types for specified regions or annotations.

    This function generates pie charts or a single stacked bar plot to visualize the proportions of different cell types
    within specified regions or annotations. It can optionally save the plot to a file and
    return the composition data.
    """
    if isinstance(palette, list):
        palette = ListedColormap(palette)
        palette_is_dict = False
    elif isinstance(palette, ListedColormap):
        palette_is_dict = False
        pass
    elif isinstance(palette, dict):
        palette_is_dict = True
        pass
    else:
        raise ValueError(f"palette must be a list of colors or a ListedColormap. Instead: {type(palette)}")

    if geom_key is not None:
        if modality is None:
            raise ValueError("If `geom_key` is not None, modality must not be None. Choose either 'regions' or 'annotations'.")

    compositions_df = calc_cellular_composition(
        data=data,
        cell_type_col=cell_type_col, cell_type_values=cell_type_values,
        cells_layer=cells_layer, mask_col=mask_col,
        geom_key=geom_key, geom_values=geom_values,
        modality=modality, groupby=groupby,
        normalize=normalize, force_assignment=force_assignment,
    )

    # retrieve names from data
    geom_names = compositions_df.columns.levels[0].values
    data_names = compositions_df.columns.levels[1].values
    cell_type_names = compositions_df.index.values

    # check data
    is_experiment = _is_experiment(data)
    if is_experiment:
        # get colors for plotting
        if cell_type_col in data.colors:
            color_dict = data.colors[cell_type_col]
        else:
            color_dict = None
    else:
        # assume it is an InSituData object
        celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)

        if palette_is_dict:
            color_dict = palette
        else:
            try:
                color_dict = map_to_colors(
                    sorted(celldata.matrix.obs[cell_type_col].unique()),
                    palette=palette)
                #color_dict = celldata.matrix.uns[f"{cell_type_col}_colors"]
            except KeyError:
                color_dict = None

    # setup plot
    if len(geom_names) == 1:
        n_plots = 1
        separate_legend = False
    elif len(geom_names) > 1:
        n_plots = len(geom_names) + 1
        separate_legend = True
    else:
        raise ValueError(f"geom_names has length 0.")

    if plot_type == "bar":
        subplot_width = (0.5+len(data_names)*1) / aspect_factor
        subplot_height = 7.5
    elif plot_type == "barh":
        subplot_width = 7.5
        subplot_height = (0.5+len(data_names)*1) * aspect_factor
    else:
        raise ValueError(f"plot_type must be either 'bar' or 'barh'. Instead: {plot_type}")

    # generate the subplots based on number of data
    fig, axs = _generate_subplots(
        n_plots=n_plots, n_keys=1,
        max_cols=max_cols,
        subplot_width=subplot_width,
        subplot_height=subplot_height
    )

    for i, geom_name in enumerate(geom_names):
        compositions = compositions_df.loc[:, geom_name]
        n_cats = compositions.shape[1]
        ax = axs[i]
        # Plot a single stacked bar plot
        if plot_type == "bar":
            ylabel = "%"
            xlabel = "Dataset"
            inverty = False
        else:
            ylabel = "Dataset"
            xlabel = "%"
            inverty = True

        if color_dict is None:
            color_list = palette.colors
        else:
            color_list = []
            for elem in compositions.index:
                try:
                    color_list.append(color_dict[elem])
                except KeyError as e:
                    if elem == "Others":
                        color_list.append("#d3d3d3") # append lightgrey hex code
                    else:
                        raise KeyError(e)

            #color_list = [color_dict[elem] for elem in compositions.index]
        compositions.T.plot(kind=plot_type, stacked=True,
                            #figsize=(fig_width, fig_height),
                            width=0.7, ax=ax, legend=not separate_legend,
                            #color=color_list
                            color=color_list
                            )

        if not separate_legend:
            ax.legend(title=cell_type_col, bbox_to_anchor=(1.05, 1), loc='upper left')

        if inverty:
            plt.gca().invert_yaxis()
        ax.set_title(geom_name)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    if separate_legend:
        # map colors to cell type names
        if color_dict is None:
            color_dict = map_to_colors(cat_list=cell_type_names, palette=palette)
        #color_dict = {cat: palette(i % palette.N) for i, cat in enumerate(cell_type_names)}
        # create legend in additional plot
        _add_colorlegend_to_axis(
            color_dict=color_dict,
            ax=axs[len(geom_names)],
            max_per_col=legend_max_per_col,
            loc='center',
            bbox_to_anchor=(0.5, 0.5),
            mode="rectangle",
            remove_axis=True
            )

    save_and_show_figure(
        savepath=savepath,
        fig=fig,
        save_only=save_only,
        dpi_save=dpi_save,
        tight=separate_legend
        )

    if return_data:
        return compositions_df

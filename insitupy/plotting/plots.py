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
from napari.viewer import Viewer

import insitupy._core._config as _config
from insitupy._constants import DEFAULT_CATEGORICAL_CMAP
from insitupy._core._checks import _check_assignment, _is_experiment
from insitupy._core._utils import _get_cell_layer
from insitupy._core.insitudata import InSituData
from insitupy._core.insituexperiment import InSituExperiment
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting._colors import _add_colorlegend_to_axis, _data_to_rgba
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
    viewer: Viewer,
    layer_name: Optional[str] = None,
    max_per_row: int = 10,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    ):
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
    values = layer.properties["value"]

    # create color mapping
    rgba_list, mapping, cmap = _data_to_rgba(values, rgba_values=layer.face_color, nan_val=None)

    if isinstance(mapping, dict):
        # categorical colorbar
        # create a figure for the colorbar
        fig, ax = plt.subplots(
            #figsize=(5, 3)
            )
        fig.subplots_adjust(bottom=0.5)

        # add color legend to axis
        _add_colorlegend_to_axis(color_dict=mapping, ax=ax, max_per_row=max_per_row)

    else:
        # continuous colorlegend
        # create a figure for the colorbar
        fig, ax = plt.subplots(
            figsize=(6, 1)
            )
        fig.subplots_adjust(bottom=0.5)

        # Add the colorbar to the figure
        cbar = fig.colorbar(mapping, orientation='horizontal', cax=ax)
        cbar.ax.set_title(layer_name)

    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=False)
    plt.show()

def calc_cellular_composition(
    data: Union[InSituData, InSituExperiment],
    cell_type_col: str,
    cells_layer: Optional[str] = None,
    geom_key: Optional[str] = None,
    modality: Literal["regions", "annotations"] = "regions",
    uid_column: str = "sample_id",
    normalize: bool = True,
    force_assignment: bool = False,
    fill_missing_categories: bool = True
    ) -> pd.DataFrame:

    # check data
    is_experiment = _is_experiment(data)
    if is_experiment:
        exp = data
    else:
        exp = InSituExperiment()
        exp.add(data, metadata={uid_column: data.sample_id})

    all_data_names = exp.metadata[uid_column].values

    if not len(all_data_names) == len(all_data_names):
        raise ValueError(f"Values in {uid_column} were found to be not unique. Please choose a column with unique values in `.metadata`.")

    # retrieve cell type compositions
    compositions_dict = {}
    for m, d in exp.iterdata():
        celldata = _get_cell_layer(
            cells=d.cells,
            cells_layer=cells_layer,
            verbose=False
            )
        adata = celldata.matrix
        data_name = m[uid_column]

        if geom_key is not None:
            # check whether the key exists in the selected geometry
            if geom_key in d.get_modality(modality).keys():
                # check whether the cells were already assigned to the requested geometry
                _check_assignment(
                    data=d,
                    cells_layer=cells_layer,
                    key=geom_key,
                    force_assignment=force_assignment,
                    modality=modality)

                assignment_series = adata.obsm[modality][geom_key]
                cats = sorted([elem for elem in assignment_series.unique() if (elem != "unassigned") & ("&" not in elem)])

                # calculate compositions
                compositions = {}
                for cat in cats:
                    idx = assignment_series[assignment_series == cat].index
                    compositions[cat] = adata.obs[cell_type_col].loc[idx].value_counts(normalize=normalize) * 100 # calculate percentage
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

    compositions_df.columns.names = [geom_key, uid_column]

    return compositions_df


def plot_cellular_composition(
    data: Union[InSituData, InSituExperiment],
    cell_type_col: str,
    cells_layer: Optional[str] = None,
    geom_key: Optional[str] = None,
    modality: Literal["regions", "annotations"] = "regions",
    plot_type: Literal["bar", "barh"] = "barh",
    uid_column: str = "sample_id",
    normalize: bool = True,
    force_assignment: bool = False,
    max_cols: int = 4,
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

    # if isinstance(palette, ListedColormap):
    #     color_list = palette.colors
    # elif isinstance(palette, list):
    #     color_list = palette
    # else:
    #     raise ValueError(f"palette must be a list of colors or a ListedColormap. Instead: {type(palette)}")

    if isinstance(palette, list):
        palette = ListedColormap(palette)
    elif isinstance(palette, ListedColormap):
        pass
    else:
        raise ValueError(f"palette must be a list of colors or a ListedColormap. Instead: {type(palette)}")

    compositions_df = calc_cellular_composition(
        data=data, cell_type_col=cell_type_col,
        cells_layer=cells_layer, geom_key=geom_key,
        modality=modality, uid_column=uid_column,
        normalize=normalize, force_assignment=force_assignment,
    )

    # retrieve names from data
    geom_names = compositions_df.columns.levels[0].values
    data_names = compositions_df.columns.levels[1].values
    cell_type_names = compositions_df.index.values

    if len(geom_names) == 1:
        n_plots = 1
        separate_legend = False
    elif len(geom_names) > 1:
        n_plots = len(geom_names) + 1
        separate_legend = True
    else:
        raise ValueError(f"geom_names has length 0.")

    if plot_type == "bar":
        subplot_width = 0.5+len(data_names)*1
        subplot_height = 8
    elif plot_type == "barh":
        subplot_width = 8
        subplot_height = len(data_names)*0.8
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
            # fig_width = 1*n_cats
            # fig_height = 6
            ylabel = "%"
            xlabel = modality
            inverty = False
        else:
            # fig_width = 8
            # fig_height = 1*n_cats
            ylabel = modality
            xlabel = "%"
            inverty = True


        compositions.T.plot(kind=plot_type, stacked=True,
                            #figsize=(fig_width, fig_height),
                            width=0.7, ax=ax, legend=not separate_legend,
                            #color=color_list
                            color=palette.colors
                            )

        if not separate_legend:
            ax.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')

        if inverty:
            plt.gca().invert_yaxis()
        ax.set_title(geom_name)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    if separate_legend:
        # create legend in additional plot
        color_dict = {cat: palette(i % palette.N) for i, cat in enumerate(cell_type_names)}
        _add_colorlegend_to_axis(
            color_dict=color_dict,
            ax=axs[len(geom_names)],
            max_per_row=np.inf,
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
        return compositions

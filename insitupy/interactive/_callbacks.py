import math
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pandas.api.types import is_numeric_dtype

from insitupy import WITH_NAPARI
from insitupy._constants import (ANNOTATIONS_SYMBOL, POINTS_SYMBOL,
                                 REGIONS_SYMBOL)
from insitupy.utils._colors import continuous_data_to_rgba

if WITH_NAPARI:
    import napari

# show cells widget
def _update_key_on_type_change(widget, viewer_config):
    current_key_type = widget.key_type.value
    widget.key.choices = viewer_config.key_dict[current_key_type]

# geometry widget
def _update_keys_based_on_geom_type(widget, xdata):
    # retrieve current value
    current_geom_type = widget.geom_type.value
    current_key = widget.key.value

    # get either regions or annotations object
    geom_data = getattr(xdata, current_geom_type.lower())
    widget.key.choices = sorted(geom_data.metadata.keys(), key=str.casefold)

def _update_classes_on_key_change(widget, xdata):
    # get current values for geom_type and key
    current_geom_type = widget.geom_type.value
    current_key = widget.key.value

    # get either regions or annotations object
    geom_data = getattr(xdata, current_geom_type.lower())

    # update annot_class choices
    widget.annot_class.choices = ["all"] + sorted(geom_data.metadata[current_key]['classes'])

def _set_show_names_based_on_geom_type(widget):
    # retrieve current value
    current_geom_type = widget.geom_type.value

    # set the show_names tick box
    if current_geom_type == "Annotations":
        widget.show_names.value = False

    if current_geom_type == "Regions":
        widget.show_names.value = True


# Function to update the legend
def _update_categorical_legend(
    static_canvas,
    mapping: dict,
    label: str,
    max_rows: int = 6,
    marker: str = "o",
    marker_mode: Literal["face", "edge"] = "face"
    ):

    # Calculate the number of columns needed
    num_items = len(mapping)
    ncols = math.ceil(num_items / max_rows)

    # prepare figue
    static_canvas.figure.clear()  # Clear the current figure
    axes = static_canvas.figure.subplots()  # Create new axes

    if marker_mode == "face":
        # create legend handles with face colored
        legend_handles = [Line2D([0], [0],
                                marker=marker, color='w', label=n,
                                markerfacecolor=c, markeredgecolor='k',
                                markersize=7) for n,c in mapping.items()]
    elif marker_mode == "edge":
        # create legend handles with edges colored
        legend_handles = [Line2D([0], [0],
                                marker=marker, color='w', label=n,
                                markerfacecolor='w', markeredgecolor=c,
                                markeredgewidth=2,
                                markersize=7) for n,c in mapping.items()]
    else:
        raise ValueError(f"marker_mode must be either 'face' or 'edge'. Instead: {marker_mode}")

    # add legend to axis
    axes.legend(handles=legend_handles, loc="center", title=label, ncols=ncols,
                fontsize=8, title_fontsize=10,
                labelspacing=0.7, borderpad=0.5)
    axes.set_axis_off()

    static_canvas.draw()  # Redraw the canvas

def _update_continuous_legend(static_canvas, mapping, label):
    static_canvas.figure.clear()  # Clear the current figure
    gs = GridSpec(1, 1, top=1.2, bottom=0.6, left=-0.5, right=1.5)  # Define the grid spec
    axes = static_canvas.figure.add_subplot(gs[0])  # Add subplot with the grid spec

    colorbar = static_canvas.figure.colorbar(mapping, ax=axes, orientation='horizontal')
    colorbar.set_label(label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)  # Adjust tick label size
    #colorbar.set_ticks(np.linspace(norm.vmin, norm.vmax, num=5))  # Set the number of ticks
    axes.set_axis_off()
    static_canvas.draw()  # Redraw the canvas

def _update_colorlegend(viewer, viewer_config):
    layer = viewer.layers.selection.active

    if isinstance(layer, napari.layers.points.points.Points):
        try:
            # get values
            values = layer.properties["value"]
            color_values = layer.face_color
        except KeyError:
            first_char = layer.name[:1]
            if first_char == POINTS_SYMBOL:
                # collect the layer names and edge colors of the respective layer
                layer_names = []
                face_colors = []
                for elem in viewer.layers:
                    if elem.name.startswith(first_char):
                        layer_names.append(elem.name.strip(first_char + " "))
                        face_colors.append(elem.current_face_color)

                # create mapping from collected values
                mapping = dict(zip(layer_names, face_colors))

                _update_categorical_legend(
                    static_canvas=viewer_config.static_canvas,
                    mapping=mapping,
                    label="Points",
                    marker="o",
                    marker_mode="face"
                    )
        else:
            if is_numeric_dtype(values):
                rgba_list, mapping = continuous_data_to_rgba(data=values,
                                        cmap=layer.face_colormap.name,
                                        #upper_climit_pct=upper_climit_pct,
                                        return_mapping=True
                                        )

                _update_continuous_legend(
                    static_canvas=viewer_config.static_canvas,
                    mapping=mapping,
                    label=layer.name)

            else:
                # substitute pd.NA with np.nan
                values = pd.Series(values).fillna(np.nan).values
                # assume the data is categorical
                #mapping = {category: tuple(rgba) for category, rgba in zip(values, color_values)}
                unique_values = list(set(values))
                mapping = {str(v): tuple(color_values[list(values).index(v)]) for v in unique_values}
                # sort mapping dict
                mapping = {elem: mapping[elem] for elem in sorted(mapping.keys())}

                _update_categorical_legend(
                    static_canvas=viewer_config.static_canvas,
                    mapping=mapping,
                    label=layer.name
                    )

    elif isinstance(layer, napari.layers.shapes.shapes.Shapes):
        # check if the layer is a annotations or regions layer
        first_char = layer.name[:1]
        if first_char in [ANNOTATIONS_SYMBOL, REGIONS_SYMBOL]:
            # collect the layer names and edge colors of the respective layer
            layer_names = []
            face_colors = []
            for elem in viewer.layers:
                if elem.name.startswith(first_char):
                    layer_names.append(elem.name.strip(first_char + " "))
                    face_colors.append(elem.current_edge_color)

            # create mapping from collected values
            mapping = dict(zip(layer_names, face_colors))

            _update_categorical_legend(
                static_canvas=viewer_config.static_canvas,
                mapping=mapping,
                label="Annotations" if first_char == ANNOTATIONS_SYMBOL else "Regions",
                marker="s",
                marker_mode="edge"
                )
    else:
        pass


def _refresh_widgets_after_data_change(xdata, viewer, viewer_config, select_data_widget, show_cells_widget, boundaries_widget, filter_widget):
    #_config.init_viewer_config(xdata)
    #viewer_config.__init__(xdata) # re-initialize ViewerConfig

    if select_data_widget is not None:
        select_data_widget.data_name.value = viewer_config.data_name
        select_data_widget.layer_name.value = viewer_config.layer_name

    if boundaries_widget is not None:
        # set choices
        boundaries_widget.key.choices = viewer_config.masks

    if show_cells_widget is not None:
        # reset the currently selected key to None
        show_cells_widget.key.value = None

        # update choices for key
        show_cells_widget.key.choices = viewer_config.key_dict[show_cells_widget.key_type.value]

        # add last addition to recent
        show_cells_widget.recent.choices = sorted(viewer_config.recent_selections)
        show_cells_widget.recent.value = None

    if filter_widget is not None:
        # update obs in filter widget
        filter_widget.obs_key.choices = viewer_config.key_dict["obs"]

    # # set only the last cell layer visible
    # cell_layers = []
    # for elem in viewer.layers:
    #     if isinstance(elem, napari.layers.points.points.Points):
    #         if not elem.name.startswith(POINTS_SYMBOL):
    #             # only if the layer is not a point annotation layer, it is added
    #             cell_layers.append(elem)
    # #point_layers = [elem for elem in xdata.viewer.layers if isinstance(elem, napari.layers.points.points.Points)]
    # n_cell_layers = len(cell_layers)

    # # make only last cell layer visible
    # for i, l in enumerate(cell_layers):
    #     if i < n_cell_layers-1:
    #         l.visible = False
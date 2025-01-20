from numbers import Number
from typing import List, Optional
from warnings import warn

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from shapely.geometry.multipolygon import MultiPolygon

import insitupy._core._callbacks
import insitupy._core.config as config
from insitupy import WITH_NAPARI
from insitupy._core._layers import _create_points_layer, _update_points_layer

from .._constants import (ANNOTATIONS_SYMBOL, POINTS_SYMBOL, REGION_CMAP,
                          REGIONS_SYMBOL)
from ..images.utils import create_img_pyramid
from ._callbacks import (_refresh_widgets_after_data_change,
                         _set_show_names_based_on_geom_type,
                         _update_classes_on_key_change, _update_colorlegend,
                         _update_keys_based_on_geom_type)

if WITH_NAPARI:
    import napari
    from magicgui import magic_factory, magicgui
    from magicgui.widgets import FunctionGui

    from ._layers import _add_annotations_as_layer

    def _initialize_widgets(
        xdata # xenium data object
        ) -> List[FunctionGui]:

        # access viewer from xeniumdata
        viewer = xdata.viewer

        if xdata.cells is None:
            add_cells_widget = None
            move_to_cell_widget = None
            add_boundaries_widget = None,
            filter_cells_widget = None
        else:
            # initialize data_name of viewer
            config.init_data_name()
            # initialize viewer configuration
            config.init_viewer_config(xdata=xdata,
                                        #data_name=config.current_data_name
                                        )
            config.init_recent_selections()

            data_names = xdata.cells.get_all_keys()

            @magicgui(
                call_button=False,
                data_name= {'choices': data_names, 'label': 'Dataset:'}
            )
            def select_data(
                data_name="main"
            ):
                pass

            # connect key change with update function
            @select_data.data_name.changed.connect
            def update_widgets_on_data_change(event=None):
                config.current_data_name = select_data.data_name.value
                insitupy._core._callbacks._refresh_widgets_after_data_change(xdata,
                                                    add_cells_widget,
                                                    add_boundaries_widget,
                                                    filter_cells_widget
                                                    )

            if len(config.masks) > 0:
                @magicgui(
                    call_button='Add',
                    key={'choices': config.masks, 'label': 'Masks:'}
                )
                def add_boundaries_widget(
                    key
                ):
                    layer_name = f"{config.current_data_name}-{key}"

                    if layer_name not in viewer.layers:
                        # get geopandas dataframe with regions
                        mask = config.boundaries[key]

                        # get metadata for mask
                        metadata = config.boundaries.metadata
                        pixel_size = metadata[key]["pixel_size"]

                        if not isinstance(mask, list):
                            # generate pyramid of the mask
                            mask_pyramid = create_img_pyramid(img=mask, nsubres=6)
                        else:
                            mask_pyramid = mask

                        # add masks as labels to napari viewer
                        viewer.add_labels(mask_pyramid, name=layer_name, scale=(pixel_size,pixel_size))
                    else:
                        print(f"Layer '{layer_name}' already in layer list.", flush=True)
            else:
                add_boundaries_widget = None

            def _update_values_on_key_change(widget):
                current_key = widget.key.value
                widget.value.choices = config.value_dict[current_key]

            @magicgui(
                call_button='Add',
                key={'choices': ["genes", "obs", "obsm"], 'label': 'Key:'},
                value={'choices': config.genes, 'label': "Value:"},
                size={'label': 'Size [µm]'},
                recent={'choices': [""], 'label': "Recent:"},
                add_new_layer={'label': 'Add new layer'}
                )
            def add_cells_widget(
                key="genes",
                value=None,
                size=8,
                recent=None,
                add_new_layer=False,
                viewer=viewer
                ) -> napari.types.LayerDataTuple:

                # get names of cells
                cell_names = config.adata.obs_names.values

                #layers_to_add = []
                if value is not None or recent is not None:
                    if value is None:
                        key = recent.split(":", maxsplit=1)[0]
                        value = recent.split(":", maxsplit=1)[1]
                    #if gene not in viewer.layers:
                    # get expression values
                    if key == "genes":
                        gene_loc = config.adata.var_names.get_loc(value)
                        color_value = config.X[:, gene_loc]
                    elif key == "obs":
                        color_value = config.adata.obs[value]
                    elif key == "obsm":
                        #TODO: Implement it for obsm
                        obsm_key = value.split("#", maxsplit=1)[0]
                        obsm_col = value.split("#", maxsplit=1)[1]
                        data = config.adata.obsm[obsm_key]

                        if isinstance(data, pd.DataFrame):
                            color_value = data[obsm_col].values
                        elif isinstance(data, np.ndarray):
                            color_value = data[:, int(obsm_col)-1].values
                        else:
                            warn("Data in `obsm` needs to be either pandas DataFrame or numpy array to be parsed.")
                        pass
                    else:
                        print("Unknown key selected.", flush=True)

                    new_layer_name = f"{config.current_data_name}-{value}"

                    # get layer names from the current data
                    layer_names_for_current_data = [elem.name for elem in viewer.layers if elem.name.startswith(config.current_data_name)]

                    # select only point layers
                    layer_names_for_current_data = [elem for elem in layer_names_for_current_data if isinstance(viewer.layers[elem], napari.layers.points.points.Points)]

                    # save last addition to add it to recent in the callback
                    config.recent_selections.append(f"{key}:{value}")

                    if len(layer_names_for_current_data) == 0:

                        # create points layer for genes
                        gene_layer = _create_points_layer(
                            points=config.points,
                            color_values=color_value,
                            #name=f"{config.current_data_name}-{gene}",
                            name=new_layer_name,
                            point_names=cell_names,
                            point_size=size,
                            upper_climit_pct=99
                        )
                        return gene_layer
                        #layers_to_add.append(gene_layer)
                    else:
                        if not add_new_layer:
                            #print(f"Key '{gene}' already in layer list.", flush=True)
                            # update the existing points layer
                            layer = viewer.layers[layer_names_for_current_data[0]]
                            _update_points_layer(
                                layer=layer,
                                new_color_values=color_value,
                                new_name=new_layer_name,
                            )
                        else:
                            # create new points layer for genes
                            gene_layer = _create_points_layer(
                                points=config.points,
                                color_values=color_value,
                                #name=f"{config.current_data_name}-{gene}",
                                name=new_layer_name,
                                point_names=cell_names,
                                point_size=size,
                                upper_climit_pct=99
                            )
                            return gene_layer

            @magicgui(
                call_button='Filter',
                obs_key={'choices': config.value_dict["obs"], 'label': "Obs:"},
                operation_type={'choices': ["contains", "is equal to"], 'label': 'Operation:'},
                obs_value={'label': 'Value:'},
                reset={'label': 'Reset'}
                )
            def filter_cells_widget(
                obs_key=None,
                operation_type="contains",
                obs_value: str = "",
                reset: bool = False,
                viewer=viewer
            ):
                # find currently selected layer
                layers = viewer.layers
                selected_layers = list(layers.selection)

                if not reset:
                    # create filtering mask
                    if operation_type == "contains":
                        mask = config.adata.obs[obs_key].str.contains(obs_value)
                    elif operation_type == "is equal to":
                        mask = config.adata.obs[obs_key] == obs_value
                    else:
                        raise ValueError(f"Unknown operation type: {operation_type}.")

                    # iterate through selected layers
                    for current_layer in selected_layers:
                        if isinstance(current_layer, napari.layers.points.points.Points):
                            # set visibility
                            fc = current_layer.face_color.copy()
                            fc[:, -1] = 0.
                            fc[mask, -1] = 1.
                            current_layer.face_color = fc
                else:
                    for current_layer in selected_layers:
                        # reset visibility
                        fc = current_layer.face_color.copy()
                        fc[:, -1] = 1.
                        current_layer.face_color = fc

            @add_cells_widget.key.changed.connect
            @add_cells_widget.call_button.changed.connect
            def update_values_on_key_change(event=None):
                _update_values_on_key_change(add_cells_widget)

            @magicgui(
                call_button='Show',
                cell={'label': "Cell:"},
                zoom={'label': 'Zoom:'},
                highlight={'label': 'Highlight'}
                )
            def move_to_cell_widget(
                cell="",
                zoom=5,
                highlight=True,
                ) -> Optional[napari.types.LayerDataTuple]:
                if cell in config.adata.obs_names.astype(str):
                    # get location of selected cell
                    cell_loc = config.adata.obs_names.get_loc(cell)
                    cell_position = config.points[cell_loc]

                    # move center of camera to cell position
                    viewer.camera.center = (0, cell_position[0], cell_position[1])
                    viewer.camera.zoom = zoom

                    if highlight:
                        name = f"cell-{cell}"
                        if name not in viewer.layers:
                            viewer.add_points(
                                data=np.array([cell_position]),
                                name=name,
                                size=6,
                                face_color=[0,0,0,0],
                                opacity=1,
                                edge_color='red',
                                edge_width=0.1
                            )
                else:
                    print(f"Cell '{cell}' not found in `xeniumdata.cells.matrix.obs_names()`.")

            def callback_refresh(event=None):
                # after the points widget is run, the widgets have to be refreshed to current data layer
                _refresh_widgets_after_data_change(xdata,
                                                        points_widget=add_cells_widget,
                                                        boundaries_widget=add_boundaries_widget,
                                                        filter_widget=filter_cells_widget
                                                        )

            def callback_update_legend(event=None):
                _update_colorlegend()

            if add_cells_widget is not None:
                add_cells_widget.call_button.clicked.connect(callback_refresh)
                add_cells_widget.call_button.clicked.connect(callback_update_legend)
            if add_boundaries_widget is not None:
                add_boundaries_widget.call_button.clicked.connect(callback_refresh)
                add_boundaries_widget.call_button.clicked.connect(callback_update_legend)

            viewer.layers.selection.events.active.connect(callback_update_legend)

        if xdata.annotations is None and xdata.regions is None:
            show_geometries_widget = None
        else:
            # check which geometries are available
            if xdata.annotations is not None:
                if xdata.regions is not None:
                    choices = ["Annotations", "Regions"]
                else:
                    choices = ["Annotations"]
            else:
                choices = ["Regions"]
            # extract geometry object
            geom = getattr(xdata, choices[0].lower())

            # extract annotations keys
            annot_keys = list(geom.metadata.keys())
            try:
                first_annot_key = list(annot_keys)[0] # for dropdown menu
            except IndexError:
                show_geometries_widget = None
            else:
                first_classes = ["all"] + sorted(geom.metadata[first_annot_key]['classes'])

                @magicgui(
                    call_button='Show',
                    geom_type={"choices": choices, "label": "Type:"},
                    key={"choices": annot_keys, "label": "Key:"},
                    annot_class={"choices": first_classes, "label": "Class:"},
                    show_names={'label': 'Show names'}
                )
                def show_geometries_widget(
                    geom_type,
                    key,
                    annot_class,
                    tolerance: Number = 2,
                    show_names: bool = False
                    ):

                    if geom_type == "Annotations":
                        # get annotation dataframe
                        annot_df = xdata.annotations[key]
                        all_keys = list(xdata.annotations.metadata.keys())
                    elif geom_type == "Regions":
                        # get regions dataframe
                        annot_df = xdata.regions[key]
                        all_keys = list(xdata.regions.metadata.keys())
                    else:
                        TypeError(f"Unknown geometry type: {geom_type}")

                    if annot_class == "all":
                        # get classes
                        classes = annot_df['name'].unique()
                    else:
                        classes = [annot_class]

                    # # check which layer types are in the dataframe
                    # layer_types = annot_df["layer_type"].unique()

                    # iterate through classes
                    for cl in classes:
                        # for layer_type in layer_types:
                        #     # generate layer name
                        #     if geom_type == "Annotations":
                        #         if layer_type == "Shapes":
                        #             layer_type_symbol = SHAPES_SYMBOL
                        #         elif layer_type == "Points":
                        #             layer_type_symbol = POINTS_SYMBOL
                        #         else:
                        #             TypeError(f"Unknown layer type: {layer_type}")
                        #     else:
                        #         layer_type_symbol = REGIONS_SYMBOL

                        layer_name = f"{cl} ({key})"

                        if layer_name not in viewer.layers:
                            # get dataframe for this class
                            class_df = annot_df[annot_df["name"] == cl].copy()

                            # simplify polygons for visualization
                            class_df["geometry"] = class_df["geometry"].simplify(tolerance)

                            # extract scale
                            #scale_factor = class_df.iloc[0]["scale"]

                            if not "color" in class_df.columns:
                                # create a RGB color with range 0-255 for this key
                                rgb_color = [elem * 255 for elem in REGION_CMAP(all_keys.index(key))][:3]
                            else:
                                rgb_color = None

                            # add layer to viewer
                            _add_annotations_as_layer(
                                dataframe=class_df,
                                viewer=viewer,
                                layer_name=layer_name,
                                #scale_factor=scale_factor,
                                rgb_color=rgb_color,
                                show_names=show_names,
                                mode=geom_type
                            )

                # connect key change with update function
                @show_geometries_widget.geom_type.changed.connect
                @show_geometries_widget.key.changed.connect
                @show_geometries_widget.call_button.clicked.connect
                @viewer.layers.events.removed.connect # somehow the values change when layers are inserted
                @viewer.layers.events.inserted.connect # or remoed. Therefore, this update is necessary
                def update_annotation_widget_after_changes(event=None):
                    _update_keys_based_on_geom_type(show_geometries_widget, xdata=xdata)
                    _update_classes_on_key_change(show_geometries_widget, xdata=xdata)
                    _set_show_names_based_on_geom_type(show_geometries_widget)

        return add_cells_widget, move_to_cell_widget, show_geometries_widget, add_boundaries_widget, select_data, filter_cells_widget #add_genes, add_observations


    @magic_factory(
        call_button='Add geometry layer',
        key={"choices": ["Geometric annotations", "Point annotations", "Regions"], "label": "Type:"},
        annot_key={'label': 'Key:'},
        class_name={'label': 'Class:'}
        )
    def add_new_geometries_widget(
        key: str = "Geometric annotations",
        annot_key: str = "TestKey",
        class_name: str = "TestClass",
    ) -> napari.types.LayerDataTuple:
        # name pattern of layer name
        name_pattern: str = "{type_symbol} {class_name} ({annot_key})"

        if (class_name != "") & (annot_key != ""):
            if key == "Geometric annotations":
                # generate name
                name = name_pattern.format(
                    type_symbol=ANNOTATIONS_SYMBOL,
                    class_name=class_name,
                    annot_key=annot_key
                    )

                # generate shapes layer for annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'shape_type': 'polygon',
                        'edge_width': 10,
                        'edge_color': 'red',
                        'face_color': 'transparent',
                        #'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object')
                        }
                        },
                    'shapes'
                    )
            elif key == "Point annotations":
                # generate name
                name = name_pattern.format(
                    type_symbol=POINTS_SYMBOL,
                    class_name=class_name,
                    annot_key=annot_key
                    )

                # generate points layer for annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'size': 25,
                        'edge_color': 'black',
                        'face_color': 'blue',
                        #'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object')
                        }
                        },
                    'points'
                    )

            elif key == "Regions":
                # generate name
                name = name_pattern.format(
                    type_symbol=REGIONS_SYMBOL,
                    class_name=class_name,
                    annot_key=annot_key
                    )

                # generate shapes layer for annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'shape_type': 'polygon',
                        'edge_width': 10,
                        'edge_color': '#ffaa00ff',
                        'face_color': 'transparent',
                        #'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object')
                        }
                        },
                    'shapes'
                    )

            else:
                layer = None

            # reset class name to nothing
            add_new_geometries_widget.class_name.value = ""

            return layer

        else:
            return None

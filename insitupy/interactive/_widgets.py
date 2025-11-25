from insitupy import WITH_NAPARI

if WITH_NAPARI:
    from typing import List, Optional

    import napari
    import numpy as np
    from magicgui import magic_factory, magicgui
    from magicgui.widgets import FunctionGui
    from matplotlib.colors import ListedColormap
    from napari.utils.notifications import show_info, show_warning
    from qtpy.QtCore import QSize, Qt
    from qtpy.QtGui import QFontMetrics, QIcon
    from qtpy.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                                QVBoxLayout, QWidget)

    from insitupy._constants import (ANNOTATIONS_SYMBOL, POINTS_SYMBOL,
                                     REGION_CMAP, REGIONS_SYMBOL)
    from insitupy.images.utils import create_img_pyramid
    from insitupy.interactive._callbacks import (
        _refresh_widgets_after_data_change, _set_show_names_based_on_geom_type,
        _update_classes_on_key_change, _update_colorlegend,
        _update_key_on_type_change, _update_keys_based_on_geom_type)
    from insitupy.interactive._configs import (ViewerConfig, _get_viewer_uid,
                                               config_manager)
    from insitupy.interactive._layers import (_create_points_layer,
                                              _update_points_layer)
    from insitupy.interactive.viewer import save_colorlegends, sync_geometries
    from insitupy.utils._helpers import _get_expression_values

    from ._layers import _add_geometries_as_layer

    def _initialize_widgets(
        viewer: napari.Viewer,
        viewer_config: ViewerConfig
        #xdata # InSituData object
        ) -> List[FunctionGui]:

        # access viewer from InSituData
        #viewer = xdata.viewer
        data = viewer_config.data

        if not viewer_config.has_cells:
            show_cells_widget = None
            move_to_cell_widget = None
            show_boundaries_widget = None,
            filter_cells_widget = None
        else:
            data_names = data.cells.get_all_keys()
            layer_names = ["main"] + list(data.cells.matrix.layers)

            @magicgui(
                call_button=False,
                data_name= {'choices': data_names, 'label': 'CellData layer:'},
                layer_name = {'choices': layer_names, 'label': 'AnnData layer:'},
            )
            def select_data_widget(
                data_name=viewer_config.data_name,
                layer_name=viewer_config.layer_name
            ):
                pass

            if len(viewer_config.masks) > 0:
                @magicgui(
                    call_button='Show',
                    key={'choices': viewer_config.masks, 'label': 'Masks:'}
                )
                def show_boundaries_widget(
                    key
                ):
                    layer_name = f"{viewer_config.data_name}-boundaries-{key}"

                    if layer_name not in viewer.layers:
                        # get geopandas dataframe with regions
                        mask = viewer_config.boundaries[key]

                        # get metadata for mask
                        metadata = viewer_config.boundaries.metadata
                        pixel_size = metadata[key]["pixel_size"]

                        if not isinstance(mask, list):
                            # generate pyramid of the mask
                            mask_pyramid = create_img_pyramid(img=mask, nsubres=6)
                        else:
                            mask_pyramid = mask

                        # add masks as labels to napari viewer
                        viewer.add_labels(mask_pyramid, name=layer_name, scale=(pixel_size,pixel_size))
                        if key == "cells":
                            viewer.layers[layer_name].contour = 1
                    else:
                        print(f"Layer '{layer_name}' already in layer list.", flush=True)
            else:
                show_boundaries_widget = None

            @magicgui(
                call_button='Show',
                key_type={'choices': ["genes", "obs", "obsm"], 'label': 'Type:'},
                key={'choices': viewer_config.genes, 'label': "Key:"},
                size={'label': 'Size [µm]'},
                recent={'choices': [""], 'label': "Recent:"},
                add_new_layer={'label': 'Add new layer'}
                )
            def show_cells_widget(
                key_type="genes",
                key=None,
                size=8,
                recent=None,
                add_new_layer=False,
                viewer=viewer
                ) -> napari.types.LayerDataTuple:

                # get names of cells
                cell_names = viewer_config.adata.obs_names.values

                #layers_to_add = []
                if key is not None or recent is not None:
                    if key is None:
                        key_type = recent.split(":", maxsplit=1)[0]
                        key = recent.split(":", maxsplit=1)[1]

                    # get expression values
                    color_value = _get_expression_values(
                        adata=viewer_config.adata,
                        X=viewer_config.X,
                        key_type=key_type, key=key
                    )

                    if viewer_config.layer_name != "main":
                        if key_type in ["obs", "obsm"]:
                            show_warning(f"Other layer than 'main' not valid for key type {key_type}. Changed layer to 'main'.")
                            viewer_config.layer_name = "main"

                    if viewer_config.layer_name == "main":
                        new_layer_name = f"{viewer_config.data_name}-{key}"
                    else:
                        new_layer_name = f"{viewer_config.data_name}-{key} [{viewer_config.layer_name}]"

                    # get layer names from the current data
                    if viewer_config.layer_name == "main":
                        layer_names_for_current_data = [elem.name for elem in viewer.layers if elem.name.startswith(viewer_config.data_name) and not elem.name.endswith(f"[{viewer_config.layer_name}]")]
                    else:
                        layer_names_for_current_data = [elem.name for elem in viewer.layers if elem.name.startswith(viewer_config.data_name) and elem.name.endswith(f"[{viewer_config.layer_name}]")]

                    # select only point layers
                    layer_names_for_current_data = [elem for elem in layer_names_for_current_data if isinstance(viewer.layers[elem], napari.layers.points.points.Points)]

                    # save last addition to add it to recent in the callback
                    viewer_config.recent_selections.append(f"{key_type}:{key}")

                    if f"{key}_colors" in viewer_config.adata.uns.keys():
                        # Convert hex colors to RGB format
                        def hex_to_rgb(hex_color):
                            hex_color = hex_color.lstrip('#')
                            return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                        rgb_colors = [hex_to_rgb(color) for color in viewer_config.adata.uns[f"{key}_colors"]]

                        # Transform to ListedColormap
                        colormap = ListedColormap(rgb_colors)
                    else:
                        colormap = None

                    if len(layer_names_for_current_data) == 0:

                        # create points layer for genes
                        gene_layer = _create_points_layer(
                            points=viewer_config.points,
                            color_values=color_value,
                            #name=f"{config.current_data_name}-{gene}",
                            name=new_layer_name,
                            point_names=cell_names,
                            point_size=size,
                            upper_climit_pct=99,
                            categorical_cmap = colormap
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
                                categorical_cmap = colormap
                            )
                            # move new layer to the top
                            was_moved = viewer.layers.move(viewer.layers.index(new_layer_name), len(viewer.layers))

                        else:
                            # create new points layer for genes
                            gene_layer = _create_points_layer(
                                points=viewer_config.points,
                                color_values=color_value,
                                #name=f"{config.current_data_name}-{gene}",
                                name=new_layer_name,
                                point_names=cell_names,
                                point_size=size,
                                upper_climit_pct=99,
                                categorical_cmap = colormap
                            )
                            return gene_layer

            if len(viewer_config.key_dict["obs"]) > 0:
                obs_choices = viewer_config.key_dict["obs"]
            else:
                obs_choices = ["No filtering options available"]

            @magicgui(
                call_button='Filter',
                obs_key={'choices': obs_choices, 'label': "Obs:"},
                operation_type={'choices': ["contains", "is equal to", "is not", "is in"],
                                'label': 'Operation:'},
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
                        mask = viewer_config.adata.obs[obs_key].str.contains(obs_value)
                    elif operation_type == "is equal to":
                        mask = viewer_config.adata.obs[obs_key].astype(str) == str(obs_value)
                    elif operation_type == "is not":
                        mask = viewer_config.adata.obs[obs_key].astype(str) != str(obs_value)
                    elif operation_type == "is in":
                        obs_value_list = [elem.strip().strip("'").strip('"') for elem in obs_value.split(",")]
                        mask = viewer_config.adata.obs[obs_key].isin(obs_value_list)
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
                if cell in viewer_config.adata.obs_names.astype(str):
                    # get location of selected cell
                    cell_loc = viewer_config.adata.obs_names.get_loc(cell)
                    cell_position = viewer_config.points[cell_loc]

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
                                border_color='red',
                                border_width=0.1
                            )
                else:
                    print(f"Cell '{cell}' not found.")

            # ---CALLBACKS---
            # connect key change with update function
            @select_data_widget.data_name.changed.connect
            @select_data_widget.layer_name.changed.connect
            @show_cells_widget.key_type.changed.connect
            @show_cells_widget.call_button.changed.connect
            @viewer.layers.events.removed.connect
            @viewer.layers.events.inserted.connect
            def update_widgets_on_data_change(event=None):
                # update data name in config and refresh the variables in the config class
                viewer_config.data_name = select_data_widget.data_name.value
                viewer_config.layer_name = select_data_widget.layer_name.value

                _refresh_widgets_after_data_change(
                    data,
                    viewer=viewer,
                    viewer_config=viewer_config,
                    select_data_widget=select_data_widget,
                    show_cells_widget=show_cells_widget,
                    boundaries_widget=show_boundaries_widget,
                    filter_widget=filter_cells_widget
                    )

            def callback_refresh(event=None):
                # after the points widget is run, the widgets have to be refreshed to current data layer
                _refresh_widgets_after_data_change(
                    data,
                    viewer=viewer,
                    viewer_config=viewer_config,
                    select_data_widget=select_data_widget,
                    show_cells_widget=show_cells_widget,
                    boundaries_widget=show_boundaries_widget,
                    filter_widget=filter_cells_widget
                    )

            def callback_update_legend(event=None):
                _update_colorlegend(viewer=viewer, viewer_config=viewer_config)

            if show_cells_widget is not None:
                show_cells_widget.call_button.clicked.connect(callback_refresh)
                show_cells_widget.call_button.clicked.connect(callback_update_legend)
            if show_boundaries_widget is not None:
                show_boundaries_widget.call_button.clicked.connect(callback_refresh)
                show_boundaries_widget.call_button.clicked.connect(callback_update_legend)

            viewer.layers.selection.events.active.connect(callback_update_legend)

        if data.annotations.is_empty and data.regions.is_empty:
            show_geometries_widget = None
        else:
            #TODO: The following section is weirdly complicated and should be simplified.
            # check which geometries are available
            if not data.annotations.is_empty:
                if not data.regions.is_empty:
                    choices = ["Annotations", "Regions"]
                else:
                    choices = ["Annotations"]
            else:
                choices = ["Regions"]

            for c in choices:
                if len(getattr(data, c.lower()).metadata.keys()) == 0:
                    choices.remove(c)

            if len(choices) == 0:
                show_geometries_widget = None
            else:

                # extract geometry object
                geom = getattr(data, choices[0].lower())

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
                        edge_width={'min': 1, 'max': 40, 'step': 1, 'label': 'Edge width:'},
                        # opacity={'min': 0.0, 'max': 1.0, 'step': 0.1, 'label': 'Opacity:'},
                        # tolerance={'min': 0, 'step': 1, 'label': 'Tolerance:'},
                        show_names={'label': 'Show names'}
                    )
                    def show_geometries_widget(
                        geom_type,
                        key,
                        annot_class,
                        edge_width: int = 4,
                        # opacity: float = 1,
                        # tolerance: int = 1,
                        show_names: bool = False
                        ):
                        opacity = 1
                        tolerance = 1

                        if geom_type == "Annotations":
                            # get annotation dataframe
                            annot_df = data.annotations[key]
                            all_keys = list(data.annotations.metadata.keys())
                        elif geom_type == "Regions":
                            # get regions dataframe
                            annot_df = data.regions[key]
                            all_keys = list(data.regions.metadata.keys())
                        else:
                            TypeError(f"Unknown geometry type: {geom_type}")

                        if annot_class == "all":
                            # get classes
                            classes = annot_df['name'].unique()
                        else:
                            classes = [annot_class]

                        # iterate through classes
                        for cl in classes:
                            layer_name = f"{cl} ({key})"
                            #if layer_name not in viewer.layers: # this cannot be checked here because the symbol is missing which is added in the adding process below
                            # get dataframe for this class
                            class_df = annot_df[annot_df["name"] == cl].copy()

                            # simplify polygons for visualization
                            # class_df["geometry"] = class_df["geometry"].simplify(tolerance)

                            if not "color" in class_df.columns:
                                # create a RGB color with range 0-255 for this key
                                rgb_color = [elem * 255 for elem in REGION_CMAP(all_keys.index(key))][:3]
                            else:
                                rgb_color = None

                            # add layer to viewer
                            _add_geometries_as_layer(
                                dataframe=class_df,
                                viewer=viewer,
                                layer_name=layer_name,
                                #scale_factor=scale_factor,
                                edge_width=edge_width,
                                opacity=opacity,
                                rgb_color=rgb_color,
                                show_names=show_names,
                                mode=geom_type,
                                tolerance=tolerance
                            )

                    # connect key change with update function
                    @show_geometries_widget.geom_type.changed.connect
                    @show_geometries_widget.key.changed.connect
                    @show_geometries_widget.call_button.clicked.connect
                    @viewer.layers.events.removed.connect # somehow the values change when layers are inserted
                    @viewer.layers.events.inserted.connect # or removed. Therefore, this update is necessary
                    def update_annotation_widget_after_changes(event=None):
                        _update_keys_based_on_geom_type(show_geometries_widget, xdata=data)
                        _update_classes_on_key_change(show_geometries_widget, xdata=data)
                        _set_show_names_based_on_geom_type(show_geometries_widget)
                        #_update_key_on_type_change(show_cells_widget, viewer_config=viewer_config)

        return (
            show_cells_widget,
            move_to_cell_widget,
            show_geometries_widget,
            show_boundaries_widget,
            select_data_widget,
            filter_cells_widget,
            #add_new_geometries_widget
            )

    # Difference between magicgui and magic_factory decorators:
    # - the magicgui decorator directly returns the widget
    # - the magic_factory decorator returns a factory function that can be called to generate the widget
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

        # get current viewer and config
        viewer = napari.current_viewer()
        viewer_config = config_manager[_get_viewer_uid(viewer)]

        if (class_name != "") & (annot_key != ""):
            if key == "Geometric annotations":
                # print(annot_key)
                # print(class_name)
                # print(viewer_config)
                _test_existance(viewer_config, annot_key, class_name, modality="annotations")
                # generate name
                name = name_pattern.format(
                    type_symbol=ANNOTATIONS_SYMBOL,
                    class_name=class_name,
                    annot_key=annot_key
                    )

                # generate shapes layer for geometric annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'shape_type': 'polygon',
                        'edge_width': 4,
                        'edge_color': 'red',
                        'face_color': 'transparent',
                        #'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object'),
                            'type': np.array([], dtype='object')
                            }
                        },
                    'shapes'
                    )
            elif key == "Point annotations":
                _test_existance(viewer_config, annot_key, class_name, modality="annotations")
                # generate name
                name = name_pattern.format(
                    type_symbol=POINTS_SYMBOL,
                    class_name=class_name,
                    annot_key=annot_key
                    )

                # generate points layer for point annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'size': 10,
                        'edge_color': 'black',
                        'face_color': 'blue',
                        #'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object'),
                            'type': np.array([], dtype='object')
                        }
                        },
                    'points'
                    )

            elif key == "Regions":
                _test_existance(viewer_config, annot_key, class_name, modality="regions")
                # generate name
                name = name_pattern.format(
                    type_symbol=REGIONS_SYMBOL,
                    class_name=class_name,
                    annot_key=annot_key
                    )

                # generate shapes layer for region
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
                            'uid': np.array([], dtype='object'),
                            'type': np.array([], dtype='object')
                        }
                        },
                    'shapes'
                    )

            else:
                layer = None

            if name in viewer.layers:
                return None

            # reset class name to nothing
            add_new_geometries_widget.class_name.value = ""

            return layer

        else:
            show_warning("Please provide a class name and an annotation key.")
            return None


    def _test_existance(viewer_config, annot_key, class_name, modality):
        try:
            geom_df = getattr(viewer_config.data, modality)

            if geom_df is not None:
                exists = (geom_df[annot_key]["name"] == class_name).any()
            else:
                exists = False
        except KeyError:
            exists = False

        if exists:
            show_warning((
                    f"Data contains already {modality} with key '{annot_key}' and class '{class_name}'. "
                    f"To show them use the 'Show geometries' widget."
                    ))

    class SyncButton(QWidget):
        def __init__(self):
            super().__init__()
            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

            # create the sync button
            self.sync_button = QPushButton("Sync Geometries")
            self.sync_button.clicked.connect(self._sync_geometries)
            self.layout.addWidget(self.sync_button)

        def _sync_geometries(self):
            sync_geometries()



    class SaveWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

            self.path_layout = QHBoxLayout()

            self.label = QLabel("No folder selected")
            self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.label.setMinimumWidth(150)
            self.label.setMaximumWidth(200)
            self.label.setToolTip("No folder selected")
            self.path_layout.addWidget(self.label)

            self.select_button = QPushButton()
            self.select_button.setText("Select")
            self.select_button.setIconSize(QSize(16, 16))
            self.select_button.setToolTip("Select Output Folder")
            self.select_button.clicked.connect(self.select_folder)
            self.path_layout.addWidget(self.select_button)

            self.layout.addLayout(self.path_layout)

            self.save_button = QPushButton("Save")
            self.save_button.clicked.connect(self.save_data)
            self.layout.addWidget(self.save_button)

            self.output_folder = None

        def select_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if folder:
                self.output_folder = folder
                self.update_label(folder)

        def update_label(self, text):
            metrics = QFontMetrics(self.label.font())
            elided_text = metrics.elidedText(text, Qt.ElideMiddle, self.label.width())
            self.label.setText(elided_text)
            self.label.setToolTip(text)

        def save_data(self):
            if self.output_folder:
                save_colorlegends(output_folder=self.output_folder)
            else:
                self.label.setText("Please select a folder first.")


    # class SaveWidget(QWidget):
    #     def __init__(self):
    #         super().__init__()
    #         self.layout = QVBoxLayout()
    #         self.setLayout(self.layout)

    #         self.label = QLabel("No folder selected")
    #         self.layout.addWidget(self.label)

    #         self.select_button = QPushButton("Select Output Folder")
    #         self.select_button.clicked.connect(self.select_folder)
    #         self.layout.addWidget(self.select_button)

    #         self.save_button = QPushButton("Save")
    #         self.save_button.clicked.connect(self.save_data)
    #         self.layout.addWidget(self.save_button)

    #         self.output_folder = None

    #     def select_folder(self):
    #         folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
    #         if folder:
    #             self.output_folder = folder
    #             # Truncate the path if it's too long
    #             max_length = 40  # Adjust as needed
    #             if len(folder) > max_length:
    #                 truncated = "..." + folder[-(max_length - 3):]
    #             else:
    #                 truncated = folder
    #             self.label.setText(f"{truncated}")

    #     def save_data(self):
    #         if self.output_folder:
    #             # Replace this with your actual saving function
    #             save_colorlegends(output_folder=self.output_folder)
    #         else:
    #             self.label.setText("Please select a folder first.")

import logging

from insitupy._constants import WITH_NAPARI

logger = logging.getLogger(__name__)

if WITH_NAPARI:
    from typing import List, Optional

    import matplotlib.pyplot as plt
    import napari
    import numpy as np
    import pandas as pd
    from magicgui import magic_factory, magicgui
    from magicgui.widgets import FunctionGui
    from matplotlib.colors import ListedColormap
    from napari.utils import DirectLabelColormap
    from napari.utils.notifications import show_info, show_warning
    from qtpy.QtCore import QSize, Qt
    from qtpy.QtGui import QFontMetrics, QIcon
    from qtpy.QtWidgets import (QComboBox, QCompleter, QFileDialog,
                                QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                                QPushButton, QVBoxLayout, QWidget)
    from scipy.sparse import issparse

    from insitupy._constants import (ANNOTATIONS_SYMBOL,
                                     DEFAULT_CATEGORICAL_CMAP, POINTS_SYMBOL,
                                     REGION_CMAP, REGIONS_SYMBOL)
    from insitupy.images.utils import create_img_pyramid
    from insitupy.interactive._callbacks import (
        _refresh_widgets_after_data_change, _update_colorlegend,
        _update_key_on_type_change)
    from insitupy.interactive._configs import (ViewerConfig, _get_viewer_uid,
                                               config_manager)
    from insitupy.interactive._layers import (_create_points_layer,
                                              _create_units_layer,
                                              _update_points_layer,
                                              _update_units_layer)
    from insitupy.interactive.viewer import save_colorlegends, sync_geometries
    from insitupy.utils._helpers import _get_expression_values

    from ._layers import (_add_geometries_as_layer, _apply_colors_from_features,
                          _connect_color_propagation, _get_or_assign_color)
    from insitupy.palettes import ANNOTATIONS_PALETTE, REGIONS_PALETTE

    # Maximum number of unique colors for labels (napari limitation)
    MAX_LABEL_COLORS = 500

    def _as_positional_array(values) -> np.ndarray:
        """Convert array-like input to a NumPy array for position-based indexing."""
        if isinstance(values, np.ndarray):
            return values
        return np.asarray(values)

    def _is_missing_label_value(value) -> bool:
        """Return True for missing values used in cell coloring."""
        try:
            return bool(pd.isna(value))
        except TypeError:
            return False

    def _unique_non_missing_categories(values: np.ndarray) -> list:
        """Collect categorical values without sorting mixed Python types."""
        valid_values = [value for value in values if not _is_missing_label_value(value)]
        if len(valid_values) == 0:
            return []
        return list(pd.unique(np.asarray(valid_values, dtype=object)))

    def _has_non_missing_label_values(values: np.ndarray) -> bool:
        """Return True if at least one value is present for coloring."""
        return any(not _is_missing_label_value(value) for value in values)

    def _build_labels_properties(
        viewer_config: "ViewerConfig",
        label_ids: np.ndarray,
        cell_names_boundary: np.ndarray,
        mask_key: str,
        key: Optional[str] = None,
        color_values: Optional[np.ndarray] = None,
    ) -> dict:
        """Build per-label properties used by napari status and tooltips."""
        boundaries = viewer_config.boundaries
        prop_names = ["cell_area", "surface_area"]
        cell_names_boundary = _as_positional_array(cell_names_boundary)

        if color_values is not None:
            color_values = _as_positional_array(color_values)

        if mask_key == "nuclei" and boundaries.nucleus_to_cell_map is not None:
            nucleus_to_cell_map = boundaries.nucleus_to_cell_map
            cell_indices = [nucleus_to_cell_map.get(label_id - 1, None) for label_id in label_ids]
        else:
            cell_indices = [i for i in range(len(label_ids))]

        names = [
            cell_names_boundary[cell_idx]
            if cell_idx is not None and cell_idx < len(cell_names_boundary)
            else "unmapped"
            for cell_idx in cell_indices
        ]

        properties = {
            'index': label_ids,
            'name': names,
        }

        for prop_name in prop_names:
            if prop_name in viewer_config.adata.obs.columns:
                prop_values = viewer_config.adata.obs[prop_name].values
                properties[prop_name] = [
                    prop_values[cell_idx] if cell_idx is not None and cell_idx < len(prop_values) else None
                    for cell_idx in cell_indices
                ]

        if key is not None and color_values is not None:
            value_list = [
                color_values[cell_idx] if cell_idx is not None and cell_idx < len(color_values) and not _is_missing_label_value(color_values[cell_idx]) else None
                for cell_idx in cell_indices
            ]
            properties['value'] = value_list
            properties[key] = value_list

        return properties

    def _create_outline_colormap(
        label_ids: np.ndarray,
        hidden_label_ids: Optional[set[int]] = None,
    ) -> DirectLabelColormap:
        """Create a direct colormap that renders every non-background label in black."""
        if hidden_label_ids is None:
            hidden_label_ids = set()

        color_dict = {
            None: np.array([0.0, 0.0, 0.0, 1.0]),
            0: np.array([0.0, 0.0, 0.0, 0.0]),
        }
        color_dict.update({
            int(label_id): np.array([0.0, 0.0, 0.0, 0.0]) if int(label_id) in hidden_label_ids else np.array([0.0, 0.0, 0.0, 1.0])
            for label_id in label_ids
        })
        return DirectLabelColormap(color_dict=color_dict)

    def _create_colored_labels_layer(
        viewer: napari.Viewer,
        viewer_config: "ViewerConfig",
        color_values: np.ndarray,
        layer_name: str,
        mask_key: str,
        key: str,
        colormap: Optional[ListedColormap] = None,
        add_new_layer: bool = False,
    ) -> None:
        """Create or update a labels layer with colors based on expression values.

        Args:
            viewer: The napari viewer instance.
            viewer_config: ViewerConfig with boundaries data.
            color_values: Array of values to color by (one per cell).
            layer_name: Name for the layer.
            mask_key: Key for the mask ("cells" or "nuclei").
            key: The key/feature name being displayed.
            colormap: Optional categorical colormap from adata.uns.
            add_new_layer: If True, always create a new layer.

        Returns:
            None (adds layer directly to viewer).
        """
        boundaries = viewer_config.boundaries
        if boundaries is None:
            show_warning("No boundaries data available.")
            return None

        # Get the mask data
        try:
            mask = boundaries[mask_key]
        except KeyError:
            show_warning(f"Key '{mask_key}' not found in boundaries masks.")
            return None

        if mask is None:
            show_warning(f"No mask data available for key '{mask_key}'.")
            return None

        # Get metadata for mask
        metadata = boundaries.metadata
        pixel_size = metadata[mask_key]["pixel_size"]

        # Generate pyramid if needed
        if not isinstance(mask, list):
            mask_pyramid = create_img_pyramid(img=mask, axes="YX", nsubres=6)
        else:
            mask_pyramid = mask

        # Get label IDs and cell names
        label_ids = boundaries.seg_mask_value.compute()
        cell_names_boundary = boundaries.cell_names.compute()
        color_values = _as_positional_array(color_values)

        if not _has_non_missing_label_values(color_values):
            show_warning(f"All values for '{key}' are missing. No labels layer was added.")
            return None

        # Handle nuclei mapping if needed
        if mask_key == "nuclei" and boundaries.nucleus_to_cell_map is not None:
            nucleus_to_cell_map = boundaries.nucleus_to_cell_map
            # Map nucleus label_ids to cell indices for color lookup
            cell_indices = [nucleus_to_cell_map.get(label_id - 1, None) for label_id in label_ids]
        else:
            # Direct mapping: label_id corresponds to cell index
            cell_indices = [i for i in range(len(label_ids))]

        # Determine if values are categorical or continuous
        is_categorical = (
            colormap is not None or
            (hasattr(color_values, 'dtype') and color_values.dtype == object) or
            isinstance(color_values[0], str) if len(color_values) > 0 else False
        )

        # Build color dictionary for DirectLabelColormap
        color_dict = {
            None: np.array([0.5, 0.5, 0.5, 0.5]),  # Default for unmapped labels
            0: np.array([0.0, 0.0, 0.0, 0.0]),      # Background transparent
        }
        hidden_label_ids = set()

        if is_categorical:
            # Categorical coloring
            if colormap is not None:
                # Use provided colormap
                unique_categories = _unique_non_missing_categories(color_values)
                cat_to_idx = {cat: i for i, cat in enumerate(unique_categories)}

                for label_id, cell_idx in zip(label_ids, cell_indices):
                    if cell_idx is not None and cell_idx < len(color_values):
                        cat = color_values[cell_idx]
                        if _is_missing_label_value(cat):
                            hidden_label_ids.add(int(label_id))
                            color_dict[int(label_id)] = np.array([0.0, 0.0, 0.0, 0.0])
                            continue
                        cat_idx = cat_to_idx.get(cat, 0)
                        # Limit color index to avoid exceeding colormap
                        color_idx = cat_idx % min(len(colormap.colors), MAX_LABEL_COLORS)
                        color = np.array(colormap.colors[color_idx])
                        if len(color) == 3:
                            color = np.append(color, 1.0)  # Add alpha
                        color_dict[int(label_id)] = color
            else:
                # Use the same default categorical colormap as points mode.
                unique_categories = _unique_non_missing_categories(color_values)
                cat_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
                cmap_mpl = DEFAULT_CATEGORICAL_CMAP
                n_colors = cmap_mpl.N if hasattr(cmap_mpl, 'N') else 20

                for label_id, cell_idx in zip(label_ids, cell_indices):
                    if cell_idx is not None and cell_idx < len(color_values):
                        cat = color_values[cell_idx]
                        if _is_missing_label_value(cat):
                            hidden_label_ids.add(int(label_id))
                            color_dict[int(label_id)] = np.array([0.0, 0.0, 0.0, 0.0])
                            continue
                        cat_idx = cat_to_idx.get(cat, 0)
                        color_idx = cat_idx % n_colors
                        norm_idx = color_idx / (n_colors - 1) if n_colors > 1 else 0
                        color_dict[int(label_id)] = np.array(cmap_mpl(norm_idx))
        else:
            # Continuous coloring
            # Normalize values to 0-1
            values = np.array(color_values, dtype=float)
            # Use percentile for robust normalization
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmax == vmin:
                vmax = vmin + 1  # Avoid division by zero

            cmap_mpl = plt.cm.viridis

            for label_id, cell_idx in zip(label_ids, cell_indices):
                if cell_idx is not None and cell_idx < len(color_values):
                    value = color_values[cell_idx]
                    if _is_missing_label_value(value):
                        hidden_label_ids.add(int(label_id))
                        color_dict[int(label_id)] = np.array([0.0, 0.0, 0.0, 0.0])
                    else:
                        norm_val = np.clip((value - vmin) / (vmax - vmin), 0, 1)
                        color_dict[int(label_id)] = np.array(cmap_mpl(norm_val))

        # Create DirectLabelColormap
        direct_cmap = DirectLabelColormap(color_dict=color_dict)

        # Determine layer name with mask type suffix
        full_layer_name = f"{layer_name} ({mask_key})"
        outline_layer_name = f"{full_layer_name} outline"

        # Build properties
        properties = _build_labels_properties(
            viewer_config=viewer_config,
            label_ids=label_ids,
            cell_names_boundary=cell_names_boundary,
            mask_key=mask_key,
            key=key,
            color_values=color_values,
        )
        outline_cmap = _create_outline_colormap(label_ids, hidden_label_ids=hidden_label_ids)
        legend_metadata = {
            "legend_key": key,
            "legend_is_categorical": is_categorical,
            "legend_continuous_cmap": "viridis",
            "legend_upper_climit_pct": 99,
        }
        outline_metadata = {
            "legend_source_layer": full_layer_name,
            "is_outline_layer": True,
        }

        # Check if layer exists and handle accordingly
        if full_layer_name in viewer.layers and not add_new_layer:
            # Update existing layer's colormap
            layer = viewer.layers[full_layer_name]
            layer.colormap = direct_cmap
            layer.properties = properties
            layer.metadata.update(legend_metadata)
            # Move to top
            viewer.layers.move(viewer.layers.index(full_layer_name), len(viewer.layers))
        else:
            if full_layer_name in viewer.layers:
                show_warning(f"Layer '{full_layer_name}' already exists. Uncheck 'Add new layer' to update it instead.")
                return None

            # Add new labels layer
            viewer.add_labels(
                mask_pyramid,
                name=full_layer_name,
                scale=(pixel_size, pixel_size),
                properties=properties,
                metadata=legend_metadata,
                colormap=direct_cmap,
            )

        if outline_layer_name in viewer.layers and not add_new_layer:
            outline_layer = viewer.layers[outline_layer_name]
            outline_layer.colormap = outline_cmap
            outline_layer.properties = properties
            outline_layer.metadata.update(outline_metadata)
            outline_layer.contour = 1
            viewer.layers.move(viewer.layers.index(outline_layer_name), len(viewer.layers))
        else:
            if outline_layer_name in viewer.layers:
                show_warning(f"Layer '{outline_layer_name}' already exists. Uncheck 'Add new layer' to update it instead.")
                return None

            outline_layer = viewer.add_labels(
                mask_pyramid,
                name=outline_layer_name,
                scale=(pixel_size, pixel_size),
                properties=properties,
                metadata=outline_metadata,
                colormap=outline_cmap,
                opacity=1.0,
            )
            outline_layer.contour = 1

        return None

    def _initialize_widgets(
        viewer: napari.Viewer,
        viewer_config: ViewerConfig
        #xdata # InSituData object
        ) -> List[FunctionGui]:

        # access viewer from InSituData
        #viewer = xdata.viewer
        data = viewer_config.data

        show_cells_widget = None
        move_to_cell_widget = None
        show_geometries_widget = None
        show_boundaries_widget = None
        select_data_widget = None
        filter_cells_widget = None
        show_units_widget = None

        def callback_update_legend(event=None):
            _update_colorlegend(viewer=viewer, viewer_config=viewer_config)

        viewer.layers.selection.events.active.connect(callback_update_legend)

        # else:
        if viewer_config.has_cells:
            data_names = data.cells.keys()
            layer_names = ["main"] + list(data.cells.table.layers)

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
                        try:
                            mask = viewer_config.boundaries[key]
                        except KeyError:
                            show_warning(f"Key '{key}' not found in boundaries masks.")
                            return

                        if mask is None:
                            show_warning(f"No mask data available for key '{key}'.")
                            return

                        # get metadata for mask
                        metadata = viewer_config.boundaries.metadata
                        pixel_size = metadata[key]["pixel_size"]

                        if not isinstance(mask, list):
                            # generate pyramid of the mask - segmentation masks are 2D (YX)
                            mask_pyramid = create_img_pyramid(img=mask, axes="YX", nsubres=6)
                        else:
                            mask_pyramid = mask

                        # Create properties DataFrame with label IDs as index
                        label_ids = viewer_config.boundaries.seg_mask_value.compute()
                        cell_names = viewer_config.boundaries.cell_names.compute()

                        if key not in {"cells", "nuclei"}:
                            show_warning(f"Unknown key for boundaries: {key}.")
                            return

                        properties = _build_labels_properties(
                            viewer_config=viewer_config,
                            label_ids=label_ids,
                            cell_names_boundary=cell_names,
                            mask_key=key,
                        )

                        # Add masks as labels to napari viewer
                        layer = viewer.add_labels(
                            mask_pyramid,
                            name=layer_name,
                            scale=(pixel_size, pixel_size),
                            properties=properties,
                        )

                        if key == "cells":
                            viewer.layers[layer_name].contour = 1
                    else:
                        logger.info(f"Layer '{layer_name}' already in layer list.")
            else:
                show_boundaries_widget = None

            # Determine available display modes based on boundaries data
            _display_mode_choices = ["points"]
            if viewer_config.boundaries is not None:
                if "cells" in viewer_config.boundaries._data and viewer_config.boundaries["cells"] is not None:
                    _display_mode_choices.append("cells")
                if "nuclei" in viewer_config.boundaries._data and viewer_config.boundaries["nuclei"] is not None:
                    _display_mode_choices.append("nuclei")

            @magicgui(
                call_button='Show',
                key_type={'choices': ["genes", "obs", "obsm"], 'label': 'Type:'},
                key={'choices': viewer_config.genes, 'label': "Key:"},
                display_mode={'choices': _display_mode_choices, 'label': 'Display as:'},
                size={'label': 'Size [µm]'},
                recent={'choices': [""], 'label': "Recent:"},
                add_new_layer={'label': 'Add new layer'}
                )
            def show_cells_widget(
                key_type="genes",
                key=None,
                display_mode="points",
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

                    # Handle display as labels (cells or nuclei boundaries)
                    if display_mode in ["cells", "nuclei"]:
                        return _create_colored_labels_layer(
                            viewer=viewer,
                            viewer_config=viewer_config,
                            color_values=color_value,
                            layer_name=new_layer_name,
                            mask_key=display_mode,
                            key=key,
                            colormap=colormap,
                            add_new_layer=add_new_layer,
                        )

                    # Display as points (default behavior)
                    # get layer names from the current data
                    if viewer_config.layer_name == "main":
                        layer_names_for_current_data = [elem.name for elem in viewer.layers if elem.name.startswith(viewer_config.data_name) and not elem.name.endswith(f"[{viewer_config.layer_name}]")]
                    else:
                        layer_names_for_current_data = [elem.name for elem in viewer.layers if elem.name.startswith(viewer_config.data_name) and elem.name.endswith(f"[{viewer_config.layer_name}]")]

                    # select only point layers
                    layer_names_for_current_data = [elem for elem in layer_names_for_current_data if isinstance(viewer.layers[elem], napari.layers.points.points.Points)]

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
                            layer = viewer.layers[layer_names_for_current_data[-1]]
                            _update_points_layer(
                                layer=layer,
                                new_color_values=color_value,
                                new_name=new_layer_name,
                                categorical_cmap = colormap
                            )
                            # move new layer to the top
                            was_moved = viewer.layers.move(viewer.layers.index(new_layer_name), len(viewer.layers))

                        else:
                            # Check if layer with this name already exists
                            if new_layer_name in viewer.layers:
                                show_warning(f"Layer '{new_layer_name}' already exists. Uncheck 'Add new layer' to update it instead.")
                                return None

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
                    logger.warning(f"Cell '{cell}' not found.")

            # ---CALLBACKS---
            # connect key change with update function
            @select_data_widget.data_name.changed.connect
            @select_data_widget.layer_name.changed.connect
            def update_widgets_on_data_change(event=None):
                # update data name in config and refresh the variables in the config class
                viewer_config.data_name = select_data_widget.data_name.value
                viewer_config.layer_name = select_data_widget.layer_name.value
                viewer_config.refresh_variables()

                _refresh_widgets_after_data_change(
                    data,
                    viewer=viewer,
                    viewer_config=viewer_config,
                    select_data_widget=select_data_widget,
                    show_cells_widget=show_cells_widget,
                    boundaries_widget=show_boundaries_widget,
                    filter_widget=filter_cells_widget
                    )

            @show_cells_widget.key_type.changed.connect
            def update_show_cells_key_choices(event=None):
                show_cells_widget.key.value = None
                _update_key_on_type_change(show_cells_widget, viewer_config=viewer_config)

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

            if show_cells_widget is not None:
                show_cells_widget.call_button.clicked.connect(callback_refresh)
                show_cells_widget.call_button.clicked.connect(callback_update_legend)
            if show_boundaries_widget is not None:
                show_boundaries_widget.call_button.clicked.connect(callback_refresh)
                show_boundaries_widget.call_button.clicked.connect(callback_update_legend)


        # ====== SPATIAL UNITS WIDGET ======
        # if not viewer_config.has_units:
        #     show_units_widget = None
        # else:
        if viewer_config.has_units:
            # Prepare choices
            obs_choices = [""] + sorted(list(viewer_config.unit_obs))
            obsm_choices = [""] + sorted(list(viewer_config.unit_obsm))

            @magicgui(
                call_button='Show',
                gene={'label': "Gene (search):"},
                obs={'choices': obs_choices, 'label': 'Obs:'},
                obsm={'choices': obsm_choices, 'label': 'Obsm:'},
                add_new_layer={'label': 'Add new layer'}
            )
            def show_units_widget(
                gene="",
                obs="",
                obsm="",
                add_new_layer=False
            ) -> napari.types.LayerDataTuple:

                key = None
                key_type = None

                # Determine which key to use (priority: gene > obs > obsm)
                if gene != "":
                    key = gene
                    key_type = "genes"
                elif obs != "":
                    key = obs
                    key_type = "obs"
                elif obsm != "":
                    key = obsm
                    key_type = "obsm"

                if key is None or key == "":
                    show_warning("Please select a key to visualize.")
                    return None

                # Validate key for genes
                if key_type == "genes":
                    if key not in viewer_config.unit_vars:
                        show_warning(f"Key '{key}' not found in genes.")
                        return None

                # get expression values
                color_values = _get_expression_values(
                    adata=viewer_config.units.data,
                    X=viewer_config.units.data.X,
                    key_type=key_type, key=key
                )

                # Create layer name
                layer_name = f"units-{key}"

                # Get existing spatial unit layers
                unit_layer_names = [elem.name for elem in viewer.layers if elem.name.startswith("units-") and isinstance(elem, napari.layers.shapes.shapes.Shapes)]

                if len(unit_layer_names) == 0:
                    # Create new spatial units layer
                    unit_layer = _create_units_layer(
                        gdf=viewer_config.units.shapes,
                        color_values=color_values,
                        name=layer_name,
                        unit_names=viewer_config.units.shapes.index.tolist(),
                        edge_width=0,
                        opacity=0.5,
                        upper_climit_pct=99
                    )
                    return unit_layer
                else:
                    if not add_new_layer:
                        # Update the existing spatial units layer
                        layer = viewer.layers[unit_layer_names[-1]]
                        _update_units_layer(
                            layer=layer,
                            new_color_values=color_values,
                            new_name=layer_name
                        )
                        # Move layer to the top
                        viewer.layers.move(viewer.layers.index(layer_name), len(viewer.layers))
                    else:
                        # Check if layer with this name already exists
                        if layer_name in viewer.layers:
                            show_warning(f"Layer '{layer_name}' already exists. Uncheck 'Add new layer' to update it instead.")
                            return None

                        # Create new units layer
                        unit_layer = _create_units_layer(
                            gdf=viewer_config.units.shapes,
                            color_values=color_values,
                            name=layer_name,
                            unit_names=viewer_config.units.shapes.index.tolist(),
                            edge_width=0,
                            opacity=0.5,
                            upper_climit_pct=99
                        )
                        return unit_layer

            # Make the gene text field searchable with completer
            def _setup_searchable_textfield(widget, full_choices):
                """Configure QLineEdit to be searchable with QCompleter."""
                if hasattr(widget, 'native') and isinstance(widget.native, QLineEdit):
                    # Create completer with full list for efficient searching
                    completer = QCompleter(full_choices)
                    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
                    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                    widget.native.setCompleter(completer)

            # Setup searchable text field for genes
            _setup_searchable_textfield(show_units_widget.gene, viewer_config.unit_vars)

            # Connect callbacks to ensure mutual exclusivity
            @show_units_widget.gene.changed.connect
            def _on_gene_changed(event=None):
                if show_units_widget.gene.value != "":
                    show_units_widget.obs.value = ""
                    show_units_widget.obsm.value = ""

            @show_units_widget.obs.changed.connect
            def _on_obs_changed(event=None):
                if show_units_widget.obs.value != "":
                    show_units_widget.gene.value = ""
                    show_units_widget.obsm.value = ""

            @show_units_widget.obsm.changed.connect
            def _on_obsm_changed(event=None):
                if show_units_widget.obsm.value != "":
                    show_units_widget.gene.value = ""
                    show_units_widget.obs.value = ""

            show_units_widget.call_button.clicked.connect(callback_update_legend)

        geometries_widget = GeometriesWidget(viewer=viewer, viewer_config=viewer_config)

        return (
            show_cells_widget,
            move_to_cell_widget,
            geometries_widget,
            show_boundaries_widget,
            select_data_widget,
            filter_cells_widget,
            show_units_widget,
            )

    _ALL = "(all)"

    # NOTE: must remain inside the `if WITH_NAPARI:` block — napari types are
    # referenced in the signature and body.
    def _get_next_region_name_for_layer(viewer: napari.Viewer, key: str, viewer_config) -> str:
        """Return the next auto-incremented region name for *key* across viewer + InSituData."""
        import re
        existing: set = set()
        layer_name = f"{REGIONS_SYMBOL} {key}"
        if layer_name in viewer.layers:
            layer = viewer.layers[layer_name]
            for n in layer.features.get('name', []):
                if isinstance(n, str):
                    existing.add(n)
        data = viewer_config.data
        if not data.regions.is_empty and key in data.regions.keys():
            df = data.regions[key]
            if 'name' in df.columns:
                for n in df['name'].dropna():
                    if isinstance(n, str):
                        existing.add(n)
        max_n = 0
        for n in existing:
            m = re.match(r'^Region (\d+)$', n)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"Region {max_n + 1}"

    class GeometriesWidget(QWidget):
        """Unified widget for showing and adding geometry layers (annotations/regions)."""

        def __init__(self, viewer: napari.Viewer, viewer_config: "ViewerConfig",
                     max_width: int = 500):
            super().__init__()
            self.viewer = viewer
            self.viewer_config = viewer_config
            self.setMaximumWidth(max_width)

            layout = QVBoxLayout()
            self.setLayout(layout)

            # Type row
            type_row = QHBoxLayout()
            type_row.addWidget(QLabel("Type:"))
            self.type_combo = QComboBox()
            self.type_combo.addItems(["Annotations", "Point annotations", "Regions"])
            type_row.addWidget(self.type_combo)
            layout.addLayout(type_row)

            # Key row
            key_row = QHBoxLayout()
            key_row.addWidget(QLabel("Key:"))
            self.key_combo = QComboBox()
            self.key_combo.setEditable(True)
            key_row.addWidget(self.key_combo)
            layout.addLayout(key_row)

            # Name row
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel("Name:"))
            self.name_combo = QComboBox()
            self.name_combo.setEditable(False)
            self.name_combo.setToolTip(
                "Filter by name. '(all)' loads all shapes for this key."
            )
            name_row.addWidget(self.name_combo)
            layout.addLayout(name_row)

            # Show / Add new buttons
            btn_row = QHBoxLayout()
            self.show_btn = QPushButton("Show")
            self.show_btn.setToolTip("Load shapes from InSituData into the viewer")
            self.show_btn.clicked.connect(self._on_show)
            btn_row.addWidget(self.show_btn)
            self.add_new_btn = QPushButton("Add new")
            self.add_new_btn.setToolTip("Create a new empty layer for drawing")
            self.add_new_btn.clicked.connect(self._on_add_new)
            btn_row.addWidget(self.add_new_btn)
            layout.addLayout(btn_row)

            # Features Table button
            self.features_btn = QPushButton("Open Features Table")
            self.features_btn.clicked.connect(self._open_features_table)
            layout.addWidget(self.features_btn)

            # Connect signals
            self.type_combo.currentIndexChanged.connect(self._on_type_changed)
            self.key_combo.currentIndexChanged.connect(self._refresh_name_combo)
            self.key_combo.editTextChanged.connect(self._refresh_name_combo)
            viewer.layers.events.inserted.connect(self._refresh_key_combo)
            viewer.layers.events.removed.connect(self._refresh_key_combo)

            # Initial population
            self._refresh_key_combo()

        def closeEvent(self, event):
            self.viewer.layers.events.inserted.disconnect(self._refresh_key_combo)
            self.viewer.layers.events.removed.disconnect(self._refresh_key_combo)
            super().closeEvent(event)

        def _get_geom_container(self):
            type_text = self.type_combo.currentText()
            data = self.viewer_config.data
            if type_text in ("Annotations", "Point annotations"):
                return data.annotations
            return data.regions

        def _on_type_changed(self, event=None):
            """Reset Key and Name to defaults when the type selection changes."""
            self.key_combo.blockSignals(True)
            self.key_combo.setCurrentIndex(-1)
            self.key_combo.clearEditText()
            self.key_combo.blockSignals(False)
            self.name_combo.blockSignals(True)
            self.name_combo.setCurrentIndex(0)  # (all)
            self.name_combo.blockSignals(False)
            self._refresh_key_combo()

        def _refresh_key_combo(self, event=None):
            geom = self._get_geom_container()
            keys = (sorted(geom.keys(), key=str.casefold)
                    if not geom.is_empty else [])
            self.key_combo.blockSignals(True)
            current = self.key_combo.currentText()
            self.key_combo.clear()
            self.key_combo.addItems(keys)
            idx = self.key_combo.findText(current)
            if idx >= 0:
                self.key_combo.setCurrentIndex(idx)
            elif current:
                self.key_combo.setEditText(current)
            else:
                self.key_combo.setCurrentIndex(-1)
                self.key_combo.clearEditText()
            self.key_combo.blockSignals(False)
            # Explicit call required: setEditText above fires no signal while
            # blocked, so _refresh_name_combo would not run otherwise.
            self._refresh_name_combo()

        def _refresh_name_combo(self, event=None):
            import warnings
            key_text = self.key_combo.currentText().strip()
            geom = self._get_geom_container()
            self.name_combo.blockSignals(True)
            current = self.name_combo.currentText()
            self.name_combo.clear()
            self.name_combo.addItem(_ALL)
            if key_text and not geom.is_empty:
                try:
                    meta = geom.metadata.get(key_text, {})
                    if 'names' in meta:
                        names = meta['names']
                    elif 'classes' in meta:
                        warnings.warn(
                            f"Geometry metadata for key '{key_text}' uses deprecated field "
                            "'classes'. Please resave the data to upgrade to the new format.",
                            DeprecationWarning,
                        )
                        names = meta['classes']
                    else:
                        names = []
                    for n in sorted(names, key=str.casefold):
                        self.name_combo.addItem(n)
                except (KeyError, AttributeError):
                    pass
            idx = self.name_combo.findText(current)
            if idx >= 0:
                self.name_combo.setCurrentIndex(idx)
            else:
                self.name_combo.setCurrentIndex(0)  # default to (all)
            self.name_combo.blockSignals(False)

        def _get_next_region_name(self, key_text: str) -> str:
            return _get_next_region_name_for_layer(self.viewer, key_text, self.viewer_config)

        def _on_show(self):
            """Load existing shapes from InSituData into the viewer."""
            name_text = self.name_combo.currentText().strip()
            key_text = self.key_combo.currentText().strip()
            type_text = self.type_combo.currentText()
            data = self.viewer_config.data

            if not key_text:
                show_warning("Please enter a key.")
                return

            load_all = (name_text == _ALL)

            if type_text in ("Annotations", "Point annotations"):
                geom_attr = "annotations"
                symbol = (ANNOTATIONS_SYMBOL if type_text == "Annotations"
                          else POINTS_SYMBOL)
                internal_mode = type_text  # "Annotations" or "Point annotations"
            else:
                geom_attr = "regions"
                symbol = REGIONS_SYMBOL
                internal_mode = "Regions"

            geom_container = getattr(data, geom_attr)
            layer_name = f"{symbol} {key_text}"

            key_exists = (not geom_container.is_empty
                          and key_text in geom_container.keys())

            if not key_exists:
                show_warning(
                    f"Key '{key_text}' not found in {geom_attr}. "
                    "Use 'Add new' to create a new layer."
                )
                return

            # Load data, filtered by name if requested
            df = geom_container[key_text]
            if not load_all:
                df = df[df['name'] == name_text]
                if df.empty:
                    show_info(f"No shapes with name '{name_text}' in key '{key_text}'.")
                    return

            if type_text == "Regions":
                if layer_name in self.viewer.layers:
                    filter_hint = (f" (filter: '{name_text}')"
                                   if not load_all else "")
                    show_warning(
                        f"Layer '{layer_name}' already exists{filter_hint}. Use Sync to update."
                    )
                    return
            else:
                # Annotations: merge new UIDs into existing layer if present.
                # Only check the layer that matches the current type to avoid
                # deduplicating against UIDs from a different geometry type.
                existing_ln = layer_name
                if existing_ln in self.viewer.layers:
                    if 'uid' in df.columns:
                        existing_uids = set(
                            self.viewer.layers[existing_ln].features['uid']
                        )
                        df = df[~df['uid'].isin(existing_uids)]
                        if df.empty:
                            show_info(
                                "No new geometries to add — all are already in the viewer."
                            )
                            return

            _add_geometries_as_layer(
                dataframe=df,
                viewer=self.viewer,
                layer_name=key_text,
                mode=internal_mode,
            )

            # napari derives current_properties from the last row of the
            # features table whenever layer.features is assigned (see
            # _get_default_column in layer_utils.py: value = column.iloc[-1]).
            # For annotation layers this would carry the last loaded shape's
            # name into subsequent draws. Reset it to "" so new shapes start
            # unnamed.
            if type_text in ("Annotations", "Point annotations"):
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    cp = dict(layer.current_properties)
                    cp['name'] = np.array([''], dtype='object')
                    layer.current_properties = cp


        def _on_add_new(self):
            """Create a new empty layer for drawing, prompting for a key name."""
            type_text = self.type_combo.currentText()
            config = self.viewer_config

            if type_text in ("Annotations", "Point annotations"):
                symbol = (ANNOTATIONS_SYMBOL if type_text == "Annotations"
                          else POINTS_SYMBOL)
                internal_mode = "Annotations"
                effective_name = ""
            else:
                symbol = REGIONS_SYMBOL
                internal_mode = "Regions"
                effective_name = None  # determined after key is known

            key_text, ok = QInputDialog.getText(
                self, "Add new layer", "Key:"
            )
            key_text = key_text.strip()
            if not ok or not key_text:
                return

            if effective_name is None:
                effective_name = self._get_next_region_name(key_text)

            layer_name = f"{symbol} {key_text}"
            self._create_new_layer(type_text, key_text, effective_name,
                                   layer_name, internal_mode, config)

        def _create_new_layer(self, type_text, key_text, name_text,
                              layer_name, internal_mode, config):
            geom_type_str = ('annotation' if type_text in ("Annotations", "Point annotations")
                             else 'region')
            features = {
                'uid': np.array([], dtype='object'),
                'type': np.array([], dtype='object'),
                'name': np.array([], dtype='object'),
                'geometry_type': np.array([], dtype='object'),
            }
            current_props = {
                'name': np.array([name_text], dtype='object'),
                'uid': np.array([''], dtype='object'),
                'type': np.array([''], dtype='object'),
                'geometry_type': np.array([geom_type_str], dtype='object'),
            }

            if layer_name in self.viewer.layers:
                # Layer exists — update current_properties for new name and
                # ensure color propagation is wired (may not be if layer was
                # created in a prior session without this call).
                layer = self.viewer.layers[layer_name]
                layer.current_properties = current_props
                if name_text:
                    self._set_layer_draw_color(layer, type_text, key_text, name_text, config)
                _connect_color_propagation(layer, config, key_text, type_text)
                return

            if type_text == "Annotations":
                self.viewer.add_shapes(
                    [],
                    name=layer_name,
                    shape_type='polygon',
                    edge_width=4,
                    edge_color='#808080',
                    face_color='transparent',
                    features=features,
                    text={'string': '{name}', 'anchor': 'upper_left',
                          'size': 8, 'color': 'white'},
                )
            elif type_text == "Point annotations":
                self.viewer.add_points(
                    np.zeros((0, 2)),
                    name=layer_name,
                    size=10,
                    border_color='black',
                    face_color='#808080',
                    features=features,
                    text={'string': '{name}', 'anchor': 'upper_left',
                          'size': 8, 'color': 'white'},
                )
            elif type_text == "Regions":
                self.viewer.add_shapes(
                    [],
                    name=layer_name,
                    shape_type='polygon',
                    edge_width=10,
                    edge_color='#ffaa00ff',
                    face_color='transparent',
                    features=features,
                    text={'string': '{name}', 'anchor': 'upper_left',
                          'size': 8, 'color': 'white'},
                )

            if layer_name in self.viewer.layers:
                new_layer = self.viewer.layers[layer_name]
                new_layer.current_properties = current_props
                if name_text:
                    self._set_layer_draw_color(new_layer, type_text, key_text, name_text, config)
                _connect_color_propagation(
                    new_layer, config, key_text, type_text
                )

        def _set_layer_draw_color(self, layer, type_text, key_text, name_text, config):
            """Set current_edge_color / current_border_color from the palette for name_text."""
            import napari.layers as _nl
            if type_text == "Regions":
                color = _get_or_assign_color(
                    config.region_colors, REGIONS_PALETTE,
                    '_region_color_idx', config, key_text, name_text
                )
            else:
                color = _get_or_assign_color(
                    config.annot_point_colors, ANNOTATIONS_PALETTE,
                    '_annot_point_color_idx', config, key_text, name_text
                )
            try:
                if isinstance(layer, _nl.Points):
                    layer.current_face_color = color
                else:
                    layer.current_edge_color = color
            except Exception:
                pass

        def _open_features_table(self):
            # Prefer the currently active layer; fall back to key combo lookup.
            active = self.viewer.layers.selection.active
            if active is not None and isinstance(
                active, (napari.layers.Shapes, napari.layers.Points)
            ):
                layer = active
            else:
                key_text = self.key_combo.currentText().strip()
                type_text = self.type_combo.currentText()
                if type_text == "Annotations":
                    layer_name = f"{ANNOTATIONS_SYMBOL} {key_text}"
                elif type_text == "Point annotations":
                    layer_name = f"{POINTS_SYMBOL} {key_text}"
                else:
                    layer_name = f"{REGIONS_SYMBOL} {key_text}"

                if layer_name not in self.viewer.layers:
                    show_warning("Layer not found in viewer. Add it first.")
                    return
                layer = self.viewer.layers[layer_name]

            result = self.viewer.window.add_plugin_dock_widget(
                plugin_name="napari", widget_name="Features table widget"
            )
            if result is not None:
                dock = result[0] if isinstance(result, tuple) else result
                dock.setFloating(True)
                dock.show()
                dock.raise_()
            self.viewer.layers.selection.active = layer



    class ResetWidgetsButton(QWidget):
        """Button widget to reset/restore all closed widgets in the napari viewer."""

        def __init__(self, widgets_max_width: int = 500):
            super().__init__()
            self.widgets_max_width = widgets_max_width
            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

            # create the reset button
            self.reset_button = QPushButton("Reset Widgets")
            self.reset_button.setToolTip("Restore all closed widgets")
            self.reset_button.clicked.connect(self._reset_widgets)
            self.layout.addWidget(self.reset_button)

        def _reset_widgets(self):
            """Re-add all widgets to the current napari viewer."""
            viewer = napari.current_viewer()
            if viewer is None:
                show_warning("No active napari viewer found.")
                return

            viewer_config = config_manager[_get_viewer_uid(viewer)]

            # Get list of currently open dock widget names
            existing_widgets = set()
            for dock_widget in viewer.window._dock_widgets.values():
                existing_widgets.add(dock_widget.name)

            # Initialize widgets
            (
                show_cells_widget,
                locate_cells_widget,
                geometries_widget,
                show_boundaries_widget,
                select_data,
                filter_cells_widget,
                show_units_widget,
            ) = _initialize_widgets(
                viewer=viewer,
                viewer_config=viewer_config
            )

            # Define widgets to add with their properties
            widgets_config = [
                (select_data, "Select data", 80, False),
                (show_cells_widget, "Show data", 170, False),
                (show_units_widget, "Show spatial units", None, True),
                (show_boundaries_widget, "Show boundaries", None, False),
                (locate_cells_widget, "Navigate to cell", None, False),
                (filter_cells_widget, "Filter cells", 150, True),
                (geometries_widget, "Geometries", None, True),
            ]

            # Add widgets that are not already open
            for widget, name, max_height, tabify in widgets_config:
                if widget is not None and name not in existing_widgets:
                    viewer.window.add_dock_widget(widget, name=name, area="right", tabify=tabify)
                    if max_height is not None:
                        widget.max_height = max_height
                    widget.max_width = self.widgets_max_width

            show_info("Widgets have been reset.")


    class UtilityButtonsWidget(QWidget):
        """Combined dock widget grouping Sync, Refresh, and Reset Widgets buttons."""

        def __init__(self, widgets_max_width: int = 500):
            super().__init__()
            self.widgets_max_width = widgets_max_width
            layout = QVBoxLayout()
            self.setLayout(layout)

            # Sync + Refresh in a shared row
            sync_refresh_row = QHBoxLayout()
            sync_btn = QPushButton("Sync")
            sync_btn.setToolTip("Sync geometries to data, then refresh text labels and colors")
            sync_btn.clicked.connect(self._sync_geometries)
            sync_refresh_row.addWidget(sync_btn)

            refresh_btn = QPushButton("Refresh")
            refresh_btn.setToolTip(
                "Re-apply text labels and colors after editing names in the Features Table"
            )
            refresh_btn.clicked.connect(self._refresh_all_geometry_layers)
            sync_refresh_row.addWidget(refresh_btn)
            layout.addLayout(sync_refresh_row)

            reset_btn = QPushButton("Reset Widgets")
            reset_btn.setToolTip("Restore all closed widgets")
            reset_btn.clicked.connect(self._reset_widgets)
            layout.addWidget(reset_btn)

        def _sync_geometries(self):
            self._refresh_all_geometry_layers()
            sync_geometries()

        def _refresh_all_geometry_layers(self):
            viewer = napari.current_viewer()
            if viewer is None:
                return
            viewer_config = config_manager[_get_viewer_uid(viewer)]
            for layer in viewer.layers:
                if (isinstance(layer, (napari.layers.Shapes, napari.layers.Points))
                        and 'name' in layer.features.columns):
                    layer.refresh_text()
                    _apply_colors_from_features(layer, viewer_config)

        def _reset_widgets(self):
            viewer = napari.current_viewer()
            if viewer is None:
                show_warning("No active napari viewer found.")
                return

            viewer_config = config_manager[_get_viewer_uid(viewer)]

            existing_widgets = set()
            for dock_widget in viewer.window._dock_widgets.values():
                existing_widgets.add(dock_widget.name)

            (
                show_cells_widget,
                locate_cells_widget,
                geometries_widget,
                show_boundaries_widget,
                select_data,
                filter_cells_widget,
                show_units_widget,
            ) = _initialize_widgets(
                viewer=viewer,
                viewer_config=viewer_config
            )

            widgets_config = [
                (select_data, "Select data", 80, False),
                (show_cells_widget, "Show data", 170, False),
                (show_units_widget, "Show spatial units", None, True),
                (show_boundaries_widget, "Show boundaries", None, False),
                (locate_cells_widget, "Navigate to cell", None, False),
                (filter_cells_widget, "Filter cells", 150, True),
                (geometries_widget, "Geometries", None, True),
            ]

            for widget, name, max_height, tabify in widgets_config:
                if widget is not None and name not in existing_widgets:
                    viewer.window.add_dock_widget(widget, name=name, area="right", tabify=tabify)
                    if max_height is not None:
                        widget.max_height = max_height
                    widget.max_width = self.widgets_max_width

            show_info("Widgets have been reset.")


    class ColorLegendWidget(QWidget):
        """Combined widget showing the colour-legend canvas with save controls beneath it."""

        def __init__(self, static_canvas):
            super().__init__()
            layout = QVBoxLayout()
            self.setLayout(layout)

            # Colour legend canvas at the top
            layout.addWidget(static_canvas)

            # Save controls below
            path_layout = QHBoxLayout()

            self.label = QLabel("No folder selected")
            self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.label.setMinimumWidth(150)
            self.label.setMaximumWidth(200)
            self.label.setToolTip("No folder selected")
            path_layout.addWidget(self.label)

            self.select_button = QPushButton("Select")
            self.select_button.setIconSize(QSize(16, 16))
            self.select_button.setToolTip("Select Output Folder")
            self.select_button.clicked.connect(self._select_folder)
            path_layout.addWidget(self.select_button)

            layout.addLayout(path_layout)

            self.save_button = QPushButton("Save")
            self.save_button.clicked.connect(self._save_data)
            layout.addWidget(self.save_button)

            self.output_folder = None

        def _select_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if folder:
                self.output_folder = folder
                metrics = QFontMetrics(self.label.font())
                elided_text = metrics.elidedText(folder, Qt.ElideMiddle, self.label.width())
                self.label.setText(elided_text)
                self.label.setToolTip(folder)

        def _save_data(self):
            if self.output_folder:
                save_colorlegends(output_folder=self.output_folder)
            else:
                self.label.setText("Please select a folder first.")


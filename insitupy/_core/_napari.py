import logging
from typing import TYPE_CHECKING, List, Literal, Optional, Union
from uuid import uuid4
from warnings import warn

import dask.array as da
import numpy as np
from scipy.sparse import issparse

from insitupy._version import __version__
from insitupy._constants import WITH_NAPARI
from insitupy._constants import ANNOTATIONS_SYMBOL, FLUO_CMAP, REGIONS_SYMBOL
from insitupy._exceptions import InSituDataMissingObject
from insitupy.containers._utils import _get_cell_layer
from insitupy.images.axes import ImageAxes
from insitupy.images.utils import _get_contrast_limits, create_img_pyramid
from insitupy.utils._helpers import _get_expression_values
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger("insitupy")


def _sync_cells_for_viewer_if_needed(
    data: Union["InSituData"],
) -> bool:
    cells = getattr(data, "cells", None)
    if cells is None or cells.is_empty:
        return False

    if cells.is_synced:
        return False

    logger.warning("Cell table and boundaries are out of sync. Visualization may show incomplete metadata until `data.cells.sync()` is run.")
    return False

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from insitupy._core.data import InSituData

# optional packages that are not always installed
if WITH_NAPARI:
    import napari
    from napari.layers import Layer, Points, Shapes
    from napari.utils.notifications import show_info, show_warning

    from insitupy.interactive._configs import _get_viewer_uid, config_manager
    from insitupy.interactive._layers import _apply_colors_from_features, _create_points_layer
    from insitupy.interactive._transcript_viewer import (
        TranscriptViewerConfig, create_transcript_viewer_widget)
    from insitupy.interactive._widgets import (ColorLegendWidget, UtilityButtonsWidget)

    #from napari.layers.shapes.shapes import Shapes
    from ..interactive._widgets import (_initialize_widgets, GeometriesWidget)

################################
### NAPARI-RELATED FUNCTIONS ###
################################
if WITH_NAPARI:
    def _add_images_to_viewer(
        data: Union["InSituData"],
        viewer: napari.Viewer,
        grayscale_colormap: List[str] = FLUO_CMAP,
        ):
        images_attr = data._images
        n_images = len(images_attr.metadata)
        n_grayscales = 0 # number of grayscale images
        for i, (img_name, img_metadata) in enumerate(images_attr.metadata.items()):
            # get image
            img = images_attr[img_name]

            # only last image is set visible
            is_visible = False if i < n_images - 1 else True
            pixel_size = img_metadata['pixel_size']

            # check if the current image is RGB
            #is_rgb = self._images.metadata[img_name]["rgb"]
            axes_str = data._images.metadata[img_name]["axes"]
            #shape = data._images.metadata[img_name]["shape"]
            if isinstance(img, list):
                shape = img[0].shape
            else:
                shape = img.shape

            if data._images.metadata[img_name]["axes"] == "CYX":
                if len(shape) == 2:
                    warn((
                        f"Axes information ({axes_str}) and shape ({shape}) do not fit together. Assumed grayscale image with axes 'YX'.\n"
                        f"Error is likely caused by inconsistencies in the metadata file occuring in insitupy versions < 0.9.0."
                        )
                        )
                    axes_str = "YX"

            axes = ImageAxes(axes_str)
            is_rgb = axes.is_rgb

            if axes.C is not None:
                if shape[axes.C] == 1:
                    # only one channel available -> select this channel
                    if not isinstance(img, list):
                        img = da.take(img, indices=0, axis=axes.C)
                    else:
                        img = [da.take(elem, indices=0, axis=axes.C) for elem in img]
                    multichannel = False
                else:
                    multichannel = True
            else:
                multichannel = False

            if not is_rgb and multichannel:
                if not isinstance(img, list):
                    n_channels = img.shape[axes.C]
                else:
                    n_channels = img[0].shape[axes.C]

                try:
                    # get channel names
                    channel_names = [
                        elem["Name"]
                        for elem
                        in data._images.metadata[img_name]['OME']['Image']['Pixels']['Channel']
                        ]
                except KeyError:
                    channel_names = [f"Channel {i+1}" for i in range(n_channels)]

                # Multichannel grayscale image
                for ch in range(n_channels):
                    # get channel name
                    ch_name = channel_names[ch]

                    # select channel
                    if not isinstance(img, list):
                        channel_img = da.take(img, indices=ch, axis=axes.C)
                    else:
                        channel_img = [da.take(elem, indices=ch, axis=axes.C) for elem in img]
                    #channel_img = img[ch]

                    # select color map
                    if ch_name in ["nuclei", "nucleus"] or "DAPI" in ch_name:
                        cmap = "blue"
                    else:
                        cmap = grayscale_colormap[n_grayscales % len(grayscale_colormap)]
                        n_grayscales += 1

                    if not isinstance(channel_img, list):
                        # create image pyramid for lazy loading
                        img_pyramid = create_img_pyramid(
                            img=channel_img,
                            axes=axes_str,
                            nsubres=6
                            )
                    else:
                        img_pyramid = channel_img

                    # get contrast limits
                    contrast_limits = _get_contrast_limits(img_pyramid)

                    if contrast_limits[1] == 0:
                        warn("The maximum value of the image is 0. Is the image really completely empty?")
                        contrast_limits = (0, 255)

                    # add image to viewer
                    viewer.add_image(
                        img_pyramid,
                        name=f"{img_name}: {ch_name}",
                        colormap=cmap,
                        blending="additive",
                        rgb=False,
                        contrast_limits=contrast_limits,
                        scale=(pixel_size, pixel_size),
                        visible=is_visible
                    )

            else:
                if is_rgb:
                    cmap = None  # default value of cmap
                    blending = "translucent_no_depth"  # set blending mode
                else:
                    if img_name in ["nuclei", "nucleus"] or "DAPI" in img_name:
                        cmap = "blue"
                    else:
                        cmap = grayscale_colormap[n_grayscales % len(grayscale_colormap)]
                        n_grayscales += 1
                    blending = "additive"  # set blending mode

                if not isinstance(img, list):
                    # create image pyramid for lazy loading
                    img_pyramid = create_img_pyramid(
                        img=img,
                        axes=axes_str,
                        nsubres=6
                        )
                else:
                    img_pyramid = img

                # infer contrast limits
                contrast_limits = _get_contrast_limits(img_pyramid)

                if contrast_limits[1] == 0:
                    warn("The maximum value of the image is 0. Is the image really completely empty?")
                    contrast_limits = (0, 255)

                # add img pyramid to napari viewer
                viewer.add_image(
                        img_pyramid,
                        name=img_name,
                        colormap=cmap,
                        blending=blending,
                        rgb=is_rgb,
                        contrast_limits=contrast_limits,
                        scale=(pixel_size, pixel_size),
                        visible=is_visible
                    )

    def _add_cells_to_viewer(
        data: Union["InSituData", "StructuredSpatialData"],
        viewer: napari.viewer,
        keys: str,
        key_type: Literal["genes", "obs", "obsm"] = "genes",
        cells_layer: Optional[str] = None,
        point_size: int = 8
        ):
        if data.cells.is_empty:
            raise InSituDataMissingObject("cells")
        else:
            celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)

            if cells_layer is None:
                cells_layer_name = data.cells.main_key
            else:
                cells_layer_name = cells_layer

            # convert keys to list
            keys = convert_to_list(keys)

            # get point coordinates
            points = np.flip(celldata.table.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])
            #points *= pixel_size # convert to length unit (e.g. µm)

            # get expression matrix
            if issparse(celldata.table.X):
                X = celldata.table.X.toarray()
            else:
                X = celldata.table.X

            for i, k in enumerate(keys):
                # get expression values
                color_value = _get_expression_values(
                    adata=celldata.table,
                    X=X,
                    key_type=key_type, key=k
                )

                # extract names of cells
                cell_names = celldata.table.obs_names.values

                # create points layer
                layer = _create_points_layer(
                    points=points,
                    color_values=color_value,
                    name=f"{cells_layer_name}-{k}",
                    point_names=cell_names,
                    point_size=point_size,
                    visible=True
                )

                # add layer programmatically - does not work for all types of layers
                # see: https://forum.image.sc/t/add-layerdatatuple-to-napari-viewer-programmatically/69878
                #self._viewer.add_layer(Layer.create(*layer))
                viewer.add_layer(Layer.create(*layer))

    def _add_widgets_to_viewer(
        data: Union["InSituData", "StructuredSpatialData"],
        viewer: napari.Viewer,
        widgets_max_width: int = 500
        ):
        # get viewer configuration from configuration manager
        viewer_config = config_manager[_get_viewer_uid(viewer)]

        # initialize the widgets
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

        # add widgets to napari window
        if select_data is not None:
            viewer.window.add_dock_widget(select_data, name="Select data", area="right", tabify=False)
            select_data.max_height = 80
            select_data.max_width = widgets_max_width

        if show_cells_widget is not None:
            viewer.window.add_dock_widget(show_cells_widget, name="Show cells", area="right", tabify=False)
            show_cells_widget.max_height = 200
            show_cells_widget.max_width = widgets_max_width

        if show_units_widget is not None:
            viewer.window.add_dock_widget(show_units_widget, name="Show spatial units", area="right", tabify=True)
            show_units_widget.max_width = widgets_max_width

        if show_boundaries_widget is not None:
            viewer.window.add_dock_widget(show_boundaries_widget, name="Show boundaries", area="right", tabify=False)
            show_boundaries_widget.max_width = widgets_max_width

        if locate_cells_widget is not None:
            viewer.window.add_dock_widget(locate_cells_widget, name="Navigate to cell", area="right", tabify=False)
            locate_cells_widget.max_width = widgets_max_width

        if filter_cells_widget is not None:
            viewer.window.add_dock_widget(filter_cells_widget, name="Filter cells", area="right", tabify=True)
            filter_cells_widget.max_height = 150
            filter_cells_widget.max_width = widgets_max_width

        # add unified geometries widget
        viewer.window.add_dock_widget(geometries_widget, name="Geometries", area="right", tabify=True)
        geometries_widget.max_width = widgets_max_width


    def _add_events_to_viewer(viewer: napari.Viewer):
        # get viewer configuration from configuration manager
        viewer_config = config_manager[_get_viewer_uid(viewer)]

        # Assign function to an layer addition event
        def _update_uid(event):
            global uids_before_removal
            if event is not None:
                layer = event.source
                logger.debug(event.action) if viewer_config.verbose else None
                if event.action == "added" and viewer_config._auto_set_uid:
                    if isinstance(layer, Shapes):
                        type_last = layer.shape_type[-1]
                        if type_last in ["polygon", "rectangle", "ellipse"]:
                            geom_type = "polygon_exterior"
                        elif type_last in ["path", "line"]:
                            geom_type = "line"
                        else:
                            show_warning(f"Unsupported shape type '{type_last}' for UID assignment. Only 'polygon' and 'path' are supported.")
                            return
                    elif isinstance(layer, Points):
                        geom_type = "point"
                        type_last = "point"
                    else:
                        return

                    uid = str(uuid4())
                    logger.debug(f"Added '{type_last}' with UID '{uid}'") if viewer_config.verbose else None

                    if layer.name.startswith(ANNOTATIONS_SYMBOL):
                        geometry_type_label = "annotation"
                    elif layer.name.startswith(REGIONS_SYMBOL):
                        geometry_type_label = "region"
                    else:
                        geometry_type_label = ""

                    # update via layer.features assignment to fire the features_update signal
                    # so the Features Table widget refreshes automatically
                    updated = layer.features.copy()
                    # Read the intended name from current_properties if set
                    cp = getattr(layer, 'current_properties', {})
                    intended_name = ""
                    if 'name' in cp:
                        val = cp['name']
                        intended_name = val[0] if hasattr(val, '__len__') and len(val) else str(val)

                    updated.loc[updated.index[-1], 'uid'] = uid
                    updated.loc[updated.index[-1], 'type'] = geom_type
                    if 'name' not in updated.columns:
                        updated['name'] = ""
                    updated.loc[updated.index[-1], 'name'] = intended_name
                    updated.loc[updated.index[-1], 'geometry_type'] = geometry_type_label
                    layer.features = updated

                elif event.action == "removing":
                    uids_before_removal = set(layer.features['uid'])
                elif event.action == "removed":
                    removed_uids = uids_before_removal ^ set(layer.features['uid'])
                    logger.debug(f"Removed following UIDs: {removed_uids}") if viewer_config.verbose else None
                    viewer_config._removal_tracker += list(removed_uids)
                else:
                    pass

        def _on_features_changed(e, _l=None):
            _l.refresh_text()
            _apply_colors_from_features(_l, viewer_config)

        # Assign the function to data of all existing layers
        for layer in viewer.layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                layer.events.data.connect(_update_uid)
                layer.events.features.connect(
                    lambda e, _l=layer: _on_features_changed(e, _l)
                )

        # Connect the function to the data of existing shapes and points layers in the viewer
        def connect_to_all_shapes_layers(event):
            layer = event.source[event.index]
            if event is not None:
                if isinstance(layer, Shapes) or isinstance(layer, Points):
                    layer.events.data.connect(_update_uid)
                    layer.events.features.connect(
                        lambda e, _l=layer: _on_features_changed(e, _l)
                    )

        # Connect the function to any new layers added to the viewer
        viewer.layers.events.inserted.connect(connect_to_all_shapes_layers)

    def _add_color_legend_to_viewer(viewer: napari.Viewer):
        config = config_manager[_get_viewer_uid(viewer)]
        color_legend_widget = ColorLegendWidget(static_canvas=config.static_canvas)
        viewer.window.add_dock_widget(color_legend_widget, area='left', name='Color legend')

    def _add_buttons_to_viewer(viewer: napari.Viewer, widgets_max_width: int = 500):
        # create combined utility buttons widget
        utility_buttons = UtilityButtonsWidget(widgets_max_width=widgets_max_width)

        # add the combined widget to viewer
        viewer.window.add_dock_widget(utility_buttons, area='right', name="Utilities")

    def _add_transcript_viewer_to_viewer(
        data: Union["InSituData", "StructuredSpatialData"],
        viewer: napari.Viewer,
        lazy_loading: bool = True,
        transcript_config: Optional[TranscriptViewerConfig] = None,
        widgets_max_width: int = 500,
    ) -> None:
        """Add transcript viewer widget to the napari viewer.

        Args:
            data: InSituData or StructuredSpatialData object containing transcript data.
            viewer: The napari viewer instance.
            lazy_loading: If True, use lazy loading mode for large datasets.
            transcript_config: Configuration for the transcript viewer widget.
            widgets_max_width: Maximum width of the widget in pixels.
        """
        if data.transcripts is None:
            return

        # Get viewer config for color legend integration
        viewer_config = config_manager[_get_viewer_uid(viewer)]

        transcript_widget = create_transcript_viewer_widget(
            viewer=viewer,
            transcripts=data.transcripts,
            lazy_loading=lazy_loading,
            config=transcript_config,
            viewer_config=viewer_config,
        )
        transcript_widget.setMaximumWidth(widgets_max_width)

        # Add transcript viewer widget
        transcript_dock = viewer.window.add_dock_widget(
            transcript_widget,
            area='right',
            name="Transcript Viewer",
            tabify=False  # Don't auto-tabify, we'll manually tabify with "Show cells"
        )

        # Explicitly tabify with "Show cells" widget if it exists
        main_window = viewer.window._qt_window
        show_cells_dock = None
        for dock in main_window.findChildren(type(transcript_dock)):
            if dock.windowTitle() == "Show cells":
                show_cells_dock = dock
                break

        if show_cells_dock is not None:
            main_window.tabifyDockWidget(show_cells_dock, transcript_dock)
            # Ensure "Show cells" tab is active (first tab)
            show_cells_dock.raise_()


    def _show(
        data: Union["InSituData", "StructuredSpatialData"],
        keys: Optional[str] = None,
        key_type: Literal["genes", "obs", "obsm"] = "genes",
        cells_layer: Optional[str] = None,
        point_size: int = 8,
        scalebar: bool = True,
        unit: str = "µm",
        return_viewer: bool = False,
        widgets_max_width: int = 500,
        verbose: bool = False,
        show_transcripts: bool = True,
        transcript_lazy_loading: bool = True,
        transcript_config: Optional[TranscriptViewerConfig] = None,
        ):

        _sync_cells_for_viewer_if_needed(data)

        # initialize a config class manager with new ID
        uid_viewer = config_manager.add_config(data=data)
        current_viewer_config = config_manager[uid_viewer] # get current viewer config
        if verbose:
            current_viewer_config.verbose = True

        # create viewer
        current_viewer = napari.Viewer(title=f"#{uid_viewer}: {data.sample_id}")

        # IMAGES
        if data.images.is_empty:
            warn("No images found.")
        else:
            _add_images_to_viewer(
                data=data,
                viewer=current_viewer
                )

        # CELLS
        if keys is not None:
            _add_cells_to_viewer(
                data=data,
                viewer=current_viewer,
                keys=keys,
                key_type=key_type,
                cells_layer=cells_layer,
                point_size=point_size
                )

        # WIDGETS
        _add_widgets_to_viewer(
            data=data,
            viewer=current_viewer,
            widgets_max_width=widgets_max_width
        )

        # TRANSCRIPT VIEWER
        if show_transcripts and data.transcripts is not None:
            _add_transcript_viewer_to_viewer(
                data=data,
                viewer=current_viewer,
                lazy_loading=transcript_lazy_loading,
                transcript_config=transcript_config,
                widgets_max_width=widgets_max_width,
            )

        # BUTTONS
        _add_buttons_to_viewer(
            viewer=current_viewer,
            widgets_max_width=widgets_max_width
        )

        # EVENTS
        _add_events_to_viewer(viewer=current_viewer)

        # COLOR LEGEND
        _add_color_legend_to_viewer(viewer=current_viewer)

        # NAPARI SETTINGS
        if scalebar:
            # add scale bar
            current_viewer.scale_bar.visible = True
            current_viewer.scale_bar.unit = unit

        napari.run()

        if return_viewer:
            return current_viewer
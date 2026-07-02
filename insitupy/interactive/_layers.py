import logging

from insitupy._constants import WITH_NAPARI

logger = logging.getLogger(__name__)

if WITH_NAPARI:
    from numbers import Number
    from typing import Literal
    from warnings import warn

    import matplotlib
    import napari
    import numpy as np
    import pandas as pd
    from geopandas import GeoDataFrame
    from matplotlib.colors import rgb2hex, to_rgba
    from napari.types import LayerDataTuple
    from napari.utils.notifications import show_info, show_warning
    from shapely import LinearRing, LineString, MultiPoint, MultiPolygon, Point, Polygon

    from insitupy._constants import (
        ANNOTATIONS_SYMBOL,
        DEFAULT_CATEGORICAL_CMAP,
        DEFAULT_CONTINUOUS_CMAP,
        POINTS_SYMBOL,
        REGIONS_SYMBOL,
    )
    from insitupy.interactive._configs import _get_viewer_uid, config_manager
    from insitupy.palettes import ANNOTATIONS_PALETTE, REGIONS_PALETTE
    from insitupy.utils._checks import check_rgb_column
    from insitupy.utils._colors import _data_to_rgba

    def _get_or_assign_color(registry: dict, palette: list, idx_attr: str,
                              config, key: str, name: str) -> str:
        """Return the registered hex colour for (key, name), assigning a new one if needed."""
        if name is None or name == "":
            return "#808080"  # grey for unnamed geometries
        pair = (key, name)
        if pair not in registry:
            idx = getattr(config, idx_attr)
            registry[pair] = palette[idx % len(palette)]
            setattr(config, idx_attr, idx + 1)
        return registry[pair]

    def _connect_color_propagation(layer, config, key: str, geom_type: str):
        """Propagate manual colour changes to all shapes/points with the same name."""
        if layer is None:
            return
        is_points = isinstance(layer, napari.layers.Points)
        # Points encode name-color in face_color (analogous to edge_color for Shapes).
        event = (layer.events.current_face_color if is_points
                 else layer.events.current_edge_color)

        def _on_color_change(event=None):
            # Skip during drawing modes — emitting edge_color there corrupts
            # napari's mid-draw state (_moving_shape / _last_cursor_position).
            if hasattr(layer, 'mode') and layer.mode not in ('select', 'direct'):
                return
            selected = list(layer.selected_data)
            if not selected:
                return
            new_color = (layer.current_face_color if is_points
                         else layer.current_edge_color)
            if 'name' not in layer.features.columns:
                return
            names = layer.features.loc[selected, 'name'].unique()
            for name in names:
                if name == "" or name is None:
                    continue
                mask = layer.features['name'] == name
                indices = list(layer.features.index[mask])
                if is_points:
                    layer.face_color[indices] = new_color
                    layer.events.face_color()
                else:
                    # _data_view.update_edge_color is used intentionally here instead of
                    # layer.edge_color = [...]: the public setter may fire
                    # events.current_edge_color as a side effect, which would re-trigger
                    # this very callback and cause an infinite loop.
                    for i in indices:
                        layer._data_view.update_edge_color(i, new_color)
                    layer.events.edge_color()
                registry = (config.region_colors if geom_type == "Regions"
                            else config.annot_point_colors)
                registry[(key, name)] = rgb2hex(to_rgba(new_color)[:3])

        event.connect(_on_color_change)

    def _apply_colors_from_features(layer, viewer_config):
        """Re-apply palette colors to all shapes/points based on the current 'name' feature column.

        Called on layer.events.features so that editing a name (e.g. via the Features Table)
        immediately updates the visible edge/border colors.
        """
        if 'name' not in layer.features.columns:
            return
        n = len(layer.data)
        if n == 0:
            return

        # Strip symbol prefix to recover the bare key
        key = layer.name
        for sym in (REGIONS_SYMBOL, ANNOTATIONS_SYMBOL, POINTS_SYMBOL):
            if key.startswith(sym + " "):
                key = key[len(sym) + 1:]
                break

        is_region = layer.name.startswith(REGIONS_SYMBOL + " ")
        is_points = isinstance(layer, napari.layers.Points)

        if is_region:
            registry = viewer_config.region_colors
            palette = REGIONS_PALETTE
            idx_attr = '_region_color_idx'
        else:
            registry = viewer_config.annot_point_colors
            palette = ANNOTATIONS_PALETTE
            idx_attr = '_annot_point_color_idx'

        names = layer.features['name'].tolist()
        if len(names) != n:
            return

        colors = [_get_or_assign_color(registry, palette, idx_attr, viewer_config, key, nm)
                  for nm in names]
        try:
            if is_points:
                layer.face_color = [to_rgba(c) for c in colors]
            else:
                # Use the public edge_color setter (fires events.edge_color automatically).
                layer.edge_color = [to_rgba(c) for c in colors]
        except Exception:
            logger.exception("_apply_colors_from_features failed")

    def _add_geometries_as_layer(
        dataframe: pd.DataFrame,
        viewer: napari.Viewer,
        layer_name: str,
        opacity: float = 1,
        mode: Literal["Annotations", "Point annotations", "Regions"] = "Annotations",
        tolerance: Number = 5
        ):

        # list to store information on shapes
        shape_list = []
        shape_type_list = []

        # lists to store the point x and y coordinates
        point_x_list = []
        point_y_list = []

        # iterate through annotations of this class and collect them as list
        color_list = {"Points": [], "Shapes": []}
        uid_list = {"Points": [], "Shapes": []}
        type_list = {"Points": [], "Shapes": []} # list to store whether the polygon is exterior or interior
        names_list = {"Points": [], "Shapes": []}
        size_list = {"Points": []}

        # hardcode edge_width based on type
        edge_width = 10 if mode == "Regions" else 4

        # determine geometry_type string for features column
        if mode == "Regions":
            geometry_type = "region"
        else:
            geometry_type = "annotation"

        # get the config
        viewer_config = config_manager[_get_viewer_uid(viewer)]

        # prepare layer names
        if mode == "Regions":
            shapes_layer_name_with_symbol = REGIONS_SYMBOL + " " + layer_name
        elif mode in ("Annotations", "Point annotations"):
            shapes_layer_name_with_symbol = ANNOTATIONS_SYMBOL + " " + layer_name
        else:
            raise ValueError(f"Unknown value for `mode`: {mode}")

        points_layer_name_with_symbol = POINTS_SYMBOL + " " + layer_name

        if shapes_layer_name_with_symbol in viewer.layers:
            shapes_layer_exists = True
        else:
            shapes_layer_exists = False

        if points_layer_name_with_symbol in viewer.layers:
            points_layer_exists = True
        else:
            points_layer_exists = False

        # check if per-row colors are given in the dataframe
        has_df_colors = "color" in dataframe.columns
        if has_df_colors:
            rgbs_valid = check_rgb_column(dataframe, "color")
            if not rgbs_valid:
                warn('Not all RGB values given in column "color" are valid. Used grey for all geometries.')
                dataframe = dataframe.copy()
                dataframe["color"] = [(128, 128, 128)] * len(dataframe)

        # pre-compute sets for O(1) duplicate UID lookups
        # Guard: layers loaded from synced data may lack a 'uid' column — fall
        # back to an empty set so all rows are treated as new (no dedup).
        _shape_feats = viewer.layers[shapes_layer_name_with_symbol].features if shapes_layer_exists else None
        existing_shape_uids = (set(_shape_feats["uid"])
                               if (_shape_feats is not None and "uid" in _shape_feats.columns)
                               else set())
        _point_feats = viewer.layers[points_layer_name_with_symbol].features if points_layer_exists else None
        existing_point_uids = (set(_point_feats["uid"])
                               if (_point_feats is not None and "uid" in _point_feats.columns)
                               else set())

        # colour registry for this call
        if mode == "Regions":
            _registry = viewer_config.region_colors
            _palette = REGIONS_PALETTE
            _idx_attr = '_region_color_idx'
        else:
            _registry = viewer_config.annot_point_colors
            _palette = ANNOTATIONS_PALETTE
            _idx_attr = '_annot_point_color_idx'

        for uid, row in dataframe.iterrows():
            # get coordinates
            geometry = row["geometry"]

            if has_df_colors:
                rgb = row["color"]
                hexcolor = rgb2hex([elem / 255 for elem in rgb])
                rgbacolor = [elem / 255 for elem in rgb] + [1]
            else:
                name_val = row["name"] if "name" in dataframe.columns else ""
                hexcolor = _get_or_assign_color(
                    _registry, _palette, _idx_attr, viewer_config, layer_name, name_val
                )
                rgbacolor = list(to_rgba(hexcolor))

            # check if polygon is a MultiPolygon or just a simple Polygon object
            if isinstance(geometry, MultiPolygon):
                data = list(geometry.geoms)
                annotation_type = "polygon_like"
            elif isinstance(geometry, Polygon):
                data = [geometry]
                annotation_type = "polygon_like"
            elif isinstance(geometry, LineString):
                data = geometry
                annotation_type = "line_like"
            elif isinstance(geometry, Point) or isinstance(geometry, MultiPoint):
                data = geometry
                annotation_type = "point_like"
            else:
                raise ValueError(f"Received unknown geometry type: {type(geometry)}")

            # check if this uid is already in the current layer. If so, skip it
            if annotation_type in ["polygon_like", "line_like"]:
                if uid in existing_shape_uids:
                    logger.debug(f"Already in layer: {uid}") if viewer_config.verbose else None
                    continue

            if annotation_type == "point_like":
                if uid in existing_point_uids:
                    logger.debug(f"Already in layer: {uid}") if viewer_config.verbose else None
                    continue

            if annotation_type == "polygon_like":
                for p in data:
                    # simplify polygon for visualization
                    p = p.simplify(tolerance)
                    # extract exterior coordinates from shapely object
                    # Note: the last coordinate is removed since it is identical with the first
                    # in shapely objects, leading sometimes to visualization bugs in napari
                    exterior_array = np.array([p.exterior.coords.xy[1].tolist()[:-1],
                                            p.exterior.coords.xy[0].tolist()[:-1]]).T
                    shape_list.append(exterior_array)  # collect shape
                    color_list["Shapes"].append(hexcolor)  # collect corresponding color
                    uid_list["Shapes"].append(uid)  # collect corresponding unique id
                    type_list["Shapes"].append("polygon_exterior")
                    names_list["Shapes"].append(row["name"])
                    shape_type_list.append("polygon")

                    # if polygon has interiors, plot them as well
                    # for information on donut-shaped polygons in napari see: https://forum.image.sc/t/is-it-possible-to-generate-doughnut-shapes-in-napari-shapes-layer/88834
                    if len(p.interiors) > 0:
                        for linear_ring in p.interiors:
                            if isinstance(linear_ring, LinearRing):
                                interior_array = np.array([linear_ring.coords.xy[1].tolist()[:-1],
                                                        linear_ring.coords.xy[0].tolist()[:-1]]).T
                                shape_list.append(interior_array)  # collect shape
                                color_list["Shapes"].append(hexcolor)  # collect corresponding color
                                uid_list["Shapes"].append(uid)  # collect corresponding unique id
                                type_list["Shapes"].append("polygon_interior")
                                names_list["Shapes"].append(row["name"])
                                shape_type_list.append("polygon")
                            else:
                                raise ValueError(f"Input must be a LinearRing object. Received: {type(linear_ring)}")

            elif annotation_type == "line_like":
                line_array = np.array([data.coords.xy[1].tolist(), data.coords.xy[0].tolist()]).T

                # collect data
                shape_list.append(line_array)
                color_list["Shapes"].append(hexcolor)  # collect corresponding color
                uid_list["Shapes"].append(uid)  # collect corresponding unique id
                type_list["Shapes"].append("line") # information on type of coordinates - important for interior/exterior of polygons
                names_list["Shapes"].append(row["name"])
                shape_type_list.append("path")

            elif annotation_type == "point_like":
                try:
                    # in case of MultiPoints we first have to extract the individual geometries
                    point_coords = [elem.coords.xy for elem in geometry.geoms]
                except AttributeError:
                    # a normal Point object does not have multiple geometries and coordinates can be accessed directly
                    point_coords = [geometry.coords.xy]

                row_size = float(row["size"]) if "size" in dataframe.columns else 10.0

                # collect coordinates and other data on the points
                for coord in point_coords:
                    point_x_list.append(coord[1].tolist()[0])
                    point_y_list.append(coord[0].tolist()[0])
                    color_list["Points"].append(rgbacolor)  # collect corresponding color
                    uid_list["Points"].append(uid)  # collect corresponding unique id
                    type_list["Points"].append("point") # information on type of coordinates - important for interior/exterior of polygons
                    names_list["Points"].append(row["name"])
                    size_list["Points"].append(row_size)

        if len(shape_list) > 0:
            features_dict = {
                    'uid': uid_list["Shapes"],
                    'type': type_list["Shapes"],
                    'name': names_list["Shapes"],
                    'geometry_type': [geometry_type] * len(uid_list["Shapes"]),
                }
            text_dict = {
                'string': '{name}',
                'anchor': 'upper_left',
                'size': 8,
                'color': 'white'
                }

            # add shapes to viewer
            if shapes_layer_name_with_symbol in viewer.layers:
                show_warning(f"A layer with the name '{shapes_layer_name_with_symbol}' already exists. Shapes added to this layer.")
                layer = viewer.layers[shapes_layer_name_with_symbol]

                viewer_config._auto_set_uid = False
                layer.add(
                    data=shape_list,
                    shape_type=shape_type_list,
                    edge_width=edge_width, # µm
                    edge_color=color_list["Shapes"],
                    face_color='transparent',
                )
                viewer_config._auto_set_uid = True

                # layer.add() already appended empty rows to layer.features; fill them in
                # and reassign so the Features Table widget is notified via features_update signal
                updated = layer.features.copy()
                n = len(shape_list)
                updated.loc[updated.index[-n:], 'uid'] = features_dict['uid']
                updated.loc[updated.index[-n:], 'type'] = features_dict['type']
                updated.loc[updated.index[-n:], 'name'] = features_dict['name']
                updated.loc[updated.index[-n:], 'geometry_type'] = features_dict['geometry_type']
                layer.features = updated

            else:
                viewer.add_shapes(
                    data=shape_list,
                    name=shapes_layer_name_with_symbol,
                    features=features_dict,
                    shape_type=shape_type_list,
                    edge_width=edge_width, # µm
                    edge_color=color_list["Shapes"],
                    face_color='transparent',
                    opacity=opacity,
                    #scale=scale_factor,
                    text=text_dict
                    )
                show_info(f"New layer '{shapes_layer_name_with_symbol}' created.")
                _connect_color_propagation(
                    viewer.layers[shapes_layer_name_with_symbol],
                    viewer_config, layer_name, mode
                )

        point_data = np.stack([point_x_list, point_y_list]).T
        if len(point_data) > 0:
            features_dict = {
                        'uid': uid_list["Points"],
                        'type': type_list["Points"],
                        'name': names_list["Points"],
                        'geometry_type': [geometry_type] * len(uid_list["Points"]),
                        'size': size_list["Points"],
                    }

            if points_layer_name_with_symbol in viewer.layers:
                show_warning(f"A layer with the name '{points_layer_name_with_symbol}' already exists. Points added to this layer.")
                layer = viewer.layers[points_layer_name_with_symbol]

                layer.add(
                    coords=point_data,
                )

                # change colors and sizes of the newly added data
                layer.face_color[-len(point_data):] = color_list["Points"]
                layer.size[-len(point_data):] = size_list["Points"]
                layer.refresh() # refresh layer to show new colors

                # layer.add() already appended empty rows to layer.features; fill them in
                # and reassign so the Features Table widget is notified via features_update signal
                updated = layer.features.copy()
                n = len(point_data)
                updated.loc[updated.index[-n:], 'uid'] = features_dict['uid']
                updated.loc[updated.index[-n:], 'type'] = features_dict['type']
                updated.loc[updated.index[-n:], 'name'] = features_dict['name']
                updated.loc[updated.index[-n:], 'geometry_type'] = features_dict['geometry_type']
                updated.loc[updated.index[-n:], 'size'] = features_dict['size']
                layer.features = updated
            else:
                viewer.add_points(
                    data=point_data,
                    name=points_layer_name_with_symbol,
                    features=features_dict,
                    size=size_list["Points"],
                    border_color="black",
                    face_color=color_list["Points"],
                    text={'string': '{name}', 'anchor': 'upper_left',
                          'size': 8, 'color': 'white'},
                    #scale=scale_factor
                )
                _connect_color_propagation(
                    viewer.layers[points_layer_name_with_symbol],
                    viewer_config, layer_name, mode
                )

    def _create_points_layer(points,
                            color_values: list[Number],
                            name: str,
                            point_names: list[str],
                            point_size: int = 6, # is in scale unit (so mostly µm)
                            opacity: float = 1,
                            visible: bool = True,
                            border_width: float = 0,
                            border_color: str = 'red',
                            upper_climit_pct: int = 99,
                            categorical_cmap: matplotlib.colors.ListedColormap = DEFAULT_CATEGORICAL_CMAP,
                            continuous_cmap = DEFAULT_CONTINUOUS_CMAP,
                            display_scope: tuple | None = None,
                            ) -> LayerDataTuple:
        if categorical_cmap is None:
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP
        # get colors
        colors, mapping, cmap = _data_to_rgba(data=color_values,
                               continuous_cmap=continuous_cmap,
                               categorical_cmap=categorical_cmap,
                               upper_climit_pct=upper_climit_pct)

        # generate point layer
        layer = (
            points,
            {
                'name': name,
                'properties': {
                    "value": color_values,
                    "cell_name": point_names
                    },
                'symbol': 'o',
                'size': point_size,
                'face_color': colors,
                'opacity': opacity,
                'visible': visible,
                'border_width': border_width,
                'border_color': border_color,
                'metadata': {"upper_climit_pct": upper_climit_pct, "display_scope": display_scope}
                },
            'points'
            )
        return layer

    def _update_points_layer(
        layer: napari.layers.Layer,
        new_color_values: list[Number],
        new_name: str | None = None,
        upper_climit_pct: int = 99,
        categorical_cmap: matplotlib.colors.ListedColormap = DEFAULT_CATEGORICAL_CMAP,
        continuous_cmap = DEFAULT_CONTINUOUS_CMAP,
        # cmap: str = "viridis"
        ) -> None:
        # get the RGBA colors for the new values
        if categorical_cmap is None:
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP
        new_colors, mapping, cmap = _data_to_rgba(data=new_color_values,
                                                continuous_cmap=continuous_cmap,
                                                categorical_cmap=categorical_cmap,
                                                upper_climit_pct=upper_climit_pct)

        # change the colors of the layer
        layer.face_color = new_colors

        # change properties of layer
        new_props = layer.properties.copy()
        new_props['value'] = new_color_values
        layer.properties = new_props

        if new_name is not None:
            layer.name = new_name

        # re-show the layer in case the user had hidden it — updating it with new
        # values means the user wants to see it again
        layer.visible = True

    def _update_units_layer(
        layer: napari.layers.Layer,
        new_color_values: list[Number],
        new_name: str | None = None,
        upper_climit_pct: int = 99,
        categorical_cmap: matplotlib.colors.ListedColormap = DEFAULT_CATEGORICAL_CMAP,
        continuous_cmap = DEFAULT_CONTINUOUS_CMAP,
        ) -> None:
        """
        Update an existing spatial units (shapes) layer with new color values.

        Args:
            layer: Existing napari shapes layer to update
            new_color_values: New values to color polygons by
            new_name: New name for the layer (optional)
            upper_climit_pct: Upper percentile for color limits
            categorical_cmap: Colormap for categorical data
            continuous_cmap: Colormap for continuous data
        """
        if categorical_cmap is None:
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP

        # Get the RGBA colors for the new values
        new_colors, mapping, cmap = _data_to_rgba(
            data=new_color_values,
            continuous_cmap=continuous_cmap,
            categorical_cmap=categorical_cmap,
            upper_climit_pct=upper_climit_pct
        )

        # Update face colors
        layer.face_color = new_colors

        # Update properties
        new_props = layer.properties.copy()
        new_props['value'] = new_color_values
        layer.properties = new_props

        if new_name is not None:
            layer.name = new_name

    def _create_units_layer(
            gdf: GeoDataFrame,
            color_values: list[Number] | None = None,
            name: str = "units",
            unit_names: list[str] | None = None,
            edge_width: Number = 2,
            opacity: float = 0.5,
            upper_climit_pct: int = 99,
            categorical_cmap: matplotlib.colors.ListedColormap = DEFAULT_CATEGORICAL_CMAP,
            continuous_cmap = DEFAULT_CONTINUOUS_CMAP,
            tolerance: Number = 1
        ) -> LayerDataTuple:
        """
        Create a napari shapes layer from SpatialUnitsData GeoDataFrame.

        Args:
            gdf: GeoDataFrame with polygon geometries
            color_values: Values to color polygons by (optional)
            name: Layer name
            unit_names: Names of spatial units for properties
            edge_width: Edge width in physical units
            opacity: Polygon opacity
            upper_climit_pct: Upper percentile for color limits
            categorical_cmap: Colormap for categorical data
            continuous_cmap: Colormap for continuous data
            tolerance: Simplification tolerance for polygons

        Returns:
            LayerDataTuple for napari
        """
        from shapely import MultiPolygon, Polygon

        if categorical_cmap is None:
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP

        # Extract polygon coordinates from geometries
        shapes_list = []
        shape_types = []

        for geom in gdf.geometry:
            # Simplify for performance
            geom = geom.simplify(tolerance)

            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            else:
                show_warning(f"Skipping non-polygon geometry of type {type(geom).__name__}")
                continue

            for poly in polys:
                # Extract exterior coordinates (Y, X order for napari)
                coords = np.array(poly.exterior.coords[:-1])  # Remove last point (duplicate)
                # Swap X, Y to Y, X for napari
                shapes_list.append(coords[:, [1, 0]])
                shape_types.append('polygon')

        # Handle coloring
        if color_values is not None:
            # Replicate color values for multi-polygons
            expanded_colors = []
            for i, geom in enumerate(gdf.geometry):
                geom = geom.simplify(tolerance)
                if isinstance(geom, (Polygon, MultiPolygon)):
                    if isinstance(geom, MultiPolygon):
                        n_polys = len(list(geom.geoms))
                    else:
                        n_polys = 1
                    expanded_colors.extend([color_values[i]] * n_polys)

            colors, mapping, cmap = _data_to_rgba(
                data=expanded_colors,
                continuous_cmap=continuous_cmap,
                categorical_cmap=categorical_cmap,
                upper_climit_pct=upper_climit_pct
            )
            face_color = colors
        else:
            face_color = 'transparent'

        # Check if any shapes were created
        if len(shapes_list) == 0:
            show_warning("No valid polygon geometries found in SpatialUnitsData. Cannot create shapes layer.")
            return None

        # Prepare properties
        properties = {}
        if unit_names is not None:
            # Expand units names for multi-polygons
            expanded_names = []
            for i, geom in enumerate(gdf.geometry):
                geom = geom.simplify(tolerance)
                if isinstance(geom, (Polygon, MultiPolygon)):
                    if isinstance(geom, MultiPolygon):
                        n_polys = len(list(geom.geoms))
                    else:
                        n_polys = 1
                    expanded_names.extend([unit_names[i]] * n_polys)
            properties['unit_name'] = expanded_names

        if color_values is not None:
            properties['value'] = expanded_colors

        # Create layer tuple
        layer = (
            shapes_list,
            {
                'name': name,
                'properties': properties,
                'shape_type': shape_types,
                'edge_width': edge_width,
                'edge_color': 'white',
                'face_color': face_color,
                'opacity': opacity,
                'metadata': {'upper_climit_pct': upper_climit_pct}
            },
            'shapes'
        )
        return layer


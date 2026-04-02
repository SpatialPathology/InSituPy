from insitupy._version import __version__
from insitupy._constants import WITH_NAPARI

if WITH_NAPARI:
    import os
    from pathlib import Path
    from typing import Union

    import napari
    from geopandas import GeoDataFrame
    from napari.layers import Layer, Points, Shapes
    from napari.utils.notifications import show_info, show_warning
    from parse import parse
    from shapely import Point

    from insitupy.interactive._checks import _check_geometry_symbol_and_layer
    from insitupy.interactive._configs import _get_viewer_uid, config_manager
    from insitupy.utils.utils import convert_napari_shape_to_polygon_or_line

    def _get_current_viewer_config(action: str):
        viewer = napari.current_viewer()  # get the viewer that was open last
        if viewer is None:
            show_warning(
                f"No napari viewer open to {action}. First, use `.show()` to open a napari viewer."
            )
            return None, None

        viewer_id = _get_viewer_uid(viewer)
        try:
            config = config_manager[viewer_id]
        except KeyError:
            show_warning(
                "Could not find viewer configuration for the current napari viewer. "
                "Please reopen the viewer via `.show()` and try again."
            )
            return None, None

        return viewer, config

    def sync_geometries():
        """Synchronise annotation and region shapes from the active napari viewer back into the :class:`InSituData` object.

        Iterates all napari Shapes and Points layers whose names match the
        expected pattern, classifies each as an annotation or region, and
        writes the geometries into the corresponding :class:`~insitupy.containers.shapes_data.ShapesData`
        object.  Geometries present in the data but absent from the viewer are
        also removed.
        """
        new_pattern = "{type_symbol} {annot_key}"
        old_pattern = "{type_symbol} {class_name} ({annot_key})"

        # get current viewer config
        viewer, config = _get_current_viewer_config("synchronize geometries")
        if viewer is None:
            return

        data = config.data

        # iterate through layers and save them as annotation or region if they meet requirements
        layers = viewer.layers
        for layer in layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                name_parsed = parse(new_pattern, layer.name)
                if name_parsed is not None:
                    type_symbol = name_parsed.named["type_symbol"]
                    annot_key = name_parsed.named["annot_key"]

                    checks_passed, object_type = _check_geometry_symbol_and_layer(
                        layer=layer, type_symbol=type_symbol
                    )

                    if checks_passed:
                        shapesdata = data.annotations if object_type == "annotation" else data.regions

                        # import all geometries from viewer into ShapesData object within InSituData
                        _store_geometries(
                            layer=layer,
                            shapesdata=shapesdata,
                            object_type=object_type,
                            annot_key=annot_key,
                        )

                        # remove entries in InSituData that are not present in viewer
                        _remove_geometries(
                            layer=layer,
                            shapesdata=shapesdata,
                            config=config,
                            object_type=object_type,
                            annot_key=annot_key,
                        )
                else:
                    name_parsed_old = parse(old_pattern, layer.name)
                    if name_parsed_old is not None:
                        import warnings
                        warnings.warn(
                            f"Layer '{layer.name}' uses the old naming pattern. "
                            "Please re-add it via 'Show geometries' to use the new format.",
                            DeprecationWarning
                        )
                        type_symbol = name_parsed_old.named["type_symbol"]
                        annot_key = name_parsed_old.named["annot_key"]
                        class_name = name_parsed_old.named["class_name"]

                        checks_passed, object_type = _check_geometry_symbol_and_layer(
                            layer=layer, type_symbol=type_symbol
                        )

                        if checks_passed:
                            shapesdata = data.annotations if object_type == "annotation" else data.regions

                            _store_geometries(
                                layer=layer,
                                shapesdata=shapesdata,
                                object_type=object_type,
                                annot_key=annot_key,
                                class_name=class_name,
                            )

                            _remove_geometries(
                                layer=layer,
                                shapesdata=shapesdata,
                                config=config,
                                object_type=object_type,
                                annot_key=annot_key,
                                class_name=class_name,
                            )

    def save_colorlegends(
        output_folder: Union[str, os.PathLike, Path] = "figures",
        #savepath: Optional[Union[str, os.PathLike, Path]] = None,
        #from_canvas: bool = False,
        max_per_col: int = 10,
        save_only: bool = True
        ):
        """Save colour legends for all currently selected napari layers to PDF files.

        For each selected layer, a colour-legend figure is generated via
        :func:`~insitupy.plotting.plots.colorlegend` and saved to
        *output_folder* as ``colorlegend-<layer_name>.pdf``.

        Args:
            output_folder: Directory to write legend PDFs into.  Created
                automatically if it does not exist.
            max_per_col: Maximum number of legend entries per column.
            save_only: If True, save the figure without displaying it.
        """
        from insitupy.plotting.plots import colorlegend

        viewer, _ = _get_current_viewer_config("save color legends")
        if viewer is None:
            return

        # create output folder path
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        selected_layers = viewer.layers.selection
        for layer in selected_layers:
            savepath = output_folder / f"colorlegend-{layer.name}.pdf"

            plotted = colorlegend(
                viewer=viewer,
                mapping=None,
                layer_name=layer.name,
                max_per_col=max_per_col,
                savepath=savepath,
                save_only=save_only,
                verbose=False,
                return_status=True
                )

            if plotted:
                show_info(f"Saved color legend to '{savepath}'")

    def _remove_geometries(
        layer,
        shapesdata,
        config,
        object_type: str,
        annot_key: str,
        class_name: str = None
    ):
        # remove entries in InSituData that are not present in viewer
        current_ids = layer.features['uid'].values

        try:
            geom_df = shapesdata[annot_key]
        except KeyError:
            return

        if class_name is not None:
            unique_names = [class_name]
        elif 'name' in layer.features.columns:
            unique_names = layer.features['name'].unique()
        else:
            import warnings
            warnings.warn(
                f"Layer '{layer.name}' has no 'name' column in features — cannot determine "
                "which geometries to remove. Skipping removal for this layer.",
                RuntimeWarning
            )
            return

        for name in unique_names:
            ids_stored = geom_df[geom_df["name"] == name].index

            # filter geom_df and keep only those entries that are also present in viewer
            removal_mask = ~ids_stored.isin(current_ids)
            ids_to_remove = ids_stored[removal_mask]

            # remove only elements that were actively removed from the viewer
            ids_to_remove = [elem for elem in ids_to_remove if elem in config._removal_tracker]
            n_removed = len(ids_to_remove)

            # drop entries from geometries dataframe
            geom_df.drop(ids_to_remove, inplace=True)

            if n_removed > 0:
                object_str = object_type + "s" if n_removed > 1 else object_type
                show_info(f"Removed {n_removed} {object_str} with key {annot_key} and class {name}.")

    def _store_geometries(
        layer,
        shapesdata,
        object_type: str,
        annot_key: str,
        class_name: str = None,
        uid_col: str = "id"
        ):
        # extract shapes coordinates and colors
        layer_data = layer.data
        scale = layer.scale

        if isinstance(layer, Points):
            colors = layer.face_color.tolist()
        else:
            colors = layer.edge_color.tolist()

        name_values = layer.features['name'].values if 'name' in layer.features.columns else class_name

        if isinstance(layer, Shapes):
            # extract shape types
            shape_types = layer.shape_type
            # build annotation GeoDataFrame
            geom_df = {
                uid_col: layer.features["uid"].values,
                "objectType": object_type,
                "geometry": [convert_napari_shape_to_polygon_or_line(napari_shape_data=ar, shape_type=st) for ar, st in zip(layer_data, shape_types)],
                "name": name_values,
                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
            }

        elif isinstance(layer, Points):
            # build annotation GeoDataFrame
            geom_df = {
                uid_col: layer.features["uid"].values,
                "objectType": object_type,
                "geometry": [Point(d[1], d[0]) for d in layer_data],  # switch x/y
                "name": name_values,
                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
            }

        # generate GeoDataFrame
        geom_df = GeoDataFrame(geom_df, geometry="geometry")

        if len(geom_df) > 0:
            # add annotations
            shapesdata.add_data(
                data=geom_df,
                key=annot_key,
                scale_factor=scale[0],
                verbose=True,
                in_napari=True
                )
        else:
            show_info(f"No geometries found in layer {layer.name}.")
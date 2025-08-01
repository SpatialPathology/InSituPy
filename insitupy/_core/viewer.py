import os
from pathlib import Path
from typing import Optional, Union

import napari
import numpy as np
from geopandas import GeoDataFrame
from napari.utils.notifications import show_info, show_warning
from parse import parse
from shapely import Point

from insitupy import WITH_NAPARI, __version__
from insitupy._core._checks import _check_geometry_symbol_and_layer
from insitupy._core._configs import _get_viewer_uid
from insitupy._core.dataclasses import AnnotationsData, RegionsData
from insitupy.utils.utils import convert_napari_shape_to_polygon_or_line

if WITH_NAPARI:
    import napari
    from napari.layers import Layer, Points, Shapes
    from napari.utils.notifications import show_info, show_warning

    from insitupy._core._configs import config_manager

    def sync_geometries():
        name_pattern = "{type_symbol} {class_name} ({annot_key})"

        # get current viewer config
        viewer = napari.current_viewer() # get the viewer that was open last
        viewer_id = viewer.title.rsplit("#", 1)[1]
        config = config_manager[viewer_id]

        data = config.data

        if viewer is None:
            print("No napari viewer open to synchronize from. First, use `.show()` to open a napari viewer.")
            return

        # iterate through layers and save them as annotation or region if they meet requirements
        layers = viewer.layers
        for layer in layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                name_parsed = parse(name_pattern, layer.name)
                if name_parsed is not None:
                    type_symbol = name_parsed.named["type_symbol"]
                    annot_key = name_parsed.named["annot_key"]
                    class_name = name_parsed.named["class_name"]

                    checks_passed, object_type = _check_geometry_symbol_and_layer(
                        layer=layer, type_symbol=type_symbol
                    )

                    if checks_passed:
                        if object_type == "annotation":
                            # if the InSituData object does not have an annotations attribute, initialize it
                            # if data.annotations is None:
                            #     data.annotations = AnnotationsData() # initialize empty object

                            shapesdata = data.annotations
                        else:
                            # if the InSituData object does not have an regions attribute, initialize it
                            # if data.regions is None:
                            #     data.regions = RegionsData() # initialize empty object

                            shapesdata = data.regions

                        # import all geometries from viewer into ShapesData object within InSituData
                        _store_geometries(
                            layer=layer,
                            shapesdata=shapesdata,
                            object_type=object_type,
                            annot_key=annot_key,
                            class_name=class_name
                        )

                        # remove entries in InSituData that are not present in viewer
                        _remove_geometries(
                            layer=layer,
                            shapesdata=shapesdata,
                            config=config,
                            object_type=object_type,
                            annot_key=annot_key,
                            class_name=class_name
                        )

    def save_colorlegends(
        output_folder: Union[str, os.PathLike, Path] = "figures",
        #savepath: Optional[Union[str, os.PathLike, Path]] = None,
        #from_canvas: bool = False,
        max_per_col: int = 10,
        save_only: bool = True
        ):
        from insitupy.plotting.plots import plot_colorlegend

        viewer = napari.current_viewer()
        config = config_manager[_get_viewer_uid(viewer)]

        # create output folder path
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        selected_layers = viewer.layers.selection
        for layer in selected_layers:
            savepath = output_folder / f"colorlegend-{layer.name}.pdf"

            plotted = plot_colorlegend(
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

        # if from_canvas:
        #     # Check if static_canvas exists
        #     if not hasattr(config, 'static_canvas'):
        #         print("Warning: 'static_canvas' attribute not found in config. "
        #             "Please display data in the napari viewer using '.show()' first.")
        #         return

        #     try:
        #         # Save the figure to a PDF file
        #         config.static_canvas.figure.savefig(savepath)
        #         print(f"Figure saved as {savepath}")
        #     except RuntimeError as e:
        #         if 'FigureCanvasQTAgg has been deleted' in str(e):
        #             print("Warning: The color legend has been deleted and cannot be saved.")
        #         else:
        #             raise  # Re-raise the exception if it's a different error
        # else:


    def _remove_geometries(
        layer,
        shapesdata,
        config,
        object_type: str,
        annot_key: str,
        class_name: str
    ):
        # remove entries in InSituData that are not present in viewer
        current_ids = layer.properties['uid'] # get ids from current layer

        try:
            geom_df = shapesdata[annot_key]
        except KeyError:
            pass
        else:
            ids_stored = geom_df[geom_df["name"] == class_name].index

            # filter geom_df and keep only those entries that are also present in viewer
            removal_mask = ~ids_stored.isin(current_ids)
            ids_to_remove = ids_stored[removal_mask]

            # remove only elements that were actively removed from the viewer
            ids_to_remove = [elem for elem in ids_to_remove if elem in config._removal_tracker]
            n_removed = len(ids_to_remove)

            # drop entries from geometries dataframe
            geom_df.drop(
                ids_to_remove,
                inplace=True
                )

            if n_removed > 0:
                if n_removed > 1:
                    object_str = object_type + "s"
                else:
                    object_str = object_type

                show_info(f"Removed {n_removed} {object_str} with key {annot_key} and class {class_name}.")

    def _store_geometries(
        layer,
        shapesdata,
        object_type: str,
        annot_key: str,
        class_name: str,
        uid_col: str = "id"
        ):
        # extract shapes coordinates and colors
        layer_data = layer.data
        scale = layer.scale

        if isinstance(layer, Points):
            colors = layer.face_color.tolist()
        else:
            colors = layer.edge_color.tolist()

        if isinstance(layer, Shapes):
            # extract shape types
            shape_types = layer.shape_type
            # build annotation GeoDataFrame
            geom_df = {
                uid_col: layer.properties["uid"],
                "objectType": object_type,
                "geometry": [convert_napari_shape_to_polygon_or_line(napari_shape_data=ar, shape_type=st) for ar, st in zip(layer_data, shape_types)],
                "name": class_name,
                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
            }

        elif isinstance(layer, Points):
            # build annotation GeoDataFrame
            geom_df = {
                uid_col: layer.properties["uid"],
                "objectType": object_type,
                "geometry": [Point(d[1], d[0]) for d in layer_data],  # switch x/y
                "name": class_name,
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
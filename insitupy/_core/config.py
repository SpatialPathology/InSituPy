import dask
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from insitupy import WITH_NAPARI

if WITH_NAPARI:
    import napari

    def init_data_name():
        global current_data_name
        current_data_name = "main"

    def init_recent_selections():
        global recent_selections
        recent_selections = []

    # set viewer configurations
    def set_viewer_config(
        xdata,
        pixel_size_param = None
        ):
        if current_data_name == "main":
            # access adata, viewer and metadata from xeniumdata
            global adata
            adata = xdata.cells.matrix
            global boundaries
            boundaries = xdata.cells.boundaries
        else:
            adata = xdata.alt[current_data_name].matrix
            boundaries = xdata.alt[current_data_name].boundaries

        # get keys from var_names, obs and obsm
        global genes, observations, value_dict
        genes = adata.var_names.tolist()
        observations = adata.obs.columns.tolist()

        obsm_keys = list(adata.obsm.keys())
        obsm_cats = []
        for k in sorted(obsm_keys):
            data = adata.obsm[k]
            if isinstance(data, pd.DataFrame):
                for col in data.columns:
                    obsm_cats.append(f"{k}-{col}")
            elif isinstance(data, np.ndarray):
                for i in range(data.shape[1]):
                    obsm_cats.append(f"{k}-{i+1}")
            else:
                pass

        value_dict = {
            "genes": genes,
            "obs": observations,
            "obsm": obsm_cats
        }

        # get point coordinates
        global points
        points = np.flip(adata.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])

        # get expression matrix
        global X
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X

        global masks
        masks = []
        for n in boundaries.metadata.keys():
            b = getattr(boundaries, n)
            if isinstance(b, dask.array.core.Array) or np.all([isinstance(elem, dask.array.core.Array) for elem in b]):
                masks.append(n)

        # get image metadata
        global pixel_size
        if pixel_size_param is None:
            first_key = list(xdata.images.metadata.keys())[0]
            pixel_size = xdata.images.metadata[first_key]["pixel_size"]
        else:
            pixel_size = pixel_size_param

    def _refresh_widgets_after_data_change(xdata, points_widget, boundaries_widget):
        set_viewer_config(xdata)

        # set choices
        boundaries_widget.key.choices = masks

        # reset the currently selected key to None
        points_widget.value.value = None

        # add last addition to recent
        points_widget.recent.choices = sorted(recent_selections)
        points_widget.recent.value = None

        # set only the last layer visible
        point_layers = [elem for elem in xdata.viewer.layers if isinstance(elem, napari.layers.points.points.Points)]
        n_point_layers = len(point_layers)

        for i, l in enumerate(point_layers):
            if i < n_point_layers-1:
                l.visible = False
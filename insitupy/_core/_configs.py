from __future__ import \
    annotations  # this prevents circular imports of type hints such as InSituExperiment in this case

from typing import Dict
from uuid import uuid4

import dask
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from scipy.sparse import issparse

from insitupy import WITH_NAPARI


def _get_viewer_uid(viewer):
    return viewer.title.rsplit("#", 1)[1]

if WITH_NAPARI:
    class ViewerConfig:
        def __init__(
            self,
            data: InSituData
            ):
            self.data = data # required to import changes from Viewer into InSituData

            if data.cells is not None:
                self.data_name = data.cells.main_key # by default, main_key is the first data layer
                self.has_cells = True
            else:
                self.data_name = None
                self.has_cells = False

            #self.adata = xdata.cells[self.data_name].matrix
            #self.boundaries = xdata.cells[self.data_name].boundaries
            #self.viewer = data.viewer
            #self.genes = sorted(self.adata.var_names.tolist())
            #self.observations = sorted(self.adata.obs.columns.tolist())
            self.key_dict = self._build_key_dict()
            #self.points = np.flip(self.adata.obsm["spatial"].copy(), axis=1)
            #self.X = self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X
            self.masks = self._extract_masks()
            self.pixel_size = self._get_pixel_size()
            self.recent_selections = []
            self.static_canvas = FigureCanvas(Figure(figsize=(5, 5))) # static canvas for color legend


        @property
        def adata(self):
            """Return the AnnData object."""
            return self.data.cells[self.data_name].matrix

        @property
        def boundaries(self):
            return self.data.cells[self.data_name].boundaries

        @property
        def genes(self):
            """Return the gene names."""
            return sorted(self.adata.var_names.tolist())

        @property
        def observations(self):
            """Return the observation names."""
            return sorted(self.adata.obs.columns.tolist())

        @property
        def obsm(self):
            obsm_keys = list(self.adata.obsm.keys())
            obsm_cats = []
            for k in sorted(obsm_keys):
                data = self.adata.obsm[k]
                if isinstance(data, pd.DataFrame):
                    obsm_cats.extend([f"{k}#{col}" for col in data.columns])
                elif isinstance(data, np.ndarray):
                    obsm_cats.extend([f"{k}#{i+1}" for i in range(data.shape[1])])

            return obsm_cats

        @property
        def points(self):
            """Return the spatial coordinates of the points."""
            return np.flip(self.adata.obsm["spatial"].copy(), axis=1)

        @property
        def X(self):
            """Return the data matrix as a dense array."""
            if issparse(self.adata.X):
                return self.adata.X.toarray()
            return self.adata.X

        def _build_key_dict(self):
            return {
                "genes": self.genes,
                "obs": self.observations,
                "obsm": self.obsm
            }

        def _extract_masks(self):
            masks = []
            # for n, b in self.boundaries.metadata.items():
            #     if b is not None and (
            #         isinstance(b, dask.array.core.Array) or
            #         all(isinstance(elem, dask.array.core.Array) for elem in b)
            #     ):
            #         masks.append(n)

            boundaries = self.data.cells[self.data_name].boundaries

            for n in boundaries.metadata.keys():
                b = boundaries[n]
                if b is not None:
                    if isinstance(b, dask.array.core.Array) or np.all([isinstance(elem, dask.array.core.Array) for elem in b]):
                        masks.append(n)

            return masks

        def _get_pixel_size(self):
            if self.data.images is not None:
                first_key = list(self.data.images.metadata.keys())[0]
                return self.data.images.metadata[first_key]["pixel_size"]
            return None

    class ViewerConfigManager:
        def __init__(self):
            self._configs: Dict[str, ViewerConfig] = {}

        def add_config(self, data) -> str:
            """Create and store a new ViewerConfig instance with a unique ID."""
            uid = str(uuid4()).split("-")[0]
            self._configs[uid] = ViewerConfig(data)
            print(self._configs)
            return uid

        def __getitem__(self, config_id: str) -> ViewerConfig:
            """Allow dictionary-like access to ViewerConfig instances."""
            return self._configs[config_id]

        def list_configs(self) -> Dict[str, ViewerConfig]:
            """Return all stored ViewerConfig instances with their IDs."""
            return self._configs

        def __repr__(self) -> str:
            config_count = len(self._configs)
            config_ids = ', '.join(list(self._configs.keys())[:5])  # Show up to 5 IDs
            if config_count > 5:
                config_ids += ', ...'
            return f"<ViewerConfigManager with {config_count} configs: [{config_ids}]>"

    # initialize config manager only if it doesn't already exist
    if 'config_manager' not in globals():
        config_manager = ViewerConfigManager()
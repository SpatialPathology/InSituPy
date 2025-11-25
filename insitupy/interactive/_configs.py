from typing import Dict, List, Optional
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

        """
        ViewerConfig manages the configuration and data access for the InSituPy napari viewer.

        This class acts as a bridge between the viewer interface and the underlying InSituData,
        providing convenient access to AnnData matrices, spatial coordinates, gene and observation
        metadata, and image boundaries. It also manages viewer-specific state such as the currently
        selected data layer.

        Attributes:
            data (InSituData): The input data object containing single-cell spatial transcriptomics data.
            data_name (str or None): The key identifying the currently selected data layer.
            layer_name (str or None): The name of the selected layer ('main' or a layer key).
            has_cells (bool): Indicates whether cell data is available.
            static_canvas (FigureCanvas): A static canvas used for rendering legends or overlays.
            recent_selections (list): A list of recently selected items in the viewer.
            verbose (bool): Flag to enable verbose output.
            _removal_tracker (list): Internal tracker for removed elements.
            _auto_set_uid (bool): Flag to automatically set UIDs for added shapes.

        Properties:
            adata (AnnData): The AnnData object for the selected data layer.
            boundaries: The boundary data for the selected data layer.
            genes (list): Sorted list of gene names.
            observations (list): Sorted list of observation names.
            obsm (list): List of available obsm keys with subcategories.
            points (ndarray): Spatial coordinates of the cells.
            X (ndarray): Dense data matrix of gene expression values.
            key_dict (dict): Dictionary mapping data categories to their respective keys.
            masks (list): List of mask names extracted from the boundary metadata.
            pixel_size (float or None): The pixel size of the image, if available.
        """

        __slots__ = [
            'data',
            'data_name',
            'layer_name',
            'has_cells',
            'static_canvas',
            'recent_selections',
            'verbose',
            '_removal_tracker',
            '_auto_set_uid'
        ]

        def __init__(self, data):
            self.data = data

            if not data.cells.is_empty:
                self.data_name = data.cells.main_key
                self.layer_name = "main"
                self.has_cells = True
            else:
                self.data_name = None
                self.layer_name = None
                self.has_cells = False

            self.static_canvas = FigureCanvas(Figure(figsize=(5, 5)))
            self._removal_tracker = []
            self.recent_selections = []
            self.verbose = False
            self._auto_set_uid = True

        @property
        def adata(self):
            """Return the AnnData object for the selected data layer."""
            if not self.data.cells.is_empty:
                return self.data.cells[self.data_name].matrix
            return None

        @property
        def boundaries(self):
            """Return the boundary data for the selected data layer."""
            if not self.data.cells.is_empty:
                return self.data.cells[self.data_name].boundaries
            return None

        @property
        def genes(self) -> List[str]:
            """Return sorted list of gene names."""
            if self.adata is not None:
                return sorted(self.adata.var_names.tolist())
            return []

        @property
        def observations(self) -> List[str]:
            """Return sorted list of observation column names."""
            if self.adata is not None:
                return sorted(self.adata.obs.columns.tolist())
            return []

        @property
        def obsm(self) -> List[str]:
            """Return list of obsm keys with subcategories in format 'key#column'."""
            if self.adata is None:
                return []

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
        def points(self) -> Optional[np.ndarray]:
            """Return spatial coordinates with flipped axes for napari display."""
            if self.adata is not None:
                return np.flip(self.adata.obsm["spatial"].copy(), axis=1)
            return None

        @property
        def X(self) -> Optional[np.ndarray]:
            """Return the data matrix as a dense array."""
            if self.adata is None:
                return None

            X = self.adata.X if self.layer_name == "main" else self.adata.layers[self.layer_name]
            return X.toarray() if issparse(X) else X

        @property
        def key_dict(self) -> Dict[str, List[str]]:
            """Return dictionary mapping data categories to their keys."""
            return {
                "genes": self.genes,
                "obs": self.observations,
                "obsm": self.obsm
            }

        @property
        def masks(self) -> List[str]:
            """Return list of mask names from boundary metadata containing dask arrays."""
            boundaries = self.boundaries
            if boundaries is None:
                return []

            m = []
            for n in boundaries.metadata.keys():
                b = boundaries[n]
                if b is not None:
                    if isinstance(b, dask.array.core.Array) or np.all([isinstance(elem, dask.array.core.Array) for elem in b]):
                        m.append(n)

            return m

        @property
        def pixel_size(self) -> Optional[float]:
            """Return pixel size from image metadata, or None if no images available."""
            if self.data.images.is_empty:
                return None

            metadata_keys = list(self.data.images.metadata.keys())
            if not metadata_keys:
                return None

            first_key = metadata_keys[0]
            return self.data.images.metadata[first_key].get("pixel_size")

    class ViewerConfigManager:
        """
        Manages multiple ViewerConfig instances, each associated with a unique identifier.

        This class provides methods to create, store, retrieve, and list ViewerConfig
        objects, enabling organized access to multiple viewer configurations.

        Attributes:
            _configs (Dict[str, ViewerConfig]): A dictionary mapping unique IDs to ViewerConfig instances.

        Methods:
            add_config(data) -> str:
                Creates a new ViewerConfig from the given data and stores it with a unique ID.
            __getitem__(config_id: str) -> ViewerConfig:
                Retrieves a ViewerConfig by its unique ID using dictionary-like access.
            list_configs() -> Dict[str, ViewerConfig]:
                Returns all stored ViewerConfig instances with their associated IDs.
            __repr__() -> str:
                Returns a string representation summarizing the stored configurations.
        """

        __slots__ = ['_configs']

        def __init__(self):
            self._configs: Dict[str, ViewerConfig] = {}

        def add_config(self, data) -> str:
            """Create and store a new ViewerConfig instance with a unique ID."""
            uid = str(uuid4()).split("-")[0]
            self._configs[uid] = ViewerConfig(data)
            return uid

        def __getitem__(self, config_id: str) -> ViewerConfig:
            """Allow dictionary-like access to ViewerConfig instances."""
            return self._configs[config_id]

        def list_configs(self) -> Dict[str, ViewerConfig]:
            """Return all stored ViewerConfig instances with their IDs."""
            return self._configs

        def __repr__(self) -> str:
            config_count = len(self._configs)
            config_ids = ', '.join(list(self._configs.keys())[:5])
            if config_count > 5:
                config_ids += ', ...'
            return f"<ViewerConfigManager with {config_count} configs: [{config_ids}]>"

    if 'config_manager' not in globals():
        config_manager = ViewerConfigManager()
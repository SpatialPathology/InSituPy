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

        """
        ViewerConfig manages the configuration and data access for the InSituPy napari viewer.

        This class acts as a bridge between the viewer interface and the underlying InSituData,
        providing convenient access to AnnData matrices, spatial coordinates, gene and observation
        metadata, and image boundaries. It also manages viewer-specific state such as the currently
        selected data layer and cached variables for rendering.

        Attributes:
            data (InSituData): The input data object containing single-cell spatial transcriptomics data.
            data_name (str or None): The key identifying the currently selected data layer.
            has_cells (bool): Indicates whether cell data is available.
            static_canvas (FigureCanvas): A static canvas used for rendering legends or overlays.
            key_dict (dict): A dictionary mapping data categories to their respective keys.
            masks (list): A list of mask names extracted from the boundary metadata.
            pixel_size (float or None): The pixel size of the image, if available.
            recent_selections (list): A list of recently selected items in the viewer.

        Properties:
            adata (AnnData): The AnnData object for the selected data layer.
            boundaries: The boundary data for the selected data layer.
            genes (list): Sorted list of gene names.
            observations (list): Sorted list of observation names.
            obsm (list): List of available obsm keys with subcategories.
            points (ndarray): Spatial coordinates of the cells.
            X (ndarray): Dense data matrix of gene expression values.

        Methods:
            refresh_variables(): Updates internal variables such as key_dict, masks, and pixel size.
            update_data_name(new_data_name): Updates the selected data layer.
        """

        __slots__ = [
            'data',
            'data_name',
            'layer_name',
            'has_cells',
            'static_canvas',
            'key_dict',
            'masks',
            'pixel_size',
            'recent_selections',
            'verbose',
            '_removal_tracker',
            '_auto_set_uid'
        ] # using slots reduces he memory consumption and accelerates attribute access

        def __init__(self, data):
            self.data = data # required to import changes from Viewer into InSituData

            if not data.cells.is_empty:
                self.data_name = data.cells.main_key # by default, main_key is the first data layer
                self.layer_name = "main"
                self.has_cells = True
            else:
                self.data_name = None
                self.layer_name = None
                self.has_cells = False

            # canvas for static elements like color legends
            self.static_canvas = FigureCanvas(Figure(figsize=(5, 5))) # static canvas for color legend

            # list to track the removal of elements
            self._removal_tracker = []

            # set boolean variables
            self.verbose = False
            self._auto_set_uid = True # boolean switch to automatically set the uid of an added shape

            self.refresh_variables()

        @property
        def adata(self):
            if not self.data.cells.is_empty:
                """Return the AnnData object."""
                return self.data.cells[self.data_name].matrix
            else:
                return None

        @property
        def boundaries(self):
            if not self.data.cells.is_empty:
                return self.data.cells[self.data_name].boundaries
            else:
                return None

        @property
        def genes(self):
            if not self.adata is None:
                """Return the gene names."""
                return sorted(self.adata.var_names.tolist())
            else:
                return []

        @property
        def observations(self):
            if not self.adata is None:
                """Return the observation names."""
                return sorted(self.adata.obs.columns.tolist())
            else:
                return []

        @property
        def obsm(self):
            if not self.adata is None:
                obsm_keys = list(self.adata.obsm.keys())
                obsm_cats = []
                for k in sorted(obsm_keys):
                    data = self.adata.obsm[k]
                    if isinstance(data, pd.DataFrame):
                        obsm_cats.extend([f"{k}#{col}" for col in data.columns])
                    elif isinstance(data, np.ndarray):
                        obsm_cats.extend([f"{k}#{i+1}" for i in range(data.shape[1])])

                return obsm_cats
            else:
                return []

        @property
        def points(self):
            if not self.adata is None:
                """Return the spatial coordinates of the points."""
                return np.flip(self.adata.obsm["spatial"].copy(), axis=1)
            else:
                return None

        @property
        def X(self):
            if not self.adata is None:
                """Return the data matrix as a dense array."""
                if self.layer_name == "main":
                    X = self.adata.X
                else:
                    X = self.adata.layers[self.layer_name]

                if issparse(X):
                    return X.toarray()
                return X
            else:
                None

        def refresh_variables(self):
            self.key_dict = self._build_key_dict()
            self.masks = self._extract_masks()
            self.pixel_size = self._get_pixel_size()
            self.recent_selections = []

        # def update_data_name(self, new_data_name):
        #     self.data_name = new_data_name

        def _build_key_dict(self):
            return {
                "genes": self.genes,
                "obs": self.observations,
                "obsm": self.obsm
            }

        def _extract_masks(self):
            if not self.data.cells.is_empty:
                masks = []
                boundaries = self.data.cells[self.data_name].boundaries

                for n in boundaries.metadata.keys():
                    b = boundaries[n]
                    if b is not None:
                        if isinstance(b, dask.array.core.Array) or np.all([isinstance(elem, dask.array.core.Array) for elem in b]):
                            masks.append(n)

                return masks

        def _get_pixel_size(self):
            if not self.data.images.is_empty:
                first_key = list(self.data.images.metadata.keys())[0]
                return self.data.images.metadata[first_key]["pixel_size"]
            return None

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
            config_ids = ', '.join(list(self._configs.keys())[:5])  # Show up to 5 IDs
            if config_count > 5:
                config_ids += ', ...'
            return f"<ViewerConfigManager with {config_count} configs: [{config_ids}]>"

    # initialize config manager only if it doesn't already exist
    if 'config_manager' not in globals():
        config_manager = ViewerConfigManager()

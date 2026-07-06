from uuid import uuid4

import dask
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from scipy.sparse import issparse

from insitupy._constants import WITH_NAPARI


def _get_viewer_uid(viewer):
    return viewer.title.split(":", 1)[0].rsplit("#", 1)[1]

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
            'has_units',
            'static_canvas',
            'recent_selections',
            'verbose',
            '_removal_tracker',
            '_auto_set_uid',
            'key_dict',
            'masks',
            'pixel_size',
            'annot_point_colors',
            'region_colors',
            '_annot_point_color_idx',
            '_region_color_idx',
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

            # Check if units are available
            self.has_units = not data.units.is_empty

            # canvas for static elements like color legends
            self.static_canvas = FigureCanvas(Figure(figsize=(5, 5))) # static canvas for color legend

            # list to track the removal of elements
            self._removal_tracker = []
            self.recent_selections = []
            self.verbose = False
            self._auto_set_uid = True

            # colour registries for geometry layers: (key, name) -> hex colour
            self.annot_point_colors: dict = {}
            self.region_colors: dict = {}
            self._annot_point_color_idx: int = 0
            self._region_color_idx: int = 0

            # Initialize masks, key_dict, and pixel_size
            self.refresh_variables()

        @property
        def adata(self):
            """Return the AnnData object for the selected data layer."""
            if not self.data.cells.is_empty:
                """Return the AnnData object."""
                return self.data.cells[self.data_name].table
            else:
                return None

        @property
        def boundaries(self):
            """Return the boundary data for the selected data layer."""
            if not self.data.cells.is_empty:
                return self.data.cells[self.data_name].boundaries
            return None

        @property
        def genes(self) -> list[str]:
            """Return sorted list of gene names."""
            if self.adata is not None:
                return sorted(self.adata.var_names.tolist())
            return []

        @property
        def observations(self) -> list[str]:
            """Return sorted list of observation column names."""
            if self.adata is not None:
                return sorted(self.adata.obs.columns.tolist())
            return []

        @property
        def obsm(self) -> list[str]:
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
        def points(self) -> np.ndarray | None:
            """Return spatial coordinates with flipped axes for napari display."""
            if self.adata is not None:
                return np.flip(self.adata.obsm["spatial"].copy(), axis=1)
            return None

        @property
        def X(self) -> np.ndarray | None:
            """Return the data matrix as a dense array."""
            if self.adata is None:
                return None

            X = self.adata.X if self.layer_name == "main" else self.adata.layers[self.layer_name]
            return X.toarray() if issparse(X) else X

        @property
        def units(self):
            """Return the main SpatialUnitsData layer, if available."""
            if self.has_units:
                return self.data.units[self.data.units.main_key]
            return None

        @property
        def unit_vars(self):
            """Return variable names of spatial unit."""
            if self.has_units and self.units.table is not None:
                return sorted(self.units.table.var_names.tolist())
            else:
                return []

        @property
        def unit_obs(self):
            """Return observation names of spatial unit."""
            if self.has_units and self.units.table is not None:
                return sorted(self.units.table.obs.columns.tolist())
            else:
                return []

        @property
        def unit_obsm(self):
            """Return units obsm keys."""
            if self.has_units and self.units.table is not None:
                obsm_keys = list(self.units.table.obsm.keys())
                obsm_cats = []
                for k in sorted(obsm_keys):
                    fdata = self.units.table.obsm[k]
                    if isinstance(fdata, pd.DataFrame):
                        obsm_cats.extend([f"{k}#{col}" for col in fdata.columns])
                    elif isinstance(fdata, np.ndarray):
                        obsm_cats.extend([f"{k}#{i+1}" for i in range(fdata.shape[1])])
                return obsm_cats
            else:
                return []

        def refresh_variables(self):
            self.key_dict = self._build_key_dict()
            self.masks = self._extract_masks()
            self.pixel_size = self._get_pixel_size()
            self.recent_selections = []

        # def update_data_name(self, new_data_name):
        #     self.data_name = new_data_name

        def _build_key_dict(self):
            key_dict = {
                "genes": self.genes,
                "obs": self.observations,
                "obsm": self.obsm
            }
            if self.has_units:
                key_dict["unit_vars"] = self.unit_vars
                key_dict["unit_obs"] = self.unit_obs
                key_dict["unit_obsm"] = self.unit_obsm
            return key_dict

        def _extract_masks(self):
            if not self.data.cells.is_empty:
                masks = []
                boundaries = self.data.cells[self.data_name].boundaries

                for n in boundaries._data.keys():
                    b = boundaries[n]
                    if b is not None:
                        if isinstance(b, dask.array.core.Array) or np.all([isinstance(elem, dask.array.core.Array) for elem in b]):
                            masks.append(n)

                return masks

        def _get_pixel_size(self):
            if not self.data.images.is_empty:
                first_key = list(self.data.images._data.keys())[0]
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
            self._configs: dict[str, ViewerConfig] = {}

        def add_config(self, data) -> str:
            """Create and store a new ViewerConfig instance with a unique ID."""
            uid = str(uuid4()).split("-")[0]
            self._configs[uid] = ViewerConfig(data)
            return uid

        def __getitem__(self, config_id: str) -> ViewerConfig:
            """Allow dictionary-like access to ViewerConfig instances."""
            return self._configs[config_id]

        def list_configs(self) -> dict[str, ViewerConfig]:
            """Return all stored ViewerConfig instances with their IDs."""
            return self._configs

        def __repr__(self) -> str:
            config_count = len(self._configs)
            config_ids = ', '.join(list(self._configs.keys())[:5])
            if config_count > 5:
                config_ids += ', ...'
            return f"<ViewerConfigManager with {config_count} configs: [{config_ids}]>"

    config_manager = ViewerConfigManager()

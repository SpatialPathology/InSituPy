import os
from pathlib import Path
from typing import Dict, Literal, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData

try:
    from spatialdata import read_zarr
except ImportError:
    raise ImportError("Please install spatialdata with `pip install spatialdata`.")
else:
    from spatialdata.transformations import get_transformation

from insitupy._constants import MODALITIES_COLOR_DICT, SAMPLE_STR
from insitupy._textformat import textformat as tf
from insitupy.spatialdata._sdio import _silent_read_zarr

# --------------------
# Wrapper dataclasses
# --------------------


class StructuredImageData:
    def __init__(self):
        self._data: Dict[str, object] = {}
        self._metadata: Dict[str, object] = {}

    def __getitem__(self, key):
        dt = self._data.get(key) # get datatree
        dask_array_list = [dt[group].ds['image'].data for group in dt.groups if group != "/"]
        return dask_array_list

    def __repr__(self):
        if len(self._data) == 0:
            return "empty"
        return "\n".join([f"{tf.Bold}{k}{tf.ResetAll}\t{tuple(dict(v.scale0.dims).values())}" for k, v in self._data.items()])

    @property
    def is_empty(self):
        return len(self._data) == 0

    @property
    def metadata(self):
        return self._metadata

    def add_image(self, key, value, scale_obj):
        self._data[key] = value
        pixel_size = scale_obj.scale[scale_obj.axes.index('x')]
        axes = "".join(scale_obj.axes).upper()
        self._metadata[key] = {'pixel_size': pixel_size, 'axes': axes}

class StructuredBoundariesData:
    def __init__(self):
        self._data: Dict[str, object] = {}

    def __getitem__(self, key): return self._data.get(key)
    def __setitem__(self, key, value): self._data[key] = value

    def __repr__(self):
        if len(self._data) == 0:
            return "empty"
        repr_str = f"BoundariesData object with {len(self._data)} entries:"
        for k in self._data:
            repr_str += f"\n{tf.SPACER}{tf.Bold}{k}{tf.ResetAll}"
        return repr_str


class StructuredCellData:
    def __init__(self):
        self.matrix: Optional[AnnData] = None
        self.boundaries = StructuredBoundariesData()

    def __repr__(self):
        repr_str = ""
        if self.matrix is not None:
            repr_str += f"{tf.Bold+'matrix'+tf.ResetAll}\n{tf.SPACER}{self.matrix.__repr__()}"
        if len(self.boundaries._data) > 0:
            bound_repr = self.boundaries.__repr__().replace("\n", f"\n{tf.SPACER}")
            repr_str += f"\n{tf.Bold+'boundaries'+tf.ResetAll}\n{tf.SPACER}{bound_repr}"
        if repr_str == "":
            repr_str = "empty"
        return repr_str


class StructuredMultiCellData:
    def __init__(self):
        self._layers: Dict[str, StructuredCellData] = {}
        self._main_key: Optional[str] = None

    def __getitem__(self, key):
        return self._layers.get(key)

    def __setitem__(self, key, item: StructuredCellData):
        if isinstance(item, StructuredCellData):
            # check whether this is the first data that is added
            is_first_key = True if len(self._layers) == 0 else False

            # add data
            self._layers[key] = item

            # set key as main key if it is the first data to be added to the layer
            if is_first_key:
                self.main_key = key
        else:
            raise ValueError(f"Item must be of type StructuredCellData. Instead: {type(item)}.")

    def __repr__(self):
        if len(self._layers) == 0:
            return "empty"

        repr_str = f"{tf.Bold}MultiCellData with layers{tf.ResetAll}"
        if self._main_key:
            repr_str += f" (main='{self._main_key}')"

        for k, v in self._layers.items():
            indented = v.__repr__().replace("\n", f"\n{tf.SPACER}")
            repr_str += f"\n{tf.SPACER}{tf.Bold}{k}{tf.ResetAll}\n{tf.SPACER}{indented}"
        return repr_str

    @property
    def is_empty(self):
        return len(self._layers) == 0

    def keys(self):
        return self._layers.keys()

    @property
    def main_key(self):
        return self._main_key

    @main_key.setter
    def main_key(self, key):
        if key not in self._layers:
            raise KeyError(f"{key} not found in layers.")
        self._main_key = key

    @property
    def matrix(self):
        try:
            return self._layers[self._main_key].matrix
        except KeyError:
            print("MultiCellData object is empty.")
            return None
        except AttributeError:
            print("No matrix available.")
            return None

    @property
    def boundaries(self):
        try:
            return self._layers[self._main_key].boundaries
        except KeyError:
            print("MultiCellData object is empty.")
            return None
        except AttributeError:
            print("No boundaries available.")
            return None


class StructuredShapesData:
    def __init__(self, shape_name="shapes"):
        self._data: Dict[str, object] = {}
        self._shape_name = shape_name

    def __getitem__(self, key):
        return self._data.get(key)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self):
        if len(self._data) == 0:
            return "empty"
        return f"{self._shape_name} with keys: {list(self._data.keys())}"

    @property
    def is_empty(self):
        return len(self._data) == 0

    @property
    def metadata(self):
        """Compute metadata on-demand from current data state."""
        meta = {}
        for key, df in self._data.items():
            meta[key] = {
                f"n_{self._shape_name}": len(df),
                "classes": sorted(df['name'].unique().tolist()) if 'name' in df.columns else ["unnamed"],
            }
        return meta

    def keys(self):
        return self._data.keys()

class StructuredAnnotationsData(StructuredShapesData):
    def __init__(self):
        super().__init__(shape_name="annotations")


class StructuredRegionsData(StructuredShapesData):
    def __init__(self):
        super().__init__(shape_name="regions")


# --------------------
# StructuredSpatialData
# --------------------

class StructuredSpatialData:
    def __init__(
        self,
        path: Optional[Union[str, os.PathLike, Path]] = None
        ):
        self._path = Path(path) if path else None
        self._sdata = None

        # modalities
        self._images = StructuredImageData()
        self._cells = StructuredMultiCellData()
        self._annotations = StructuredAnnotationsData()
        self._regions = StructuredRegionsData()
        self._transcripts: Optional[pd.DataFrame] = None

        # if path is not None:
        #     self.read(path)

    def __repr__(self):
        repr_str = f"{tf.Bold+tf.Red}StructuredSpatialData{tf.ResetAll}\n"
        repr_str += f"{tf.Bold}Path:{tf.ResetAll}\t{self._path}\n"

        if len(self._images._data) == 0 and \
           len(self._cells._layers) == 0 and \
           len(self._annotations._data) == 0 and \
           len(self._regions._data) == 0 and \
           self._transcripts is None:
            repr_str += "\nNo modalities loaded."
            return repr_str

        if len(self._images._data) > 0:
            repr_str += f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['images']} images{tf.ResetAll}\n{tf.SPACER}   {list(self._images._data.keys())}"
        if len(self._cells._layers) > 0:
            repr_str += f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['cells']} cells{tf.ResetAll}\n{tf.SPACER}   {list(self._cells._layers.keys())}"
        if self._transcripts is not None:
            repr_str += f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['transcripts']} transcripts{tf.ResetAll}\n{tf.SPACER}   DataFrame with shape {self._transcripts.shape}"
        if len(self._annotations._data) > 0:
            repr_str += f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['annotations']} annotations{tf.ResetAll}\n{tf.SPACER}   {list(self._annotations._data.keys())}"
        if len(self._regions._data) > 0:
            repr_str += f"\n{tf.SPACER+tf.RARROWHEAD+MODALITIES_COLOR_DICT['regions']} regions{tf.ResetAll}\n{tf.SPACER}   {list(self._regions._data.keys())}"

        return repr_str

    # Properties
    @property
    def images(self): return self._images
    @property
    def cells(self): return self._cells
    @property
    def annotations(self): return self._annotations
    @property
    def regions(self): return self._regions
    @property
    def transcripts(self): return self._transcripts

    # Load from SpatialData
    @classmethod
    def read(cls, path: Union[str, Path]):
        path = Path(path)
        sdata = _silent_read_zarr(path)
        data = cls(path)

        for elem_type, key, elem in sdata.gen_elements():
            if key.startswith(SAMPLE_STR):
                raise ValueError("Multi-sample data is not supported in `StructuredSpatialData`. Use `InSituExperiment` instead.")

            parts = key.split(".")
            if parts[0] == "IMAGES":
                # self._images[parts[1]] = elem
                scale_obj = get_transformation(elem)
                data._images.add_image(parts[1], elem, scale_obj=scale_obj)
            elif parts[0] == "CELLS":
                cell_key = parts[1]
                if cell_key not in data._cells._layers:
                    data._cells[cell_key] = StructuredCellData()
                if parts[2] == "matrix":
                    data._cells[cell_key].matrix = elem
                elif parts[2] == "boundaries":
                    data._cells[cell_key].boundaries[parts[3]] = elem
            elif parts[0] == "ANNOTATIONS":
                data._annotations[parts[1]] = elem
            elif parts[0] == "REGIONS":
                data._regions[parts[1]] = elem
            elif parts[0] == "TRANSCRIPTS":
                data._transcripts = elem
            else:
                warn(f"Unrecognized element: {key}")

        return data

    def show(self,
        keys: Optional[str] = None,
        key_type: Literal["genes", "obs", "obsm"] = "genes",
        cells_layer: Optional[str] = None,
        point_size: int = 8,
        scalebar: bool = True,
        unit: str = "µm",
        return_viewer: bool = False,
        widgets_max_width: int = 500,
        verbose: bool = False
        ):
        # check whether napari is installed
        try:
            import napari

            from insitupy._core._napari import _show
        except ImportError:
            raise ImportError("Napari is not installed. Please install napari with `pip install napari[all]` to use this functionality.")

        _show(
            data=self,
            keys=keys,
            key_type=key_type,
            cells_layer=cells_layer,
            point_size=point_size,
            scalebar=scalebar,
            unit=unit,
            return_viewer=return_viewer,
            widgets_max_width=widgets_max_width,
            verbose=verbose
        )

from __future__ import annotations

import logging
import os
from numbers import Number
from pathlib import Path

from shapely import MultiPolygon, Polygon

from insitupy._io.files import check_overwrite_and_remove_if_true, write_dict_to_json
from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy._version import __version__

logger = logging.getLogger(__name__)


class MultiSpatialUnitsData(DeepCopyMixin):
    '''
    Data object containing multiple SpatialUnitsData objects.
    '''
    def __init__(self):
        self._layers: dict[str, SpatialUnitsData] = {}
        self._main_key: str | None = None

    @classmethod
    def read(cls, path):
        """Read MultiSpatialUnitsData from a saved directory.

        Args:
            path: Path to the MultiSpatialUnitsData directory.

        Returns:
            MultiSpatialUnitsData: The loaded MultiSpatialUnitsData object.
        """
        from insitupy.containers.io import _read_multispatialunitsdata
        return _read_multispatialunitsdata(path)

    def __len__(self):
        return len(self._layers)

    def __repr__(self):
        if len(self._layers) > 0:
            if self._main_key is not None:
                indented_repr = self._layers[self._main_key].__repr__().replace('\n', f'\n{tf.SPACER}')
                repr = (
                    f"{tf.Bold}MultiSpatialUnitsData with main layer{tf.ResetAll} '{self._main_key}'\n"
                    f"{tf.SPACER}{indented_repr}"
                )

            non_main_keys = [f"'{k}'" for k in self._layers.keys() if k != self._main_key]
            if len(non_main_keys) > 0:
                repr += f"\n\nAdditional layers with keys: {', '.join(non_main_keys)}"
        else:
            repr = "empty"
        return repr

    def __getitem__(self, key):
        if key not in self._layers:
            raise KeyError(key)
        return self._layers[key]

    def __contains__(self, key):
        return key in self.keys()

    def __setitem__(self, key: str, item):
        from .spatial_units_data import SpatialUnitsData
        if isinstance(item, SpatialUnitsData):
            # check whether this is the first data that is added
            is_first_key = True if len(self._layers) == 0 else False

            # add data
            self._layers[key] = item

            # set key as main key if it is the first data to be added to the layer
            if is_first_key:
                self.main_key = key
        else:
            raise TypeError(f"Item must be of type SpatialUnitsData. Instead: {type(item)}.")

    def __delitem__(self, key: str):
        if key in self._layers.keys():
            if key == self._main_key:
                raise KeyError(
                    f"Cannot delete the main key '{self._main_key}'. "
                    "Please use `set_main()` to set another key as main first."
                )
            del self._layers[key]
        else:
            raise KeyError(f"Key '{key}' not found in MultiSpatialUnitsData.")

    def keys(self):
        """Return the keys of all stored spatial units layers."""
        return self._layers.keys()

    @property
    def is_empty(self):
        """True if no spatial units layers have been added."""
        return len(self._layers) == 0

    @property
    def main_key(self):
        """Key of the currently active (main) spatial units layer."""
        return self._main_key

    @main_key.setter
    def main_key(self, value: str):
        """Set the main layer key.

        Raises:
            ValueError: If *value* is not an existing layer key.
        """
        if value not in self._layers.keys():
            raise ValueError("Such layer does not exist.")
        self._main_key = value

    def set_main(self, key):
        """Set the active (main) layer by key.

        Args:
            key: Key of the layer to promote to main.  Silently ignored if
                *key* is not present.
        """
        if key in self.keys():
            self._main_key = key

    @property
    def shapes(self):
        """Shapes GeoDataFrame of the main layer, or None if the object is empty."""
        try:
            return self._layers[self._main_key].shapes
        except KeyError:
            logger.warning("MultiSpatialUnitsData object is empty.")
            return None
        except AttributeError:
            logger.warning("No shapes available.")
            return None

    @property
    def table(self):
        """AnnData table of the main layer, or None if the object is empty."""
        try:
            return self._layers[self._main_key].table
        except KeyError:
            logger.warning("MultiSpatialUnitsData object is empty.")
            return None
        except AttributeError:
            logger.warning("No table available.")
            return None

    @property
    def unit_type(self):
        """Unit type of the main layer, or None if the object is empty."""
        try:
            return self._layers[self._main_key].unit_type
        except KeyError:
            logger.warning("MultiSpatialUnitsData object is empty.")
            return None
        except AttributeError:
            logger.warning("No unit_type available.")
            return None

    def add_units(self,
                  su,
                  key: str,
                  is_main: bool = False,
                  overwrite: bool = False):
        """Add a SpatialUnitsData layer to the MultiSpatialUnitsData object.

        Args:
            su: The SpatialUnitsData object to add.
            key: String key under which the layer is stored.
            is_main: If True, set this layer as the main (active) layer.
            overwrite: If True, allow replacing an existing layer with the
                same ``key``.  Raises ``KeyError`` when the key already
                exists and ``overwrite`` is False.

        Raises:
            TypeError: If ``su`` is not a SpatialUnitsData instance.
            KeyError: If ``key`` already exists and ``overwrite`` is False.
        """
        from .spatial_units_data import SpatialUnitsData
        if not isinstance(su, SpatialUnitsData):
            raise TypeError(f"su must be of type SpatialUnitsData. Instead: {type(su)}.")

        if key in self._layers.keys():
            if not overwrite:
                raise KeyError(
                    f"Key '{key}' already exists in MultiSpatialUnitsData. "
                    f"Set overwrite=True to replace it."
                )
            logger.info(f"Overwriting '{key}' in MultiSpatialUnitsData.")
        self._layers[key] = su
        if is_main:
            self._main_key = key

    def crop(self,
            xlim: tuple[Number, Number] | None = None,
            ylim: tuple[Number, Number] | None = None,
            shape: Polygon | MultiPolygon | None = None,
            inplace: bool = False,
            verbose: bool = True):
        """Crop all spatial units layers to a spatial bounding box or polygon.

        Delegates to :meth:`~insitupy.containers.spatial_units_data.SpatialUnitsData.crop`
        on each layer.  Either a *shape* or both *xlim* and *ylim* must be
        provided.

        Args:
            xlim: ``(x_min, x_max)`` bounding box in physical units.
            ylim: ``(y_min, y_max)`` bounding box in physical units.
            shape: Shapely polygon defining the crop region.  Takes
                precedence over *xlim* / *ylim* if provided.
            inplace: If True, modify this object in place; otherwise
                return a cropped copy.
            verbose: Passed to each layer's ``crop`` call.

        Returns:
            MultiSpatialUnitsData or None: Cropped copy when ``inplace=False``,
            otherwise None.
        """
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        for key in _self._layers.keys():
            _self._layers[key].crop(
                xlim=xlim,
                ylim=ylim,
                shape=shape,
                inplace=True,
                verbose=verbose)

        if not inplace:
            return _self

    def transform(self,
                  transformation_matrix,
                  source_pixel_size: Number | None = None,
                  reference_pixel_size: Number | None = None,
                  inplace: bool = False,
                  verbose: bool = False):
        """Apply an affine transformation to all spatial units layers.

        Delegates to :meth:`~insitupy.containers.spatial_units_data.SpatialUnitsData.transform`
        on each layer.

        Args:
            transformation_matrix: Either a 2x3 or 3x3 numpy array or a path to a CSV/Excel file.
            source_pixel_size: Pixel size (in um/pixel) of the source image.
            reference_pixel_size: Pixel size (in um/pixel) of the reference image.
            inplace: If True, modify this object in place; otherwise
                return a transformed copy.
            verbose: Passed to each layer's ``transform`` call.

        Returns:
            MultiSpatialUnitsData or None: Transformed copy when ``inplace=False``,
            otherwise None.
        """
        if inplace:
            _self = self
        else:
            _self = self.copy()

        for key in _self._layers.keys():
            _self._layers[key].transform(
                transformation_matrix=transformation_matrix,
                source_pixel_size=source_pixel_size,
                reference_pixel_size=reference_pixel_size,
                inplace=True,
                verbose=verbose)

        if not inplace:
            return _self

    def save(self,
             path: str | os.PathLike | Path,
             overwrite: bool = False):
        """Save all spatial units layers to a directory on disk.

        Each layer is saved into a subdirectory named after its key.
        A ``.multispatialunitsdata`` JSON file stores the main-key and
        layer-key metadata required to reload the object.

        Args:
            path: Output directory path.
            overwrite: If True, remove an existing directory at *path*
                before saving.  Defaults to False.
        """
        path = Path(path)
        musd_metadata = {"key_main": self._main_key, "all_keys": list(self._layers.keys())}

        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)
        for key in self._layers.keys():
            save_path = path / key
            self._layers[key].save(
                path=save_path,
                overwrite=overwrite)

        # add version to metadata
        musd_metadata["version"] = __version__

        # save metadata
        write_dict_to_json(dictionary=musd_metadata, file=path / ".multispatialunitsdata")

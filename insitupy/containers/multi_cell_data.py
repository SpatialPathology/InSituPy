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


class MultiCellData(DeepCopyMixin):
    '''
    Data object containing multiple CellData objects.
    '''
    def __init__(self):
        self._layers: dict[str, CellData] = dict()
        self._main_key: str | None = None

    @classmethod
    def read(cls, path, path_upper=None, alt_path_dict=None):
        """Read MultiCellData from a saved directory.

        Args:
            path: Path to the MultiCellData directory.
            path_upper: Optional upper path for alternative segmentations.
            alt_path_dict: Optional dictionary of alternative paths.

        Returns:
            MultiCellData: The loaded MultiCellData object.
        """
        from insitupy.containers.io import _read_multicelldata
        return _read_multicelldata(path, path_upper, alt_path_dict)

    def __len__(self):
        return len(self._layers)

    def __repr__(self):
        if len(self._layers) > 0:
            if self._main_key is not None:
                indented_repr = self._layers[self._main_key].__repr__().replace('\n', f'\n{tf.SPACER}')
                repr = (
                    f"{tf.Bold}MultiCellData with main layer{tf.ResetAll} '{self._main_key}'\n"
                    f"{tf.SPACER}{indented_repr}"
                )

            non_main_keys = [f"'{k}'" for k in self._layers.keys() if k != self._main_key]
            if len(non_main_keys) > 0:
                repr += f"\n\nAdditional layers with keys: {', '.join(non_main_keys)}"
        else:
            repr = "empty"
        return repr

    def __getitem__(self, key):
        return self._layers.get(key)

    def __contains__(self, key):
        return key in self.keys()

    def __setitem__(self, key: str, item):
        from .cell_data import CellData
        if isinstance(item, CellData):
            # check whether this is the first data that is added
            is_first_key = True if len(self._layers) == 0 else False

            # add data
            self._layers[key] = item

            # set key as main key if it is the first data to be added to the layer
            if is_first_key:
                self.main_key = key
        else:
            raise TypeError(f"Item must be of type CellData. Instead: {type(item)}.")

    def __delitem__(self, key: str):
        if key in self._layers.keys():
            if key == self._main_key:
                raise KeyError(f"Cannot delete the main key '{self._main_key}'. Please use `set_main()` to set another key as main first.")
            layer = self._layers[key]
            close_method = getattr(layer, "close", None)
            if callable(close_method):
                close_method()
            del self._layers[key]
        else:
            raise KeyError(f"Key '{key}' not found in MultiCellData.")

    def close(self):
        """Close underlying resources owned by all contained CellData layers."""
        for layer in self._layers.values():
            close_method = getattr(layer, "close", None)
            if callable(close_method):
                close_method()

    @property
    def layers(self):
        """Dict mapping layer keys to :class:`~insitupy.containers.cell_data.CellData` objects."""
        return self._layers

    @property
    def matrix(self):
        """Deprecated alias for the ``table`` of the main layer. Use ``table`` instead."""
        import warnings
        warnings.warn(
            "The 'matrix' property is deprecated and will be removed in a future version. "
            "Please use 'table' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return self._layers[self._main_key].table
        except KeyError:
            logger.warning("MultiCellData object is empty.")
            return None
        except AttributeError:
            logger.warning("No matrix available.")
            return None

    @property
    def table(self):
        """Alias for matrix property. This is the preferred name going forward."""
        try:
            return self._layers[self._main_key].table
        except KeyError:
            logger.warning("MultiCellData object is empty.")
            return None
        except AttributeError:
            logger.warning("No table available.")
            return None

    @property
    def boundaries(self):
        """Boundaries of the main layer, or None if the object is empty."""
        try:
            return self._layers[self._main_key].boundaries
        except KeyError:
            logger.warning("MultiCellData object is empty.")
            return None
        except AttributeError:
            logger.warning("No boundaries available.")
            return None

    @property
    def is_synced(self) -> bool:
        """True if all layers have their table and boundaries in sync."""
        if self.is_empty:
            return True

        return all(layer.is_synced for layer in self._layers.values())

    @property
    def main_key(self):
        """Key of the currently active (main) cell data layer."""
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

    @property
    def is_empty(self):
        """True if no cell data layers have been added."""
        return len(self._layers) == 0

    def add_celldata(self,
                     cd,
                     key: str,
                     is_main: bool = False,
                     overwrite: bool = False):
        """Add a CellData layer to the MultiCellData object.

        Args:
            cd: The CellData object to add.
            key: String key under which the layer is stored.
            is_main: If True, set this layer as the main (active) layer.
            overwrite: If True, allow replacing an existing layer with the
                same ``key``.  Raises ``KeyError`` when the key already
                exists and ``overwrite`` is False.

        Raises:
            TypeError: If ``cd`` is not a CellData instance.
            KeyError: If ``key`` already exists and ``overwrite`` is False.
        """
        from .cell_data import CellData
        if not isinstance(cd, CellData):
            raise TypeError(f"cd must be of type CellData. Instead: {type(cd)}.")

        if key in self._layers.keys():
            if not overwrite:
                raise KeyError(
                    f"Key '{key}' already exists in MultiCellData. "
                    f"Set overwrite=True to replace it."
                )
            logger.info(f"Overwriting '{key}' in MultiCellData.")
        self._layers[key] = cd
        if is_main:
            self._main_key = key

    def add_proseg(self,
                   path: str | os.PathLike | Path,
                   counts_file: str | None = None,
                   cell_metadata_file: str | None = None,
                   polygons_file: str | None = None,
                   pixel_size: Number = 1,
                   key: str = "proseg",
                   is_main: bool = False,
                   overwrite: bool = False
                   ):
        """
            Adds output of Proseg https://github.com/dcjones/proseg segmentation to the object.

            Args:
                path (Union[str, os.PathLike, Path]): Path to proseg output. Can be either:
                    - A directory containing individual files (counts, metadata, and polygon files) for legacy proseg output
                    - A .zarr directory containing a SpatialData object with proseg results.
                      The SpatialData object should contain:
                      * tables['table']: AnnData with counts and cell metadata
                      * shapes['cell_boundaries']: GeoDataFrame with cell polygons
                counts_file (Optional[str]): Name of the counts file. Only used with legacy directory input.
                cell_metadata_file (Optional[str]): Name of the cell metadata file. Only used with legacy directory input.
                polygons_file (Optional[str]): Name of the polygons file. Only used with legacy directory input.
                pixel_size (float): Size of the pixel for scaling.
                key (str, optional): Key to store the data. Defaults to "proseg".
                is_main (bool, optional): Flag to indicate if this is the main data. Defaults to False.
                overwrite (bool, optional): If True, allow overwriting an existing key. Defaults to False.
        """
        from ._segmentations import _read_proseg, _read_proseg_from_spatialdata
        from .boundaries_data import BoundariesData
        from .cell_data import CellData

        # Convert to Path object
        path = Path(path)

        # Check if this is a zarr file containing spatialdata
        if path.suffix == '.zarr' or path.name.endswith('.zarr'):
            # Read SpatialData from zarr
            try:
                import spatialdata
            except ImportError:
                raise ImportError(
                    "Reading proseg output from zarr requires the spatialdata package. "
                    "Please install with `pip install spatialdata`."
                )

            # File-specific parameters are ignored for spatialdata input
            if any([counts_file, cell_metadata_file, polygons_file]):
                logger.warning(
                    "File-specific parameters (counts_file, cell_metadata_file, polygons_file) "
                    "are ignored when reading from .zarr (SpatialData) format."
                )

            # Read spatialdata from zarr
            sdata = spatialdata.read_zarr(path)

            # Process spatialdata object
            adata, boundaries_mask, cell_names, seg_mask_value = _read_proseg_from_spatialdata(
                sdata,
                pixel_size=pixel_size
            )
            del sdata  # free spatialdata object before heavy allocations
        else:
            # Legacy path-based input (directory with individual files)
            adata, boundaries_mask, cell_names, seg_mask_value = _read_proseg(
                path, counts_file=counts_file, cell_metadata_file=cell_metadata_file,
                polygons_file=polygons_file, pixel_size=pixel_size
            )

        # generate boundaries data object
        boundaries = BoundariesData(
            cell_names=cell_names,
            seg_mask_value=seg_mask_value
            )

        # add cellular boundaries
        boundaries.add_boundaries(
            cell_boundaries=boundaries_mask,
            pixel_size=pixel_size
            )
        del boundaries_mask  # no longer needed after add_boundaries

        # Create cell data and add to object
        celldata = CellData(table=adata, boundaries=boundaries)
        del adata, boundaries  # now owned by celldata

        self.add_celldata(cd=celldata, key=key, is_main=is_main, overwrite=overwrite)


    def add_baysor(
                    self,
                    xd: str | os.PathLike | Path, # XeniumRanger output
                    path: str | os.PathLike | Path, # baysor output
                    counts_file: str | None = None,
                    cell_metadata_file: str | None = None,
                    polygons_file: str | None = None,
                    pixel_size: Number = 1,
                    key: str = "baysor",
                    is_main: bool = False,
                    overwrite: bool = False
                    ):
        """
        Add Baysor (https://github.com/kharchenkolab/Baysor) segmentation output to the object.

        Reads Baysor output files (counts, cell metadata, and polygon files)
        together with the XeniumRanger output directory and registers the
        resulting cell data as a new segmentation layer named ``key``.

        Args:
            xd (Union[str, os.PathLike, Path]): Path to the XeniumRanger output
                directory. Used alongside the Baysor output to reconstruct cell
                boundaries.
            path (Union[str, os.PathLike, Path]): Path to the Baysor output
                directory containing individual result files.
            counts_file (Optional[str], optional): Name of the counts file
                within ``path``. Defaults to None (auto-detected).
            cell_metadata_file (Optional[str], optional): Name of the cell
                metadata file within ``path``. Defaults to None (auto-detected).
            polygons_file (Optional[str], optional): Name of the polygons file
                within ``path``. Defaults to None (auto-detected).
            pixel_size (Number, optional): Size of one pixel in micrometers,
                used to scale coordinates. Defaults to 1.
            key (str, optional): Key under which the segmentation layer is
                stored. Defaults to ``'baysor'``.
            is_main (bool, optional): If True, set this layer as the main
                (active) cell segmentation layer. Defaults to False.
            overwrite (bool, optional): If True, allow overwriting an existing
                layer with the same ``key``. Defaults to False.

        Returns:
            None: Modifies the object in place by adding a new cell segmentation
                layer accessible via ``self[key]``.
        """

        from ._segmentations import _read_baysor
        from .boundaries_data import BoundariesData
        from .cell_data import CellData

        # Convert to Path object
        path = Path(path)
        logger.debug(path)
        # Legacy path-based input (directory with individual files)
        adata, boundaries_mask, cell_names, seg_mask_value = _read_baysor(path, xd,
            counts_file=counts_file, cell_metadata_file=cell_metadata_file,
            polygons_file=polygons_file, pixel_size=pixel_size
            )

        # generate boundaries data object
        boundaries = BoundariesData(
            cell_names=cell_names,
            seg_mask_value=seg_mask_value
            )

        # add cellular boundaries
        boundaries.add_boundaries(
            cell_boundaries=boundaries_mask,
            pixel_size=pixel_size
            )
        del boundaries_mask  # no longer needed after add_boundaries

        # Create cell data and add to object
        celldata = CellData(table=adata, boundaries=boundaries)
        del adata, boundaries  # now owned by celldata

        self.add_celldata(cd=celldata, key=key, is_main=is_main, overwrite=overwrite)


    def crop(self,
            xlim: tuple[int, int] | None = None,
            ylim: tuple[int, int] | None = None,
            shape: Polygon | MultiPolygon | None = None,
            inplace: bool = False,
            verbose: bool = True):
        """Crop all cell data layers to a spatial bounding box or polygon.

        Delegates to :meth:`~insitupy.containers.cell_data.CellData.crop` on
        each layer.  Either a *shape* or both *xlim* and *ylim* must be
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
            MultiCellData or None: Cropped copy when ``inplace=False``,
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

    def keys(self):
        """Return the keys of all stored cell data layers."""
        return self._layers.keys()

    def save(self,
             path: str | os.PathLike | Path,
             zipped: bool = False,
             overwrite: bool = False,
             max_resolution_boundaries: Number | None = None
             ):
        """Save all cell data layers to a directory on disk.

        Each layer is saved into a subdirectory named after its key.
        A ``.multicelldata`` JSON file stores the main-key and layer-key
        metadata required to reload the object.

        Args:
            path: Output directory path.
            zipped: If True, save boundary zarr stores as ``.zarr.zip``
                archives.  Defaults to False.
            overwrite: If True, remove an existing directory at *path*
                before saving.  Defaults to False.
            max_resolution_boundaries: Maximum spatial resolution for
                downsampling boundary pyramids.  If None, boundaries are
                saved at their original resolution.
        """
        path = Path(path)
        multicelldata_metadata = {"key_main": self._main_key, "all_keys": list(self._layers.keys())}
        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)
        for key in self._layers.keys():
            save_path = path / key
            self._layers[key].save(
                path=save_path,
                zipped=zipped,
                max_resolution_boundaries=max_resolution_boundaries,
                overwrite=overwrite)

        # add version to metadata
        multicelldata_metadata["version"] = __version__

        # save metadata
        write_dict_to_json(dictionary=multicelldata_metadata, file=path / ".multicelldata")

    def set_main(self, key):
        """Set the active (main) layer by key.

        Args:
            key: Key of the layer to promote to main.  Silently ignored if
                *key* is not present.
        """
        if key in self.keys():
            self._main_key = key

    def sync(self, return_summary: bool = False):
        """
        Synchronize table and boundaries across all layers in this MultiCellData object.

        Calls :meth:`CellData.sync` on each layer. You need to call this manually
        after directly mutating a layer's ``.table`` (e.g. filtering rows by index)
        without going through the normal InSituPy API, so that the boundaries stay
        consistent with the table.

        Args:
            return_summary (bool, optional): If True, return a dict mapping each
                layer key to the per-layer summary dict returned by
                :meth:`CellData.sync`. Defaults to False.

        Returns:
            None | dict: ``None`` normally. When ``return_summary=True``, a dict
                mapping layer keys to per-layer summary dicts (see
                :meth:`CellData.sync` for the structure of each summary dict).
        """
        current_keys = self._layers.keys()
        summaries = {}
        for key in current_keys:
            summary = self._layers[key].sync(return_summary=True)
            summaries[key] = summary
            if summary["had_boundaries"]:
                if summary["changed"]:
                    logger.info(
                        "MultiCellData.sync(): layer '%s' updated (%s table entries removed, %s boundary entries removed, reordered=%s).",
                        key,
                        summary["removed_table"],
                        summary["removed_boundaries"],
                        summary["reordered_boundaries"],
                    )
                else:
                    logger.info("MultiCellData.sync(): layer '%s' already aligned.", key)
            else:
                logger.info("MultiCellData.sync(): layer '%s' has no boundaries.", key)

        if return_summary:
            return summaries
        return None

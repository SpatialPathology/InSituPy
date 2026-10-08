from __future__ import annotations

import logging
import os
import warnings
from numbers import Number
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import MultiPoint, Point, Polygon, affinity

from insitupy._constants import FORBIDDEN_ANNOTATION_NAMES, RED, WITH_NAPARI
from insitupy._io.files import check_overwrite_and_remove_if_true
from insitupy._io.geo import parse_geopandas, write_qupath_geojson
from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)

if WITH_NAPARI:
    from napari.utils.notifications import show_info, show_warning


class ShapesData(DeepCopyMixin):
    """Object to store shape data (annotations, regions, or generic shapes).

    Args:
        files: Optional list of file paths to load on construction.
        keys: Keys to assign to each file.  Required when ``files`` is given.
        pixel_size: Scale factor (µm/pixel) applied to all geometries when
            loading from ``files``.  Required when ``files`` is given.
        assert_uniqueness: If True, reject a key whose name column contains
            duplicate values.
        polygons_only: If True, discard non-Polygon geometries on import.
        forbidden_names: Names that may not appear in the name column.
        shape_name: Human-readable label used in log messages and
            ``__repr__`` (e.g. ``"annotations"``, ``"regions"``).
        name_col: Column name that holds the class/category label of each
            shape.  Defaults to ``"name"``.
        color_col: Column name that holds the display colour of each shape.
            Defaults to ``"color"``.
        uid_col: Column name used as the row index (unique shape identifier).
            Defaults to ``"id"``.
    """
    def __init__(self,
                 files: list[str | os.PathLike | Path] | None = None,
                 keys: list[str] | None = None,
                 pixel_size: float | None = None,
                 assert_uniqueness: bool = False,
                 polygons_only: bool = False,
                 forbidden_names: list[str] | None = None,
                 shape_name: str | None = None,
                 name_col: str = "name",
                 color_col: str = "color",
                 uid_col: str = "id",
                 ) -> None:
        self._shape_name = shape_name if shape_name is not None else "shapes"
        self._name_col = name_col
        self._color_col = color_col
        self._uid_col = uid_col

        # add hidden variables
        self._data = {}
        self._assert_uniqueness = assert_uniqueness
        self._polygons_only = polygons_only
        self._forbidden_names = forbidden_names

        if files is not None:
            # make sure files and keys are a list
            if keys is None:
                raise ValueError("If `files` are given, also corresponding `keys` need to be given.")
            files = convert_to_list(files)
            keys = convert_to_list(keys)
            if len(files) != len(keys):
                raise ValueError("Number of files does not match number of keys.")

            if pixel_size is None:
                raise ValueError("If files and `keys` are given, also `pixel_size` needs to be specified")

            if files is not None:
                for key, file in zip(keys, files):
                    # read annotation and store in dictionary
                    self.add_data(data=file,
                                  key=key,
                                  scale_factor=pixel_size,
                                  uid_col=self._uid_col,
                                  )

    def __repr__(self):
        if len(self._data) > 0:
            repr_strings = []
            for l, m in self.metadata.items():
                # get metadata
                n = m[f"n_{self._shape_name}"]
                classes = m["names"]
                classes_str = [f"'{elem}'" for elem in classes]
                lc = len(classes)

                # create string
                r = (
                    f'{tf.Bold}{l}:{tf.ResetAll}\t{n} '
                    f'{self._shape_name}, {lc} '
                    f'{"names" if lc>1 else "name"} '
                )
                if lc < 10:
                    r += f'({", ".join(classes_str)})'
                repr_strings.append(r)

            s = "\n".join(repr_strings)
        else:
            s = "empty"

        return s

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def metadata(self):
        """Compute metadata on-demand from current data state."""
        meta = {}
        for key, df in self._data.items():
            meta[key] = {
                f"n_{self._shape_name}": len(df),
                "names": sorted(df[self._name_col].unique().tolist()) if self._name_col in df.columns else ["unnamed"],
            }
        return meta

    @property
    def is_empty(self):
        """True if no shape data has been added yet."""
        return len(self._data) == 0

    def _check_uniqueness(self,
                          dataframe: gpd.GeoDataFrame | None = None,
                          key: str | None = None,
                          verbose: bool = True,
                          name_col: str | None = None,
                          ) -> bool:
        """Check that every index entry maps to a unique name-column value.

        Args:
            dataframe: GeoDataFrame to check.  When ``None``, the stored
                layer for ``key`` is used.
            key: Key of the stored layer (used for the warning message and
                to look up the dataframe when ``dataframe`` is ``None``).
            verbose: If True, log a confirmation message when names are unique.
            name_col: Column name holding the class label.  Falls back to
                the instance default (``self._name_col``) when ``None``.

        Returns:
            True if the name column contains only unique values, False otherwise.
        """
        _name_col = name_col if name_col is not None else self._name_col

        if dataframe is None:
            annot_df = self[key]
        else:
            annot_df = dataframe

        if len(annot_df.index.unique()) != len(annot_df[_name_col].unique()):
            warnings.warn(
                    f"The names of the {self._shape_name} for key '{key}' were not unique and thus "
                    f"the key was skipped. In regions only one geometry per class is allowed."
                    f"To achieve this in the napari viewer select one layer per region and give each layer a unique name.",
                    UserWarning,
                    stacklevel=2
                )
            return False
        else:
            if verbose:
                logger.info(f"Names of {self._shape_name} for key '{key}' are unique.")
            return True

    def add_data(self,
                 data: gpd.GeoDataFrame | pd.DataFrame | dict | str | os.PathLike | Path,
                 key: str,
                 scale_factor: Number | None = None,
                 verbose: bool = False,
                 in_napari: bool = False,
                 uid_col: str | None = None,
                 name_col: str | None = None,
                 color_col: str | None = None,
                   ):
        """Add shape data from a file or dataframe to the ShapesData object.

        Parses the input into a GeoDataFrame, scales geometries by
        ``scale_factor``, and stores the result under ``key``.  If the
        key already exists, new shapes are appended and duplicates
        (by index) are resolved by keeping the latest entry.

        The ``name_col``, ``color_col``, and ``uid_col`` parameters act as
        **source column mappings**: they identify which columns in the input
        data carry the class label, colour, and unique ID respectively.  On
        import these columns are renamed to the canonical internal names
        (``self._name_col``, ``self._color_col``, ``self._uid_col``), so the
        stored dataframe always has a consistent structure regardless of the
        input naming convention.

        Args:
            data: Shape data as a GeoDataFrame, DataFrame, dict, or a
                file path to a GeoJSON / shapefile.
            key: String key under which the shapes are stored.
            scale_factor: Multiplicative factor applied to all geometry
                coordinates. Pass the pixel size in µm/pixel (e.g. ``0.2125``)
                when importing raw pixel-coordinate data from files. Pass
                ``1.0`` when the input is already in µm (e.g. data originating
                from another :class:`ShapesData` object). This argument is
                required — omitting it raises :exc:`ValueError`.
            verbose: If True, log a summary of added shapes.
            in_napari: If True, use napari notification functions for
                reporting instead of the logger.
            uid_col: Column name in the *input* data that holds the unique
                shape identifier.  Renamed to ``self._uid_col`` on import.
                Falls back to ``self._uid_col`` (typically ``"id"``) when
                ``None``.
            name_col: Column name in the *input* data that holds the
                class/category label.  Renamed to ``self._name_col`` on
                import.  Falls back to ``self._name_col`` (typically
                ``"name"``) when ``None``.
            color_col: Column name in the *input* data that holds the display
                colour.  Renamed to ``self._color_col`` on import.  Falls
                back to ``self._color_col`` (typically ``"color"``) when
                ``None``.
        """
        if scale_factor is None:
            raise ValueError(
                "scale_factor is required. Pass the pixel size in µm/pixel "
                "(e.g. 0.2125) if your data is in pixel coordinates, or pass "
                "1.0 if it is already in µm."
            )

        _uid_col = uid_col if uid_col is not None else self._uid_col
        _name_col = name_col if name_col is not None else self._name_col
        _color_col = color_col if color_col is not None else self._color_col

        # parse geopandas data from dataframe or file
        new_df = parse_geopandas(
            data,
            uid_col=_uid_col
            )

        if new_df is None:
            logger.warning(f"Data for key '{key}' was empty. Skipped import.")
        else:
            # Rename source columns to canonical internal names so the stored
            # dataframe always has a consistent structure (required by the I/O
            # layer, e.g. write_qupath_geojson).
            rename_map = {}
            if _name_col in new_df.columns and _name_col != self._name_col:
                rename_map[_name_col] = self._name_col
            if _color_col in new_df.columns and _color_col != self._color_col:
                rename_map[_color_col] = self._color_col
            if rename_map:
                new_df = new_df.rename(columns=rename_map)
            if new_df.index.name != self._uid_col:
                new_df.index.name = self._uid_col

            if self._name_col not in new_df.columns:
                new_df[self._name_col] = [None] * len(new_df)

            if self._color_col not in new_df.columns:
                warnings.warn(f"No '{_color_col}' column found in the imported data. Setting all colors to red.", UserWarning, stacklevel=2)
                new_df[self._color_col] = [RED] * len(new_df)

            if self._forbidden_names is not None:
                try:
                    new_names = new_df[self._name_col].tolist()
                except KeyError:
                    pass
                else:
                    if np.any([elem in new_names for elem in self._forbidden_names]):
                        raise ValueError(f"One of the forbidden names for annotations ({self._forbidden_names}) has been used in the imported dataset. Please change the respective change to prevent interference with downstream functions.")

            # convert geometries into unit (e.g. µm) values
            new_df["geometry"] = new_df["geometry"].scale(xfact=scale_factor, yfact=scale_factor, origin=(0,0))

            # determine the type of layer that needs to be used in napari later
            layer_types = []
            for geom in new_df["geometry"]:
                if isinstance(geom, Point) or isinstance(geom, MultiPoint):
                    layer_types.append("Points")
                else:
                    layer_types.append("Shapes")
            new_df["layer_type"] = layer_types

            if key not in self._data.keys():
                # if key does not exist yet, the new df is the whole annotation dataframe
                annot_df = new_df

                # collect additional variables for reporting
                new_geometries_added = True
                existing_str = ""
                old_n = 0
                new_n = len(annot_df)
            else:
                # concatenate old and new annoation dataframe
                annot_df = self[key]
                old_n = len(annot_df)
                annot_df = pd.concat([annot_df, new_df], ignore_index=False)

                # remove all duplicated shapes - leaving only the newly added
                dup_mask = annot_df.index.duplicated(keep="last")
                annot_df = annot_df[~dup_mask]
                new_n = len(annot_df)

                # collect additional variables for reporting
                new_geometries_added = new_n > old_n
                existing_str = "existing "

            if new_geometries_added:
                if self._assert_uniqueness:
                    # check if the shapes data for this key is unique
                    is_unique = self._check_uniqueness(dataframe=annot_df, key=key, verbose=False)

                    if not is_unique:
                        return

                if self._polygons_only:
                    # check if any of the shapes are not shapely Polygons
                    is_not_polygon = np.array([not isinstance(p, Polygon) for p in annot_df.geometry])
                    if np.any(is_not_polygon):
                        annot_df = annot_df.loc[~is_not_polygon]
                        show_warning(f"Some {self._shape_name} were not shapely.Polygon objects and skipped.")

            # check that the dataframe is not empty
            if len(annot_df) > 0:
                # add dataframe to ShapesData object
                self._data[key] = annot_df

                if verbose:
                    # report
                    if in_napari:
                        _show_func = show_info
                    else:
                        _show_func = logger.info
                    if new_geometries_added:
                        _show_func(f"Added {new_n - old_n} new {self._shape_name} to {existing_str}key '{key}'")
                    else:
                        _show_func(f"Updated {self._shape_name} to {existing_str}key '{key}'")

    def crop(self,
             shape,
             xlim,
             ylim,
             verbose: bool = True,
             inplace: bool = False
             ):
        """Crop shapes to a spatial region.

        Shapes that intersect with the cropping region are kept and their
        coordinates are shifted so that the crop origin becomes (0, 0).
        Keys whose shapes are entirely outside the region are removed.

        Either ``shape`` or both ``xlim`` and ``ylim`` must be provided.
        When ``shape`` is given it takes precedence.

        Args:
            shape: A Shapely geometry defining the crop region.  If None,
                a rectangle is constructed from ``xlim`` and ``ylim``.
            xlim: Tuple of (min_x, max_x) for rectangular cropping.
            ylim: Tuple of (min_y, max_y) for rectangular cropping.
            verbose: If True, log a warning when both xlim/ylim and shape
                are provided.
            inplace: If True, modify this object in place. Otherwise
                return a cropped copy.

        Returns:
            ShapesData or None: A new cropped ShapesData if
                ``inplace=False``, otherwise None.

        Raises:
            ValueError: If ``shape`` is None and either ``xlim`` or
                ``ylim`` is None.
        """
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        if shape is None:
            if (xlim is None) or (ylim is None):
                raise ValueError("If shape is None, both xlim and ylim must not be None.")
            else:
                shape = Polygon([(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])])
        else:
            if (xlim is not None) and (ylim is not None):
                if verbose:
                    logger.warning("Both xlim/ylim and shape are provided. Shape will be used for cropping.")

        keys_to_remove = []
        for key in list(_self._data.keys()):
            shapesdf = _self[key]

            # select annotations that intersect with the selected area
            mask = [shape.intersects(elem) for elem in shapesdf["geometry"]]
            shapesdf = shapesdf.loc[mask, :].copy()

            # move origin to zero after cropping
            shapesdf["geometry"] = shapesdf["geometry"].apply(affinity.translate, xoff=-xlim[0], yoff=-ylim[0])

            # check if there are annotations left or if it has to be deleted
            if len(shapesdf) > 0:
                # add new dataframe back to annotations object
                _self._data[key] = shapesdf
            else:
                # mark for deletion
                keys_to_remove.append(key)

        # delete empty keys
        for key in keys_to_remove:
            del _self._data[key]

        if not inplace:
            return _self

    def keys(self):
        """Return the keys of all stored shape layers."""
        return self._data.keys()

    def remove_key(
        self,
        key_to_remove: str,
        classes_to_remove: Literal["all"] | list[str] | str = "all"
        ):
        """Remove a shape layer or specific classes within it.

        Args:
            key_to_remove: Key of the shape layer to modify or delete.
            classes_to_remove: Which classes to remove.  ``"all"`` deletes
                the entire key; a string or list of strings removes only the
                rows whose name-column value (``self._name_col``) matches one
                of the given class names.
        """
        if classes_to_remove == "all":
            try:
                del self._data[key_to_remove]
            except KeyError:
                logger.warning(f"Key '{key_to_remove}' not found in ShapesData object. Nothing to remove.")
        else:
            classes_to_remove = convert_to_list(classes_to_remove)
            geom_df = self[key_to_remove]
            self._data[key_to_remove] = geom_df[~geom_df[self._name_col].isin(classes_to_remove)]

    @classmethod
    def read(cls, path, scale_factor=None):
        """Read ShapesData from a saved directory.

        Args:
            path: Path to the ShapesData directory.
            scale_factor: Optional scale factor for coordinates.

        Returns:
            ShapesData: The loaded ShapesData object.
        """
        from insitupy.containers.io import _read_shapesdata
        return _read_shapesdata(path, mode="shapes", scale_factor=scale_factor)

    def save(self,
             path: str | os.PathLike | Path,
             overwrite: bool = False
             ):
        """Save all shape layers to disk as GeoJSON files.

        Each key in the ShapesData object is written as a separate
        ``<key>.geojson`` file inside ``path``.

        Args:
            path: Directory to save into. Created if it does not exist.
            overwrite: If True, remove ``path`` first if it already
                exists.
        """
        path = Path(path)

        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # create directory
        path.mkdir(parents=True, exist_ok=True)

        # save each shape layer as geojson
        for key in self.keys():
            df = self[key]
            shapes_file = path / f"{key}.geojson"
            write_qupath_geojson(dataframe=df, file=shapes_file)


class AnnotationsData(ShapesData):
    """Container for spatial annotation shapes (non-exclusive, mixed geometry).

    Subclass of :class:`ShapesData` configured for annotations:
    duplicate names within a key are allowed, all geometry types (points,
    lines, polygons) are accepted, and a set of reserved names is
    forbidden to prevent conflicts with downstream analysis functions.

    Args:
        files: Optional list of file paths to load on construction.
        keys: Keys to assign to each file.  Required when ``files`` is given.
        pixel_size: Scale factor (µm/pixel) applied to all geometries when
            loading from ``files``.  Required when ``files`` is given.
        name_col: Column name that holds the class/category label of each
            shape.  Defaults to ``"name"``.
        color_col: Column name that holds the display colour of each shape.
            Defaults to ``"color"``.
        uid_col: Column name used as the row index (unique shape identifier).
            Defaults to ``"id"``.
    """
    def __init__(self,
                 files: list[str | os.PathLike | Path] | None = None,
                 keys: list[str] | None = None,
                 pixel_size: float | None = None,
                 name_col: str = "name",
                 color_col: str = "color",
                 uid_col: str = "id",
                 ) -> None:

        ShapesData.__init__(
            self,
            files=files,
            keys=keys,
            pixel_size=pixel_size,
            assert_uniqueness=False,
            polygons_only=False,
            forbidden_names=FORBIDDEN_ANNOTATION_NAMES,
            shape_name="annotations",
            name_col=name_col,
            color_col=color_col,
            uid_col=uid_col,
            )

    @classmethod
    def read(cls, path, scale_factor=None):
        """Read AnnotationsData from a saved directory.

        Args:
            path: Path to the AnnotationsData directory.
            scale_factor: Optional scale factor for coordinates.

        Returns:
            AnnotationsData: The loaded AnnotationsData object.
        """
        from insitupy.containers.io import _read_shapesdata
        return _read_shapesdata(path, mode="annotations", scale_factor=scale_factor)

    def to_regions(
        self,
        keys: str | list[str] | None = None,
        name_filter: str | list[str] | None = None,
    ) -> RegionsData:
        """Convert annotations to a new :class:`RegionsData` object.

        Only Polygon geometries are carried over.  Each key must have unique
        names after polygon filtering (and optional ``name_filter``) because
        ``RegionsData`` enforces uniqueness — a ``ValueError`` is raised if
        duplicates remain.

        Args:
            keys: Which annotation keys to include.  ``None`` converts all
                keys.
            name_filter: Keep only annotations whose name matches this value
                or list of values.  ``None`` keeps all names.

        Returns:
            RegionsData: A new object containing the converted regions.

        Raises:
            ValueError: If any key contains duplicate names after filtering.
        """
        target_keys = convert_to_list(keys) if keys is not None else list(self.keys())
        name_filter_list = convert_to_list(name_filter) if name_filter is not None else None

        result = RegionsData()
        for key in target_keys:
            if key not in self._data:
                warnings.warn(f"Key '{key}' not found in annotations — skipped.", UserWarning, stacklevel=2)
                continue

            df = self._data[key].copy()

            # Keep only Polygon geometries
            poly_mask = df.geometry.apply(lambda g: isinstance(g, Polygon))
            df = df.loc[poly_mask]
            if len(df) == 0:
                warnings.warn(f"Key '{key}' had no Polygon geometries — skipped.", UserWarning, stacklevel=2)
                continue

            # Optional name filter
            if name_filter_list is not None:
                df = df.loc[df[self._name_col].isin(name_filter_list)]
                if len(df) == 0:
                    warnings.warn(
                        f"Key '{key}' was empty after name_filter — skipped.",
                        UserWarning, stacklevel=2,
                    )
                    continue

            # Require unique names — RegionsData does not allow duplicates
            dup_names = [n for n, cnt in df[self._name_col].value_counts().items() if cnt > 1]
            if dup_names:
                raise ValueError(
                    f"Key '{key}' contains duplicate annotation names: {dup_names}. "
                    "RegionsData requires unique names per key. Rename or remove the "
                    "duplicate annotations before converting."
                )

            result.add_data(data=df, key=key, scale_factor=1.0)

        return result

    @classmethod
    def from_regions(
        cls,
        regions: RegionsData,
        keys: str | list[str] | None = None,
        on_forbidden: Literal["error", "rename", "skip"] = "error",
    ) -> AnnotationsData:
        """Construct an :class:`AnnotationsData` from a :class:`RegionsData`.

        Delegates to :meth:`RegionsData.to_annotations`.

        Args:
            regions: Source ``RegionsData`` object.
            keys: Which keys to include.  ``None`` converts all keys.
            on_forbidden: How to handle region names that clash with
                :data:`~insitupy._constants.FORBIDDEN_ANNOTATION_NAMES`.

        Returns:
            AnnotationsData: A new object containing the converted annotations.
        """
        return regions.to_annotations(keys=keys, on_forbidden=on_forbidden)


class RegionsData(ShapesData):
    """Container for spatial region shapes (unique names, polygons only).

    Subclass of :class:`ShapesData` configured for regions:
    each class name within a key must be unique, only
    :class:`~shapely.geometry.Polygon` geometries are accepted, and no
    forbidden-name list is applied.

    Args:
        files: Optional list of file paths to load on construction.
        keys: Keys to assign to each file.  Required when ``files`` is given.
        pixel_size: Scale factor (µm/pixel) applied to all geometries when
            loading from ``files``.  Required when ``files`` is given.
        name_col: Column name that holds the class/category label of each
            shape.  Defaults to ``"name"``.
        color_col: Column name that holds the display colour of each shape.
            Defaults to ``"color"``.
        uid_col: Column name used as the row index (unique shape identifier).
            Defaults to ``"id"``.
    """
    def __init__(self,
                 files: list[str | os.PathLike | Path] | None = None,
                 keys: list[str] | None = None,
                 pixel_size: float | None = None,
                 name_col: str = "name",
                 color_col: str = "color",
                 uid_col: str = "id",
                 ) -> None:

        ShapesData.__init__(
            self,
            files=files,
            keys=keys,
            pixel_size=pixel_size,
            assert_uniqueness=True,
            polygons_only=True,
            forbidden_names=None,
            shape_name="regions",
            name_col=name_col,
            color_col=color_col,
            uid_col=uid_col,
            )

    @classmethod
    def read(cls, path, scale_factor=None):
        """Read RegionsData from a saved directory.

        Args:
            path: Path to the RegionsData directory.
            scale_factor: Optional scale factor for coordinates.

        Returns:
            RegionsData: The loaded RegionsData object.
        """
        from insitupy.containers.io import _read_shapesdata
        return _read_shapesdata(path, mode="regions", scale_factor=scale_factor)

    def to_annotations(
        self,
        keys: str | list[str] | None = None,
        on_forbidden: Literal["error", "rename", "skip"] = "error",
    ) -> AnnotationsData:
        """Convert regions to a new :class:`AnnotationsData` object.

        Since ``RegionsData`` is already filtered to unique-named Polygons,
        the structural conversion is straightforward.  The only complication
        is region names that clash with
        :data:`~insitupy._constants.FORBIDDEN_ANNOTATION_NAMES`.

        Args:
            keys: Which region keys to include.  ``None`` converts all keys.
            on_forbidden: How to handle names that appear in
                ``FORBIDDEN_ANNOTATION_NAMES``.

                - ``"error"`` — raise ``ValueError`` listing the offending
                  names (default; safest choice).
                - ``"rename"`` — append ``"_region"`` suffix and warn.
                - ``"skip"`` — drop the offending rows and warn.

        Returns:
            AnnotationsData: A new object containing the converted annotations.
        """
        target_keys = convert_to_list(keys) if keys is not None else list(self.keys())
        forbidden = set(FORBIDDEN_ANNOTATION_NAMES)

        result = AnnotationsData()
        for key in target_keys:
            if key not in self._data:
                warnings.warn(f"Key '{key}' not found in regions — skipped.", UserWarning, stacklevel=2)
                continue

            df = self._data[key].copy()

            bad_names = [n for n in df[self._name_col].unique() if n in forbidden]
            if bad_names:
                if on_forbidden == "error":
                    raise ValueError(
                        f"Key '{key}' contains forbidden annotation names: {bad_names}. "
                        "Use on_forbidden='rename' or 'skip' to handle them automatically."
                    )
                elif on_forbidden == "rename":
                    rename_map = {n: f"{n}_region" for n in bad_names}
                    df[self._name_col] = df[self._name_col].replace(rename_map)
                    warnings.warn(
                        f"Renamed forbidden names in key '{key}': {rename_map}",
                        UserWarning, stacklevel=2,
                    )
                elif on_forbidden == "skip":
                    df = df.loc[~df[self._name_col].isin(bad_names)]
                    warnings.warn(
                        f"Dropped forbidden names from key '{key}': {bad_names}",
                        UserWarning, stacklevel=2,
                    )
                    if len(df) == 0:
                        warnings.warn(
                            f"Key '{key}' was empty after dropping forbidden names — skipped.",
                            UserWarning, stacklevel=2,
                        )
                        continue

            result.add_data(data=df, key=key, scale_factor=1.0)

        return result

    @classmethod
    def from_annotations(
        cls,
        annotations: AnnotationsData,
        keys: str | list[str] | None = None,
        name_filter: str | list[str] | None = None,
    ) -> RegionsData:
        """Construct a :class:`RegionsData` from an :class:`AnnotationsData`.

        Delegates to :meth:`AnnotationsData.to_regions`.

        Args:
            annotations: Source ``AnnotationsData`` object.
            keys: Which keys to include.  ``None`` converts all keys.
            name_filter: Keep only annotations with these names.

        Returns:
            RegionsData: A new object containing the converted regions.
        """
        return annotations.to_regions(keys=keys, name_filter=name_filter)

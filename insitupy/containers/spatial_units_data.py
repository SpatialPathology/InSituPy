from __future__ import annotations

import logging
import os
from numbers import Number
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from shapely import MultiPolygon, Polygon, affinity

from insitupy._io.files import check_overwrite_and_remove_if_true, write_dict_to_json
from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy._version import __version__

logger = logging.getLogger(__name__)


class SpatialUnitsData(DeepCopyMixin):
    """
    Object to store spatial units (e.g., functional tissue units, niches)
    with their associated omics data.

    Geometric information about the spatial units are stored as GeoDataFrames
    with polygon geometries, and their omics readouts are stored
    as AnnData objects. This provides flexibility for defining various spatial units beyond cells.

    Note: All coordinates in the geometries are assumed to be given as physical
    coordinates (usually um).
    """

    def __init__(
        self,
        shapes: gpd.GeoDataFrame | None,
        data: AnnData | None,
        unit_type: str = "unit"
    ):
        """
        Initialize SpatialUnitsData object.

        Args:
            shapes: GeoDataFrame containing polygon geometries for spatial units.
                Should have columns: 'geometry', 'name' (unit identifier),
                and optionally 'color', 'type', etc.
                All coordinates are assumed to be in physical units (usually um).
            data: AnnData object with omics readouts. obs_names should
                match unit names in the GeoDataFrame.
            unit_type: Description of unit type (e.g., 'niche', 'functional_unit').
        """
        self._shapes = shapes.copy() if shapes is not None else gpd.GeoDataFrame()
        self._data = data.copy()
        self._unit_type = unit_type

        # Convert Point geometries with radius to circles
        if not self._shapes.empty and 'radius' in self._shapes.columns:
            # Check if any geometries are Points
            point_mask = self._shapes.geometry.geom_type.isin(['Point', 'MultiPoint'])
            if point_mask.any():
                # Only convert Point geometries that have a valid (non-NA) radius
                radius_valid = ~self._shapes['radius'].isna()
                convert_mask = point_mask & radius_valid

                if convert_mask.any():
                    logger.info(f"Converting {convert_mask.sum()} Point geometries with radius to circular polygons using buffer.")
                    self._shapes.loc[convert_mask, 'geometry'] = self._shapes.loc[convert_mask].apply(
                        lambda row: row.geometry.buffer(row.radius), axis=1
                    )

        # Validate consistency if both features and data are provided
        if not self._shapes.empty and self._data is not None:
            self._validate_consistency()

            # rename feature index to match data.obs_names
            self._shapes.index = self._data.obs_names

    @classmethod
    def read(cls, path):
        """Read SpatialUnitsData from a saved directory.

        Args:
            path: Path to the SpatialUnitsData directory.

        Returns:
            SpatialUnitsData: The loaded object.
        """
        import json

        from anndata import read_h5ad
        path = Path(path)

        # Read metadata
        meta_file = path / "metadata.json"
        with open(meta_file) as f:
            meta = json.load(f)

        # Read shapes
        shapes = gpd.read_parquet(path / "shapes.parquet")

        # Read data if exists
        data_file = path / "data.h5ad"
        data = read_h5ad(data_file) if data_file.exists() else None

        return cls(shapes=shapes, data=data, unit_type=meta.get("unit_type", "unit"))

    def __repr__(self):
        n_units = len(self._shapes)
        has_data = self._data is not None

        if n_units > 0:
            repr_str = (
                f"{tf.Bold}SpatialUnitsData{tf.ResetAll} (Type: '{self._unit_type}')\n"
            )

            if has_data:
                repr_str += (
                    f"{tf.SPACER}.table: {self._data.n_obs} obs x "
                    f"{self._data.n_vars} vars\n"
                    f"{tf.SPACER}.shapes: {n_units} geometries"
                )
        else:
            repr_str = "Empty SpatialUnitsData object"

        return repr_str

    def __len__(self):
        return len(self._shapes)

    def __getitem__(self, key):
        """Subset SpatialUnitsData by unit indices or names."""
        new_obj = self.copy()

        if isinstance(key, (int, slice, list, np.ndarray, pd.Series)):
            new_obj._shapes = new_obj._shapes.iloc[key].copy()
        elif isinstance(key, str):
            # Assume string key is a unit name
            new_obj._shapes = new_obj._shapes[
                new_obj._shapes['name'] == key
            ].copy()
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

        # Sync data if present
        if new_obj._data is not None:
            unit_names = new_obj._shapes.index.tolist()
            new_obj._data = new_obj._data[unit_names, :].copy()

        return new_obj

    @property
    def shapes(self) -> gpd.GeoDataFrame:
        """GeoDataFrame containing geometries of `.shapes`."""
        return self._shapes

    @shapes.setter
    def shapes(self, value: gpd.GeoDataFrame):
        """Set the geometry GeoDataFrame.

        Raises:
            TypeError: If *value* is not a :class:`~geopandas.GeoDataFrame`.
        """
        if not isinstance(value, gpd.GeoDataFrame):
            raise TypeError(f"`.shapes` must be GeoDataFrame, not {type(value)}")
        self._shapes = value

    @property
    def table(self) -> AnnData | None:
        """AnnData object with omics readouts. This is the preferred name going forward."""
        return self._data

    @table.setter
    def table(self, value: AnnData | None):
        """Set the AnnData table. This is the preferred name going forward."""
        if value is not None and not isinstance(value, AnnData):
            raise TypeError(f"table must be AnnData object, not {type(value)}")
        self._data = value

    @property
    def unit_type(self) -> str:
        """Type of spatial units stored."""
        return self._unit_type

    @property
    def is_empty(self) -> bool:
        """True if the shapes GeoDataFrame contains no geometries."""
        return len(self._shapes) == 0

    def _validate_consistency(self):
        """Validate that shapes and data indices match."""
        if self._data is None:
            return

        unit_names = self._shapes.index
        data_names = self._data.obs_names

        if len(unit_names) != len(data_names):
            raise ValueError(
                f"Number of shapes ({len(unit_names)}) does not match "
                f"number of data obs ({len(data_names)})."
            )

        if not np.all(unit_names == data_names):
            logger.warning(
                "Indices in `.shapes` do not match `.data.obs_names`. Shapes will be renamed according to the `obs_names`. "
                "For this to be valid, please make sure that the order of elements in `.shapes` and `.data` matches."
            )

    def crop(
        self,
        xlim: tuple[Number, Number] | None = None,
        ylim: tuple[Number, Number] | None = None,
        shape: Polygon | MultiPolygon | None = None,
        inplace: bool = False,
        verbose: bool = True
    ):
        """
        Crop spatial units to a specified region.

        Args:
            xlim: X-axis limits (min, max).
            ylim: Y-axis limits (min, max).
            shape: Polygon/MultiPolygon to crop to. Takes precedence over xlim/ylim.
            inplace: Modify object in place.
            verbose: Print status messages.

        Returns:
            Cropped SpatialUnitsData if not inplace, else None.
        """
        _self = self if inplace else self.copy()

        # Create crop shape
        if shape is None:
            if xlim is None or ylim is None:
                raise ValueError("Must provide either shape or both xlim and ylim.")
            shape = Polygon([
                (xlim[0], ylim[0]), (xlim[1], ylim[0]),
                (xlim[1], ylim[1]), (xlim[0], ylim[1])
            ])
        else:
            if xlim is not None and ylim is not None and verbose:
                logger.warning("Both shape and xlim/ylim provided. Using shape.")
            # make sure there are no negative values in the limits, consistent
            # with InSituData.crop, which clips the region bounds before
            # cropping the images
            xlim = max(0.0, shape.bounds[0]), shape.bounds[2]
            ylim = max(0.0, shape.bounds[1]), shape.bounds[3]

        # Filter features that intersect
        mask = _self._shapes.geometry.intersects(shape)
        _self._shapes = _self._shapes[mask].copy()

        # Translate to origin
        _self._shapes["geometry"] = _self._shapes["geometry"].apply(
            affinity.translate, xoff=-xlim[0], yoff=-ylim[0]
        )

        # Crop data if present
        if _self._data is not None:
            feature_names = _self._shapes.index.tolist()
            _self._data = _self._data[feature_names, :].copy()

        logger.info(f"Cropped to {len(_self._shapes)} features.")

        if not inplace:
            return _self

    def sync(self, verbose: bool = False):
        """
        Synchronize spatial units and data to have matching indices.
        Keeps only units present in both.
        """
        if self._data is None:
            logger.info("No data to sync.")
            return

        unit_names = set(self._shapes.index)
        data_names = set(self._data.obs_names)
        common_names = unit_names & data_names

        # Filter units
        self._shapes = self._shapes.loc[list(common_names)]

        # Filter data
        self._data = self._data[list(common_names), :].copy()

        logger.info(f"Synced to {len(common_names)} common features.")

    def transform(
        self,
        transformation_matrix: np.ndarray | str | os.PathLike | Path,
        source_pixel_size: Number | None = None,
        reference_pixel_size: Number | None = None,
        inplace: bool = False,
        verbose: bool = False
    ):
        """Apply an affine transformation to all geometries in the SpatialUnitsData object.

        Args:
            transformation_matrix: Either a 2x3 or 3x3 numpy array or a path to a CSV/Excel file.
            source_pixel_size: Pixel size (in um/pixel) of the source image.
            reference_pixel_size: Pixel size (in um/pixel) of the reference image.
            inplace: If True, modify the object in place.
            verbose: If True, print status messages.

        Returns:
            SpatialUnitsData: Transformed SpatialUnitsData object if inplace=False, else None.
        """
        _self = self if inplace else self.copy()

        # Load transformation matrix if it's a file path
        if isinstance(transformation_matrix, (str, os.PathLike, Path)):
            transformation_matrix = Path(transformation_matrix)
            if not transformation_matrix.exists():
                raise FileNotFoundError(f"Transformation matrix file not found: {transformation_matrix}")

            # Read file based on extension
            if transformation_matrix.suffix.lower() in ['.csv', '.txt']:
                M = pd.read_csv(transformation_matrix, header=None).values
            elif transformation_matrix.suffix.lower() in ['.xlsx', '.xls']:
                M = pd.read_excel(transformation_matrix, header=None).values
            else:
                raise ValueError(f"Unsupported file format: {transformation_matrix.suffix}. Use .csv, .txt, .xlsx, or .xls")
        else:
            M = np.array(transformation_matrix)

        # Validate matrix dimensions
        if M.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Transformation matrix must be 2x3 or 3x3, got shape {M.shape}. "
                f"Expected format:\n"
                f"[[a, b, xoff],\n"
                f" [d, e, yoff]] or with [0, 0, 1] as third row."
            )

        # Extract transformation parameters
        if M.shape == (3, 3):
            # Validate that the third row is [0, 0, 1]
            if not np.allclose(M[2, :], [0, 0, 1]):
                raise ValueError("For 3x3 matrix, third row must be [0, 0, 1]")
            M = M[:2, :]

        # Convert pixel-based matrix to physical coordinates if reference_pixel_size is provided
        if reference_pixel_size is not None:
            M = M.copy().astype(np.float64)

            if source_pixel_size is not None:
                M[:2, :2] *= (reference_pixel_size / source_pixel_size)

            M[0, 2] *= reference_pixel_size  # Convert x offset: pixels -> um
            M[1, 2] *= reference_pixel_size  # Convert y offset: pixels -> um
            logger.info(f"Converted transformation matrix from pixel coordinates "
                        f"(reference: {reference_pixel_size} um/pixel) to physical coordinates.")

        # Apply transformation to geometries using shapely's affine_transform
        # Matrix format for shapely: [a, b, d, e, xoff, yoff]
        a, b, xoff = M[0, :]
        d, e, yoff = M[1, :]

        logger.info(f"Applying transformation (in physical coordinates): "
                    f"a={a}, b={b}, d={d}, e={e}, xoff={xoff}, yoff={yoff}")

        _self._shapes["geometry"] = _self._shapes["geometry"].apply(
            lambda geom: affinity.affine_transform(geom, [a, b, d, e, xoff, yoff])
        )

        logger.info(f"Transformed {len(_self._shapes)} features.")

        if not inplace:
            return _self

    def save(
        self,
        path: str | os.PathLike | Path,
        overwrite: bool = False
    ):
        """
        Save SpatialUnitsData to directory.

        Args:
            path: Output directory path.
            overwrite: If True, overwrite existing files.
        """
        path = Path(path)

        # Check overwrite
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)

        # Create directory
        path.mkdir(parents=True, exist_ok=True)

        # Save shapes as parquet (unconditional, matching what .read() expects)
        self._shapes.to_parquet(path / "shapes.parquet")

        # Save data as h5ad
        if self._data is not None:
            data_file = path / "data.h5ad"
            self._data.write_h5ad(data_file)

        # Save metadata
        metadata = {
            "version": __version__,
            "unit_type": self._unit_type,
        }
        write_dict_to_json(dictionary=metadata, file=path / "metadata.json")

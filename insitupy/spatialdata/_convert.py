try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    from spatialdata._core.validation import check_valid_name
    from spatialdata.models import (
        Image2DModel,
        Labels2DModel,
        PointsModel,
        ShapesModel,
        TableModel,
    )
    from spatialdata.transformations import Identity, Scale, get_transformation
    from spatialdata.transformations.transformations import Scale

import logging
from numbers import Number
from typing import Literal

import dask.dataframe as dd
import numpy as np
from anndata import AnnData
from xarray import DataArray

from insitupy._constants import (
    DEFAULT_CHUNK_SIZE_X,
    DEFAULT_CHUNK_SIZE_Y,
    MODALITIES,
    SAMPLE_STR,
)
from insitupy._core.data import InSituData
from insitupy.containers import BoundariesData
from insitupy.images.axes import ImageAxes
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)


def _generate_spatialdata_key(
    sample_id: str,
    modality: Literal[MODALITIES], # type: ignore
    locator: str | tuple | list | None
    ):
    # from spatialdata._core.validation import check_valid_name
    if modality.lower() not in MODALITIES:
        raise ValueError(f"Modality '{modality}' not recognized. Choose from {MODALITIES}.")

    if modality == "transcripts":
        if locator is not None:
            raise ValueError("Locator must be None for modality 'transcripts'.")
    else:
        if locator is None:
            raise ValueError(f"Locator cannot be None for modality '{modality}'.")

    if sample_id is None:
        sample_part = ""
    else:
        sample_part = f"{SAMPLE_STR}.{sample_id}.."

    if locator is not None:
        locator = convert_to_list(locator)

        locator_checked = []
        for elem in locator:
            try:
                check_valid_name(elem)
            except ValueError as e:
                raise ValueError(f"Name '{elem}' does not meet the naming requirements of SpatialData. {e}")

            if "." in elem or "-" in elem:
                logger.warning(f"Replacing '.' and '-' in '{elem}' with '_' to meet naming requirements.")
                elem = elem.replace(".", "_").replace("-", "_")

            locator_checked.append(elem)

        key = f"{sample_part}{modality.upper()}.{'.'.join(locator_checked)}"
    else:
        key = f"{sample_part}{modality.upper()}"

    # check key for validity
    check_valid_name(key)
    return key


def _transform_anndata_for_spatialdata(
    adata: AnnData,
    #cells_as_circles: bool = True
    cells_key: str,
    cell_area_key: str | None = "cell_area"
    ):
    """Convert an AnnData table to spatialdata-compatible format with circle shapes.

    Adds ``spatialdata_attrs`` metadata to ``adata.uns``, creates unit circles
    for all cells, and — if *cell_area_key* is present in ``obs`` — creates
    area-sized circles whose radii are derived from the cell area.

    Args:
        adata: AnnData with spatial coordinates in ``obsm["spatial"]``.
        cells_key: SpatialData element key used for the ``region`` attribute.
        cell_area_key: Column in ``adata.obs`` containing cell areas in
            squared pixels.  Used to compute per-cell circle radii.  Pass
            ``None`` to skip sized circles.

    Returns:
        A tuple ``(adata, circles_sized, circles)`` where *circles_sized* is
        a :class:`~geopandas.GeoDataFrame` of area-scaled circles (or ``None``)
        and *circles* is a GeoDataFrame of unit-radius circles.
    """
    # from spatialdata.models import ShapesModel
    adata = adata.copy()
    region_str = "region"
    attrs = {}
    #attrs["instance_key"] = "cell_id" if cells_as_circles else "cell_labels"
    attrs["instance_key"] = "cell_id"
    adata.obs["cell_id"] = adata.obs.index
    #attrs[region_str] = "cell_circles" if cells_as_circles else "cell_labels"
    attrs[region_str] = cells_key
    adata.obs[region_str] = cells_key
    adata.obs[region_str] = adata.obs[region_str].astype("category")
    attrs["region_key"] = region_str
    adata.uns["spatialdata_attrs"] = attrs

    if cell_area_key is not None:
        try:
            cell_areas = adata.obs[cell_area_key].to_numpy()
        except KeyError:
            logger.warning(f"Key '{cell_area_key}' not found in AnnData. Skipped generation of sized circles.")
            circles_sized = None
        else:
            radius = np.sqrt(cell_areas / np.pi)
            circles_sized = ShapesModel.parse(
                    adata.obsm["spatial"].copy(),
                    geometry=0, # means "Circles" (3 is Polygon, 6 is MultiPolygon)
                    radius=radius,
                    index=adata.obs.index.copy(),
            )
    else:
        circles_sized = None

    circles = ShapesModel.parse(
        adata.obsm["spatial"].copy(),
        geometry=0,
        radius=5,
        index=adata.obs.index.copy()
        )

    return adata, circles_sized, circles


def _transform_images_for_spatialdata(
    xd: InSituData,
    n_pyramids: int = 5,
    sample_id: str | None = None
    ):
    """Extract images from an :class:`InSituData` and parse them into SpatialData Image2DModel elements.

    Reads each named image from ``xd.images``, applies a pixel-size scale
    transformation, and parses the array into a :class:`~spatialdata.models.Image2DModel`
    with a multi-resolution pyramid.

    Args:
        xd: Source :class:`InSituData` object.
        n_pyramids: Number of pyramid levels to generate.
        sample_id: Optional prefix for the SpatialData element key.

    Returns:
        A dict mapping SpatialData element keys to parsed Image2DModel arrays.
    """
    # from spatialdata.models import Image2DModel
    # from spatialdata.transformations.transformations import Scale

    images = {}
    if xd.images is not None:
        for name in xd.images.names:
            image_list =  xd.images[name]
            meta = xd.images.metadata[name]
            pixel_size = meta["pixel_size"]
            transformations = {"global": Scale([pixel_size, pixel_size], axes=("x", "y"))}
            img = image_list[0]
            axes_isp = xd.images.metadata[name]["axes"]
            axes_config = ImageAxes(axes_isp)
            is_rgb = xd.images.metadata[name]["rgb"]

            dict_key = _generate_spatialdata_key(
                sample_id=sample_id,
                modality="images",
                locator=name
            )
            if is_rgb:
                axes = tuple(list(axes_isp.lower().replace("s", "c")))
                array = DataArray(
                    data=image_list[0],
                    name=name,
                    dims=axes
                    )
                images[dict_key] = Image2DModel.parse(
                    array,
                    rgb=True,
                    scale_factors=[2 for _ in range(n_pyramids)],
                    transformations=transformations,
                    chunks=(3, DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X)
                    )
            else:
                c_dim = axes_config.C if axes_config.C is not None else 1

                img_reshaped = img.reshape(
                    c_dim,
                    img.shape[axes_config.Y],
                    img.shape[axes_config.X]
                )
                array = DataArray(
                    data=img_reshaped,
                    name=name,
                    dims=("c", "y", "x")
                    )
                images[dict_key] = Image2DModel.parse(
                    array,
                    scale_factors=[2 for _ in range(n_pyramids)],
                    transformations=transformations,
                    chunks=(1, DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X)
                    )
    return images


def _transform_transcripts_for_spatialdata(
    xd: InSituData,
    sample_id: str | None = None
    ):
    """Parse transcript coordinates from an :class:`InSituData` into a SpatialData PointsModel element.

    Args:
        xd: Source :class:`InSituData` object with transcripts loaded.
        sample_id: Optional prefix for the SpatialData element key.

    Returns:
        A dict mapping a SpatialData element key to a parsed PointsModel
        DataFrame, or an empty dict if no transcripts are available.
    """
    # from spatialdata.models import PointsModel
    points = {}
    if xd.transcripts is not None:
        df = xd.transcripts
        if isinstance(df, dd.DataFrame):
            # Pre-compute known categories for the feature column. Without this,
            # PointsModel.parse(..., sort=True) treats the categories as unknown and
            # pays for an extra internal dask pass to determine them (measured ~2.25x
            # slower on a real 42.6M-row transcript table).
            df = df.assign(feature_name=df["feature_name"].astype("category").cat.as_known())
        parsed_points = PointsModel.parse(
            df,
            coordinates={"x": "x_location", "y": "y_location", "z": "z_location"},
            feature_key="feature_name",
            instance_key="cell_id",
            sort=True
            )

        dict_key = _generate_spatialdata_key(
            sample_id=sample_id,
            modality="transcripts",
            locator=None
        )

        points = {dict_key: parsed_points}
    return points


def _transform_table_for_spatialdata(
    xd: InSituData,
    sample_id: str | None = None
    ):
    """Convert cell AnnData tables and circle shapes from an :class:`InSituData` into SpatialData elements.

    For each cell layer with a loaded table, parses the AnnData into a
    :class:`~spatialdata.models.TableModel` and the corresponding cell
    positions into circle :class:`~spatialdata.models.ShapesModel` elements
    (both unit-radius and area-scaled variants when cell areas are available).

    Args:
        xd: Source :class:`InSituData` object.
        sample_id: Optional prefix for SpatialData element keys.

    Returns:
        A tuple ``(tables, cell_shapes)`` where both are dicts mapping
        SpatialData element keys to their parsed model objects.
    """
    # from spatialdata.models import TableModel
    tables, cell_shapes = {}, {}
    #if xd.cells is not None and xd.cells.table is not None:
    if xd.cells is not None:
        for cell_key in xd.cells.keys():
            if xd.cells[cell_key].table is not None:
                tables_key = _generate_spatialdata_key(
                    sample_id=sample_id,
                    modality="cells",
                    locator=[cell_key, "table"]
                )

                circles_dict_key = _generate_spatialdata_key(
                    sample_id=sample_id,
                    modality="cells",
                    locator=[cell_key, "circles"]
                )

                adata, circles_sized, circles = _transform_anndata_for_spatialdata(
                    xd.cells[cell_key].table,
                    cells_key=circles_dict_key
                    )

                # see https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/tables.html#construct-a-table-annotating-1-or-more-spatialelements
                tables[tables_key] = TableModel.parse(adata)
                cell_shapes[circles_dict_key] = circles

                if circles_sized is not None:
                    circles_sized_dict_key = _generate_spatialdata_key(
                        sample_id=sample_id,
                        modality="cells",
                        locator=[cell_key, "circles_sized"]
                    )

                    # add sized circles
                    cell_shapes[circles_sized_dict_key] = circles_sized

    return tables, cell_shapes


def _transform_units_for_spatialdata(
    xd: InSituData,
    sample_id: str | None = None
    ):
    """Convert spatial units tables and shapes from an :class:`InSituData` into SpatialData elements.

    For each units layer with a table, parses the AnnData into a
    :class:`~spatialdata.models.TableModel` and the corresponding polygon
    geometries into a :class:`~spatialdata.models.ShapesModel`. Unlike cells,
    units already carry real polygon geometries (see :class:`SpatialUnitsData`),
    so no circles are synthesized and no scale transform is applied.

    Args:
        xd: Source :class:`InSituData` object.
        sample_id: Optional prefix for SpatialData element keys.

    Returns:
        A tuple ``(tables, unit_shapes)`` where both are dicts mapping
        SpatialData element keys to their parsed model objects.
    """
    tables, unit_shapes = {}, {}
    if xd.units is not None:
        for unit_key in xd.units.keys():
            sud = xd.units[unit_key]
            if sud.table is not None and not sud.is_empty:
                shapes_key = _generate_spatialdata_key(
                    sample_id=sample_id,
                    modality="units",
                    locator=[unit_key, "shapes"]
                )
                table_dict_key = _generate_spatialdata_key(
                    sample_id=sample_id,
                    modality="units",
                    locator=[unit_key, "table"]
                )

                adata = sud.table.copy()
                adata.obs["unit_id"] = adata.obs.index
                adata.obs["region"] = shapes_key
                adata.obs["region"] = adata.obs["region"].astype("category")
                adata.uns["spatialdata_attrs"] = {
                    "region": shapes_key,
                    "region_key": "region",
                    "instance_key": "unit_id",
                }

                tables[table_dict_key] = TableModel.parse(adata)
                unit_shapes[shapes_key] = ShapesModel.parse(sud.shapes)
    return tables, unit_shapes


def _transform_cell_boundaries_for_spatialdata(
    xd: InSituData,
    n_pyramids: int = 5,
    sample_id: str | None = None
    ):
    """Convert cell boundary label arrays from an :class:`InSituData` into SpatialData Labels2DModel elements.

    For each cell layer with boundaries loaded, wraps the top-level label
    array in a :class:`~xarray.DataArray` and parses it into a
    :class:`~spatialdata.models.Labels2DModel` with a multi-resolution pyramid
    and a pixel-size scale transformation.

    Args:
        xd: Source :class:`InSituData` object.
        n_pyramids: Number of pyramid down-sampling levels.
        sample_id: Optional prefix for SpatialData element keys.

    Returns:
        A dict mapping SpatialData element keys to parsed Labels2DModel arrays.
    """
    # from spatialdata.models import Labels2DModel
    # from spatialdata.transformations.transformations import Scale

    cell_boundaries = {}
    #if xd.cells is not None and xd.cells.boundaries is not None:
    if xd.cells is not None:
        for cell_key in xd.cells.keys():
            if xd.cells[cell_key].boundaries is not None:
                celldata = xd.cells[cell_key]
                for name in celldata.boundaries.metadata.keys():
                    meta = celldata.boundaries.metadata[name]
                    pixel_size = meta["pixel_size"]
                    transformations = {"global": Scale([pixel_size, pixel_size], axes=("x", "y"))}

                    # get list of available boundaries
                    labels_list =  celldata.boundaries[name]

                    if isinstance(labels_list, list):
                        top_array = labels_list[0]
                    else:
                        top_array = labels_list
                    array = DataArray(data=top_array, name=name, dims=("y", "x"))

                    dict_key = _generate_spatialdata_key(
                        sample_id=sample_id,
                        modality="cells",
                        locator=[cell_key, "boundaries", name]
                    )

                    cell_boundaries[dict_key] = Labels2DModel.parse(
                        array,
                        scale_factors=[2 for _ in range(n_pyramids)],
                        transformations=transformations,
                        chunks=(DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X)
                        )
    return cell_boundaries


def _transform_annotations_for_spatialdata(
    xd: InSituData,
    sample_id: str | None = None
    ):
    """Parse annotation GeoDataFrames from an :class:`InSituData` into SpatialData ShapesModel elements.

    Args:
        xd: Source :class:`InSituData` object with annotations loaded.
        sample_id: Optional prefix for SpatialData element keys.

    Returns:
        A dict mapping SpatialData element keys to parsed ShapesModel GeoDataFrames,
        or an empty dict if no annotations are available.
    """
    # from spatialdata.models import ShapesModel
    shapes = {}
    if xd.annotations is not None:
        for key in xd.annotations.metadata.keys():
            gdf = ShapesModel.parse(
                xd.annotations[key],
                )

            dict_key = _generate_spatialdata_key(
                sample_id=sample_id,
                modality="annotations",
                locator=key
            )

            shapes[dict_key] = gdf
    return shapes


def _transform_regions_for_spatialdata(
    xd: InSituData,
    sample_id: str | None = None
    ):
    """Parse region GeoDataFrames from an :class:`InSituData` into SpatialData ShapesModel elements.

    Args:
        xd: Source :class:`InSituData` object with regions loaded.
        sample_id: Optional prefix for SpatialData element keys.

    Returns:
        A dict mapping SpatialData element keys to parsed ShapesModel GeoDataFrames,
        or an empty dict if no regions are available.
    """
    # from spatialdata.models import ShapesModel
    shapes = {}
    if xd.regions is not None:
        for key in xd.regions.metadata.keys():
            gdf = ShapesModel.parse(
                xd.regions[key],
                )
            dict_key = _generate_spatialdata_key(
                sample_id=sample_id,
                modality="regions",
                locator=key
            )

            shapes[dict_key] = gdf
    return shapes


def _merge_dicts_with_warning(*dicts):
    merged = {}
    seen_keys = set()
    for d in dicts:
        for key in d:
            if key in seen_keys:
                logger.warning(f"Duplicate key detected - '{key}'")
            seen_keys.add(key)
        merged.update(d)
    return merged


def _extract_pixel_size_from_spatialdata(
    sdata: SpatialData,
    # pixel_size: Optional[float],
    element_to_extract_from: str,
    coordinate_system: str | None = None,
    verbose: bool = True
) -> float | None:
    """Extract pixel size from SpatialData or use provided value."""

    # if pixel_size is not None:
    #     return pixel_size
    if element_to_extract_from is None:
        raise ValueError("Element to extract pixel size from must be specified.")

    # Try to extract from specified element
    if element_to_extract_from in sdata:
        try:
            transform = get_transformation(
                element=sdata[element_to_extract_from],
                to_coordinate_system=coordinate_system
                )

            if isinstance(transform, Scale):
                ps = 1 / transform.scale[0].item()
            elif isinstance(transform, Identity):
                ps = 1.0
            else:
                raise ValueError(f"Transformation type '{type(transform)}' not supported for pixel size extraction.")

            if verbose:
                logger.info(f"Extracted pixel size {ps} from '{element_to_extract_from}'")
            return ps

        except Exception as e:
            if verbose:
                logger.warning(f"Could not extract pixel size from '{element_to_extract_from}': {e}")

    else:
        raise ValueError(f"Element '{element_to_extract_from}' not found in SpatialData for pixel size extraction")

    # # Try to extract from any available element
    # for elem_type, key, elem in sdata.gen_elements():
    # # for key in sdata.keys():
    #     try:
    #         transform = get_transformation(sdata[key])
    #         ps = 1 / transform.scale[0].item()
    #         if verbose:
    #             print(f"Extracted pixel size {ps} from '{key}'", flush=True)
    #         return ps
    #     except:
    #         continue

    # if verbose:
    #     print("Warning: Could not extract pixel size from any element", flush=True)
    # return None


def _add_images_to_insitudata(
    data: InSituData,
    sdata: SpatialData,
    image_data: tuple[str, Number] | tuple[str, Number, bool] | list[tuple[str, Number] | tuple[str, Number, bool]] | dict[str, tuple[str, Number] | tuple[str, Number, bool]],
    # pixel_size: Optional[float],
    verbose: bool
):
    """Add images to InSituData.

    Args:
        data: InSituData object to add images to.
        sdata: SpatialData object containing the images.
        image_data: Image data in one of the supported formats:
            - Single tuple: (image_key, pixel_size) or (image_key, pixel_size, is_rgb)
            - List of tuples: [(image_key, pixel_size), ...] or [(image_key, pixel_size, is_rgb), ...]
            - Dictionary: {name: (image_key, pixel_size), ...} or {name: (image_key, pixel_size, is_rgb), ...}
            The optional is_rgb flag (default: False) indicates if the image is RGB.
        verbose: Whether to print status messages.
    """

    # Normalize to dict format
    if isinstance(image_data, tuple):
        image_dict = {image_data[0]: image_data}
    elif isinstance(image_data, list):
        image_dict = {t[0]: t for t in image_data}
    else:
        image_dict = image_data

    for name, image_tuple in image_dict.items():
        key = image_tuple[0]
        pixel_size = image_tuple[1]
        is_rgb = image_tuple[2] if len(image_tuple) > 2 else False
        if key not in sdata:
            if verbose:
                logger.warning(f"Image key '{key}' not found in SpatialData")
            continue
        img_data = sdata[key]
        try:
            data_array = img_data.scale0['image']
        except AttributeError:
            data_array = img_data

        # get information about axis configuration
        axes_str = "".join(data_array.dims).upper()
        if is_rgb:
            axes_str = axes_str.replace("C", "S")

        axes = ImageAxes(axes_str)

        da_img = data_array.data

        if da_img.shape[0] == 1:
            # if the channel axis has length 1, remove it
            da_img = da_img.squeeze(axes.C)
            axes_str = axes_str[1:]
            axes = ImageAxes(axes_str)

        channel_names = name
        if not is_rgb and axes.C is not None and da_img.shape[axes.C] > 1:
            channel_names = [f"{name}_{i}" for i in range(da_img.shape[axes.C])]

        data.images.add_image(
            image=da_img,
            channel_names=channel_names,
            axes=axes_str,
            pixel_size=pixel_size,
            overwrite=False,
            verbose=verbose
        )


def _create_boundaries_from_spatialdata(
    sdata: SpatialData,
    cell_names: np.ndarray,
    cell_boundaries_data: tuple[str, Number] | None = None, # tuple as (cell_boundaries_key, pixel_size)
    nucleus_boundaries_data: tuple[str, Number] | None = None, # tuple as (nucleus_boundaries_key, pixel_size)
    ) -> BoundariesData:
    """Create BoundariesData from SpatialData labels."""

    if cell_boundaries_data[1] != nucleus_boundaries_data[1]:
        raise ValueError("Pixel sizes for cell boundaries and nucleus boundaries must be the same.")
    else:
        pixel_size = cell_boundaries_data[1]

    # Generate seg_mask_value
    logger.warning("For the segmentation mask values of the boundaries, it is assumed that the order of the cells matches the ascending values of the segmentation mask.")
    seg_mask_value = np.array(range(1, len(cell_names) + 1))

    boundaries = BoundariesData(
        cell_names=cell_names,
        seg_mask_value=seg_mask_value
    )

    # Add cell boundaries if provided
    cell_bounds = None
    cell_boundaries_key = cell_boundaries_data[0]
    if cell_boundaries_key and cell_boundaries_key in sdata:
        cell_bounds = sdata[cell_boundaries_key].scale0['image'].data

    # Add nucleus boundaries if provided
    nuc_bounds = None
    nucleus_boundaries_key = nucleus_boundaries_data[0]
    if nucleus_boundaries_key and nucleus_boundaries_key in sdata:
        nuc_bounds = sdata[nucleus_boundaries_key].scale0['image'].data

    if cell_bounds is not None or nuc_bounds is not None:
        boundaries.add_boundaries(
            cell_boundaries=cell_bounds,
            nuclei_boundaries=nuc_bounds,
            pixel_size=pixel_size
        )

    return boundaries

def _validate_image_data_format(
    image_data: tuple[str, Number] | tuple[str, Number, bool] | list[tuple[str, Number] | tuple[str, Number, bool]] | dict[str, tuple[str, Number] | tuple[str, Number, bool]] | None
) -> None:
    """
    Validate the format of image_data parameter.

    Args:
        image_data: Image data in one of the supported formats:
            - Single tuple: (image_key, pixel_size) or (image_key, pixel_size, is_rgb)
            - List of tuples: [(image_key, pixel_size), ...] or [(image_key, pixel_size, is_rgb), ...]
            - Dictionary: {name: (image_key, pixel_size), ...} or {name: (image_key, pixel_size, is_rgb), ...}

    Raises:
        ValueError: If the format structure is invalid.
        TypeError: If element types are incorrect.
    """
    if image_data is None:
        return

    def _validate_tuple(t, context=""):
        """Validate a single image tuple."""
        if len(t) not in (2, 3):
            raise ValueError(f"{context}tuple must have 2 or 3 elements (image_key, pixel_size[, is_rgb]), got {len(t)}")
        if not isinstance(t[0], str):
            raise TypeError(f"{context}image_key must be a string, got {type(t[0])}")
        if not isinstance(t[1], Number):
            raise TypeError(f"{context}pixel_size must be a number, got {type(t[1])}")
        if len(t) == 3 and not isinstance(t[2], bool):
            raise TypeError(f"{context}is_rgb must be a boolean, got {type(t[2])}")

    if isinstance(image_data, tuple):
        # Single tuple: (image_key, pixel_size) or (image_key, pixel_size, is_rgb)
        _validate_tuple(image_data, "image_data ")
    elif isinstance(image_data, list):
        # List of tuples
        for i, item in enumerate(image_data):
            if not isinstance(item, tuple):
                raise ValueError(f"image_data list must contain tuples, got {type(item)} at index {i}")
            _validate_tuple(item, f"image_data[{i}] ")
    elif isinstance(image_data, dict):
        # Dictionary: {name: (image_key, pixel_size[, is_rgb])}
        for name, value in image_data.items():
            if not isinstance(value, tuple):
                raise ValueError(f"image_data['{name}'] must be a tuple, got {type(value)}")
            _validate_tuple(value, f"image_data['{name}'] ")
    else:
        raise TypeError(
            f"image_data must be a tuple, list of tuples, or dict, got {type(image_data)}. "
            f"Expected format: (image_key, pixel_size[, is_rgb]), [(image_key, pixel_size[, is_rgb]), ...], "
            f"or {{name: (image_key, pixel_size[, is_rgb]), ...}}"
        )

def _validate_boundaries_data_format(
    boundaries_data: tuple[str, Number] | None,
    param_name: str = "boundaries_data"
) -> None:
    """
    Validate the format of cell_boundaries_data or nucleus_boundaries_data parameter.

    Args:
        boundaries_data: Boundaries data as a tuple (boundaries_key, pixel_size)
        param_name: Name of the parameter being validated (for error messages)

    Raises:
        ValueError: If the format structure is invalid.
        TypeError: If element types are incorrect.
    """
    if boundaries_data is None:
        return

    if not isinstance(boundaries_data, tuple):
        raise TypeError(
            f"{param_name} must be a tuple, got {type(boundaries_data)}. "
            f"Expected format: (boundaries_key, pixel_size)"
        )

    if len(boundaries_data) != 2:
        raise ValueError(
            f"{param_name} tuple must have 2 elements (boundaries_key, pixel_size), "
            f"got {len(boundaries_data)}"
        )

    if not isinstance(boundaries_data[0], str):
        raise TypeError(
            f"boundaries_key in {param_name} must be a string, got {type(boundaries_data[0])}"
        )

    if not isinstance(boundaries_data[1], Number):
        raise TypeError(
            f"pixel_size in {param_name} must be a number, got {type(boundaries_data[1])}"
        )

try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    from spatialdata.transformations import get_transformation, Identity, Scale
    from spatialdata._core.validation import check_valid_name
    from spatialdata.models import (Image2DModel, Labels2DModel, PointsModel,
                                    ShapesModel, TableModel)
    from spatialdata.transformations.transformations import Scale

import logging
from numbers import Number
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from anndata import AnnData
from xarray import DataArray

from insitupy._constants import (DEFAULT_CHUNK_SIZE_X, DEFAULT_CHUNK_SIZE_Y,
                                 MODALITIES, SAMPLE_STR)
from insitupy._core.data import InSituData
from insitupy.dataclasses import BoundariesData, CellData
from insitupy.images.axes import ImageAxes
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)


def _generate_spatialdata_key(
    sample_id: str,
    modality: Literal[MODALITIES], # type: ignore
    locator: Optional[Union[str, tuple, List]]
    ):
    # from spatialdata._core.validation import check_valid_name
    if not modality.lower() in MODALITIES:
        raise ValueError(f"Modality '{modality}' not recognized. Choose from {MODALITIES}.")

    if modality == "transcripts":
        assert locator is None, "Locator must be None for modality 'transcripts'."
    else:
        assert locator is not None, f"Locator cannot be None for modality '{modality}'."

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
    cell_area_key: Optional[str] = "cell_area"
    ):
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
            print(f"Key '{cell_area_key}' not found in AnnData. Skipped generation of sized circles.")
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
    sample_id: Optional[str] = None
    ):
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
    sample_id: Optional[str] = None
    ):
    # from spatialdata.models import PointsModel
    points = {}
    if xd.transcripts is not None:
        df = xd.transcripts
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
    sample_id: Optional[str] = None
    ):
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


def _transform_cell_boundaries_for_spatialdata(
    xd: InSituData,
    n_pyramids: int = 5,
    sample_id: Optional[str] = None
    ):
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
    sample_id: Optional[str] = None
    ):
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
    sample_id: Optional[str] = None
    ):
    # from spatialdata.models import ShapesModel
    shapes = {}
    if xd.annotations is not None:
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
                print(f"Warning: Duplicate key detected - '{key}'")
            seen_keys.add(key)
        merged.update(d)
    return merged


def _extract_pixel_size_from_spatialdata(
    sdata: SpatialData,
    # pixel_size: Optional[float],
    element_to_extract_from: str,
    coordinate_system: Optional[str] = None,
    verbose: bool = True
) -> Optional[float]:
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
                print(f"Extracted pixel size {ps} from '{element_to_extract_from}'", flush=True)
            return ps

        except Exception as e:
            if verbose:
                print(f"Warning: Could not extract pixel size from '{element_to_extract_from}': {e}", flush=True)

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
    image_data: Union[
        Tuple[str, Number],
        Tuple[str, Number, bool],
        List[Union[Tuple[str, Number], Tuple[str, Number, bool]]],
        Dict[str, Union[Tuple[str, Number], Tuple[str, Number, bool]]]
    ],
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
                print(f"Warning: Image key '{key}' not found in SpatialData", flush=True)
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

        data.images.add_image(
            image=da_img,
            channel_names=name,
            axes=axes_str,
            pixel_size=pixel_size,
            overwrite=False,
            verbose=verbose
        )


def _create_boundaries_from_spatialdata(
    sdata: SpatialData,
    cell_names: np.ndarray,
    cell_boundaries_data: Optional[Tuple[str, Number]] = None, # tuple as (cell_boundaries_key, pixel_size)
    nucleus_boundaries_data: Optional[Tuple[str, Number]] = None, # tuple as (nucleus_boundaries_key, pixel_size)
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
    image_data: Optional[Union[
        Tuple[str, Number],
        Tuple[str, Number, bool],
        List[Union[Tuple[str, Number], Tuple[str, Number, bool]]],
        Dict[str, Union[Tuple[str, Number], Tuple[str, Number, bool]]]
    ]]
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
    boundaries_data: Optional[Tuple[str, Number]],
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
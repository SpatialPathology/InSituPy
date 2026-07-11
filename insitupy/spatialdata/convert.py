try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    pass

import logging
from collections import defaultdict
from numbers import Number
from typing import Union

import numpy as np

from insitupy._constants import MODALITIES, SPATIALDATA_DIALECT_VERSION
from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy.containers import CellData, SpatialUnitsData
from insitupy.spatialdata._convert import (
    _add_images_to_insitudata,
    _create_boundaries_from_spatialdata,
    _merge_dicts_with_warning,
    _transform_annotations_for_spatialdata,
    _transform_cell_boundaries_for_spatialdata,
    _transform_images_for_spatialdata,
    _transform_regions_for_spatialdata,
    _transform_table_for_spatialdata,
    _transform_transcripts_for_spatialdata,
    _transform_units_for_spatialdata,
    _validate_boundaries_data_format,
    _validate_image_data_format,
)

logger = logging.getLogger(__name__)


def check_and_fix_case_insensitive_conflicts(
    sdata: SpatialData
    ):
    """
    Check for case-insensitive key conflicts in a SpatialData object and fix them in place.

    When keys differ only in capitalization (e.g., 'ANNOTATIONS.Demo' and 'ANNOTATIONS.demo'),
    this can cause conflicts when writing to disk on case-insensitive filesystems.

    Args:
        sdata: SpatialData object to check and modify in place.

    Returns:
        tuple: (sdata, rename_map)
            - sdata: The same SpatialData object, with conflicting keys renamed in place.
            - rename_map: Dictionary mapping old keys to new keys (empty if no conflicts)

    Example:
        >>> sdata, renames = check_and_fix_case_insensitive_conflicts(sdata)
        >>> print(renames)
        {'ANNOTATIONS.demo': 'ANNOTATIONS.demo_v2'}
    """
    # Collect all keys from all element types
    all_keys = []
    for attr in ['images', 'labels', 'points', 'shapes', 'tables']:
        if hasattr(sdata, attr):
            element_dict = getattr(sdata, attr)
            if element_dict is not None:
                all_keys.extend(element_dict.keys())

    # Group keys by their lowercase version
    grouped = defaultdict(list)
    for key in all_keys:
        grouped[key.lower()].append(key)

    # Find conflicts (where multiple keys map to same lowercase)
    conflicts = {k: v for k, v in grouped.items() if len(v) > 1}

    if not conflicts:
        logger.info("No case-insensitive conflicts found.")
        return sdata, {}

    # Generate rename map
    rename_map = {}
    all_existing_keys = set(all_keys)  # Track all keys including future renames

    for lower_key, variants in conflicts.items():
        # Keep first variant unchanged, rename the rest
        for i, key in enumerate(variants[1:], start=2):
            new_key = key
            suffix_num = 2

            # Find a unique suffix that doesn't conflict
            while new_key in all_existing_keys or new_key.lower() in [k.lower() for k in all_existing_keys if k != key]:
                new_key = f"{key}_v{suffix_num}"
                suffix_num += 1

            rename_map[key] = new_key
            all_existing_keys.add(new_key)
            all_existing_keys.discard(key)

    if not rename_map:
        return sdata, {}

    # Log warning with conflicts and renames
    message_lines = ["Case-insensitive key conflicts detected and automatically fixed:"]
    for old_key, new_key in rename_map.items():
        message_lines.append(f"  '{old_key}' -> '{new_key}'")
    logger.warning("\n".join(message_lines))

    # Apply renames in place
    for attr in ['images', 'labels', 'points', 'shapes', 'tables']:
        if hasattr(sdata, attr):
            element_dict = getattr(sdata, attr)
            if element_dict is not None:
                # Create new dict with renamed keys
                items_to_rename = [(k, v) for k, v in element_dict.items() if k in rename_map]
                for old_key, value in items_to_rename:
                    new_key = rename_map[old_key]
                    del element_dict[old_key]
                    element_dict[new_key] = value

    return sdata, rename_map

def convert_to_spatialdata_dict(
    data: Union[InSituData, "InSituExperiment"], # type: ignore
    n_pyramids: int = 5,
    include_transcripts: bool = True,
    ):

    """
    Converts an InSituData object to a dictionary for SpatialData object.

    This function integrates various data elements such as images, labels, transcripts, and annotations
    into a SpatialData object. It requires the spatialdata framework to be installed.

    Args:
        data: Source InSituData or InSituExperiment object to convert.
        n_pyramids: Number of resolution pyramid levels to generate for image elements.
        include_transcripts: If False, skip transcript export entirely. Transcript export is
            the dominant cost for large experiments; set to False to omit it.

    Raises:
        ImportError: If the spatialdata framework is not installed.

    Returns:
        Dict: a dictionary with all modalities saved in SpatialData format.
    """
    is_experiment = _is_experiment(data)
    if is_experiment:
        iterator = data.iterdata()
    else:
        iterator = iter([(None, data)])

    merged_dict = {}
    for meta, d in iterator:
        if meta is None:
            sample_id = None
        else:
            sample_id = meta["uid"]
        # create SpatialData dictionary
        if include_transcripts:
            transcripts = _transform_transcripts_for_spatialdata(d, sample_id=sample_id)
        else:
            transcripts = {}
        tables, cell_shapes = _transform_table_for_spatialdata(d, sample_id=sample_id)
        units_tables, units_shapes = _transform_units_for_spatialdata(d, sample_id=sample_id)
        annotations = _transform_annotations_for_spatialdata(d, sample_id=sample_id)
        regions = _transform_regions_for_spatialdata(d, sample_id=sample_id)
        images = _transform_images_for_spatialdata(d, n_pyramids=n_pyramids, sample_id=sample_id)
        labels = _transform_cell_boundaries_for_spatialdata(d, sample_id=sample_id)
        md = _merge_dicts_with_warning(
            transcripts, tables, cell_shapes, units_tables, units_shapes,
            annotations, regions, images, labels
            )

        # collect resulting dictionary
        merged_dict = _merge_dicts_with_warning(merged_dict, md)

    return merged_dict

def convert_to_spatialdata(
    data: Union[InSituData, "InSituExperiment"], # type: ignore
    n_pyramids: int = 5,
    include_transcripts: bool = True,
    ):

    """
    Convert an InSituData or InSituExperiment object to a SpatialData object.

    Integrates images, cell tables, cell shapes, spatial units, transcripts,
    annotations, regions, and cell boundary labels into a single SpatialData object.
    Automatically detects and resolves case-insensitive key conflicts that would
    cause problems when writing to disk.

    Requires the ``spatialdata`` package (``pip install spatialdata``).

    Args:
        data (Union[InSituData, InSituExperiment]): Source data object to convert.
            For an ``InSituExperiment``, all samples are merged into one SpatialData
            object with sample-prefixed element keys.
        n_pyramids (int, optional): Number of resolution pyramid levels to generate
            for image elements. Defaults to 5.
        include_transcripts (bool, optional): If False, skip transcript export
            entirely. Transcript export is the dominant cost for large experiments;
            set to False to omit it. Defaults to True.

    Returns:
        SpatialData: A SpatialData object whose elements are keyed as follows
            (all keys are prefixed with ``'SAMPLE.<sample_uid>..'`` when converting
            an ``InSituExperiment``; see ``insitupy/spatialdata/DIALECT.md`` for the
            full naming spec):

            - **images**: one entry per image channel (e.g. ``'nuclei'``, ``'morphology_focus'``).
            - **labels**: cell boundary label images (e.g. ``'cell_boundaries'``).
            - **shapes**: cell circle shapes, spatial units polygons, and annotation/region shapes.
            - **tables**: cell expression table(s) and spatial units table(s).
            - **points**: transcript coordinates (if available and ``include_transcripts=True``).

            The store also carries a versioned dialect descriptor at
            ``sdata.attrs["insitupy_spatialdata_dialect"]``.

    Raises:
        ImportError: If the ``spatialdata`` package is not installed.
    """
    sd_dict = convert_to_spatialdata_dict(
        data,
        n_pyramids=n_pyramids,
        include_transcripts=include_transcripts,
        )

    dialect_attrs = {
        "insitupy_spatialdata_dialect": {
            "version": SPATIALDATA_DIALECT_VERSION,
            "modalities": list(MODALITIES),
            "sample_prefix_pattern": "SAMPLE.<uid>..",
        }
    }
    sdata = SpatialData.init_from_elements(sd_dict, attrs=dialect_attrs)

    # Check and fix case-insensitive conflicts
    sdata, rename_map = check_and_fix_case_insensitive_conflicts(sdata)

    return sdata

def convert_from_spatialdata(
    sdata: SpatialData,
    # Image parameters
    image_data: tuple[str, Number] | tuple[str, Number, bool] | list[tuple[str, Number] | tuple[str, Number, bool]] | dict[str, tuple[str, Number] | tuple[str, Number, bool]] | None = None,
    # Table parameters
    table_key: str = 'table',
    # Cell parameters
    cells_key: str | None = None,
    # Spatial units parameters
    units_key: str | None = None,
    unit_type: str | None = None,
    # Boundaries parameters
    cell_boundaries_data: tuple[str, Number] | None = None, # tuple as (cell_boundaries_key, pixel_size)
    nucleus_boundaries_data: tuple[str, Number] | None = None, # tuple as (nucleus_boundaries_key, pixel_size)
    # Transcripts parameters
    transcripts_key: str | None = "transcripts",
    # Metadata
    slide_id: str | None = None,
    sample_id: str | None = None,
    metadata: dict | None = None,
    method_name: str = "",

    # Other parameters
    spatial_key: str = "spatial",
    verbose: bool = True
) -> InSituData:
    """
    Convert a SpatialData object to an InSituData object.

    Args:
        sdata: SpatialData object to convert.
        image_data: Image data specification. Supports:
            - Single tuple: (image_key, pixel_size) or (image_key, pixel_size, is_rgb)
            - List of tuples: [(image_key, pixel_size), ...] or [(image_key, pixel_size, is_rgb), ...]
            - Dictionary: {name: (image_key, pixel_size), ...} or {name: (image_key, pixel_size, is_rgb), ...}
            The optional is_rgb flag (default: False) indicates if the image should be treated as RGB.
        table_key: Key for the cell expression table in SpatialData.
        cells_key: Key for cell shapes in SpatialData.
        units_key: Key for spatial units in SpatialData.
        unit_type: Type of spatial unit.
        cell_boundaries_data: Tuple of (label_key, pixel_size) for cell segmentation masks.
        nucleus_boundaries_data: Tuple of (label_key, pixel_size) for nucleus segmentation masks.
        transcripts_key: Key for transcript points in SpatialData.
        slide_id: Identifier for the slide.
        sample_id: Identifier for the sample.
        metadata: Additional metadata dictionary.
        method_name: Name of the spatial method (e.g., "Xenium").
        spatial_key: Key for spatial coordinates in obsm.
        verbose: Whether to print status messages.

    Returns:
        InSituData: Converted InSituData object.
    """

    # Initialize InSituData
    data = InSituData(
        path=None,
        metadata=metadata,
        slide_id=slide_id,
        sample_id=sample_id,
        method_name=method_name,
        method_params=sdata.attrs,
    )

    if 'global' in sdata.coordinate_systems:
        logger.info("Using 'global' coordinate system for pixel size extraction.")
        cs = 'global'
    elif units_key in sdata.coordinate_systems:
        logger.info(f"Using '{units_key}' coordinate system for pixel size extraction.")
        cs = units_key
    elif cells_key in sdata.coordinate_systems:
        logger.info(f"Using '{cells_key}' coordinate system for pixel size extraction.")
        cs = cells_key
    else:
        raise ValueError("Cannot determine coordinate system for pixel size extraction.")

    # pixel_size = _extract_pixel_size_from_spatialdata(
    #     sdata=sdata,
    #     # pixel_size,
    #     element_to_extract_from=element_to_extract_from,
    #     coordinate_system=cs,
    #     verbose=verbose
    #     )

    # LOAD IMAGES
    if image_data:
        # Validate image_data format
        _validate_image_data_format(image_data)

        if verbose:
            logger.info("Adding images...")
        _add_images_to_insitudata(data, sdata, image_data, verbose)

    if cells_key:
        # Validate boundaries data formats
        _validate_boundaries_data_format(cell_boundaries_data, param_name="cell_boundaries_data")
        _validate_boundaries_data_format(nucleus_boundaries_data, param_name="nucleus_boundaries_data")

        # LOAD CELLS (table + boundaries)
        if table_key is not None:
            if table_key in sdata:
                if verbose:
                    logger.info("Adding cell data...")
            table = sdata[table_key]
            cell_names = np.array(table.obs_names)

            if spatial_key in table.obsm:
                logger.warning(f"Spatial coordinates in `obsm['{spatial_key}']` are overwritten using centroids from `'{cells_key}'`.")

            table.obsm[spatial_key] = sdata[cells_key].centroid.get_coordinates().values

            # Prepare boundaries if keys provided
            boundaries = None
            if cell_boundaries_data or nucleus_boundaries_data:
                boundaries = _create_boundaries_from_spatialdata(
                    sdata,
                    cell_names,
                    cell_boundaries_data,
                    nucleus_boundaries_data,
                    # pixel_size
                )

            cd = CellData(table=table, boundaries=boundaries)
            data.cells.add_celldata(cd=cd, key="main", is_main=True)
        elif verbose:
            logger.warning(f"Table key '{table_key}' not found in SpatialData")

    if units_key:
        data.add_units(
            SpatialUnitsData(
                shapes=sdata.shapes[units_key],
                data=sdata[table_key],
                unit_type=unit_type if unit_type is not None else "unit",
                # pixel_size=pixel_size
                )
            )

    # LOAD TRANSCRIPTS
    if transcripts_key and transcripts_key in sdata:
        if verbose:
            logger.info("Adding transcripts...")

        # Rename coordinate columns lazily
        transcripts_df = sdata[transcripts_key]
        rename_map = {}
        if 'x' in transcripts_df.columns:
            rename_map['x'] = 'x_location'
        if 'y' in transcripts_df.columns:
            rename_map['y'] = 'y_location'
        if 'z' in transcripts_df.columns:
            rename_map['z'] = 'z_location'

        if rename_map:
            transcripts_df = transcripts_df.rename(columns=rename_map)

        transcripts_df['feature_name'] = transcripts_df['feature_name'].astype(str)

        data.transcripts = transcripts_df
    elif verbose and transcripts_key:
        logger.warning(f"Transcripts key '{transcripts_key}' not found in SpatialData")

    return data

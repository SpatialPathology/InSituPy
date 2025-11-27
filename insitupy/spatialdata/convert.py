try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    from spatialdata.transformations import get_transformation

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np

from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy.dataclasses import CellData
from insitupy.spatialdata._convert import (
    _add_images_to_insitudata, _create_boundaries_from_spatialdata,
    _extract_pixel_size_from_spatialdata, _merge_dicts_with_warning,
    _transform_annotations_for_spatialdata,
    _transform_cell_boundaries_for_spatialdata,
    _transform_images_for_spatialdata, _transform_matrix_for_spatialdata,
    _transform_regions_for_spatialdata, _transform_transcripts_for_spatialdata)

logger = logging.getLogger(__name__)


def check_and_fix_case_insensitive_conflicts(
    sdata: SpatialData,
    inplace: bool = False
    ):
    """
    Check for case-insensitive key conflicts in a SpatialData object and optionally fix them.

    When keys differ only in capitalization (e.g., 'ANNOTATIONS.Demo' and 'ANNOTATIONS.demo'),
    this can cause conflicts when writing to disk on case-insensitive filesystems.

    Args:
        sdata: SpatialData object to check
        inplace: If True, modify the SpatialData object in place. If False, return a modified copy.

    Returns:
        tuple: (modified_sdata, rename_map)
            - modified_sdata: SpatialData object with renamed keys (original if inplace=True)
            - rename_map: Dictionary mapping old keys to new keys (empty if no conflicts)

    Example:
        >>> sdata_fixed, renames = check_and_fix_case_insensitive_conflicts(sdata, inplace=False)
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

    # Apply renames
    if not inplace:
        # Create a new SpatialData dict
        new_elements = {}
        for attr in ['images', 'labels', 'points', 'shapes', 'tables']:
            if hasattr(sdata, attr):
                element_dict = getattr(sdata, attr)
                if element_dict is not None:
                    for key, value in element_dict.items():
                        new_key = rename_map.get(key, key)
                        new_elements[new_key] = value

        sdata = SpatialData.from_elements_dict(new_elements)
    else:
        # Modify in place (more complex, need to handle each element type)
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
    ):

    """
    Converts an InSituData object to a dictionary for SpatialData object.

    This function integrates various data elements such as images, labels, transcripts, and annotations
    into a SpatialData object. It requires the spatialdata framework to be installed.

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
        transcripts = _transform_transcripts_for_spatialdata(d, sample_id=sample_id)
        tables, cell_shapes = _transform_matrix_for_spatialdata(d, sample_id=sample_id)
        annotations = _transform_annotations_for_spatialdata(d, sample_id=sample_id)
        regions = _transform_regions_for_spatialdata(d, sample_id=sample_id)
        images = _transform_images_for_spatialdata(d, n_pyramids=n_pyramids, sample_id=sample_id)
        labels = _transform_cell_boundaries_for_spatialdata(d, sample_id=sample_id)
        md = _merge_dicts_with_warning(
            transcripts, tables, cell_shapes, annotations, regions, images, labels
            )

        # collect resulting dictionary
        merged_dict = _merge_dicts_with_warning(merged_dict, md)

    return merged_dict

def convert_to_spatialdata(
    data: Union[InSituData, "InSituExperiment"], # type: ignore
    n_pyramids: int = 5
    ):

    """
    Converts an InSituData object to a SpatialData object.

    This function integrates various data elements such as images, labels, transcripts, and annotations
    into a SpatialData object. It requires the spatialdata framework to be installed.

    The function automatically checks for and fixes case-insensitive key conflicts that could cause
    issues when writing to disk.

    Returns:
        SpatialData: A SpatialData object containing the integrated data elements.

    """
    # is_experiment = _is_experiment(data)

    # if is_experiment:

    sd_dict = convert_to_spatialdata_dict(
        data,
        n_pyramids=n_pyramids
        )
    sdata = SpatialData.from_elements_dict(sd_dict)

    # Check and fix case-insensitive conflicts
    sdata, rename_map = check_and_fix_case_insensitive_conflicts(sdata, inplace=True)

    return sdata

def convert_from_spatialdata(
    sdata: SpatialData,
    # Image parameters
    image_keys: Optional[Union[str, List[str], Dict[str, str]]] = None,
    # Table parameters
    table_key: str = 'table',
    # Cell parameters
    cells_key: Optional[str] = None,
    # Features parameters
    features_key: Optional[str] = None,
    # Boundaries parameters
    cell_boundaries_key: Optional[str] = None,
    nucleus_boundaries_key: Optional[str] = None,
    # Transcripts parameters
    transcripts_key: Optional[str] = "transcripts",
    # Metadata
    slide_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    method_name: str = "",

    # Other parameters
    spatial_key: str = "spatial",
    verbose: bool = True
) -> InSituData:
    """
    Convert a SpatialData object to an InSituData object.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object
    image_keys : str, list of str, or dict, optional
        Images to add. Can be:
        - Single string: adds one image with that key as name
        - List of strings: adds multiple images using keys as names
        - Dict {name: key}: adds images with custom names
        If None, no images are added
    table_key : str, default "table"
        Key for the expression matrix table in sdata
    cell_boundaries_key : str, optional
        Key for cell boundaries/labels in sdata
    nucleus_boundaries_key : str, optional
        Key for nucleus boundaries/labels in sdata
    transcripts_key : str, optional
        Key for transcripts in sdata. If None, transcripts are not added
    pixel_size : float, optional
        Explicit pixel size. If None, will try to extract from transformation
    pixel_size_from_element : str, optional
        Element name to extract pixel size from via transformation.
        If None and pixel_size is None, uses first available element
    slide_id : str, optional
        Slide identifier
    sample_id : str, optional
        Sample identifier
    metadata : dict, optional
        Additional metadata
    method_name : str, default ""
        Name of the method used
    verbose : bool, default True
        Print progress messages

    Returns
    -------
    InSituData
        Converted InSituData object
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

    # Determine pixel size
    ps = _extract_pixel_size_from_spatialdata(
        sdata=sdata,
        # pixel_size,
        element_to_extract_from=cells_key,
        verbose=verbose
        )

    # LOAD IMAGES
    if image_keys is not None:
        if verbose:
            print("Adding images...", flush=True)
        _add_images_to_insitudata(data, sdata, image_keys, ps, verbose)

    # LOAD CELLS (matrix + boundaries)
    if table_key is not None:
        if table_key in sdata:
            if verbose:
                print("Adding cell data...", flush=True)

            matrix = sdata[table_key]
            cell_names = np.array(matrix.obs_names)

            if spatial_key in matrix.obsm:
                logger.warning(f"Spatial coordinates in `obsm['{spatial_key}']` are overwritten using centroids from `'{cells_key}'`.")

            matrix.obsm[spatial_key] = sdata[cells_key].centroid.get_coordinates().values

            # Prepare boundaries if keys provided
            boundaries = None
            if cell_boundaries_key or nucleus_boundaries_key:
                boundaries = _create_boundaries_from_spatialdata(
                    sdata,
                    cell_names,
                    cell_boundaries_key,
                    nucleus_boundaries_key,
                    ps
                )

            cd = CellData(matrix=matrix, boundaries=boundaries)
            data.cells.add_celldata(cd=cd, key="main", is_main=True)
        elif verbose:
            logger.warning(f"Warning: Table key '{table_key}' not found in SpatialData", flush=True)

    # LOAD TRANSCRIPTS
    if transcripts_key and transcripts_key in sdata:
        if verbose:
            print("Adding transcripts...", flush=True)

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
        print(f"Warning: Transcripts key '{transcripts_key}' not found in SpatialData", flush=True)

    return data
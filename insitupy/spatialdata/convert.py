try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    pass

import logging
import os
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData

from insitupy._constants import (
    MODALITIES,
    SPATIALDATA_DERIVED_MODALITIES,
    SPATIALDATA_DIALECT_VERSION,
)
from insitupy._core._checks import _is_experiment
from insitupy._core.data import InSituData
from insitupy.containers import CellData, SpatialUnitsData
from insitupy.experiment.data import InSituExperiment, TableAccessor
from insitupy.spatialdata._convert import (
    _add_images_to_insitudata,
    _build_insitudata_from_elements,
    _create_boundaries_from_spatialdata,
    _generate_spatialdata_key,
    _group_elements_by_sample,
    _merge_dicts_with_warning,
    _transform_annotations_for_spatialdata,
    _transform_cell_boundaries_for_spatialdata,
    _transform_concat_tables_for_spatialdata,
    _transform_images_for_spatialdata,
    _transform_nucleus_map_for_spatialdata,
    _transform_regions_for_spatialdata,
    _transform_table_for_spatialdata,
    _transform_transcripts_for_spatialdata,
    _transform_units_for_spatialdata,
    _validate_boundaries_data_format,
    _validate_image_data_format,
)
from insitupy.utils.utils import convert_to_list

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
    include_concat_tables: bool = True,
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
        include_concat_tables: If False, skip exporting InSituExperiment.build_table()'s
            concatenated union table(s) even if built. Ignored for a bare InSituData (which has
            no build_table()). Defaults to True: any layer with a built table is exported as a
            TABLES.<layer> element - export is opt-in by virtue of having called build_table().

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
        nucleus_maps = _transform_nucleus_map_for_spatialdata(d, sample_id=sample_id)
        md = _merge_dicts_with_warning(
            transcripts, tables, cell_shapes, units_tables, units_shapes,
            annotations, regions, images, labels, nucleus_maps
            )

        # collect resulting dictionary
        merged_dict = _merge_dicts_with_warning(merged_dict, md)

    if is_experiment and include_concat_tables:
        concat_tables = _transform_concat_tables_for_spatialdata(
            data, exported_keys=set(merged_dict.keys())
        )
        merged_dict = _merge_dicts_with_warning(merged_dict, concat_tables)

    return merged_dict

def convert_to_spatialdata(
    data: Union[InSituData, "InSituExperiment"], # type: ignore
    n_pyramids: int = 5,
    include_transcripts: bool = True,
    include_concat_tables: bool = True,
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
        include_concat_tables (bool, optional): If False, skip exporting
            ``InSituExperiment.build_table()``'s concatenated union table(s) even if built.
            Ignored for a bare ``InSituData``. Defaults to True.

    Returns:
        SpatialData: A SpatialData object whose elements are keyed as follows
            (all keys are prefixed with ``'SAMPLE.<sample_uid>..'`` when converting
            an ``InSituExperiment``; see ``insitupy/spatialdata/DIALECT.md`` for the
            full naming spec):

            - **images**: one entry per image channel (e.g. ``'nuclei'``, ``'morphology_focus'``).
            - **labels**: cell boundary label images (e.g. ``'cell_boundaries'``).
            - **shapes**: cell circle shapes, spatial units polygons, and annotation/region shapes.
            - **tables**: cell expression table(s), spatial units table(s), - when a cell
              layer's boundaries carry a populated ``nucleus_to_cell_map`` (multinucleated-cell
              support) - a small ``CELLS.<key>.nucleus_map`` table linking nucleus labels to
              parent cells, and - for an ``InSituExperiment`` with a built table - a
              ``TABLES.<layer>`` element holding ``build_table()``'s concatenated union table.
            - **points**: transcript coordinates (if available and ``include_transcripts=True``).

            The store also carries a versioned dialect descriptor at
            ``sdata.attrs["insitupy_spatialdata_dialect"]``, including ``uid``/``slide_id``/
            ``sample_id`` identity for each sample (or flat ``slide_id``/``sample_id`` for a
            bare ``InSituData``).

    Raises:
        ImportError: If the ``spatialdata`` package is not installed.
    """
    sd_dict = convert_to_spatialdata_dict(
        data,
        n_pyramids=n_pyramids,
        include_transcripts=include_transcripts,
        include_concat_tables=include_concat_tables,
        )

    dialect_attrs = {
        "insitupy_spatialdata_dialect": {
            "version": SPATIALDATA_DIALECT_VERSION,
            "modalities": [*MODALITIES, *SPATIALDATA_DERIVED_MODALITIES],
            "sample_prefix_pattern": "SAMPLE.<uid>..",
        }
    }

    if _is_experiment(data):
        dialect_attrs["insitupy_spatialdata_dialect"]["samples"] = {
            meta["uid"]: {"slide_id": d.slide_id, "sample_id": d.sample_id}
            for meta, d in data.iterdata()
        }
    else:
        dialect_attrs["insitupy_spatialdata_dialect"]["slide_id"] = data.slide_id
        dialect_attrs["insitupy_spatialdata_dialect"]["sample_id"] = data.sample_id

    sdata = SpatialData.init_from_elements(sd_dict, attrs=dialect_attrs)

    # Check and fix case-insensitive conflicts
    sdata, rename_map = check_and_fix_case_insensitive_conflicts(sdata)

    return sdata

def _convert_from_spatialdata_manual(
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
    Convert a SpatialData object to an InSituData object using caller-supplied keys.

    Low-level, manual-parameter primitive for SpatialData objects that carry no
    InSituPy dialect descriptor (foreign / labels-native stores, e.g.
    ``spatialdata-io`` output). Not part of the public API (not exported from
    ``insitupy.spatialdata``) - InSituPy's own dialect round trip uses the
    dialect-driven :func:`convert_from_spatialdata` instead, which needs none
    of these keys. This function is the enabling groundwork WP4 hardens for
    general foreign-store import (fixing, among other things, the
    ``units_key``/``table_key`` mixup below - not fixed here, since it is
    structurally impossible in the dialect-driven path and this function is
    being handed to WP4 regardless; see
    ``.log/reports/260711/spatialdata-work-packages/report-wp4-labels-native-import.md``).

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


def convert_from_spatialdata(
    sdata: SpatialData,
    verbose: bool = True,
) -> InSituData | InSituExperiment:
    """
    Convert an InSituPy-dialect SpatialData object back into an InSituData or InSituExperiment.

    A true inverse of :func:`convert_to_spatialdata`: the dialect and every
    modality-naming detail (pixel sizes, RGB-ness, per-cell-layer boundaries,
    per-unit-layer tables, ...) are auto-detected from the store itself and
    ``sdata.attrs["insitupy_spatialdata_dialect"]`` - no caller-supplied keys
    or pixel sizes are needed. Returns a bare ``InSituData`` for a
    single-sample store (no ``SAMPLE.<uid>..`` prefix in any element key), or
    an ``InSituExperiment`` of ``InSituData`` objects for a multi-sample store.

    The returned object has no backing project directory - call ``.saveas(path)``
    to persist it as a ``.insitupy`` project before ``.save()`` can be used.

    Args:
        sdata: A SpatialData object written by :func:`convert_to_spatialdata`.
        verbose: If True, log progress for each modality.

    Returns:
        InSituData or InSituExperiment: The reconstructed object(s).

    Raises:
        ValueError: If ``sdata.attrs`` carries no InSituPy dialect descriptor
            (i.e. this is a foreign/labels-native store InSituPy did not
            write - see ``insitupy.spatialdata._convert._convert_from_spatialdata_manual``,
            the low-level primitive WP4 is hardening for that case) or if the
            descriptor's version is not one this InSituPy version supports.
    """
    dialect = sdata.attrs.get("insitupy_spatialdata_dialect")
    if dialect is None:
        raise ValueError(
            "sdata was not written by insitupy.spatialdata.convert_to_spatialdata "
            "(no 'insitupy_spatialdata_dialect' key in sdata.attrs). Reading a "
            "foreign/labels-native SpatialData store is not supported by this "
            "function; see insitupy.spatialdata._convert._convert_from_spatialdata_manual "
            "(WP4, in progress)."
        )

    version = dialect.get("version")
    if version != SPATIALDATA_DIALECT_VERSION:
        raise ValueError(
            f"Unsupported insitupy_spatialdata_dialect version {version!r}; "
            f"this InSituPy version reads dialect version {SPATIALDATA_DIALECT_VERSION} only."
        )

    grouped = _group_elements_by_sample(sdata)

    if len(grouped) == 1 and None in grouped:
        return _build_insitudata_from_elements(
            grouped[None],
            slide_id=dialect.get("slide_id"),
            sample_id=dialect.get("sample_id"),
            method_params=dict(sdata.attrs),
            verbose=verbose,
        )

    sample_meta = dialect.get("samples", {})
    data, uids, slide_ids, sample_ids = [], [], [], []
    for uid, elements in grouped.items():
        if uid is None:
            # Global (non-per-sample) elements, e.g. TABLES.<layer> - not a sample. Reading
            # a concatenated table back requires convert_table_from_spatialdata(sdata, layer).
            logger.debug(
                "Skipping %d global (non-per-sample) element(s) while reconstructing "
                "per-sample InSituData objects: %s", len(elements), sorted(elements),
            )
            continue
        meta = sample_meta.get(uid, {})
        xd = _build_insitudata_from_elements(
            elements,
            slide_id=meta.get("slide_id"),
            sample_id=meta.get("sample_id"),
            method_params=dict(sdata.attrs),
            verbose=verbose,
        )
        xd._uid = uid
        data.append(xd)
        uids.append(uid)
        slide_ids.append(meta.get("slide_id"))
        sample_ids.append(meta.get("sample_id"))

    experiment = InSituExperiment()
    experiment._metadata = pd.DataFrame({
        "uid": uids,
        "slide_id": slide_ids,
        "sample_id": sample_ids,
    })
    experiment._data = data
    return experiment


def read_spatialdata(
    path: str | os.PathLike | Path,
    verbose: bool = True,
) -> InSituData | InSituExperiment:
    """
    Read an InSituPy-dialect SpatialData zarr store into an InSituData or InSituExperiment.

    Thin convenience wrapper around ``spatialdata.read_zarr`` +
    :func:`convert_from_spatialdata`. Matches the ``insitupy.io`` reader
    convention (``read_xenium``, ``read_visium``, ...) rather than
    auto-detecting the on-disk format inside ``InSituData.read()`` /
    ``InSituExperiment.read()``.

    Args:
        path: Path to a SpatialData ``.zarr`` store written by
            :func:`convert_to_spatialdata`.
        verbose: If True, log progress for each modality.

    Returns:
        InSituData or InSituExperiment: The reconstructed object(s), with no
        backing project directory - call ``.saveas(path)`` to persist as a
        ``.insitupy`` project before ``.save()`` can be used.
    """
    import spatialdata

    sdata = spatialdata.read_zarr(path)
    return convert_from_spatialdata(sdata, verbose=verbose)


def convert_table_from_spatialdata(
    sdata: SpatialData,
    cells_layer: str,
    covered_labels: list[str] | str | None = None,
) -> AnnData:
    """
    Reconstruct the ``.table``-equivalent AnnData for *cells_layer* from a ``TABLES.<layer>`` element.

    Applies the same inner-over-covered reconstruction that
    :class:`~insitupy.experiment.data.TableAccessor` /
    :class:`~insitupy.experiment.data.ViewTableAccessor` use for a disk-built table
    (:meth:`TableAccessor._reconstruct`, unmodified), sourced from the SpatialData element
    instead of a local zarr store.

    Args:
        sdata: A SpatialData object written by :func:`convert_to_spatialdata` with
            ``include_concat_tables=True`` (the default) and a built table for *cells_layer*.
        cells_layer: Cell layer whose concatenated table to reconstruct (e.g. ``"main"``).
        covered_labels: If ``None`` (default), reconstructs the full-experiment table
            (equivalent to ``exp.table[cells_layer]``) - the inner gene set over every sample
            the table was built from. If a subset of labels (e.g. sample uids) is given,
            reconstructs the inner-over-that-subset, row-filtered table (equivalent to
            ``view.table[cells_layer]``).

    Returns:
        AnnData: The reconstructed inner-over-covered table.

    Raises:
        KeyError: If no ``TABLES.<cells_layer>`` element exists in *sdata*.
    """
    key = _generate_spatialdata_key(sample_id=None, modality="tables", locator=cells_layer)
    if key not in sdata.tables:
        raise KeyError(
            f"No concatenated table found for cells_layer='{cells_layer}' "
            f"(looked for '{key}')."
        )
    full = sdata.tables[key]
    labels = np.array([str(label) for label in full.uns["_insitupy_presence_labels"]])
    presence = np.asarray(full.uns["_insitupy_gene_presence"], dtype=bool)
    label_col = full.uns.get("_insitupy_build_params", {}).get("label_col", "uid")

    if covered_labels is None:
        covered_labels, row_filter = labels, False
    else:
        covered_labels, row_filter = convert_to_list(covered_labels), True

    return TableAccessor._reconstruct(
        full,
        covered_labels=covered_labels,
        labels=labels,
        presence=presence,
        label_col=label_col,
        row_filter=row_filter,
    )

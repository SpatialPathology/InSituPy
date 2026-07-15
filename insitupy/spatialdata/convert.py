try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    pass

import logging
import os
from collections import defaultdict
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
    _assign_sdata_transcripts,
    _build_insitudata_from_elements,
    _centroids_from_labels,
    _create_boundaries_from_spatialdata,
    _extract_pixel_size_from_element,
    _generate_spatialdata_key,
    _get_base_resolution_array,
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
    _validate_foreign_spec,
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

def convert_from_foreign_spatialdata(
    sdata: SpatialData,
    images: dict[str, dict] | None = None,
    cells: dict[str, dict] | None = None,
    units: dict[str, dict] | None = None,
    transcripts: str | None = None,
    slide_id: str | None = None,
    sample_id: str | None = None,
    metadata: dict | None = None,
    method_name: str = "",
    spatial_key: str = "spatial",
    coordinate_system: str | None = None,
    verbose: bool = True,
) -> InSituData:
    """
    Convert a foreign / labels-native SpatialData object into an InSituData.

    For SpatialData objects that carry no InSituPy dialect descriptor - e.g.
    ``spatialdata_io.xenium()`` output, or any other labels-native store following
    the standard SpatialData ``TableModel`` annotation contract
    (``region``/``region_key``/``instance_key`` in ``table.uns["spatialdata_attrs"]``).
    InSituPy's own dialect round trip uses the dialect-driven
    :func:`convert_from_spatialdata` instead.

    ``images``, ``cells``, and ``units`` are keyed dicts - one entry per
    InSituData-side image/layer - because ``InSituData`` supports multiple cell
    layers (``MultiCellData``) and multiple spatial-units layers
    (``MultiSpatialUnitsData``), each built from its own SpatialData table. The
    first entry of each dict becomes the main layer (``is_main = (i == 0)``).
    ``transcripts`` stays a scalar SpatialData points key, since
    ``InSituData.transcripts`` is single-cardinality. Every modality parameter
    defaults to ``None`` - nothing is imported unless asked.

    Segmentation identity (``seg_mask_value``) and, when no shapes/circles element
    is available, cell centroids are derived from the table's own real data rather
    than fabricated. A cells spec's ``cells_key``/``cell_boundaries_data`` are
    auto-detected from the table's declared ``region`` when not given explicitly -
    the minimal cells entry is ``cells={"main": {"table_key": "table"}}``. Explicit
    spec fields always override auto-detection.

    Args:
        sdata: SpatialData object to convert.
        images: ``{name: spec}`` - one entry per image to import. Spec keys:

            - ``key`` (str, required): SpatialData image element key.
            - ``pixel_size`` (Number, required): microns/pixel (foreign stores
              don't carry it reliably).
            - ``is_rgb`` (bool, optional, default ``False``): forwarded to
              ``add_image``.
        cells: ``{layer: spec}`` - one entry per cell layer to import. Spec keys:

            - ``table_key`` (str, required): SpatialData table element key for
              this layer.
            - ``cells_key`` (str, optional): cell shapes key; auto-detected from
              the table's declared ``region`` when omitted.
            - ``cell_boundaries_data`` (tuple of (str, Number), optional):
              ``(labels_key, pixel_size)``; auto-detected from ``region`` when
              ``region`` is a labels element and neither this nor ``cells_key``
              is given.
            - ``nucleus_boundaries_data`` (tuple of (str, Number), optional):
              ``(labels_key, pixel_size)``; never auto-detected - no standard
              annotation identifies a nucleus region.
        units: ``{layer: spec}`` - one entry per spatial-units layer to import.
            Spec keys:

            - ``table_key`` (str, required): SpatialData table element key for
              this units layer.
            - ``units_key`` (str, required): SpatialData shapes key for the unit
              geometries.
            - ``unit_type`` (str, optional, default ``"unit"``): stored on the
              ``SpatialUnitsData``.
        transcripts: SpatialData points key for transcripts.
        slide_id: Identifier for the slide.
        sample_id: Identifier for the sample.
        metadata: Additional metadata dictionary.
        method_name: Name of the spatial method (e.g., "Xenium").
        spatial_key: Key for spatial coordinates in obsm.
        coordinate_system: Coordinate system to resolve pixel sizes in, for any
            auto-detected boundaries. Defaults to ``'global'`` when present in
            ``sdata.coordinate_systems`` (virtually always true), else falls back
            to the table's ``region``.
        verbose: Whether to print status messages.

    Returns:
        InSituData: Converted InSituData object, with no backing project directory
        (call ``.saveas(path)`` before ``.save()`` can be used).

    Raises:
        TypeError: If a spec value is not a dict.
        ValueError: If a spec is missing required keys, has unknown keys, or
            names a table/shapes key not present in ``sdata``.

    Examples:
        Xenium-style: cells + boundaries + transcripts + images::

            convert_from_foreign_spatialdata(
                sdata,
                images={"nuclei": {"key": "morphology_focus", "pixel_size": 0.2125}},
                cells={"main": {
                    "table_key": "table",
                    "cells_key": "cell_circles",
                    "cell_boundaries_data": ("cell_labels", 0.2125),
                    "nucleus_boundaries_data": ("nucleus_labels", 0.2125),
                }},
                transcripts="transcripts",
            )

        Minimal labels-native: boundaries / seg-values / centroids derived from
        the table's region::

            convert_from_foreign_spatialdata(sdata, cells={"main": {"table_key": "table"}})

        Visium, units-only (no cells)::

            convert_from_foreign_spatialdata(
                sdata,
                units={"visium": {"table_key": "table", "units_key": "spots", "unit_type": "visium"}},
            )
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

    # LOAD IMAGES (caller-supplied pixel size / RGB-ness - unchanged contract)
    if images:
        if verbose:
            logger.info("Adding images...")
        _add_images_to_insitudata(data, sdata, images, verbose)

    # LOAD CELLS (table + boundaries) per layer.
    if cells:
        if verbose:
            logger.info("Adding cell data...")
        for i, (layer_name, spec) in enumerate(cells.items()):
            _validate_foreign_spec(
                spec, layer_name, "cells",
                required=("table_key",),
                allowed=("table_key", "cells_key", "cell_boundaries_data", "nucleus_boundaries_data"),
            )
            table_key = spec["table_key"]
            if not isinstance(table_key, str):
                raise TypeError(f"cells spec for '{layer_name}': 'table_key' must be a string, got {type(table_key)}.")
            if table_key not in sdata:
                raise ValueError(
                    f"cells layer '{layer_name}': table_key '{table_key}' not found in SpatialData."
                )

            # per-entry locals - re-read from spec every iteration so an
            # auto-detected value for one layer never leaks into the next.
            cells_key = spec.get("cells_key")
            cell_boundaries_data = spec.get("cell_boundaries_data")
            nucleus_boundaries_data = spec.get("nucleus_boundaries_data")
            table = sdata[table_key]

            spatialdata_attrs = table.uns.get("spatialdata_attrs", {}) or {}
            region = spatialdata_attrs.get("region")
            region_key = spatialdata_attrs.get("region_key")
            instance_key = spatialdata_attrs.get("instance_key")

            if isinstance(region, (list, tuple)):
                if len(region) != 1:
                    raise ValueError(
                        f"Table '{table_key}' annotates {len(region)} regions ({list(region)!r}) - "
                        "multi-region tables are not supported by the foreign-store importer."
                    )
                region = region[0]

            if region is not None and region_key is not None and region_key in table.obs.columns:
                unexpected = set(table.obs[region_key].unique()) - {region}
                if unexpected:
                    raise ValueError(
                        f"Table '{table_key}' region_key column '{region_key}' contains values other "
                        f"than the declared region {region!r}: {sorted(map(str, unexpected))} - "
                        "multi-region tables are not supported by the foreign-store importer."
                    )

            cell_names = np.array(table.obs_names)

            # Real segmentation identity: prefer the table's own instance_key column
            # over fabricating a 1..N mapping (only a last resort for stores that don't
            # follow the standard SpatialData table-annotation contract at all).
            if instance_key is not None and instance_key in table.obs.columns:
                seg_mask_value = table.obs[instance_key].to_numpy()
            else:
                logger.warning(
                    "No usable 'instance_key' found in the table's spatialdata_attrs - "
                    "falling back to an assumed 1..N mapping between obs order and mask value. "
                    "This is very likely wrong for a real segmentation mask."
                )
                seg_mask_value = np.arange(1, len(cell_names) + 1)

            # Auto-detect cells_key / cell_boundaries_data from the table's own region
            # when the caller didn't supply either explicitly. Explicit args always win.
            if cells_key is None and cell_boundaries_data is None and region is not None:
                if region in sdata.shapes:
                    cells_key = region
                elif region in sdata.labels:
                    if coordinate_system is not None:
                        cs = coordinate_system
                    elif 'global' in sdata.coordinate_systems:
                        cs = 'global'
                    elif region in sdata.coordinate_systems:
                        cs = region
                    else:
                        raise ValueError("Cannot determine coordinate system for pixel size extraction.")

                    pixel_size = _extract_pixel_size_from_element(sdata[region], coordinate_system=cs, verbose=verbose)
                    cell_boundaries_data = (region, pixel_size)

            # Validate boundaries data formats (whatever the final values are, explicit or auto-detected)
            _validate_boundaries_data_format(cell_boundaries_data, param_name="cell_boundaries_data")
            _validate_boundaries_data_format(nucleus_boundaries_data, param_name="nucleus_boundaries_data")

            # Spatial coordinates
            if cells_key and cells_key in sdata:
                if spatial_key in table.obsm:
                    logger.warning(f"Spatial coordinates in `obsm['{spatial_key}']` are overwritten using centroids from `'{cells_key}'`.")
                table.obsm[spatial_key] = sdata[cells_key].centroid.get_coordinates().values
            elif cell_boundaries_data is not None:
                if spatial_key in table.obsm:
                    logger.warning(
                        f"Spatial coordinates in `obsm['{spatial_key}']` are overwritten using "
                        f"centroids derived from '{cell_boundaries_data[0]}'."
                    )
                label_key, label_pixel_size = cell_boundaries_data
                label_array = _get_base_resolution_array(sdata[label_key]).data
                table.obsm[spatial_key] = _centroids_from_labels(label_array, seg_mask_value, label_pixel_size)
            elif spatial_key not in table.obsm:
                raise ValueError(
                    f"No shapes element ('cells_key') or labels element ('cell_boundaries_data') "
                    f"available to derive obsm['{spatial_key}'] from, and none is already present."
                )

            # Prepare boundaries if keys resolved (explicit or auto-detected)
            boundaries = None
            if cell_boundaries_data or nucleus_boundaries_data:
                boundaries = _create_boundaries_from_spatialdata(
                    sdata,
                    cell_names,
                    seg_mask_value,
                    cell_boundaries_data,
                    nucleus_boundaries_data,
                )

            cd = CellData(table=table, boundaries=boundaries)
            data.cells.add_celldata(cd=cd, key=layer_name, is_main=(i == 0))

    # LOAD SPATIAL UNITS per layer.
    if units:
        if verbose:
            logger.info("Adding spatial units...")
        for i, (layer_name, spec) in enumerate(units.items()):
            _validate_foreign_spec(
                spec, layer_name, "units",
                required=("table_key", "units_key"),
                allowed=("table_key", "units_key", "unit_type"),
            )
            table_key = spec["table_key"]
            units_key = spec["units_key"]
            if not isinstance(table_key, str):
                raise TypeError(f"units spec for '{layer_name}': 'table_key' must be a string, got {type(table_key)}.")
            if not isinstance(units_key, str):
                raise TypeError(f"units spec for '{layer_name}': 'units_key' must be a string, got {type(units_key)}.")
            if table_key not in sdata:
                raise ValueError(f"units layer '{layer_name}': table_key '{table_key}' not found in SpatialData.")
            if units_key not in sdata.shapes:
                raise ValueError(f"units layer '{layer_name}': units_key '{units_key}' not in sdata.shapes.")

            su = SpatialUnitsData(
                shapes=sdata.shapes[units_key],
                data=sdata[table_key],
                unit_type=spec.get("unit_type") or "unit",
            )
            data.add_units(su, key=layer_name, is_main=(i == 0))

    # LOAD TRANSCRIPTS
    if transcripts and transcripts in sdata:
        if verbose:
            logger.info("Adding transcripts...")
        _assign_sdata_transcripts(data, sdata[transcripts])
    elif verbose and transcripts:
        logger.warning(f"Transcripts key '{transcripts}' not found in SpatialData")

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
            write - see :func:`insitupy.spatialdata.convert_from_foreign_spatialdata`)
            or if the descriptor's version is not one this InSituPy version supports.
    """
    dialect = sdata.attrs.get("insitupy_spatialdata_dialect")
    if dialect is None:
        raise ValueError(
            "sdata was not written by insitupy.spatialdata.convert_to_spatialdata "
            "(no 'insitupy_spatialdata_dialect' key in sdata.attrs). Reading a "
            "foreign/labels-native SpatialData store is not supported by this "
            "function; see insitupy.spatialdata.convert_from_foreign_spatialdata "
            "instead."
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

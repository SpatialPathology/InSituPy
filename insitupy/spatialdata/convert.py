try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")
else:
    from spatialdata.transformations import Scale, get_transformation

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
    _assign_sdata_transcripts,
    _build_insitudata_from_elements,
    _centroids_from_labels,
    _create_boundaries_from_spatialdata,
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
    Refuse case-insensitive element-name conflicts in a SpatialData object (SD-B6).

    When two element keys differ only in capitalization (e.g. 'ANNOTATIONS.Demo' and
    'ANNOTATIONS.demo'), they cannot be stored distinctly on a case-insensitive
    filesystem, and the InSituPy dialect cannot round-trip them without a durable
    rename map (deferred to dialect 4). Rather than silently auto-renaming one element
    - which changed the layer name, left the table's ``region`` attr dangling and was
    not recorded, so the layer was dropped or mislabelled on read - this raises and
    asks the caller to rename.

    Args:
        sdata: SpatialData object to check.

    Returns:
        tuple: ``(sdata, {})`` - the same object and an empty rename map, kept for
        backward-compatible call sites. A non-empty rename map is never returned.

    Raises:
        ValueError: If any two element keys collide case-insensitively.
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

    # SD-B6: refuse rather than silently auto-rename. Auto-renaming one variant to
    # "..._v2" changed the layer name, left the table's `region` attr dangling, and
    # was not recorded, so the layer was dropped or mislabelled on read. A lossless
    # fix needs a durable rename map in the store descriptor (deferred to dialect 4);
    # for now, ask the caller to rename.
    conflict_lines = [f"  {sorted(variants)}" for variants in conflicts.values()]
    raise ValueError(
        "Case-insensitive element-name conflict(s) in the SpatialData dialect - the "
        "following key groups differ only by case and cannot be stored or round-tripped "
        "distinctly:\n" + "\n".join(conflict_lines) + "\nRename one element in each group "
        "so the names differ by more than case."
    )

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

def _resolve_foreign_pixel_size(
    sdata: SpatialData,
    region: str,
    pixel_size_spec: Number | None,
    coordinate_system: str,
    verbose: bool,
) -> float:
    """Resolve the pixel size (um/pixel) for an auto-detected foreign labels element (SD-B1).

    Never trusts a labels element's ``Identity`` transform as 1 um/pixel - that shortcut is
    only valid for InSituPy's own dialect writer (see ``_extract_pixel_size_from_element``,
    which stays unchanged for that reader; the dialect reader's ``Identity -> 1.0`` is
    correct for its own store). Resolution order:

    1. ``pixel_size_spec``, if the caller passed one explicitly in the cells spec.
    2. The first sibling points/shapes element (in ``coordinate_system``) carrying a pure
       ``Scale`` transform - spatialdata-io writes such elements as ``Scale([1/ps, 1/ps])``,
       the inverse of InSituPy's own writer's ``Scale([ps, ps])`` convention, so the pixel
       size is ``1 / scale_x``. Logged at info level, naming the element used.
    3. Otherwise raise, naming both fixes.

    Args:
        sdata: SpatialData object.
        region: the labels element key whose pixel size is being resolved (for error
            messages only).
        pixel_size_spec: the cells spec's ``pixel_size`` key, if given.
        coordinate_system: coordinate system to resolve transforms in (already picked by
            the caller - 'global' when present, else the table's own region).
        verbose: if True, log the resolution.

    Returns:
        Pixel size in micrometers per pixel.

    Raises:
        ValueError: if no pixel size can be resolved from any source.
    """
    if pixel_size_spec is not None:
        return float(pixel_size_spec)

    for group_name in ("points", "shapes"):
        group = getattr(sdata, group_name, None)
        if not group:
            continue
        for elem_key, elem in group.items():
            try:
                transform = get_transformation(element=elem, to_coordinate_system=coordinate_system)
            except Exception:
                # Element has no transform to this coordinate system - not a usable sibling.
                continue
            if not isinstance(transform, Scale):
                continue
            axis_index = transform.axes.index("x") if "x" in transform.axes else 0
            scale_value = transform.scale[axis_index].item()
            if scale_value == 0:
                continue
            pixel_size = 1.0 / scale_value
            if verbose:
                logger.info(
                    f"Resolved pixel size {pixel_size} for labels element '{region}' from "
                    f"sibling {group_name[:-1]} element '{elem_key}' (Scale {scale_value})."
                )
            return pixel_size

    raise ValueError(
        f"Cannot resolve a pixel size for labels element '{region}': its transform does not "
        "reliably imply 1 micrometer/pixel (an Identity transform on a foreign labels raster "
        "is not trustworthy, unlike InSituPy's own dialect writer). Fix by either: (1) passing "
        f"an explicit 'pixel_size' key in the cells spec for this layer, or (2) ensuring a "
        f"sibling points/shapes element in coordinate system '{coordinate_system}' carries a "
        "Scale transform (spatialdata_io.xenium() output does, e.g. via its 'transcripts' "
        "points element)."
    )


def _resolve_nucleus_to_cell_map(
    sdata: SpatialData,
    label_map_spec: str,
    cell_names: np.ndarray,
) -> dict:
    """Build a nucleus-index -> parent-cell-name map from a foreign store's map source (SD-B2).

    Args:
        sdata: SpatialData object.
        label_map_spec: key of a shapes element (spatialdata-io's ``nucleus_boundaries``, a
            GeoDataFrame indexed by nucleus label with a ``cell_id`` column) or a table
            element with ``nucleus_label``/``cell_id`` columns (InSituPy dialect,
            ``_transform_nucleus_map_for_spatialdata``'s output).
        cell_names: this cell layer's ``obs_names`` - every mapped ``cell_id`` must be one
            of these.

    Returns:
        ``{nucleus_index_0based: parent_cell_name}``, matching the convention documented on
        ``BoundariesData.__init__``'s ``nucleus_to_cell_map`` parameter (mask value N maps to
        key N - 1, since mask values are 1-indexed).

    Raises:
        ValueError: if ``label_map_spec`` names a missing element, is neither a shapes nor a
            table element, has a non-integer/non-unique index (shapes) or missing columns, or
            maps to a ``cell_id`` not present in ``cell_names``.
    """
    if label_map_spec not in sdata:
        raise ValueError(
            f"label_map '{label_map_spec}' not found in SpatialData - cannot resolve the "
            "nucleus-to-cell map."
        )

    if label_map_spec in sdata.shapes:
        gdf = sdata[label_map_spec]
        if "cell_id" not in gdf.columns:
            raise ValueError(
                f"label_map shapes element '{label_map_spec}' has no 'cell_id' column."
            )
        try:
            indices = gdf.index.astype(int)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"label_map shapes element '{label_map_spec}' has a non-integer index - "
                "expected integer nucleus label values (1-based mask values)."
            ) from e
        if indices.duplicated().any():
            raise ValueError(
                f"label_map shapes element '{label_map_spec}' has a non-unique index."
            )
        nucleus_to_cell_map = {
            int(idx) - 1: str(cell_id)
            for idx, cell_id in zip(indices, gdf["cell_id"])
        }
    elif label_map_spec in sdata.tables:
        table = sdata[label_map_spec]
        obs = table.obs
        missing_cols = {"nucleus_label", "cell_id"} - set(obs.columns)
        if missing_cols:
            raise ValueError(
                f"label_map table element '{label_map_spec}' is missing column(s) "
                f"{sorted(missing_cols)} - expected 'nucleus_label' and 'cell_id'."
            )
        nucleus_to_cell_map = {
            int(row.nucleus_label) - 1: str(row.cell_id)
            for row in obs.itertuples()
        }
    else:
        raise ValueError(
            f"label_map '{label_map_spec}' is neither a shapes nor a table element in "
            "SpatialData."
        )

    cell_name_set = {str(c) for c in cell_names}
    missing_cells = {cid for cid in nucleus_to_cell_map.values() if cid not in cell_name_set}
    if missing_cells:
        raise ValueError(
            f"label_map '{label_map_spec}' maps to cell_id(s) not present in this cell "
            f"layer's cell_names: {sorted(missing_cells)[:10]}"
        )

    return nucleus_to_cell_map


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
              annotation identifies a nucleus region. Requires ``label_map``
              (raises otherwise) - a nucleus raster has no reliable 1:1
              mapping to cells.
            - ``pixel_size`` (Number, optional): microns/pixel for
              auto-detected label-derived centroids and boundaries. Used only
              when ``cell_boundaries_data`` is not given explicitly - an
              ``Identity`` transform on a foreign labels element is never
              trusted as 1 micrometer/pixel. When omitted, resolved from a
              sibling points/shapes element's ``Scale`` transform; raises if
              neither is available.
            - ``spatial_source`` (Literal["table", "labels", "shapes"],
              optional): where ``obsm["spatial"]`` comes from. Default: keep
              an existing ``obsm["spatial"]`` (the vendor micrometre
              centroids) if present, else derive from ``cells_key`` shapes,
              else from ``cell_boundaries_data`` labels. ``"labels"``/
              ``"shapes"`` force derivation even when ``obsm["spatial"]``
              already exists; ``"table"`` requires it to already be present.
            - ``label_map`` (str, optional): the nucleus-to-cell map source.
              Either a shapes element indexed by nucleus label with a
              ``cell_id`` column (spatialdata-io's ``nucleus_boundaries``), or
              a table element with ``nucleus_label``/``cell_id`` columns
              (InSituPy dialect). Required when ``nucleus_boundaries_data`` is
              given.
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
                allowed=(
                    "table_key", "cells_key", "cell_boundaries_data", "nucleus_boundaries_data",
                    "pixel_size", "spatial_source", "label_map",
                ),
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
            pixel_size_spec = spec.get("pixel_size")
            spatial_source = spec.get("spatial_source")
            label_map_spec = spec.get("label_map")

            if spatial_source is not None and spatial_source not in ("table", "shapes", "labels"):
                raise ValueError(
                    f"cells spec for '{layer_name}': unknown spatial_source {spatial_source!r} - "
                    "expected one of 'table', 'shapes', 'labels', or None."
                )

            # Copy the table before any mutation (SD-B8): obsm['spatial'] may be overwritten
            # below, and the caller's SpatialData object must not be mutated by an import.
            table = sdata[table_key].copy()

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

                    pixel_size = _resolve_foreign_pixel_size(sdata, region, pixel_size_spec, cs, verbose)
                    cell_boundaries_data = (region, pixel_size)

            # Validate boundaries data formats (whatever the final values are, explicit or auto-detected)
            _validate_boundaries_data_format(cell_boundaries_data, param_name="cell_boundaries_data")
            _validate_boundaries_data_format(nucleus_boundaries_data, param_name="nucleus_boundaries_data")

            # Nucleus -> cell map (SD-B2): a nucleus raster has no reliable 1:1 relationship to
            # cells (multinucleated cells, non-cell-ordered nucleus labels on XOA v2-v4) - require
            # an explicit map source rather than silently falling back to the wrong identity rule.
            nucleus_to_cell_map = None
            if nucleus_boundaries_data is not None:
                if label_map_spec is None:
                    raise ValueError(
                        f"cells spec for '{layer_name}': nucleus_boundaries_data given without "
                        "label_map; pass label_map=<shapes-or-table key> (e.g. 'nucleus_boundaries' "
                        "for spatialdata-io) so nuclei are parented to the correct cells. Nucleus "
                        "rasters have no reliable 1:1 mapping."
                    )
                nucleus_to_cell_map = _resolve_nucleus_to_cell_map(sdata, label_map_spec, cell_names)

            # Spatial coordinates (SD-B1 + SD-B4 + SD-B8), resolved per `spatial_source`.
            # Default keeps an already-present obsm['spatial'] (a foreign store's own,
            # typically micrometre, centroids) instead of silently overwriting it with
            # label-derived pixel centroids.
            if spatial_source == "table":
                if spatial_key not in table.obsm:
                    raise ValueError(
                        f"cells spec for '{layer_name}': spatial_source='table' requires "
                        f"obsm['{spatial_key}'] to already be present on the table."
                    )
            elif spatial_source == "shapes":
                if not (cells_key and cells_key in sdata):
                    raise ValueError(
                        f"cells spec for '{layer_name}': spatial_source='shapes' requires a "
                        "resolvable 'cells_key' shapes element."
                    )
                table.obsm[spatial_key] = sdata[cells_key].centroid.get_coordinates().values
            elif spatial_source == "labels":
                if cell_boundaries_data is None:
                    raise ValueError(
                        f"cells spec for '{layer_name}': spatial_source='labels' requires a "
                        "resolvable 'cell_boundaries_data'."
                    )
                label_key, label_pixel_size = cell_boundaries_data
                label_array = _get_base_resolution_array(sdata[label_key]).data
                table.obsm[spatial_key] = _centroids_from_labels(label_array, seg_mask_value, label_pixel_size)
            elif spatial_key in table.obsm:
                pass  # keep the vendor centroids already on the table (SD-B1 fix)
            elif cells_key and cells_key in sdata:
                table.obsm[spatial_key] = sdata[cells_key].centroid.get_coordinates().values
            elif cell_boundaries_data is not None:
                label_key, label_pixel_size = cell_boundaries_data
                label_array = _get_base_resolution_array(sdata[label_key]).data
                table.obsm[spatial_key] = _centroids_from_labels(label_array, seg_mask_value, label_pixel_size)
            else:
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
                    nucleus_to_cell_map=nucleus_to_cell_map,
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
    strict: bool = False,
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
        strict: If True, raise on an incompletely-written store (a cell/units layer
            present but missing its ``table``, or a cell layer with boundaries but no
            recorded ``_insitupy_seg_mask_value``) instead of warning and skipping it
            (SD-B7). Default ``False`` preserves the lenient warn-and-skip behaviour.

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
            strict=strict,
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
            strict=strict,
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
    strict: bool = False,
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
    return convert_from_spatialdata(sdata, verbose=verbose, strict=strict)


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

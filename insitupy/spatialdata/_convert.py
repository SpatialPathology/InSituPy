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
import re
import warnings
from collections import defaultdict
from numbers import Number
from typing import Literal

import anndata
import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from skimage.measure import regionprops_table
from xarray import DataArray

from insitupy._constants import (
    DEFAULT_CHUNK_SIZE_X,
    DEFAULT_CHUNK_SIZE_Y,
    MODALITIES,
    SAMPLE_STR,
    SPATIALDATA_DERIVED_MODALITIES,
)
from insitupy._core.data import InSituData
from insitupy._exceptions import InSituPyError
from insitupy.containers import BoundariesData, CellData, SpatialUnitsData
from insitupy.experiment.data import InSituExperiment
from insitupy.images.axes import ImageAxes
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)


# obs column names the SpatialData-dialect writer injects as bookkeeping and the
# reader strips again (SD-B3). A user obs column of one of these names would be
# silently overwritten on write and dropped on read, so the writer refuses it.
_RESERVED_CELL_OBS = ("cell_id", "region", "_insitupy_seg_mask_value")
_RESERVED_UNIT_OBS = ("unit_id", "region")


def _assert_no_reserved_obs_collision(adata, reserved, layer_desc):
    """Raise if a user table's ``obs`` already carries a dialect-reserved column name (SD-B3).

    Must be called on the user's table before the writer injects its own
    bookkeeping columns, so it inspects only user-provided columns.
    """
    collisions = [name for name in reserved if name in adata.obs.columns]
    if collisions:
        raise InSituPyError(
            f"{layer_desc} has obs column(s) {collisions} that collide with reserved "
            f"SpatialData dialect bookkeeping name(s) {list(reserved)}. These names are "
            "reserved for round-trip identity and would be overwritten on export and "
            "dropped on read. Rename the column(s) before exporting to SpatialData."
        )


def _assert_key_absent(d, key, modality_desc="element"):
    """Raise if ``key`` is already present in ``d`` (SD-B5) - refuse a silent overwrite.

    Two source names differing only by ``.``/``-`` (both sanitised to ``_`` by
    :func:`_generate_spatialdata_key`) or by case collapse to one dialect key; the
    second would otherwise silently overwrite the first in a modality's element dict.
    """
    if key in d:
        raise InSituPyError(
            f"Two {modality_desc}s map to the same SpatialData dialect key '{key}'. "
            "This happens when names differ only by '.', '-' (both become '_') or case. "
            "Rename one so the names differ by more than those characters."
        )


def _generate_spatialdata_key(
    sample_id: str,
    modality: Literal[MODALITIES], # type: ignore
    locator: str | tuple | list | None
    ):
    # from spatialdata._core.validation import check_valid_name
    if modality.lower() not in MODALITIES and modality.lower() not in SPATIALDATA_DERIVED_MODALITIES:
        raise ValueError(
            f"Modality '{modality}' not recognized. "
            f"Choose from {MODALITIES + SPATIALDATA_DERIVED_MODALITIES}."
        )

    if modality == "transcripts":
        if locator is not None:
            raise ValueError("Locator must be None for modality 'transcripts'.")
    else:
        if locator is None:
            raise ValueError(f"Locator cannot be None for modality '{modality}'.")

    if sample_id is None:
        sample_part = ""
    else:
        # A uid is identity - never silently sanitise it. A ".." in a uid would
        # collide with the "SAMPLE.<uid>.." prefix separator and make the key
        # un-parseable (SD-B10), so reject it; a single "." round-trips fine
        # (_parse_dialect_key splits the prefix on "..") but is worth a warning.
        if ".." in str(sample_id):
            raise InSituPyError(
                f"Sample uid '{sample_id}' contains '..', which collides with the "
                f"'{SAMPLE_STR}.<uid>..' dialect key separator and cannot be encoded. "
                "Rename the sample uid."
            )
        if "." in str(sample_id):
            logger.warning(
                f"Sample uid '{sample_id}' contains '.'; it round-trips through the "
                "SpatialData dialect but a uid without '.' is recommended."
            )
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


def _parse_dialect_key(key: str) -> tuple[str | None, str, list[str]]:
    """Parse a dialect element key into ``(sample_uid, modality, locator_parts)``.

    The precise inverse of :func:`_generate_spatialdata_key`. ``sample_uid`` is
    ``None`` when the key carries no ``SAMPLE.<uid>..`` prefix (a bare
    ``InSituData`` export). ``modality`` is upper-case, matching how
    :func:`_generate_spatialdata_key` emits it.
    """
    if key.startswith(f"{SAMPLE_STR}."):
        # Split off the "SAMPLE.<uid>.." prefix on the ".." delimiter first, so a uid
        # that itself contains a single "." (SD-B10) is not mis-split. The write side
        # rejects a uid containing "..", so the first ".." is always the prefix separator.
        prefix, _sep, rest_str = key.partition("..")
        sample_uid = prefix[len(SAMPLE_STR) + 1:]
        rest = rest_str.split(".")
    else:
        sample_uid = None
        rest = key.split(".")

    modality = rest[0]
    locator_parts = rest[1:]
    return sample_uid, modality, locator_parts


def _group_elements_by_sample(sdata: SpatialData) -> dict[str | None, dict[str, tuple[str, object]]]:
    """Group SpatialData elements by sample uid, per the ``SAMPLE.<uid>..`` dialect prefix.

    Returns a dict mapping sample uid (or a single ``None`` key when the store
    carries no ``SAMPLE.`` prefix at all - a bare ``InSituData`` export) to a
    dict of ``{element_key: (elem_type, elem)}`` for that sample's elements,
    with keys stripped of the ``SAMPLE.<uid>..`` prefix.
    """
    samples: dict[str | None, dict[str, tuple[str, object]]] = defaultdict(dict)
    for elem_type, key, elem in sdata.gen_elements():
        sample_uid, modality, locator_parts = _parse_dialect_key(key)
        stripped_key = ".".join([modality, *locator_parts]) if locator_parts else modality
        samples[sample_uid][stripped_key] = (elem_type, elem)
    return dict(samples)


def _transform_anndata_for_spatialdata(
    adata: AnnData,
    #cells_as_circles: bool = True
    cells_key: str,
    cell_area_key: str | None = "cell_area",
    seg_mask_value: np.ndarray | None = None
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
        seg_mask_value: Optional per-cell segmentation mask values, in the same
            row order as ``adata.obs_names``. When given, stored under the
            reserved ``_insitupy_seg_mask_value`` obs column so a reader can
            recover which raster pixel value corresponds to which cell name
            without assuming a contiguous 1..N mapping.

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

    if seg_mask_value is not None:
        adata.obs["_insitupy_seg_mask_value"] = seg_mask_value

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
            img = image_list[0] if isinstance(image_list, list) else image_list
            axes_isp = xd.images.metadata[name]["axes"]
            axes_config = ImageAxes(axes_isp)
            is_rgb = xd.images.metadata[name]["rgb"]

            dict_key = _generate_spatialdata_key(
                sample_id=sample_id,
                modality="images",
                locator=name
            )
            _assert_key_absent(images, dict_key, "image")
            if is_rgb:
                axes = tuple(list(axes_isp.lower().replace("s", "c")))
                array = DataArray(
                    data=img,
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

        # Build the coordinate / feature / instance mapping from the columns actually
        # present (SD-B9): a 2D transcript frame carries no "z_location", and an
        # unassigned frame carries no "cell_id" - neither should raise KeyError inside
        # spatialdata. x/y and feature_name are genuinely required, so raise clearly.
        missing_xy = [c for c in ("x_location", "y_location") if c not in df.columns]
        if missing_xy:
            raise InSituPyError(
                f"Transcript frame is missing required coordinate column(s) {missing_xy}; "
                "cannot export transcripts to SpatialData."
            )
        coordinates = {"x": "x_location", "y": "y_location"}
        if "z_location" in df.columns:
            coordinates["z"] = "z_location"

        if "feature_name" not in df.columns:
            raise ValueError(
                "Transcript frame is missing the required 'feature_name' column; "
                "cannot export transcripts to SpatialData."
            )

        instance_key = "cell_id" if "cell_id" in df.columns else None

        if isinstance(df, dd.DataFrame):
            # Pre-compute known categories for the feature column. Without this,
            # PointsModel.parse(..., sort=True) treats the categories as unknown and
            # pays for an extra internal dask pass to determine them (measured ~2.25x
            # slower on a real 42.6M-row transcript table).
            df = df.assign(feature_name=df["feature_name"].astype("category").cat.as_known())
        parsed_points = PointsModel.parse(
            df,
            coordinates=coordinates,
            feature_key="feature_name",
            instance_key=instance_key,
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
            celldata = xd.cells[cell_key]
            if celldata.table is not None:
                _assert_no_reserved_obs_collision(
                    celldata.table, _RESERVED_CELL_OBS, f"Cell layer '{cell_key}'"
                )
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

                seg_mask_value = None
                if celldata.boundaries is not None:
                    seg_mask_value = celldata.boundaries.seg_mask_value.compute()

                adata, circles_sized, circles = _transform_anndata_for_spatialdata(
                    celldata.table,
                    cells_key=circles_dict_key,
                    seg_mask_value=seg_mask_value
                    )

                # see https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/tables.html#construct-a-table-annotating-1-or-more-spatialelements
                _assert_key_absent(tables, tables_key, "cell table")
                _assert_key_absent(cell_shapes, circles_dict_key, "cell shapes")
                tables[tables_key] = TableModel.parse(adata)
                cell_shapes[circles_dict_key] = circles

                if circles_sized is not None:
                    circles_sized_dict_key = _generate_spatialdata_key(
                        sample_id=sample_id,
                        modality="cells",
                        locator=[cell_key, "circles_sized"]
                    )

                    # add sized circles
                    _assert_key_absent(cell_shapes, circles_sized_dict_key, "cell shapes")
                    cell_shapes[circles_sized_dict_key] = circles_sized

    return tables, cell_shapes


def _transform_nucleus_map_for_spatialdata(
    xd: InSituData,
    sample_id: str | None = None
    ):
    """Convert per-cell layer nucleus-to-cell mappings into small SpatialData tables.

    For each cell layer whose boundaries carry a populated
    ``nucleus_to_cell_map`` (multinucleated-cell support, Xenium v2.0+),
    builds a dedicated table - one row per mapped nucleus, independent of and
    much smaller than the main cells table - annotating the layer's
    ``nuclei`` :class:`~spatialdata.models.Labels2DModel` raster via
    SpatialData's own ``region``/``instance_key`` mechanism. This keeps the
    mapping entirely out of the main cells table's ``obs``. ``nucleus_count``
    is not exported; it is derivable on read via a simple group-by on this
    table. Nuclei with no valid cell assignment (orphan nuclei, or a stale
    map entry after boundaries were filtered without a ``.sync()``) are
    skipped rather than exported with a bogus ``cell_id``.

    Args:
        xd: Source :class:`InSituData` object.
        sample_id: Optional prefix for SpatialData element keys.

    Returns:
        A dict mapping SpatialData element keys to parsed TableModel objects,
        or an empty dict if no cell layer has a populated ``nucleus_to_cell_map``.
    """
    tables = {}
    if xd.cells is not None:
        for cell_key in xd.cells.keys():
            celldata = xd.cells[cell_key]
            if celldata.boundaries is None:
                continue

            mapping = celldata.boundaries.nucleus_to_cell_map
            if mapping is None:
                continue

            cell_names = celldata.table.obs_names
            cell_names_set = set(cell_names)
            nucleus_labels, cell_ids = [], []
            n_skipped = 0
            for nucleus_idx, cell_name in mapping.items():
                if cell_name not in cell_names_set:
                    # orphan nucleus (no assigned cell) or a stale map entry
                    n_skipped += 1
                    continue
                nucleus_labels.append(nucleus_idx + 1)
                cell_ids.append(cell_name)

            if n_skipped > 0:
                warnings.warn(
                    f"Skipped {n_skipped} nucleus_to_cell_map entr{'y' if n_skipped == 1 else 'ies'} "
                    f"for cell layer '{cell_key}' referencing an unknown cell name (orphan nucleus, "
                    "or a stale map from before CellData.sync()).", stacklevel=2)

            if not nucleus_labels:
                continue

            nucleus_map_obs = pd.DataFrame({
                "nucleus_label": nucleus_labels,
                "cell_id": cell_ids,
            })

            nuclei_labels_key = _generate_spatialdata_key(
                sample_id=sample_id,
                modality="cells",
                locator=[cell_key, "boundaries", "nuclei"]
            )

            nucleus_map_adata = AnnData(X=np.empty((len(nucleus_map_obs), 0)), obs=nucleus_map_obs)
            nucleus_map_adata.obs["region"] = nuclei_labels_key
            nucleus_map_adata.obs["region"] = nucleus_map_adata.obs["region"].astype("category")
            nucleus_map_adata.uns["spatialdata_attrs"] = {
                "region": nuclei_labels_key,
                "region_key": "region",
                "instance_key": "nucleus_label",
            }

            dict_key = _generate_spatialdata_key(
                sample_id=sample_id,
                modality="cells",
                locator=[cell_key, "nucleus_map"]
            )
            _assert_key_absent(tables, dict_key, "nucleus-map table")
            tables[dict_key] = TableModel.parse(nucleus_map_adata)

    return tables


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
                _assert_no_reserved_obs_collision(
                    adata, _RESERVED_UNIT_OBS, f"Units layer '{unit_key}'"
                )
                adata.obs["unit_id"] = adata.obs.index
                adata.obs["region"] = shapes_key
                adata.obs["region"] = adata.obs["region"].astype("category")
                adata.uns["spatialdata_attrs"] = {
                    "region": shapes_key,
                    "region_key": "region",
                    "instance_key": "unit_id",
                }

                _assert_key_absent(tables, table_dict_key, "units table")
                _assert_key_absent(unit_shapes, shapes_key, "units shapes")
                tables[table_dict_key] = TableModel.parse(adata)
                unit_shapes[shapes_key] = ShapesModel.parse(sud.shapes)
    return tables, unit_shapes


def _transform_concat_tables_for_spatialdata(
    experiment: "InSituExperiment",
    exported_keys: set,
    ):
    """Serialize each built ``build_table()`` union table as a ``TABLES.<layer>`` element.

    Only cell layers with an on-disk built table (``experiment.table.keys()``) are considered -
    this is what makes the concatenated element "opt-in": it is exported iff ``build_table()``
    was called for that layer. Each row is linked back to its origin sample's real
    ``CELLS.<layer>.circles`` instance via a per-row ``region``/``cell_id`` (instance_key) pair
    reconstructed from the table's own ``label_col`` obs column and the recorded
    ``make_obs_names_unique`` build parameter - not just a plausible-looking region list. See
    ``insitupy/spatialdata/DIALECT.md`` for the full element spec.

    Args:
        experiment: Source InSituExperiment object.
        exported_keys: Element keys already produced for per-sample modalities earlier in this
            export - used to confirm every referenced per-sample circles element actually exists
            in the store before referencing it in ``region``.

    Returns:
        A dict mapping SpatialData element keys (e.g. ``"TABLES.main"``) to parsed TableModel
        objects. A layer is skipped (with a warning) if its on-disk table carries no
        gene-presence record (a pre-format-version table), or if any covered sample's circles
        element is missing from *exported_keys* (a table built from a different set of samples
        than the current export) - the region list must reference only elements that actually
        exist, so a partially-resolvable layer is omitted rather than partially exported.
    """
    tables = {}
    for cells_layer in experiment.table.keys():
        table_path = experiment._get_table_path(cells_layer)
        if table_path is None or not table_path.exists():
            logger.warning(
                f"Table path for cells layer '{cells_layer}' not found; skipping its "
                "concatenated-table SpatialData export."
            )
            continue

        adata = anndata.read_zarr(table_path)

        if "_insitupy_presence_labels" not in adata.uns or "_insitupy_gene_presence" not in adata.uns:
            logger.warning(
                f"Concatenated table for cells layer '{cells_layer}' has no gene-presence "
                "record (a pre-format-version table?) - skipping its SpatialData export."
            )
            continue

        build_params = dict(adata.uns.get("_insitupy_build_params", {}))
        label_col = build_params.get("label_col", "uid")
        make_obs_names_unique = build_params.get("make_obs_names_unique", True)

        # Restrict to only the samples `experiment` currently represents. A no-op for a
        # full InSituExperiment whose metadata already covers every built sample; for an
        # InSituExperimentView, narrows the raw union table down to just the view's own
        # samples. Row-only - gene/var columns and the presence matrix in `.uns` are left
        # untouched, preserving the existing "export the raw union, reconstruct genes on
        # read" contract for both cases. Must happen before `presence_labels` is derived,
        # so an out-of-scope sample's label never reaches `label_to_circles_key` below and
        # can never be spuriously flagged "unresolved" against this export's `exported_keys`.
        if label_col in adata.obs.columns and label_col in experiment._metadata.columns:
            covered_ids = set(str(v) for v in experiment._metadata[label_col])
            row_mask = adata.obs[label_col].astype(str).isin(covered_ids)
            if not row_mask.all():
                adata = adata[row_mask.values].copy()

        if adata.n_obs == 0:
            logger.warning(
                f"No samples in this export are covered by the concatenated table "
                f"for cells layer '{cells_layer}'; skipping its SpatialData export."
            )
            continue

        if label_col in adata.obs.columns:
            presence_labels = sorted(set(adata.obs[label_col].astype(str)))
        else:
            presence_labels = [str(label) for label in adata.uns["_insitupy_presence_labels"]]

        if label_col == "uid":
            label_to_uid = {label: label for label in presence_labels}
        elif label_col in experiment._metadata.columns:
            label_to_uid = {
                str(row[label_col]): row["uid"]
                for _, row in experiment._metadata.iterrows()
            }
        else:
            label_to_uid = {}

        label_to_circles_key = {}
        unresolved = []
        for label in presence_labels:
            uid = label_to_uid.get(label)
            circles_key = None
            if uid is not None:
                circles_key = _generate_spatialdata_key(
                    sample_id=uid,
                    modality="cells",
                    locator=[cells_layer, "circles"],
                )
            if circles_key is None or circles_key not in exported_keys:
                unresolved.append(label)
                continue
            label_to_circles_key[label] = circles_key

        if unresolved:
            logger.warning(
                f"Concatenated table for cells layer '{cells_layer}': could not resolve a "
                f"valid, exported circles element for label(s) {unresolved} - skipping this "
                "layer's SpatialData export (likely a table built from a different set of "
                "samples than the current export)."
            )
            continue

        obs_label = adata.obs[label_col].astype(str)
        row_region = obs_label.map(label_to_circles_key)

        if make_obs_names_unique:
            obs_names = adata.obs_names.astype(str)
            suffixes = "-" + obs_label
            orig_names = [
                name[: -len(suffix)] if name.endswith(suffix) else name
                for name, suffix in zip(obs_names, suffixes, strict=True)
            ]
        else:
            orig_names = list(adata.obs_names.astype(str))

        adata.obs["cell_id"] = orig_names
        region_list = sorted(set(row_region))
        adata.obs["region"] = pd.Categorical(row_region, categories=region_list)
        adata.uns["spatialdata_attrs"] = {
            "region": region_list,
            "region_key": "region",
            "instance_key": "cell_id",
        }

        key = _generate_spatialdata_key(
            sample_id=None,
            modality="tables",
            locator=cells_layer,
        )
        _assert_key_absent(tables, key, "concatenated table")
        tables[key] = TableModel.parse(adata)

    return tables


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
                    # get list of available boundaries
                    labels_list = celldata.boundaries[name]
                    if labels_list is None:
                        # e.g. an unpopulated "nuclei" slot when nuclei_boundaries=None was
                        # passed to add_boundaries() - metadata still carries the key, but
                        # there is no raster to export.
                        continue

                    meta = celldata.boundaries.metadata[name]
                    pixel_size = meta["pixel_size"]
                    transformations = {"global": Scale([pixel_size, pixel_size], axes=("x", "y"))}

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

                    _assert_key_absent(cell_boundaries, dict_key, "cell boundary label")
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

            _assert_key_absent(shapes, dict_key, "annotation")
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

            _assert_key_absent(shapes, dict_key, "region")
            shapes[dict_key] = gdf
    return shapes


def _merge_dicts_with_warning(*dicts):
    """Merge element dicts, raising on any duplicate key (SD-B5 backstop).

    The per-modality builders already refuse within-modality collisions via
    :func:`_assert_key_absent`; this catches a duplicate arising when the
    per-modality / per-sample dicts are combined (e.g. two samples sharing a uid).
    """
    merged = {}
    for d in dicts:
        for key in d:
            if key in merged:
                raise InSituPyError(
                    f"Duplicate SpatialData dialect key '{key}' while merging elements. "
                    "Two elements (possibly from different samples sharing a uid, or names "
                    "differing only by '.', '-' or case) map to the same key."
                )
        merged.update(d)
    return merged


def _extract_pixel_size_from_element(
    elem,
    coordinate_system: str = "global",
    verbose: bool = True,
    ) -> float:
    """Extract the pixel size (µm/pixel) from an already-fetched SpatialData element's transform.

    InSituPy's own writer always uses a single ``"global"`` coordinate system
    with a ``Scale`` (or, for untransformed elements, an ``Identity``)
    transformation, so this is unambiguous for InSituPy-dialect stores.

    Args:
        elem: An already-fetched SpatialData element (e.g. ``sdata[key]``).
        coordinate_system: Coordinate system to resolve the transform in.
        verbose: If True, log the extracted value.

    Returns:
        The pixel size in micrometers per pixel.

    Raises:
        ValueError: If the element's transformation is neither ``Scale`` nor
            ``Identity``.
    """
    transform = get_transformation(element=elem, to_coordinate_system=coordinate_system)

    if isinstance(transform, Scale):
        # transform.axes is not guaranteed to be ("x", "y") in that order or
        # length - verified empirically that after a disk write/read round
        # trip, spatialdata rewrites the Scale to cover every array dim
        # (e.g. ("c", "y", "x")), so scale[0] can silently become the channel
        # axis's 1.0 instead of the pixel size. Look up "x" explicitly
        # (pixel size is always isotropic in this dialect, so "x" and "y"
        # give the same value).
        axis_index = transform.axes.index("x") if "x" in transform.axes else 0
        pixel_size = transform.scale[axis_index].item()
    elif isinstance(transform, Identity):
        pixel_size = 1.0
    else:
        raise ValueError(f"Transformation type '{type(transform)}' not supported for pixel size extraction.")

    if verbose:
        logger.info(f"Extracted pixel size {pixel_size}")
    return pixel_size


def _get_base_resolution_array(elem):
    """Return the base-resolution DataArray from a possibly-multiscale SpatialData element.

    A pyramidal (multiscale) ``Image2DModel``/``Labels2DModel`` element (the
    default for InSituPy's own writer) loads as a ``DataTree`` whose own
    ``.dims``/``.data``/``.coords`` are empty; the actual array lives at
    ``elem.scale0["image"]``. A non-pyramidal element loads as a plain
    DataArray already - verified empirically against ``spatialdata==0.8.0``
    for both Image2DModel and Labels2DModel. Mirrors the same
    try/except pattern already used by ``_add_images_to_insitudata``.
    """
    try:
        return elem.scale0["image"]
    except AttributeError:
        return elem


def _is_rgb_element(elem) -> bool:
    """Detect whether a parsed Image2DModel element is RGB via its own ``c`` coordinate.

    Verified empirically against ``spatialdata==0.8.0``:
    ``Image2DModel.parse(..., rgb=True)`` produces a ``c`` coordinate with
    string labels ``['r', 'g', 'b']``; non-RGB images get an integer/positional
    ``c`` coordinate.
    """
    base = _get_base_resolution_array(elem)
    c_coord = base.coords.get("c")
    if c_coord is None:
        return False
    try:
        return list(c_coord.values) == ["r", "g", "b"]
    except Exception:
        return False


def _build_images_into_insitudata(
    data: InSituData,
    images_elements: dict[str, tuple[str, object]],
    verbose: bool = True,
    ) -> None:
    """Reconstruct ``IMAGES.<name>`` elements into an :class:`InSituData`, in place.

    Auto-detects pixel size and RGB-ness from each element itself, so no
    caller-supplied image naming/pixel-size arguments are needed.
    """
    for name, (_elem_type, elem) in images_elements.items():
        pixel_size = _extract_pixel_size_from_element(elem, verbose=verbose)
        is_rgb = _is_rgb_element(elem)

        base = _get_base_resolution_array(elem)
        axes_str = "".join(base.dims).upper()
        img_data = base.data

        if is_rgb:
            axes_str = axes_str.replace("C", "S")
        elif img_data.shape[0] == 1:
            # The writer always synthesizes a length-1 channel axis for non-RGB
            # images (c_dim = axes_config.C if axes_config.C is not None else 1) -
            # undo it here to match the original array shape.
            img_data = img_data.squeeze(0)
            axes_str = axes_str[1:]

        data.images.add_image(
            image=img_data,
            channel_names=name,
            axes=axes_str,
            pixel_size=pixel_size,
            overwrite=False,
            verbose=verbose,
        )


def _build_cells_into_insitudata(
    data: InSituData,
    cells_elements: dict[str, tuple[str, object]],
    verbose: bool = True,
    strict: bool = False,
    ) -> None:
    """Reconstruct ``CELLS.<key>.*`` elements (table, boundaries, nucleus map) into an :class:`InSituData`.

    ``circles``/``circles_sized`` elements are ignored on read - the table's
    own ``obsm["spatial"]``/``.X`` are trusted directly since they were
    already synced pre-export; the circles are exported only for external
    SpatialData-based viewers.
    """
    by_cell_key: dict[str, dict[str, tuple[str, object]]] = defaultdict(dict)
    for locator_str, entry in cells_elements.items():
        cell_key, _, rest = locator_str.partition(".")
        by_cell_key[cell_key][rest] = entry

    for i, (cell_key, parts) in enumerate(by_cell_key.items()):
        table_entry = parts.get("table")
        if table_entry is None:
            msg = (
                f"Cell layer '{cell_key}' has elements but no 'table' - the SpatialData "
                "store may be partially written."
            )
            if strict:
                raise InSituPyError(msg + " (strict=True)")
            logger.warning(msg + " Skipping this layer. Pass strict=True to raise instead.")
            warnings.warn(msg + " Skipping this layer.", stacklevel=2)
            continue
        _, table = table_entry
        cell_names = table.obs_names

        boundaries = None
        cell_boundary_entry = parts.get("boundaries.cells")
        if cell_boundary_entry is not None:
            _, cell_label_elem = cell_boundary_entry

            if "_insitupy_seg_mask_value" in table.obs.columns:
                seg_mask_value = table.obs["_insitupy_seg_mask_value"].to_numpy()
            else:
                msg = (
                    f"Cell layer '{cell_key}' has boundaries but no "
                    "'_insitupy_seg_mask_value' column in its table (a "
                    "pre-dialect-v2 store?)"
                )
                if strict:
                    raise ValueError(
                        msg + " - refusing to assume a 1..N obs-order-to-mask-value "
                        "mapping under strict=True."
                    )
                seg_mask_value = np.arange(1, len(cell_names) + 1)
                logger.warning(
                    msg + " - falling back to an assumed 1..N mapping between obs "
                    "order and mask value."
                )

            nucleus_to_cell_map = None
            nucleus_count = None
            nucleus_map_entry = parts.get("nucleus_map")
            if nucleus_map_entry is not None:
                _, nmap = nucleus_map_entry
                nucleus_to_cell_map = {
                    int(row.nucleus_label) - 1: str(row.cell_id)
                    for row in nmap.obs.itertuples()
                }
                nucleus_count = (
                    nmap.obs["cell_id"].value_counts()
                    .reindex(cell_names, fill_value=0).to_numpy()
                )

            boundaries = BoundariesData(
                cell_names=cell_names,
                seg_mask_value=seg_mask_value,
                nucleus_to_cell_map=nucleus_to_cell_map,
                nucleus_count=nucleus_count,
            )

            pixel_size = _extract_pixel_size_from_element(cell_label_elem, verbose=verbose)
            nuclei_boundary_entry = parts.get("boundaries.nuclei")
            nuclei_data = None
            if nuclei_boundary_entry is not None:
                _, nuclei_label_elem = nuclei_boundary_entry
                nuclei_data = _get_base_resolution_array(nuclei_label_elem).data

            boundaries.add_boundaries(
                cell_boundaries=_get_base_resolution_array(cell_label_elem).data,
                nuclei_boundaries=nuclei_data,
                pixel_size=pixel_size,
            )

        # Drop writer-injected bookkeeping columns/uns before handing the table
        # back to the user - they were never part of the original in-memory table.
        table = table.copy()
        for col in ("cell_id", "region", "_insitupy_seg_mask_value"):
            if col in table.obs.columns:
                del table.obs[col]
        table.uns.pop("spatialdata_attrs", None)

        cd = CellData(table=table, boundaries=boundaries)
        data.cells.add_celldata(cd=cd, key=cell_key, is_main=(i == 0))


def _build_units_into_insitudata(
    data: InSituData,
    units_elements: dict[str, tuple[str, object]],
    verbose: bool = True,
    strict: bool = False,
    ) -> None:
    """Reconstruct ``UNITS.<key>.*`` elements - each unit key's own table + shapes."""
    by_unit_key: dict[str, dict[str, tuple[str, object]]] = defaultdict(dict)
    for locator_str, entry in units_elements.items():
        unit_key, _, rest = locator_str.partition(".")
        by_unit_key[unit_key][rest] = entry

    for unit_key, parts in by_unit_key.items():
        table_entry = parts.get("table")
        shapes_entry = parts.get("shapes")
        if table_entry is None or shapes_entry is None:
            missing = [n for n, e in (("table", table_entry), ("shapes", shapes_entry)) if e is None]
            msg = (
                f"Units layer '{unit_key}' is missing element(s) {missing} - the "
                "SpatialData store may be partially written."
            )
            if strict:
                raise InSituPyError(msg + " (strict=True)")
            logger.warning(msg + " Skipping this layer. Pass strict=True to raise instead.")
            warnings.warn(msg + " Skipping this layer.", stacklevel=2)
            continue
        _, table = table_entry
        _, shapes = shapes_entry

        table = table.copy()
        for col in ("unit_id", "region"):
            if col in table.obs.columns:
                del table.obs[col]
        table.uns.pop("spatialdata_attrs", None)

        su = SpatialUnitsData(shapes=shapes, data=table, unit_type=unit_key)
        data.add_units(su, key=unit_key)


def _assign_sdata_transcripts(data: InSituData, transcripts_df) -> None:
    """Rename a SpatialData transcripts frame's x/y/z coordinate columns to
    InSituPy's ``*_location`` names and assign it to ``data.transcripts``.

    Shared by both SpatialData importers (the dialect reader and the foreign
    reader). The dtype is left untouched on purpose: a categorical
    ``feature_name`` (as produced by the exporter) is preserved, and the
    serialization-time dictionary-width normalization is handled at the write
    boundary (``_save_transcripts``), not here.
    """
    rename_map = {}
    if "x" in transcripts_df.columns:
        rename_map["x"] = "x_location"
    if "y" in transcripts_df.columns:
        rename_map["y"] = "y_location"
    if "z" in transcripts_df.columns:
        rename_map["z"] = "z_location"
    if rename_map:
        transcripts_df = transcripts_df.rename(columns=rename_map)
    data.transcripts = transcripts_df


def _build_transcripts_into_insitudata(
    data: InSituData,
    transcripts_elements: dict[str, tuple[str, object]],
    verbose: bool = True,
    ) -> None:
    """Reconstruct the (at most one) ``TRANSCRIPTS`` element into an :class:`InSituData`."""
    if not transcripts_elements:
        return
    _, transcripts_df = next(iter(transcripts_elements.values()))
    _assign_sdata_transcripts(data, transcripts_df)


def _build_annotations_into_insitudata(
    data: InSituData,
    annotations_elements: dict[str, tuple[str, object]],
    verbose: bool = True,
    ) -> None:
    """Reconstruct ``ANNOTATIONS.<key>`` elements into an :class:`InSituData`."""
    for key, (_elem_type, elem) in annotations_elements.items():
        data.annotations.add_data(data=elem, key=key, scale_factor=1.0, verbose=verbose)


def _build_regions_into_insitudata(
    data: InSituData,
    regions_elements: dict[str, tuple[str, object]],
    verbose: bool = True,
    ) -> None:
    """Reconstruct ``REGIONS.<key>`` elements into an :class:`InSituData`."""
    for key, (_elem_type, elem) in regions_elements.items():
        data.regions.add_data(data=elem, key=key, scale_factor=1.0, verbose=verbose)


def _build_insitudata_from_elements(
    elements: dict[str, tuple[str, object]],
    slide_id: str | None = None,
    sample_id: str | None = None,
    method_params: dict | None = None,
    verbose: bool = True,
    strict: bool = False,
    ) -> InSituData:
    """Reconstruct a single :class:`InSituData` from one sample's grouped SpatialData elements.

    Args:
        elements: Maps dialect-stripped element keys (no ``SAMPLE.<uid>..``
            prefix) to ``(elem_type, elem)`` tuples, as returned per-sample by
            :func:`_group_elements_by_sample`.
        slide_id: Slide identifier to set on the reconstructed object, if known.
        sample_id: Sample identifier to set on the reconstructed object, if known.
        method_params: Forwarded to :class:`InSituData`'s ``method_params``.
        verbose: If True, log progress for each modality.

    Returns:
        A fully populated, in-memory :class:`InSituData` with no backing
        project directory (``.saveas()`` is required before it can be saved).
    """
    data = InSituData(
        path=None,
        slide_id=slide_id,
        sample_id=sample_id,
        method_name="",
        method_params=method_params or {},
    )

    by_modality: dict[str, dict[str, tuple[str, object]]] = defaultdict(dict)
    for key, entry in elements.items():
        _, modality, locator_parts = _parse_dialect_key(key)
        locator_str = ".".join(locator_parts) if locator_parts else ""
        by_modality[modality][locator_str] = entry

    _build_images_into_insitudata(data, by_modality.get("IMAGES", {}), verbose=verbose)
    _build_cells_into_insitudata(data, by_modality.get("CELLS", {}), verbose=verbose, strict=strict)
    _build_units_into_insitudata(data, by_modality.get("UNITS", {}), verbose=verbose, strict=strict)
    _build_transcripts_into_insitudata(data, by_modality.get("TRANSCRIPTS", {}), verbose=verbose)
    _build_annotations_into_insitudata(data, by_modality.get("ANNOTATIONS", {}), verbose=verbose)
    _build_regions_into_insitudata(data, by_modality.get("REGIONS", {}), verbose=verbose)

    return data


def _sanitize_channel_name(name) -> str:
    """Sanitize a channel name for use as an ``ImageData`` key (SD-B11).

    ``ImageData.save`` writes each channel to ``<name>.zarr`` under the output
    directory (``insitupy/containers/image_data.py``), so a literal path
    separator in a real channel name (e.g. a Xenium morphology channel named
    ``"ATP1A1/CD45/E-Cadherin"``) would otherwise be read as a subdirectory.
    Replaces filesystem-unsafe characters with an underscore.
    """
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def _add_images_to_insitudata(
    data: InSituData,
    sdata: SpatialData,
    images: dict[str, dict],
    verbose: bool
):
    """Add images to InSituData from keyed image specs.

    Args:
        data: InSituData object to add images to.
        sdata: SpatialData object containing the images.
        images: {name: spec}, where spec is a dict with keys:
            - key (str, required): SpatialData image element key.
            - pixel_size (Number, required): microns/pixel.
            - is_rgb (bool, optional, default False): forwarded to ``add_image``.
        verbose: Whether to print status messages.
    """

    for name, spec in images.items():
        _validate_foreign_spec(
            spec, name, "images",
            required=("key", "pixel_size"),
            allowed=("key", "pixel_size", "is_rgb"),
        )
        key = spec["key"]
        pixel_size = spec["pixel_size"]
        is_rgb = spec.get("is_rgb", False)
        if not isinstance(key, str):
            raise TypeError(f"images spec for '{name}': 'key' must be a string, got {type(key)}.")
        if not isinstance(pixel_size, Number):
            raise TypeError(f"images spec for '{name}': 'pixel_size' must be a number, got {type(pixel_size)}.")
        if not isinstance(is_rgb, bool):
            raise TypeError(f"images spec for '{name}': 'is_rgb' must be a boolean, got {type(is_rgb)}.")

        if key not in sdata:
            if verbose:
                logger.warning(f"Image key '{key}' not found in SpatialData")
            continue
        img_data = sdata[key]
        data_array = _get_base_resolution_array(img_data)

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
            n_channels = da_img.shape[axes.C]
            channel_names = None
            if "c" in data_array.coords:
                c_values = data_array.coords["c"].values
                if len(c_values) == n_channels:
                    channel_names = [_sanitize_channel_name(v) for v in c_values]
            if channel_names is None:
                # Fall back to numbered channels when no usable `c` coordinate is present
                # (SD-B11: real channel names, e.g. "DAPI"/"ATP1A1/CD45/E-Cadherin", are
                # preferred whenever the DataArray actually carries them).
                channel_names = [f"{name}_{i}" for i in range(n_channels)]

        data.images.add_image(
            image=da_img,
            channel_names=channel_names,
            axes=axes_str,
            pixel_size=pixel_size,
            is_rgb=is_rgb,
            overwrite=False,
            verbose=verbose
        )


def _create_boundaries_from_spatialdata(
    sdata: SpatialData,
    cell_names: np.ndarray,
    seg_mask_value: np.ndarray,
    cell_boundaries_data: tuple[str, Number] | None = None, # tuple as (cell_boundaries_key, pixel_size)
    nucleus_boundaries_data: tuple[str, Number] | None = None, # tuple as (nucleus_boundaries_key, pixel_size)
    nucleus_to_cell_map: dict[int, str] | None = None,
    ) -> BoundariesData:
    """Create BoundariesData from SpatialData labels.

    Cell and nucleus boundaries are independently optional and may carry
    independent pixel sizes (a foreign store's cell/nucleus label rasters
    are not guaranteed to share a resolution, unlike InSituPy's own
    exporter). ``seg_mask_value`` is supplied by the caller - this function
    does not fabricate it.

    ``nucleus_to_cell_map`` (SD-B2), when given, is passed straight through
    to the ``BoundariesData`` constructor so nuclei are parented to their
    real parent cell instead of falling back to the 1:1 identity rule.
    """
    if nucleus_boundaries_data is not None and cell_boundaries_data is None:
        raise ValueError(
            "cell_boundaries_data must be provided when nucleus_boundaries_data is given "
            "without cell boundaries - nucleus-only masks are not supported."
        )

    boundaries = BoundariesData(
        cell_names=cell_names,
        seg_mask_value=seg_mask_value,
        nucleus_to_cell_map=nucleus_to_cell_map,
    )

    # Add cell boundaries if provided
    cell_bounds = None
    cell_pixel_size = None
    if cell_boundaries_data is not None:
        cell_boundaries_key, cell_pixel_size = cell_boundaries_data
        if cell_boundaries_key and cell_boundaries_key in sdata:
            cell_bounds = _get_base_resolution_array(sdata[cell_boundaries_key]).data

    # Add nucleus boundaries if provided
    nuc_bounds = None
    nucleus_pixel_size = None
    if nucleus_boundaries_data is not None:
        nucleus_boundaries_key, nucleus_pixel_size = nucleus_boundaries_data
        if nucleus_boundaries_key and nucleus_boundaries_key in sdata:
            nuc_bounds = _get_base_resolution_array(sdata[nucleus_boundaries_key]).data

    if cell_bounds is not None or nuc_bounds is not None:
        if cell_bounds is None:
            raise ValueError(
                "cell_boundaries_data resolved to no array in sdata - cannot add "
                "nucleus-only boundaries."
            )
        boundaries.add_boundaries(
            cell_boundaries=cell_bounds,
            pixel_size=cell_pixel_size,
            nuclei_boundaries=nuc_bounds,
            nucleus_pixel_size=nucleus_pixel_size,
        )

    return boundaries


def _centroids_from_labels(
    label_array,
    seg_mask_value: np.ndarray,
    pixel_size: Number,
    ) -> np.ndarray:
    """Derive (x, y) centroids for each mask id from a label raster.

    Used when no shapes/circles element is available to source centroids
    from (a fully labels-native foreign store). Materializes the label
    array into memory once (unavoidable without precomputed centroids -
    bounded by the same array already being loaded for ``BoundariesData``
    regardless) and runs :func:`skimage.measure.regionprops_table` over it.

    Args:
        label_array: 2-D label mask (dask or numpy array).
        seg_mask_value: Ordered array of mask ids to return centroids for,
            in the order they should appear in the output (typically
            matching ``table.obs`` order).
        pixel_size: Isotropic pixel size (µm/pixel) to scale pixel-space
            centroids into physical coordinates. Only ``Scale``/``Identity``
            transforms are supported anywhere in this importer, so no
            affine/rotation handling is attempted here either.

    Returns:
        Array of shape ``(len(seg_mask_value), 2)`` with ``(x, y)`` centroids.

    Raises:
        ValueError: If any value in ``seg_mask_value`` has no matching
            region in the label mask.
    """
    mask = np.asarray(label_array.compute() if hasattr(label_array, "compute") else label_array)

    props = regionprops_table(mask, properties=("label", "centroid"))
    # regionprops_table's centroid-0/centroid-1 are (row, col) = (y, x).
    # Cast label keys to plain int - seg_mask_value and the mask's own label
    # dtype are not guaranteed to be the same numpy integer type.
    by_label = {
        int(label): (x, y)
        for label, x, y in zip(props["label"], props["centroid-1"], props["centroid-0"], strict=True)
    }

    missing = [v for v in seg_mask_value if int(v) not in by_label]
    if missing:
        raise ValueError(
            f"seg_mask_value contains {len(missing)} value(s) with no matching region in the "
            f"label mask (e.g. {missing[:5]}) - mask and table disagree on segmentation identity."
        )

    centroids = np.array([by_label[int(v)] for v in seg_mask_value], dtype=float)
    centroids *= pixel_size
    return centroids

def _validate_foreign_spec(
    spec: dict,
    layer_name: str,
    modality: str,
    required: tuple[str, ...],
    allowed: tuple[str, ...],
) -> None:
    """
    Validate a single keyed-dict spec entry for ``convert_from_foreign_spatialdata``.

    Args:
        spec: The spec dict for one layer/image name.
        layer_name: The dict key (layer/image name) this spec belongs to - used
            in error messages.
        modality: One of "images", "cells", "units" - used in error messages.
        required: Required spec keys.
        allowed: All allowed spec keys (superset of ``required``).

    Raises:
        TypeError: If spec is not a dict.
        ValueError: If a required key is missing or an unknown key is present.
    """
    if not isinstance(spec, dict):
        raise TypeError(
            f"{modality} spec for '{layer_name}' must be a dict, got {type(spec)}."
        )

    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(
            f"{modality} spec for '{layer_name}' is missing required key(s): {missing}. "
            f"Allowed keys: {list(allowed)}"
        )

    unknown = [k for k in spec if k not in allowed]
    if unknown:
        raise ValueError(
            f"{modality} spec for '{layer_name}' has unknown key(s): {unknown}. "
            f"Allowed keys: {list(allowed)}"
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

import contextlib
import logging
import os
import shutil
import sys
import warnings
from contextlib import ExitStack
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import Literal
from warnings import warn

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import scanpy as sc
import zarr
from zarr.errors import ArrayNotFoundError

from insitupy._exceptions import StaleNucleusMapWarning
from insitupy._io.files import atomic_replace_dir, check_overwrite_and_remove_if_true, read_json

logger = logging.getLogger(__name__)
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData
from insitupy.containers.image_data import ImageData
from insitupy.containers.multi_cell_data import MultiCellData
from insitupy.containers.multi_spatial_units_data import MultiSpatialUnitsData
from insitupy.containers.shapes_data import AnnotationsData, RegionsData, ShapesData
from insitupy.containers.spatial_units_data import SpatialUnitsData
from insitupy.containers._zarr_compat import ZARR_V3, _get_zarr_store
from insitupy.images.io import _sorted_pyramid_levels
from insitupy.utils.utils import (
    _encode_xenium_cell_ids,
    _generate_time_based_uid,
    convert_to_list,
    glob_visible,
)

# Categorical transcript columns (e.g. feature_name) are written with a uniformly
# wide dictionary index so that independently-written parquet partitions - each
# with its own locally-inferred int8/int16 index - all convert against one schema.
# Value type is string: every categorical transcript column produced today is
# name-like. An int/float-categorical column would need its value type derived
# from the column instead (none exist today).
_CATEGORICAL_PARQUET_DTYPE = pa.dictionary(pa.int32(), pa.string())


def _read_table_from_celldata(
    path: Path,
    metadata: dict
) -> sc.AnnData:
    """
    Read the AnnData table from CellData directory.

    Parameters
    ----------
    path : Path
        Path to the CellData directory
    metadata : dict
        Metadata dictionary from .celldata file

    Returns
    -------
    sc.AnnData
        The loaded AnnData table
    """
    try:
        table = sc.read_h5ad(path / metadata["table"])
    except KeyError:
        # backward compatibility: previously it was called matrix
        table = sc.read_h5ad(path / metadata["matrix"])
    return table


def _read_boundaries_from_celldata_zarr(
    bound_path: Path,
) -> BoundariesData:
    """
    Read BoundariesData from a zarr store.

    Parameters
    ----------
    bound_path : Path
        Path to the boundaries zarr store

    Returns
    -------
    BoundariesData
        The boundaries object with cells and nuclei masks
    """
    # check whether it is zipped or not
    suffix = bound_path.name.split(".", maxsplit=1)[-1]

    try:
        # read cell ids and seg_mask_values
        cell_names = da.from_zarr(bound_path, component="cell_names")
    except ArrayNotFoundError:
        # if cell names is not found, the data might come from an older InSituPy version which contained a cell_id instead
        try:
            # read cell ids and seg_mask_values
            cell_ids = da.from_zarr(bound_path, component="cell_id").compute()
            cell_names = _encode_xenium_cell_ids(cell_ids).astype(str)
        except ArrayNotFoundError:
            # neither cell_names nor cell_id on disk: the store lacks an identity axis, so
            # cell_names would be left unbound and blow up later with UnboundLocalError. Raise a
            # clear, domain-specific error instead (the store is readable, just incomplete).
            raise ValueError(
                f"Boundaries store at {bound_path} has neither a 'cell_names' nor a "
                "'cell_id' array; cannot determine cell identities. The store is likely "
                "incomplete or was not written by InSituPy."
            )

    try:
        # in older datasets sometimes seg_mask_value is missing
        seg_mask_value = da.from_zarr(bound_path, component="seg_mask_value")
    except ArrayNotFoundError:
        warn("No `seg_mask_value` component found in boundaries zarr storage. This can lead to problems when syncing `.boundaries` and `.table`.")
        seg_mask_value = None

    # Read the empty-map marker up front: it distinguishes a known-but-empty map ({}) from
    # None ("unknown, assume v1.x 1:1"). Absent on stores written before this attribute
    # existed, where it defaults to False and the old behaviour is preserved.
    with ExitStack() as marker_stack:
        marker_store = _get_zarr_store(bound_path, mode="r", zipped=(suffix == "zarr.zip"))
        # In Zarr v2, stores are context managers and need to be entered
        if not ZARR_V3:
            marker_store = marker_stack.enter_context(marker_store)
        root_attrs = dict(zarr.open_group(store=marker_store, mode="r").attrs)
    has_map = bool(root_attrs.get("insitupy_has_nucleus_map", False))

    # Read nucleus_to_cell_map (for multinucleated cell support, Xenium v2.0+).
    # drop_reason is set when the saved map is discarded; a single StaleNucleusMapWarning
    # is emitted for the store after the consistency check below.
    drop_reason = None
    try:
        # current format (0.12.0b7+): two parallel arrays [nucleus_index] / [cell_name]
        nuc_idx_arr = da.from_zarr(bound_path, component="nucleus_to_cell_map/nucleus_index").compute()
        cell_name_arr = da.from_zarr(bound_path, component="nucleus_to_cell_map/cell_name").compute()
        nucleus_to_cell_map = dict(zip(
            np.asarray(nuc_idx_arr).astype(np.int64).tolist(),
            np.asarray(cell_name_arr).astype(str).tolist(),
        ))
    except (ArrayNotFoundError, TypeError):
        try:
            # legacy (pre-0.12.0b7) position-based format: flat [[nucleus_idx, row_position], ...]
            legacy_arr = np.asarray(
                da.from_zarr(bound_path, component="nucleus_to_cell_map").compute()
            ).reshape(-1, 2)
            names = np.asarray(cell_names).astype(str)
            pos = legacy_arr[:, 1].astype(np.int64)
            if len(pos) == 0 or (pos.min() >= 0 and pos.max() < len(names)):
                nucleus_to_cell_map = dict(zip(
                    legacy_arr[:, 0].astype(np.int64).tolist(),
                    names[pos].tolist(),
                ))
            else:
                nucleus_to_cell_map = None
                drop_reason = "legacy"
        except (ArrayNotFoundError, TypeError):
            # no map arrays on disk: {} when the writer marked a known-empty map,
            # else None (unknown -> assume v1.x 1:1, the older-dataset behaviour)
            nucleus_to_cell_map = {} if has_map else None

    # Read nucleus_count (number of nuclei per cell)
    try:
        nucleus_count = da.from_zarr(bound_path, component="nucleus_count").compute()
    except (ArrayNotFoundError, TypeError):
        nucleus_count = None  # Not available in older datasets

    # initialize boundaries data object
    boundaries = BoundariesData(
        cell_names=cell_names,
        seg_mask_value=seg_mask_value,
        nucleus_to_cell_map=nucleus_to_cell_map,
        nucleus_count=nucleus_count
    )

    if drop_reason is None and not boundaries.nucleus_map_is_consistent(np.asarray(cell_names)):
        drop_reason = "inconsistent"
    if drop_reason is not None:
        # message is path-free so that identical warnings can be deduplicated/summarized
        # Emitted with a fresh registry (like warnings.warn(..., stacklevel=2) but without
        # "default"/"module" dedup) so that every stale store is reported and can be counted
        # by InSituExperiment load loops; a user filter on the class still silences it.
        caller = sys._getframe(1)
        warnings.warn_explicit(
            StaleNucleusMapWarning(
                "Saved nucleus_to_cell_map / nucleus_count does not match the cell table "
                "(written by an older InSituPy that did not maintain it, or legacy "
                "position-based format). Dropped it; nuclei are treated as 1:1 with cells. "
                "Harmless for cell-level analyses; re-read from raw data to restore "
                "multinucleated-cell mapping."
            ),
            StaleNucleusMapWarning,
            caller.f_code.co_filename,
            caller.f_lineno,
            module=caller.f_globals.get("__name__", "<unknown>"),
            registry={},
        )
        boundaries._nucleus_to_cell_map = None
        boundaries._nucleus_count = None

    # retrieve the boundaries data
    bound_data = {}
    meta = {}
    zipped = True if suffix == "zarr.zip" else False
    # Use ExitStack to handle context manager differences between Zarr v2 and v3
    with ExitStack() as stack:
        dirstore = _get_zarr_store(bound_path, mode="r", zipped=zipped)

        # In Zarr v2, stores are context managers and need to be entered
        if not ZARR_V3:
            dirstore = stack.enter_context(dirstore)

        # open zarr group
        root = zarr.open_group(store=dirstore, mode='r')

        for k in ["cells", "nuclei"]:
            comp = f"masks/{k}"
            if comp in root:
                # boundary masks are always stored as a pyramid group: masks/{k}/{level}
                subresolutions = _sorted_pyramid_levels(root[comp].keys())
                bound_data[k] = []
                for subres in subresolutions:
                    if not subres.startswith("."):
                        if zipped:
                            bound_data[k].append(
                                da.from_zarr(dirstore, component=f"{comp}/{subres}").persist()
                            )
                        else:
                            bound_data[k].append(
                                da.from_zarr(dirstore, component=f"{comp}/{subres}")
                            )

                # retrieve boundaries metadata
                store = zarr.open(dirstore)
                meta[k] = store[f"masks/{k}"].attrs.asdict()

    cell_boundaries = bound_data.get("cells")
    nuclei_boundaries = bound_data.get("nuclei")

    # add boundaries, reading each layer's own pixel size (they can differ: a foreign store
    # may resolve the cell and nucleus rasters independently)
    if cell_boundaries is not None:
        boundaries.add_boundaries(
            cell_boundaries=cell_boundaries,
            nuclei_boundaries=nuclei_boundaries,
            pixel_size=meta["cells"]["pixel_size"],
            nucleus_pixel_size=meta["nuclei"]["pixel_size"] if "nuclei" in meta else None,
        )
    elif nuclei_boundaries is not None:
        raise ValueError(
            f"Boundaries store at {bound_path} contains a 'nuclei' mask but no 'cells' mask. "
            "Nucleus-only boundaries are not supported."
        )
    # no masks on disk: leave `boundaries` mask-less (BoundariesData.is_empty)

    return boundaries


def _read_celldata(
    path: str | os.PathLike | Path,
    boundaries: bool = True,
) -> CellData:
    """
    Read CellData from a saved directory.

    Parameters
    ----------
    path : Union[str, os.PathLike, Path]
        Path to the CellData directory
    boundaries : bool
        If False, the segmentation boundaries (including the nucleus map) are not read
        and the returned CellData has ``boundaries=None``.

    Returns
    -------
    CellData
        The loaded CellData object
    """
    # validate path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CellData directory not found: {path}")
    celldata_metadata = read_json(path / ".celldata")

    # read table (AnnData)
    table = _read_table_from_celldata(path, celldata_metadata)

    # read boundaries (absent when CellData was saved without boundaries)
    if boundaries and "boundaries" in celldata_metadata:
        bound_path = path / celldata_metadata["boundaries"]
        boundaries_data = _read_boundaries_from_celldata_zarr(bound_path)
    else:
        boundaries_data = None

    # extract configuration
    config = celldata_metadata.get("config", {})

    # create CellData object
    celldata = CellData(table=table, boundaries=boundaries_data, config=config)

    return celldata


def _read_shapesdata(
    path: str | os.PathLike | Path,
    mode: Literal["annotations", "regions", "shapes"],
    scale_factor: Number | None = None
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ShapesData directory not found: {path}")

    # e.g. when reading from a shapesdata object, it is assumed that it was saved as µm
    if scale_factor is None:
        scale_factor = 1

    # read metadata and retrieve keys and files from it
    # metadata = read_json(path / "metadata.json")
    # keys = metadata.keys()
    # files = [path / f"{k}.geojson" for k in keys]
    files_dict = {f.stem: f for f in glob_visible(path, "*.geojson") if f.stem != "metadata"}

    # check which type of ShapesData is read here
    if mode == "annotations":
        data = AnnotationsData()
    elif mode == "regions":
        data = RegionsData()
    elif mode == "shapes":
        data = ShapesData()
    else:
        raise ValueError(f"Unknown `mode`: {mode}")

    # make sure files and keys are a list
    # files = convert_to_list(files)
    # keys = convert_to_list(keys)

    # for k, f in zip(keys, files):
    for k, f in files_dict.items():
        data.add_data(
            data=f, key=k,
            scale_factor=scale_factor
            )

    # overwrite metadata
    # data.metadata = metadata
    return data

def _read_multicelldata(
        path: str | os.PathLike | Path,
        path_upper: str | os.PathLike | Path | None = None,
        alt_path_dict: dict | None = None,
        keys: str | list[str] | None = None,
        boundaries: bool = True,
    ) -> MultiCellData:
    """Read a MultiCellData from a saved directory.

    ``keys`` restricts the read to the listed layers (default: all) and ``boundaries=False``
    skips the segmentation boundaries of every layer. A result that lacks a saved layer or
    its boundaries is marked partial (``MultiCellData.is_partial``) so that it cannot be
    saved over the complete data on disk. The saved main layer must be among ``keys``.
    """
    path = Path(path)
    if os.path.exists(path / ".multicelldata"):
        old = False
    elif os.path.exists(path / ".celldata"):
        old = True
    else:
        raise FileNotFoundError(f"Metadata file for cells dimension in {path} was not found.")

    # layer layout on disk: key -> directory, plus the saved main key
    if not old:
        celldata_metadata = read_json(path / ".multicelldata")
        key_main = celldata_metadata["key_main"]
        layer_paths = {k: path / k for k in celldata_metadata["all_keys"]}
    else:
        key_main = "main"
        layer_paths = {"main": path}
        if path_upper is not None and alt_path_dict is not None:
            path_upper = Path(path_upper)
            for k, p in alt_path_dict.items():
                layer_paths[k] = path_upper / p

    # validate the requested keys before anything is read
    if keys is None:
        selected = list(layer_paths)
    else:
        selected_set = set(convert_to_list(keys))
        unknown = [k for k in selected_set if k not in layer_paths]
        if unknown:
            raise KeyError(
                f"Cell layer(s) {sorted(unknown)} not found in {path}. "
                f"Available layers: {list(layer_paths)}."
            )
        if key_main not in selected_set:
            raise ValueError(
                f"`keys` must include the saved main layer '{key_main}' (got {sorted(selected_set)}). "
                f"Available layers: {list(layer_paths)}."
            )
        selected = [k for k in layer_paths if k in selected_set]

    mcd = MultiCellData()
    for key in selected:
        cd = _read_celldata(layer_paths[key], boundaries=boundaries)
        mcd.add_celldata(cd=cd, key=key, is_main=(key == key_main))

    if len(selected) < len(layer_paths) or not boundaries:
        mcd._partial = {"keys": list(selected), "boundaries": bool(boundaries)}
    return mcd


def _read_multispatialunitsdata(path: str | os.PathLike | Path) -> MultiSpatialUnitsData:
    path = Path(path)
    marker = path / ".multispatialunitsdata"
    musd = MultiSpatialUnitsData()
    if marker.exists():
        meta = read_json(marker)
        for key in meta["all_keys"]:
            su = SpatialUnitsData.read(path / key)
            musd.add_units(su=su, key=key, is_main=(key == meta["key_main"]))
    elif (path / "shapes.parquet").exists():
        # legacy: pre-multi-unit InSituPy versions stored one flat layer directly under `units/`
        su = SpatialUnitsData.read(path)
        musd.add_units(su=su, key="main", is_main=True)
    else:
        raise FileNotFoundError(f"No spatial units data found at {path}")
    return musd


@contextlib.contextmanager
def _discard_on_failure(path):
    """Delete the freshly created save directory *path* if the block raises.

    A half-written versioned save directory would otherwise sit next to the
    committed one and, for stores without an ``.ispy`` pointer, be mistaken for
    the newest save.
    """
    try:
        yield
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise


def _save_images(imagedata: ImageData,
                 path: str | os.PathLike,
                 metadata: dict | None = None,
                 images_as_zarr: bool = True,
                 max_resolution: Number | None = None, # in µm per pixel,
                 debug: bool = False,
                 verbose: bool = False
                 ):
    img_path = (path / "images")

    savepaths = imagedata.save(
        path=img_path,
        as_zarr=images_as_zarr,
        return_savepaths=True,
        max_resolution=max_resolution,
        debug=debug,
        verbose=verbose
        )

    #if metadata is not None:
    metadata["data"]["images"] = {}
    for n in imagedata.metadata.keys():
        s = savepaths[n]
        # collect metadata
        metadata["data"]["images"][n] = Path(relpath(s, path)).as_posix()


def _save_cells(cells: MultiCellData,
                path,
                metadata,
                max_resolution_boundaries: Number | None = None, # in µm per pixel
                overwrite=False
                ):
    # create path for cells
    uid = _generate_time_based_uid()
    cells_path = path / "cells" / uid

    # save cells to path and write info to metadata
    with _discard_on_failure(cells_path):
        cells.save(
            path=cells_path,
            max_resolution_boundaries=max_resolution_boundaries,
            overwrite=overwrite
            )

    #if metadata is not None:
    try:
        # move old celldata paths to history
        old_path = metadata["data"]["cells"]
    except KeyError:
        pass
    else:
        metadata["history"]["cells"].append(old_path)

    # move new paths to data
    metadata["data"]["cells"] = Path(relpath(cells_path, path)).as_posix()


def _save_transcripts(transcripts, path, metadata):
    # create file path
    trans_path = path / "transcripts"
    trans_path.mkdir(parents=True, exist_ok=True) # create directory
    trans_file = trans_path / "transcripts.parquet"

    # save transcripts as parquet and modify metadata
    if isinstance(transcripts, dd.DataFrame):
        # Any categorical column read back from a partitioned parquet store (e.g.
        # a SpatialData Points element) has its dictionary index width inferred
        # independently per partition (int8 for <=127 local categories, int16
        # otherwise, ...). to_parquet() infers its target schema from a single
        # partition, so a differently-sized partition later on fails to convert.
        # Forcing a wide-enough dictionary index here avoids the mismatch without
        # requiring a full-data pass to unify categories across partitions.
        schema = "infer"
        cat_cols = [
            col for col, dt in transcripts.dtypes.items()
            if isinstance(dt, pd.CategoricalDtype)
        ]
        if cat_cols:
            schema = {col: _CATEGORICAL_PARQUET_DTYPE for col in cat_cols}

        # Use the synchronous scheduler so dask reads and writes one partition
        # at a time. The default threaded scheduler reads N partitions
        # concurrently (N ≈ CPU cores), multiplying peak RAM by N. For large
        # transcript datasets this causes OOM even on systems with adequate
        # total RAM due to concurrent allocation pressure.
        transcripts.to_parquet(trans_file, schema=schema, compute_kwargs={"scheduler": "synchronous"})
    else:
        transcripts.to_parquet(trans_file)

    #if metadata is not None:
    metadata["data"]["transcripts"] = Path(relpath(trans_file, path)).as_posix()


def _save_units(units: MultiSpatialUnitsData, path, metadata, overwrite: bool = False):
    units_path = path / "units"
    if overwrite and units_path.exists():
        # Stage then swap: the existing units are only replaced once the new ones are
        # completely written, so a failure mid-write cannot leave the project without units.
        staging = path / "units.__ispy_tmp__"
        check_overwrite_and_remove_if_true(staging, overwrite=True)  # stale staging from a crash
        try:
            units.save(path=staging)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        atomic_replace_dir(staging, units_path, what="saving units")
    else:
        units.save(path=units_path, overwrite=overwrite)
    metadata["data"]["units"] = Path(relpath(units_path, path)).as_posix()


def _save_annotations(annotations, path, metadata):
    uid = _generate_time_based_uid()
    annot_path = path / "annotations" / uid

    # save annotations
    with _discard_on_failure(annot_path):
        annotations.save(annot_path)

    # metadata=None: standalone write outside a project (e.g. quicksave)
    if metadata is None:
        return

    try:
        # move old paths to history
        old_path = metadata["data"]["annotations"]
    except KeyError:
        pass
    else:
        metadata["history"]["annotations"].append(old_path)

    # add new paths
    metadata["data"]["annotations"] = Path(relpath(annot_path, path)).as_posix()


def _save_regions(regions, path, metadata):
    uid = _generate_time_based_uid()
    annot_path = path / "regions" / uid

    # save annotations
    with _discard_on_failure(annot_path):
        regions.save(annot_path)

    # metadata=None: standalone write outside a project
    if metadata is None:
        return

    try:
        # move old paths to history
        old_path = metadata["data"]["regions"]
    except KeyError:
        pass
    else:
        metadata["history"]["regions"].append(old_path)

    # add new paths
    metadata["data"]["regions"] = Path(relpath(annot_path, path)).as_posix()


# ---------------------------------------------------------------------------
# Deprecation wrappers for renamed public functions
# ---------------------------------------------------------------------------

def read_celldata(path, **kwargs):
    """Deprecated: Use CellData.read() instead."""
    warnings.warn(
        "read_celldata() is deprecated. Use CellData.read() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _read_celldata(path, **kwargs)


def read_shapesdata(path, mode, scale_factor=None):
    """Deprecated: Use ShapesData.read(), AnnotationsData.read(), or RegionsData.read() instead."""
    warnings.warn(
        "read_shapesdata() is deprecated. Use ShapesData.read() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _read_shapesdata(path, mode, scale_factor)


def read_multicelldata(path, path_upper=None, alt_path_dict=None):
    """Deprecated: Use MultiCellData.read() instead."""
    warnings.warn(
        "read_multicelldata() is deprecated. Use MultiCellData.read() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _read_multicelldata(path, path_upper, alt_path_dict)

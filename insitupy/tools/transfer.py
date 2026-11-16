from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from insitupy._core.data import InSituData
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)

_KEY_BASE = 100_000  # the Visium HD grid is < 100k bins per side
_MAPPING_KEY = "visium_hd_bin_mapping"
_MAPPING_BIN_SIZE = 2  # um; the mapping is given on the 2 um grid
_Layer = tuple[Literal["cells", "units"], str]


def _grid_key(rows, cols) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64) * _KEY_BASE + np.asarray(cols, dtype=np.int64)


def _get_layer(data: InSituData, spec: _Layer, role: str):
    if not (isinstance(spec, tuple) and len(spec) == 2):
        raise ValueError(f"`{role}` must be a tuple (kind, layer_name), e.g. ('units', 'square_008um').")
    kind, name = spec
    if kind == "cells":
        container = data.cells
    elif kind == "units":
        container = data.units
    else:
        raise ValueError(f"`{role}`: kind must be 'cells' or 'units', not '{kind}'.")
    if name not in container.keys():
        raise ValueError(f"`{role}`: {kind} layer '{name}' not found. Available: {list(container.keys())}.")
    layer = container[name]
    if layer.table is None:
        raise ValueError(f"`{role}`: {kind} layer '{name}' has no table.")
    return layer


def _bin_info(layer, role: str):
    """Return (bin_size_um, grid keys) of a Visium HD bin layer."""
    table = layer.table
    if layer.unit_type != "visium_hd_bin":
        raise ValueError(
            f"`{role}` must be a 'visium_hd_bin' units layer, but unit_type is '{layer.unit_type}'."
        )
    missing = [c for c in ("array_row", "array_col") if c not in table.obs]
    if missing or "bin_size_um" not in table.uns:
        raise ValueError(
            f"`{role}` needs obs columns 'array_row'/'array_col' and uns['bin_size_um'] "
            "(as set by `read_visium_hd`)."
        )
    keys = _grid_key(table.obs["array_row"], table.obs["array_col"])
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"`{role}` contains duplicated (array_row, array_col) positions.")
    return int(table.uns["bin_size_um"]), keys


def _ratio(big: int, small: int) -> int:
    if big % small != 0:
        raise ValueError(
            f"Bin sizes {big} um and {small} um are not nested (ratio {big / small:.3g} is not an integer)."
        )
    return big // small


def _parent_key(keys: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return keys
    return _grid_key((keys // _KEY_BASE) // k, (keys % _KEY_BASE) // k)


def _get_mapping(cell_layer) -> pd.DataFrame:
    mapping = cell_layer.table.uns.get(_MAPPING_KEY)
    if mapping is None:
        raise ValueError(
            f"The cell table has no `uns['{_MAPPING_KEY}']`. Read the data with "
            "`read_visium_hd(..., load_bin_mapping=True)` (needs the segmentation)."
        )
    return mapping


def _resolve_agg(values: pd.Series, agg: str) -> str:
    is_numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if agg == "auto":
        return "mean" if is_numeric else "majority"
    if agg in ("mean", "sum") and not is_numeric:
        raise ValueError(f"agg='{agg}' needs a numeric column, but '{values.name}' is {values.dtype}.")
    return agg


def _reindex_aggregated(res: pd.Series, index, agg: str) -> pd.Series:
    """Align aggregated values to the target; for ``sum``, targets without values get 0."""
    out = res.reindex(index)
    return out.fillna(0) if agg == "sum" else out


def _aggregate(values: pd.Series, groups: np.ndarray, agg: str) -> pd.Series:
    """Aggregate `values` per group. Returns a Series indexed by the unique groups.

    Missing values are ignored. For ``majority`` ties go to the first category (category order
    for categoricals, sorted order otherwise).
    """
    values = values.reset_index(drop=True)
    valid = values.notna().to_numpy()
    values = values[valid].reset_index(drop=True)
    groups = np.asarray(groups)[valid]

    if agg in ("mean", "sum"):
        return getattr(values.groupby(groups), agg)()

    is_cat = isinstance(values.dtype, pd.CategoricalDtype)
    if is_cat:
        cats = values.cat.categories
        codes = values.cat.codes.to_numpy()
    else:
        codes, cats = pd.factorize(values, sort=True)
    counts = (
        pd.DataFrame({"g": groups, "c": codes})
        .groupby(["g", "c"]).size().rename("n").reset_index()
        .sort_values(["g", "n", "c"], ascending=[True, False, True])
        .drop_duplicates("g")
    )
    if is_cat:
        out_values = pd.Categorical.from_codes(counts["c"].to_numpy(), categories=cats)
    else:
        out_values = np.asarray(cats)[counts["c"].to_numpy()]
    return pd.Series(out_values, index=counts["g"].to_numpy())


def transfer_obs(
    data: InSituData,
    keys: str | list[str],
    source: _Layer,
    target: _Layer,
    agg: Literal["auto", "majority", "mean", "sum"] = "auto",
    key_added: str | list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Transfer obs columns between Visium HD bin layers and cells (experimental).

    Writes into ``.obs`` of the target table. Supported combinations:

    - **units -> units** (both ``visium_hd_bin``): the bin sizes must be nested (integer ratio).
      Coarse -> fine: each fine bin takes the value of its parent bin. Fine -> coarse: the fine
      bins of each parent are aggregated. Same size: values are matched by grid position.
    - **units -> cells**: each 2 um bin of a cell (``uns['visium_hd_bin_mapping']`` of the
      cell table, see ``read_visium_hd``) looks up its source bin; values are aggregated per cell.
    - **cells -> units**: each 2 um bin takes the value of its cell; values are aggregated per
      target bin. Bins without any cell get NaN (0 with ``agg="sum"``).

    The nesting of 2 -> 8 -> 16 um bins is derived from ``array_row``/``array_col`` by integer
    division, so the transfer also works after cropping. Bins that are missing in the source
    (filtered or cropped away) are ignored in the aggregation.

    Args:
        data: The sample to operate on.
        keys: Obs column(s) of the source table to transfer.
        source: Source layer as ``("cells" | "units", layer_name)``.
        target: Target layer as ``("cells" | "units", layer_name)``.
        agg: How values are combined where several source values map to one target element.
            ``"auto"`` uses ``"majority"`` for categorical/object/boolean and ``"mean"`` for
            numeric columns. ``"majority"`` takes the most frequent value (each 2 um bin or fine
            bin counts once); ties go to the first category (category order for categoricals,
            sorted order otherwise). Categorical results keep the source categories.
            Target elements without any source value get NaN, except for ``"sum"``, where they
            get 0. Ignored where each target element has a single source value.
        key_added: Name(s) of the new target column(s). Defaults to ``keys``.
        overwrite: Overwrite existing target columns. Defaults to False.

    Raises:
        ValueError: For unsupported layer combinations, non-nested bin sizes, a missing bin
            mapping, an incompatible ``agg`` or an existing target column without ``overwrite``.
        KeyError: If a key is not an obs column of the source table.
    """
    keys = convert_to_list(keys)
    key_added = keys if key_added is None else convert_to_list(key_added)
    if len(key_added) != len(keys):
        raise ValueError("`key_added` must have the same length as `keys`.")
    if agg not in ("auto", "majority", "mean", "sum"):
        raise ValueError(f"Unknown agg '{agg}'.")

    src_layer = _get_layer(data, source, "source")
    tgt_layer = _get_layer(data, target, "target")
    src_kind, tgt_kind = source[0], target[0]
    if src_kind == "cells" and tgt_kind == "cells":
        raise ValueError("Transfer between two cell layers is not supported.")

    src_obs, tgt_obs = src_layer.table.obs, tgt_layer.table.obs
    for k in keys:
        if k not in src_obs:
            raise KeyError(f"Key '{k}' not found in the obs of the source table.")
    if not overwrite:
        existing = [k for k in key_added if k in tgt_obs]
        if existing:
            raise ValueError(f"Target column(s) {existing} exist already. Use `overwrite=True`.")

    # per case: a function mapping a source column to a Series aligned with the target obs
    if src_kind == "units" and tgt_kind == "units":
        s_src, src_keys = _bin_info(src_layer, "source")
        s_tgt, tgt_keys = _bin_info(tgt_layer, "target")

        def transfer(col: pd.Series) -> pd.Series:
            by_pos = col.set_axis(src_keys)
            if s_src >= s_tgt:
                # coarse -> fine (or same size): look up the parent bin
                k = _ratio(s_src, s_tgt)
                return by_pos.reindex(_parent_key(tgt_keys, k))
            k = _ratio(s_tgt, s_src)
            how = _resolve_agg(col, agg)
            res = _aggregate(col, _parent_key(src_keys, k), how)
            return _reindex_aggregated(res, tgt_keys, how)

    elif src_kind == "units":
        s_src, src_keys = _bin_info(src_layer, "source")
        mapping = _get_mapping(tgt_layer)
        if s_src < _MAPPING_BIN_SIZE:
            raise ValueError(f"Source bin size {s_src} um is smaller than the 2 um mapping grid.")
        k = _ratio(s_src, _MAPPING_BIN_SIZE)
        map_src = _parent_key(_grid_key(mapping["array_row"], mapping["array_col"]), k)
        cell_ids = mapping["cell_id"].astype(object).to_numpy()

        def transfer(col: pd.Series) -> pd.Series:
            vals = col.set_axis(src_keys).reindex(map_src)
            how = _resolve_agg(col, agg)
            res = _aggregate(vals, cell_ids, how)
            return _reindex_aggregated(res, tgt_obs.index, how)

    else:  # cells -> units
        s_tgt, tgt_keys = _bin_info(tgt_layer, "target")
        mapping = _get_mapping(src_layer)
        if s_tgt < _MAPPING_BIN_SIZE:
            raise ValueError(f"Target bin size {s_tgt} um is smaller than the 2 um mapping grid.")
        k = _ratio(s_tgt, _MAPPING_BIN_SIZE)
        map_tgt = _parent_key(_grid_key(mapping["array_row"], mapping["array_col"]), k)
        cell_ids = mapping["cell_id"].astype(object).to_numpy()

        def transfer(col: pd.Series) -> pd.Series:
            vals = col.reindex(cell_ids)
            how = _resolve_agg(col, agg)
            res = _aggregate(vals, map_tgt, how)
            return _reindex_aggregated(res, tgt_keys, how)

    for k_src, k_dst in zip(keys, key_added, strict=True):
        result = transfer(src_obs[k_src])
        tgt_layer.table.obs[k_dst] = pd.Series(result.array, index=tgt_obs.index)
        logger.info(
            "Transferred '%s' from %s['%s'] to %s['%s'] as '%s'.",
            k_src, src_kind, source[1], tgt_kind, target[1], k_dst,
        )

from __future__ import annotations

import logging

from insitupy.containers.multi_cell_data import MultiCellData

logger = logging.getLogger(__name__)

_NO_CELLS_MESSAGE = (
    "This sample has no cell layers (cells not loaded or not part of this data, e.g. Visium). "
    "Call load_cells() if the sample has cells; for spatial units pass units_layer=... "
    "(pl.spatial) or use the units table directly (functions that accept an AnnData)."
)


def _get_cell_layer(
    cells: MultiCellData,
    cells_layer: str | None,
    verbose: bool = False,
    return_layer_name: bool = False,
):
    if cells.is_empty:
        raise ValueError(_NO_CELLS_MESSAGE)

    if cells_layer is None:
        cells_layer = cells.main_key
    else:
        all_keys = cells.keys()
        if cells_layer not in all_keys:
            raise ValueError(f"cells_layer {cells_layer} not in layers: {all_keys}")

    if verbose:
        logger.info(f"Using CellData from MultiCellData layer '{cells_layer}'.")
    layer = cells[cells_layer]

    if return_layer_name:
        return layer, cells_layer
    else:
        return layer


def _get_table_layer(
    data,
    cells_layer: str | None = None,
    units_layer: str | None = None,
    verbose: bool = False,
    return_layer_name: bool = False,
):
    """Resolve a cell layer or a units layer of an InSituData object.

    Returns a `CellData` or a `SpatialUnitsData`; both expose `.table`.
    `cells_layer` and `units_layer` are mutually exclusive. With neither given,
    the main cell layer is used.
    """
    if cells_layer is not None and units_layer is not None:
        raise ValueError("cells_layer and units_layer are mutually exclusive")

    if units_layer is None:
        if data.cells.is_empty:
            msg = _NO_CELLS_MESSAGE
            if data.units is not None and not data.units.is_empty:
                msg += f" Units layers: {list(data.units.keys())}."
            raise ValueError(msg)
        return _get_cell_layer(
            cells=data.cells,
            cells_layer=cells_layer,
            verbose=verbose,
            return_layer_name=return_layer_name,
        )

    units = data.units
    if units is None or units.is_empty:
        raise ValueError("No units loaded - call `load_units()` first.")
    keys = list(units.keys())
    if units_layer not in keys:
        raise ValueError(f"units_layer '{units_layer}' not in units layers: {keys}")
    layer = units[units_layer]
    if layer.table is None:
        raise ValueError(f"Units layer '{units_layer}' has no table.")

    if verbose:
        logger.info(f"Using SpatialUnitsData from units layer '{units_layer}'.")
    if return_layer_name:
        return layer, units_layer
    return layer


def _has_table_layer(xd, cells_layer: str | None = None, units_layer: str | None = None) -> bool:
    """Whether `_get_table_layer` would find the requested layer (never raises)."""
    if units_layer is not None:
        units = xd.units
        return (
            units is not None
            and not units.is_empty
            and units_layer in units.keys()
            and units[units_layer].table is not None
        )
    return not xd.cells.is_empty and (cells_layer is None or cells_layer in xd.cells.keys())

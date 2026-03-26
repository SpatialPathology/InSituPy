import logging
from typing import Optional

from insitupy.containers.multi_cell_data import MultiCellData

logger = logging.getLogger(__name__)


def _get_cell_layer(
    cells: MultiCellData,
    cells_layer: Optional[str],
    verbose: bool = False,
    return_layer_name: bool = False,
):
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
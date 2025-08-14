from insitupy.dataclasses.dataclasses import MultiCellData


def _get_cell_layer(
    cells: MultiCellData,
    cells_layer: str,
    verbose: bool = False,
    return_layer_name: bool = False,
):
    if cells_layer is None:
        cells_layer = cells.main_key
    else:
        all_keys = cells.get_all_keys()
        if cells_layer not in all_keys:
            raise ValueError(f"cells_layer {cells_layer} not in layers: {all_keys}")

    print(f"Using CellData from MultiCellData layer '{cells_layer}'.") if verbose else None
    layer = cells[cells_layer]

    if return_layer_name:
        return layer, cells_layer
    else:
        return layer

from insitupy.containers._utils import _get_cell_layer
from insitupy.utils._checks import _is_list_unique


def _all_obs_names_unique(
    exp,
    cells_layer: str | None,
    ):

    all_obs_names = []
    for meta, data in exp.iterdata():
        celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
        all_obs_names += celldata.table.obs_names.tolist()

    return _is_list_unique(all_obs_names)

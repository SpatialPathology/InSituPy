from typing import Literal

from insitupy.dataclasses._utils import _get_cell_layer


def _check_assignment(
    data,
    cells_layer: str,
    key: str,
    modality: Literal["annotations", "regions"],
    force_assignment: bool = False,
    verbose: bool = False
):
    celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
    try:
        column = celldata.table.obsm[modality].columns
    except KeyError:
        do_assignment = True
    else:
        if key in column:
            do_assignment = False
        else:
            do_assignment = True

    if do_assignment or force_assignment:
        if modality == "annotations":
            # assign annotations
            data.assign_annotations(keys=key, cells_layers=cells_layer)
        elif modality == "regions":
            # assign regions
            data.assign_regions(keys=key, cells_layers=cells_layer)
    else:
        if verbose:
            print(f"{modality.capitalize()} with key '{key}' have already been assigned to the dataset.")


def _is_experiment(obj):
    from insitupy._core.data import InSituData
    from insitupy.experiment.data import InSituExperiment

    # if isinstance(obj, InSituData):
    if obj.__class__ is InSituData:
        return False
    elif obj.__class__ is InSituExperiment:
    # elif isinstance(obj, InSituExperiment):
        return True
    else:
        raise ValueError(f"Object is neither InSituData or InSituExperiment. Instead: {type(obj)}")
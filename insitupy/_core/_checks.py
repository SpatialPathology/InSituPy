import logging
from typing import Literal

from insitupy.containers._utils import _get_cell_layer

logger = logging.getLogger(__name__)


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
            logger.info(f"{modality.capitalize()} with key '{key}' have already been assigned to the dataset.")


def _is_experiment(obj, *, allow_view: bool = True):
    """Return whether *obj* is an experiment-level object (True) or single-sample (False).

    Args:
        obj: The object to classify. Must be exactly `InSituData`, exactly
            `InSituExperiment`, or an `InSituExperimentView` (a subclass of
            `InSituExperiment` returned by subsetting-as-view).
        allow_view: If False, reject an `InSituExperimentView` with a clear
            NotImplementedError instead of returning True. Every current call
            site passes the default (True); this exists so a future call site
            that genuinely cannot support a linked view (e.g. it needs to
            mutate view-only, per-instance state that isn't shared with the
            parent) can opt out explicitly rather than the helper silently
            growing to allow everything.

    Raises:
        NotImplementedError: If *obj* is an `InSituExperimentView` and
            `allow_view` is False.
        ValueError: If *obj* is neither `InSituData`, `InSituExperiment`, nor
            `InSituExperimentView`.
    """
    from insitupy._core.data import InSituData
    from insitupy.experiment.data import InSituExperiment, InSituExperimentView

    if isinstance(obj, InSituExperimentView):
        if not allow_view:
            raise NotImplementedError(
                "This function does not support InSituExperimentView. Call "
                "`.copy()` on the view to get an independent InSituExperiment "
                "first, or pass the parent InSituExperiment directly."
            )
        return True
    elif obj.__class__ is InSituData:
        return False
    elif obj.__class__ is InSituExperiment:
        return True
    else:
        raise ValueError(f"Object is neither InSituData or InSituExperiment. Instead: {type(obj)}")

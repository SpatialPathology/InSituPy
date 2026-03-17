from warnings import warn

from insitupy._constants import WITH_NAPARI
from insitupy._constants import (ANNOTATIONS_SYMBOL, POINTS_SYMBOL,
                                 REGIONS_SYMBOL)

if WITH_NAPARI:
    from napari.layers import Layer, Points, Shapes


def _check_geometry_symbol_and_layer(layer, type_symbol):
    if type_symbol == ANNOTATIONS_SYMBOL:
        checks_passed = True
        object_type = "annotation"
    elif type_symbol == POINTS_SYMBOL:
        checks_passed = True
        object_type = "annotation"
    elif type_symbol == REGIONS_SYMBOL:
        #is_region_layer = True
        object_type = "region"
        if isinstance(layer, Points):
            warn(f'Layer "{layer.name}" is a point layer and at the same time classified as "Region". This is not allowed. Skipped this layer.')
            checks_passed = False
        else:
            checks_passed = True
    else:
        warn(f'Layer "{layer.name}" is not a valid geometry type. Skipped this layer.')
        checks_passed = False
        object_type = None

    return checks_passed, object_type
import contextlib
import logging
import os
import re
import sys
from datetime import datetime
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from shapely import MultiPolygon, Polygon

logger = logging.getLogger(__name__)


def _get_expression_values(adata, X, key_type, key):
    # get expression values
    if key_type == "genes":
        try:
            gene_loc = adata.var_names.get_loc(key)
            color_value = X[:, gene_loc]

            if issparse(color_value):
                color_value = color_value.toarray().flatten()
        except KeyError:
            if key in adata.obs.columns:
                color_value = adata.obs[key]
    elif key_type == "obs":
            color_value = adata.obs[key]
    elif key_type == "obsm":
        #TODO: Implement it for obsm
        obsm_key = key.split("#", maxsplit=1)[0]
        obsm_col = key.split("#", maxsplit=1)[1]
        data = adata.obsm[obsm_key]

        if isinstance(data, pd.DataFrame):
            color_value = data[obsm_col].values
        elif isinstance(data, np.ndarray):
            color_value = data[:, int(obsm_col)-1]
        else:
            warn("Data in `obsm` needs to be either pandas DataFrame or numpy array to be parsed.")
        pass
    else:
        logger.warning("Unknown key selected.")

    return color_value

def _fill_multipolygon(mp):
    filled_polygons = [Polygon(p.exterior) for p in mp.geoms]
    filled_multipolygon = MultiPolygon(filled_polygons)
    return filled_multipolygon

def _format_multipolygon_coords(coords):
    # Convert to proper format
    formatted_coords = []
    for poly in coords:
        exterior = poly[0]
        holes = poly[1:] if len(poly) > 1 else []
        formatted_coords.append((exterior, holes))

    return formatted_coords

def _convert_to_float_coords(coords, mode):
    # Convert Decimal to float

    # if len(coords) == 1:
    if mode == "Polygon":
        float_coords = [[(float(x), float(y)) for x, y in ring] for ring in coords]
        poly = Polygon(float_coords[0])
    elif mode == "MultiPolygon":
        float_coords = [[[(float(x), float(y)) for x, y in ring] for ring in poly] for poly in coords]
        formatted_coords = _format_multipolygon_coords(float_coords)

        # Create MultiPolygon
        mp = MultiPolygon(formatted_coords)

        # fill holes in the polygons
        poly = _fill_multipolygon(mp)
        #poly = MultiPolygon(float_coords)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")
    return poly

def _generate_mask(values, xmax, ymax, seg_mask_value):
    try:
        from rasterio.features import rasterize
    except ImportError:
        raise ImportError("This function requires the rasterio package, please install with `pip install rasterio`.")
    # rasterize polygons
    boundaries_mask = rasterize(
        list(zip(values, seg_mask_value)),
        out_shape=(ymax,xmax))
    boundaries_mask = da.from_array(boundaries_mask)

    return boundaries_mask


_SAVE_DIR_NAME = re.compile(r"(\d{6})-(\d{12})(?:-|$)")


def parse_save_dir_datetime(name: str) -> datetime | None:
    """Parse the datetime encoded in the name of an InSituPy save directory.

    Save directories are named ``YYMMDD-HHMMSSffffff-<hash>``
    (e.g. ``250805-115555000343-2c58ca86``).

    Args:
        name: Directory name (not a full path).

    Returns:
        The encoded datetime, or ``None`` if *name* does not follow the pattern
        (e.g. a folder the user created by hand).
    """
    match = _SAVE_DIR_NAME.match(name)
    if match is None:
        return None
    try:
        # YYMMDD + HHMMSSffffff
        return datetime.strptime(match.group(1) + match.group(2), "%y%m%d%H%M%S%f")
    except ValueError:
        return None


def sort_paths_by_datetime(paths):
    """Sort a list of paths by the datetime encoded in their names, newest first.

    Assumes directory names follow the pattern ``YYMMDD-HHMMSSffffff-<hash>``
    (e.g. ``250805-115555000343-2c58ca86``). Paths whose name does not follow
    it are not InSituPy save directories: they are left out of the result and
    reported in a single warning.

    Args:
        paths: Iterable of :class:`~pathlib.Path` objects to sort.

    Returns:
        A new list of the recognised paths sorted from most-recent to oldest.
    """
    parsed = []
    skipped = []
    for p in paths:
        dt = parse_save_dir_datetime(p.name)
        if dt is None:
            skipped.append(p.name)
        else:
            parsed.append((dt, p))

    if skipped:
        warn(
            f"Ignoring {len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'} that "
            f"{'is' if len(skipped) == 1 else 'are'} not an InSituPy save directory: "
            f"{sorted(skipped)}.",
            UserWarning,
            stacklevel=2,
        )

    return [p for _, p in sorted(parsed, key=lambda item: item[0], reverse=True)]




@contextlib.contextmanager
def suppress_output():
    """Context manager that silences all stdout and stderr output.

    Redirects both ``sys.stdout`` and ``sys.stderr`` to ``/dev/null`` for the
    duration of the ``with`` block, then restores them unconditionally.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stderr = old_stderr

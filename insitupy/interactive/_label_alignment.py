import warnings

import numpy as np

from insitupy.utils.utils import is_valid_boundary_index


def map_boundary_to_adata_positions(obs_names, cell_names_boundary):
    """Boundary-index -> adata-position, matched by cell name.

    Returns a list where entry b is the position of cell_names_boundary[b] in
    obs_names, or None if that name is absent from the table.
    """
    obs_names_arr = np.asarray(obs_names)
    name_to_pos = {str(n): i for i, n in enumerate(obs_names_arr)}
    if len(name_to_pos) < len(obs_names_arr):
        warnings.warn(
            "obs_names contains duplicate values; boundary cells matching a "
            "duplicated name will all be aligned to the last occurrence in the table.",
            stacklevel=2,
        )
    return [name_to_pos.get(str(cn), None) for cn in np.asarray(cell_names_boundary)]


def compute_label_cell_indices(label_ids, cell_names_boundary, obs_names,
                               nucleus_to_cell_map, mask_key):
    """Return (boundary_indices, adata_indices), one per label_id.

    boundary_indices index cell_names_boundary (boundary order); adata_indices
    index obs_names / color_values (table order). Callers use boundary_indices
    for boundary-derived fields (the tooltip cell name) and adata_indices for
    table-derived fields (colour values, obs columns).
    """
    boundary_to_adata = map_boundary_to_adata_positions(obs_names, cell_names_boundary)
    n_boundary = len(boundary_to_adata)
    if mask_key == "nuclei" and nucleus_to_cell_map is not None:
        name_to_boundary_pos = {str(n): i for i, n in enumerate(np.asarray(cell_names_boundary))}
        boundary_indices = [
            name_to_boundary_pos.get(nucleus_to_cell_map.get(int(lid) - 1), None)
            for lid in label_ids
        ]
    else:
        if len(label_ids) != n_boundary:
            raise ValueError(
                f"label_ids ({len(label_ids)}) and cell_names_boundary ({n_boundary}) "
                "must have matching length when boundary positions are assigned "
                "by index."
            )
        boundary_indices = list(range(n_boundary))

    def _to_adata(b):
        if not is_valid_boundary_index(b, n_boundary):
            return None
        return boundary_to_adata[b]

    adata_indices = [_to_adata(b) for b in boundary_indices]
    return boundary_indices, adata_indices

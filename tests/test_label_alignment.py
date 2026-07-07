"""Tests for the render-time name-based label/cell alignment helper.

Covers the correctness fix for coloring cells/nuclei labels layers when the
cell table was filtered without a following ``CellData.sync()`` call: labels
must be matched to table rows by cell name, not by positional index.
"""

import numpy as np
import pytest

from insitupy.interactive._label_alignment import (
    compute_label_cell_indices,
    map_boundary_to_adata_positions,
)


def test_synced_identity():
    """When boundary order matches table order, alignment is the identity map."""
    obs_names = ["A", "B", "C", "D"]
    cell_names_boundary = ["A", "B", "C", "D"]
    label_ids = np.array([1, 2, 3, 4])

    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=None,
        mask_key="cells",
    )

    assert boundary_indices == [0, 1, 2, 3]
    assert adata_indices == [0, 1, 2, 3]


def test_filtered_without_sync_reindexes_by_name():
    """The bug: table filtered+reordered relative to boundaries must still map by name."""
    obs_names = ["C", "A"]  # table has only cells C and A, reordered
    cell_names_boundary = ["A", "B", "C", "D"]
    label_ids = np.array([1, 2, 3, 4])

    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=None,
        mask_key="cells",
    )

    assert boundary_indices == [0, 1, 2, 3]
    # A -> position 1, B -> dropped (None), C -> position 0, D -> dropped (None)
    assert adata_indices == [1, None, 0, None]

    color_values = np.array([100.0, 200.0])  # indexed by obs_names order (C, A)
    picked = [
        color_values[i] if i is not None else None
        for i in adata_indices
    ]
    assert picked == [200.0, None, 100.0, None]


def test_nuclei_branch_maps_through_nucleus_to_cell_map():
    """Nuclei labels resolve boundary index via nucleus_to_cell_map, then by name."""
    obs_names = ["C", "A"]
    cell_names_boundary = ["A", "B", "C", "D"]
    label_ids = np.array([1, 2])  # nucleus label ids (1-based)
    # nucleus label_id - 1 -> boundary index
    nucleus_to_cell_map = {0: 2, 1: 0}  # nucleus 1 -> boundary idx 2 ("C"), nucleus 2 -> boundary idx 0 ("A")

    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=nucleus_to_cell_map,
        mask_key="nuclei",
    )

    assert boundary_indices == [2, 0]
    assert adata_indices == [0, 1]  # "C" -> position 0, "A" -> position 1


def test_missing_name_maps_to_none_without_error():
    """A boundary cell absent from obs_names resolves to None, no KeyError."""
    obs_names = ["A"]
    cell_names_boundary = ["A", "Z"]

    result = map_boundary_to_adata_positions(obs_names, cell_names_boundary)

    assert result == [0, None]


def test_nuclei_branch_without_nucleus_to_cell_map_falls_back_to_position():
    """With no nucleus_to_cell_map (e.g. v1.x data), nuclei behave like cells:
    boundary_indices is positional, adata_indices is still resolved by name."""
    obs_names = ["C", "A", "B", "D"]
    cell_names_boundary = ["A", "B", "C", "D"]
    label_ids = np.array([1, 2, 3, 4])

    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=None,
        mask_key="nuclei",
    )

    assert boundary_indices == [0, 1, 2, 3]
    assert adata_indices == [1, 2, 0, 3]


def test_stale_nucleus_to_cell_map_out_of_range_maps_to_none():
    """A stale nucleus_to_cell_map entry pointing past the current boundary
    array (e.g. after a sync() that dropped boundary rows without updating the
    map) resolves to None instead of raising an IndexError."""
    obs_names = ["A", "B"]
    cell_names_boundary = ["A", "B"]
    label_ids = np.array([1, 2])
    nucleus_to_cell_map = {0: 0, 1: 5}  # position 5 no longer exists

    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=nucleus_to_cell_map,
        mask_key="nuclei",
    )

    assert boundary_indices == [0, 5]
    assert adata_indices == [0, None]


def test_mismatched_lengths_raise_in_positional_branch():
    """label_ids and cell_names_boundary must match length when boundary
    positions are assigned positionally (cells branch / nuclei without a map)."""
    with pytest.raises(ValueError):
        compute_label_cell_indices(
            label_ids=np.array([1, 2, 3]),
            cell_names_boundary=["A", "B"],
            obs_names=["A", "B"],
            nucleus_to_cell_map=None,
            mask_key="cells",
        )


def test_duplicate_obs_names_warns_and_uses_last_occurrence():
    """Duplicate obs_names collapse to the last occurrence; this is surfaced
    with a warning rather than failing silently."""
    obs_names = ["A", "B", "A"]
    cell_names_boundary = ["A"]

    with pytest.warns(UserWarning, match="duplicate"):
        result = map_boundary_to_adata_positions(obs_names, cell_names_boundary)

    assert result == [2]

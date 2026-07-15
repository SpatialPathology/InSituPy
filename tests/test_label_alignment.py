"""Tests for the render-time name-based label/cell alignment helper.

Covers the correctness fix for coloring cells/nuclei labels layers when the
cell table was filtered without a following ``CellData.sync()`` call: labels
must be matched to table rows by cell name, not by positional index.
"""

import numpy as np
import pytest

from insitupy.containers.boundaries_data import BoundariesData
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
    # nucleus label_id - 1 -> parent cell name
    nucleus_to_cell_map = {0: "C", 1: "A"}  # nucleus 1 -> cell "C", nucleus 2 -> cell "A"

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
    """A stale nucleus_to_cell_map entry referencing a cell name no longer present
    in the current boundaries (e.g. after a sync() that dropped boundary rows
    without updating the map) resolves to None instead of raising an error."""
    obs_names = ["A", "B"]
    cell_names_boundary = ["A", "B"]
    label_ids = np.array([1, 2])
    nucleus_to_cell_map = {0: "A", 1: "Z"}  # "Z" no longer exists

    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=nucleus_to_cell_map,
        mask_key="nuclei",
    )

    assert boundary_indices == [0, None]
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


def _make_multinucleated_boundaries():
    """3 cells, 5 nuclei; nuclei 1&2 -> c1, nucleus 3 -> c2, nuclei 4&5 -> c3."""
    boundaries = BoundariesData(
        cell_names=["c1", "c2", "c3"],
        seg_mask_value=[10, 20, 30],
        nucleus_to_cell_map={0: "c1", 1: "c1", 2: "c2", 3: "c3", 4: "c3"},
    )
    cell_mask = np.array([[0, 10, 20], [30, 20, 10]], dtype=np.uint32)
    nuclei_mask = np.array([[0, 1, 3], [5, 4, 2]], dtype=np.uint32)
    boundaries.add_boundaries(cell_boundaries=cell_mask, nuclei_boundaries=nuclei_mask, pixel_size=1)
    return boundaries


def test_label_ids_for_nuclei_resolves_multinucleated_cells_through_the_real_caller():
    """Regression: the viewer used to pass seg_mask_value (cell labels) as label_ids
    for the nuclei layer, which only accidentally worked for a contiguous 1:1
    label space. label_ids_for("nuclei") must supply the actual nucleus raster
    labels so a real multinucleated dataset colors correctly."""
    boundaries = _make_multinucleated_boundaries()
    obs_names = ["c1", "c2", "c3"]

    assert list(boundaries.label_ids_for("cells")) == [10, 20, 30]

    nuclei_label_ids = boundaries.label_ids_for("nuclei")
    assert list(nuclei_label_ids) == [1, 2, 3, 4, 5]  # map keys (0-4) + 1

    cell_names_boundary = boundaries.cell_names.compute()
    boundary_indices, adata_indices = compute_label_cell_indices(
        label_ids=nuclei_label_ids,
        cell_names_boundary=cell_names_boundary,
        obs_names=obs_names,
        nucleus_to_cell_map=boundaries.nucleus_to_cell_map,
        mask_key="nuclei",
    )

    # nucleus labels 1,2 -> c1 (row 0); label 3 -> c2 (row 1); labels 4,5 -> c3 (row 2)
    assert boundary_indices == [0, 0, 1, 2, 2]
    assert adata_indices == [0, 0, 1, 2, 2]


def test_label_ids_for_falls_back_to_seg_mask_value_without_map():
    """1:1 fallback: with no nucleus_to_cell_map, nuclei labels are seg_mask_value,
    same as cells - no spurious None/unmapped entries."""
    boundaries = BoundariesData(cell_names=["c1", "c2"], seg_mask_value=[1, 2])
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint32)
    boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=mask, pixel_size=1)

    cells_ids = boundaries.label_ids_for("cells")
    nuclei_ids = boundaries.label_ids_for("nuclei")

    np.testing.assert_array_equal(cells_ids, np.array([1, 2], dtype=np.uint32))
    np.testing.assert_array_equal(nuclei_ids, cells_ids)


def test_duplicate_obs_names_warns_and_uses_last_occurrence():
    """Duplicate obs_names collapse to the last occurrence; this is surfaced
    with a warning rather than failing silently."""
    obs_names = ["A", "B", "A"]
    cell_names_boundary = ["A"]

    with pytest.warns(UserWarning, match="duplicate"):
        result = map_boundary_to_adata_positions(obs_names, cell_names_boundary)

    assert result == [2]

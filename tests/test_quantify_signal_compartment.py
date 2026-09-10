"""Tests for quantify_signal(cells_compartment="nuclei") label-space handling (C2).

The nucleus raster carries its own label space (nucleus polygon index + 1), which does
*not* coincide with the cell seg_mask_value on Xenium v2/v3 (multinucleated or independently
renumbered nuclei). quantify_signal must translate nucleus labels to cell labels before
regionprops, otherwise nucleus measurements are attributed to the wrong cell (or dropped).

See .log/reports/260910/boundary-group-a-fixes/report-boundary-group-a-fixes.md (finding C2).

Built inline (no tests/spatialdata_fixtures import, which would tie this to the optional
spatialdata dependency).
"""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData
from insitupy.containers.image_data import ImageData
from insitupy.containers.multi_cell_data import MultiCellData


def _build_insitudata(cell_names, seg_mask_value, cells_mask, nuclei_mask, image,
                      nucleus_to_cell_map=None, pixel_size=1.0):
    table = AnnData(
        X=np.ones((len(cell_names), 2)),
        obs=pd.DataFrame(index=pd.Index(cell_names, dtype=str)),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.zeros((len(cell_names), 2), dtype=float)

    boundaries = BoundariesData(
        cell_names=list(cell_names),
        seg_mask_value=list(seg_mask_value),
        nucleus_to_cell_map=nucleus_to_cell_map,
    )
    boundaries.add_boundaries(
        cell_boundaries=cells_mask.astype(np.uint32),
        nuclei_boundaries=nuclei_mask.astype(np.uint32),
        pixel_size=pixel_size,
    )

    cells = MultiCellData()
    cells.add_celldata(CellData(table=table, boundaries=boundaries), key="main", is_main=True)

    images = ImageData()
    images.add_image(image=image.astype(np.float32), channel_names="IF", axes="YX",
                     pixel_size=pixel_size, verbose=False)

    data = InSituData()
    data._cells = cells
    data._images = images
    return data


def _multinucleated_fixture(with_empty_third_cell=False):
    # c1 (seg 1) owns nuclei 1 and 2; c2 (seg 2) owns nucleus 3.
    nuclei = np.array([
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.uint32)
    cells = np.array([
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 0, 0],  # c1 extends past its nuclei into background (image 0)
    ], dtype=np.uint32)
    image = np.zeros_like(nuclei, dtype=np.float32)
    image[nuclei == 1] = 10.0
    image[nuclei == 2] = 20.0
    image[nuclei == 3] = 30.0

    cell_names = ["c1", "c2"]
    seg = [1, 2]
    nmap = {0: "c1", 1: "c1", 2: "c2"}  # nucleus idx -> parent cell name

    if with_empty_third_cell:
        # c3 (seg 3) exists in the table/cells raster but has no nucleus underneath
        cells = cells.copy()
        cells[2, 4:6] = 3
        cell_names = ["c1", "c2", "c3"]
        seg = [1, 2, 3]

    return _build_insitudata(cell_names, seg, cells, nuclei, image, nucleus_to_cell_map=nmap)


def test_multinucleated_nuclei_signal_assigned_to_parent_cell():
    data = _multinucleated_fixture()

    res = data.quantify_signal(image_name="IF", cells_compartment="nuclei",
                               method="mean", add_to_obs=False)

    # c1's two nuclei (10, 20 over equal areas) are one region -> mean 15; c2's nucleus -> 30.
    assert set(res.index) == {"c1", "c2"}
    assert res["c1"] == pytest.approx(15.0)
    assert res["c2"] == pytest.approx(30.0)
    # nothing dropped to a None index
    assert res.index.notna().all()


def test_cells_compartment_unchanged_by_c2_relabel():
    data = _multinucleated_fixture()

    res = data.quantify_signal(image_name="IF", cells_compartment="cells",
                               method="mean", add_to_obs=False)

    # "cells" labels already are seg_mask_value: relabel is a no-op. c1 covers nuclei 1+2
    # (four px at 10, four px at 20) plus four nucleus-free px at 0, 12 px total:
    # (4*10 + 4*20 + 4*0) / 12 = 10. c2 covers nucleus 3 (30) over its 4 px = 30.
    assert res["c1"] == pytest.approx(10.0)
    assert res["c2"] == pytest.approx(30.0)


def test_zero_nucleus_cell_gets_nan_not_a_neighbours_value():
    data = _multinucleated_fixture(with_empty_third_cell=True)

    data.quantify_signal(image_name="IF", cells_compartment="nuclei", method="mean")

    obs = data.cells["main"].table.obs
    col = "IF_signal_nuclei_mean"
    assert obs.loc["c1", col] == pytest.approx(15.0)
    assert obs.loc["c2", col] == pytest.approx(30.0)
    # c3 has no nucleus -> NaN, not a borrowed neighbour value
    assert np.isnan(obs.loc["c3", col])


def test_v1x_none_map_nuclei_results_match_direct_labels():
    # v1.x path: nuclei labels already equal seg_mask_value, nucleus_to_cell_map is None.
    # label_lut returns None so quantify does not relabel; the result is the pre-fix result,
    # which is already correct for 1:1 data.
    nuclei = np.array([
        [1, 1, 2, 2],
        [0, 0, 0, 0],
    ], dtype=np.uint32)
    cells = nuclei.copy()
    image = np.zeros_like(nuclei, dtype=np.float32)
    image[nuclei == 1] = 10.0
    image[nuclei == 2] = 20.0

    data = _build_insitudata(["c1", "c2"], [1, 2], cells, nuclei, image,
                             nucleus_to_cell_map=None)

    res = data.quantify_signal(image_name="IF", cells_compartment="nuclei",
                               method="mean", add_to_obs=False)
    assert res["c1"] == pytest.approx(10.0)
    assert res["c2"] == pytest.approx(20.0)


def test_v1x_one_to_one_dict_map_is_identity_relabel():
    # The shipped hbreastcancer shape: seg_mask_value == 1..N, a 1:1 nucleus map, and nuclei
    # rasters labelled 1..N. label_lut must be the identity and as_cell_labeled_mask a no-op,
    # so quantify_signal("nuclei") is unchanged. This is what guarantees tutorial 09 does not
    # regress.
    boundaries = BoundariesData(
        cell_names=["c1", "c2", "c3"],
        seg_mask_value=[1, 2, 3],
        nucleus_to_cell_map={0: "c1", 1: "c2", 2: "c3"},
    )
    mask = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32)
    boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=mask, pixel_size=1)

    lut = boundaries.label_lut("nuclei")
    np.testing.assert_array_equal(lut, np.array([0, 1, 2, 3], dtype=lut.dtype))

    relabeled = boundaries.as_cell_labeled_mask("nuclei", da.from_array(mask))
    np.testing.assert_array_equal(relabeled.compute(), mask)


def test_label_lut_none_for_cells_and_inconsistent_map():
    boundaries = BoundariesData(
        cell_names=["c1", "c2"],
        seg_mask_value=[10, 20],
        nucleus_to_cell_map={0: "c1", 1: "c2"},
    )
    # "cells" labels already are seg_mask_value -> no translation
    assert boundaries.label_lut("cells") is None

    inconsistent = BoundariesData(
        cell_names=["c1", "c2"],
        seg_mask_value=[10, 20],
        nucleus_to_cell_map={0: "ghost"},  # names a cell that does not exist
    )
    assert inconsistent.label_lut("nuclei") is None


def test_label_lut_table_maps_labels_to_seg_and_gaps_to_zero():
    # non-contiguous seg_mask_value, and a gap in the nucleus index space (idx 1 missing)
    boundaries = BoundariesData(
        cell_names=["c1", "c2"],
        seg_mask_value=[10, 20],
        nucleus_to_cell_map={0: "c1", 2: "c2"},
    )
    lut = boundaries.label_lut("nuclei")

    assert lut.dtype == np.uint32
    assert lut[0] == 0            # background
    assert lut[1] == 10           # nucleus 1 (idx 0) -> c1 seg 10
    assert lut[2] == 0            # gap: no nucleus idx 1 in the map
    assert lut[3] == 20           # nucleus 3 (idx 2) -> c2 seg 20

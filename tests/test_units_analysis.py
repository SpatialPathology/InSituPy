"""Tests for using units layers (Visium HD bins / spots) in pl/pp/tl and `tl.transfer_obs`.

Synthetic sample: a 2 um layer (16 x 16 bins) and an 8 um layer (4 x 4 bins) covering the same
32 x 32 um area, plus a few cells whose table holds a hand-built bin -> cell mapping.
"""

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.collections import PathCollection
from shapely.geometry import box

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers._utils import _get_table_layer
from insitupy.containers.cell_data import CellData
from insitupy.containers.spatial_units_data import SpatialUnitsData
from insitupy.experiment.data import InSituExperiment

N2 = 16  # 2 um bins per side
N8 = 4   # 8 um bins per side


def _bin_layer(size, side, seed=0, n_genes=20, **obs_columns):
    rng = np.random.default_rng(seed)
    rows, cols = np.divmod(np.arange(side * side), side)
    names = [f"s_{size:03d}um_{r:05d}_{c:05d}-1" for r, c in zip(rows, cols, strict=True)]
    obs = pd.DataFrame(
        {"array_row": rows.astype("int32"), "array_col": cols.astype("int32"), **obs_columns},
        index=names,
    )
    X = np.log1p(rng.poisson(2.0, size=(side * side, n_genes)).astype(float))
    table = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    centers = np.c_[(cols + 0.5) * size, (rows + 0.5) * size]
    table.obsm["spatial"] = centers
    table.uns["bin_size_um"] = size
    shapes = gpd.GeoDataFrame(
        {"name": names, "geometry": [box(c * size, r * size, (c + 1) * size, (r + 1) * size)
                                     for r, c in zip(rows, cols, strict=True)]},
        index=names,
    )
    return SpatialUnitsData(shapes=shapes, data=table, unit_type="visium_hd_bin")


# (cell, [(row, col) of 2 um bins]); parents in the 8 um grid are (r // 4, c // 4)
CELL_BINS = {
    "c0": [(0, 0), (0, 1), (1, 0), (1, 1)],   # 4 bins in 8 um (0, 0)
    "c1": [(0, 4), (0, 5), (1, 4)],           # 3 bins in 8 um (0, 1)
    "c2": [(0, 3), (1, 3), (2, 3), (3, 4)],   # 3 bins in 8 um (0, 0), 1 in (0, 1)
    "c3": [(8, 8)],                           # 1 bin in 8 um (2, 2)
    "c4": [],                                 # no bins
}


def _make_xd():
    xd = InSituData(path=None, metadata=None, slide_id="s", sample_id="x",
                    method_name="t", method_params={})

    # 2 um layer: tie in 8 um bin (0, 0) (8 x "y", 8 x "x"); categories ordered so "b" < "a"
    lab2 = np.full(N2 * N2, "a", dtype=object)
    rows2, cols2 = np.divmod(np.arange(N2 * N2), N2)
    in_00 = (rows2 < 4) & (cols2 < 4)
    lab2[in_00 & (rows2 < 2)] = "b"
    lab2 = pd.Categorical(lab2, categories=["b", "a"])
    u2 = _bin_layer(2, N2, seed=1, lab=lab2, n=cols2.astype(float))

    # 8 um layer
    rows8, cols8 = np.divmod(np.arange(N8 * N8), N8)
    grp = pd.Categorical(np.where((rows8 == 0) & (cols8 == 1), "B", "A"), categories=["A", "B"])
    u8 = _bin_layer(8, N8, seed=2, grp=grp, val=(rows8 * 10 + cols8).astype(float))

    xd.add_units(u2, key="square_002um")
    xd.add_units(u8, key="square_008um")

    # cells
    names = list(CELL_BINS)
    ctable = AnnData(X=np.ones((len(names), 2)), obs=pd.DataFrame(
        {"ctype": pd.Categorical(["T", "B", "T", "B", "T"], categories=["T", "B"])}, index=names),
        var=pd.DataFrame(index=["g0", "g1"]))
    ctable.obsm["spatial"] = np.array([[2.0, 2.0], [10.0, 2.0], [6.0, 4.0], [18.0, 18.0], [30.0, 30.0]])
    mapping = pd.DataFrame(
        [(cid, r, c) for cid, bins in CELL_BINS.items() for r, c in bins],
        columns=["cell_id", "array_row", "array_col"],
    )
    mapping["cell_id"] = pd.Categorical(mapping["cell_id"])
    mapping["in_nucleus"] = False
    ctable.uns["visium_hd_bin_mapping"] = mapping
    xd.cells.add_celldata(cd=CellData(table=ctable, boundaries=None), key="main", is_main=True)

    # annotations: two overlapping polygons A (x 0-16) and B (x 8-32)
    gdf = gpd.GeoDataFrame({
        "id": ["A_0", "B_0"], "name": ["A", "B"],
        "geometry": [box(0, 0, 16, 32), box(8, 0, 32, 32)],
        "color": ["#ff0000", "#0000ff"],
    })
    xd._annotations.add_data(data=gdf, key="zones", scale_factor=1.0)
    return xd


# 1 -------------------------------------------------------------------------------------------

def test_resolver_rejects_both_layers_and_unknown_units_key():
    xd = _make_xd()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _get_table_layer(xd, cells_layer="main", units_layer="square_008um")
    with pytest.raises(ValueError, match="square_002um"):
        _get_table_layer(xd, units_layer="nope")
    assert _get_table_layer(xd, units_layer="square_008um").table is xd.units["square_008um"].table
    # cells behaviour unchanged
    assert _get_table_layer(xd).table is xd.cells["main"].table
    # no units loaded
    empty = InSituData(path=None, metadata=None, slide_id="s", sample_id="x",
                       method_name="t", method_params={})
    with pytest.raises(ValueError, match="load_units"):
        _get_table_layer(empty, units_layer="square_008um")


# 2 -------------------------------------------------------------------------------------------

def test_filter_cells_on_units_keeps_shapes_in_sync_and_order():
    xd = _make_xd()
    layer = xd.units["square_008um"]
    mask = np.zeros(len(layer.table), dtype=bool)
    mask[[10, 3, 7, 1, 14]] = True
    expected = list(layer.table.obs_names[mask])
    isp.pp.filter_cells(xd, units_layer="square_008um", mask=mask)
    assert list(layer.table.obs_names) == expected
    assert list(layer.shapes.index) == expected

    # scanpy path: zero out two bins
    layer = xd.units["square_002um"]
    X = np.asarray(layer.table.X).copy()
    X[[5, 100, 200]] = 0
    layer.table.X = X
    isp.pp.filter_cells(xd, units_layer="square_002um", min_counts=1)
    assert layer.table.n_obs == N2 * N2 - 3
    assert list(layer.shapes.index) == list(layer.table.obs_names)
    # cells untouched
    assert xd.cells["main"].table.n_obs == len(CELL_BINS)


# 3 -------------------------------------------------------------------------------------------

def test_assign_annotations_units_layer_and_overlap_label():
    xd = _make_xd()
    # default call: cell layers only, units untouched
    xd.assign_annotations(keys="zones")
    assert "annotations" not in xd.units["square_008um"].table.obsm

    xd.assign_annotations(keys="zones", cells_layers=[], units_layers="square_008um")
    table = xd.units["square_008um"].table
    labels = table.obsm["annotations"]["zones"]
    cx = table.obsm["spatial"][:, 0]
    expected = np.where(cx < 8, "A", np.where(cx < 16, "A & B", "B"))
    assert list(labels) == list(expected)

    xd.assign_annotations(keys="zones", cells_layers=[], units_layers="square_002um")
    assert set(xd.units["square_002um"].table.obsm["annotations"]["zones"]) == {"A", "A & B", "B"}

    with pytest.raises(ValueError, match="mutually exclusive"):
        xd.assign_geometries("annotations", keys="zones", cells_layer="main", units_layer="square_008um")


def test_assign_annotations_units_layer_after_crop():
    """Cropping moves the units table coordinates together with the shapes and annotations,
    so assignment after a crop gives the same labels as before it."""
    xd = _make_xd()
    cropped = xd.crop(xlim=(9, 32), ylim=(0, 32))
    table = cropped.units["square_008um"].table
    shapes = cropped.units["square_008um"].shapes
    np.testing.assert_allclose(table.obsm["spatial"], np.c_[shapes.centroid.x, shapes.centroid.y])

    cropped.assign_annotations(keys="zones", cells_layers=[], units_layers="square_008um")
    labels = table.obsm["annotations"]["zones"]
    cx_orig = table.obsm["spatial"][:, 0] + 9
    expected = np.where(cx_orig < 16, "A & B", "B")
    assert list(labels) == list(expected)


# 4 -------------------------------------------------------------------------------------------

def test_dge_on_units_layer_records_layer():
    xd = _make_xd()
    res = isp.tl.dge(
        xd, target_annotation_tuple=("zones", "A"), ref_annotation_tuple="rest",
        units_layer="square_008um", exclude_ambiguous_assignments=True,
    )
    assert res.config.units_layer == "square_008um"
    assert res.config.cells_layer is None
    assert len(res.main) == 20


# 5 -------------------------------------------------------------------------------------------

def test_transfer_obs_coarse_to_fine_inherits():
    xd = _make_xd()
    isp.tl.transfer_obs(xd, ["grp", "val"], source=("units", "square_008um"),
                        target=("units", "square_002um"))
    obs = xd.units["square_002um"].table.obs
    r, c = obs["array_row"].to_numpy(), obs["array_col"].to_numpy()
    assert np.allclose(obs["val"], (r // 4) * 10 + (c // 4))
    assert list(obs["grp"].cat.categories) == ["A", "B"]
    assert (obs["grp"] == np.where(((r // 4) == 0) & ((c // 4) == 1), "B", "A")).all()


def test_transfer_obs_fine_to_coarse_majority_and_mean():
    xd = _make_xd()
    isp.tl.transfer_obs(xd, ["lab", "n"], source=("units", "square_002um"),
                        target=("units", "square_008um"))
    obs = xd.units["square_008um"].table.obs
    by_pos = obs.set_index(["array_row", "array_col"])
    # tie (8 x "b", 8 x "a" in bin (0, 0)) goes to the first category, "b" (not alphabetical)
    assert by_pos.loc[(0, 0), "lab"] == "b"
    assert by_pos.loc[(0, 1), "lab"] == "a"
    assert list(obs["lab"].cat.categories) == ["b", "a"]
    assert by_pos.loc[(0, 0), "n"] == pytest.approx(1.5)   # mean of columns 0..3
    assert by_pos.loc[(0, 1), "n"] == pytest.approx(5.5)   # mean of columns 4..7


def test_transfer_obs_units_to_cells_weighted_majority():
    xd = _make_xd()
    isp.tl.transfer_obs(xd, ["grp", "val"], source=("units", "square_008um"),
                        target=("cells", "main"))
    obs = xd.cells["main"].table.obs
    # c2 has 3 bins in 8 um (0, 0) -> "A" and 1 bin in (0, 1) -> "B"
    assert obs.loc["c0", "grp"] == "A"
    assert obs.loc["c1", "grp"] == "B"
    assert obs.loc["c2", "grp"] == "A"
    assert obs.loc["c3", "grp"] == "A"
    assert pd.isna(obs.loc["c4", "grp"])
    assert obs.loc["c2", "val"] == pytest.approx(0.25)     # (3 * 0 + 1 * 1) / 4
    assert obs.loc["c3", "val"] == pytest.approx(22.0)
    assert pd.isna(obs.loc["c4", "val"])


def test_transfer_obs_units_to_cells_ignores_bins_missing_in_source():
    xd = _make_xd()
    # drop 8 um bin (0, 1): c1 loses all bins, c2 keeps only its (0, 0) bins
    layer = xd.units["square_008um"]
    layer.table = layer.table[~((layer.table.obs["array_row"] == 0) & (layer.table.obs["array_col"] == 1))]
    isp.tl.transfer_obs(xd, "val", source=("units", "square_008um"), target=("cells", "main"))
    obs = xd.cells["main"].table.obs
    assert pd.isna(obs.loc["c1", "val"])
    assert obs.loc["c2", "val"] == pytest.approx(0.0)


def test_transfer_obs_cells_to_units_nan_for_empty_bins():
    xd = _make_xd()
    isp.tl.transfer_obs(xd, "ctype", source=("cells", "main"), target=("units", "square_008um"))
    obs = xd.units["square_008um"].table.obs.set_index(["array_row", "array_col"])
    assert obs.loc[(0, 0), "ctype"] == "T"   # c0 (4 x T) and c2 (3 x T)
    assert obs.loc[(0, 1), "ctype"] == "B"   # c1 (3 x B) beats c2 (1 x T)
    assert obs.loc[(2, 2), "ctype"] == "B"
    assert obs["ctype"].isna().sum() == N8 * N8 - 3


def test_transfer_obs_sum_gives_zero_for_empty_targets_mean_gives_nan():
    xd = _make_xd()
    xd.cells["main"].table.obs["one"] = 1.0
    src, tgt = ("cells", "main"), ("units", "square_008um")
    isp.tl.transfer_obs(xd, "one", source=src, target=tgt, agg="sum", key_added="n_sum")
    isp.tl.transfer_obs(xd, "one", source=src, target=tgt, agg="mean", key_added="n_mean")
    obs = xd.units["square_008um"].table.obs.set_index(["array_row", "array_col"])
    # sum = number of mapped 2 um bins per 8 um bin
    assert obs.loc[(0, 0), "n_sum"] == 7   # c0 (4) + c2 (3)
    assert obs.loc[(0, 1), "n_sum"] == 4   # c1 (3) + c2 (1)
    assert obs.loc[(2, 2), "n_sum"] == 1   # c3
    assert (obs["n_sum"] == 0).sum() == N8 * N8 - 3
    assert obs["n_mean"].isna().sum() == N8 * N8 - 3


def test_transfer_obs_rejects_bad_inputs():
    xd = _make_xd()
    u8, u2, cells = ("units", "square_008um"), ("units", "square_002um"), ("cells", "main")

    # non-integer ratio
    xd.units["square_008um"].table.uns["bin_size_um"] = 5
    with pytest.raises(ValueError, match="not nested"):
        isp.tl.transfer_obs(xd, "val", source=u8, target=u2)
    xd.units["square_008um"].table.uns["bin_size_um"] = 8

    # not a visium_hd_bin layer
    xd.add_units(SpatialUnitsData(shapes=xd.units["square_008um"].shapes,
                                  data=xd.units["square_008um"].table, unit_type="niche"), key="other")
    with pytest.raises(ValueError, match="visium_hd_bin"):
        isp.tl.transfer_obs(xd, "val", source=("units", "other"), target=u2)

    # missing mapping
    del xd.cells["main"].table.uns["visium_hd_bin_mapping"]
    with pytest.raises(ValueError, match="visium_hd_bin_mapping"):
        isp.tl.transfer_obs(xd, "val", source=u8, target=cells)

    # missing key, existing target column, cells -> cells
    with pytest.raises(KeyError):
        isp.tl.transfer_obs(xd, "nope", source=u8, target=u2)
    isp.tl.transfer_obs(xd, "val", source=u8, target=u2)
    with pytest.raises(ValueError, match="overwrite"):
        isp.tl.transfer_obs(xd, "val", source=u8, target=u2)
    isp.tl.transfer_obs(xd, "val", source=u8, target=u2, overwrite=True)
    with pytest.raises(ValueError, match="two cell layers"):
        isp.tl.transfer_obs(xd, "ctype", source=cells, target=cells)


# 7 -------------------------------------------------------------------------------------------

def _plotted_points(fig):
    """Number of scatter points (PathCollections only, which skips the colorbar)."""
    return sum(
        len(c.get_offsets())
        for a in fig.get_axes() for c in a.collections
        if isinstance(c, PathCollection)
    )


def _markers(fig):
    """(marker, size in um) of the data-unit marker collections drawn by pl.spatial."""
    from insitupy.plotting.spatial import _DataUnitMarkers

    return [
        (c.marker, c.data_size)
        for a in fig.get_axes() for c in a.collections
        if isinstance(c, _DataUnitMarkers)
    ]


def test_spatial_units_layer_smoke():
    xd = _make_xd()
    plt.close("all")
    isp.pl.spatial(xd, "val", units_layer="square_008um", show=False)
    fig = plt.gcf()
    n_points = _plotted_points(fig)
    assert n_points == N8 * N8
    assert _markers(fig) == [("s", 8.0)]   # square markers with the bin edge as size
    plt.close("all")

    # categorical key, explicit spot_size is respected
    isp.pl.spatial(xd, "lab", units_layer="square_002um", spot_size=3, show=False)
    assert _markers(plt.gcf())[-1][1] == 3
    plt.close("all")

    # experiment with two samples
    exp = InSituExperiment()
    exp._data.append(_make_xd())
    exp._data.append(_make_xd())
    exp._metadata = pd.DataFrame({"uid": ["s1", "s2"]})
    isp.pl.spatial(exp, "grp", units_layer="square_008um", show=False)
    fig = plt.gcf()
    n_points = _plotted_points(fig)
    assert n_points == 2 * N8 * N8
    plt.close("all")

    with pytest.raises(ValueError, match="mutually exclusive"):
        isp.pl.spatial(xd, "val", cells_layer="main", units_layer="square_008um", show=False)


def test_spatial_units_layer_without_coordinates_raises():
    xd = _make_xd()
    del xd.units["square_008um"].table.obsm["spatial"]
    with pytest.raises(ValueError, match="centroid"):
        isp.pl.spatial(xd, "val", units_layer="square_008um", show=False)
    plt.close("all")

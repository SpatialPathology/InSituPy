"""Samples without the requested cell/units layer.

Single samples raise a clear ValueError (used to be ``KeyError: None``); pl.spatial on an
experiment skips samples without the layer (one warning, empty titled panel) and only
raises if no sample has it.
"""
import warnings

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import box

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.spatial_units_data import SpatialUnitsData
from insitupy.experiment.data import InSituExperiment
from insitupy.plotting.spatial import _DataUnitMarkers

UNITS = "square_016um"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _table(n=12, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {"val": rng.random(n), "grp": pd.Categorical(rng.choice(["A", "B"], n))},
        index=[f"c{i}" for i in range(n)],
    )
    table = AnnData(X=rng.random((n, 2)), obs=obs, var=pd.DataFrame(index=["g0", "g1"]))
    table.obsm["spatial"] = rng.uniform(10, 90, size=(n, 2))
    return table


def _xd(sample_id, cells_layer=None, units=False, bin_size=None):
    xd = InSituData(path=None, metadata=None, slide_id=sample_id, sample_id=sample_id,
                    method_name="t", method_params={})
    if cells_layer is not None:
        xd.cells.add_celldata(cd=CellData(table=_table(), boundaries=None),
                              key=cells_layer, is_main=True)
    if units:
        table = _table(seed=1)
        n = len(table)
        cols, rows = np.arange(n) % 4, np.arange(n) // 4
        table.obsm["spatial"] = np.c_[(cols + 0.5) * 16, (rows + 0.5) * 16]
        if bin_size is not None:
            table.uns["bin_size_um"] = bin_size
        shapes = gpd.GeoDataFrame(
            {"name": table.obs_names,
             "geometry": [box(c * 16, r * 16, (c + 1) * 16, (r + 1) * 16)
                          for c, r in zip(cols, rows, strict=True)]},
            index=table.obs_names,
        )
        xd.add_units(SpatialUnitsData(shapes=shapes, data=table, unit_type="visium_hd_bin"),
                     key=UNITS)
    return xd


def _experiment(*xds):
    exp = InSituExperiment()
    for xd in xds:
        exp._data.append(xd)
    exp._metadata = pd.DataFrame({"uid": [xd.sample_id for xd in xds]})
    return exp


def _marker_axes(fig):
    return [a for a in fig.axes if any(isinstance(c, _DataUnitMarkers) for c in a.collections)]


def test_no_cell_layer_single_sample_raises_clear_error():
    xd = _xd("s0", units=True)
    with pytest.raises(ValueError, match="units_layer"):
        isp.pl.spatial(xd, keys="val", show=False)
    with pytest.raises(ValueError, match="load_cells"):
        isp.plotting.qc.plot_qc_metrics(xd)


def test_experiment_missing_units_layer_skips_sample():
    exp = _experiment(_xd("s0"), _xd("s1", units=True, bin_size=16))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        isp.pl.spatial(exp, keys=["val", "grp"], units_layer=UNITS, show=False)
    skipped = [w for w in rec if "not found in sample" in str(w.message)]
    assert len(skipped) == 1
    assert issubclass(skipped[0].category, UserWarning)
    assert "s0" in str(skipped[0].message)

    fig = plt.gcf()
    assert len(_marker_axes(fig)) == 2  # sample s1, one panel per key
    empty = [a for a in fig.axes if a.get_title() in ("val", "grp") and a not in _marker_axes(fig)]
    assert len(empty) == 2  # sample s0: titled, nothing drawn
    assert all(not a.collections for a in empty)


def test_experiment_missing_cells_layer_skips_sample():
    exp = _experiment(_xd("s0", cells_layer="main"), _xd("s1", cells_layer="other"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        isp.pl.spatial(exp, keys=["grp", "val"], cells_layer="other", show=False)
    assert len([w for w in rec if "not found in sample" in str(w.message)]) == 1

    fig = plt.gcf()
    drawn = [a for a in fig.axes if a.collections]
    assert len(drawn) >= 2  # s1 panels
    empty = [a for a in fig.axes if a.get_title() in ("grp", "val") and not a.collections]
    assert len(empty) == 2


def test_experiment_no_sample_has_layer_raises():
    exp = _experiment(_xd("s0"), _xd("s1"))
    with pytest.raises(ValueError, match="No sample has"):
        isp.pl.spatial(exp, keys="val", units_layer=UNITS, show=False)


def test_units_marker_defaults_from_first_sample_with_layer():
    exp = _experiment(_xd("s0"), _xd("s1", units=True, bin_size=16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        isp.pl.spatial(exp, keys="val", units_layer=UNITS, show=False)
    fig = plt.gcf()
    fig.canvas.draw()
    ax = _marker_axes(fig)[0]
    coll = next(c for c in ax.collections if isinstance(c, _DataUnitMarkers))
    t = ax.transData
    px_per_um = abs(t.transform((0, 1))[1] - t.transform((0, 0))[1])
    drawn_px = np.sqrt(coll.get_sizes()[0]) * fig.dpi / 72
    assert drawn_px == pytest.approx(16 * px_per_um, abs=0.5)

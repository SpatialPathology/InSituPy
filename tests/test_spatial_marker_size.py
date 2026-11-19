"""pl.spatial draws markers at their true size in um (square edge / circle diameter).

The marker size used to be converted from um to points before the final layout (the
equal aspect and the colorbar/legend axes shrink the axes at draw time), so markers came
out 1.2-4x too large. These tests render the figure and measure the drawn marker.
"""

import io

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import ListedColormap
from shapely.geometry import box

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers.spatial_units_data import SpatialUnitsData
from insitupy.plotting.spatial import PlotConfig, _DataUnitMarkers

BIN = 16.0
# elongated grid (aspect 5): the equal aspect shrinks the axes height a lot at draw time
NX, NY = 40, 8
BLUE, RED = "#0000ff", "#ff0000"


def _make_xd(bin_size=BIN, nx=NX, ny=NY):
    rows, cols = np.divmod(np.arange(nx * ny), nx)
    names = [f"b{i}" for i in range(nx * ny)]
    grp = np.full(nx * ny, "A", dtype=object)
    grp[-1] = "B"  # one marked bin, drawn last (on top), so its full extent is visible
    obs = pd.DataFrame(
        {"grp": pd.Categorical(grp, categories=["A", "B"]), "val": cols.astype(float),
         "mark": (grp == "B").astype(float)},
        index=names,
    )
    table = AnnData(X=np.ones((nx * ny, 2)), obs=obs, var=pd.DataFrame(index=["g0", "g1"]))
    table.obsm["spatial"] = np.c_[(cols + 0.5) * bin_size, (rows + 0.5) * bin_size]
    table.uns["bin_size_um"] = bin_size
    shapes = gpd.GeoDataFrame(
        {"name": names, "geometry": [
            box(c * bin_size, r * bin_size, (c + 1) * bin_size, (r + 1) * bin_size)
            for r, c in zip(rows, cols, strict=True)]},
        index=names,
    )
    xd = InSituData(path=None, metadata=None, slide_id="s", sample_id="x",
                    method_name="t", method_params={})
    xd.add_units(SpatialUnitsData(shapes=shapes, data=table, unit_type="visium_hd_bin"),
                 key="square_016um")
    return xd


def _px_per_um(ax):
    t = ax.transData
    return abs(t.transform((0, 1))[1] - t.transform((0, 0))[1])


def _marker_collections(fig):
    return [(a, c) for a in fig.get_axes() for c in a.collections if isinstance(c, _DataUnitMarkers)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("keys", ["val", ["val", "val"], "grp", ["grp", "grp"]])
def test_drawn_marker_edge_equals_bin_size(keys):
    """Continuous (colorbar) and categorical (legend axes), one and two panels."""
    isp.pl.spatial(_make_xd(), keys=keys, units_layer="square_016um", show=False)
    fig = plt.gcf()
    fig.canvas.draw()

    found = _marker_collections(fig)
    assert len(found) == (1 if isinstance(keys, str) else 2)
    for ax, coll in found:
        drawn_px = np.sqrt(coll.get_sizes()[0]) * fig.dpi / 72
        assert drawn_px == pytest.approx(BIN * _px_per_um(ax), abs=0.5)


@pytest.mark.parametrize("dpi", [80, 200])
@pytest.mark.parametrize("key", ["grp", "mark"])
def test_saved_square_has_bin_size_in_pixels(key, dpi):
    """Measure the marked (red) square in the rendered PNG, at screen and another save dpi.

    ``grp`` is categorical (legend axes below), ``mark`` continuous (colorbar right).
    """
    two_colors = ListedColormap([BLUE, RED])
    # a caller-given plot_config skips the units marker defaults, so set them explicitly
    pc = PlotConfig(palette=two_colors, cmap=two_colors, crange=[0, 1], show_legend=False,
                    spot_size=BIN, spot_type="s")
    isp.pl.spatial(_make_xd(), keys=key, units_layer="square_016um", plot_config=pc, show=False)
    fig = plt.gcf()
    ax = _marker_collections(fig)[0][0]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = plt.imread(buf)[..., :3]
    # only look inside the main axes (the colorbar is red, too)
    scale = dpi / fig.dpi
    bb = ax.get_window_extent()
    height = img.shape[0]
    r0, r1 = int(height - bb.y1 * scale), int(np.ceil(height - bb.y0 * scale))
    c0, c1 = int(bb.x0 * scale), int(np.ceil(bb.x1 * scale))
    crop = img[r0:r1, c0:c1]
    red = (crop[..., 0] > 0.6) & (crop[..., 1] < 0.4) & (crop[..., 2] < 0.4)
    ys, xs = np.nonzero(red)

    expected_px = BIN * _px_per_um(ax) * dpi / fig.dpi  # transData is at screen dpi again
    assert expected_px > 8  # the measurement is meaningful
    assert xs.max() - xs.min() + 1 == pytest.approx(expected_px, abs=1.5)
    assert ys.max() - ys.min() + 1 == pytest.approx(expected_px, abs=1.5)


def test_markers_follow_zoom():
    isp.pl.spatial(_make_xd(), keys="val", units_layer="square_016um", show=False)
    fig = plt.gcf()
    ax, coll = _marker_collections(fig)[0]
    ax.set_xlim(0, 4 * BIN)
    ax.set_ylim(4 * BIN, 0)
    fig.canvas.draw()
    drawn_px = np.sqrt(coll.get_sizes()[0]) * fig.dpi / 72
    assert drawn_px == pytest.approx(BIN * _px_per_um(ax), abs=0.5)


def test_subpixel_markers_are_drawn_at_one_pixel(capsys):
    # 2 um bins over 8 mm: far below one pixel at the default figure size
    xd = _make_xd(bin_size=2.0, nx=4000, ny=4)
    isp.pl.spatial(xd, keys="val", units_layer="square_016um", show=False)
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.draw()  # the notice is logged once per plot, not per draw
    ax, coll = _marker_collections(fig)[0]
    assert 2.0 * _px_per_um(ax) < 1
    assert np.sqrt(coll.get_sizes()[0]) * fig.dpi / 72 == pytest.approx(1.0)
    # the insitupy logger prints to stdout (propagate=False)
    assert capsys.readouterr().out.count("drawn at 1 px") == 1


def test_default_limits_include_whole_markers():
    isp.pl.spatial(_make_xd(), keys="val", units_layer="square_016um", show=False)
    ax = _marker_collections(plt.gcf())[0][0]
    assert ax.get_xlim() == pytest.approx((0, NX * BIN))
    assert sorted(ax.get_ylim()) == pytest.approx([0, NY * BIN])

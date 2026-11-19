"""pl.spatial with images stored as a single array instead of a pyramid (list of levels).

Readers that go through spatialdata (e.g. read_visium_hd) store a single dask array. The
level selection used to call len() on it, which counts rows: > 1024 rows overflowed
(2**1024), fewer rows silently picked a single pixel row as the background image.
"""
import matplotlib

matplotlib.use("Agg")

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.plotting.spatial import _extract_image_information


def _make_xd(img_shape):
    rng = np.random.default_rng(0)
    n = 20
    table = AnnData(
        X=rng.random((n, 3)),
        obs=pd.DataFrame({"score": rng.random(n)}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    table.obsm["spatial"] = rng.uniform(100, min(img_shape) - 100, size=(n, 2))
    xd = InSituData(
        path=None, metadata=None, slide_id="s1", sample_id="s1",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    img = da.from_array(np.arange(np.prod(img_shape), dtype=np.uint16).reshape(img_shape) % 1000)
    xd.images.add_image(img, channel_names="he", axes="YX", pixel_size=1.0, verbose=False)
    return xd


@pytest.mark.parametrize("n_rows", [1500, 600])
def test_single_array_image_is_one_level(n_rows):
    xd = _make_xd((n_rows, 800))
    assert not isinstance(xd.images["he"], list)  # precondition: stored without pyramid

    image, pixel_size, _, _, pixel_xlim, pixel_ylim = _extract_image_information(
        xd.images, image_key="he", xlim=(0, 800), ylim=(0, n_rows),
        pixelwidth_per_subplot=200, histogram_setting=None,
    )
    assert pixel_size == 1.0
    # covers the whole image (not one pixel row); the array itself is strided down
    assert pixel_xlim == [0, 800]
    assert pixel_ylim == [0, n_rows]
    assert image.shape[0] > 1


def test_spatial_with_large_single_array_image_smoke():
    xd = _make_xd((1500, 1500))
    isp.pl.spatial(xd, keys="score", image_key="he", show=False)
    plt.close("all")


# ---------------------------------------------------------------------------
# image placement
# ---------------------------------------------------------------------------

def _make_placed_xd(sample_id="s1", coord_range=(300, 600), pixel_size=1.0,
                    image_shape=(1000, 1000), with_image=True, rgb=False):
    """Sample whose cells sit at ``coord_range`` um; the image value equals the column index."""
    rng = np.random.default_rng(1)
    n = 20
    table = AnnData(
        X=rng.random((n, 3)),
        obs=pd.DataFrame({"score": rng.random(n), "score2": rng.random(n)},
                         index=[f"{sample_id}c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    coords = rng.uniform(*coord_range, size=(n, 2))
    coords[0] = coord_range  # pin the minimum
    table.obsm["spatial"] = coords
    xd = InSituData(
        path=None, metadata=None, slide_id=sample_id, sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    if with_image:
        cols = np.tile(np.arange(image_shape[1], dtype=np.uint16), (image_shape[0], 1))
        if rgb:
            cols = np.stack([cols] * 3, axis=-1).astype(np.uint8)
        xd.images.add_image(
            da.from_array(cols), channel_names="he", axes="YXS" if rgb else "YX",
            pixel_size=pixel_size, verbose=False,
        )
    return xd, coords


def _image_axes():
    return [a for a in plt.gcf().axes if a.images]


def test_origin_zero_shifts_image_with_points():
    xd, coords = _make_placed_xd()
    xmin = coords[:, 0].min()
    isp.pl.spatial(
        xd, keys="score", image_key="he", xlim=(350, 450), ylim=(350, 450),
        plot_config=isp.pl.PlotConfig(origin_zero=True), show=False,
    )
    (ax,) = _image_axes()
    im = ax.images[0]
    # the crop starts at the absolute limit (column index == um at pixel size 1) ...
    assert im.get_array()[0, 0] == pytest.approx(350, abs=2)
    # ... and is drawn at the shifted position, where the shifted cells are
    assert im.get_extent()[0] == pytest.approx(350 - xmin, abs=2)
    assert ax.get_xlim()[0] == pytest.approx(350 - xmin, abs=1e-6)
    plt.close("all")


def test_origin_zero_shifts_annotations():
    import geopandas as gpd
    from matplotlib.collections import PatchCollection
    from shapely.geometry import Polygon

    xd, coords = _make_placed_xd()
    xmin, ymin = coords.min(axis=0)
    gdf = gpd.GeoDataFrame({
        "id": ["A_0"], "name": ["A"], "color": [[255, 0, 0]],
        "geometry": [Polygon([(400, 410), (450, 410), (450, 470), (400, 470)])],
    })
    xd.annotations.add_data(data=gdf, key="zones", scale_factor=1.0)

    def drawn_bounds(origin_zero):
        isp.pl.spatial(
            xd, keys="score", annotation_tuple="zones",
            plot_config=isp.pl.PlotConfig(origin_zero=origin_zero), show=False,
        )
        verts = np.concatenate([
            p.vertices for c in plt.gcf().axes[0].collections
            if isinstance(c, PatchCollection) for p in c.get_paths()
        ])
        plt.close("all")
        return verts.min(axis=0), verts.max(axis=0)

    lo0, hi0 = drawn_bounds(False)
    lo1, hi1 = drawn_bounds(True)
    np.testing.assert_allclose(lo0, [400, 410], atol=1e-6)
    np.testing.assert_allclose(lo1, [400 - xmin, 410 - ymin], atol=1e-6)
    np.testing.assert_allclose(hi1, [450 - xmin, 470 - ymin], atol=1e-6)
    # the stored annotations must not be mutated by plotting
    assert xd.annotations["zones"].geometry.iloc[0].bounds == (400, 410, 450, 470)


def test_extent_has_no_half_micron_offset():
    xd, _ = _make_placed_xd(pixel_size=0.5, image_shape=(1000, 1000))
    isp.pl.spatial(xd, keys="score", image_key="he", xlim=(100, 200), ylim=(100, 200), show=False)
    (ax,) = _image_axes()
    extent = ax.images[0].get_extent()
    assert extent[0] == pytest.approx(100.0)
    assert extent[1] == pytest.approx(200.0)
    assert extent[3] == pytest.approx(100.0)  # top edge (origin upper)
    assert extent[2] == pytest.approx(200.0)
    plt.close("all")


def test_missing_image_key_single_sample_raises():
    xd, _ = _make_placed_xd()
    with pytest.raises(ValueError, match="Available images"):
        isp.pl.spatial(xd, keys="score", image_key="nuclei", show=False)
    plt.close("all")


def _make_image_experiment(with_image=(True, False)):
    exp = isp.InSituExperiment()
    for i, has in enumerate(with_image):
        exp._data.append(_make_placed_xd(sample_id=f"s{i}", with_image=has)[0])
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(len(with_image))]})
    return exp


def test_missing_image_key_in_one_experiment_sample_warns():
    exp = _make_image_experiment((True, False))
    with pytest.warns(UserWarning, match="not found in sample") as rec:
        isp.pl.spatial(exp, keys=["score", "score2"], image_key="he", show=False)
    assert len([w for w in rec if "not found in sample" in str(w.message)]) == 1
    assert "s1" in str([w.message for w in rec if "not found in sample" in str(w.message)][0])
    # one panel per (key, sample) combination; only the sample with the image has an image
    n_with_image = len(_image_axes())
    assert n_with_image == 2  # two keys x one sample
    plt.close("all")


def test_missing_image_key_in_all_experiment_samples_raises():
    exp = _make_image_experiment((True, True))
    with pytest.raises(ValueError, match="not found"):
        isp.pl.spatial(exp, keys="score", image_key="nuclei", show=False)
    plt.close("all")


def test_single_array_is_downsampled():
    xd = _make_xd((1500, 1500))
    image, pixel_size, _, _, pxl, pyl = _extract_image_information(
        xd.images, image_key="he", xlim=(0, 1500), ylim=(0, 1500),
        pixelwidth_per_subplot=200, histogram_setting=None,
    )
    assert max(image.shape[:2]) <= 2 * 200
    assert pixel_size == 1.0  # the extent is still computed from the full pixel limits
    assert (pxl[1] - pxl[0]) * pixel_size == 1500
    assert (pyl[1] - pyl[0]) * pixel_size == 1500


def test_non_factor_two_pyramid_extent():
    from insitupy.containers import ImageData

    base = da.zeros((4096, 4096), dtype=np.uint8)
    imd = ImageData()
    imd._data["img"] = [base, base[::4, ::4], base[::16, ::16]]
    imd._names.append("img")
    imd._metadata["img"] = {"pixel_size": 1.0, "shape": base.shape, "axes": "YX", "rgb": False}
    _, ps, _, _, pxl, _ = _extract_image_information(
        imd, "img", xlim=(0, 4096), ylim=(0, 4096),
        pixelwidth_per_subplot=200, histogram_setting=None,
    )
    assert (pxl[1] - pxl[0]) * ps == 4096


def test_auto_histogram_skipped_for_rgb():
    xd, _ = _make_placed_xd(rgb=True, image_shape=(300, 300))
    assert xd.images.metadata["he"]["rgb"]
    _, _, vmin, vmax, *_ = _extract_image_information(
        xd.images, image_key="he", xlim=(0, 300), ylim=(0, 300),
        pixelwidth_per_subplot=200, histogram_setting="auto",
    )
    assert vmin is None and vmax is None


def test_limits_outside_image_give_no_image():
    xd = _make_xd((500, 500))
    image, *_ = _extract_image_information(
        xd.images, image_key="he", xlim=(900, 1000), ylim=(900, 1000),
        pixelwidth_per_subplot=200, histogram_setting="auto",
    )
    assert image is None

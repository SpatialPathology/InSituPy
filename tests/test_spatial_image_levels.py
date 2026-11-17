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

    image, pixel_size, *_ = _extract_image_information(
        xd.images, image_key="he", xlim=(0, 800), ylim=(0, n_rows),
        pixelwidth_per_subplot=200, histogram_setting=None,
    )
    assert pixel_size == 1.0
    assert image.shape == (n_rows, 800)  # the whole image, not one pixel row


def test_spatial_with_large_single_array_image_smoke():
    xd = _make_xd((1500, 1500))
    isp.pl.spatial(xd, keys="score", image_key="he", show=False)
    plt.close("all")

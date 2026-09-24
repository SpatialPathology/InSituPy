"""InSituData.crop: per-image overlap, empty-region detection, and atomic in-place crops.

Layout used by all tests (physical units = pixels, pixel_size=1):
- image "big" covers 0-200 x 0-200
- image "small" covers 0-50 x 0-50 (e.g. a registered IF image over part of the tissue)
- cells and transcripts lie in 0-50 x 0-50 only
"""

import warnings

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData


def _make_insitudata(with_cells=True, with_transcripts=True, with_images=True):
    rng = np.random.default_rng(0)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="slide1", sample_id="s1",
        method_name="test", method_params={},
    )
    if with_cells:
        n = 10
        table = AnnData(
            X=rng.integers(0, 20, size=(n, 3)).astype(float),
            obs=pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n)])),
            var=pd.DataFrame(index=pd.Index([f"g{j}" for j in range(3)])),
        )
        table.obsm["spatial"] = rng.random((n, 2)) * 50
        xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    if with_transcripts:
        n = 100
        xd.transcripts = pd.DataFrame({
            "x_location": rng.random(n) * 50,
            "y_location": rng.random(n) * 50,
            "feature_name": rng.choice(["g0", "g1", "g2"], n),
        })
    if with_images:
        xd.images.add_image(image=da.zeros((200, 200), dtype=np.uint8), channel_names="big",
                            axes="YX", pixel_size=1.0, verbose=False)
        xd.images.add_image(image=da.zeros((50, 50), dtype=np.uint8), channel_names="small",
                            axes="YX", pixel_size=1.0, verbose=False)
    return xd


def test_region_missing_one_image_drops_it_with_warning():
    xd = _make_insitudata()

    with pytest.warns(UserWarning, match=r"\['small'\]"):
        cropped = xd.crop(xlim=(100, 150), ylim=(100, 150))

    assert list(cropped.images.keys()) == ["big"]
    assert cropped.images["big"].shape == (50, 50)
    assert "small" not in cropped.images.metadata
    # the original keeps both images
    assert sorted(xd.images.keys()) == ["big", "small"]


def test_region_outside_all_images_drops_all_images():
    xd = _make_insitudata(with_images=False)
    xd.images.add_image(image=da.zeros((20, 20), dtype=np.uint8), channel_names="tiny",
                        axes="YX", pixel_size=1.0, verbose=False)

    with pytest.warns(UserWarning, match=r"\['tiny'\]"):
        cropped = xd.crop(xlim=(25, 50), ylim=(25, 50))

    assert cropped.images.is_empty


def test_region_with_zero_transcripts_and_no_image_raises():
    # transcripts are loaded but none fall inside the region, and no image overlaps
    xd = _make_insitudata(with_cells=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="does not contain any image data or omic data"):
            xd.crop(xlim=(300, 350), ylim=(300, 350))


def test_region_with_zero_cells_and_no_image_raises():
    xd = _make_insitudata(with_transcripts=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="does not contain any image data or omic data"):
            xd.crop(xlim=(300, 350), ylim=(300, 350))


def test_failed_inplace_crop_leaves_object_unchanged():
    xd = _make_insitudata()
    cells_before = xd.cells["main"].table.n_obs
    transcripts_before = len(xd.transcripts)
    images_before = sorted(xd.images.keys())
    uids_before = list(xd.metadata["uids"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError):
            xd.crop(xlim=(300, 350), ylim=(300, 350), inplace=True)

    assert xd.cells["main"].table.n_obs == cells_before
    assert len(xd.transcripts) == transcripts_before
    assert sorted(xd.images.keys()) == images_before
    assert xd.images["big"].shape == (200, 200)
    assert xd.metadata["uids"] == uids_before
    assert "cropping_history" not in xd.metadata


def test_successful_inplace_crop_is_committed():
    xd = _make_insitudata()

    xd.crop(xlim=(0, 25), ylim=(0, 25), inplace=True)

    spatial = xd.cells["main"].table.obsm["spatial"]
    assert 0 < len(spatial) < 10
    assert xd.images["big"].shape == (25, 25)
    assert xd.images["small"].shape == (25, 25)
    assert len(xd.transcripts) < 100

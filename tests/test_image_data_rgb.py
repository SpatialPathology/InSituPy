"""RGB handling in ImageData.add_image: the canonical axes are the only source of truth."""

import dask.array as da
import numpy as np
import pytest

from insitupy.containers.image_data import ImageData


def _arr(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _assert_rgb_matches_axes(data: ImageData):
    for name in data.names:
        meta = data.metadata[name]
        assert meta["axes"] in {"YX", "YXS"}
        assert meta["rgb"] == (meta["axes"] == "YXS"), name


def test_rgba_yxs_is_rgb():
    data = ImageData()
    data.add_image(da.from_array(_arr((20, 16, 4))), channel_names="he", axes="YXS",
                   pixel_size=1.0, verbose=False)

    assert data.metadata["he"]["rgb"] is True
    _assert_rgb_matches_axes(data)


def test_yxs_with_two_samples_raises():
    data = ImageData()
    with pytest.raises(ValueError, match="3 \\(RGB\\) or 4 \\(RGBA\\)"):
        data.add_image(_arr((20, 16, 2)), channel_names="x", axes="YXS", pixel_size=1.0, verbose=False)


def test_is_rgb_true_on_grayscale_raises():
    data = ImageData()
    with pytest.raises(ValueError, match="no channel axis"):
        data.add_image(_arr((20, 16)), channel_names="g", axes="YX", pixel_size=1.0,
                       is_rgb=True, verbose=False)


def test_is_rgb_true_keeps_three_channel_cyx_as_one_rgb_image():
    arr = _arr((3, 20, 16))
    data = ImageData()
    data.add_image(arr, channel_names="he", axes="CYX", pixel_size=1.0, is_rgb=True, verbose=False)

    assert data.names == ["he"]
    assert data.metadata["he"]["axes"] == "YXS"
    assert data.metadata["he"]["rgb"] is True
    np.testing.assert_array_equal(np.asarray(data["he"]), np.moveaxis(arr, 0, -1))


def test_rgb_flag_matches_axes_for_mixed_images():
    data = ImageData()
    data.add_image(_arr((20, 16, 3)), channel_names="he", axes="YXS", pixel_size=1.0, verbose=False)
    data.add_image(_arr((20, 16)), channel_names="dapi", axes="YX", pixel_size=1.0, verbose=False)
    data.add_image(_arr((2, 20, 16)), channel_names=["a", "b"], axes="CYX", pixel_size=1.0, verbose=False)
    data.add_image(_arr((3, 20, 16)), channel_names="planar", axes="SYX", pixel_size=1.0, verbose=False)

    assert data.names == ["he", "dapi", "a", "b", "planar"]
    _assert_rgb_matches_axes(data)


def test_rgba_saved_as_ome_tiff_reads_back_as_yxs(tmp_path):
    # before WP-12 RGBA got rgb=False, was written as minisblack and read back as CYX
    arr = _arr((40, 30, 4), seed=1)
    src = ImageData()
    src.add_image(da.from_array(arr), channel_names="rgba", axes="YXS", pixel_size=0.5,
                  ome_meta={"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5}, verbose=False)
    paths = src.save(tmp_path / "imgs", as_zarr=False, return_savepaths=True)

    data = ImageData()
    data.add_image(paths["rgba"], channel_names="rgba", verbose=False)

    assert data.metadata["rgba"]["axes"] == "YXS"
    assert data.metadata["rgba"]["rgb"] is True
    img = data["rgba"]
    np.testing.assert_array_equal(np.asarray(img[0] if isinstance(img, list) else img), arr)

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy.tools import registration


class _DummyImages:
    def __init__(self):
        self.is_empty = False
        self.names = ["nuclei"]
        self.metadata = {"nuclei": {"pixel_size": 1.0, "axes": "YX"}}
        self.added_images = []

    def __contains__(self, key):
        return key == "nuclei"

    def __getitem__(self, key):
        if key != "nuclei":
            raise KeyError(key)
        return [np.zeros((8, 8), dtype=np.uint8)]

    def add_image(self, *args, **kwargs):
        self.added_images.append({"args": args, "kwargs": kwargs})
        return None


class _DummyData:
    def __init__(self, root: Path):
        self.path = root / "experiment.xenium"
        self.slide_id = "slide"
        self.sample_id = "sample"
        self.images = _DummyImages()


@pytest.fixture()
def dummy_data(tmp_path):
    return _DummyData(tmp_path)


@pytest.fixture()
def image_file(tmp_path):
    path = tmp_path / "image.tif"
    path.write_bytes(b"dummy")
    return path


def test_if_channel_count_mismatch_raises(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (np.zeros((2, 8, 8), dtype=np.uint8), {}, "CYX", 1.0),
    )

    with pytest.raises(ValueError, match="Mismatch between `channel_names` and image channels"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["DAPI", "FITC", "TRITC"],
            channel_name_for_registration="DAPI",
        )


def test_if_registration_channel_not_in_names_raises(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (np.zeros((2, 8, 8), dtype=np.uint8), {}, "CYX", 1.0),
    )

    with pytest.raises(ValueError, match="was not found in `channel_names`"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["FITC", "TRITC"],
            channel_name_for_registration="DAPI",
        )


def test_if_duplicate_channel_names_raises(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (np.zeros((2, 8, 8), dtype=np.uint8), {}, "CYX", 1.0),
    )

    with pytest.raises(ValueError, match="`channel_names` must be unique"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["DAPI", "DAPI"],
            channel_name_for_registration="DAPI",
        )


def test_if_no_channels_left_to_register_raises(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (np.zeros((1, 8, 8), dtype=np.uint8), {}, "CYX", 1.0),
    )

    with pytest.raises(ValueError, match="No channels remain to register"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["DAPI"],
            channel_name_for_registration="DAPI",
        )


def test_decon_scale_factor_non_positive_raises(dummy_data, image_file):
    with pytest.raises(ValueError, match="`decon_scale_factor` must be > 0"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["HE"],
            decon_scale_factor=0,
        )


def _dummy_register_images_standalone(moving, fixed, **kwargs):
    """Stub for register_images_standalone: return zeros + identity matrix."""
    h, w = fixed.shape[:2] if hasattr(fixed, "shape") else (8, 8)
    registered = np.zeros((h, w), dtype=np.uint8)
    T = np.eye(2, 3, dtype=np.float64)
    return registered, T


def _dummy_apply_warp(image, T, dsize, axes):
    """Stub for apply_warp: return zeros with the requested output size."""
    w, h = dsize
    return np.zeros((h, w), dtype=np.uint8)


def test_if_positive_path_smoke(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (da.from_array(np.zeros((2, 8, 8), dtype=np.uint8)), {}, "CYX", 1.0),
    )
    monkeypatch.setattr(registration, "register_images_standalone", _dummy_register_images_standalone)
    monkeypatch.setattr(registration, "apply_warp", _dummy_apply_warp)

    registration.register_images(
        data=dummy_data,
        image_to_be_registered=image_file,
        channel_names=["DAPI", "FITC"],
        channel_name_for_registration="DAPI",
        save_registered_images=False,
    )

    assert len(dummy_data.images.added_images) == 1
    added = dummy_data.images.added_images[0]["kwargs"]
    assert added["channel_names"] == "FITC"
    assert added["axes"] == "YX"
    assert added["image"].shape == (8, 8)

    # Ensure that the registration channel ("DAPI") is not included
    # in any of the images added to the collection.
    for img in dummy_data.images.added_images:
        ch = img["kwargs"].get("channel_names")
        if isinstance(ch, (list, tuple, set)):
            assert "DAPI" not in ch
        else:
            assert ch != "DAPI"


def test_if_positive_path_with_pyramid_list_input(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: ([np.zeros((2, 8, 8), dtype=np.uint8)], {}, "CYX", 1.0),
    )
    monkeypatch.setattr(registration, "register_images_standalone", _dummy_register_images_standalone)
    monkeypatch.setattr(registration, "apply_warp", _dummy_apply_warp)

    registration.register_images(
        data=dummy_data,
        image_to_be_registered=image_file,
        channel_names=["DAPI", "FITC"],
        channel_name_for_registration="DAPI",
        save_registered_images=False,
    )

    assert len(dummy_data.images.added_images) == 1
    added = dummy_data.images.added_images[0]["kwargs"]
    assert added["channel_names"] == "FITC"
    assert added["axes"] == "YX"
    assert added["image"].shape == (8, 8)


def test_template_metadata_pixel_size_is_used(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (da.from_array(np.zeros((2, 8, 8), dtype=np.uint8)), {}, "CYX", 4.0),
    )
    monkeypatch.setattr(registration, "register_images_standalone", _dummy_register_images_standalone)
    monkeypatch.setattr(registration, "apply_warp", _dummy_apply_warp)

    registration.register_images(
        data=dummy_data,
        image_to_be_registered=image_file,
        channel_names=["DAPI", "FITC"],
        channel_name_for_registration="DAPI",
        save_registered_images=False,
    )

    assert len(dummy_data.images.added_images) == 1
    added = dummy_data.images.added_images[0]["kwargs"]
    assert added["pixel_size"] == dummy_data.images.metadata["nuclei"]["pixel_size"]


def test_image_path_alias_is_accepted(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (da.from_array(np.zeros((2, 8, 8), dtype=np.uint8)), {}, "CYX", 1.0),
    )
    monkeypatch.setattr(registration, "register_images_standalone", _dummy_register_images_standalone)
    monkeypatch.setattr(registration, "apply_warp", _dummy_apply_warp)

    registration.register_images(
        data=dummy_data,
        image_path=image_file,
        channel_names=["DAPI", "FITC"],
        channel_name_for_registration="DAPI",
        save_registered_images=False,
    )

    assert len(dummy_data.images.added_images) == 1
    added = dummy_data.images.added_images[0]["kwargs"]
    assert added["channel_names"] == "FITC"


def test_image_path_and_legacy_name_together_raises(dummy_data, image_file):
    with pytest.raises(ValueError, match="Provide only one of `image_to_be_registered` or `image_path`"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            image_path=image_file,
            channel_names=["HE"],
        )


def _raise_not_enough_matches(*args, **kwargs):
    raise NotEnoughFeatureMatchesError(number=1, threshold=5)


def test_insufficient_matches_warns_and_returns_when_configured(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (np.zeros((8, 8, 3), dtype=np.uint8), {}, "YXS", 1.0),
    )
    monkeypatch.setattr(registration, "register_images_standalone", _raise_not_enough_matches)

    with pytest.warns(UserWarning, match="Registration skipped"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["HE"],
            save_registered_images=False,
            raise_on_insufficient_matches=False,
        )

    assert len(dummy_data.images.added_images) == 0


def test_insufficient_matches_raises_when_configured(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (np.zeros((8, 8, 3), dtype=np.uint8), {}, "YXS", 1.0),
    )
    monkeypatch.setattr(registration, "register_images_standalone", _raise_not_enough_matches)

    with pytest.raises(NotEnoughFeatureMatchesError):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["HE"],
            save_registered_images=False,
            raise_on_insufficient_matches=True,
        )

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


# ---------------------------------------------------------------------------
# method="elastix"
# ---------------------------------------------------------------------------

class _MetadataImages(_DummyImages):
    """Like the real ImageData, add_image creates a metadata entry for the added name."""

    def add_image(self, *args, **kwargs):
        super().add_image(*args, **kwargs)
        self.metadata[kwargs["channel_names"]] = {"pixel_size": kwargs["pixel_size"], "axes": kwargs["axes"]}


def _identity_transform(fixed_shape=(8, 8), moving_shape=(8, 8), dice=0.95):
    from insitupy.images.registration_elastix import DisplacementTransform
    jy, jx = np.mgrid[0:fixed_shape[0], 0:fixed_shape[1]].astype(np.float32)
    return DisplacementTransform(
        map_xy=np.stack([jx, jy], -1), grid_f=(1.0, 1.0),
        fixed_shape=fixed_shape, moving_shape=moving_shape,
        pixel_size_fixed=1.0, pixel_size_moving=1.0, affine=np.eye(3)[:2],
        metrics={"nonrigid": True, "tissue_dice": dice, "structure_ncc": 0.5,
                 "min_relative_jacobian": 0.9, "precision_note": "tissue-scale"},
    )


def test_elastix_if_path_warps_other_channels_and_records_metadata(dummy_data, image_file, monkeypatch):
    import json

    dummy_data.images = _MetadataImages()
    img = np.zeros((2, 8, 8), dtype=np.uint16)
    img[1, 2:5, 3:6] = 1000
    monkeypatch.setattr(registration, "read_image", lambda _p: ([da.from_array(img)], {}, "CYX", 1.0))
    calls = {}

    def _fake_elastix(moving, fixed, **kwargs):
        calls.update(kwargs, moving=moving)
        return None, _identity_transform()

    monkeypatch.setattr(registration, "register_images_elastix", _fake_elastix)

    registration.register_images(
        data=dummy_data,
        image_path=image_file,
        channel_names=["DAPI", "FITC"],
        channel_name_for_registration="DAPI",
        save_registered_images=False,
        method="elastix",
        test_flipping=False,
        elastix_config={"nonrigid": False},
    )

    # registration ran on the DAPI channel only, without a full-res warp
    assert calls["axes_moving"] == "YX" and calls["warp"] is False
    assert calls["nonrigid"] is False and calls["test_flipping"] is False
    assert calls["pixel_size_fixed"] == 1.0
    assert len(calls["moving"]) == 1 and calls["moving"][0].shape == (8, 8)

    # the other channel went through the (identity) displacement transform
    assert [a["kwargs"]["channel_names"] for a in dummy_data.images.added_images] == ["FITC"]
    np.testing.assert_array_equal(dummy_data.images.added_images[0]["kwargs"]["image"], img[1])

    reg = dummy_data.images.metadata["FITC"]["registration"]
    assert reg["method"] == "elastix"
    json.dumps(reg)  # persisted via zarr attrs


def test_elastix_low_quality_warns(dummy_data, image_file, monkeypatch):
    dummy_data.images = _MetadataImages()
    monkeypatch.setattr(registration, "read_image", lambda _p: (np.zeros((8, 8, 3), np.uint8), {}, "YXS", 1.0))
    monkeypatch.setattr(
        registration, "register_images_elastix",
        lambda moving, fixed, **kw: (np.zeros((8, 8, 3), np.uint8), _identity_transform(dice=0.4)),
    )
    with pytest.warns(UserWarning, match="may have failed"):
        registration.register_images(
            data=dummy_data, image_path=image_file, channel_names=["HE"],
            save_registered_images=False, method="elastix",
        )
    assert dummy_data.images.metadata["HE"]["registration"]["tissue_dice"] == 0.4


def test_unknown_method_raises(dummy_data, image_file):
    with pytest.raises(ValueError, match="`method` must be"):
        registration.register_images(
            data=dummy_data, image_path=image_file, channel_names=["HE"], method="sift",
        )


def test_unknown_elastix_config_key_raises(dummy_data, image_file, monkeypatch):
    monkeypatch.setattr(registration, "read_image", lambda _p: (np.zeros((8, 8, 3), np.uint8), {}, "YXS", 1.0))
    with pytest.raises(TypeError, match="Unknown elastix_config keys"):
        registration.register_images(
            data=dummy_data, image_path=image_file, channel_names=["HE"],
            save_registered_images=False, method="elastix", elastix_config={"nonrigd": False},
        )

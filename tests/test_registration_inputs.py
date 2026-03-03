from pathlib import Path

import dask.array as da
import numpy as np
import pytest

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


def test_pixel_size_overrides_non_positive_raise(dummy_data, image_file):
    with pytest.raises(ValueError, match="`pixel_size_image_override` must be > 0"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["HE"],
            pixel_size_image_override=0,
        )

    with pytest.raises(ValueError, match="`pixel_size_template_override` must be > 0"):
        registration.register_images(
            data=dummy_data,
            image_to_be_registered=image_file,
            channel_names=["HE"],
            pixel_size_template_override=-1,
        )


def test_if_positive_path_smoke(dummy_data, image_file, monkeypatch):
    class _DummyImageRegistration:
        def __init__(self, image, template, *args, **kwargs):
            self.image = image
            self.template = template
            self.image_resized = None
            self.image_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.template_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.registered = None

        def load_and_scale_images(self, *args, **kwargs):
            return None

        def extract_features(self, *args, **kwargs):
            return None

        def calculate_transformation_matrix(self):
            return None

        def perform_registration(self):
            if self.image is None:
                self.registered = np.zeros((8, 8), dtype=np.uint8)
            elif hasattr(self.image, "compute"):
                self.registered = self.image.compute()
            else:
                self.registered = np.asarray(self.image)

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (da.from_array(np.zeros((2, 8, 8), dtype=np.uint8)), {}, "CYX", 1.0),
    )
    monkeypatch.setattr(registration, "ImageRegistration", _DummyImageRegistration)

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
    class _DummyImageRegistration:
        def __init__(self, image, template, *args, **kwargs):
            self.image = image
            self.template = template
            self.image_resized = None
            self.image_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.template_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.registered = None

        def load_and_scale_images(self, *args, **kwargs):
            return None

        def extract_features(self, *args, **kwargs):
            return None

        def calculate_transformation_matrix(self):
            return None

        def perform_registration(self):
            self.registered = np.asarray(self.image)

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: ([np.zeros((2, 8, 8), dtype=np.uint8)], {}, "CYX", 1.0),
    )
    monkeypatch.setattr(registration, "ImageRegistration", _DummyImageRegistration)

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


def test_pixel_size_overrides_are_used(dummy_data, image_file, monkeypatch):
    class _DummyImageRegistration:
        def __init__(self, image, template, *args, **kwargs):
            self.image = image
            self.template = template
            self.image_resized = None
            self.image_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.template_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.registered = None

        def load_and_scale_images(self, *args, **kwargs):
            return None

        def extract_features(self, *args, **kwargs):
            return None

        def calculate_transformation_matrix(self):
            return None

        def perform_registration(self):
            if hasattr(self.image, "compute"):
                self.registered = self.image.compute()
            else:
                self.registered = np.asarray(self.image)

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (da.from_array(np.zeros((2, 8, 8), dtype=np.uint8)), {}, "CYX", 4.0),
    )
    monkeypatch.setattr(registration, "ImageRegistration", _DummyImageRegistration)

    registration.register_images(
        data=dummy_data,
        image_to_be_registered=image_file,
        channel_names=["DAPI", "FITC"],
        channel_name_for_registration="DAPI",
        save_registered_images=False,
        pixel_size_image_override=2.5,
        pixel_size_template_override=0.8,
    )

    assert len(dummy_data.images.added_images) == 1
    added = dummy_data.images.added_images[0]["kwargs"]
    assert added["pixel_size"] == 0.8


def test_image_path_alias_is_accepted(dummy_data, image_file, monkeypatch):
    class _DummyImageRegistration:
        def __init__(self, image, template, *args, **kwargs):
            self.image = image
            self.template = template
            self.image_resized = None
            self.image_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.template_scaled = np.zeros((1, 1), dtype=np.uint8)
            self.registered = None

        def load_and_scale_images(self, *args, **kwargs):
            return None

        def extract_features(self, *args, **kwargs):
            return None

        def calculate_transformation_matrix(self):
            return None

        def perform_registration(self):
            if hasattr(self.image, "compute"):
                self.registered = self.image.compute()
            else:
                self.registered = np.asarray(self.image)

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        registration,
        "read_image",
        lambda _p: (da.from_array(np.zeros((2, 8, 8), dtype=np.uint8)), {}, "CYX", 1.0),
    )
    monkeypatch.setattr(registration, "ImageRegistration", _DummyImageRegistration)

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

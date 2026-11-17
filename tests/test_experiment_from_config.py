"""Tests for the reader dispatch of ``InSituExperiment.from_config`` and ``InSituExperiment.add``."""
import pandas as pd
import pytest

import insitupy.io.data as io_data
from insitupy import InSituData, InSituExperiment
from insitupy._io._dispatch import _detect_reader
from insitupy.datasets import xenium_test_dataset_v4_nucex
from insitupy.datasets.datasets import _atera_test_dataset_dir


@pytest.fixture(scope="module")
def xenium_dir():
    return xenium_test_dataset_v4_nucex().path


@pytest.fixture(scope="module")
def atera_dir():
    return _atera_test_dataset_dir()


def _touch(root, *names):
    for name in names:
        p = root / name
        if name.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
    return root


def _dummy_data(slide_id="s"):
    return InSituData(
        path=None, metadata=None, slide_id=slide_id, sample_id=slide_id,
        method_name="test", method_params={},
    )


# ---------------------------------------------------------------- detection

@pytest.mark.parametrize("markers, expected", [
    ([".ispy"], "insitupy"),
    (["experiment.xenium"], "xenium"),
    (["experiment.spatial"], "atera"),
    (["morphology_2d/"], "atera"),
    (["experiment.xenium", "experiment.spatial"], "atera"),
    (["binned_outputs/"], "visium_hd"),
    (["segmented_outputs/"], "visium_hd"),
    (["spatial/scalefactors_json.json"], "visium"),
    (["binned_outputs/", "spatial/tissue_hires_image.png"], "visium_hd"),
])
def test_detect_reader_each_format(tmp_path, markers, expected):
    _touch(tmp_path, *markers)
    assert _detect_reader(tmp_path) == expected


def test_detect_reader_no_marker_raises(tmp_path):
    with pytest.raises(ValueError, match="experiment.xenium"):
        _detect_reader(tmp_path)


@pytest.mark.parametrize("markers, readers", [
    ([".ispy", "experiment.xenium"], ("insitupy", "xenium")),
    (["experiment.xenium", "binned_outputs/"], ("xenium", "visium_hd")),
])
def test_detect_reader_ambiguous_raises(tmp_path, markers, readers):
    _touch(tmp_path, *markers)
    with pytest.raises(ValueError, match="Ambiguous") as exc:
        _detect_reader(tmp_path)
    for r in readers:
        assert r in str(exc.value)


# ----------------------------------------------------------------- dispatch

def test_from_config_mixed_formats(tmp_path, xenium_dir, atera_dir):
    project = tmp_path / "project"
    io_data.read_xenium(xenium_dir, verbose=False).saveas(project, verbose=False)

    config = pd.DataFrame({
        "directory": [str(xenium_dir), str(atera_dir), str(project)],
        "condition": ["a", "b", "c"],
    })
    exp = InSituExperiment.from_config(config, mode="auto")

    assert len(exp.data) == 3
    assert [d.metadata["method"] for d in exp.data] == ["Xenium", "Atera", "Xenium"]
    assert "directory" not in exp.metadata.columns
    assert "reader" not in exp.metadata.columns
    assert list(exp.metadata["condition"]) == ["a", "b", "c"]


def test_reader_column_overrides_mode(xenium_dir):
    config = pd.DataFrame({
        "directory": [str(xenium_dir), str(xenium_dir)],
        "reader": ["atera", ""],
    })
    exp = InSituExperiment.from_config(config, mode="xenium")

    assert [d.metadata["method"] for d in exp.data] == ["Atera", "Xenium"]
    assert "reader" not in exp.metadata.columns


def test_reader_column_nan_falls_back_to_mode(tmp_path):
    d = _touch(tmp_path / "a", "binned_outputs/")
    called = []
    config = pd.DataFrame({"directory": [str(d)], "reader": [float("nan")]})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(io_data, "read_visium_hd", lambda p, **kw: called.append(kw) or _dummy_data())
        InSituExperiment.from_config(config, mode="visium_hd")
    assert len(called) == 1


def test_reader_kwargs_routing(tmp_path, monkeypatch):
    hd = _touch(tmp_path / "hd", "binned_outputs/")
    vis = _touch(tmp_path / "vis", "spatial/scalefactors_json.json")
    calls = {}

    def recorder(name):
        def _read(path, **kwargs):
            calls.setdefault(name, []).append(kwargs)
            return _dummy_data(name)
        return _read

    monkeypatch.setattr(io_data, "read_visium", recorder("visium"))
    monkeypatch.setattr(io_data, "read_visium_hd", recorder("visium_hd"))

    config = pd.DataFrame({"directory": [str(hd), str(vis)]})
    InSituExperiment.from_config(config, reader_kwargs={
        "visium": {"fullres_pixel_size": 0.5},
        "visium_hd": {"bin_size": 16},
    })
    assert calls["visium"] == [{"verbose": False, "fullres_pixel_size": 0.5}]
    assert calls["visium_hd"] == [{"verbose": False, "bin_size": 16}]

    calls.clear()
    InSituExperiment.from_config(config, reader_kwargs={
        "visium": {"fullres_pixel_size": 0.5},
        "visium_hd": {"verbose": True},
    })
    assert calls["visium_hd"] == [{"verbose": True}]


def test_from_config_reports_all_bad_rows_before_reading(tmp_path, monkeypatch):
    good = _touch(tmp_path / "good", "binned_outputs/")
    empty = tmp_path / "empty"
    empty.mkdir()

    def must_not_run(path, **kwargs):
        raise AssertionError("reader called before validation finished")

    monkeypatch.setattr(io_data, "read_visium_hd", must_not_run)

    config = pd.DataFrame({
        "directory": [str(good), str(tmp_path / "missing"), str(good), str(empty)],
        "reader": ["", "", "bogus", ""],
    })
    with pytest.raises(ValueError) as exc:
        InSituExperiment.from_config(config, mode="auto")
    msg = str(exc.value)
    assert "row 1" in msg and "row 2" in msg and "row 3" in msg
    assert "row 0" not in msg
    assert "No such directory" in msg and "bogus" in msg and "auto-detect" in msg


def test_unknown_reader_kwargs_key_raises(xenium_dir):
    config = pd.DataFrame({"directory": [str(xenium_dir)]})
    with pytest.raises(ValueError, match="xenum"):
        InSituExperiment.from_config(config, reader_kwargs={"xenum": {}})


def test_add_auto_and_reader_kwargs(atera_dir):
    exp = InSituExperiment()
    exp.add(atera_dir, mode="auto")
    assert exp.data[0].metadata["method"] == "Atera"

    with pytest.raises(ValueError, match="only apply when"):
        exp.add(_dummy_data(), foo=1)
    with pytest.raises(ValueError, match="no reader arguments"):
        exp.add(atera_dir, mode="insitupy", foo=1)
    with pytest.raises(ValueError, match="Invalid mode"):
        exp.add(atera_dir, mode="bogus")

"""Round-trip tests for InSituData.quicksave / list_quicksaves / load_quicksave (B4).

All three were broken end to end before 0.12 (TypeError in quicksave, AttributeError in
list_quicksaves on a fresh object, NameError / missing scale_factor in load_quicksave).
The quicksave cache is redirected to ``tmp_path``.
"""

import logging

import geopandas as gpd
import pytest
from shapely.geometry import Point

import insitupy._core.data as core_data
from insitupy._core.data import InSituData
from insitupy.containers.shapes_data import AnnotationsData


@pytest.fixture
def quicksave_dir(tmp_path, monkeypatch):
    qdir = tmp_path / "quicksaves"
    monkeypatch.setattr(core_data, "_QUICKSAVE_DIR", qdir)
    return qdir


def _make_xd(slide_id="slide1", sample_id="sampleA"):
    xd = InSituData(path=None, metadata=None, slide_id=slide_id, sample_id=sample_id,
                    method_name="t", method_params={})
    poly = gpd.GeoDataFrame({
        "id": ["a_0"], "name": ["Tumor"], "color": [[255, 0, 0]],
        "geometry": [Point(5, 5).buffer(5)],
    })
    xd._annotations.add_data(data=poly, key="pathology", scale_factor=1.0)
    return xd


@pytest.mark.parametrize("slide_id", ["slide1", "run__7"])  # "__" is the name separator
def test_quicksave_roundtrip(quicksave_dir, slide_id):
    xd = _make_xd(slide_id=slide_id)
    original = xd.annotations["pathology"].geometry.iloc[0]

    xd.quicksave(note="before edit")

    listing = xd.list_quicksaves()
    assert len(listing) == 1
    assert listing["note"].iloc[0] == "before edit"
    assert listing["slide_id"].iloc[0] == slide_id
    uid = listing["uid"].iloc[0]

    xd._annotations = AnnotationsData()
    xd.load_quicksave(uid)

    assert "pathology" in xd.annotations.keys()
    restored = xd.annotations["pathology"].geometry.iloc[0]
    assert restored.equals_exact(original, tolerance=1e-9)


def test_list_quicksaves_filters_to_this_object(quicksave_dir):
    _make_xd(slide_id="slide1").quicksave()
    _make_xd(slide_id="other").quicksave()
    # a folder that is not a quicksave must be ignored, not crash the parser
    (quicksave_dir / "my_notes").mkdir()

    listing = _make_xd(slide_id="slide1").list_quicksaves()

    assert list(listing["slide_id"]) == ["slide1"]


def test_list_quicksaves_without_cache_dir_is_empty(quicksave_dir):
    listing = _make_xd().list_quicksaves()

    assert listing.empty
    assert list(listing.columns) == ["slide_id", "sample_id", "savetime", "uid", "note"]


def test_load_unknown_quicksave_warns(quicksave_dir, caplog):
    xd = _make_xd()
    # the `insitupy` logger does not propagate to the root logger caplog listens on
    logger = logging.getLogger("insitupy")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="insitupy"):
            xd.load_quicksave("nope")
    finally:
        logger.removeHandler(caplog.handler)

    assert "No quicksave with uid 'nope'" in caplog.text
    assert list(xd.annotations.keys()) == ["pathology"]

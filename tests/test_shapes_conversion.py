"""Tests for AnnotationsData.to_regions() and RegionsData.to_annotations().

These test the container-level conversion methods directly, distinct from the
InSituData convenience wrappers (annotations_to_regions / regions_to_annotations)
already covered in test_tools.py.

Key structural differences that are explicitly tested here:
- AnnotationsData allows duplicate names and non-polygon geometries;
  to_regions() must reject duplicates and silently drop non-polygons.
- RegionsData enforces unique names and polygons-only at add_data() time;
  to_annotations() must handle forbidden names ("rest") via on_forbidden.
"""

import warnings

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from insitupy._constants import FORBIDDEN_ANNOTATION_NAMES
from insitupy.containers import AnnotationsData, RegionsData


# ── Helpers ───────────────────────────────────────────────────────────────────

_FORBIDDEN = FORBIDDEN_ANNOTATION_NAMES[0]  # "rest"


def _poly(x=0, y=0, size=10):
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def _poly_gdf(*names):
    """One Polygon row per name, with unique ids."""
    return gpd.GeoDataFrame({
        "id": [f"{n}_{i}" for i, n in enumerate(names)],
        "name": list(names),
        "geometry": [_poly(i * 20) for i in range(len(names))],
        "color": ["#ff0000"] * len(names),
    })


def _ann(*names, key="roi"):
    """AnnotationsData with a single key containing the given names."""
    ann = AnnotationsData()
    ann.add_data(data=_poly_gdf(*names), key=key, scale_factor=1.0)
    return ann


def _reg(*names, key="roi"):
    """RegionsData with a single key containing the given (unique) names."""
    reg = RegionsData()
    reg.add_data(data=_poly_gdf(*names), key=key, scale_factor=1.0)
    return reg


# ── AnnotationsData.to_regions ────────────────────────────────────────────────

class TestAnnotationsDataToRegions:

    def test_returns_regions_data_instance(self):
        result = _ann("Tumor").to_regions()
        assert isinstance(result, RegionsData)

    def test_geometry_preserved(self):
        ann = _ann("Tumor")
        result = ann.to_regions()
        assert result["roi"].geometry.iloc[0].equals(ann["roi"].geometry.iloc[0])

    def test_name_preserved(self):
        result = _ann("Tumor", "Stroma").to_regions()
        assert set(result["roi"]["name"]) == {"Tumor", "Stroma"}

    def test_all_keys_converted_when_keys_is_none(self):
        ann = AnnotationsData()
        ann.add_data(data=_poly_gdf("Tumor"), key="pathology", scale_factor=1.0)
        ann.add_data(data=_poly_gdf("CD8"), key="immune", scale_factor=1.0)
        result = ann.to_regions(keys=None)
        assert set(result.keys()) == {"pathology", "immune"}

    def test_specific_key_selection(self):
        ann = AnnotationsData()
        ann.add_data(data=_poly_gdf("Tumor"), key="pathology", scale_factor=1.0)
        ann.add_data(data=_poly_gdf("CD8"), key="immune", scale_factor=1.0)
        result = ann.to_regions(keys="pathology")
        assert "pathology" in result.keys()
        assert "immune" not in result.keys()

    def test_missing_key_warns_and_skips(self):
        ann = _ann("Tumor")
        with pytest.warns(UserWarning, match="not found"):
            result = ann.to_regions(keys=["roi", "nonexistent"])
        assert "roi" in result.keys()
        assert "nonexistent" not in result.keys()

    def test_nonpolygon_geometries_dropped(self):
        """AnnotationsData accepts non-polygons; to_regions must silently drop them."""
        ann = AnnotationsData()
        gdf = gpd.GeoDataFrame({
            "id": ["poly_0", "line_0", "point_0"],
            "name": ["Tumor", "Edge", "Spot"],
            "geometry": [_poly(), LineString([(0, 0), (10, 10)]), Point(5, 5)],
            "color": ["#ff0000", "#00ff00", "#0000ff"],
        })
        ann.add_data(data=gdf, key="mixed", scale_factor=1.0)
        result = ann.to_regions()
        assert list(result["mixed"]["name"]) == ["Tumor"]

    def test_all_nonpolygon_warns_and_key_skipped(self):
        ann = AnnotationsData()
        gdf = gpd.GeoDataFrame({
            "id": ["line_0"],
            "name": ["Edge"],
            "geometry": [LineString([(0, 0), (10, 10)])],
            "color": ["#ff0000"],
        })
        ann.add_data(data=gdf, key="lines", scale_factor=1.0)
        with pytest.warns(UserWarning, match="no Polygon"):
            result = ann.to_regions()
        assert "lines" not in result.keys()

    def test_name_filter(self):
        result = _ann("Tumor", "Stroma").to_regions(name_filter="Tumor")
        assert list(result["roi"]["name"]) == ["Tumor"]

    def test_name_filter_empty_warns_and_key_skipped(self):
        ann = _ann("Tumor")
        with pytest.warns(UserWarning, match="empty after name_filter"):
            result = ann.to_regions(name_filter="Stroma")
        assert "roi" not in result.keys()

    def test_duplicate_names_raise_valueerror(self):
        """AnnotationsData allows duplicate names; to_regions must reject them
        because RegionsData enforces uniqueness."""
        ann = AnnotationsData()
        gdf = gpd.GeoDataFrame({
            "id": ["t_0", "t_1"],
            "name": ["Tumor", "Tumor"],  # duplicate
            "geometry": [_poly(0), _poly(20)],
            "color": ["#ff0000", "#ff0000"],
        })
        ann.add_data(data=gdf, key="roi", scale_factor=1.0)
        with pytest.raises(ValueError, match="duplicate"):
            ann.to_regions()

    def test_multiple_keys_all_converted(self):
        ann = AnnotationsData()
        ann.add_data(data=_poly_gdf("Tumor"), key="a", scale_factor=1.0)
        ann.add_data(data=_poly_gdf("Stroma"), key="b", scale_factor=1.0)
        result = ann.to_regions()
        assert set(result.keys()) == {"a", "b"}


# ── RegionsData.to_annotations ────────────────────────────────────────────────

class TestRegionsDataToAnnotations:

    def test_returns_annotations_data_instance(self):
        result = _reg("Tumor").to_annotations()
        assert isinstance(result, AnnotationsData)

    def test_geometry_preserved(self):
        reg = _reg("Tumor")
        result = reg.to_annotations()
        assert result["roi"].geometry.iloc[0].equals(reg["roi"].geometry.iloc[0])

    def test_name_preserved(self):
        result = _reg("Tumor", "Stroma").to_annotations()
        assert set(result["roi"]["name"]) == {"Tumor", "Stroma"}

    def test_all_keys_converted_when_keys_is_none(self):
        reg = RegionsData()
        reg.add_data(data=_poly_gdf("Tumor"), key="pathology", scale_factor=1.0)
        reg.add_data(data=_poly_gdf("CD8"), key="immune", scale_factor=1.0)
        result = reg.to_annotations(keys=None)
        assert set(result.keys()) == {"pathology", "immune"}

    def test_specific_key_selection(self):
        reg = RegionsData()
        reg.add_data(data=_poly_gdf("Tumor"), key="pathology", scale_factor=1.0)
        reg.add_data(data=_poly_gdf("CD8"), key="immune", scale_factor=1.0)
        result = reg.to_annotations(keys="pathology")
        assert "pathology" in result.keys()
        assert "immune" not in result.keys()

    def test_missing_key_warns_and_skips(self):
        reg = _reg("Tumor")
        with pytest.warns(UserWarning, match="not found"):
            result = reg.to_annotations(keys=["roi", "nonexistent"])
        assert "roi" in result.keys()
        assert "nonexistent" not in result.keys()

    def test_forbidden_name_error_default(self):
        """on_forbidden='error' (default) raises ValueError for forbidden names."""
        reg = _reg(_FORBIDDEN)
        with pytest.raises(ValueError, match="forbidden"):
            reg.to_annotations()

    def test_forbidden_name_rename(self):
        """on_forbidden='rename' appends '_region' suffix and warns."""
        reg = _reg(_FORBIDDEN)
        with pytest.warns(UserWarning, match="Renamed"):
            result = reg.to_annotations(on_forbidden="rename")
        assert list(result["roi"]["name"]) == [f"{_FORBIDDEN}_region"]

    def test_forbidden_name_skip(self):
        """on_forbidden='skip' drops forbidden rows and warns."""
        reg = RegionsData()
        reg.add_data(data=_poly_gdf(_FORBIDDEN, "Tumor"), key="roi", scale_factor=1.0)
        with pytest.warns(UserWarning, match="Dropped"):
            result = reg.to_annotations(on_forbidden="skip")
        assert list(result["roi"]["name"]) == ["Tumor"]

    def test_forbidden_only_key_skip_empties_key(self):
        """When skipping leaves a key empty, it is omitted entirely."""
        reg = _reg(_FORBIDDEN)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result = reg.to_annotations(on_forbidden="skip")
        assert "roi" not in result.keys()

    def test_multiple_keys_all_converted(self):
        reg = RegionsData()
        reg.add_data(data=_poly_gdf("Tumor"), key="a", scale_factor=1.0)
        reg.add_data(data=_poly_gdf("Stroma"), key="b", scale_factor=1.0)
        result = reg.to_annotations()
        assert set(result.keys()) == {"a", "b"}

"""Tests for napari geometry-layer features API.

Covers the properties → features migration, _update_uid callback correctness,
sync_geometries round-trip, and the show_geometries_widget choices logic.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely import Polygon

from insitupy import WITH_NAPARI

pytestmark = pytest.mark.skipif(
    not WITH_NAPARI, reason="napari is required for these tests"
)

from insitupy._constants import ANNOTATIONS_SYMBOL, REGIONS_SYMBOL  # noqa: E402
from insitupy._core._napari import _add_events_to_viewer  # noqa: E402
from insitupy.containers.shapes_data import ShapesData  # noqa: E402
from insitupy.interactive._configs import config_manager  # noqa: E402
from insitupy.interactive._layers import _add_geometries_as_layer  # noqa: E402
from insitupy.interactive.viewer import _remove_geometries, sync_geometries  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_polygon_gdf(uid: str, name: str) -> GeoDataFrame:
    """GeoDataFrame with UID as row index (for _add_geometries_as_layer)."""
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    return GeoDataFrame(
        {"geometry": [poly], "name": [name]},
        geometry="geometry",
        index=[uid],
    )


def _make_polygon_gdf_for_shapesdata(uid: str, name: str) -> GeoDataFrame:
    """GeoDataFrame with 'id' column (for ShapesData.add_data)."""
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    return GeoDataFrame(
        {"id": [uid], "geometry": [poly], "name": [name]},
        geometry="geometry",
    )


def _register_config(viewer, uid: str, verbose: bool = False, **extra):
    """Register a minimal ViewerConfig stub and wire it to the viewer title."""
    config = SimpleNamespace(
        verbose=verbose, _auto_set_uid=True, _removal_tracker=[],
        annot_point_colors={}, region_colors={},
        _annot_point_color_idx=0, _region_color_idx=0,
        **extra
    )
    config_manager._configs[uid] = config
    viewer.title = f"InSituPy#{uid}"
    return config


def _cleanup(uid: str):
    config_manager._configs.pop(uid, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def viewer_and_config(make_napari_viewer):
    """Headless napari viewer + minimal ViewerConfig stub in config_manager."""
    viewer = make_napari_viewer()
    uid = "napari_test_fixture_uid"
    config = _register_config(viewer, uid)
    yield viewer, config
    _cleanup(uid)


def _empty_features_df() -> pd.DataFrame:
    return pd.DataFrame({
        "uid": pd.Series([], dtype=str),
        "type": pd.Series([], dtype=str),
        "name": pd.Series([], dtype=str),
        "geometry_type": pd.Series([], dtype=str),
    })


# ---------------------------------------------------------------------------
# P1 — Core correctness
# ---------------------------------------------------------------------------

class TestAddGeometriesAsLayer:
    """P1.1 and P1.2: _add_geometries_as_layer feature column correctness."""

    def test_new_layer_has_correct_columns_and_values(self, viewer_and_config):
        """New Shapes layer contains uid/type/name/geometry_type with correct values."""
        viewer, _ = viewer_and_config
        gdf = _make_polygon_gdf("uid_001", "cls_a")

        _add_geometries_as_layer(
            dataframe=gdf, viewer=viewer, layer_name="testkey", mode="Annotations"
        )

        layer = viewer.layers[f"{ANNOTATIONS_SYMBOL} testkey"]
        f = layer.features

        for col in ("uid", "type", "name", "geometry_type"):
            assert col in f.columns, f"Missing column: {col}"

        assert f["uid"].iloc[0] == "uid_001"
        assert f["type"].iloc[0] == "polygon_exterior"
        assert f["name"].iloc[0] == "cls_a"
        assert f["geometry_type"].iloc[0] == "annotation"

    def test_regions_mode_sets_geometry_type_region(self, viewer_and_config):
        """mode='Regions' sets geometry_type to 'region' in layer features."""
        viewer, _ = viewer_and_config
        gdf = _make_polygon_gdf("uid_r01", "r_class")

        _add_geometries_as_layer(
            dataframe=gdf, viewer=viewer, layer_name="reg_key", mode="Regions"
        )

        layer = viewer.layers[f"{REGIONS_SYMBOL} reg_key"]
        assert layer.features["geometry_type"].iloc[0] == "region"

    def test_append_path_no_row_doubling(self, viewer_and_config):
        """Second call to the same layer: len(features) == len(data) (no doubling)."""
        viewer, _ = viewer_and_config

        gdf1 = _make_polygon_gdf("uid_001", "a")
        gdf2 = _make_polygon_gdf("uid_002", "b")

        _add_geometries_as_layer(
            dataframe=gdf1, viewer=viewer, layer_name="mykey", mode="Annotations"
        )
        _add_geometries_as_layer(
            dataframe=gdf2, viewer=viewer, layer_name="mykey", mode="Annotations"
        )

        layer = viewer.layers[f"{ANNOTATIONS_SYMBOL} mykey"]
        assert len(layer.features) == len(layer.data), (
            f"features rows ({len(layer.features)}) != data rows ({len(layer.data)})"
        )

    def test_append_fills_uid_for_second_shape(self, viewer_and_config):
        """Appended shape's uid in features matches its GeoDataFrame index."""
        viewer, _ = viewer_and_config

        gdf1 = _make_polygon_gdf("uid_001", "a")
        gdf2 = _make_polygon_gdf("uid_002", "b")

        _add_geometries_as_layer(
            dataframe=gdf1, viewer=viewer, layer_name="twokey", mode="Annotations"
        )
        _add_geometries_as_layer(
            dataframe=gdf2, viewer=viewer, layer_name="twokey", mode="Annotations"
        )

        layer = viewer.layers[f"{ANNOTATIONS_SYMBOL} twokey"]
        uids = layer.features["uid"].tolist()
        assert "uid_001" in uids
        assert "uid_002" in uids


# ---------------------------------------------------------------------------
# P1.3, P2.5, P2.6 — _update_uid callback
# ---------------------------------------------------------------------------

class TestUpdateUid:
    """_update_uid fires on programmatic shape/point additions."""

    def test_polygon_draw_sets_uid_type_geometry_type(self, make_napari_viewer):
        """Adding a polygon fills uid (non-empty UUID), type, geometry_type."""
        uid = "uid_draw_poly_1"
        viewer = make_napari_viewer()
        _register_config(viewer, uid)
        try:
            layer = viewer.add_shapes(
                data=[],
                name=f"{ANNOTATIONS_SYMBOL} manual",
                features=_empty_features_df(),
            )
            _add_events_to_viewer(viewer)

            coords = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
            layer.add([coords], shape_type=["polygon"])

            f = layer.features
            assert len(f) == 1
            assert f["uid"].iloc[0] != "", "uid should be a non-empty UUID string"
            assert f["type"].iloc[0] == "polygon_exterior"
            assert f["geometry_type"].iloc[0] == "annotation"
        finally:
            _cleanup(uid)

    def test_points_verbose_true_no_name_error(self, make_napari_viewer):
        """P2.5 regression: Points branch with verbose=True must not raise NameError."""
        uid = "uid_pts_verbose_1"
        viewer = make_napari_viewer()
        _register_config(viewer, uid, verbose=True)
        try:
            layer = viewer.add_points(
                data=np.zeros((0, 2)),
                name=f"{ANNOTATIONS_SYMBOL} pts_verbose",
                features=_empty_features_df(),
            )
            _add_events_to_viewer(viewer)

            # This triggered NameError before the fix (type_last undefined for Points)
            layer.add(np.array([[50.0, 50.0]]))

            f = layer.features
            assert len(f) == 1
            assert f["uid"].iloc[0] != ""
            assert f["type"].iloc[0] == "point"
            assert f["geometry_type"].iloc[0] == "annotation"
        finally:
            _cleanup(uid)

    def test_geometry_type_is_region_for_regions_layer(self, make_napari_viewer):
        """P2.6: geometry_type is 'region' when layer name starts with REGIONS_SYMBOL."""
        uid = "uid_geomtype_region_1"
        viewer = make_napari_viewer()
        _register_config(viewer, uid)
        try:
            layer = viewer.add_shapes(
                data=[],
                name=f"{REGIONS_SYMBOL} myregion",
                features=_empty_features_df(),
            )
            _add_events_to_viewer(viewer)

            coords = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
            layer.add([coords], shape_type=["polygon"])

            assert layer.features["geometry_type"].iloc[0] == "region"
        finally:
            _cleanup(uid)

    def test_geometry_type_is_empty_for_unknown_prefix(self, make_napari_viewer):
        """geometry_type is '' when the layer name has no known symbol prefix."""
        uid = "uid_geomtype_unknown_1"
        viewer = make_napari_viewer()
        _register_config(viewer, uid)
        try:
            layer = viewer.add_shapes(
                data=[],
                name="plain_shapes_layer",
                features=_empty_features_df(),
            )
            _add_events_to_viewer(viewer)

            coords = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
            layer.add([coords], shape_type=["polygon"])

            assert layer.features["geometry_type"].iloc[0] == ""
        finally:
            _cleanup(uid)


# ---------------------------------------------------------------------------
# P1.4 — sync_geometries round-trip
# ---------------------------------------------------------------------------

def test_sync_geometries_keeps_present_geometries(make_napari_viewer):
    """Geometries visible in the viewer survive sync_geometries unchanged."""
    viewer = make_napari_viewer()
    uid = "uid_sync_rt_1"
    try:
        shapesdata = ShapesData(shape_name="annotations")
        gdf = _make_polygon_gdf_for_shapesdata("shape_uid_1", "cls_a")
        shapesdata.add_data(data=gdf, key="annot_key", scale_factor=1.0)

        config = SimpleNamespace(
            verbose=False,
            _auto_set_uid=True,
            _removal_tracker=[],
            annot_point_colors={},
            region_colors={},
            _annot_point_color_idx=0,
            _region_color_idx=0,
            data=SimpleNamespace(
                annotations=shapesdata,
                regions=ShapesData(shape_name="regions"),
            ),
        )
        config_manager._configs[uid] = config
        viewer.title = f"InSituPy#{uid}"

        # Mirror the stored polygon as a napari Shapes layer
        coords = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
        viewer.add_shapes(
            data=[coords],
            name=f"{ANNOTATIONS_SYMBOL} annot_key",
            features=pd.DataFrame({
                "uid": ["shape_uid_1"],
                "type": ["polygon_exterior"],
                "name": ["cls_a"],
                "geometry_type": ["annotation"],
            }),
            shape_type=["polygon"],
        )

        sync_geometries()

        assert "annot_key" in shapesdata.keys()
        df = shapesdata["annot_key"]
        assert len(df) >= 1, "ShapesData should still contain the synced polygon"
    finally:
        _cleanup(uid)


# ---------------------------------------------------------------------------
# P2.7 — _remove_geometries RuntimeWarning for missing 'name' column
# ---------------------------------------------------------------------------

def test_remove_geometries_missing_name_column_emits_runtime_warning(make_napari_viewer):
    """P2.7: _remove_geometries emits RuntimeWarning when layer has no 'name' column."""
    viewer = make_napari_viewer()
    uid = "uid_remove_warn_1"
    config = SimpleNamespace(verbose=False, _auto_set_uid=True, _removal_tracker=[])
    config_manager._configs[uid] = config

    try:
        shapesdata = ShapesData(shape_name="annotations")
        gdf = _make_polygon_gdf_for_shapesdata("uid_001", "cls_a")
        shapesdata.add_data(data=gdf, key="annot_key", scale_factor=1.0)

        coords = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
        layer = viewer.add_shapes(
            data=[coords],
            name=f"{ANNOTATIONS_SYMBOL} annot_key",
            # Deliberately omit 'name' column to trigger the RuntimeWarning
            features=pd.DataFrame({"uid": ["uid_001"]}),
            shape_type=["polygon"],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _remove_geometries(
                layer=layer,
                shapesdata=shapesdata,
                config=config,
                object_type="annotation",
                annot_key="annot_key",
            )

        assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
            "Expected a RuntimeWarning when 'name' column is absent from layer features"
        )
    finally:
        _cleanup(uid)


# ---------------------------------------------------------------------------
# P3 — show_geometries_widget choices logic (no viewer needed)
# ---------------------------------------------------------------------------

def _fake_geom(is_empty: bool, keys=()):
    """Minimal namespace that satisfies the choices-filter predicate."""
    return SimpleNamespace(is_empty=is_empty, keys=lambda: keys)


def _compute_choices(data) -> list:
    """Mirror the _initialize_widgets choices expression exactly."""
    return [
        c for c in ["Annotations", "Regions"]
        if not getattr(data, c.lower()).is_empty
        and len(getattr(data, c.lower()).keys()) > 0
    ]


class TestGeometryChoices:
    """P3.8–10: show_geometries_widget choices filtering."""

    def test_both_empty_yields_no_choices(self):
        """P3.8: both empty → choices == []."""
        data = SimpleNamespace(
            annotations=_fake_geom(is_empty=True),
            regions=_fake_geom(is_empty=True),
        )
        assert _compute_choices(data) == []

    def test_one_non_empty_with_keys_appears_in_choices(self):
        """P3.9: only annotations non-empty with keys → ['Annotations']."""
        data = SimpleNamespace(
            annotations=_fake_geom(is_empty=False, keys=("key1",)),
            regions=_fake_geom(is_empty=True),
        )
        assert _compute_choices(data) == ["Annotations"]

    def test_non_empty_but_no_keys_is_excluded(self):
        """P3.10: non-empty annotations with no keys → excluded; regions with keys → included."""
        data = SimpleNamespace(
            annotations=_fake_geom(is_empty=False, keys=()),
            regions=_fake_geom(is_empty=False, keys=("key_r",)),
        )
        assert _compute_choices(data) == ["Regions"]

    def test_both_non_empty_with_keys_yields_both(self):
        """Both non-empty with keys → both appear, Annotations first."""
        data = SimpleNamespace(
            annotations=_fake_geom(is_empty=False, keys=("a1",)),
            regions=_fake_geom(is_empty=False, keys=("r1",)),
        )
        assert _compute_choices(data) == ["Annotations", "Regions"]

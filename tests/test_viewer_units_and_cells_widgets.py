"""Tests for the units-layer selector and cells-combo UI polish (2026-07-06/07).

Covers `ViewerConfig.units_key`/`refresh_unit_variables()`, the `cells_layer` threading
into `ViewerConfig.__init__`, and the searchable-combo / call-button enable-gating
wiring added to `show_cells_widget`/`show_units_widget` inside `_initialize_widgets`.

Uses a hand-built, in-memory `InSituData` (no disk I/O) plus a headless
`napari.viewer.ViewerModel` -- see `.log/reports/260707/widget-ui-test-plan/`
for why `ViewerModel` (not `make_napari_viewer`) is used: `_initialize_widgets` only
touches layer data/magicgui state, never a real GL canvas.
"""

from collections import namedtuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point, Polygon

from insitupy import WITH_NAPARI

pytestmark = pytest.mark.skipif(not WITH_NAPARI, reason="napari is required for these tests")

from insitupy._core.data import InSituData  # noqa: E402
from insitupy.containers.boundaries_data import BoundariesData  # noqa: E402
from insitupy.containers.cell_data import CellData  # noqa: E402
from insitupy.containers.spatial_units_data import SpatialUnitsData  # noqa: E402
from insitupy.interactive._configs import ViewerConfig  # noqa: E402
from insitupy.interactive._widgets import _initialize_widgets  # noqa: E402

# ---------------------------------------------------------------------------
# Builders (mirror the in-memory patterns already used across the test suite,
# e.g. tests/test_units_multi.py, tests/test_celldata_sync_safeguards.py)
# ---------------------------------------------------------------------------

def _make_cells_table(obs_names, n_genes=3):
    n = len(obs_names)
    rng = np.random.default_rng(0)
    table = AnnData(
        X=rng.random((n, n_genes)),
        obs=pd.DataFrame(
            {"celltype": (["A", "B"] * n)[:n]},
            index=pd.Index(obs_names, dtype=str),
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    table.obsm["spatial"] = np.array([[float(i * 2), float(i * 2)] for i in range(n)])
    return table


def _make_boundaries():
    boundaries = BoundariesData(cell_names=["c1", "c2", "c3"], seg_mask_value=[1, 2, 3])
    mask = np.zeros((20, 20), dtype=np.uint32)
    mask[0:5, 0:5] = 1
    mask[5:10, 5:10] = 2
    mask[10:15, 10:15] = 3
    nuclei = np.zeros((20, 20), dtype=np.uint32)
    nuclei[1:3, 1:3] = 1
    nuclei[6:8, 6:8] = 2
    nuclei[11:13, 11:13] = 3
    boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=nuclei, pixel_size=1)
    return boundaries


def _make_units(names, unit_type="unit", n_vars=2, seed=0):
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame(
        {"name": names, "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))]}
    )
    table = AnnData(
        X=rng.random((len(names), n_vars)),
        obs=pd.DataFrame(
            {"cluster": ["x"] * len(names)}, index=pd.Index(names, dtype=str)
        ),
        var=pd.DataFrame(index=[f"v{i}" for i in range(n_vars)]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def _make_polygon_gdf(name):
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return gpd.GeoDataFrame(
        {"id": [f"{name}_0"], "name": [name], "geometry": [poly], "color": ["#ff0000"]}
    )


def _build_insitudata(second_cells_layer=False):
    """InSituData with one cell layer, two named units layers ('visium'/'niche'),
    and minimal annotations/regions (GeometriesWidget touches both)."""
    xd = InSituData(
        path=None, metadata=None,
        slide_id="slide1", sample_id="s1",
        method_name="test", method_params={},
    )
    celldata = CellData(table=_make_cells_table(["c1", "c2", "c3"]), boundaries=_make_boundaries())
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    if second_cells_layer:
        alt_celldata = CellData(table=_make_cells_table(["c1", "c2", "c3"]), boundaries=_make_boundaries())
        xd.cells.add_celldata(cd=alt_celldata, key="alt", is_main=False)

    xd.add_units(_make_units(["v1", "v2"], unit_type="visium", seed=1))
    xd.add_units(_make_units(["n1"], unit_type="niche", seed=2), key="niche")

    xd._annotations.add_data(data=_make_polygon_gdf("myannot"), key="my_annot", scale_factor=1.0)
    xd._regions.add_data(data=_make_polygon_gdf("myregion"), key="pathology", scale_factor=1.0)

    return xd


Widgets = namedtuple(
    "Widgets",
    "show_cells move_to_cell geometries show_boundaries select_data filter_cells show_units",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def viewer_model(qapp):
    """Headless napari ViewerModel (data layer only, no Qt window/GL canvas) --
    see module docstring / widget-ui-test-plan for why this works here."""
    from napari.viewer import ViewerModel
    return ViewerModel()


@pytest.fixture
def xdata():
    return _build_insitudata()


@pytest.fixture
def viewer_config(xdata):
    return ViewerConfig(xdata)


@pytest.fixture
def widgets(viewer_model, viewer_config):
    return Widgets(*_initialize_widgets(viewer_model, viewer_config))


# ---------------------------------------------------------------------------
# Tier 1 -- pure ViewerConfig logic, no widget construction
# ---------------------------------------------------------------------------

class TestViewerConfigUnitsKey:
    def test_units_key_defaults_to_main_key(self, viewer_config):
        assert viewer_config.units_key == "visium"
        assert viewer_config.units.table.obs_names.tolist() == ["v1", "v2"]

    def test_refresh_unit_variables_rebuilds_key_dict_from_new_layer(self, viewer_config):
        assert viewer_config.key_dict["unit_vars"] == ["v0", "v1"]

        viewer_config.units_key = "niche"
        viewer_config.refresh_unit_variables()

        # niche layer has its own AnnData (seed=2) but the same var-name scheme;
        # the important assertion is that .units now resolves through the new key.
        assert viewer_config.units.table.obs_names.tolist() == ["n1"]
        assert viewer_config.key_dict["unit_vars"] == ["v0", "v1"]

    def test_units_property_returns_none_if_units_key_unset(self, viewer_config):
        viewer_config.units_key = None
        assert viewer_config.units is None


class TestCellsLayerValidation:
    def test_invalid_cells_layer_raises_value_error(self, xdata):
        with pytest.raises(ValueError, match="not in layers"):
            ViewerConfig(xdata, cells_layer="does_not_exist")

    def test_explicit_valid_cells_layer_resolves_data_name(self, xdata):
        config = ViewerConfig(xdata, cells_layer="main")
        assert config.data_name == "main"

    def test_default_cells_layer_falls_back_to_main_key(self, xdata):
        config = ViewerConfig(xdata)
        assert config.data_name == xdata.cells.main_key


# ---------------------------------------------------------------------------
# Tier 2 -- Qt widget state via qapp + headless ViewerModel
# ---------------------------------------------------------------------------

class TestCellsKeyCombo:
    def test_key_combo_is_searchable_with_contains_completer(self, widgets):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QComboBox

        native = widgets.show_cells.key.native
        assert isinstance(native, QComboBox)
        assert native.isEditable()
        completer = native.completer()
        assert completer is not None
        assert completer.filterMode() == Qt.MatchContains
        assert completer.caseSensitivity() == Qt.CaseInsensitive

    def test_call_button_disabled_until_valid_key_or_recent(self, widgets, viewer_config):
        assert widgets.show_cells.call_button.enabled is False

        widgets.show_cells.key.value = viewer_config.genes[0]
        assert widgets.show_cells.call_button.enabled is True

    def test_call_button_enabled_via_recent_without_key(self, widgets):
        assert widgets.show_cells.call_button.enabled is False
        # "recent" choices are populated by _refresh_widgets_after_data_change from
        # viewer_config.recent_selections; set choices directly to isolate the
        # _cells_show_enabled() gating logic under test.
        widgets.show_cells.recent.choices = ["obs:celltype"]
        widgets.show_cells.recent.value = "obs:celltype"
        assert widgets.show_cells.call_button.enabled is True

    def test_call_button_resets_on_key_type_change(self, widgets, viewer_config):
        widgets.show_cells.key.value = viewer_config.genes[0]
        assert widgets.show_cells.call_button.enabled is True

        widgets.show_cells.key_type.value = "obs"
        assert widgets.show_cells.call_button.enabled is False

    def test_invalid_typed_key_is_rejected_at_call_time(self, widgets, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            "insitupy.interactive._widgets.show_warning", lambda msg: warnings.append(msg)
        )

        result = widgets.show_cells(key="not_a_real_gene", key_type="genes", recent=None)

        assert result is None
        assert len(warnings) == 1
        assert "not_a_real_gene" in warnings[0]

    def test_valid_typed_key_returns_layer_tuple(self, widgets, viewer_config):
        result = widgets.show_cells(key=viewer_config.genes[0], key_type="genes", recent=None)
        assert result is not None
        _, kwargs, layer_type = result
        assert layer_type == "points"
        assert kwargs["name"] == f"main-{viewer_config.genes[0]}"


class TestUnitsWidget:
    def test_units_key_choices_match_data_units_keys(self, widgets, xdata):
        assert set(widgets.show_units.units_key.choices) == set(xdata.units.keys())

    def test_call_button_disabled_until_gene_obs_or_obsm_set(self, widgets):
        assert widgets.show_units.call_button.enabled is False
        widgets.show_units.gene.value = "v0"
        assert widgets.show_units.call_button.enabled is True

    def test_gene_obs_obsm_are_mutually_exclusive(self, widgets):
        widgets.show_units.gene.value = "v0"
        widgets.show_units.obs.value = "cluster"
        assert widgets.show_units.gene.value == ""
        assert widgets.show_units.call_button.enabled is True

    def test_switching_units_key_resets_choices_values_and_call_button(self, widgets, viewer_config):
        widgets.show_units.gene.value = "v0"
        assert widgets.show_units.call_button.enabled is True

        widgets.show_units.units_key.value = "niche"

        assert viewer_config.units_key == "niche"
        assert widgets.show_units.gene.value == ""
        assert widgets.show_units.obs.value == ""
        assert widgets.show_units.obsm.value == ""
        assert widgets.show_units.call_button.enabled is False

    def test_layer_name_is_scoped_by_units_key(self, widgets):
        result = widgets.show_units(units_key="visium", gene="v0", obs="", obsm="")
        _, kwargs, _ = result
        assert kwargs["name"] == "units-visium-v0"

    def test_switching_units_layer_does_not_clobber_previous_layer(self, widgets, viewer_model):
        r1 = widgets.show_units(units_key="visium", gene="v0", obs="", obsm="")
        viewer_model._add_layer_from_data(*r1)

        widgets.show_units.units_key.value = "niche"
        r2 = widgets.show_units(units_key="niche", gene="v0", obs="", obsm="")
        viewer_model._add_layer_from_data(*r2)

        layer_names = {layer.name for layer in viewer_model.layers}
        assert {"units-visium-v0", "units-niche-v0"} <= layer_names

    def test_reshowing_hidden_layer_makes_it_visible_again(self, widgets, viewer_model):
        # Regression test: re-selecting a units key whose layer already exists but
        # was hidden by the user should un-hide it, not just move it to the top
        # (mirrors the existing behavior of _update_points_layer for cells).
        r1 = widgets.show_units(units_key="visium", gene="v0", obs="", obsm="")
        viewer_model._add_layer_from_data(*r1)

        layer = viewer_model.layers["units-visium-v0"]
        layer.visible = False

        result = widgets.show_units(units_key="visium", gene="v0", obs="", obsm="", add_new_layer=False)

        assert result is None  # updates the existing layer in place, returns nothing
        assert layer.visible is True

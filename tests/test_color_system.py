"""Tests for the layer-aware, write-through color system (``ExperimentColors``).

Covers: write-through to ``.uns``, partial manual dicts, the cells-layer dimension,
``pl.embedding`` reach via ``.uns``, colors.json nesting + legacy migration,
layer-aware ``sync_colors``, and view ``save_colors`` two-level merges.
"""

import contextlib
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import insitupy as isp
from insitupy import WITH_NAPARI
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_xd(seed, categories, col="celltype", n_cells=None, slide_id="s"):
    """Minimal InSituData with a categorical obs column under the 'main' cells layer."""
    cats = list(categories)
    if n_cells is None:
        n_cells = len(cats)
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, 3))
    obs = pd.DataFrame(
        {col: pd.Categorical([cats[i % len(cats)] for i in range(n_cells)])},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(3)])
    table = AnnData(X=X, obs=obs, var=var)
    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id=slide_id, sample_id=slide_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


def _make_xd_two_layers(seed, main_cats, layerb_cats, col="celltype", slide_id="s"):
    """InSituData with two cells layers ('main', 'layerB') sharing an obs column name."""
    xd = InSituData(
        path=None, metadata=None,
        slide_id=slide_id, sample_id=slide_id,
        method_name="test", method_params={},
    )
    for key, cats, is_main in [("main", main_cats, True), ("layerB", layerb_cats, False)]:
        n_cells = len(cats)
        rng = np.random.default_rng(seed + (0 if key == "main" else 100))
        X = rng.random((n_cells, 3))
        obs = pd.DataFrame(
            {col: pd.Categorical(list(cats))},
            index=[f"{key}_c{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"g{j}" for j in range(3)])
        table = AnnData(X=X, obs=obs, var=var)
        celldata = CellData(table=table, boundaries=None)
        xd.cells.add_celldata(cd=celldata, key=key, is_main=is_main)
    return xd


def _make_experiment(n=2, categories=("A", "B", "C"), col="celltype"):
    exp = InSituExperiment()
    for i in range(n):
        exp._data.append(_make_xd(seed=i, categories=categories, col=col, slide_id=f"s{i}"))
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(n)]})
    return exp


@contextlib.contextmanager
def _capture_insitupy_logs(caplog, level):
    """caplog's handler is attached at the root logger, but `insitupy` sets
    `propagate=False` (see insitupy/_logging.py), so records from `insitupy.*`
    loggers never reach it. Attach the handler directly to the source logger.
    """
    logger = logging.getLogger("insitupy.experiment.data")
    caplog.set_level(level, logger="insitupy.experiment.data")
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


def _all_legend_labels(fig):
    labels = set()
    for a in fig.get_axes():
        leg = a.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                labels.add(t.get_text())
    return labels


# ── 1. write-through ─────────────────────────────────────────────────────────


def test_assignment_writes_through_to_uns():
    exp = _make_experiment(n=2, categories=("A", "B", "C"))
    color_dict = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}

    exp.colors["celltype"] = color_dict

    for xd in exp._data:
        cats = xd.cells["main"].table.obs["celltype"].cat.categories
        uns_colors = xd.cells["main"].table.uns["celltype_colors"]
        assert len(uns_colors) == len(cats)
        for cat, hexcolor in zip(cats, uns_colors, strict=False):
            assert hexcolor.lower() == color_dict[cat].lower()
            assert hexcolor.startswith("#")

    assert exp.colors["celltype"] == color_dict


# ── 2. partial manual dict fills gaps ────────────────────────────────────────


def test_partial_manual_dict_fills_gaps():
    exp = _make_experiment(n=1, categories=("A", "B", "C"))
    partial = {"A": "#ff0000"}

    exp.colors["celltype"] = partial

    xd = exp._data[0]
    cats = list(xd.cells["main"].table.obs["celltype"].cat.categories)
    uns_colors = xd.cells["main"].table.uns["celltype_colors"]

    assert len(uns_colors) == len(cats)
    assert uns_colors[cats.index("A")].lower() == "#ff0000"
    for c in uns_colors:
        assert c.startswith("#") and len(c) == 7

    # stored intent keeps only the assigned subset
    assert exp.colors["celltype"] == partial


# ── 3. layer dimension is independent ────────────────────────────────────────


def test_layer_dimension_independent():
    exp = InSituExperiment()
    exp._data.append(_make_xd_two_layers(0, ["A", "B"], ["X", "Y"], slide_id="s0"))
    exp._metadata = pd.DataFrame({"uid": ["s0"]})

    main_colors = {"A": "#ff0000", "B": "#00ff00"}
    layerb_colors = {"X": "#0000ff", "Y": "#ffff00"}

    exp.colors["celltype"] = main_colors
    exp.colors.set("celltype", layerb_colors, cells_layer="layerB")

    assert exp.colors["celltype"] == main_colors
    assert exp.colors.get("celltype", cells_layer="layerB") == layerb_colors

    xd = exp._data[0]
    main_uns = [c.lower() for c in xd.cells["main"].table.uns["celltype_colors"]]
    layerb_uns = [c.lower() for c in xd.cells["layerB"].table.uns["celltype_colors"]]
    assert main_uns == ["#ff0000", "#00ff00"]
    assert layerb_uns == ["#0000ff", "#ffff00"]

    # main is unaffected by the layerB write
    assert exp.colors["celltype"] == main_colors


# ── 4. pl.embedding picks up manual colors via .uns (no signature change) ───


def test_embedding_picks_up_manual_colors():
    exp = _make_experiment(n=1, categories=("A", "B"))
    exp.colors["celltype"] = {"A": "#123456", "B": "#654321"}

    adata = exp.data[0].cells.table
    adata.obsm["X_umap"] = np.random.default_rng(0).random((adata.n_obs, 2))

    plt.close("all")
    fig = isp.pl.embedding(
        adata, keys="celltype", render_mode="matplotlib", show=False, return_fig=True
    )
    try:
        assert fig is not None
        labels = _all_legend_labels(fig)
        assert {"A", "B"}.issubset(labels)
    finally:
        plt.close("all")

    assert [c.lower() for c in adata.uns["celltype_colors"]] == ["#123456", "#654321"]


# ── 5. colors.json nested round-trip ─────────────────────────────────────────


def test_colors_json_roundtrip_nested(tmp_path):
    exp = InSituExperiment()
    exp._data.append(_make_xd_two_layers(0, ["A", "B"], ["X", "Y"], slide_id="s0"))
    exp._metadata = pd.DataFrame({"uid": ["s0"]})
    exp._path = tmp_path

    exp.colors["celltype"] = {"A": "#ff0000", "B": "#00ff00"}
    exp.colors.set("celltype", {"X": "#0000ff", "Y": "#ffff00"}, cells_layer="layerB")
    exp.save_colors(path=tmp_path)

    with open(tmp_path / "colors.json") as f:
        raw = json.load(f)

    normalized = InSituExperiment._normalize_colors_store(raw, "main")
    assert normalized == exp._colors
    assert normalized["main"]["celltype"] == {"A": "#ff0000", "B": "#00ff00"}
    assert normalized["layerB"]["celltype"] == {"X": "#0000ff", "Y": "#ffff00"}

    # .uns stays consistent with what was written at assignment time
    xd = exp._data[0]
    assert [c.lower() for c in xd.cells["main"].table.uns["celltype_colors"]] == ["#ff0000", "#00ff00"]
    assert [c.lower() for c in xd.cells["layerB"].table.uns["celltype_colors"]] == ["#0000ff", "#ffff00"]


# ── 6. legacy flat colors.json migration ─────────────────────────────────────


def test_migrate_legacy_flat_colors_json(tmp_path, caplog):
    legacy = {"celltype": {"A": "#ff0000", "B": "#00ff00"}}
    (tmp_path / "colors.json").write_text(json.dumps(legacy))

    exp = InSituExperiment()
    exp._data.append(_make_xd(seed=0, categories=("A", "B"), slide_id="s0"))
    exp._metadata = pd.DataFrame({"uid": ["s0"]})
    exp._path = tmp_path

    with open(tmp_path / "colors.json") as f:
        raw = json.load(f)

    with _capture_insitupy_logs(caplog, logging.INFO):
        exp._colors = exp._normalize_colors_store(raw, exp._default_color_layer())

    assert exp._colors == {"main": {"celltype": {"A": "#ff0000", "B": "#00ff00"}}}
    assert any("Migrating legacy flat colors.json" in r.message for r in caplog.records)
    assert exp.colors["celltype"] == {"A": "#ff0000", "B": "#00ff00"}


def test_normalize_colors_store_empty_returns_empty():
    assert InSituExperiment._normalize_colors_store({}, "main") == {}


# ── 7. sync_colors is layer-aware ────────────────────────────────────────────


def test_sync_colors_layer_aware_gate(caplog):
    exp = InSituExperiment()
    xd = _make_xd_two_layers(0, ["A", "B"], ["A", "B"], slide_id="s0")
    exp._data.append(xd)
    exp._metadata = pd.DataFrame({"uid": ["s0"]})

    exp.sync_colors("celltype", cells_layer="main")
    assert "celltype" in exp.colors.layer("main")

    # today this would be a no-op; layer-aware storage means it independently populates
    exp.sync_colors("celltype", cells_layer="layerB")
    assert "celltype" in exp.colors.layer("layerB")
    assert exp.colors.layer("main")["celltype"] != {}
    assert exp.colors.layer("layerB")["celltype"] != {}

    with _capture_insitupy_logs(caplog, logging.WARNING):
        exp.sync_colors("celltype", cells_layer="main")  # overwrite=False -> warns, no crash

    assert any("already exists" in r.message for r in caplog.records)


# ── 8. view save_colors two-level merge ──────────────────────────────────────


def test_view_save_colors_two_level_merge(tmp_path):
    exp = InSituExperiment()
    exp._data.append(_make_xd(seed=0, categories=("A", "B"), slide_id="s0"))
    exp._data.append(_make_xd(seed=1, categories=("A", "B"), slide_id="s1"))
    exp._metadata = pd.DataFrame({"uid": ["s0", "s1"]})
    exp._path = tmp_path
    exp._colors = {
        "main": {"celltype": {"A": "#ff0000", "B": "#00ff00"}},
        "layerB": {"other": {"X": "#111111"}},
    }
    exp.save_colors(path=tmp_path)

    view = exp._subset(slice(0, 1), as_view=True)
    view._colors = {"main": {"celltype2": {"C": "#222222"}}}
    view.save_colors()

    with open(tmp_path / "colors.json") as f:
        merged = json.load(f)

    assert merged["main"]["celltype"] == {"A": "#ff0000", "B": "#00ff00"}
    assert merged["main"]["celltype2"] == {"C": "#222222"}
    assert merged["layerB"]["other"] == {"X": "#111111"}


# ── 9-11. napari-guarded ──────────────────────────────────────────────────────


@pytest.mark.skipif(not WITH_NAPARI, reason="napari is required for these tests")
class TestNapariFallbackColormap:
    """Reuses the 260630 report's Part-1 test spec for _resolve_categorical_colormap."""

    def test_fallback_colormap_is_rgb_not_hex(self):
        from insitupy.interactive._widgets import _resolve_categorical_colormap

        color_value = pd.Series(pd.Categorical(["A", "B", "A"]))
        _, colormap = _resolve_categorical_colormap(color_value, uns={}, key="k")
        assert colormap is not None
        assert len(np.asarray(colormap.colors[0])) == 3
        # sanity: labels-renderer expression must not raise
        np.array(colormap.colors[0])

    def test_cells_and_points_agree_categorical(self):
        from matplotlib.colors import to_rgb, to_rgba

        from insitupy._constants import DEFAULT_CATEGORICAL_CMAP
        from insitupy.interactive._widgets import _resolve_categorical_colormap
        from insitupy.utils._colors import create_cmap_mapping

        color_value = pd.Series(
            pd.Categorical(["A", "A", "B"], categories=["B", "A", "C"])
        )
        cv, colormap = _resolve_categorical_colormap(color_value, uns={}, key="k")
        mapping = create_cmap_mapping(cv, colormap)

        n = DEFAULT_CATEGORICAL_CMAP.N
        for i, cat in enumerate(cv.cat.categories):
            expected = to_rgba(to_rgb(DEFAULT_CATEGORICAL_CMAP.colors[i % n]))
            assert mapping[cat] == pytest.approx(expected)
            assert to_rgba(colormap.colors[i]) == pytest.approx(expected)

    def test_object_column_normalized(self):
        from insitupy.interactive._widgets import _resolve_categorical_colormap

        color_value = pd.Series(["B", "A", "B"], dtype=object)
        cv, colormap = _resolve_categorical_colormap(color_value, uns={}, key="k")
        assert hasattr(cv, "cat")
        assert list(cv.cat.categories) == sorted(cv.cat.categories)
        assert colormap is not None

    def test_uns_branch_unchanged(self):
        from insitupy.interactive._widgets import _resolve_categorical_colormap

        color_value = pd.Series(pd.Categorical(["A", "B"]))
        uns = {"k_colors": ["#ff0000", "#00ff00"]}
        cv, colormap = _resolve_categorical_colormap(color_value, uns=uns, key="k")
        assert colormap.colors[0] == pytest.approx((1.0, 0.0, 0.0))
        assert colormap.colors[1] == pytest.approx((0.0, 1.0, 0.0))
        assert hasattr(cv, "cat")

    def test_numeric_returns_none(self):
        from insitupy.interactive._widgets import _resolve_categorical_colormap

        color_value = np.array([1.0, 2.0, 3.0])
        cv, colormap = _resolve_categorical_colormap(color_value, uns={}, key="k")
        assert colormap is None
        assert cv is color_value


# ── Part 2 — exp.show() pre-flight (napari-independent, monkeypatched) ──────


def test_show_preflight_populates_uns(monkeypatch):
    exp = _make_experiment(n=2, categories=("A", "B"))
    calls = []

    def fake_show(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(InSituData, "show", fake_show)

    exp.show(0)

    for xd in exp._data:
        assert "celltype_colors" in xd.cells["main"].table.uns
    assert "celltype" in exp.colors
    assert len(calls) == 1
    assert calls[0] == {"cells_layer": None, "verbose": False}


def test_show_auto_sync_false_skips(monkeypatch):
    exp = _make_experiment(n=2, categories=("A", "B"))

    monkeypatch.setattr(InSituData, "show", lambda self, **kwargs: None)

    exp.show(0, auto_sync_colors=False)

    for xd in exp._data:
        assert "celltype_colors" not in xd.cells["main"].table.uns
    assert len(exp.colors) == 0

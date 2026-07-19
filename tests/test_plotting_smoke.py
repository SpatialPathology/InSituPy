import importlib

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

plots_module = importlib.import_module("insitupy.plotting.plots")
scatter_module = importlib.import_module("insitupy.plotting.scatter")
spatial_module = importlib.import_module("insitupy.plotting.spatial")

matplotlib.use("Agg")


def _make_adata_with_umap(n_obs: int = 10) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "celltype": pd.Categorical(["A", "B"] * (n_obs // 2) + (["A"] if n_obs % 2 else [])),
            "score": np.linspace(0.0, 1.0, n_obs),
        }
    )
    adata = ad.AnnData(X=np.random.rand(n_obs, 3), obs=obs)
    adata.obsm["X_umap"] = np.random.rand(n_obs, 2)
    return adata


def test_embedding_plotly_smoke_with_monkeypatched_backend(monkeypatch):
    adata = _make_adata_with_umap()

    monkeypatch.setattr(scatter_module, "_check_plotly", lambda: True)
    monkeypatch.setattr(
        scatter_module,
        "_plot_plotly",
        lambda *args, **kwargs: {"ok": True, "title": args[5]},
    )

    out = scatter_module.embedding(
        adata=adata,
        basis="X_umap",
        color=["celltype", "score"],
        interactive=True,
        render_mode="plotly",
    )

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(o["ok"] for o in out)


def test_embedding_raises_when_basis_missing():
    adata = ad.AnnData(X=np.random.rand(5, 2), obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))

    with pytest.raises(KeyError, match=r"not found in adata\.obsm"):
        scatter_module.embedding(adata=adata, basis="X_umap", color=None)


def test_embedding_raises_for_one_dimensional_basis():
    adata = ad.AnnData(X=np.random.rand(5, 2), obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))
    adata.obsm["X_umap"] = np.random.rand(5, 1)

    with pytest.raises(ValueError, match="must have at least 2 dimensions"):
        scatter_module.embedding(adata=adata, basis="X_umap", color=None)


def test_embedding_static_works_without_datashader(monkeypatch):
    adata = _make_adata_with_umap()
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)

    # Static mode falls back to matplotlib when datashader is unavailable — must not raise
    scatter_module.embedding(adata=adata, basis="X_umap", color=None, interactive=False)


def test_spatial_smoke_calls_subplot_pipeline(monkeypatch):
    calls = {"setup": 0, "plot": 0, "save": 0}

    class _DummyColorConfig:
        def __init__(self, *args, **kwargs):
            pass

    def _fake_calc_subplot_params(self, keys, n_data, color_config):
        self.n_plots = len(keys) * n_data
        self.n_rows = 1
        self.n_cols = self.n_plots
        self.figsize = (4 * self.n_cols, 4)

    def _fake_setup_subplots(layout_config, verbose=False):
        calls["setup"] += 1
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        return fig, np.array([ax])

    def _fake_plot_to_subplots(*args, **kwargs):
        calls["plot"] += 1

    def _fake_save_and_show_figure(*args, **kwargs):
        calls["save"] += 1

    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", _DummyColorConfig)
    monkeypatch.setattr(spatial_module.LayoutConfig, "calc_subplot_params", _fake_calc_subplot_params)
    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", _fake_plot_to_subplots)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", _fake_save_and_show_figure)

    spatial_module.spatial(data=object(), keys=["gene_a"], show=False)

    assert calls == {"setup": 1, "plot": 1, "save": 1}


def test_dataconfig_filter_mode_defaults_to_none():
    """Regression for a trailing-comma typo (`filter_mode: FilterMode | None = None,`)
    that silently made the dataclass default a 1-tuple `(None,)` instead of `None`.
    """
    data_config = spatial_module.DataConfig()
    assert data_config.filter_mode is None
    assert data_config.filter_tuple is None


def test_spatial_plot_config_and_data_config_not_clobbered(monkeypatch):
    """A caller-provided plot_config/data_config's own field values must survive
    when the caller doesn't also touch the matching top-level kwarg - regression for
    the update_values() unconditional-override bug (fixed via _explicit_overrides()).
    Explicit top-level kwargs must still win when actually passed.
    """
    captured = {}

    def _fake_setup_subplots(layout_config, verbose=False):
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        return fig, np.array([ax])

    def _fake_plot_to_subplots(
        data, keys, cells_layer, fig, axs,
        plot_config, layout_config, data_config, color_config,
    ):
        captured["plot_config"] = plot_config
        captured["data_config"] = data_config

    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", _fake_plot_to_subplots)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    custom_plot = spatial_module.PlotConfig(spot_size=99, alpha=0.3)
    custom_data = spatial_module.DataConfig(region_tuple=("r", "name"))

    spatial_module.spatial(
        data=object(), keys=["gene_a"],
        plot_config=custom_plot, data_config=custom_data, show=False,
    )
    assert captured["plot_config"].spot_size == 99
    assert captured["plot_config"].alpha == 0.3
    assert captured["data_config"].region_tuple == ("r", "name")

    # explicit top-level kwargs still win when actually passed
    spatial_module.spatial(
        data=object(), keys=["gene_a"],
        plot_config=spatial_module.PlotConfig(spot_size=99), spot_size=7, show=False,
    )
    assert captured["plot_config"].spot_size == 7

    plt.close("all")


def test_spatial_figsize_subplot_width_height_passthrough(monkeypatch):
    """subplot_width/subplot_height/figsize must reach LayoutConfig.figsize via the
    real calc_subplot_params(), mirroring the top-level params already on
    pl.embedding(). Also guards against re-introducing the known update_values()
    clobber bug: a caller-provided layout_config's own fields must survive when
    the new top-level shortcuts are left at their None default.
    """
    captured = []

    def _fake_setup_subplots(layout_config, verbose=False):
        captured.append(layout_config)
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        return fig, np.array([ax])

    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    # subplot_width/subplot_height reach LayoutConfig and drive the computed figsize
    spatial_module.spatial(
        data=object(), keys=["gene_a", "gene_b"], max_cols=2,
        subplot_width=10, subplot_height=3, show=False,
    )
    assert captured[0].subplot_width == 10
    assert captured[0].subplot_height == 3
    assert captured[0].figsize == (10 * captured[0].n_cols, 3 * captured[0].n_rows)

    # explicit figsize overrides the per-panel computation entirely
    spatial_module.spatial(
        data=object(), keys=["gene_a"], figsize=(20.0, 15.0), show=False,
    )
    assert captured[1].figsize == (20.0, 15.0)

    # a caller-provided layout_config's own subplot_width/height must survive when
    # the top-level shortcuts aren't touched
    custom_layout = spatial_module.LayoutConfig(subplot_width=12, subplot_height=9)
    spatial_module.spatial(
        data=object(), keys=["gene_a"], layout_config=custom_layout, show=False,
    )
    assert captured[2].subplot_width == 12
    assert captured[2].subplot_height == 9

    plt.close("all")


def test_spatial_figsize_with_subplot_width_warns_and_wins(monkeypatch):
    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", lambda *a, **k: None)

    def _fake_setup_subplots(layout_config, verbose=False):
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        return fig, np.array([ax])

    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    with pytest.warns(UserWarning, match="figsize sets the total figure size"):
        spatial_module.spatial(
            data=object(), keys=["gene_a"],
            figsize=(20.0, 15.0), subplot_width=10, show=False,
        )
    plt.close("all")


def test_spatial_default_valued_kwarg_overrides_caller_config(monkeypatch):
    """A top-level kwarg passed at its own default value must still override a
    caller-provided config - regression for the `_explicit_overrides` heuristic
    that used to conflate "not passed" with "passed == default" and silently
    dropped alpha=1.0 / spot_size=10 / max_cols=4.
    """
    captured = {}

    def _fake_setup_subplots(layout_config, verbose=False):
        captured["layout_config"] = layout_config
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        return fig, np.array([ax])

    def _fake_plot_to_subplots(
        data, keys, cells_layer, fig, axs,
        plot_config, layout_config, data_config, color_config,
    ):
        captured["plot_config"] = plot_config

    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", _fake_plot_to_subplots)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    spatial_module.spatial(
        data=object(), keys=["gene_a"], alpha=1.0, spot_size=10,
        plot_config=spatial_module.PlotConfig(alpha=0.3, spot_size=99),
        show=False,
    )
    assert captured["plot_config"].alpha == 1.0
    assert captured["plot_config"].spot_size == 10

    spatial_module.spatial(
        data=object(), keys=["gene_a"], max_cols=4,
        layout_config=spatial_module.LayoutConfig(max_cols=2),
        show=False,
    )
    assert captured["layout_config"].max_cols == 4

    plt.close("all")


def test_spatial_ndarray_xlim_does_not_raise(monkeypatch):
    """xlim passed as a bare numpy array must not raise - regression for
    `_explicit_overrides`'s old `value != default` comparison, which produced an
    elementwise array and raised "ambiguous truth value" for ndarray inputs.
    """
    captured = {}

    def _fake_setup_subplots(layout_config, verbose=False):
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        return fig, np.array([ax])

    def _fake_plot_to_subplots(
        data, keys, cells_layer, fig, axs,
        plot_config, layout_config, data_config, color_config,
    ):
        captured["plot_config"] = plot_config

    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", _fake_plot_to_subplots)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    spatial_module.spatial(
        data=object(), keys=["gene_a"], xlim=np.array([0, 100]), show=False,
    )
    assert np.array_equal(captured["plot_config"].xlim, np.array([0, 100]))

    plt.close("all")


def test_spatial_figsize_from_caller_layout_config_warns_with_subplot_width(monkeypatch):
    """The figsize/subplot precedence warning must also fire when the effective
    figsize comes from a caller-provided layout_config, not just a top-level
    figsize kwarg - regression for the warning only checking the top-level param.
    """
    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", lambda *a, **k: None)

    def _fake_setup_subplots(layout_config, verbose=False):
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        return fig, np.array([ax])

    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    with pytest.warns(UserWarning, match="figsize sets the total figure size"):
        spatial_module.spatial(
            data=object(), keys=["gene_a"],
            layout_config=spatial_module.LayoutConfig(figsize=(30, 20)),
            subplot_width=10, show=False,
        )
    plt.close("all")


def test_spatial_reused_layout_config_not_mutated_across_calls(monkeypatch):
    """Reusing one ``layout_config`` across spatial() calls must not leak a stale
    figsize. Regression for oversized markers / wrong figure size when plotting a
    subset experiment (e.g. ``exp[:3]``) after the full ``exp`` with the same
    ``layout_config`` object.
    """
    captured_figsizes = []

    class _FakeExp:
        def __init__(self, n_data):
            self._n_data = n_data

        def __len__(self):
            return self._n_data

        def sync_colors(self, **kwargs):
            pass

    class _FakeColorConfig:
        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, key):
            # categorical key -> non-None color_dict drives the multidata layout
            return {"color_dict": {"A": "#ffffff", "B": "#000000"}}

        def keys(self):
            return ["k"]

    def _fake_setup_subplots(layout_config, verbose=False):
        # record the figsize that calc_subplot_params produced for this call
        captured_figsizes.append(layout_config.figsize)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        return fig, np.array([ax])

    monkeypatch.setattr(spatial_module, "_is_experiment", lambda data: True)
    monkeypatch.setattr(spatial_module, "_ColorConfigMultiPlot", _FakeColorConfig)
    monkeypatch.setattr(spatial_module, "_setup_subplots", _fake_setup_subplots)
    monkeypatch.setattr(spatial_module, "_plot_to_subplots", lambda *a, **k: None)
    monkeypatch.setattr(spatial_module, "save_and_show_figure", lambda *a, **k: None)

    layout_config = spatial_module.LayoutConfig()

    # First call on the "full" experiment, then the same object on a "subset".
    spatial_module.spatial(_FakeExp(6), keys=["k"], layout_config=layout_config, show=False)
    spatial_module.spatial(_FakeExp(3), keys=["k"], layout_config=layout_config, show=False)

    # The subset plot must get a smaller figure than the full one, not a stale copy.
    assert captured_figsizes[0] != captured_figsizes[1]
    assert captured_figsizes[1][1] < captured_figsizes[0][1]

    # The caller's config must be left untouched (no derived state leaked into it).
    assert layout_config.figsize is None
    plt.close("all")


def test_cellular_composition_smoke_with_stubbed_composition(monkeypatch):
    compositions = pd.DataFrame(
        {("region_1", "sample_1"): [60.0, 40.0]},
        index=pd.Index(["A", "B"], name="cell_type"),
    )
    compositions.columns = pd.MultiIndex.from_tuples(
        compositions.columns,
        names=["region_key", "uid"],
    )

    class _DummyCellLayer:
        def __init__(self):
            self.table = ad.AnnData(
                X=np.zeros((4, 1), dtype=float),
                obs=pd.DataFrame(
                    {
                        "cell_type": pd.Categorical(["A", "B", "A", "B"]),
                    }
                ),
            )
            self.table.uns["cell_type_colors"] = ["#1f77b4", "#ff7f0e"]

    class _DummyData:
        def __init__(self):
            self.cells = object()

    monkeypatch.setattr(plots_module, "calc_cellular_composition", lambda *args, **kwargs: compositions)
    monkeypatch.setattr(plots_module, "_is_experiment", lambda data: False)
    monkeypatch.setattr(plots_module, "_get_cell_layer", lambda *args, **kwargs: _DummyCellLayer())
    monkeypatch.setattr(plots_module, "save_and_show_figure", lambda *args, **kwargs: None)

    data = _DummyData()
    out = plots_module.cellular_composition(
        data=data,
        cell_type_col="cell_type",
        plot_type="barh",
        return_data=True,
    )

    assert isinstance(out, pd.DataFrame)
    assert out.shape == compositions.shape


def test_cellular_composition_requires_modality_when_geom_key_set():
    with pytest.raises(ValueError, match="modality must not be None"):
        plots_module.cellular_composition(
            data=object(),
            cell_type_col="cell_type",
            geom_key="region",
            modality=None,
        )


def test_embedding_layer_static_smoke(monkeypatch):
    adata = _make_adata_with_umap()
    adata.var.index = pd.Index([f"g{i}" for i in range(adata.n_vars)])
    adata.layers["counts"] = adata.X * 2
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)

    scatter_module.embedding(
        adata=adata, basis="X_umap", keys="g0", layer="counts",
        interactive=False, show=False,
    )

    with pytest.raises(KeyError, match=r"Layer 'missing' not found"):
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="g0", layer="missing",
            interactive=False, show=False,
        )


# ── palette parameter tests ───────────────────────────────────────────────────

class TestEmbeddingPalette:
    """Tests for the palette parameter in pl.embedding() / pl.umap()."""

    def _make_adata(self):
        return _make_adata_with_umap()

    def _make_adata_with_nan(self, n_obs=10, n_nan=3):
        rng = np.random.default_rng(42)
        labels = np.array(["A", "B"] * (n_obs // 2))
        labels[:n_nan] = None
        obs = pd.DataFrame(
            {"celltype": pd.Categorical(labels, categories=["A", "B"])},
            index=[f"c{i}" for i in range(n_obs)],
        )
        adata = ad.AnnData(X=rng.random((n_obs, 3)), obs=obs)
        adata.obsm["X_umap"] = rng.random((n_obs, 2))
        return adata

    def test_colormap_name_writes_uns(self, monkeypatch):
        from matplotlib.colors import is_color_like
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette="viridis", render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert "celltype_colors" in adata.uns
        colors = adata.uns["celltype_colors"]
        assert len(colors) == 2
        assert all(is_color_like(c) for c in colors)

    def test_sequence_assigned_in_category_order(self, monkeypatch):
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette=["#ff0000", "#00ff00"], render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert adata.uns["celltype_colors"] == ["#ff0000", "#00ff00"]

    def test_palette_overrides_existing_uns(self, monkeypatch):
        adata = self._make_adata()
        adata.uns["celltype_colors"] = ["#111111", "#222222"]
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette=["#ff0000", "#00ff00"], render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert adata.uns["celltype_colors"] == ["#ff0000", "#00ff00"]

    def test_consistency_across_calls(self, monkeypatch):
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette=["#ff0000", "#00ff00"], render_mode="matplotlib", show=False,
        )
        plt.close("all")
        saved = list(adata.uns["celltype_colors"])
        # Second call without palette — should reuse written colors
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette=None, render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert list(adata.uns["celltype_colors"]) == saved

    def test_umap_forwards_palette(self, monkeypatch):
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        scatter_module.umap(
            adata, keys="celltype",
            palette=["#ff0000", "#00ff00"], render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert adata.uns["celltype_colors"] == ["#ff0000", "#00ff00"]

    def test_cycler_palette(self, monkeypatch):
        from cycler import cycler
        from matplotlib.colors import is_color_like
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        pal = cycler(color=["#ff0000", "#00ff00"])
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette=pal, render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert "celltype_colors" in adata.uns
        assert all(is_color_like(c) for c in adata.uns["celltype_colors"])

    def test_shorter_palette_warns_and_recycles(self, monkeypatch):
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        with pytest.warns(UserWarning, match="fewer colors"):
            scatter_module.embedding(
                adata=adata, basis="X_umap", keys="celltype",
                palette=["#ff0000"], render_mode="matplotlib", show=False,
            )
        plt.close("all")
        assert all(c == "#ff0000" for c in adata.uns["celltype_colors"])

    def test_invalid_colormap_raises(self, monkeypatch):
        adata = self._make_adata()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        with pytest.raises(ValueError, match="not a valid matplotlib colormap"):
            scatter_module.embedding(
                adata=adata, basis="X_umap", keys="celltype",
                palette="definitely_not_a_cmap", render_mode="matplotlib", show=False,
            )
        plt.close("all")

    def test_palette_with_nan_color(self, monkeypatch):
        adata = self._make_adata_with_nan()
        monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="celltype",
            palette=["#ff0000", "#00ff00"], nan_color="lightgray",
            render_mode="matplotlib", show=False,
        )
        plt.close("all")
        assert "celltype_colors" in adata.uns
        # Only real categories written back — no "NaN" entry
        assert len(adata.uns["celltype_colors"]) == 2

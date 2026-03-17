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

    with pytest.raises(KeyError, match="not found in adata\.obsm"):
        scatter_module.embedding(adata=adata, basis="X_umap", color=None)


def test_embedding_raises_for_one_dimensional_basis():
    adata = ad.AnnData(X=np.random.rand(5, 2), obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))
    adata.obsm["X_umap"] = np.random.rand(5, 1)

    with pytest.raises(ValueError, match="must have at least 2 dimensions"):
        scatter_module.embedding(adata=adata, basis="X_umap", color=None)


def test_embedding_static_requires_datashader(monkeypatch):
    adata = _make_adata_with_umap()
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)

    with pytest.raises(ImportError, match="datashader and matplotlib are required"):
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

import importlib
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

scatter_module = importlib.import_module("insitupy.plotting.scatter")


def _make_adata(n_obs: int = 20, n_keys: int = 4) -> ad.AnnData:
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({f"k{i}": rng.random(n_obs) for i in range(n_keys)})
    adata = ad.AnnData(X=rng.random((n_obs, 3)), obs=obs)
    adata.obsm["X_umap"] = rng.random((n_obs, 2))
    return adata


# --- test cases ---

def test_default_size_unchanged(monkeypatch):
    """Regression guard: default output must be (ncols*5+2, nrows*5)."""
    adata = _make_adata(n_keys=4)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    fig = scatter_module.embedding(
        adata=adata, basis="X_umap", keys=[f"k{i}" for i in range(4)],
        ncols=3, render_mode="matplotlib", show=False, return_fig=True,
    )
    w, h = fig.get_size_inches()
    # ncols_plot=3, nrows=2 → (17, 10)
    assert w == pytest.approx(17.0)
    assert h == pytest.approx(10.0)
    plt.close("all")


def test_per_panel_scaling_figsize(monkeypatch):
    """subplot_width/subplot_height drive total figure size."""
    adata = _make_adata(n_keys=4)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    fig = scatter_module.embedding(
        adata=adata, basis="X_umap", keys=[f"k{i}" for i in range(4)],
        ncols=2, subplot_width=4, subplot_height=3,
        render_mode="matplotlib", show=False, return_fig=True,
    )
    w, h = fig.get_size_inches()
    # ncols_plot=2, nrows=2 → (2*4+2, 2*3) = (10, 6)
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(6.0)
    ax = fig.axes[0]
    if hasattr(ax, "get_box_aspect"):
        assert ax.get_box_aspect() == pytest.approx(3 / 4)
    plt.close("all")


def test_single_panel_per_panel(monkeypatch):
    """Single key with panel dims: (1*w+2, 1*h)."""
    adata = _make_adata(n_keys=1)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    fig = scatter_module.embedding(
        adata=adata, basis="X_umap", keys="k0",
        subplot_width=4, subplot_height=3,
        render_mode="matplotlib", show=False, return_fig=True,
    )
    w, h = fig.get_size_inches()
    assert w == pytest.approx(6.0)
    assert h == pytest.approx(3.0)
    plt.close("all")


def test_figsize_still_means_whole_figure(monkeypatch):
    """figsize override is passed through to plt.subplots unchanged."""
    adata = _make_adata(n_keys=4)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    fig = scatter_module.embedding(
        adata=adata, basis="X_umap", keys=[f"k{i}" for i in range(4)],
        figsize=(8, 6), render_mode="matplotlib", show=False, return_fig=True,
    )
    w, h = fig.get_size_inches()
    assert w == pytest.approx(8.0)
    assert h == pytest.approx(6.0)
    plt.close("all")


def test_conflict_warning_figsize_plus_panel(monkeypatch):
    """Passing both figsize and subplot_width warns; figsize wins."""
    adata = _make_adata(n_keys=2)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    with pytest.warns(UserWarning, match="subplot_width/subplot_height are ignored"):
        fig = scatter_module.embedding(
            adata=adata, basis="X_umap", keys=["k0", "k1"],
            figsize=(8, 6), subplot_width=4,
            render_mode="matplotlib", show=False, return_fig=True,
        )
    w, h = fig.get_size_inches()
    assert w == pytest.approx(8.0)
    assert h == pytest.approx(6.0)
    plt.close("all")


def test_wspace_no_spurious_warning_under_panel_sizing(monkeypatch):
    """With subplot_width set, wspace must NOT trigger the spacing warning."""
    adata = _make_adata(n_keys=2)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys=["k0", "k1"],
            subplot_width=4, wspace=0.3,
            render_mode="matplotlib", show=False,
        )
    spacing_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "no effect" in str(w.message)
    ]
    assert spacing_warnings == [], "Unexpected spacing warning when panel sizing is active"
    plt.close("all")


def test_wspace_still_warns_in_default_mode(monkeypatch):
    """Without figsize or panel dims, wspace still warns."""
    adata = _make_adata(n_keys=2)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    with pytest.warns(UserWarning, match="no effect"):
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys=["k0", "k1"],
            wspace=0.3, render_mode="matplotlib", show=False,
        )
    plt.close("all")


def test_umap_wrapper_forwards_panel_params(monkeypatch):
    """umap() forwards subplot_width/subplot_height through **kwargs."""
    adata = _make_adata(n_keys=4)
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    fig = scatter_module.umap(
        adata=adata, keys=[f"k{i}" for i in range(4)],
        ncols=2, subplot_width=4, subplot_height=3,
        render_mode="matplotlib", show=False, return_fig=True,
    )
    w, h = fig.get_size_inches()
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(6.0)
    plt.close("all")

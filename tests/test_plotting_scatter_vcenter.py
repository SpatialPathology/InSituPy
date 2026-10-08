import importlib

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize, TwoSlopeNorm

scatter_module = importlib.import_module("insitupy.plotting.scatter")


def _make_adata(n_obs: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(42)
    obs = pd.DataFrame({"score": np.linspace(-1.0, 3.0, n_obs)})
    adata = ad.AnnData(X=rng.random((n_obs, 3)), obs=obs)
    adata.obsm["X_umap"] = rng.random((n_obs, 2))
    return adata


# --- _build_norm unit tests ---

def test_build_norm_without_vcenter_is_plain_normalize():
    norm = scatter_module._build_norm(0.0, 5.0, None)
    assert isinstance(norm, Normalize) and not isinstance(norm, TwoSlopeNorm)


def test_build_norm_centers_value_at_midpoint():
    norm = scatter_module._build_norm(-1.0, 3.0, 0.0)
    assert isinstance(norm, TwoSlopeNorm)
    assert abs(float(norm(0.0)) - 0.5) < 1e-9
    assert float(norm(-1.0)) == pytest.approx(0.0)
    assert float(norm(3.0)) == pytest.approx(1.0)


@pytest.mark.parametrize("vmin,vmax", [
    (0, 5),    # center below vmin
    (-5, 0),   # center above vmax
    (2, 5),    # center well below range
    (-5, -2),  # center well above range
    (3, 3),    # constant positive
    (0, 0),    # constant zero
])
def test_build_norm_never_raises_when_center_outside_range(vmin, vmax):
    norm = scatter_module._build_norm(float(vmin), float(vmax), 0.0)
    assert abs(float(norm(0.0)) - 0.5) < 1e-9


# --- static render smoke tests ---

def test_embedding_vcenter_static_matplotlib_fallback(monkeypatch):
    adata = _make_adata()
    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    # 'score' spans [-1, 3]; vcenter=0 is inside the range — normal TwoSlopeNorm
    scatter_module.embedding(
        adata=adata, basis="X_umap", keys="score",
        cmap="coolwarm", vcenter=0.0, render_mode="matplotlib", show=False,
    )
    plt.close("all")


def test_embedding_vcenter_below_range_does_not_crash(monkeypatch):
    """vcenter=0 with all-positive data triggers the nudge path — must not raise."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"expr": rng.uniform(0.5, 3.0, 20)})
    adata = ad.AnnData(X=rng.random((20, 3)), obs=obs)
    adata.obsm["X_umap"] = rng.random((20, 2))

    monkeypatch.setattr(scatter_module, "_check_datashader", lambda: False)
    scatter_module.embedding(
        adata=adata, basis="X_umap", keys="expr",
        cmap="coolwarm", vcenter=0.0, render_mode="matplotlib", show=False,
    )
    plt.close("all")


def test_embedding_vcenter_warns_when_interactive(monkeypatch):
    adata = _make_adata()
    monkeypatch.setattr(scatter_module, "_check_plotly", lambda: True)
    monkeypatch.setattr(
        scatter_module,
        "_plot_plotly",
        lambda *args, **kwargs: {"ok": True},
    )
    with pytest.warns(UserWarning, match="vcenter is only supported for static"):
        scatter_module.embedding(
            adata=adata, basis="X_umap", keys="score",
            cmap="coolwarm", vcenter=0.0,
            interactive=True, render_mode="plotly",
        )

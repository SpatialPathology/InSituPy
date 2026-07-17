"""Tests for insitupy.utils._adata - AnnData element selection helpers.

Direct regression coverage for the anndata 0.13 `layers[None]` bug: 0.13 stores `X`
internally as `adata.layers[None]`, so code that enumerates and deletes `.layers` keys
without filtering `None` silently destroys `X`. See
.log/reports/260717/anndata-013-layers-fix/report-anndata-013-layers-fix.md.
"""

import anndata as ad
import numpy as np
import pandas as pd

from insitupy.utils._adata import _select_anndata_elements


def _make_adata(n_cells=5, n_genes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 30, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


class TestSelectAnndataElementsPreservesX:
    def test_layer_keys_none_keeps_x(self):
        """'drop all layers' must not drop X - the actual bug."""
        adata = _make_adata()
        X_before = np.asarray(adata.X).copy()

        result = _select_anndata_elements(adata, layer_keys=None)

        assert result.X is not None
        np.testing.assert_array_equal(np.asarray(result.X), X_before)
        assert "counts" not in result.layers

    def test_layer_keys_list_keeps_x(self):
        """'keep only these layers' must not drop X either."""
        adata = _make_adata()
        X_before = np.asarray(adata.X).copy()

        result = _select_anndata_elements(adata, layer_keys=["counts"])

        assert result.X is not None
        np.testing.assert_array_equal(np.asarray(result.X), X_before)
        assert "counts" in result.layers

    def test_layer_keys_all_keeps_x_and_layers(self):
        adata = _make_adata()
        X_before = np.asarray(adata.X).copy()

        result = _select_anndata_elements(adata, layer_keys="all")

        assert result.X is not None
        np.testing.assert_array_equal(np.asarray(result.X), X_before)
        assert "counts" in result.layers

    def test_layer_keys_subset_drops_unlisted_real_layers(self):
        """Real (non-X) layers are still dropped/kept as documented - the fix must not
        make the function stop removing layers it's supposed to remove."""
        adata = _make_adata()
        adata.layers["norm"] = np.asarray(adata.X) * 2

        result = _select_anndata_elements(adata, layer_keys=["counts"])

        assert "counts" in result.layers
        assert "norm" not in result.layers
        assert result.X is not None

    def test_inplace_true_preserves_x(self):
        adata = _make_adata()
        X_before = np.asarray(adata.X).copy()

        out = _select_anndata_elements(adata, layer_keys=None, inplace=True)

        assert out is None
        assert adata.X is not None
        np.testing.assert_array_equal(np.asarray(adata.X), X_before)

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from insitupy.plotting.scatter import _get_color_values


def test_get_color_values_reads_from_layer():
    X = np.array([[1.0], [2.0], [3.0]])
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["GeneA"]))
    adata.layers["counts"] = X * 10

    vals_x, ctype = _get_color_values(adata, "GeneA")
    vals_layer, _ = _get_color_values(adata, "GeneA", layer="counts")

    assert ctype == "continuous"
    np.testing.assert_array_equal(vals_x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(vals_layer, [10.0, 20.0, 30.0])


def test_get_color_values_invalid_layer_raises():
    adata = ad.AnnData(X=np.zeros((2, 1)), var=pd.DataFrame(index=["GeneA"]))
    with pytest.raises(KeyError, match=r"Layer 'missing' not found"):
        _get_color_values(adata, "GeneA", layer="missing")


def test_get_color_values_obs_ignores_layer():
    adata = ad.AnnData(
        X=np.zeros((3, 1), dtype=float),
        obs=pd.DataFrame({"score": [0.1, 0.2, 0.3]}),
        var=pd.DataFrame(index=["GeneA"]),
    )
    adata.layers["counts"] = np.ones((3, 1))
    vals, ctype = _get_color_values(adata, "score", layer="counts")
    assert ctype == "continuous"
    np.testing.assert_array_equal(vals, [0.1, 0.2, 0.3])

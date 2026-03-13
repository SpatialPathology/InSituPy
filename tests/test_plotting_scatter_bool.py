import anndata as ad
import numpy as np
import pandas as pd

from insitupy.plotting.scatter import _get_color_values, _get_vmin_vmax


def test_get_color_values_treats_bool_obs_as_categorical():
    adata = ad.AnnData(
        X=np.zeros((3, 1), dtype=float),
        obs=pd.DataFrame({"low_confidence": [True, False, True]})
    )

    values, color_type = _get_color_values(adata, "low_confidence")

    assert color_type == "categorical"
    assert isinstance(values.dtype, pd.CategoricalDtype)


def test_get_vmin_vmax_handles_bool_arrays_with_percentile():
    values = np.array([True, False, True, False], dtype=bool)

    vmin, vmax = _get_vmin_vmax(values, vmax_percentile=95)

    assert vmin == 0.0
    assert 0.0 <= vmax <= 1.0

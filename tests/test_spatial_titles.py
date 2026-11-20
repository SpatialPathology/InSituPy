"""Regression tests: pl.spatial panel titles when the sample name is not a string.

With one key over several samples the panel title is ``sample_name + "\\n" + key``, which
raised a TypeError when ``sample_id`` was None or the ``name_column`` held numbers.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import insitupy as isp
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment
from insitupy.plotting.spatial import DataConfig


def _make_xd(seed, sample_id, n_cells=10):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {"celltype": pd.Categorical(["A", "B"] * (n_cells // 2))},
        index=[f"c{i}" for i in range(n_cells)],
    )
    table = AnnData(
        X=rng.random((n_cells, 3)), obs=obs,
        var=pd.DataFrame(index=[f"g{j}" for j in range(3)]),
    )
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100.0
    xd = InSituData(
        path=None, metadata=None,
        slide_id=f"slide{seed}", sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    return xd


def _titles(fig):
    return [a.get_title() for a in fig.get_axes() if a.get_title()]


@pytest.mark.parametrize("case,expected", [
    ("sample_id_none", ["sample 0\ncelltype", "s1\ncelltype"]),
    ("numeric_name_column", ["1\ncelltype", "2\ncelltype"]),
])
def test_multi_sample_title_with_non_string_name(case, expected):
    exp = InSituExperiment()
    exp._data.append(_make_xd(0, None if case == "sample_id_none" else "s0"))
    exp._data.append(_make_xd(1, "s1"))
    exp._metadata = pd.DataFrame({"uid": ["u0", "u1"], "num": [1, 2]})
    data_config = DataConfig(name_column="num") if case == "numeric_name_column" else None

    plt.close("all")
    kwargs = {"data_config": data_config} if data_config is not None else {}
    isp.pl.spatial(exp, keys="celltype", show=False, **kwargs)
    titles = _titles(plt.gcf())
    plt.close("all")

    assert titles == expected

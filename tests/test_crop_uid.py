"""Identity-model tests for InSituData.crop / copy uid handling.

A non-inplace crop() yields a new, detached dataset that belongs to no
InSituExperiment, so it must clear the experiment-slot uid (self._uid) to None -
InSituExperiment.add() then mints a fresh uid instead of colliding with the
parent. An inplace crop() and copy() must both keep the uid: an inplace crop may
still be an experiment member, and a copy is the same dataset (see decisions.md
"Identity model").
"""

import numpy as np
import pandas as pd
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData


def _make_insitudata(slide_id="slide1", sample_id="s1"):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 20, size=(5, 3)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(5)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(3)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((5, 2)) * 100
    cd = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id=slide_id, sample_id=sample_id,
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=cd, key="main", is_main=True)
    return xd


def test_noninplace_crop_clears_uid():
    # a detached crop belongs to no experiment -> uid must be reset to None,
    # and the parent's uid must be left untouched.
    xd = _make_insitudata()
    xd._uid = "parent-uid"

    cropped = xd.crop(xlim=(0, 100), ylim=(0, 100), inplace=False)

    assert cropped.uid is None
    assert xd.uid == "parent-uid"  # original unchanged


def test_inplace_crop_keeps_uid():
    # an inplace crop may still be an experiment member -> uid preserved.
    xd = _make_insitudata()
    xd._uid = "member-uid"

    result = xd.crop(xlim=(0, 100), ylim=(0, 100), inplace=True)

    assert result is None  # inplace crop returns None
    assert xd.uid == "member-uid"


def test_copy_keeps_uid():
    # regression guard: copy() is a same-dataset duplicate and must NOT reset the
    # uid (only crop's non-inplace path does).
    xd = _make_insitudata()
    xd._uid = "same-uid"

    xd_copy = xd.copy()

    assert xd_copy.uid == "same-uid"
    assert xd.uid == "same-uid"

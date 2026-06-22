"""Tests for InSituExperiment.save() with pathless / outside-path datasets.

Covers acceptance criteria from the 260622 fix-two-bugs-before-0.12.0b4 report:
- save() auto-assigns a free data-NNN slot for pathless datasets
- two pathless datasets get distinct slots (no collision)
- dataset with path outside the experiment dir raises ValueError
- existing on-disk datasets are updated (not recreated) alongside new ones
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_xd(seed, n_cells=5, n_genes=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    cd = CellData(table=table, boundaries=None)
    xd = InSituData(path=None, metadata=None,
                    slide_id="s", sample_id="s", method_name="t", method_params={})
    xd.cells.add_celldata(cd=cd, key="main", is_main=True)
    return xd


def _make_and_save_exp(tmp_path, n=2):
    """Create an experiment, save it via saveas(), and return (exp, dest)."""
    exp = InSituExperiment()
    for i in range(n):
        exp._data.append(_make_xd(i))
    exp._metadata = pd.DataFrame({"uid": [f"s{i}" for i in range(n)]})
    dest = tmp_path / "exp"
    exp.saveas(dest)
    return exp, dest


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_save_assigns_path_for_pathless_dataset(tmp_path):
    """save() must auto-assign the next free data-NNN slot for a pathless dataset."""
    exp, dest = _make_and_save_exp(tmp_path, n=2)

    xd_new = _make_xd(seed=99)
    exp.add(xd_new)  # path is None

    exp.save()

    assert xd_new.path is not None, "path must be set after save()"
    assert (dest / "data-002").exists(), "data-002 directory must exist"
    assert xd_new.path == (dest / "data-002").resolve()
    assert len(exp._metadata) == 3, "metadata must have 3 rows"


def test_save_raises_for_path_outside_experiment_dir(tmp_path):
    """save() must raise ValueError when a dataset's path is outside the experiment dir."""
    exp, dest = _make_and_save_exp(tmp_path, n=1)

    xd_outside = _make_xd(seed=10)
    xd_outside._path = (tmp_path / "elsewhere" / "data-000").resolve()

    exp._data.append(xd_outside)
    exp._metadata = pd.concat(
        [exp._metadata, pd.DataFrame([{"uid": "outside-uid"}])],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="outside-uid"):
        exp.save()


def test_save_two_nopath_datasets_get_unique_slots(tmp_path):
    """Two pathless datasets added before save() must each get a distinct free slot."""
    exp, dest = _make_and_save_exp(tmp_path, n=1)

    xd_a = _make_xd(seed=20)
    xd_b = _make_xd(seed=21)
    exp.add(xd_a)
    exp.add(xd_b)

    exp.save()

    assert xd_a.path is not None
    assert xd_b.path is not None
    assert xd_a.path != xd_b.path, "both datasets must have distinct paths"
    assert xd_a.path.exists()
    assert xd_b.path.exists()
    # Both must be inside the experiment directory
    assert xd_a.path.parent == dest.resolve()
    assert xd_b.path.parent == dest.resolve()


def test_save_mixed_existing_and_new(tmp_path):
    """Existing on-disk dataset is updated; new pathless dataset is written fresh."""
    exp, dest = _make_and_save_exp(tmp_path, n=1)

    xd_existing = exp._data[0]
    existing_path = xd_existing.path

    xd_new = _make_xd(seed=30)
    exp.add(xd_new)

    exp.save()

    # Existing dataset path unchanged
    assert xd_existing.path == existing_path
    assert existing_path.exists()

    # New dataset was written to the next free slot
    assert xd_new.path is not None
    assert xd_new.path.exists()
    assert xd_new.path != existing_path

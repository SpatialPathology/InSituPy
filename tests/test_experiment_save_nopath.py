"""Tests for InSituExperiment.save() with pathless / outside-path datasets.

Covers acceptance criteria from the 260622 fix-two-bugs-before-0.12.0b4 report:
- save() auto-assigns a free data-NNN slot for pathless datasets
- two pathless datasets get distinct slots (no collision)
- dataset with path outside the experiment dir raises ValueError
- existing on-disk datasets are updated (not recreated) alongside new ones

Also covers retry-safety (260625 report):
- Trigger A: path-less dataset is NOT mutated before a later validation error (retry succeeds)
- Trigger B: a mid-write failure is rolled back cleanly (no litter, retry succeeds)
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pathlib import Path

from insitupy._constants import ISPY_METADATA_FILE
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


def test_save_retry_trigger_a(tmp_path):
    """Trigger A: path-less dataset at lower index than an external-path dataset.

    After the first save() raises ValueError (external path), the path-less dataset
    must still have path=None (not mutated to a slot), so a retry after removing the
    offending dataset succeeds and writes it to a clean slot.
    """
    exp, dest = _make_and_save_exp(tmp_path, n=1)

    xd_new = _make_xd(seed=40)       # path is None, will be index 1
    exp.add(xd_new)

    xd_ext = _make_xd(seed=41)
    xd_ext._path = (tmp_path / "elsewhere" / "data-000").resolve()
    exp._data.append(xd_ext)
    exp._metadata = pd.concat(
        [exp._metadata, pd.DataFrame([{"uid": "ext-uid"}])],
        ignore_index=True,
    )

    # First save must raise — external path detected
    with pytest.raises(ValueError, match="ext-uid"):
        exp.save()

    # The path-less dataset must NOT have been mutated (validate-before-assign)
    assert xd_new.path is None, "path-less dataset must not be mutated before validation raises"
    assert not (dest / "data-001").exists(), "no slot dir must be created before validation raises"

    # Remove the offending external dataset and retry
    exp._data.pop()
    exp._metadata = exp._metadata.iloc[:2].reset_index(drop=True)

    exp.save()

    assert xd_new.path == (dest / "data-001").resolve()
    assert xd_new.path.exists()
    assert (xd_new.path / ISPY_METADATA_FILE).exists()


def test_save_retry_trigger_b(tmp_path):
    """Trigger B: saveas() fails mid-write; rollback removes partial dir and resets _path.

    After the first save() raises RuntimeError (simulated mid-write failure), the
    partial slot dir must be gone and the path-less dataset must have path=None.
    A retry (without the fault) succeeds and writes it to a clean slot.
    """
    exp, dest = _make_and_save_exp(tmp_path, n=1)

    xd_new = _make_xd(seed=50)
    exp.add(xd_new)

    def _boom(path, *a, **k):
        Path(path).mkdir(parents=True, exist_ok=True)  # leave partial, metadata-less dir
        raise RuntimeError("simulated mid-write failure")

    xd_new.saveas = _boom

    with pytest.raises(RuntimeError, match="call save\\(\\) again"):
        exp.save()

    # Rollback must have cleaned up
    assert xd_new.path is None, "path must be reset to None after rollback"
    assert not (dest / "data-001").exists(), "partial slot dir must be removed by rollback"

    # Remove the fault and retry
    del xd_new.saveas

    exp.save()

    assert xd_new.path == (dest / "data-001").resolve()
    assert xd_new.path.exists()
    assert (xd_new.path / ISPY_METADATA_FILE).exists()

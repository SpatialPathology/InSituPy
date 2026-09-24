"""Tests for the destructive-operation guards of InSituExperiment.

Covers the ROAD-8 batch (report 260924 "track-b-atomic-commit"):
- ``remove(delete_from_disk=True)`` refuses datasets outside the experiment folder (B5)
- ``concat(mode="move")`` refuses a non-empty destination (T2) and unexpected content in a
  source experiment root unless ``force=True`` (B5), before anything is moved; a source
  ``tables/`` folder is derived data that is not carried over (T18)
- staging/backup leftovers of an interrupted save are not mistaken for datasets
- experiment-level ``zip_output`` is rejected (its result could not be read back)
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_xd(i):
    rng = np.random.default_rng(i)
    table = AnnData(
        X=rng.integers(0, 10, size=(5, 3)).astype(float),
        obs=pd.DataFrame(index=pd.Index([f"s{i}c{j}" for j in range(5)])),
        var=pd.DataFrame(index=pd.Index([f"g{j}" for j in range(3)])),
    )
    table.obsm["spatial"] = rng.random((5, 2)) * 100
    xd = InSituData(path=None, metadata=None,
                    slide_id=f"slide{i}", sample_id=f"sample{i}", method_name="t", method_params={})
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    return xd


def _saved_experiment(tmp_path, name="exp", n=2, offset=0):
    """Save an n-dataset experiment to ``tmp_path/name`` and read it back."""
    exp = InSituExperiment()
    for i in range(n):
        exp.add(_make_xd(offset + i))
    exp.saveas(tmp_path / name)
    return InSituExperiment.read(tmp_path / name)


# ── Interrupted-save leftovers ────────────────────────────────────────────────


def test_read_skips_staging_leftovers(tmp_path):
    exp = _saved_experiment(tmp_path)
    root = tmp_path / "exp"
    (root / "data-000.__ispy_tmp__").mkdir()  # partial dataset: no .ispy inside
    (root / "data-000.__ispy_tmp__" / "half-written.bin").write_text("x")
    (root / "data-001.__ispy_bak__").mkdir()

    with pytest.warns(UserWarning, match="left over from an interrupted save"):
        reread = InSituExperiment.read(root)

    assert len(reread) == len(exp) == 2

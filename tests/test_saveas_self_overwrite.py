"""Tests for the self-overwrite guard in InSituData.saveas (roadmap B1).

saveas() deletes its target up front (shutil.rmtree) and then writes; because images
and transcripts are read lazily from the backing directory, writing onto (or into, or
over an ancestor of) the object's own project would read from a just-deleted directory
and lose data silently. The guard refuses any such overlapping target - regardless of
the `overwrite` flag - and points the user at .save() for in-place updates.
"""

import os
import re

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_xd(seed=0, n_cells=5, n_genes=3):
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


def _saved_xd(dest):
    """Return an InSituData written to `dest`, so it is project-backed (_path == dest)."""
    xd = _make_xd()
    xd.saveas(dest)
    return xd


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_saveas_onto_own_path_overwrite_true_refuses(tmp_path):
    """The core B1 regression: saveas onto the own project with overwrite=True must be refused
    and must not delete the on-disk data."""
    dest = tmp_path / "proj"
    xd = _saved_xd(dest)

    sentinel = dest / "sentinel.txt"
    sentinel.write_text("keep")

    with pytest.raises(ValueError, match=re.escape("use .save()")):
        xd.saveas(dest, overwrite=True)

    assert dest.exists(), "project directory must survive a refused self-overwrite"
    assert sentinel.exists(), "no rmtree may have run on the backing directory"


def test_saveas_onto_own_path_overwrite_false_refuses(tmp_path):
    """overwrite=False onto the own project must raise the guard's ValueError, not the
    FileExistsError that invites overwrite=True (which would then destroy the data)."""
    dest = tmp_path / "proj"
    xd = _saved_xd(dest)

    with pytest.raises(ValueError, match=re.escape("use .save()")):
        xd.saveas(dest, overwrite=False)

    assert dest.exists()


def test_saveas_onto_parent_refuses(tmp_path):
    """Ancestor case: rmtree of a parent of the backing dir would delete the project too."""
    dest = tmp_path / "nested" / "proj"
    xd = _saved_xd(dest)

    with pytest.raises(ValueError):
        xd.saveas(dest.parent, overwrite=True)

    assert dest.exists()


def test_saveas_onto_subdir_of_project_refuses(tmp_path):
    """Descendant case: rmtree of a subdir of the backing dir would delete part of the store."""
    dest = tmp_path / "proj"
    xd = _saved_xd(dest)

    with pytest.raises(ValueError):
        xd.saveas(dest / "images", overwrite=True)

    assert dest.exists()


def test_saveas_to_different_dir_still_works(tmp_path):
    """A genuine new, non-overlapping location must not trip the guard and must repath."""
    dest = tmp_path / "proj"
    xd = _saved_xd(dest)

    other = tmp_path / "copy"
    xd.saveas(other)

    assert other.exists()
    assert xd.path.resolve() == other.resolve()


def test_saveas_relative_alias_of_own_path_refuses(tmp_path, monkeypatch):
    """A relative path that resolves to the backing dir must be refused, proving the guard
    normalizes both sides via .resolve()."""
    dest = tmp_path / "proj"
    xd = _saved_xd(dest)

    # Change cwd to the project's parent so a bare relative name aliases the backing dir.
    monkeypatch.chdir(tmp_path)
    alias = os.path.join(".", "proj")

    with pytest.raises(ValueError):
        xd.saveas(alias, overwrite=True)

    assert dest.exists()

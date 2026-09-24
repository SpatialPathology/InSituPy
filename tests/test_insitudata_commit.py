"""Tests for the atomic-commit behaviour of InSituData saves.

Covers the ROAD-8 batch (report 260924 "track-b-atomic-commit"):
- the ``.ispy`` pointer, not the wall-clock directory name, decides which save is current (B2)
- a failed ``save()`` leaves the previously committed state readable (B2/B6)
- units are never deleted before their replacement is complete (B6)
- ``save(path=<another copy>)`` is refused (B3)
- ``saveas(overwrite=True)`` never deletes the target before the new data is complete, and a
  lazily backed object can be saved over its own source (B1 follow-on)
- ``saveas(zip_output=True)`` checks and writes ``<path>.zip`` only (B7)

Every test runs against a real on-disk project.
"""

import json
import os
import shutil
import warnings

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point

from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.data import InSituData
from insitupy.containers.cell_data import CellData
from insitupy.containers.spatial_units_data import SpatialUnitsData

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_units(names, unit_type="unit", seed=0):
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame(
        {"name": names, "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))]}
    )
    table = AnnData(
        X=rng.random((len(names), 2)),
        obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
        var=pd.DataFrame(index=["v0", "v1"]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def _make_xd(seed=0, n_cells=5, n_genes=3, transcripts=False, units=False):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    xd = InSituData(path=None, metadata=None,
                    slide_id="s", sample_id="s", method_name="t", method_params={})
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=None), key="main", is_main=True)
    if transcripts:
        df = pd.DataFrame({
            "x": rng.random(40) * 100,
            "y": rng.random(40) * 100,
            "feature_name": [f"g{i % n_genes}" for i in range(40)],
        })
        xd._transcripts = dd.from_pandas(df, npartitions=2)
    if units:
        xd.add_units(_make_units(["u1", "u2"], unit_type="visium", seed=1))
        xd.add_units(_make_units(["n1", "n2"], unit_type="niche", seed=2), key="niche")
    return xd


def _saved(tmp_path, name="p", **kwargs):
    """Save a fresh dataset to ``tmp_path/name`` and return ``(project_dir, read_back)``."""
    p = tmp_path / name
    _make_xd(**kwargs).saveas(p, verbose=False)
    return p, InSituData.read(p)


def _ispy(p):
    return json.loads((p / ISPY_METADATA_FILE).read_text())


def _dirs(p):
    return sorted(d.name for d in p.iterdir())


def _tree(p):
    """Relative file listing of *p*, for byte-identity style comparisons."""
    return sorted(str(f.relative_to(p)) for f in p.rglob("*"))


# ── B2: the .ispy pointer decides ─────────────────────────────────────────────


def test_clock_skew_save_keeps_fresh_state(tmp_path, monkeypatch):
    """A save whose directory name sorts OLDER than the previous one must still win."""
    p, xd = _saved(tmp_path)
    xd.cells.table.obs["flag"] = "fresh"

    older = "200101-000000000000-deadbeef"
    monkeypatch.setattr("insitupy.containers.io._generate_time_based_uid", lambda: older)
    xd.save(verbose=False)

    # reload after save picked the committed dir, not the newest-by-name one
    assert "flag" in xd.cells.table.obs.columns
    # only the committed dir remains; the previous (newer-named) one was pruned
    assert _dirs(p / "cells") == [older]
    assert _ispy(p)["data"]["cells"] == f"cells/{older}"
    assert all((p / h).is_dir() for h in _ispy(p)["history"]["cells"]), \
        "history must not list deleted saves"
    assert "flag" in InSituData.read(p).cells.table.obs.columns


def test_loader_falls_back_without_pointer(tmp_path):
    """Stores without an .ispy pointer (older projects) load the newest-by-name save."""
    p, xd = _saved(tmp_path)
    xd.cells.table.obs["flag"] = "second"
    xd.save(verbose=False, keep_history=True)
    assert len(list((p / "cells").iterdir())) == 2

    meta = _ispy(p)
    del meta["data"]["cells"]
    (p / ISPY_METADATA_FILE).write_text(json.dumps(meta))

    assert "flag" in InSituData.read(p).cells.table.obs.columns


def test_pointer_outside_project_ignored(tmp_path):
    """A pointer leaving <project>/cells/ is rejected with a warning, never followed."""
    p, xd = _saved(tmp_path)
    original = next((p / "cells").iterdir())
    xd.cells.table.obs["flag"] = "in-project"
    xd.save(verbose=False, keep_history=True)

    # a real, valid cells store outside the project (without the flag column)
    shutil.copytree(original, tmp_path / "evil")
    meta = _ispy(p)
    meta["data"]["cells"] = "../evil"
    (p / ISPY_METADATA_FILE).write_text(json.dumps(meta))

    with pytest.warns(UserWarning, match="pointer for 'cells'"):
        loaded = InSituData.read(p)
    assert "flag" in loaded.cells.table.obs.columns, "must load the in-project newest save"


def test_non_insitupy_dir_left_alone(tmp_path):
    """Folders InSituPy did not create are ignored with a warning and never deleted."""
    p, xd = _saved(tmp_path)
    backup = p / "cells" / "my_backup"
    backup.mkdir()
    (backup / "keep.txt").write_text("mine")

    xd.cells.table.obs["flag"] = "x"
    with pytest.warns(UserWarning, match="my_backup"):
        xd.save(verbose=False)

    assert (backup / "keep.txt").read_text() == "mine"
    assert "flag" in InSituData.read(p).cells.table.obs.columns


def test_crash_before_commit_keeps_old_state(tmp_path, monkeypatch):
    """A failure after the new cells are written but before .ispy is committed changes nothing."""
    p, xd = _saved(tmp_path)
    committed = _ispy(p)["data"]["cells"]
    dirs_before = _dirs(p / "cells")
    meta_before = json.loads(json.dumps(xd.metadata))

    xd.cells.table.obs["flag"] = "never committed"

    def _boom(*args, **kwargs):
        raise OSError("simulated crash before the .ispy write")

    monkeypatch.setattr("insitupy._core.data.write_dict_to_json", _boom)
    with pytest.raises(OSError, match="simulated crash"):
        xd.save(verbose=False)
    monkeypatch.undo()

    assert _ispy(p)["data"]["cells"] == committed
    assert _dirs(p / "cells") == dirs_before, "the uncommitted cells dir must be removed"
    assert "flag" not in InSituData.read(p).cells.table.obs.columns
    assert json.loads(json.dumps(xd.metadata)) == meta_before


# ── B6: units are never deleted before their replacement is complete ─────────


def test_units_failure_mid_write_keeps_old_units(tmp_path, monkeypatch):
    p, xd = _saved(tmp_path, units=True)
    old_names = {k: list(xd.units[k].table.obs_names) for k in xd.units.keys()}
    xd.units["visium"].table.obs["new_col"] = "changed"

    calls = []
    original_save = SpatialUnitsData.save

    def _fail_on_second_layer(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("simulated disk failure on the second layer")
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(SpatialUnitsData, "save", _fail_on_second_layer)
    with pytest.raises(OSError, match="second layer"):
        xd.save(verbose=False)
    monkeypatch.undo()

    assert not (p / "units.__ispy_tmp__").exists()
    on_disk = InSituData.read(p)
    assert {k: list(on_disk.units[k].table.obs_names) for k in on_disk.units.keys()} == old_names
    assert "new_col" not in on_disk.units["visium"].table.obs.columns

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
import zipfile

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point

from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.data import InSituData
from insitupy.containers.boundaries_data import BoundariesData
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


# label mask of the boundaries fixture: values 1..5 = seg_mask_value of cells c0..c4
_MASK = np.arange(6, dtype=np.uint32).reshape(2, 3)


def _make_xd(seed=0, n_cells=5, n_genes=3, transcripts=False, units=False, boundaries=False):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"g{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    xd = InSituData(path=None, metadata=None,
                    slide_id="s", sample_id="s", method_name="t", method_params={})
    bounds = None
    if boundaries:
        bounds = BoundariesData(cell_names=list(obs.index), seg_mask_value=list(range(1, n_cells + 1)))
        bounds.add_boundaries(cell_boundaries=_MASK, nuclei_boundaries=None, pixel_size=1)
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=bounds), key="main", is_main=True)
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
    """Folders InSituPy did not create are never deleted, and a normal save does not warn."""
    p, xd = _saved(tmp_path)
    backup = p / "cells" / "my_backup"
    backup.mkdir()
    (backup / "keep.txt").write_text("mine")

    xd.cells.table.obs["flag"] = "x"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xd.save(verbose=False)  # the pointer resolves, and pruning skips the folder silently
    assert not [w for w in caught if "my_backup" in str(w.message)]

    assert (backup / "keep.txt").read_text() == "mine"
    assert "flag" in InSituData.read(p).cells.table.obs.columns

    # Only the newest-by-name fallback (no pointer) reports the ignored folder.
    meta = _ispy(p)
    del meta["data"]["cells"]
    (p / ISPY_METADATA_FILE).write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="my_backup"):
        InSituData.read(p)


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


def test_save_geometries_crash_before_commit_keeps_old_state(tmp_path, monkeypatch):
    """The partial saver rolls back like save(): new annotations dir removed, metadata restored."""
    p, xd = _saved(tmp_path)
    poly = gpd.GeoDataFrame({
        "id": ["a_0"], "name": ["Tumor"], "color": ["#ff0000"],
        "geometry": [Point(5, 5).buffer(5)],
    })
    xd._annotations.add_data(data=poly, key="pathology", scale_factor=1.0)
    ispy_before = _ispy(p)
    meta_before = json.loads(json.dumps(xd.metadata))

    def _boom(*args, **kwargs):
        raise OSError("simulated crash before the .ispy write")

    monkeypatch.setattr("insitupy._core.data.write_dict_to_json", _boom)
    with pytest.raises(OSError, match="simulated crash"):
        xd.save_geometries(verbose=False)
    monkeypatch.undo()

    assert _ispy(p) == ispy_before
    annot_root = p / "annotations"
    assert not annot_root.exists() or not any(annot_root.iterdir()), \
        "the uncommitted annotations dir must be removed"
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


# ── B3: save(path=<another copy>) is refused ──────────────────────────────────


def test_save_to_other_copy_refused(tmp_path):
    xd = _make_xd()
    a, b = tmp_path / "a", tmp_path / "b"
    xd.saveas(a, verbose=False)
    shutil.copytree(a, b)
    listing_a, listing_b = _tree(a), _tree(b)

    xd.cells.table.obs["flag"] = "x"
    with pytest.raises(ValueError, match="another copy of the same dataset"):
        xd.save(path=b, verbose=False)
    # an unlinked object that shares the uid must not write into the copy either
    with pytest.raises(ValueError, match="another copy of the same dataset"):
        xd.copy().save(path=a, verbose=False)

    assert _tree(a) == listing_a
    assert _tree(b) == listing_b


def test_save_to_linked_path_and_new_path_still_work(tmp_path):
    xd = _make_xd()
    a = tmp_path / "a"
    xd.saveas(a, verbose=False)

    xd.save(path=a, verbose=False)  # explicit path that is the linked project
    xd.save(path=tmp_path / "new", verbose=False)  # non-existent path routes to saveas

    assert (tmp_path / "new" / ISPY_METADATA_FILE).exists()
    assert xd.path == (tmp_path / "new").resolve()


def test_partial_saves_to_other_path_refused_when_linked(tmp_path):
    p, xd = _saved(tmp_path)
    other = tmp_path / "other"

    with pytest.raises(ValueError, match="only writes into the project"):
        xd.save_cells(path=other)
    with pytest.raises(ValueError, match="only writes into the project"):
        xd.save_geometries(path=other)
    assert not other.exists()

    xd.save_cells(path=p)  # the linked project itself is fine


# ── saveas(overwrite=True): stage then swap ──────────────────────────────────


def test_saveas_overwrite_failure_keeps_target(tmp_path, monkeypatch):
    target = tmp_path / "other"
    _make_xd(seed=0).saveas(target, verbose=False)
    listing, ispy_before = _tree(target), (target / ISPY_METADATA_FILE).read_bytes()

    xd = _make_xd(seed=1)
    meta_before = json.loads(json.dumps(xd.metadata))

    def _boom(*args, **kwargs):
        raise OSError("simulated failure while writing cells")

    monkeypatch.setattr("insitupy._core.data._save_cells", _boom)
    with pytest.raises(OSError, match="simulated failure"):
        xd.saveas(target, overwrite=True, verbose=False)
    monkeypatch.undo()

    assert _tree(target) == listing
    assert (target / ISPY_METADATA_FILE).read_bytes() == ispy_before
    assert not (tmp_path / "other.__ispy_tmp__").exists()
    assert json.loads(json.dumps(xd.metadata)) == meta_before
    assert xd.path is None


def test_saveas_overwrite_replaces_and_cleans(tmp_path):
    target = tmp_path / "other"
    _make_xd(seed=0, n_cells=5).saveas(target, verbose=False)
    (target / "stale.txt").write_text("old")

    xd = _make_xd(seed=1, n_cells=8)
    xd.saveas(target, overwrite=True, verbose=False)

    assert not (target / "stale.txt").exists()
    assert InSituData.read(target).cells.table.n_obs == 8
    assert _dirs(tmp_path) == ["other"], "no staging or backup siblings may remain"


def test_copy_saveas_over_own_source_is_safe(tmp_path):
    """A copy that still reads lazily from the slot it overwrites must not lose data."""
    p, xd = _saved(tmp_path, transcripts=True)
    original = xd.transcripts.compute().reset_index(drop=True)

    c = xd.copy()
    assert isinstance(c.transcripts, dd.DataFrame)
    c.saveas(p, overwrite=True, verbose=False)

    on_disk = InSituData.read(p).transcripts.compute().reset_index(drop=True)
    pd.testing.assert_frame_equal(on_disk, original)
    # the copy was re-opened from the new files, so it still computes after the swap
    pd.testing.assert_frame_equal(c.transcripts.compute().reset_index(drop=True), original)
    assert c.path == p.resolve()
    assert _dirs(tmp_path) == ["p"]


def test_overwrite_reopens_lazily_read_boundaries(tmp_path):
    """Every save writes boundaries to a NEW cells/<uid> directory, so masks that a copy still
    reads lazily from the overwritten project point into a deleted directory after the swap.
    zarr reads a missing chunk as zeros instead of raising, so without re-opening them the
    segmentation masks would silently become all-zero."""
    p, xd = _saved(tmp_path, boundaries=True)
    before = sorted(d.name for d in (p / "cells").iterdir())

    c = xd.copy()  # keeps reading the masks lazily from the cells/<uid> directory in p
    c.saveas(p, overwrite=True, verbose=False)

    assert sorted(d.name for d in (p / "cells").iterdir()) != before, "expected a new cells/<uid>"
    np.testing.assert_array_equal(np.asarray(c.cells.boundaries["cells"][0]), _MASK)
    np.testing.assert_array_equal(
        np.asarray(InSituData.read(p).cells.boundaries["cells"][0]), _MASK
    )


def test_atomic_replace_dir_recovers_orphaned_backup(tmp_path, monkeypatch):
    """A lone .__ispy_bak__ is the only surviving copy of an interrupted swap."""
    from insitupy._io.files import atomic_replace_dir

    dest = tmp_path / "dest"
    backup = tmp_path / "dest.__ispy_bak__"
    staging = tmp_path / "dest.__ispy_tmp__"
    backup.mkdir()
    (backup / "old.txt").write_text("survivor")
    staging.mkdir()
    (staging / "new.txt").write_text("new")

    original_rename = os.rename

    def _fail_on_staging(src, dst):
        if os.fspath(src) == os.fspath(staging):
            raise OSError("simulated rename failure")
        return original_rename(src, dst)

    monkeypatch.setattr(os, "rename", _fail_on_staging)
    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_replace_dir(staging, dest)
    monkeypatch.undo()

    assert (dest / "old.txt").read_text() == "survivor", "the orphaned backup must not be deleted"
    assert not backup.exists()
    assert not staging.exists()


# ── B7: zip_output checks and writes <path>.zip only ─────────────────────────


def test_zip_output_checks_real_zip_path(tmp_path):
    xd = _make_xd()
    path = tmp_path / "out"
    zip_path = tmp_path / "out.zip"
    zip_path.write_text("not a real archive")
    path.mkdir()
    (path / "sentinel.txt").write_text("mine")

    with pytest.raises(FileExistsError, match=r"out\.zip"):
        xd.saveas(path, zip_output=True, verbose=False)
    assert zip_path.read_text() == "not a real archive"
    assert (path / "sentinel.txt").exists(), "the directory at `path` is never touched"

    meta_before = json.loads(json.dumps(xd.metadata))
    xd.saveas(path, zip_output=True, overwrite=True, verbose=False)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert ISPY_METADATA_FILE in names and any(n.startswith("cells/") for n in names)
    assert (path / "sentinel.txt").exists(), "the directory at `path` is never deleted"
    assert xd.path is None, "an in-memory object is not re-pointed at the archive"
    assert json.loads(json.dumps(xd.metadata)) == meta_before
    assert sorted(d.name for d in tmp_path.iterdir()) == ["out", "out.zip"]


def test_zip_export_keeps_linked_project(tmp_path):
    p, xd = _saved(tmp_path)
    xd.saveas(tmp_path / "export", zip_output=True, verbose=False)

    assert (tmp_path / "export.zip").is_file()
    assert not (tmp_path / "export").exists()
    assert xd.path == p, "the object stays backed by the project it was loaded from"

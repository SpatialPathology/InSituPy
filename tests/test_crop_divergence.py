"""After ``crop(inplace=True)`` an object still points at the project it was read from,
but its data are the cropped ones. Loading from, saving into or unloading against that
project would mix cropped and uncropped data, so these calls refuse; ``saveas()`` writes
the crop to a new path or replaces the original (U-H24a).

Every test runs against a real on-disk project.
"""

import json
import logging

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import box

from insitupy import InSituExperiment
from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.data import InSituData
from insitupy._exceptions import ProjectDivergedError
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData

# cells c0..c4 at x = 10, 30, 50, 70, 90; the crop window keeps c0..c2
_CELL_X = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
_CROP = dict(xlim=(0, 60), ylim=(0, 100))
_N_ALL, _N_CROPPED = 5, 3
_MASK = np.arange(6, dtype=np.uint32).reshape(2, 3)


def _make_xd(sample_id="s", far_region=False):
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(index=pd.Index([f"c{i}" for i in range(_N_ALL)]))
    var = pd.DataFrame(index=pd.Index(["g0", "g1", "g2"]))
    table = AnnData(X=rng.integers(0, 10, size=(_N_ALL, 3)).astype(float), obs=obs, var=var)
    table.obsm["spatial"] = np.column_stack([_CELL_X, np.full(_N_ALL, 50.0)])
    bounds = BoundariesData(cell_names=list(obs.index), seg_mask_value=list(range(1, _N_ALL + 1)))
    bounds.add_boundaries(cell_boundaries=_MASK, nuclei_boundaries=None, pixel_size=1)
    xd = InSituData(path=None, metadata=None, slide_id="slide", sample_id=sample_id,
                    method_name="t", method_params={})
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=bounds), key="main", is_main=True)
    tx = pd.DataFrame({
        "x_location": np.linspace(0, 100, 40),
        "y_location": np.full(40, 50.0),
        "feature_name": [f"g{i % 3}" for i in range(40)],
    })
    xd._transcripts = dd.from_pandas(tx, npartitions=2)
    if far_region:
        # a region outside the crop window: the crop empties the regions modality
        gdf = gpd.GeoDataFrame({"id": ["far_0"], "name": ["far"],
                                "geometry": [box(80, 80, 95, 95)], "color": ["#00ff00"]})
        xd._regions.add_data(data=gdf, key="zones", scale_factor=1.0)
    return xd


def _cropped(tmp_path, name="p"):
    """Save, read back and crop in place; return ``(project_dir, cropped_object)``."""
    p = tmp_path / name
    _make_xd().saveas(p, verbose=False)
    xd = InSituData.read(p)
    xd.crop(**_CROP, inplace=True)
    assert xd.cells.table.n_obs == _N_CROPPED
    return p, xd


def _ispy_uids(p):
    return json.loads((p / ISPY_METADATA_FILE).read_text())["uids"]


def _n_cells_on_disk(p):
    return InSituData.read(p).cells.table.n_obs


# ── refusals ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("loader", ["load_cells", "load_transcripts", "load_all"])
def test_load_after_inplace_crop_refuses(tmp_path, loader):
    """Loading would put the uncropped project data into the cropped object."""
    _, xd = _cropped(tmp_path)
    with pytest.raises(ProjectDivergedError, match="cropped in place"):
        getattr(xd, loader)()
    assert xd.cells.table.n_obs == _N_CROPPED


@pytest.mark.parametrize("saver", ["save_cells", "save_geometries"])
def test_partial_save_after_inplace_crop_leaves_parent_intact(tmp_path, saver):
    """Partial savers wrote the crop into the parent project without any uid check."""
    p, xd = _cropped(tmp_path)
    uids_before = _ispy_uids(p)
    with pytest.raises(ProjectDivergedError):
        getattr(xd, saver)()
    assert _n_cells_on_disk(p) == _N_ALL
    assert _ispy_uids(p) == uids_before


def test_save_after_inplace_crop_explains(tmp_path):
    p, xd = _cropped(tmp_path)
    with pytest.raises(ProjectDivergedError, match="saveas") as err:
        xd.save(verbose=False)
    assert isinstance(err.value, RuntimeError)  # save() raised RuntimeError before
    assert _n_cells_on_disk(p) == _N_ALL


def test_unload_after_inplace_crop_refuses(tmp_path):
    """The cropped data could not be loaded back once unloaded."""
    _, xd = _cropped(tmp_path)
    with pytest.raises(ProjectDivergedError):
        xd.unload("cells", verbose=False)
    assert xd.cells.table.n_obs == _N_CROPPED


def test_reload_after_inplace_crop_is_a_noop(tmp_path, caplog):
    _, xd = _cropped(tmp_path)
    # insitupy loggers do not propagate to the root logger caplog listens on
    logger = logging.getLogger("insitupy._core.data")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="insitupy._core.data"):
            xd.reload()
    finally:
        logger.removeHandler(caplog.handler)
    assert "Not reloading" in caplog.text
    assert xd.cells.table.n_obs == _N_CROPPED


# ── ways out ──────────────────────────────────────────────────────────────────


def test_saveas_new_path_ends_divergence(tmp_path):
    """After saveas(new) the object is linked to the crop, so save/load work again.
    Two chained in-place crops before it."""
    _, xd = _cropped(tmp_path)
    xd.crop(xlim=(0, 40), ylim=(0, 100), inplace=True)  # second crop: keeps c0, c1
    new = tmp_path / "crop"
    xd.saveas(new, verbose=False)

    xd.save(verbose=False)
    xd.load_cells()
    assert xd.cells.table.n_obs == 2
    assert _n_cells_on_disk(new) == 2
    assert _n_cells_on_disk(tmp_path / "p") == _N_ALL


def test_saveas_own_path_replaces_original_with_crop(tmp_path):
    """D1: the crop may replace its own project. Lazy data must be re-opened from the new
    files: zarr reads chunks of a deleted directory as zeros instead of raising."""
    p, xd = _cropped(tmp_path)
    mask_after_crop = np.asarray(xd.cells.boundaries["cells"][0]).copy()
    tx_after_crop = xd.transcripts.compute().reset_index(drop=True)
    assert mask_after_crop.any()

    xd.saveas(p, overwrite=True, verbose=False)

    assert _n_cells_on_disk(p) == _N_CROPPED
    np.testing.assert_array_equal(np.asarray(xd.cells.boundaries["cells"][0]), mask_after_crop)
    pd.testing.assert_frame_equal(xd.transcripts.compute().reset_index(drop=True), tx_after_crop)
    xd.load_cells()  # no longer diverged
    assert xd.cells.table.n_obs == _N_CROPPED


def test_saveas_own_path_refuses_to_delete_never_loaded_modality(tmp_path):
    """Replacing the project would delete transcripts that were never loaded (and can no
    longer be loaded and cropped): refuse, leave the project intact."""
    p = tmp_path / "p"
    _make_xd().saveas(p, verbose=False)
    xd = InSituData.read(p, load_all=False)
    xd.load_cells()
    xd.crop(**_CROP, inplace=True)

    with pytest.raises(ValueError, match="transcripts"):
        xd.saveas(p, overwrite=True, verbose=False)
    assert (p / "transcripts").is_dir()
    assert _n_cells_on_disk(p) == _N_ALL


def test_saveas_own_path_allows_modality_emptied_by_crop(tmp_path):
    """A modality that was loaded but lies outside the crop window is not 'never loaded'."""
    p = tmp_path / "p"
    _make_xd(far_region=True).saveas(p, verbose=False)
    xd = InSituData.read(p)
    assert "regions" in xd.get_loaded_modalities()
    xd.crop(**_CROP, inplace=True)
    assert "regions" not in xd.get_loaded_modalities()

    xd.saveas(p, overwrite=True, verbose=False)
    assert _n_cells_on_disk(p) == _N_CROPPED


def test_saveas_own_path_without_overwrite_points_to_overwrite(tmp_path):
    p, xd = _cropped(tmp_path)
    with pytest.raises(ValueError, match="overwrite=True"):
        xd.saveas(p, verbose=False)
    assert _n_cells_on_disk(p) == _N_ALL


# ── unaffected cases ──────────────────────────────────────────────────────────


def test_legacy_store_without_uids_still_loads(tmp_path):
    """Stores written before uids existed carry none; nothing can be told, nothing refuses."""
    p = tmp_path / "legacy"
    _make_xd().saveas(p, verbose=False)
    meta = json.loads((p / ISPY_METADATA_FILE).read_text())
    meta.pop("uids")
    (p / ISPY_METADATA_FILE).write_text(json.dumps(meta))

    xd = InSituData.read(p)
    xd.load_cells()
    assert xd.cells.table.n_obs == _N_ALL


# ── experiment ────────────────────────────────────────────────────────────────


def test_experiment_save_points_to_replace(tmp_path):
    exp = InSituExperiment()
    exp.add(_make_xd(sample_id="s1"))
    exp.add(_make_xd(sample_id="s2"))
    exp.saveas(tmp_path / "exp", verbose=False)

    exp = InSituExperiment.read(tmp_path / "exp")
    exp.data[0].load_cells()
    exp.data[0].crop(**_CROP, inplace=True)

    with pytest.raises(RuntimeError, match=r"replace\(i, exp\.data\[i\]\)"):
        exp.save()

    exp.replace(0, exp.data[0], confirm=False)
    exp.save()
    reread = InSituExperiment.read(tmp_path / "exp")
    reread.data[0].load_cells()
    assert reread.data[0].cells.table.n_obs == _N_CROPPED

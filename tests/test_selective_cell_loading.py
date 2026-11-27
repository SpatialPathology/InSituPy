"""Selective cell loading: load_cells(keys=..., boundaries=...) and the save guard.

Data loss is the risk: a partly loaded cells modality written by save()/saveas() would drop the
unloaded layers/boundaries on disk. These tests check that every writer refuses a partial object
and that nothing is written or deleted when it does.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy._exceptions import StaleNucleusMapWarning
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

from tests.test_boundaries_data_save import _write_legacy_nucleus_map_to_store

CELL_NAMES = ["c1", "c2", "c3"]


def _make_celldata(seed):
    rng = np.random.default_rng(seed)
    table = AnnData(
        X=rng.integers(1, 10, size=(3, 2)).astype(float),
        obs=pd.DataFrame(index=pd.Index(CELL_NAMES)),
        var=pd.DataFrame(index=pd.Index(["g0", "g1"])),
    )
    table.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    boundaries = BoundariesData(cell_names=CELL_NAMES, seg_mask_value=[1, 2, 3])
    boundaries.add_boundaries(
        cell_boundaries=np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32),
        nuclei_boundaries=None,
        pixel_size=1,
    )
    return CellData(table=table, boundaries=boundaries)


def _make_xd(seed):
    """In-memory dataset with two cell layers: 'main' and 'alt'."""
    xd = InSituData(path=None, metadata=None, slide_id="s", sample_id="s",
                    method_name="t", method_params={})
    xd.cells.add_celldata(cd=_make_celldata(seed), key="main", is_main=True)
    xd.cells.add_celldata(cd=_make_celldata(seed + 100), key="alt")
    return xd


def _snapshot(root):
    """Every file below *root* with size and mtime, to detect any write or delete."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _stale(record):
    return [w for w in record if issubclass(w.category, StaleNucleusMapWarning)]


@pytest.fixture
def saved_xd_dir(tmp_path):
    """A saved dataset whose main layer carries a stale (legacy, inconsistent) nucleus map."""
    dest = tmp_path / "xd"
    _make_xd(0).saveas(dest)
    (store,) = list(dest.glob("cells/*/main/boundaries.zarr"))
    _write_legacy_nucleus_map_to_store(store, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
    return dest


@pytest.fixture
def saved_exp_dir(tmp_path):
    exp = InSituExperiment()
    for i in range(3):
        exp.add(_make_xd(i), metadata={"label": f"G{i}"})
    dest = tmp_path / "exp"
    exp.saveas(dest)
    for ddir in ["data-000", "data-001"]:
        (store,) = list((dest / ddir).glob("cells/*/main/boundaries.zarr"))
        _write_legacy_nucleus_map_to_store(store, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
    return dest


class TestLoadCellsSelective:
    def test_keys_and_no_boundaries_skip_nucleus_map_entirely(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            xd.load_cells(keys=["main"], boundaries=False)

        assert _stale(record) == []
        assert list(xd.cells.keys()) == ["main"]
        assert xd.cells["main"].boundaries is None
        assert xd.cells.table.n_obs == 3
        assert xd.cells.is_partial

    def test_boundaries_true_still_warns_for_stale_map(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            xd.load_cells(keys="main")
        assert len(_stale(record)) == 1
        assert list(xd.cells.keys()) == ["main"]
        assert xd.cells["main"].boundaries is not None

    def test_unknown_key_raises_keyerror_listing_available(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        with pytest.raises(KeyError, match="nope") as exc:
            xd.load_cells(keys=["main", "nope"])
        assert "alt" in str(exc.value) and "main" in str(exc.value)
        # the failed call left the previously loaded cells untouched
        assert not xd.cells.is_partial
        assert set(xd.cells.keys()) == {"main", "alt"}

    def test_keys_without_main_layer_raises_naming_main(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        with pytest.raises(ValueError, match="main layer 'main'"):
            xd.load_cells(keys=["alt"])
        assert not xd.cells.is_partial
        assert set(xd.cells.keys()) == {"main", "alt"}

    def test_full_load_replaces_partial_and_clears_flag(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"], boundaries=False)
        assert xd.cells.is_partial
        xd.load_cells()
        assert not xd.cells.is_partial
        assert set(xd.cells.keys()) == {"main", "alt"}
        assert xd.cells["main"].boundaries is not None

    def test_load_all_forwards_cell_arguments(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_all(cell_keys=["main"], cell_boundaries=False)
        assert list(xd.cells.keys()) == ["main"]
        assert xd.cells["main"].boundaries is None

    def test_flag_survives_copy_and_crop(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"], boundaries=False)
        assert xd.copy().cells.is_partial

        cropped = xd.crop(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), verbose=False)
        assert cropped.cells.is_partial
        with pytest.raises(ValueError, match="loaded partially"):
            cropped.saveas(saved_xd_dir.parent / "cropped_copy")


class TestSaveGuardDataset:
    def test_save_raises_and_writes_nothing(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"], boundaries=False)
        before = _snapshot(saved_xd_dir)

        with pytest.raises(ValueError, match=r"load_cells\(\) without arguments"):
            xd.save()

        assert _snapshot(saved_xd_dir) == before

    def test_save_cells_raises_and_writes_nothing(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"])
        before = _snapshot(saved_xd_dir)

        with pytest.raises(ValueError, match="loaded partially"):
            xd.save_cells()

        assert _snapshot(saved_xd_dir) == before

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_saveas_raises_and_writes_nothing(self, saved_xd_dir, tmp_path, overwrite):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"], boundaries=False)
        target = tmp_path / "copy"

        with pytest.raises(ValueError, match="loaded partially"):
            xd.saveas(target, overwrite=overwrite)
        assert not target.exists()
        assert not list(tmp_path.glob("copy*"))  # no staging dir or zip left over

        # an existing target is not deleted either
        existing = tmp_path / "existing"
        _make_xd(5).saveas(existing)
        before = _snapshot(existing)
        with pytest.raises(ValueError, match="loaded partially"):
            xd.saveas(existing, overwrite=True)
        assert _snapshot(existing) == before

    def test_saveas_zip_raises(self, saved_xd_dir, tmp_path):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"])
        with pytest.raises(ValueError, match="loaded partially"):
            xd.saveas(tmp_path / "zipped", zip_output=True)
        assert not list(tmp_path.glob("zipped*"))

    def test_direct_cells_save_raises_before_touching_disk(self, saved_xd_dir, tmp_path):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"])
        target = tmp_path / "direct"
        with pytest.raises(ValueError, match="loaded partially"):
            xd.cells.save(target)
        assert not target.exists()

    def test_saving_works_after_full_load_and_keeps_all_layers(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"], boundaries=False)
        with pytest.raises(ValueError):
            xd.save()

        xd.load_cells()
        xd.save(verbose=False)

        reread = InSituData.read(saved_xd_dir)
        reread.load_cells()
        assert set(reread.cells.keys()) == {"main", "alt"}
        assert reread.cells["alt"].boundaries is not None

    def test_save_images_only_is_not_blocked(self, saved_xd_dir):
        xd = InSituData.read(saved_xd_dir)
        xd.load_cells(keys=["main"], boundaries=False)
        # writes no cells, so a partial cells modality is harmless here
        xd.save(images_only=True, verbose=False)
        reread = InSituData.read(saved_xd_dir)
        reread.load_cells()
        assert set(reread.cells.keys()) == {"main", "alt"}


class TestExperimentSelective:
    def test_load_cells_selective_no_stale_warning(self, saved_exp_dir):
        exp = InSituExperiment.read(saved_exp_dir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            exp.load_cells(keys=["main"], boundaries=False)
        assert _stale(record) == []
        for xd in exp.data:
            assert list(xd.cells.keys()) == ["main"]
            assert xd.cells["main"].boundaries is None

    def test_load_cells_keeps_stale_summary_with_boundaries(self, saved_exp_dir):
        exp = InSituExperiment.read(saved_exp_dir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            exp.load_cells(keys=["main"])
        stale = _stale(record)
        assert len(stale) == 1
        assert "2 of 3" in str(stale[0].message)

    def test_load_all_forwards(self, saved_exp_dir):
        exp = InSituExperiment.read(saved_exp_dir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            exp.load_all(cell_keys="main", cell_boundaries=False)
        assert _stale(record) == []
        assert all(list(xd.cells.keys()) == ["main"] for xd in exp.data)

    def test_save_saveas_save_cells_raise_and_write_nothing(self, saved_exp_dir, tmp_path):
        exp = InSituExperiment.read(saved_exp_dir)
        exp.load_cells(keys=["main"], boundaries=False)
        before = _snapshot(saved_exp_dir)
        siblings = sorted(p.name for p in tmp_path.iterdir())

        with pytest.raises(ValueError, match="3 dataset"):
            exp.save()
        with pytest.raises(ValueError, match="3 dataset"):
            exp.save_cells()
        with pytest.raises(ValueError, match="3 dataset"):
            exp.saveas(tmp_path / "new_exp")
        with pytest.raises(ValueError, match="3 dataset"):
            exp.saveas(saved_exp_dir, overwrite=True)

        assert _snapshot(saved_exp_dir) == before
        assert sorted(p.name for p in tmp_path.iterdir()) == siblings

    def test_one_partial_dataset_blocks_the_whole_experiment(self, saved_exp_dir):
        exp = InSituExperiment.read(saved_exp_dir)
        exp.load_cells()
        exp.data[1].load_cells(keys=["main"])
        before = _snapshot(saved_exp_dir)
        with pytest.raises(ValueError, match="1 dataset"):
            exp.save()
        assert _snapshot(saved_exp_dir) == before

    def test_view_saveas_raises(self, saved_exp_dir, tmp_path):
        exp = InSituExperiment.read(saved_exp_dir)
        exp.load_cells(keys=["main"], boundaries=False)
        view = exp[[0, 1]]
        with pytest.raises(ValueError, match="loaded partially"):
            view.saveas(tmp_path / "view_export")
        assert not (tmp_path / "view_export").exists()

    def test_full_load_unblocks_saving(self, saved_exp_dir):
        exp = InSituExperiment.read(saved_exp_dir)
        exp.load_cells(keys=["main"], boundaries=False)
        exp.load_cells()
        exp.save()
        reread = InSituExperiment.read(saved_exp_dir)
        reread.load_cells()
        assert all(set(xd.cells.keys()) == {"main", "alt"} for xd in reread.data)

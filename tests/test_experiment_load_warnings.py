"""Experiment-level loading collapses per-dataset StaleNucleusMapWarning into one summary."""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from insitupy._core.data import InSituData
from insitupy._exceptions import StaleNucleusMapWarning
from insitupy._logging import _count_warnings
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData
from insitupy.experiment.data import InSituExperiment

from tests.test_boundaries_data_save import _write_legacy_nucleus_map_to_store

CELL_NAMES = ["c1", "c2", "c3"]


def _make_xd_with_boundaries(seed):
    rng = np.random.default_rng(seed)
    table = AnnData(
        X=rng.integers(1, 10, size=(3, 2)).astype(float),
        obs=pd.DataFrame(index=pd.Index(CELL_NAMES)),
        var=pd.DataFrame(index=pd.Index(["g0", "g1"])),
    )
    boundaries = BoundariesData(cell_names=CELL_NAMES, seg_mask_value=[1, 2, 3])
    boundaries.add_boundaries(
        cell_boundaries=np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32),
        nuclei_boundaries=None,
        pixel_size=1,
    )
    xd = InSituData(path=None, metadata=None, slide_id="s", sample_id="s",
                    method_name="t", method_params={})
    xd.cells.add_celldata(cd=CellData(table=table, boundaries=boundaries),
                          key="main", is_main=True)
    return xd


@pytest.fixture
def experiment_with_two_stale_stores(tmp_path):
    exp = InSituExperiment()
    for i in range(3):
        exp.add(_make_xd_with_boundaries(i), metadata={"label": f"G{i}"})
    dest = tmp_path / "exp"
    exp.saveas(dest)
    # make the first two stores stale: positions sized for a 5-cell table, only 3 exist
    for ddir in ["data-000", "data-001"]:
        (store,) = list((dest / ddir).glob("cells/*/main/boundaries.zarr"))
        _write_legacy_nucleus_map_to_store(store, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
    return dest


def _stale(record):
    return [w for w in record if issubclass(w.category, StaleNucleusMapWarning)]


@pytest.mark.parametrize("method", ["load_cells", "load_all"])
def test_load_emits_one_summary_instead_of_per_dataset_warnings(
    experiment_with_two_stale_stores, method
):
    exp = InSituExperiment.read(experiment_with_two_stale_stores)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        getattr(exp, method)()

    stale = _stale(record)
    assert len(stale) == 1
    assert "2 of 3" in str(stale[0].message)
    # the stale map was dropped in the loaded data
    assert exp.data[0].cells["main"].boundaries.nucleus_to_cell_map is None


def test_summary_can_be_silenced_by_class(experiment_with_two_stale_stores):
    exp = InSituExperiment.read(experiment_with_two_stale_stores)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=StaleNucleusMapWarning)
        exp.load_cells()
    assert _stale(record) == []


def test_count_warnings_reemits_unrelated_warnings():
    class Other(UserWarning):
        pass

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with _count_warnings(StaleNucleusMapWarning) as recorded:
            warnings.warn("keep me", Other)
            warnings.warn("count me", StaleNucleusMapWarning)

    assert len(recorded) == 1
    assert [str(w.message) for w in record] == ["keep me"]
    assert record[0].category is Other

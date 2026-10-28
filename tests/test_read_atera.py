import json
import shutil

import numpy as np
import pytest
import zarr

from insitupy import InSituData
from insitupy._exceptions import InvalidAteraDirectory
from insitupy._io import _atera
from insitupy._io._atera import _encode_xenium_cell_ids, _read_transcripts_from_atera
from insitupy.datasets import atera_test_dataset, xenium_test_dataset_v4_nucex
from insitupy.datasets.datasets import _atera_test_dataset_dir
from insitupy.io import read_atera
from insitupy.utils.utils import convert_int_to_xenium_hex


@pytest.fixture(scope="module")
def atera_dir():
    """Download (if needed) the tiny Atera test dataset and return its directory."""
    return _atera_test_dataset_dir()


@pytest.fixture(scope="module")
def atera_data(atera_dir):
    return read_atera(atera_dir, load_transcripts=True)


@pytest.fixture(scope="module")
def cells_store(atera_dir):
    store = zarr.storage.ZipStore(atera_dir / "cells.zarr.zip", mode="r")
    yield zarr.open_group(store, mode="r")
    store.close()


def test_metadata(atera_data, atera_dir):
    meta = json.loads((atera_dir / "experiment.spatial").read_text())
    assert atera_data.metadata["method"] == "Atera"
    assert atera_data.slide_id == meta["slide_id"]
    assert atera_data.sample_id == meta["region_name"]
    assert atera_data.metadata["method_params"]["pixel_size"] == meta["pixel_size"]


def test_table_contains_only_gene_expression(atera_data, atera_dir):
    table = atera_data.cells.table
    meta = json.loads((atera_dir / "experiment.spatial").read_text())
    assert table.n_obs == meta["num_cells"]
    assert (table.var["feature_types"] == "Gene Expression").all()
    assert table.n_vars == meta["panel_wta"]["panel_num_targets"]
    # var names are gene symbols, not Ensembl ids
    assert not table.var_names.str.startswith("ENSG").any()
    assert table.var["gene_ids"].str.startswith("ENSG").all()


def test_10x_flags_do_not_hijack_scanpy(atera_data):
    table = atera_data.cells.table
    # scanpy's pca/neighbors pick up var["highly_variable"] automatically if present
    assert "highly_variable" not in table.var.columns
    assert "atera_highly_variable" in table.var.columns
    assert "atera_filtered" in table.obs.columns


def test_table_and_segmentation_are_aligned(atera_data, cells_store):
    table = atera_data.cells.table
    boundaries = atera_data.cells.boundaries
    assert list(table.obs_names) == list(boundaries.cell_names)
    # mask labels are the row position + 1
    assert int(np.asarray(cells_store["masks/1"][:]).max()) == table.n_obs


def test_multinucleated_cell_maps_to_one_cell(atera_data, cells_store):
    cell_index = cells_store["polygon_sets/0/cell_index"][:]
    multi = np.flatnonzero(np.bincount(cell_index) > 1)
    assert len(multi) > 0, "test data is expected to contain a multinucleated cell"
    cell_name = atera_data.cells.table.obs_names[multi[0]]
    nuclei = [i for i, c in atera_data.cells.boundaries.nucleus_to_cell_map.items() if c == cell_name]
    assert len(nuclei) >= 2


def test_spatial_coordinates_are_microns(atera_data, cells_store):
    centroids = cells_store["cell_summary"][:, :2]
    np.testing.assert_allclose(atera_data.cells.table.obsm["spatial"], centroids, rtol=1e-6)


def test_images(atera_dir, atera_data, cells_store):
    meta = json.loads((atera_dir / "experiment.spatial").read_text())
    assert list(atera_data.images.keys()) == ["nuclei"]
    nuclei = atera_data.images["nuclei"]
    nuclei = nuclei[0] if isinstance(nuclei, list) else nuclei
    assert nuclei.shape == cells_store["masks/1"].shape
    assert atera_data.images.metadata["nuclei"]["pixel_size"] == pytest.approx(meta["pixel_size"])

    with_stains = read_atera(atera_dir, load_cell_segmentation_images=True, verbose=False)
    assert set(with_stains.images.keys()) == {"nuclei", "atp1a1_cd45_e-cadherin", "18s", "alphasma_vimentin"}


def test_transcripts_are_opt_in(atera_dir):
    assert read_atera(atera_dir, verbose=False).transcripts is None


def test_transcripts(atera_data, atera_dir):
    tx = atera_data.transcripts.compute()
    store = zarr.storage.ZipStore(atera_dir / "transcripts.zarr.zip", mode="r")
    try:
        root = zarr.open_group(store, mode="r")
        gene_names = list(root.attrs["gene_names"])
        tile = root["grid"]["0,0"]
        gene_offset = tile["gene_offset"][:]
        n_rows = tile["location"].shape[0]
    finally:
        store.close()

    assert len(tx) == n_rows
    assert {"x_location", "y_location", "z_location", "feature_name", "qv",
            "overlaps_nucleus", "cell_id"} <= set(tx.columns)
    # feature name of each row follows the gene ranges in gene_offset
    expected = np.repeat(np.arange(len(gene_names)), gene_offset[:, 1] - gene_offset[:, 0])
    np.testing.assert_array_equal(tx["feature_name"].cat.codes.to_numpy(), expected)
    assigned = tx.loc[tx["cell_id"] != "UNASSIGNED", "cell_id"]
    assert len(assigned) > 0
    assert assigned.isin(atera_data.cells.table.obs_names).all()


def test_transcript_partitions_split_large_tiles(atera_dir, monkeypatch):
    whole = _read_transcripts_from_atera(atera_dir).compute()
    monkeypatch.setattr(_atera, "MAX_TRANSCRIPT_ROWS_PER_PARTITION", 100)
    split = _read_transcripts_from_atera(atera_dir)
    assert split.npartitions == int(np.ceil(len(whole) / 100))
    # a gene range crosses partition borders; the result must be identical
    split = split.compute().reset_index(drop=True)
    assert split.equals(whole.reset_index(drop=True))


def test_encode_xenium_cell_ids():
    ids = np.array([[7790, 1], [0, 1], [0xFFFFFFFF, 2], [0xFFFFFFFF, 0xFFFFFFFF]], dtype=np.uint32)
    encoded = _encode_xenium_cell_ids(ids)
    expected = [convert_int_to_xenium_hex(int(a), int(b)) for a, b in ids[:3]]
    assert list(encoded[:3]) == expected
    assert encoded[0] == "aaaabogo-1"
    assert encoded[3] == "UNASSIGNED"


def test_saveas_roundtrip(atera_data, tmp_path):
    out = tmp_path / "atera_project"
    atera_data.saveas(out, verbose=False)
    back = InSituData.read(out)
    back.load_all()
    assert back.metadata["method"] == "Atera"
    assert list(back.cells.table.obs_names) == list(atera_data.cells.table.obs_names)
    assert len(back.transcripts) == len(atera_data.transcripts)
    assert "nuclei" in back.images.keys()


def test_name_arguments_override_metadata(atera_dir):
    data = read_atera(atera_dir, dataset_name="my_slide", sample_name="my_sample", verbose=False)
    assert data.slide_id == "my_slide"
    assert data.sample_id == "my_sample"


def test_missing_files_raise(atera_dir, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(atera_dir, broken)
    (broken / "cells.zarr.zip").unlink()
    with pytest.raises(InvalidAteraDirectory, match="cells.zarr.zip"):
        read_atera(broken)


def test_xenium_formatted_data_is_delegated():
    # preproduction Atera data has the Xenium format; use a tiny Xenium bundle as a stand-in
    xenium_dir = xenium_test_dataset_v4_nucex().path
    data = read_atera(xenium_dir, sample_name="atera_preview", verbose=False)
    assert data.metadata["method"] == "Atera"
    assert data.sample_id == "atera_preview"
    assert data.cells.table.n_obs > 0


def test_atera_test_dataset():
    data = atera_test_dataset()
    assert isinstance(data, InSituData)
    assert data.metadata["method"] == "Atera"

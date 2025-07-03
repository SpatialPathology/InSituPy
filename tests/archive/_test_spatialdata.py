from pathlib import Path
from spatialdata import read_zarr
import pytest
from insitupy.datasets.datasets import xenium_test_dataset


SDATA_PATH = Path("tests/data/spatialdata_example")

image_x = 3522
image_y = 5789
n_cells = 23
n_transripts = 1985
n_genes = 5006
n_images = 1
image_names = ["nuclei"]

@pytest.mark.parametrize("levels", [1, 3])
def test_tospatialdata(levels: int):
    xd = xenium_test_dataset()
    xd.load_all()

    sdata = xd.to_spatialdata(levels)

    sdata_ref = read_zarr(SDATA_PATH)

    assert len(sdata.images) == n_images
    assert len(sdata.images["nuclei"].groups) == levels + 2
    assert sdata.images["nuclei"]["scale0"].dims == sdata_ref.images["morphology_focus"]["scale0"].dims


    assert len(sdata.labels) == 2
    assert len(sdata.labels["cellular"].groups) == levels + 2
    assert len(sdata.labels["nuclear"].groups) == levels + 2
    assert sdata.labels["cellular"]["scale0"].dims == sdata_ref.labels["cell_labels"]["scale0"].dims
    assert sdata.labels["nuclear"]["scale0"].dims == sdata_ref.labels["nucleus_labels"]["scale0"].dims

    assert sdata.shapes["cell_circles"].equals(sdata.shapes["cell_circles"])

    assert len(sdata.tables) == 1
    #from anndata.tests.helpers import assert_adata_equal
    #assert_adata_equal(sdata.tables["table"], sdata_ref.tables["table"], exact=False)
from pathlib import Path

import scanpy as sc

from insitupy.datasets.datasets import xenium_test_dataset
from insitupy.preprocessing import (cluster_cells, normalize_and_transform,
                                    reduce_dimensions)

BAYSOR_PATH = Path("tests/data/baysor_output-slide__region__20241212__134825__j1_b1_s1")
image_x = 3522
image_y = 5789
n_cells = 23
n_transripts = 1985
n_genes = 5006
n_images = 1
image_names = ["nuclei"]

def test_read():
    xd = xenium_test_dataset()
    xd.load_all()

    assert len(xd.images.metadata) == n_images
    assert xd.images["nuclei"][0].shape == (image_x, image_y)

    assert len(xd.cells.boundaries.metadata) == 2
    assert xd.cells.boundaries["cells"].shape == (image_x, image_y)
    assert xd.cells.boundaries["nuclei"].shape == (image_x, image_y)

    assert len(xd.transcripts) == n_transripts
    assert len(xd.transcripts.columns) == 13
    assert xd.cells.table.shape == (n_cells, n_genes)



def test_functions():
    xd = xenium_test_dataset()
    xd.load_all()
    sc.pp.filter_cells(xd.cells.table, min_counts=1, inplace=True)
    normalize_and_transform(xd, transformation_method="sqrt")
    reduce_dimensions(xd, method="umap")
    cluster_cells(xd, method="leiden")
    for key in ['spatial', 'X_pca', 'X_umap']:
        assert key in xd.cells.table.obsm.keys()
    assert "leiden" in xd.cells.table.obs.columns
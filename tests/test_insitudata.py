from pathlib import Path
from insitupy import read_xenium


DATA_PATH = Path("tests/data/Xenium_Prime_Mouse_Ileum_tiny_outs")
BAYSOR_PATH = Path("tests/data/baysor_output-slide__region__20241212__134825__j1_b1_s1")
image_x = 3522
image_y = 5789
n_cells = 23
n_transripts = 1985
n_genes = 5006
n_images = 1
image_names = ["nuclei"]

def test_read():
    xd = read_xenium(DATA_PATH)
    xd.load_all()

    assert len(xd.images.metadata) == n_images
    assert xd.images["nuclei"][0].shape == (image_x, image_y)

    assert len(xd.cells["main"].boundaries.metadata) == 2
    assert xd.cells["main"].boundaries["cellular"].shape == (image_x, image_y)
    assert xd.cells["main"].boundaries["nuclear"].shape == (image_x, image_y)

    assert xd.transcripts.shape == (n_transripts, 9)
    assert xd.cells["main"].matrix.shape == (n_cells, n_genes)


def test_baysor():
    xd = read_xenium(DATA_PATH)
    xd.load_all()
    xd.add_baysor(BAYSOR_PATH)
    assert xd.cells is not None
    assert len(xd.cells.get_all_keys()) == 2
    assert "baysor" in xd.cells.get_all_keys()
    assert xd.cells["baysor"].matrix is not None
    assert xd.cells["baysor"].matrix.shape == (18, 11)
    assert xd.cells["baysor"].boundaries is not None
    assert xd.cells["baysor"].boundaries
    assert 'cellular' in xd.cells["baysor"].boundaries.metadata.keys()
    

def test_functions():
    xd = read_xenium(DATA_PATH)
    xd.load_all()
    xd.normalize_and_transform(transformation_method="sqrt")
    xd.reduce_dimensions(umap=True, tsne=False)
    for key in ['spatial', 'X_pca', 'X_umap']:
        assert key in xd.cells["main"].matrix.obsm.keys()
    assert "leiden" in xd.cells["main"].matrix.obs.columns
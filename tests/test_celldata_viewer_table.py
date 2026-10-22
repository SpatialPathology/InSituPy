"""Tests for the X_csc.zarr viewer sidecar written by CellData.save (viewer milestone M4).

The sidecar is read here with plain zarr / anndata, never by InSituPy.
"""

import json
import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata import AnnData

from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.data import InSituData
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.cell_data import CellData


def _make_celldata(n_cells=6, n_genes=4, boundaries=True, shuffle=True, drop_from_bounds=1, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"c{i}" for i in range(n_cells)]
    X = sp.random(n_cells, n_genes, density=0.6, format="csr", random_state=seed, dtype=np.float32)
    obs = pd.DataFrame({
        "cat": pd.Categorical(["a", "b"] * (n_cells // 2)),
        "score": rng.random(n_cells),
        "count": np.arange(n_cells),
    }, index=names)
    table = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100

    bounds = None
    expected = {n: 0 for n in names}
    if boundaries:
        # boundaries know the cells in another order, and lack the last `drop_from_bounds` cells
        b_names = names[: n_cells - drop_from_bounds]
        if shuffle:
            b_names = b_names[::-1]
        labels = [10 + i for i in range(len(b_names))]
        expected.update(dict(zip(b_names, labels)))
        bounds = BoundariesData(cell_names=b_names, seg_mask_value=labels)
        bounds.add_boundaries(cell_boundaries=np.arange(6, dtype=np.uint32).reshape(2, 3),
                              nuclei_boundaries=None, pixel_size=1)
    return CellData(table=table, boundaries=bounds), expected


def _open_sidecar(layer_dir):
    cd = json.loads((layer_dir / ".celldata").read_text())
    side = layer_dir / cd["x_csc"]
    return side, zarr.open_group(side, zarr_format=2, mode="r")


class TestSidecarContent:
    def test_sidecar_matches_table_and_label_ids_by_name(self, tmp_path):
        cd, expected = _make_celldata()
        obs_before = cd.table.obs.copy()
        cd.save(tmp_path / "layer")

        side, g = _open_sidecar(tmp_path / "layer")
        assert (side / ".zmetadata").exists()
        assert not (side / "zarr.json").exists()
        assert g["X"].attrs["encoding-type"] == "csc_matrix"
        assert g["X/indptr"].shape[0] == cd.table.n_vars + 1

        back = anndata.read_zarr(side)
        assert sp.issparse(back.X) and back.X.format == "csc"
        np.testing.assert_allclose(back.X.toarray(), cd.table.X.toarray())
        np.testing.assert_allclose(back.obsm["spatial"], cd.table.obsm["spatial"])
        for col in ("cat", "score", "count"):
            assert list(back.obs[col]) == list(cd.table.obs[col])
        assert list(back.obs_names) == list(cd.table.obs_names)
        # label ids by name; the cell missing from the boundaries gets 0
        assert dict(zip(back.obs_names, back.obs["_label_id"])) == expected
        assert 0 in set(back.obs["_label_id"])
        assert back.obs["_label_id"].dtype == np.uint32

        # the user's table is not mutated
        pd.testing.assert_frame_equal(cd.table.obs, obs_before)
        assert "_label_id" not in cd.table.obs.columns

    def test_read_ignores_sidecar(self, tmp_path):
        cd, _ = _make_celldata()
        cd.save(tmp_path / "layer")
        back = CellData.read(tmp_path / "layer")
        assert "_label_id" not in back.table.obs.columns
        assert sp.issparse(back.table.X)
        np.testing.assert_allclose(back.table.X.toarray(), cd.table.X.toarray())

    def test_layer_without_boundaries(self, tmp_path):
        cd, _ = _make_celldata(boundaries=False)
        cd.save(tmp_path / "layer")
        side, g = _open_sidecar(tmp_path / "layer")
        assert "_label_id" not in g["obs"]
        assert "spatial" in g["obsm"]

    def test_dense_x_is_written_csc(self, tmp_path):
        cd, _ = _make_celldata()
        dense = cd.table.X.toarray()
        cd.table.X = dense
        cd.save(tmp_path / "layer")
        side, g = _open_sidecar(tmp_path / "layer")
        assert g["X"].attrs["encoding-type"] == "csc_matrix"
        np.testing.assert_allclose(anndata.read_zarr(side).X.toarray(), dense)

    def test_no_zarr_v2_warning_reaches_caller(self, tmp_path):
        cd, _ = _make_celldata()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cd.save(tmp_path / "layer")
        assert not [w for w in rec if "zarr v2" in str(w.message)]


def _make_xd(layers):
    xd = InSituData(path=None, metadata=None,
                    slide_id="s", sample_id="s", method_name="t", method_params={})
    for i, (key, cd) in enumerate(layers.items()):
        xd.cells.add_celldata(cd=cd, key=key, is_main=(i == 0))
    return xd


class TestSidecarInProject:
    def test_each_layer_has_its_own_sidecar(self, tmp_path):
        main, exp_main = _make_celldata(seed=0)
        other, exp_other = _make_celldata(seed=1, shuffle=False, drop_from_bounds=2)
        p = tmp_path / "p"
        _make_xd({"main": main, "proseg": other}).saveas(p, verbose=False)

        cells_dir = p / json.loads((p / ISPY_METADATA_FILE).read_text())["data"]["cells"]
        for layer, expected in (("main", exp_main), ("proseg", exp_other)):
            side, _ = _open_sidecar(cells_dir / layer)
            back = anndata.read_zarr(side)
            assert dict(zip(back.obs_names, back.obs["_label_id"])) == expected

    def test_failed_sidecar_write_keeps_old_commit(self, tmp_path, monkeypatch):
        cd, _ = _make_celldata()
        p = tmp_path / "p"
        _make_xd({"main": cd}).saveas(p, verbose=False)
        xd = InSituData.read(p)
        committed = json.loads((p / ISPY_METADATA_FILE).read_text())["data"]["cells"]
        dirs_before = sorted(d.name for d in (p / "cells").iterdir())
        values_before = xd.cells.table.X.toarray().copy()

        xd.cells.table.obs["flag"] = "never committed"

        def _boom(*args, **kwargs):
            raise OSError("simulated sidecar failure")

        monkeypatch.setattr("insitupy.containers.cell_data.write_viewer_table", _boom)
        with pytest.raises(OSError, match="simulated sidecar failure"):
            xd.save(verbose=False)
        monkeypatch.undo()

        assert json.loads((p / ISPY_METADATA_FILE).read_text())["data"]["cells"] == committed
        assert sorted(d.name for d in (p / "cells").iterdir()) == dirs_before
        back = InSituData.read(p)
        assert "flag" not in back.cells.table.obs.columns
        np.testing.assert_allclose(back.cells.table.X.toarray(), values_before)

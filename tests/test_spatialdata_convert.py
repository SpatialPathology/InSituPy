"""Tests for spatialdata.convert_to_spatialdata.

All tests are skipped when the spatialdata package is not installed.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# Skip the entire module if spatialdata is not installed
pytest.importorskip("spatialdata")

from insitupy._core.data import InSituData  # noqa: E402
from insitupy.containers.dataclasses import CellData  # noqa: E402
from insitupy.spatialdata.convert import convert_to_spatialdata  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_insitudata(n_cells=10, n_genes=5, seed=0):
    """Minimal InSituData with expression table and spatial coordinates."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(float)

    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(n_cells)]))
    var = pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_genes)]))
    table = AnnData(X=X, obs=obs, var=var)
    table.obsm["spatial"] = rng.random((n_cells, 2)) * 100

    celldata = CellData(table=table, boundaries=None)
    xd = InSituData(
        path=None, metadata=None,
        slide_id="test", sample_id="s1",
        method_name="test", method_params={},
    )
    xd.cells.add_celldata(cd=celldata, key="main", is_main=True)
    return xd


# ── convert_to_spatialdata ────────────────────────────────────────────────────

class TestConvertToSpatialdata:
    def test_returns_spatialdata_object(self):
        from spatialdata import SpatialData
        xd = _make_insitudata()
        sdata = convert_to_spatialdata(xd)
        assert isinstance(sdata, SpatialData)

    def test_tables_element_present(self):
        xd = _make_insitudata()
        sdata = convert_to_spatialdata(xd)
        assert len(sdata.tables) > 0

    def test_table_obs_shape_matches(self):
        n_cells = 8
        xd = _make_insitudata(n_cells=n_cells)
        sdata = convert_to_spatialdata(xd)
        table = next(iter(sdata.tables.values()))
        assert table.n_obs == n_cells

    def test_table_var_shape_matches(self):
        n_genes = 6
        xd = _make_insitudata(n_genes=n_genes)
        sdata = convert_to_spatialdata(xd)
        table = next(iter(sdata.tables.values()))
        assert table.n_vars == n_genes

    def test_no_images_element_when_none_loaded(self):
        xd = _make_insitudata()
        sdata = convert_to_spatialdata(xd)
        # Without images loaded, images dict should be empty
        assert len(sdata.images) == 0

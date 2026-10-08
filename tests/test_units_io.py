"""Save/load round-trip tests for the units modality.

These exercise the path that previously raised AttributeError (`_save_units`
read `units.data` instead of `.table`) and had never been covered by a test:
saveas() -> InSituData.read() with units attached, legacy on-disk layout
detection, and persisting a units update via `.save()` on an existing
project.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from shapely.geometry import Point

from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.data import InSituData
from insitupy._io.files import write_dict_to_json
from insitupy.containers.spatial_units_data import SpatialUnitsData

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_units(names, unit_type="unit", n_vars=2, seed=0):
    rng = np.random.default_rng(seed)
    gdf = gpd.GeoDataFrame(
        {
            "name": names,
            "geometry": [Point(i, i).buffer(0.4) for i in range(len(names))],
        }
    )
    table = AnnData(
        X=rng.random((len(names), n_vars)),
        obs=pd.DataFrame(index=pd.Index(names, dtype=str)),
        var=pd.DataFrame(index=[f"v{i}" for i in range(n_vars)]),
    )
    return SpatialUnitsData(shapes=gdf, data=table, unit_type=unit_type)


def _make_insitudata():
    """Minimal InSituData with no project path and no cells."""
    return InSituData(
        path=None, metadata=None,
        slide_id="slide1", sample_id="s1",
        method_name="test", method_params={},
    )


# ── saveas() -> read() round-trip with multiple unit layers ─────────────────


def test_saveas_read_roundtrip_multi_layer_units(tmp_path):
    xd = _make_insitudata()
    xd.add_units(_make_units(["v1", "v2"], unit_type="visium", seed=1))
    xd.add_units(_make_units(["n1"], unit_type="niche", seed=2), key="niche")

    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)

    xd2 = InSituData.read(proj_dir)

    assert set(xd2.units.keys()) == {"visium", "niche"}
    assert xd2.units.main_key == "visium"
    assert list(xd2.units["visium"].table.obs_names) == ["v1", "v2"]
    assert list(xd2.units["niche"].table.obs_names) == ["n1"]
    np.testing.assert_allclose(
        xd2.units["visium"].table.X, xd.units["visium"].table.X
    )


# ── Legacy flat on-disk layout (pre-multi-unit InSituPy versions) ───────────


def test_load_units_reads_legacy_flat_layout(tmp_path):
    proj_dir = tmp_path / "legacy_project"
    proj_dir.mkdir()

    # Write a flat units/ folder directly (shapes.parquet + data.h5ad +
    # metadata.json, no .multispatialunitsdata marker) -- this is exactly
    # what SpatialUnitsData.save() itself produces.
    su = _make_units(["u1", "u2"], unit_type="niche")
    su.save(path=proj_dir / "units")

    # A real (if old-format) saved project always has a `.ispy` marker --
    # only the units/ sub-layout predates multi-unit support.
    write_dict_to_json(
        dictionary={"slide_id": "slide1", "sample_id": "s1"},
        file=proj_dir / ISPY_METADATA_FILE,
    )

    xd = InSituData.read(proj_dir, load_all=False)
    xd.load_units(verbose=False)

    assert xd.units.main_key == "main"
    assert list(xd.units.keys()) == ["main"]
    assert list(xd.units.table.obs_names) == ["u1", "u2"]


# ── save() on an existing project persists newly added units ────────────────


def test_save_to_existing_project_persists_new_units_layer(tmp_path):
    xd = _make_insitudata()
    xd.add_units(_make_units(["v1"], unit_type="visium"))

    proj_dir = tmp_path / "proj"
    xd.saveas(proj_dir, verbose=False)

    xd2 = InSituData.read(proj_dir)
    xd2.add_units(_make_units(["n1", "n2"], unit_type="niche"), key="niche")
    xd2.save(verbose=False)

    # verify via a fully independent re-read from disk
    xd3 = InSituData.read(proj_dir)
    assert set(xd3.units.keys()) == {"visium", "niche"}
    assert xd3.units.main_key == "visium"
    assert list(xd3.units["niche"].table.obs_names) == ["n1", "n2"]

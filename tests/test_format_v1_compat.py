"""Backward compatibility with the InSituPy 0.12 ("format 1") on-disk layout.

The projects are written by ``tests/format_v1_fixtures.py`` directly with low-level libraries,
so they stay a frozen picture of what 0.12 wrote while InSituPy's own writers move to format 2.
This suite is the compatibility gate for the format-v2 work (A6 plan, WP-0): it must pass on
the unchanged 0.12/0.13.dev code first, and later on ``feat/format-v2`` with the documented
change for ``save()`` (format-1 projects become read-only and need ``insitupy.io.migrate()``).

Every test compares values, not just that a call succeeds.
"""

import inspect
import types

import numpy as np
import pandas as pd
import pytest
from shapely import wkt

import tests.format_v1_fixtures as fx
from insitupy import InSituData, InSituExperiment
from tests.format_v1_fixtures import V1_VARIANTS, write_v1_experiment, write_v1_project

# 0.12 writes cell names as fixed-length unicode, which zarr-python 3 flags as unspecified
pytestmark = pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")


# ── value comparison ──────────────────────────────────────────────────────────


def _assert_cells(xd, expected, *, extra_obs: dict | None = None):
    table = xd.cells.table
    want = expected["table"]
    np.testing.assert_array_equal(table.X.toarray(), want.X.toarray())
    assert list(table.var_names) == list(want.var_names)
    want_obs = want.obs.copy()
    for col, value in (extra_obs or {}).items():
        # anndata stores a string column as categorical on write
        want_obs[col] = pd.Categorical([value] * len(want_obs))
    pd.testing.assert_frame_equal(table.obs, want_obs, check_like=True)
    np.testing.assert_allclose(table.obsm["spatial"], want.obsm["spatial"])

    b = xd.cells.boundaries
    assert list(np.asarray(b.cell_names).astype(str)) == expected["cell_names"]
    np.testing.assert_array_equal(np.asarray(b.seg_mask_value), expected["seg_mask_value"])
    for layer, levels in expected["masks"].items():
        assert len(b[layer]) == len(levels)
        for got, want_level in zip(b[layer], levels, strict=True):
            np.testing.assert_array_equal(np.asarray(got), want_level)
        # a wrong mask pixel size misaligns the segmentation with images and transcripts
        assert b.metadata[layer]["pixel_size"] == expected["mask_pixel_size"]
    assert b.nucleus_to_cell_map == expected["nucleus_to_cell_map"]


def _assert_images(xd, expected):
    assert sorted(xd.images.metadata.keys()) == sorted(expected["images"])
    for name, levels in expected["images"].items():
        got = xd.images[name]
        assert len(got) == len(levels)
        for got_level, want_level in zip(got, levels, strict=True):
            # in memory RGB is always channels-last (YXS), whatever the store layout
            np.testing.assert_array_equal(np.asarray(got_level), want_level)
        meta = xd.images.metadata[name]
        assert meta["pixel_size"] == expected["image_pixel_size"]
        assert meta["rgb"] is expected["image_rgb"][name]


def _assert_transcripts(xd, expected):
    got = xd.transcripts.compute()
    want = expected["transcripts"]
    key = ["x_location", "y_location"]
    got = got.sort_values(key).reset_index(drop=True)
    want = want.sort_values(key).reset_index(drop=True)
    for col in ("x_location", "y_location", "z_location"):
        np.testing.assert_allclose(got[col].to_numpy(), want[col].to_numpy())
    for col in ("feature_name", "cell_id"):
        assert list(got[col].astype(str)) == list(want[col].astype(str))


def _assert_units(xd, expected):
    assert sorted(xd.units.keys()) == sorted(expected["units"])
    for key, want in expected["units"].items():
        su = xd.units[key]
        assert list(su.table.obs_names) == want["names"]
        np.testing.assert_allclose(np.asarray(su.table.X), want["X"])
        got_geoms = list(su.shapes.loc[want["names"]].geometry)
        assert all(g.equals(wkt.loads(w)) for g, w in zip(got_geoms, want["wkt"], strict=True))


def _assert_shapes(container, expected):
    assert sorted(container.keys()) == sorted(expected)
    for key, want in expected.items():
        gdf = container[key]
        assert list(gdf["name"]) == want["names"]
        assert all(g.equals(wkt.loads(w)) for g, w in zip(gdf.geometry, want["wkt"], strict=True))


def _assert_project(xd, expected, *, extra_obs: dict | None = None):
    assert xd.slide_id == expected["slide_id"]
    assert xd.sample_id == expected["sample_id"]
    _assert_cells(xd, expected, extra_obs=extra_obs)
    _assert_images(xd, expected)
    _assert_transcripts(xd, expected)
    _assert_units(xd, expected)
    _assert_shapes(xd.annotations, expected["annotations"])
    _assert_shapes(xd.regions, expected["regions"])


def _assert_lazy_still_valid(xd, expected):
    """Lazily backed arrays must still compute the right values (zarr reads a chunk missing
    after a directory swap as zeros instead of raising)."""
    for layer, levels in expected["masks"].items():
        np.testing.assert_array_equal(np.asarray(xd.cells.boundaries[layer][0]), levels[0])
    for name, levels in expected["images"].items():
        np.testing.assert_array_equal(np.asarray(xd.images[name][0]), levels[0])
    _assert_transcripts(xd, expected)


# ── the fixture writer itself ─────────────────────────────────────────────────


def test_fixture_writer_uses_no_insitupy_code():
    """The fixtures must not go through InSituPy writers, or they would drift with them."""
    for name, obj in vars(fx).items():
        if isinstance(obj, types.ModuleType):
            assert not obj.__name__.startswith("insitupy"), name
        elif inspect.isfunction(obj) or inspect.isclass(obj):
            assert not (obj.__module__ or "").startswith("insitupy"), name


# ── InSituData ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("variant", V1_VARIANTS)
def test_read_v1_project(tmp_path, variant):
    p = tmp_path / "proj"
    expected = write_v1_project(p, variant=variant)

    xd = InSituData.read(p)

    _assert_project(xd, expected)
    # the stale absolute path stored by 0.12 (B8) is never used to locate data
    assert xd.path.resolve() == p.resolve()


@pytest.mark.parametrize("variant", V1_VARIANTS)
def test_saveas_v1_project_to_new_path(tmp_path, variant):
    p = tmp_path / "proj"
    expected = write_v1_project(p, variant=variant)
    xd = InSituData.read(p)

    copy_path = tmp_path / "copy"
    xd.saveas(copy_path, verbose=False)

    _assert_project(InSituData.read(copy_path), expected)
    _assert_lazy_still_valid(xd, expected)
    # the source project is left as it was
    _assert_project(InSituData.read(p), expected)


def test_saveas_copy_over_v1_source(tmp_path):
    """A copy that still reads lazily from the project it overwrites must not turn masks or
    images into zeros (the swap deletes the directories the lazy arrays point into). The
    linked object itself is refused by saveas(); a copy is the supported route."""
    p = tmp_path / "proj"
    expected = write_v1_project(p)
    c = InSituData.read(p).copy()

    c.saveas(p, overwrite=True, verbose=False)

    _assert_lazy_still_valid(c, expected)
    _assert_project(InSituData.read(p), expected)


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            v,
            marks=pytest.mark.xfail(
                raises=KeyError, strict=True,
                reason="B14: save() reads .ispy 'uids' unguarded; stores without it raise",
            ),
        ) if v == "pre_uids" else v
        for v in V1_VARIANTS
    ],
)
def test_save_into_v1_project(tmp_path, variant):
    """An edit saved into a 0.12 project persists and leaves every other modality intact.

    Format v2 (A6 WP-2b) makes format-1 projects read-only: this test then changes to expect
    ``FormatVersionError``, and the value check moves to ``migrate()`` + ``save()`` (WP-8).
    """
    p = tmp_path / "proj"
    expected = write_v1_project(p, variant=variant)
    xd = InSituData.read(p)

    xd.cells.table.obs["flag"] = "edited"
    xd.save(verbose=False)

    _assert_project(InSituData.read(p), expected, extra_obs={"flag": "edited"})


# ── InSituExperiment ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("csv_only", [False, True], ids=["parquet", "csv_only"])
def test_read_v1_experiment(tmp_path, csv_only):
    root = tmp_path / "exp"
    expected = write_v1_experiment(root, n=3, csv_only=csv_only)

    exp = InSituExperiment.read(root)

    meta = exp.metadata
    want = expected["metadata"]
    # the legacy CSV's auto-generated slide_id/sample_id columns are dropped on read
    assert sorted(meta.columns) == sorted(want.columns)
    for col in want.columns:
        assert list(meta[col].astype(str)) == list(want[col].astype(str)), col

    # datasets pair with metadata rows by uid, not by directory order
    for i, uid in enumerate(want["uid"]):
        xd = exp.data[i]
        assert xd.uid == uid
        xd.load_cells()
        np.testing.assert_array_equal(
            xd.cells.table.X.toarray(), expected["datasets"][uid]["table"].X.toarray()
        )

    selected = exp.filters.view("only_a").metadata["uid"].tolist()
    assert selected == [u for u, m in zip(want["uid"], expected["filters"]["only_a"], strict=True) if m]

"""Tests for BoundariesData.save() - regression coverage for the dask
to_zarr(..., zarr_array_kwargs=...) API-incompatibility fix, and for the
removal of the broken non-pyramid save/read path.

See .log/reports/260710/boundaries-zarr-save-fix/report-boundaries-zarr-save-fix.md
See .log/reports/260715/boundaries-nonpyramid-removal/report-boundaries-nonpyramid-removal.md
"""

import warnings
from contextlib import ExitStack

import dask.array as da
import numpy as np
import pytest

from insitupy._exceptions import InvalidFileTypeError
from insitupy.containers._zarr_compat import (
    ZARR_V3,
    _get_zarr_store,
    _write_dask_array_to_zarr,
)
from insitupy.containers.boundaries_data import BoundariesData
from insitupy.containers.io import _read_boundaries_from_celldata_zarr


def _create_boundaries(cell_names=("c1", "c2", "c3"), seg_mask_value=(1, 2, 3),
                        nucleus_to_cell_map=None, nucleus_count=None):
    boundaries = BoundariesData(
        cell_names=list(cell_names),
        seg_mask_value=list(seg_mask_value),
        nucleus_to_cell_map=nucleus_to_cell_map,
        nucleus_count=nucleus_count,
    )
    mask = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32)
    boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=None, pixel_size=1)
    return boundaries


def _write_legacy_nucleus_map_to_store(path, nucleus_map: dict):
    """Hand-write the pre-0.12.0b7 flat [[nucleus_idx, row_position], ...] int64 array
    directly into an existing boundaries.zarr store, bypassing BoundariesData.save()
    (which can no longer produce this shape) - simulates a store written by an older
    InSituPy version for the back-compat read path."""
    with ExitStack() as stack:
        dirstore = _get_zarr_store(path, mode="a")
        if not ZARR_V3:
            dirstore = stack.enter_context(dirstore)
        arr = np.array([[k, v] for k, v in nucleus_map.items()], dtype=np.int64)
        _write_dask_array_to_zarr(dirstore, "nucleus_to_cell_map", da.from_array(arr))


class TestSaveRoundtripsAllMetadataArrays:
    def test_save_roundtrips_all_metadata_arrays(self, tmp_path):
        # Xenium v2.0+ multinucleated path: nucleus_to_cell_map/nucleus_count both set
        boundaries = _create_boundaries(
            nucleus_to_cell_map={0: "c1", 1: "c1", 2: "c2"},
            nucleus_count=np.array([2, 1, 0], dtype=np.int64),
        )

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        cell_names = da.from_zarr(path, component="cell_names").compute()
        seg_mask_value = da.from_zarr(path, component="seg_mask_value").compute()
        nuc_idx = da.from_zarr(path, component="nucleus_to_cell_map/nucleus_index").compute()
        cell_name = da.from_zarr(path, component="nucleus_to_cell_map/cell_name").compute()
        nucleus_count = da.from_zarr(path, component="nucleus_count").compute()

        assert list(cell_names) == ["c1", "c2", "c3"]
        np.testing.assert_array_equal(seg_mask_value, np.array([1, 2, 3], dtype=np.uint32))
        assert {int(i): str(n) for i, n in zip(nuc_idx, cell_name)} == {0: "c1", 1: "c1", 2: "c2"}
        np.testing.assert_array_equal(nucleus_count, np.array([2, 1, 0], dtype=np.int64))


class TestSaveOverwritesOnSecondCall:
    def test_save_overwrites_on_second_call(self, tmp_path):
        # No add_boundaries() call here: isolates the overwrite=True
        # semantics of the 4 metadata-array writes from the mask-array
        # writes, which are covered separately below.
        path = tmp_path / "boundaries.zarr"

        boundaries = BoundariesData(cell_names=["c1", "c2", "c3"], seg_mask_value=[1, 2, 3])
        boundaries.save(path=path)

        boundaries2 = BoundariesData(cell_names=["x1", "x2", "x3"], seg_mask_value=[4, 5, 6])
        # Must not raise (ContainsArrayError) on the second save to the same path.
        boundaries2.save(path=path)

        cell_names = da.from_zarr(path, component="cell_names").compute()
        seg_mask_value = da.from_zarr(path, component="seg_mask_value").compute()
        assert list(cell_names) == ["x1", "x2", "x3"]
        np.testing.assert_array_equal(seg_mask_value, np.array([4, 5, 6], dtype=np.uint32))


class TestSaveWithMasksOverwritesOnSecondCall:
    """Regression coverage: re-saving boundaries *with masks* to the same
    non-zipped path used to raise zarr.errors.ContainsArrayError because the
    mask-array writes did not pass overwrite=True (unlike the 4 metadata
    arrays, which already did)."""

    def test_second_save_with_masks_wins(self, tmp_path):
        path = tmp_path / "boundaries.zarr"
        mask1 = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32)
        mask2 = np.array([[0, 4, 5], [6, 0, 0]], dtype=np.uint32)

        boundaries = BoundariesData(cell_names=["c1", "c2", "c3"], seg_mask_value=[1, 2, 3])
        boundaries.add_boundaries(cell_boundaries=mask1, nuclei_boundaries=None, pixel_size=1)
        boundaries.save(path=path)

        boundaries2 = BoundariesData(cell_names=["x1", "x2", "x3"], seg_mask_value=[4, 5, 6])
        boundaries2.add_boundaries(cell_boundaries=mask2, nuclei_boundaries=None, pixel_size=2)
        # Must not raise (ContainsArrayError) on the second save to the same path.
        boundaries2.save(path=path)

        comp = "masks/cells/0"
        mask_on_disk = da.from_zarr(path, component=comp).compute()
        cell_names = da.from_zarr(path, component="cell_names").compute()
        np.testing.assert_array_equal(mask_on_disk, mask2)
        assert list(cell_names) == ["x1", "x2", "x3"]


class TestSaveWithoutNucleusMetadata:
    def test_save_without_nucleus_metadata(self, tmp_path):
        boundaries = _create_boundaries(nucleus_to_cell_map=None, nucleus_count=None)

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        import zarr
        store = zarr.open(path, mode="r")
        assert "nucleus_to_cell_map" not in store
        assert "nucleus_count" not in store


class TestIndependentCellNucleusPixelSize:
    """WP4: add_boundaries(nucleus_pixel_size=...) - cell and nucleus masks from a
    foreign store are not guaranteed to share a resolution, unlike InSituPy's own
    exporter (which always calls add_boundaries with one shared pixel_size)."""

    def test_nucleus_pixel_size_defaults_to_cell_pixel_size(self):
        boundaries = BoundariesData(cell_names=["c1", "c2"], seg_mask_value=[1, 2])
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint32)
        boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=mask, pixel_size=0.5)

        assert boundaries.metadata["cells"]["pixel_size"] == 0.5
        assert boundaries.metadata["nuclei"]["pixel_size"] == 0.5

    def test_independent_nucleus_pixel_size_is_stored_separately(self):
        boundaries = BoundariesData(cell_names=["c1", "c2"], seg_mask_value=[1, 2])
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint32)
        boundaries.add_boundaries(
            cell_boundaries=mask, nuclei_boundaries=mask,
            pixel_size=1.0, nucleus_pixel_size=0.25,
        )

        assert boundaries.metadata["cells"]["pixel_size"] == 1.0
        assert boundaries.metadata["nuclei"]["pixel_size"] == 0.25


class TestLoadInvalidatesInconsistentNucleusMap:
    """Regression coverage for the shipped demo: an older InSituPy filtered cells
    without maintaining nucleus_to_cell_map, leaving a saved map/count sized for the
    pre-filter cell count. The load path must detect and drop it (with a warning)
    rather than propagate a silently wrong mapping."""

    def test_load_drops_map_referencing_unknown_cell_name(self, tmp_path):
        # current (name-based) format: "ghost" is not among the 3 saved cell names.
        boundaries = _create_boundaries(
            nucleus_to_cell_map={0: "c1", 1: "ghost"},
            nucleus_count=np.array([1, 1, 1], dtype=np.int64),
        )

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        with pytest.warns(UserWarning, match="inconsistent with the cell table"):
            loaded = _read_boundaries_from_celldata_zarr(path)

        assert loaded.nucleus_to_cell_map is None
        assert loaded.nucleus_count is None
        # the rest of the object survives intact
        assert list(loaded.cell_names.compute()) == ["c1", "c2", "c3"]
        np.testing.assert_array_equal(
            loaded.seg_mask_value.compute(), np.array([1, 2, 3], dtype=np.uint32)
        )

    def test_load_keeps_consistent_map_and_count(self, tmp_path):
        boundaries = _create_boundaries(
            nucleus_to_cell_map={0: "c1", 1: "c1", 2: "c2"},
            nucleus_count=np.array([2, 1, 0], dtype=np.int64),
        )

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = _read_boundaries_from_celldata_zarr(path)

        assert loaded.nucleus_to_cell_map == {0: "c1", 1: "c1", 2: "c2"}
        np.testing.assert_array_equal(loaded.nucleus_count, np.array([2, 1, 0], dtype=np.int64))

    def test_load_converts_legacy_position_based_format(self, tmp_path):
        """Load-bearing regression for the compatibility requirement: a store written
        by pre-0.12.0b7 InSituPy (a flat [[nucleus_idx, row_position]] array) must
        still load correctly, converted transparently to the name-based
        representation. BoundariesData.save() can no longer produce this shape, so the
        legacy array is hand-written directly into the store."""
        boundaries = _create_boundaries()  # cell_names = ("c1", "c2", "c3")
        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        # nucleus 0, 1 -> row 0 ("c1"); nucleus 2 -> row 1 ("c2")
        _write_legacy_nucleus_map_to_store(path, {0: 0, 1: 0, 2: 1})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = _read_boundaries_from_celldata_zarr(path)

        assert loaded.nucleus_to_cell_map == {0: "c1", 1: "c1", 2: "c2"}

    def test_load_drops_inconsistent_legacy_position_based_map(self, tmp_path):
        """A legacy on-disk map sized for an original (pre-filter) cell count - the
        demo's stale identity-map shape - must still be detected and dropped."""
        boundaries = _create_boundaries()  # 3 cells on disk
        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        # positions sized for an original 5-cell table; only 3 cells exist now
        _write_legacy_nucleus_map_to_store(path, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

        with pytest.warns(UserWarning, match="legacy position-based format"):
            loaded = _read_boundaries_from_celldata_zarr(path)

        assert loaded.nucleus_to_cell_map is None
        assert loaded.nucleus_count is None


class TestSaveRejectsZarrZip:
    """The per-store `.zarr.zip` write path has been removed (never worked on
    zarr 3.2.1). `.save()` must now reject it cleanly instead of crashing
    mid-write with zipfile.BadZipFile."""

    def test_zarr_zip_path_raises_invalid_file_type_error(self, tmp_path):
        boundaries = _create_boundaries()
        path = tmp_path / "boundaries.zarr.zip"

        with pytest.raises(InvalidFileTypeError):
            boundaries.save(path=path)


class TestMaskRoundtripThroughReader:
    """Regression for the non-pyramid read crash: masks saved by BoundariesData.save() must load
    back through _read_boundaries_from_celldata_zarr with content intact. The reader's mask branch
    was previously exercised only for metadata, never for the mask arrays themselves."""

    def test_cell_mask_roundtrips_through_reader(self, tmp_path):
        mask = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32)
        boundaries = BoundariesData(cell_names=["c1", "c2", "c3"], seg_mask_value=[1, 2, 3])
        boundaries.add_boundaries(cell_boundaries=mask, nuclei_boundaries=None, pixel_size=1)

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path)

        loaded = _read_boundaries_from_celldata_zarr(path)
        # level 0 of the pyramid is the full-resolution mask
        np.testing.assert_array_equal(loaded["cells"][0].compute(), mask)

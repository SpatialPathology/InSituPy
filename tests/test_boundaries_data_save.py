"""Tests for BoundariesData.save() — regression coverage for the dask
to_zarr(..., zarr_array_kwargs=...) API-incompatibility fix.

See .log/reports/260710/boundaries-zarr-save-fix/report-boundaries-zarr-save-fix.md
"""

import dask.array as da
import numpy as np
import pytest

from insitupy._exceptions import InvalidFileTypeError
from insitupy.containers.boundaries_data import BoundariesData


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


class TestSaveRoundtripsAllMetadataArrays:
    def test_save_roundtrips_all_metadata_arrays(self, tmp_path):
        # Xenium v2.0+ multinucleated path: nucleus_to_cell_map/nucleus_count both set
        boundaries = _create_boundaries(
            nucleus_to_cell_map={0: 0, 1: 0, 2: 1},
            nucleus_count=np.array([2, 1, 0], dtype=np.int64),
        )

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path, save_as_pyramid=False)

        cell_names = da.from_zarr(path, component="cell_names").compute()
        seg_mask_value = da.from_zarr(path, component="seg_mask_value").compute()
        nucleus_map_arr = da.from_zarr(path, component="nucleus_to_cell_map").compute()
        nucleus_count = da.from_zarr(path, component="nucleus_count").compute()

        assert list(cell_names) == ["c1", "c2", "c3"]
        np.testing.assert_array_equal(seg_mask_value, np.array([1, 2, 3], dtype=np.uint32))
        assert {int(row[0]): int(row[1]) for row in nucleus_map_arr} == {0: 0, 1: 0, 2: 1}
        np.testing.assert_array_equal(nucleus_count, np.array([2, 1, 0], dtype=np.int64))


class TestSaveOverwritesOnSecondCall:
    def test_save_overwrites_on_second_call(self, tmp_path):
        # No add_boundaries() call here: isolates the overwrite=True
        # semantics of the 4 metadata-array writes from the mask-array
        # writes, which are covered separately below.
        path = tmp_path / "boundaries.zarr"

        boundaries = BoundariesData(cell_names=["c1", "c2", "c3"], seg_mask_value=[1, 2, 3])
        boundaries.save(path=path, save_as_pyramid=False)

        boundaries2 = BoundariesData(cell_names=["x1", "x2", "x3"], seg_mask_value=[4, 5, 6])
        # Must not raise (ContainsArrayError) on the second save to the same path.
        boundaries2.save(path=path, save_as_pyramid=False)

        cell_names = da.from_zarr(path, component="cell_names").compute()
        seg_mask_value = da.from_zarr(path, component="seg_mask_value").compute()
        assert list(cell_names) == ["x1", "x2", "x3"]
        np.testing.assert_array_equal(seg_mask_value, np.array([4, 5, 6], dtype=np.uint32))


class TestSaveWithMasksOverwritesOnSecondCall:
    """Regression coverage: re-saving boundaries *with masks* to the same
    non-zipped path used to raise zarr.errors.ContainsArrayError because the
    mask-array writes did not pass overwrite=True (unlike the 4 metadata
    arrays, which already did)."""

    @pytest.mark.parametrize("save_as_pyramid", [False, True])
    def test_second_save_with_masks_wins(self, tmp_path, save_as_pyramid):
        path = tmp_path / "boundaries.zarr"
        mask1 = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint32)
        mask2 = np.array([[0, 4, 5], [6, 0, 0]], dtype=np.uint32)

        boundaries = BoundariesData(cell_names=["c1", "c2", "c3"], seg_mask_value=[1, 2, 3])
        boundaries.add_boundaries(cell_boundaries=mask1, nuclei_boundaries=None, pixel_size=1)
        boundaries.save(path=path, save_as_pyramid=save_as_pyramid)

        boundaries2 = BoundariesData(cell_names=["x1", "x2", "x3"], seg_mask_value=[4, 5, 6])
        boundaries2.add_boundaries(cell_boundaries=mask2, nuclei_boundaries=None, pixel_size=2)
        # Must not raise (ContainsArrayError) on the second save to the same path.
        boundaries2.save(path=path, save_as_pyramid=save_as_pyramid)

        comp = "masks/cells/0" if save_as_pyramid else "masks/cells"
        mask_on_disk = da.from_zarr(path, component=comp).compute()
        cell_names = da.from_zarr(path, component="cell_names").compute()
        np.testing.assert_array_equal(mask_on_disk, mask2)
        assert list(cell_names) == ["x1", "x2", "x3"]


class TestSaveWithoutNucleusMetadata:
    def test_save_without_nucleus_metadata(self, tmp_path):
        boundaries = _create_boundaries(nucleus_to_cell_map=None, nucleus_count=None)

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path, save_as_pyramid=False)

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


class TestSaveRejectsZarrZip:
    """The per-store `.zarr.zip` write path has been removed (never worked on
    zarr 3.2.1). `.save()` must now reject it cleanly instead of crashing
    mid-write with zipfile.BadZipFile."""

    def test_zarr_zip_path_raises_invalid_file_type_error(self, tmp_path):
        boundaries = _create_boundaries()
        path = tmp_path / "boundaries.zarr.zip"

        with pytest.raises(InvalidFileTypeError):
            boundaries.save(path=path, save_as_pyramid=False)

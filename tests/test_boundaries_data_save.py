"""Tests for BoundariesData.save() — regression coverage for the dask
to_zarr(..., zarr_array_kwargs=...) API-incompatibility fix.

See .log/reports/260710/boundaries-zarr-save-fix/report-boundaries-zarr-save-fix.md
"""

import dask.array as da
import numpy as np

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
        # No add_boundaries() call here: this isolates the overwrite=True
        # semantics of the 4 metadata-array writes (the target of this fix)
        # from the pre-existing, out-of-scope ContainsArrayError on
        # re-saving mask arrays to the same non-zipped path (see
        # .log/backlog.md — "re-saving boundaries to the same non-zipped
        # path fails on the mask arrays").
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


class TestSaveWithoutNucleusMetadata:
    def test_save_without_nucleus_metadata(self, tmp_path):
        boundaries = _create_boundaries(nucleus_to_cell_map=None, nucleus_count=None)

        path = tmp_path / "boundaries.zarr"
        boundaries.save(path=path, save_as_pyramid=False)

        import zarr
        store = zarr.open(path, mode="r")
        assert "nucleus_to_cell_map" not in store
        assert "nucleus_count" not in store

"""Labels-native foreign-store import tests (WP4).

Uses `make_foreign_labels_native_sdata` - a hand-built SpatialData object with no
InSituPy dialect descriptor, mimicking spatialdata-io Xenium output - and exercises
`convert_from_foreign_spatialdata` (the public, hardened successor to
`_convert_from_spatialdata_manual`). See
.log/reports/260713/wp4-labels-native-import/report-wp4-labels-native-import.md.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

pytest.importorskip("spatialdata")

from spatialdata import SpatialData  # noqa: E402
from spatialdata.models import Labels2DModel, TableModel  # noqa: E402
from spatialdata.transformations import Identity, Scale  # noqa: E402
from xarray import DataArray  # noqa: E402

from insitupy._core.data import InSituData  # noqa: E402
from insitupy.spatialdata import convert_from_foreign_spatialdata  # noqa: E402
from tests.spatialdata_fixtures import make_foreign_labels_native_sdata  # noqa: E402


class TestSegMaskValueTakenFromInstanceKey:
    def test_seg_mask_value_taken_from_instance_key(self):
        """Real failure-mode: the fixture's ids are deliberately non-contiguous,
        so this only passes if seg_mask_value is read from the table's own
        instance_key column, not fabricated as arange(1, n+1)."""
        sdata = make_foreign_labels_native_sdata(contiguous_ids=False, n_cells=6)

        data = convert_from_foreign_spatialdata(
            sdata, table_key="table", cell_boundaries_data=("cell_labels", 1.0),
        )

        expected = list(sdata.tables["table"].obs["cell_id"])
        assert list(data.cells["main"].boundaries.seg_mask_value.compute()) == expected


class TestCellOnlySegmentationImportsWithoutNucleus:
    def test_cell_only_segmentation_imports_without_nucleus(self):
        """Pre-fix: comparing cell/nucleus pixel sizes before a None check raised
        TypeError whenever only cell boundaries were given - the ordinary case for
        a segmentation with no nucleus channel."""
        sdata = make_foreign_labels_native_sdata(with_nucleus=False, n_cells=4)

        data = convert_from_foreign_spatialdata(
            sdata, table_key="table", cell_boundaries_data=("cell_labels", 1.0),
        )  # must not raise

        assert data.cells["main"].boundaries is not None


class TestIndependentPixelSizesForCellAndNucleus:
    def test_independent_pixel_sizes_are_honoured(self):
        """Real failure-mode: a foreign store's cell/nucleus label rasters are not
        guaranteed to share a resolution, unlike InSituPy's own exporter."""
        sdata = make_foreign_labels_native_sdata(n_cells=4, nucleus_pixel_size=0.5)

        data = convert_from_foreign_spatialdata(
            sdata, table_key="table",
            cell_boundaries_data=("cell_labels", 1.0),
            nucleus_boundaries_data=("nucleus_labels", 0.5),
        )

        boundaries = data.cells["main"].boundaries
        assert boundaries.metadata["cells"]["pixel_size"] == 1.0
        assert boundaries.metadata["nuclei"]["pixel_size"] == 0.5


class TestIdentityTransformImageImports:
    def test_identity_transform_image_imports(self):
        sdata = make_foreign_labels_native_sdata()

        data = convert_from_foreign_spatialdata(
            sdata, image_data=("morphology", 1.0), table_key="table",
            cell_boundaries_data=("cell_labels", 1.0),
        )

        assert data.images["morphology"] is not None
        assert data.images.metadata["morphology"]["pixel_size"] == 1.0


class TestSingleScaleImageImports:
    def test_single_scale_image_imports(self):
        """Exercises the non-pyramidal (img_data.scale0 AttributeError, now
        _get_base_resolution_array's plain-DataArray branch) path against a real
        single-scale foreign raster, not just the code path's presence."""
        n_cells = 6
        sdata = make_foreign_labels_native_sdata(n_cells=n_cells)

        data = convert_from_foreign_spatialdata(
            sdata, image_data=("morphology", 1.0), table_key="table",
            cell_boundaries_data=("cell_labels", 1.0),
        )

        img = data.images["morphology"]
        arr = img[0] if isinstance(img, list) else img
        assert arr.shape == (n_cells * 4, n_cells * 4)


class TestShortDocumentedCallProducesValidInsitudata:
    def test_short_documented_call_produces_valid_insitudata(self):
        """The full auto-detection chain end to end: cells_key/cell_boundaries_data,
        seg_mask_value, and centroids are all derived from the table's own
        spatialdata_attrs with no other arguments."""
        sdata = make_foreign_labels_native_sdata()

        data = convert_from_foreign_spatialdata(sdata)

        assert isinstance(data, InSituData)
        assert data.cells["main"].is_synced


class TestCoordinateSystemSelection:
    def test_explicit_coordinate_system_overrides_global(self):
        """Real failure-mode: a silent wrong pick among multiple coordinate systems
        is a correctness bug. Hand-built (not the shared fixture) - a single label
        element carrying two coordinate systems with different Scale factors."""
        n_cells = 3
        size = n_cells * 4
        instance_ids = np.arange(1, n_cells + 1)

        cell_mask = np.zeros((size, size), dtype=np.uint32)
        for i, value in enumerate(instance_ids):
            cell_mask[i * 4, i * 4] = value
        cell_mask_arr = DataArray(cell_mask, dims=("y", "x"))
        labels = {
            "cell_labels": Labels2DModel.parse(
                cell_mask_arr,
                transformations={
                    "global": Identity(),
                    "aligned": Scale([0.3, 0.3], axes=("x", "y")),
                },
            )
        }

        obs = pd.DataFrame({
            "cell_id": instance_ids,
            "region": pd.Categorical(["cell_labels"] * n_cells),
        })
        adata = AnnData(
            X=np.random.default_rng(0).random((n_cells, 2)),
            obs=obs,
            var=pd.DataFrame(index=["gene_0", "gene_1"]),
        )
        table = TableModel.parse(adata, region="cell_labels", region_key="region", instance_key="cell_id")
        sdata = SpatialData(labels=labels, tables={"table": table})

        data = convert_from_foreign_spatialdata(sdata, table_key="table", coordinate_system="aligned")

        assert data.cells["main"].boundaries.metadata["cells"]["pixel_size"] == pytest.approx(0.3)

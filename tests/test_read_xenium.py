import pytest

from insitupy import InSituData
from insitupy.datasets import (xenium_test_dataset_v2_mm,
                               xenium_test_dataset_v2_nucex,
                               xenium_test_dataset_v3_mm,
                               xenium_test_dataset_v3_nucex,
                               xenium_test_dataset_v4_mm,
                               xenium_test_dataset_v4_nucex,
                               xenium_test_dataset_v4_protein)
from insitupy.io.data import read_xenium


@pytest.fixture(scope="module")
def v2_mm_data():
    """Download and return v2 multimodal segmentation test dataset."""
    return xenium_test_dataset_v2_mm(overwrite=False)


@pytest.fixture(scope="module")
def v2_nucex_data():
    """Download and return v2 nuclear expansion test dataset."""
    return xenium_test_dataset_v2_nucex(overwrite=False)


@pytest.fixture(scope="module")
def v3_mm_data():
    """Download and return v3 multimodal segmentation test dataset."""
    return xenium_test_dataset_v3_mm(overwrite=False)


@pytest.fixture(scope="module")
def v3_nucex_data():
    """Download and return v3 nuclear expansion test dataset."""
    return xenium_test_dataset_v3_nucex(overwrite=False)


@pytest.fixture(scope="module")
def v4_nucex_data():
    """Download and return v4 nuclear expansion test dataset."""
    return xenium_test_dataset_v4_nucex(overwrite=False)


@pytest.fixture(scope="module")
def v4_mm_data():
    """Download and return v4 multimodal segmentation test dataset."""
    return xenium_test_dataset_v4_mm(overwrite=False)


@pytest.fixture(scope="module")
def v4_protein_data():
    """Download and return v4 protein expression test dataset."""
    return xenium_test_dataset_v4_protein(overwrite=False)


class TestReadXeniumV2MM:
    """Tests for Xenium v2.0.0 multimodal segmentation dataset."""

    def test_data_loads(self, v2_mm_data):
        """Test that data loads successfully."""
        assert isinstance(v2_mm_data, InSituData)

    def test_has_cells(self, v2_mm_data):
        """Test that dataset contains cells."""
        assert v2_mm_data.cells.table.n_obs > 0

    def test_has_genes(self, v2_mm_data):
        """Test that dataset contains genes."""
        assert v2_mm_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v2_mm_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v2_mm_data.cells.table.obsm
        assert v2_mm_data.cells.table.obsm["spatial"].shape[0] == v2_mm_data.cells.table.n_obs
        assert v2_mm_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumV2Nucex:
    """Tests for Xenium v2.0.0 nuclear expansion dataset."""

    def test_data_loads(self, v2_nucex_data):
        """Test that data loads successfully."""
        assert isinstance(v2_nucex_data, InSituData)

    def test_has_cells(self, v2_nucex_data):
        """Test that dataset contains cells."""
        assert v2_nucex_data.cells.table.n_obs > 0

    def test_has_genes(self, v2_nucex_data):
        """Test that dataset contains genes."""
        assert v2_nucex_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v2_nucex_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v2_nucex_data.cells.table.obsm
        assert v2_nucex_data.cells.table.obsm["spatial"].shape[0] == v2_nucex_data.cells.table.n_obs
        assert v2_nucex_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumV3MM:
    """Tests for Xenium v3.0.0 multimodal segmentation dataset."""

    def test_data_loads(self, v3_mm_data):
        """Test that data loads successfully."""
        assert isinstance(v3_mm_data, InSituData)

    def test_has_cells(self, v3_mm_data):
        """Test that dataset contains cells."""
        assert v3_mm_data.cells.table.n_obs > 0

    def test_has_genes(self, v3_mm_data):
        """Test that dataset contains genes."""
        assert v3_mm_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v3_mm_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v3_mm_data.cells.table.obsm
        assert v3_mm_data.cells.table.obsm["spatial"].shape[0] == v3_mm_data.cells.table.n_obs
        assert v3_mm_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumV3Nucex:
    """Tests for Xenium v3.0.0 nuclear expansion dataset."""

    def test_data_loads(self, v3_nucex_data):
        """Test that data loads successfully."""
        assert isinstance(v3_nucex_data, InSituData)

    def test_has_cells(self, v3_nucex_data):
        """Test that dataset contains cells."""
        assert v3_nucex_data.cells.table.n_obs > 0

    def test_has_genes(self, v3_nucex_data):
        """Test that dataset contains genes."""
        assert v3_nucex_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v3_nucex_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v3_nucex_data.cells.table.obsm
        assert v3_nucex_data.cells.table.obsm["spatial"].shape[0] == v3_nucex_data.cells.table.n_obs
        assert v3_nucex_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumV4Nucex:
    """Tests for Xenium v4.0.0 nuclear expansion dataset."""

    def test_data_loads(self, v4_nucex_data):
        """Test that data loads successfully."""
        assert isinstance(v4_nucex_data, InSituData)

    def test_has_cells(self, v4_nucex_data):
        """Test that dataset contains cells."""
        assert v4_nucex_data.cells.table.n_obs > 0

    def test_has_genes(self, v4_nucex_data):
        """Test that dataset contains genes."""
        assert v4_nucex_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v4_nucex_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v4_nucex_data.cells.table.obsm
        assert v4_nucex_data.cells.table.obsm["spatial"].shape[0] == v4_nucex_data.cells.table.n_obs
        assert v4_nucex_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumV4MM:
    """Tests for Xenium v4.0.0 multimodal segmentation dataset."""

    def test_data_loads(self, v4_mm_data):
        """Test that data loads successfully."""
        assert isinstance(v4_mm_data, InSituData)

    def test_has_cells(self, v4_mm_data):
        """Test that dataset contains cells."""
        assert v4_mm_data.cells.table.n_obs > 0

    def test_has_genes(self, v4_mm_data):
        """Test that dataset contains genes."""
        assert v4_mm_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v4_mm_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v4_mm_data.cells.table.obsm
        assert v4_mm_data.cells.table.obsm["spatial"].shape[0] == v4_mm_data.cells.table.n_obs
        assert v4_mm_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumV4Protein:
    """Tests for Xenium v4.0.0 protein expression dataset."""

    def test_data_loads(self, v4_protein_data):
        """Test that data loads successfully."""
        assert isinstance(v4_protein_data, InSituData)

    def test_has_cells(self, v4_protein_data):
        """Test that dataset contains cells."""
        assert v4_protein_data.cells.table.n_obs > 0

    def test_has_genes(self, v4_protein_data):
        """Test that dataset contains genes."""
        assert v4_protein_data.cells.table.n_vars > 0

    def test_has_spatial_coords(self, v4_protein_data):
        """Test that spatial coordinates are present."""
        assert "spatial" in v4_protein_data.cells.table.obsm
        assert v4_protein_data.cells.table.obsm["spatial"].shape[0] == v4_protein_data.cells.table.n_obs
        assert v4_protein_data.cells.table.obsm["spatial"].shape[1] == 2


class TestReadXeniumCrossVersion:
    """Cross-version comparison tests."""

    def test_all_versions_return_insitudata(
        self,
        v2_mm_data,
        v2_nucex_data,
        v3_mm_data,
        v3_nucex_data,
        v4_nucex_data,
        v4_mm_data,
        v4_protein_data,
    ):
        """Test that all versions return InSituData objects."""
        all_datasets = [
            v2_mm_data,
            v2_nucex_data,
            v3_mm_data,
            v3_nucex_data,
            v4_nucex_data,
            v4_mm_data,
            v4_protein_data,
        ]

        for data in all_datasets:
            assert isinstance(data, InSituData), "Dataset is not an InSituData object"

    def test_all_versions_have_required_structure(
        self,
        v2_mm_data,
        v2_nucex_data,
        v3_mm_data,
        v3_nucex_data,
        v4_nucex_data,
        v4_mm_data,
        v4_protein_data,
    ):
        """Test that all versions have the required data structure."""
        all_datasets = [
            v2_mm_data,
            v2_nucex_data,
            v3_mm_data,
            v3_nucex_data,
            v4_nucex_data,
            v4_mm_data,
            v4_protein_data,
        ]

        for data in all_datasets:
            assert hasattr(data, "cells"), "Missing cells attribute"
            assert hasattr(data.cells, "table"), "Missing table attribute"
            assert hasattr(data.cells.table, "n_obs"), "Missing n_obs attribute"
            assert hasattr(data.cells.table, "n_vars"), "Missing n_vars attribute"
            assert hasattr(data.cells.table, "obsm"), "Missing obsm attribute"
            assert "spatial" in data.cells.table.obsm, "Missing spatial coordinates"


class TestReadXeniumSpatialDataBackend:
    """Tests for Xenium loading through the spatialdata backend."""

    def test_spatialdata_backend_loads_when_available(self, v2_mm_data):
        """Test backend='spatialdata' path and core output structure."""
        pytest.importorskip("spatialdata", minversion="0.7.2")
        pytest.importorskip("spatialdata_io")

        data = read_xenium(v2_mm_data.path, backend="spatialdata", verbose=False)

        assert isinstance(data, InSituData)
        assert data.cells.table.n_obs > 0
        assert "spatial" in data.cells.table.obsm

    def test_spatialdata_backend_transcripts_schema(self, v2_mm_data):
        """Test transcript schema normalization from SpatialData conversion."""
        pytest.importorskip("spatialdata", minversion="0.7.2")
        pytest.importorskip("spatialdata_io")

        data = read_xenium(v2_mm_data.path, backend="spatialdata", verbose=False)

        assert data.transcripts is not None
        assert "feature_name" in data.transcripts.columns
        assert "x_location" in data.transcripts.columns
        assert "y_location" in data.transcripts.columns

    def test_roundtrip_insitudata_spatialdata_insitudata(self, v2_mm_data):
        """Test roundtrip conversion InSituData -> SpatialData -> InSituData."""
        pytest.importorskip("spatialdata", minversion="0.7.2")

        from insitupy.spatialdata import (convert_from_spatialdata,
                                          convert_to_spatialdata)

        xd_original = v2_mm_data
        sdata = convert_to_spatialdata(xd_original)

        cell_layer_key = xd_original.cells.main_key
        if cell_layer_key is None:
            cell_layer_key = next(iter(xd_original.cells.layers.keys()))
        cell_layer = xd_original.cells[cell_layer_key]

        image_data = {
            name: (
                f"IMAGES.{name}",
                xd_original.images.metadata[name]["pixel_size"],
                xd_original.images.metadata[name]["rgb"],
            )
            for name in xd_original.images.names
        }

        cells_key = f"CELLS.{cell_layer_key}.circles"
        table_key = f"CELLS.{cell_layer_key}.table"

        cell_boundaries_data = None
        nucleus_boundaries_data = None
        if cell_layer.boundaries is not None and len(cell_layer.boundaries.metadata) > 0:
            boundary_names = list(cell_layer.boundaries.metadata.keys())

            cell_boundary_name = next((n for n in boundary_names if "cell" in n.lower()), None)
            nucleus_boundary_name = next((n for n in boundary_names if "nuc" in n.lower()), None)

            if cell_boundary_name is not None:
                cell_pixel_size = cell_layer.boundaries.metadata[cell_boundary_name]["pixel_size"]
                cell_boundaries_data = (f"CELLS.{cell_layer_key}.boundaries.{cell_boundary_name}", cell_pixel_size)
            if nucleus_boundary_name is not None:
                nucleus_pixel_size = cell_layer.boundaries.metadata[nucleus_boundary_name]["pixel_size"]
                nucleus_boundaries_data = (f"CELLS.{cell_layer_key}.boundaries.{nucleus_boundary_name}", nucleus_pixel_size)

        if (
            cell_boundaries_data is None
            or nucleus_boundaries_data is None
            or not (cell_boundaries_data[0] in sdata and nucleus_boundaries_data[0] in sdata)
        ):
            cell_boundaries_data = None
            nucleus_boundaries_data = None

        xd_roundtrip = convert_from_spatialdata(
            sdata=sdata,
            image_data=image_data,
            cells_key=cells_key,
            table_key=table_key,
            cell_boundaries_data=cell_boundaries_data,
            nucleus_boundaries_data=nucleus_boundaries_data,
            transcripts_key="TRANSCRIPTS",
            slide_id=xd_original.slide_id,
            sample_id=xd_original.sample_id,
            method_name=xd_original.metadata.get("method", ""),
            verbose=False,
        )

        assert isinstance(xd_roundtrip, InSituData)
        assert xd_roundtrip.cells.table.n_obs == xd_original.cells.table.n_obs
        assert xd_roundtrip.cells.table.n_vars == xd_original.cells.table.n_vars
        assert "spatial" in xd_roundtrip.cells.table.obsm

        assert set(xd_roundtrip.images.names) == set(xd_original.images.names)

        assert xd_roundtrip.transcripts is not None
        assert "feature_name" in xd_roundtrip.transcripts.columns
        assert "x_location" in xd_roundtrip.transcripts.columns
        assert "y_location" in xd_roundtrip.transcripts.columns
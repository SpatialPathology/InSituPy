import pytest

from insitupy import InSituData
from insitupy.datasets import (
    xenium_test_dataset_v2_mm,
    xenium_test_dataset_v2_nucex,
    xenium_test_dataset_v3_mm,
    xenium_test_dataset_v3_nucex,
    xenium_test_dataset_v4_mm,
    xenium_test_dataset_v4_nucex,
    xenium_test_dataset_v4_protein,
)
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

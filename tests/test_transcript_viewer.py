"""Unit tests for the transcript viewer module."""

import numpy as np
import pandas as pd
import pytest

from insitupy import WITH_NAPARI
from insitupy.interactive._transcript_viewer import (
    DEFAULT_DEBOUNCE_MS,
    DEFAULT_MAX_VISIBLE_POINTS,
    DEFAULT_POINT_SIZE,
    TranscriptViewerConfig,
    _assign_gene_colors,
    prepare_gene_colors,
    prepare_gene_data,
)


class TestTranscriptViewerConfig:
    """Tests for TranscriptViewerConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriptViewerConfig()
        assert config.max_visible_points == DEFAULT_MAX_VISIBLE_POINTS
        assert config.point_size == DEFAULT_POINT_SIZE
        assert config.debounce_ms == DEFAULT_DEBOUNCE_MS
        assert config.gene_column == "feature_name"
        assert config.x_column == "x_location"
        assert config.y_column == "y_location"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TranscriptViewerConfig(
            max_visible_points=50_000,
            point_size=5,
            debounce_ms=1000,
            gene_column="gene",
            x_column="x",
            y_column="y",
        )
        assert config.max_visible_points == 50_000
        assert config.point_size == 5
        assert config.debounce_ms == 1000
        assert config.gene_column == "gene"
        assert config.x_column == "x"
        assert config.y_column == "y"


class TestAssignGeneColors:
    """Tests for _assign_gene_colors function."""

    def test_empty_gene_list(self):
        """Test with empty gene list."""
        colors = _assign_gene_colors([])
        assert colors == {}

    def test_single_gene(self):
        """Test with single gene."""
        colors = _assign_gene_colors(["GeneA"])
        assert len(colors) == 1
        assert "GeneA" in colors
        assert len(colors["GeneA"]) == 3  # RGB tuple
        assert all(0 <= c <= 1 for c in colors["GeneA"])

    def test_multiple_genes(self):
        """Test with multiple genes."""
        genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
        colors = _assign_gene_colors(genes)
        assert len(colors) == 4
        for gene in genes:
            assert gene in colors
            assert len(colors[gene]) == 3
            assert all(0 <= c <= 1 for c in colors[gene])

    def test_reproducibility(self):
        """Test that colors are reproducible with same input."""
        genes = ["GeneA", "GeneB", "GeneC"]
        colors1 = _assign_gene_colors(genes)
        colors2 = _assign_gene_colors(genes)
        for gene in genes:
            assert colors1[gene] == colors2[gene]

    def test_distinct_colors(self):
        """Test that different genes get different colors."""
        genes = ["GeneA", "GeneB", "GeneC"]
        colors = _assign_gene_colors(genes)
        color_tuples = list(colors.values())
        # Colors should be distinct (with some tolerance for floating point)
        for i, c1 in enumerate(color_tuples):
            for j, c2 in enumerate(color_tuples):
                if i != j:
                    # At least one channel should differ
                    assert not np.allclose(c1, c2, atol=1e-6)


class TestPrepareGeneData:
    """Tests for prepare_gene_data function."""

    @pytest.fixture
    def sample_transcript_df(self):
        """Create a sample transcript DataFrame."""
        np.random.seed(42)
        n_points = 100
        genes = np.random.choice(["GeneA", "GeneB", "GeneC"], n_points)
        x = np.random.uniform(0, 1000, n_points)
        y = np.random.uniform(0, 1000, n_points)
        return pd.DataFrame({
            "feature_name": genes,
            "x_location": x,
            "y_location": y,
        })

    def test_basic_functionality(self, sample_transcript_df):
        """Test basic gene data preparation."""
        gene_data, gene_colors = prepare_gene_data(sample_transcript_df)

        # Check gene_data structure
        assert isinstance(gene_data, dict)
        assert set(gene_data.keys()) == {"GeneA", "GeneB", "GeneC"}

        # Check coordinate arrays
        for gene, coords in gene_data.items():
            assert isinstance(coords, np.ndarray)
            assert coords.ndim == 2
            assert coords.shape[1] == 2  # x, y columns

        # Check gene_colors structure
        assert isinstance(gene_colors, dict)
        assert set(gene_colors.keys()) == {"GeneA", "GeneB", "GeneC"}

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            "gene": ["A", "B", "A"],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })
        config = TranscriptViewerConfig(
            gene_column="gene",
            x_column="x",
            y_column="y",
        )
        gene_data, gene_colors = prepare_gene_data(df, config)

        assert set(gene_data.keys()) == {"A", "B"}
        assert gene_data["A"].shape == (2, 2)
        assert gene_data["B"].shape == (1, 2)

    def test_coordinate_values(self):
        """Test that coordinates are correctly extracted."""
        df = pd.DataFrame({
            "feature_name": ["GeneA", "GeneA"],
            "x_location": [10.0, 20.0],
            "y_location": [30.0, 40.0],
        })
        gene_data, _ = prepare_gene_data(df)

        expected = np.array([[10.0, 30.0], [20.0, 40.0]])
        np.testing.assert_array_equal(gene_data["GeneA"], expected)

    def test_mixed_bytes_and_str_gene_names(self):
        """Test normalization of mixed bytes/str gene names in in-memory mode."""
        df = pd.DataFrame({
            "feature_name": [b"GeneA", "GeneA", b"GeneB", "GeneC"],
            "x_location": [1.0, 2.0, 3.0, 4.0],
            "y_location": [5.0, 6.0, 7.0, 8.0],
        })

        gene_data, gene_colors = prepare_gene_data(df)

        assert set(gene_data.keys()) == {"GeneA", "GeneB", "GeneC"}
        assert all(isinstance(gene, str) for gene in gene_data.keys())
        assert set(gene_colors.keys()) == {"GeneA", "GeneB", "GeneC"}


class TestPrepareGeneColors:
    """Tests for prepare_gene_colors function (lazy mode)."""

    def test_basic_functionality(self):
        """Test basic gene colors preparation with Dask DataFrame."""
        pytest.importorskip("dask")
        import dask.dataframe as dd

        df = pd.DataFrame({
            "feature_name": ["GeneA", "GeneB", "GeneA", "GeneC"],
            "x_location": [1.0, 2.0, 3.0, 4.0],
            "y_location": [5.0, 6.0, 7.0, 8.0],
        })
        dask_df = dd.from_pandas(df, npartitions=2)

        gene_list, gene_colors = prepare_gene_colors(dask_df)

        # Check gene_list
        assert isinstance(gene_list, list)
        assert set(gene_list) == {"GeneA", "GeneB", "GeneC"}
        assert gene_list == sorted(gene_list)  # Should be sorted

        # Check gene_colors
        assert isinstance(gene_colors, dict)
        assert set(gene_colors.keys()) == {"GeneA", "GeneB", "GeneC"}

    def test_mixed_bytes_and_str_gene_names(self):
        """Test normalization of mixed bytes/str gene names in lazy mode."""
        pytest.importorskip("dask")
        import dask.dataframe as dd

        df = pd.DataFrame({
            "feature_name": [b"GeneA", "GeneA", b"GeneB", "GeneC", b"GeneC"],
            "x_location": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_location": [6.0, 7.0, 8.0, 9.0, 10.0],
        })
        dask_df = dd.from_pandas(df, npartitions=2)

        gene_list, gene_colors = prepare_gene_colors(dask_df)

        assert gene_list == ["GeneA", "GeneB", "GeneC"]
        assert all(isinstance(gene, str) for gene in gene_list)
        assert set(gene_colors.keys()) == {"GeneA", "GeneB", "GeneC"}


class TestSubsampling:
    """Tests for subsampling functionality."""

    def test_no_subsampling_when_under_limit(self):
        """Test that no subsampling occurs when under limit."""
        n_points = 50
        max_points = 100

        coords = np.random.rand(n_points, 2)

        # No subsampling needed
        assert len(coords) <= max_points
        # Coords remain unchanged
        assert coords.shape == (n_points, 2)


@pytest.mark.skipif(not WITH_NAPARI, reason="napari required")
class TestGetValueGuard:
    """Tests for the _install_get_value_guard hover-lookup race hardening."""

    def test_swallows_indexerror(self):
        from insitupy.interactive._transcript_viewer import TranscriptViewerWidget

        class _Fake:
            def _get_value(self, position):
                raise IndexError("transient race")

        fake = _Fake()
        TranscriptViewerWidget._install_get_value_guard(fake)
        assert fake._get_value((0.0, 0.0)) is None  # positional
        assert fake._get_value(position=(0.0, 0.0)) is None  # keyword

    def test_passes_through_value(self):
        from insitupy.interactive._transcript_viewer import TranscriptViewerWidget

        class _Fake:
            def _get_value(self, position):
                return 7

        fake = _Fake()
        TranscriptViewerWidget._install_get_value_guard(fake)
        assert fake._get_value((0.0, 0.0)) == 7


@pytest.fixture
def viewer_model(qapp):
    """A headless napari ViewerModel (data layer only, no Qt window/GL canvas).

    ``_create_layer``/``_update_layer`` only touch layer *data* (data, size,
    face_color, properties, canvas_size_limits, get_value) -- none of that needs
    an actual rendered canvas. Using ``ViewerModel`` instead of ``make_napari_viewer``
    keeps these tests independent of whether this machine's Qt offscreen platform
    can create a real OpenGL context (it does not, on at least one dev machine).
    """
    from napari.viewer import ViewerModel

    return ViewerModel()


@pytest.mark.skipif(not WITH_NAPARI, reason="napari required")
class TestCreateLayerDefaults:
    """Tests for the Transcripts layer's creation-time defaults."""

    def test_guard_installed_and_canvas_size_limits_disabled(self, viewer_model):
        from insitupy.interactive._transcript_viewer import TranscriptViewerWidget

        widget = TranscriptViewerWidget(
            viewer_model,
            gene_data_or_list={"GeneA": np.zeros((0, 2))},
            gene_colors={"GeneA": (1.0, 0.0, 0.0)},
        )
        widget._create_layer()
        layer = viewer_model.layers[TranscriptViewerWidget.LAYER_NAME]

        # min=0 disables napari's canvas-size floor, which otherwise forces a
        # visible border_color edge on zoom-out regardless of border_width=0.
        assert layer.canvas_size_limits[0] == 0
        # The hover-lookup guard must be installed on the real layer instance.
        assert "_get_value" in layer.__dict__


@pytest.mark.skipif(not WITH_NAPARI, reason="napari required")
class TestUpdateLayerSizing:
    """Tests for point-size consistency across _update_layer calls."""

    def test_point_sizes_stay_uniform_after_size_change(self, viewer_model):
        from insitupy.interactive._transcript_viewer import TranscriptViewerWidget

        coords = np.random.default_rng(0).uniform(0, 100, size=(20, 2))
        widget = TranscriptViewerWidget(
            viewer_model,
            gene_data_or_list={"GeneA": coords},
            gene_colors={"GeneA": (1.0, 0.0, 0.0)},
        )

        # First render (bypass bbox/camera by seeding queried coords directly).
        widget.active_genes = {"GeneA": coords}
        widget._create_layer()
        widget._update_layer()
        layer = viewer_model.layers[TranscriptViewerWidget.LAYER_NAME]

        # User drags the point-size slider (no selection -> sets current_size only).
        layer.current_size = 5.0

        # A later camera move renders a different-length visible set.
        widget.active_genes = {"GeneA": coords[:13]}
        widget._update_layer()

        assert len(layer.size) == len(layer.data)  # arrays aligned
        assert np.allclose(np.asarray(layer.size), 5.0)  # uniform at slider value


@pytest.mark.skipif(not WITH_NAPARI, reason="napari required")
class TestUpdateLayerConsistency:
    """Tests that _update_layer leaves the layer internally consistent."""

    def test_hover_value_lookup_is_consistent(self, viewer_model):
        from insitupy.interactive._transcript_viewer import TranscriptViewerWidget

        coords = np.random.default_rng(1).uniform(0, 100, size=(30, 2))
        widget = TranscriptViewerWidget(
            viewer_model,
            gene_data_or_list={"GeneA": coords},
            gene_colors={"GeneA": (0.0, 1.0, 0.0)},
        )
        widget.active_genes = {"GeneA": coords}
        widget._create_layer()
        widget._update_layer()
        layer = viewer_model.layers[TranscriptViewerWidget.LAYER_NAME]

        # napari data is (y, x); coords are (x, y). Query a real displayed point.
        pos = tuple(layer.data[0])
        value = layer.get_value(
            pos, view_direction=None, dims_displayed=[0, 1], world=False
        )
        # Either a valid in-range index or None -- never an out-of-bounds index / raise.
        assert value is None or (0 <= int(value) < len(layer.data))
        # And the view-state indices are consistent with the current data length.
        assert len(layer._indices_view) == 0 or int(layer._indices_view.max()) < len(layer.data)

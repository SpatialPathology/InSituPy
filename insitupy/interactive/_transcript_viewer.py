"""
Transcript Viewer Widget for InSituPy.

A scalable napari widget for visualizing transcript points (10-50M+).

Features:
    - In-memory spatial queries with numpy boolean indexing
    - Debounced camera updates (configurable interval)
    - Searchable gene selector with autocomplete
    - Per-gene color assignment using shuffled HSV colormap
    - Max point limit with random subsampling
    - Hover display showing gene names
    - Memory management for lazy loading mode

Two loading modes:
    - In-memory mode (default): All gene coordinates loaded upfront.
      Fast queries (~1GB memory for 50M points).
    - Lazy mode: Gene coordinates loaded on-demand from Dask DataFrame.
      Lower memory (~20MB per active gene), ~1.5s delay per gene.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import colormaps

from insitupy._constants import WITH_NAPARI

if TYPE_CHECKING:
    import dask.dataframe as dd
    import napari
    import pandas as pd

logger = logging.getLogger(__name__)

# === Configuration ===
# Default values that can be overridden via TranscriptViewerConfig
DEFAULT_MAX_VISIBLE_POINTS = 100_000
DEFAULT_POINT_SIZE = 0.2
DEFAULT_DEBOUNCE_MS = 500


def _normalize_gene_name(gene: object) -> str:
    """Normalize transcript gene names to Python strings.

    Qt widgets expect text entries to be ``str``. Some transcript datasets
    store gene names as raw bytes in parquet-backed columns.
    """
    if isinstance(gene, str):
        return gene
    if isinstance(gene, bytes):
        try:
            return gene.decode("utf-8")
        except UnicodeDecodeError:
            return gene.decode("utf-8", errors="replace")
    return str(gene)


class TranscriptViewerConfig:
    """Configuration class for the TranscriptViewerWidget.

    Attributes:
        max_visible_points: Maximum number of points to display before subsampling.
        point_size: Size of the transcript points in the viewer.
        debounce_ms: Debounce interval in milliseconds for camera updates.
        gene_column: Column name for gene/feature names in the transcript DataFrame.
        x_column: Column name for x coordinates.
        y_column: Column name for y coordinates.
    """

    def __init__(
        self,
        max_visible_points: int = DEFAULT_MAX_VISIBLE_POINTS,
        point_size: int = DEFAULT_POINT_SIZE,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        gene_column: str = "feature_name",
        x_column: str = "x_location",
        y_column: str = "y_location",
    ):
        """Initialize transcript viewer configuration.

        Args:
            max_visible_points: Maximum points to display before subsampling.
            point_size: Size of transcript points in pixels.
            debounce_ms: Debounce delay for camera updates in milliseconds.
            gene_column: DataFrame column name for gene/feature names.
            x_column: DataFrame column name for x coordinates.
            y_column: DataFrame column name for y coordinates.
        """
        self.max_visible_points = max_visible_points
        self.point_size = point_size
        self.debounce_ms = debounce_ms
        self.gene_column = gene_column
        self.x_column = x_column
        self.y_column = y_column


def prepare_gene_data(
    df: Union["pd.DataFrame", "dd.DataFrame"],
    config: Optional[TranscriptViewerConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float, float]]]:
    """Prepare gene data from a DataFrame (in-memory mode).

    Loads all gene coordinates into memory upfront for fast spatial queries.
    Use this for datasets that fit in memory (< ~1GB, ~50M points).

    Args:
        df: DataFrame with transcript data. Must contain columns for gene name,
            x coordinate, and y coordinate (configurable via config).
            If Dask DataFrame, will be computed to pandas.
        config: Configuration object specifying column names.
            If None, uses default column names.

    Returns:
        Tuple containing:
            - gene_data: Dict mapping gene names to coordinate arrays of shape (N, 2).
            - gene_colors: Dict mapping gene names to RGB color tuples.
    """
    if config is None:
        config = TranscriptViewerConfig()

    # Compute if dask DataFrame
    if hasattr(df, "compute"):
        logger.info("Computing Dask DataFrame to pandas...")
        df = df[[config.gene_column, config.x_column, config.y_column]].compute()

    # Normalize gene names to strings for UI compatibility
    df = df.copy()
    df[config.gene_column] = df[config.gene_column].map(_normalize_gene_name)

    # Get unique genes and assign colors
    genes = sorted(df[config.gene_column].unique())
    n_genes = len(genes)
    logger.info("Found %d unique genes", n_genes)

    gene_colors = _assign_gene_colors(genes)

    # Build per-gene coordinate arrays
    gene_data: Dict[str, np.ndarray] = {}
    grouped = df.groupby(config.gene_column)
    for gene, group in grouped:
        gene_data[gene] = group[[config.x_column, config.y_column]].values

    return gene_data, gene_colors


def prepare_gene_colors(
    dask_df: "dd.DataFrame",
    config: Optional[TranscriptViewerConfig] = None,
) -> Tuple[List[str], Dict[str, Tuple[float, float, float]]]:
    """Prepare gene list and colors from a Dask DataFrame (lazy mode).

    Only loads unique gene names, not coordinates. Coordinates are loaded
    on-demand when genes are added to the viewer.

    Args:
        dask_df: Dask DataFrame with transcript data. Must contain the gene
            column specified in config. The DataFrame is NOT fully computed.
        config: Configuration object specifying column names.

    Returns:
        Tuple containing:
            - gene_list: Sorted list of unique gene names.
            - gene_colors: Dict mapping gene names to RGB color tuples.
    """
    if config is None:
        config = TranscriptViewerConfig()

    # Only compute unique gene names
    logger.info("Extracting unique gene names from Dask DataFrame...")
    genes_raw = dask_df[config.gene_column].unique().compute().tolist()
    genes = sorted({_normalize_gene_name(gene) for gene in genes_raw})
    n_genes = len(genes)
    logger.info("Found %d unique genes", n_genes)

    gene_colors = _assign_gene_colors(genes)

    return genes, gene_colors


def _assign_gene_colors(
    genes: List[str],
) -> Dict[str, Tuple[float, float, float]]:
    """Assign distinct colors to genes using shuffled HSV colormap.

    Colors are shuffled so that alphabetically adjacent genes get
    visually distinct colors.

    Args:
        genes: List of gene names to assign colors to.

    Returns:
        Dict mapping gene names to RGB color tuples (values 0-1).
    """
    n_genes = len(genes)
    if n_genes == 0:
        return {}

    # Use HSV colormap for maximum color distinction
    cmap = colormaps["hsv"]

    # Create shuffled color indices so neighboring genes get different colors
    color_indices = np.arange(n_genes)
    rng = np.random.default_rng(42)  # reproducible shuffle
    rng.shuffle(color_indices)

    gene_colors = {
        gene: cmap(color_indices[i] / n_genes)[:3]
        for i, gene in enumerate(genes)
    }

    return gene_colors


if WITH_NAPARI:
    import napari
    from matplotlib.lines import Line2D
    from napari.utils.notifications import show_info, show_warning
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtWidgets import (QCompleter, QHBoxLayout, QLabel, QLineEdit,
                                QListWidget, QListWidgetItem, QPushButton,
                                QVBoxLayout, QWidget)

    from insitupy.interactive._configs import ViewerConfig

    class TranscriptViewerWidget(QWidget):
        """Napari dock widget for viewing transcript points by gene.

        Supports two loading modes:

        In-memory mode (lazy_loading=False, default):
            All gene coordinates are pre-loaded into memory. Fast bbox queries,
            but requires ~1GB for 50M points.

            Usage:
                gene_data, gene_colors = prepare_gene_data(df)
                widget = TranscriptViewerWidget(viewer, gene_data, gene_colors)

        Lazy mode (lazy_loading=True):
            Gene coordinates are loaded on-demand from a Dask DataFrame when
            added to the viewer. Lower memory footprint, but ~1.5s delay per gene.

            Usage:
                gene_list, gene_colors = prepare_gene_colors(dask_df)
                widget = TranscriptViewerWidget(
                    viewer, gene_list, gene_colors,
                    lazy_loading=True, dask_df=dask_df
                )

        Attributes:
            viewer: The napari viewer instance.
            gene_colors: Dict mapping gene names to RGB color tuples.
            active_genes: Dict mapping active gene names to their visible coordinates.
            config: Configuration object for viewer settings.
            viewer_config: ViewerConfig for accessing the color legend canvas.
        """

        LAYER_NAME = "Transcripts"

        def __init__(
            self,
            viewer: napari.Viewer,
            gene_data_or_list: Union[Dict[str, np.ndarray], List[str]],
            gene_colors: Dict[str, Tuple[float, float, float]],
            lazy_loading: bool = False,
            dask_df: Optional["dd.DataFrame"] = None,
            config: Optional[TranscriptViewerConfig] = None,
            viewer_config: Optional[ViewerConfig] = None,
        ):
            """Initialize the transcript viewer widget.

            Args:
                viewer: The napari viewer instance.
                gene_data_or_list: In-memory mode: dict mapping gene names to
                    coordinate arrays. Lazy mode: list of gene names.
                gene_colors: Dict mapping gene names to RGB color tuples.
                lazy_loading: If True, use lazy loading mode with on-demand
                    gene loading from dask_df.
                dask_df: Required if lazy_loading=True. Dask DataFrame with
                    transcript coordinates.
                config: Configuration object. If None, uses defaults.
                viewer_config: ViewerConfig for accessing static_canvas for color legend.

            Raises:
                ValueError: If lazy_loading=True but dask_df is None.
            """
            super().__init__()
            self.viewer = viewer
            self.gene_colors = gene_colors
            self.lazy_loading = lazy_loading
            self.config = config or TranscriptViewerConfig()
            self.viewer_config = viewer_config
            self.active_genes: Dict[str, Optional[np.ndarray]] = {}
            self._gene_query_mode: str = "str"

            if lazy_loading:
                if dask_df is None:
                    raise ValueError("dask_df is required when lazy_loading=True")
                self.dask_df = dask_df
                self.gene_list = gene_data_or_list  # list of gene names
                self.gene_data: Dict[str, np.ndarray] = {}  # cache for loaded coords
                self._init_gene_query_mode()
            else:
                self.dask_df = None
                self.gene_data = gene_data_or_list  # dict of coords
                self.gene_list = sorted(gene_data_or_list.keys())

            # Cache for reference layer info (bbox optimization)
            self._ref_layer: Optional[napari.layers.Image] = None
            self._ref_layer_multiscale: bool = False
            self._ref_layer_scale: Optional[Tuple[float, float]] = None

            self._setup_ui()
            self._setup_debounce()
            self._connect_camera()
            self._connect_layer_selection()
            self._connect_layer_removal()
            self._setup_ref_layer_cache()

        def _setup_ui(self) -> None:
            """Set up the user interface elements."""
            layout = QVBoxLayout()
            self.setLayout(layout)

            # Gene search input with autocomplete
            layout.addWidget(QLabel("Search gene:"))
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Type gene name...")

            # Setup autocomplete
            self.completer = QCompleter(self.gene_list)
            self.completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.completer.setFilterMode(Qt.MatchContains)
            self.search_input.setCompleter(self.completer)
            layout.addWidget(self.search_input)

            # Add/Remove buttons
            btn_layout = QHBoxLayout()
            self.add_btn = QPushButton("Add Gene")
            self.remove_btn = QPushButton("Remove Selected")
            self.clear_btn = QPushButton("Clear All")
            btn_layout.addWidget(self.add_btn)
            btn_layout.addWidget(self.remove_btn)
            btn_layout.addWidget(self.clear_btn)
            layout.addLayout(btn_layout)

            # Active genes list (for removal selection)
            layout.addWidget(QLabel("Active genes:"))
            self.active_list = QListWidget()
            self.active_list.setMaximumHeight(100)
            layout.addWidget(self.active_list)

            # Status label
            self.status_label = QLabel("Points: 0")
            layout.addWidget(self.status_label)

            # Connect signals
            self.add_btn.clicked.connect(self._on_add_gene)
            self.remove_btn.clicked.connect(self._on_remove_gene)
            self.clear_btn.clicked.connect(self._on_clear_all)
            self.search_input.returnPressed.connect(self._on_add_gene)

        def _setup_debounce(self) -> None:
            """Set up the debounce timer for camera updates."""
            self.timer = QTimer()
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self._do_query)

        def _connect_camera(self) -> None:
            """Connect camera events for automatic updates."""
            self.viewer.camera.events.center.connect(self._on_view_change)
            self.viewer.camera.events.zoom.connect(self._on_view_change)

        def _connect_layer_selection(self) -> None:
            """Connect layer selection event to update color legend."""
            self.viewer.layers.selection.events.active.connect(
                self._on_layer_selection_change
            )

        def _on_layer_selection_change(self, event=None) -> None:
            """Update color legend when Transcripts layer is selected."""
            layer = self.viewer.layers.selection.active
            if layer is not None and layer.name == self.LAYER_NAME:
                self._update_color_legend()

        def _connect_layer_removal(self) -> None:
            """Connect layer removal event to clear UI state when layer is deleted."""
            self.viewer.layers.events.removed.connect(self._on_layer_removed)

        def _on_layer_removed(self, event) -> None:
            """Clear active genes and UI when the Transcripts layer is removed."""
            removed_layer = event.value
            if removed_layer.name == self.LAYER_NAME:
                self.active_genes.clear()
                if self.lazy_loading:
                    self.gene_data.clear()
                self._update_active_list()
                self._update_color_legend()
                self.status_label.setText("Points: 0")

        def _on_view_change(self, event=None) -> None:
            """Restart debounce timer on camera change."""
            self.timer.stop()
            self.timer.start(self.config.debounce_ms)

        def _setup_ref_layer_cache(self) -> None:
            """Cache reference layer info for efficient bbox calculations.

            This method finds the first image layer and caches its properties
            (multiscale flag, scale) to avoid repeated lookups during
            camera pan/zoom events.
            """
            image_layers = [
                layer for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Image)
            ]
            if image_layers:
                self._ref_layer = image_layers[0]
                self._ref_layer_multiscale = self._ref_layer.multiscale
                self._ref_layer_scale = self._ref_layer.scale
                logger.debug(
                    "Cached ref layer '%s': multiscale=%s, scale=%s",
                    self._ref_layer.name,
                    self._ref_layer_multiscale,
                    self._ref_layer_scale,
                )
            else:
                self._ref_layer = None
                self._ref_layer_multiscale = False
                self._ref_layer_scale = None

        def _get_bbox(self) -> Optional[Tuple[float, float, float, float]]:
            """Get current visible bounding box from the cached reference layer.

            The bounding box is returned in world coordinates (physical units),
            accounting for the image layer's scale (pixel_size) and pyramid level
            when using multiscale images.

            For multiscale/pyramidal images, corner_pixels returns coordinates
            at the current data_level. We must multiply by the downsample_factor
            to get full-resolution pixel coordinates before converting to world
            coordinates.

            Uses cached reference layer info for efficiency since this method
            is called on every camera pan/zoom event.

            Returns:
                Tuple of (xmin, xmax, ymin, ymax) in world coordinates,
                or None if no image layer found.
            """
            if self._ref_layer is None:
                # Try to find a reference layer if not cached yet
                self._setup_ref_layer_cache()
                if self._ref_layer is None:
                    show_warning("No image layer found for reference.")
                    return None

            corners = self._ref_layer.corner_pixels
            # corner_pixels is [[y_min, x_min], [y_max, x_max]] in pixel coordinates
            # at the current pyramid level

            ymin_px, xmin_px = corners[0]
            ymax_px, xmax_px = corners[1]

            # For multiscale images, corner_pixels are at the current data_level
            # We need to multiply by the downsample factor to get full-res pixels
            if self._ref_layer_multiscale:
                current_level = self._ref_layer.data_level
                downsample_factors = self._ref_layer.downsample_factors
                # downsample_factors[level] gives (y_factor, x_factor) for each level
                y_factor, x_factor = downsample_factors[current_level]
                xmin_px = xmin_px * x_factor
                xmax_px = xmax_px * x_factor
                ymin_px = ymin_px * y_factor
                ymax_px = ymax_px * y_factor

            # Convert full-resolution pixel coordinates to world coordinates
            # Use cached scale for efficiency
            scale_y, scale_x = self._ref_layer_scale
            xmin = xmin_px * scale_x
            xmax = xmax_px * scale_x
            ymin = ymin_px * scale_y
            ymax = ymax_px * scale_y

            return xmin, xmax, ymin, ymax

        def _query_bbox(
            self,
            coords: np.ndarray,
            xmin: float,
            xmax: float,
            ymin: float,
            ymax: float,
        ) -> np.ndarray:
            """Fast vectorized bounding box query.

            Args:
                coords: Array of shape (N, 2) with [x, y] coordinates.
                xmin: Minimum x coordinate of bounding box.
                xmax: Maximum x coordinate of bounding box.
                ymin: Minimum y coordinate of bounding box.
                ymax: Maximum y coordinate of bounding box.

            Returns:
                Filtered coordinate array within the bounding box.
            """
            mask = (
                (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
                (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
            )
            return coords[mask]

        def _on_add_gene(self) -> None:
            """Add gene from search input to active list."""
            gene = self.search_input.text().strip()
            if not gene:
                return
            if gene not in self.gene_list:
                show_warning(f"Gene '{gene}' not found.")
                return
            if gene in self.active_genes:
                show_info(f"Gene '{gene}' already active.")
                return

            # Lazy loading: load gene coordinates on demand
            if self.lazy_loading and gene not in self.gene_data:
                logger.info("Loading coordinates for gene '%s'...", gene)
                self._load_gene_coords(gene)

            self.active_genes[gene] = None
            self._update_active_list()
            self.search_input.clear()

            # Create the layer if it doesn't exist yet (before querying)
            if self.LAYER_NAME not in self.viewer.layers:
                self._create_layer()

            self._do_query()

        def _init_gene_query_mode(self) -> None:
            """Infer whether gene column values are stored as strings or bytes."""
            self._gene_query_mode = "str"

            try:
                sample = self.dask_df[self.config.gene_column].dropna().head(1)
            except Exception as exc:
                logger.debug(
                    "Could not infer gene query mode from column '%s'; using string mode (%s)",
                    self.config.gene_column,
                    exc,
                )
                return

            if len(sample) == 0:
                logger.debug(
                    "Gene column '%s' has no non-null sample value; using string mode",
                    self.config.gene_column,
                )
                return

            sample_value = sample.iloc[0]
            if isinstance(sample_value, (bytes, bytearray)):
                self._gene_query_mode = "bytes"

            logger.debug(
                "Initialized gene query mode for column '%s' as '%s'",
                self.config.gene_column,
                self._gene_query_mode,
            )

        def _gene_to_query_key(self, gene: str):
            """Convert normalized gene string to query key for the Dask column."""
            if self._gene_query_mode == "bytes":
                try:
                    return gene.encode("utf-8")
                except UnicodeEncodeError:
                    logger.warning(
                        "Could not UTF-8 encode gene '%s' for bytes-backed lookup; using string key",
                        gene,
                    )
                    return gene
            return gene

        def _load_gene_coords(self, gene: str) -> None:
            """Load gene coordinates from Dask DataFrame (lazy mode only).

            Args:
                gene: Gene name to load coordinates for.
            """
            query_key = self._gene_to_query_key(gene)
            subset = self.dask_df[self.dask_df[self.config.gene_column] == query_key]
            coords = subset[[self.config.x_column, self.config.y_column]].compute().values

            # Guarded fallback for mixed/unknown columns: retry only when primary lookup is empty.
            if len(coords) == 0:
                fallback_key = None
                if self._gene_query_mode == "str":
                    try:
                        fallback_key = gene.encode("utf-8")
                    except UnicodeEncodeError:
                        fallback_key = None
                elif self._gene_query_mode == "bytes":
                    fallback_key = gene

                if fallback_key is not None and fallback_key != query_key:
                    subset = self.dask_df[self.dask_df[self.config.gene_column] == fallback_key]
                    coords = subset[[self.config.x_column, self.config.y_column]].compute().values

            self.gene_data[gene] = coords
            logger.info("Loaded %d coordinates for gene '%s'", len(coords), gene)

        def _on_remove_gene(self) -> None:
            """Remove selected gene from active list."""
            current = self.active_list.currentItem()
            if current:
                gene = current.text()
                if gene in self.active_genes:
                    del self.active_genes[gene]
                    # In lazy mode, also free the cached coordinates
                    if self.lazy_loading and gene in self.gene_data:
                        del self.gene_data[gene]
                        logger.info("Freed cached coordinates for gene '%s'", gene)
                    self._update_active_list()
                    self._update_layer()
            else:
                show_info("Please select a gene to remove.")

        def _on_clear_all(self) -> None:
            """Remove all active genes."""
            self.active_genes.clear()
            # In lazy mode, also free all cached coordinates
            if self.lazy_loading:
                self.gene_data.clear()
                logger.info("Freed all cached gene coordinates")
            self._update_active_list()
            self._update_layer()

        def _update_active_list(self) -> None:
            """Sync QListWidget with active_genes dict (without coloring)."""
            self.active_list.clear()
            for gene in sorted(self.active_genes.keys()):
                item = QListWidgetItem(gene)
                self.active_list.addItem(item)

        def _update_color_legend(self) -> None:
            """Update the color legend with active gene colors."""
            if self.viewer_config is None:
                return

            static_canvas = self.viewer_config.static_canvas
            if not self.active_genes:
                # Clear the legend
                static_canvas.figure.clear()
                static_canvas.draw()
                return

            # Build mapping of gene names to colors
            import math
            mapping = {
                gene: self.gene_colors[gene]
                for gene in sorted(self.active_genes.keys())
            }

            # Calculate layout
            num_items = len(mapping)
            max_rows = 6
            ncols = math.ceil(num_items / max_rows)

            # Update figure
            static_canvas.figure.clear()
            axes = static_canvas.figure.subplots()

            # Create legend handles
            legend_handles = [
                Line2D(
                    [0], [0],
                    marker='o', color='w', label=name,
                    markerfacecolor=color, markeredgecolor='k',
                    markersize=7
                )
                for name, color in mapping.items()
            ]

            # Add legend to axis
            axes.legend(
                handles=legend_handles, loc="center", title="Transcripts",
                ncols=ncols, fontsize=8, title_fontsize=10,
                labelspacing=0.7, borderpad=0.5
            )
            axes.set_axis_off()
            static_canvas.draw()

        def _do_query(self) -> None:
            """Query visible points for all active genes."""
            if not self.active_genes:
                return

            bbox = self._get_bbox()
            if bbox is None:
                return

            for gene in self.active_genes:
                coords = self.gene_data[gene]
                visible = self._query_bbox(coords, *bbox)
                self.active_genes[gene] = visible

            self._update_layer()

        def _create_layer(self) -> None:
            """Create the Transcripts points layer.

            Only called explicitly from _on_add_gene. This ensures the layer
            is never silently recreated after the user deletes it.
            """
            if self.LAYER_NAME in self.viewer.layers:
                return
            self.viewer.add_points(
                np.empty((0, 2)),
                name=self.LAYER_NAME,
                face_color="white",
                size=self.config.point_size,
                border_width=0,
            )

        def _update_layer(self) -> None:
            """Update the Transcripts points layer if it exists.

            This method only updates an existing layer — it never creates one.
            If the user has deleted the layer, this is a no-op, so camera
            movements won't cause the layer to reappear.
            """
            # If the layer doesn't exist (user deleted it), do nothing
            if self.LAYER_NAME not in self.viewer.layers:
                return

            if not self.active_genes:
                self.viewer.layers.remove(self.LAYER_NAME)
                self.status_label.setText("Points: 0")
                self._update_color_legend()
                return

            # Combine all visible points
            all_coords = []
            all_colors = []
            all_gene_names = []

            for gene, coords in self.active_genes.items():
                if coords is not None and len(coords) > 0:
                    all_coords.append(coords)
                    color = self.gene_colors[gene]
                    all_colors.extend([color] * len(coords))
                    all_gene_names.extend([gene] * len(coords))

            if not all_coords:
                self.status_label.setText("Points: 0 (none in view)")
                self.viewer.layers[self.LAYER_NAME].data = np.empty((0, 2))
                return

            combined = np.vstack(all_coords)
            colors = np.array(all_colors)
            gene_names = np.array(all_gene_names)
            total_points = len(combined)

            # Subsample if over limit
            subsampled = False
            if total_points > self.config.max_visible_points:
                rng = np.random.default_rng()
                idx = rng.choice(
                    total_points, self.config.max_visible_points, replace=False
                )
                combined = combined[idx]
                colors = colors[idx]
                gene_names = gene_names[idx]
                subsampled = True

            # Swap x,y to y,x for napari (napari uses row, column order)
            points_yx = combined[:, ::-1]

            # Update layer
            layer = self.viewer.layers[self.LAYER_NAME]
            properties = {"gene": gene_names}
            layer.data = points_yx
            layer.face_color = colors
            layer.properties = properties

            # Update status
            if subsampled:
                self.status_label.setText(
                    f"Points: {self.config.max_visible_points:,} / "
                    f"{total_points:,} (subsampled)"
                )
            else:
                self.status_label.setText(f"Points: {total_points:,}")

            # Update color legend
            self._update_color_legend()

    def create_transcript_viewer_widget(
        viewer: napari.Viewer,
        transcripts: Union["pd.DataFrame", "dd.DataFrame"],
        lazy_loading: bool = True,
        config: Optional[TranscriptViewerConfig] = None,
        viewer_config: Optional["ViewerConfig"] = None,
    ) -> TranscriptViewerWidget:
        """Create a TranscriptViewerWidget from transcript data.

        Convenience function that prepares gene data and creates the widget.

        Args:
            viewer: The napari viewer instance.
            transcripts: DataFrame containing transcript data with gene names
                and coordinates.
            lazy_loading: If True, use lazy loading mode (recommended for
                datasets > 50M points).
            config: Configuration object. If None, uses defaults.
            viewer_config: ViewerConfig for accessing static_canvas for color legend.

        Returns:
            Configured TranscriptViewerWidget ready to be added to the viewer.

        Example:
            >>> widget = create_transcript_viewer_widget(viewer, xdata.transcripts)
            >>> viewer.window.add_dock_widget(widget, name="Transcripts")
        """
        if config is None:
            config = TranscriptViewerConfig()

        if lazy_loading:
            if not hasattr(transcripts, "compute"):
                raise ValueError(
                    "lazy_loading=True requires a Dask DataFrame. "
                    "Pass a pandas DataFrame with lazy_loading=False instead."
                )
            gene_list, gene_colors = prepare_gene_colors(transcripts, config)
            widget = TranscriptViewerWidget(
                viewer=viewer,
                gene_data_or_list=gene_list,
                gene_colors=gene_colors,
                lazy_loading=True,
                dask_df=transcripts,
                config=config,
                viewer_config=viewer_config,
            )
        else:
            gene_data, gene_colors = prepare_gene_data(transcripts, config)
            widget = TranscriptViewerWidget(
                viewer=viewer,
                gene_data_or_list=gene_data,
                gene_colors=gene_colors,
                lazy_loading=False,
                config=config,
                viewer_config=viewer_config,
            )

        return widget

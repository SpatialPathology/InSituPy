from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from magicgui import magicgui

from insitupy import WITH_NAPARI

pytestmark = pytest.mark.skipif(not WITH_NAPARI, reason="napari is required for interactive callback tests")


from insitupy.interactive._callbacks import (  # noqa: E402
    _refresh_widgets_after_data_change,
    _update_key_on_type_change,
)
from insitupy.interactive._layers import _create_points_layer  # noqa: E402
from insitupy.interactive._widgets import (  # noqa: E402
    _build_labels_properties,
    _find_layer_by_scope,
)


def _make_field(value=None, choices=None):
    return SimpleNamespace(value=value, choices=list(choices or []))


def test_refresh_widgets_after_data_change_respects_current_key_type():
    viewer_config = SimpleNamespace(
        data_name="main",
        layer_name="main",
        masks=["cells", "nuclei"],
        key_dict={
            "genes": ["GeneA", "GeneB"],
            "obs": ["cell_type", "batch"],
            "obsm": ["X_umap#1", "X_umap#2"],
        },
        recent_selections=["obs:cell_type"],
    )

    select_data_widget = SimpleNamespace(
        data_name=_make_field(value="stale_data"),
        layer_name=_make_field(value="stale_layer"),
    )
    show_cells_widget = SimpleNamespace(
        key_type=_make_field(value="obs"),
        key=_make_field(value="GeneA", choices=["GeneA", "GeneB"]),
        recent=_make_field(value="obs:batch", choices=[]),
    )
    boundaries_widget = SimpleNamespace(key=_make_field(value=None, choices=[]))
    filter_widget = SimpleNamespace(obs_key=_make_field(value=None, choices=[]))

    _refresh_widgets_after_data_change(
        xdata=None,
        viewer=None,
        viewer_config=viewer_config,
        select_data_widget=select_data_widget,
        show_cells_widget=show_cells_widget,
        boundaries_widget=boundaries_widget,
        filter_widget=filter_widget,
    )

    assert select_data_widget.data_name.value == "main"
    assert select_data_widget.layer_name.value == "main"
    assert show_cells_widget.key.value is None
    assert show_cells_widget.key.choices == ["cell_type", "batch"]
    assert show_cells_widget.recent.choices == ["obs:cell_type"]
    assert show_cells_widget.recent.value is None
    assert boundaries_widget.key.choices == ["cells", "nuclei"]
    assert filter_widget.obs_key.choices == ["cell_type", "batch"]


def test_update_key_on_type_change_switches_choices_without_touching_type():
    viewer_config = SimpleNamespace(
        key_dict={
            "genes": ["GeneA", "GeneB"],
            "obs": ["cell_type", "batch"],
            "obsm": ["X_umap#1", "X_umap#2"],
        }
    )
    show_cells_widget = SimpleNamespace(
        key_type=_make_field(value="obsm"),
        key=_make_field(value=None, choices=["GeneA", "GeneB"]),
    )

    _update_key_on_type_change(show_cells_widget, viewer_config=viewer_config)

    assert show_cells_widget.key_type.value == "obsm"
    assert show_cells_widget.key.choices == ["X_umap#1", "X_umap#2"]


def test_update_key_on_type_change_updates_magicgui_reset_defaults():
    viewer_config = SimpleNamespace(
        key_dict={
            "genes": ["GeneA", "GeneB"],
            "obs": ["cell_type", "batch"],
            "obsm": ["X_umap#1", "X_umap#2"],
        }
    )

    @magicgui(
        key_type={"choices": ["genes", "obs", "obsm"]},
        key={"choices": viewer_config.key_dict["genes"]},
    )
    def widget(key_type="genes", key=None):
        pass

    widget.key_type.value = "obs"
    _update_key_on_type_change(widget, viewer_config=viewer_config)
    widget.key.reset_choices()

    assert widget.key_type.value == "obs"
    assert list(widget.key.choices) == [None, "cell_type", "batch"]


def test_build_labels_properties_keeps_all_columns_aligned_for_cells():
    viewer_config = SimpleNamespace(
        boundaries=SimpleNamespace(nucleus_to_cell_map=None),
        adata=SimpleNamespace(
            obs=pd.DataFrame({"cell_area": [1.0, 2.0]}),
            obs_names=pd.Index(["cell_a", "cell_b"]),
        ),
    )

    properties = _build_labels_properties(
        viewer_config=viewer_config,
        label_ids=np.array([11, 12, 13], dtype=np.uint32),
        cell_names_boundary=np.array(["cell_a", "cell_b", "cell_c"]),
        mask_key="cells",
    )

    assert len(properties["index"]) == 3
    assert properties["name"] == ["cell_a", "cell_b", "cell_c"]
    assert properties["cell_area"] == [1.0, 2.0, None]


def test_build_labels_properties_handles_unmapped_or_out_of_bounds_nuclei():
    viewer_config = SimpleNamespace(
        boundaries=SimpleNamespace(nucleus_to_cell_map={0: 0, 1: 5}),
        adata=SimpleNamespace(
            obs=pd.DataFrame({"cell_area": [4.0]}),
            obs_names=pd.Index(["cell_a"]),
        ),
    )

    properties = _build_labels_properties(
        viewer_config=viewer_config,
        label_ids=np.array([1, 2, 3], dtype=np.uint32),
        cell_names_boundary=np.array(["cell_a"]),
        mask_key="nuclei",
    )

    assert len(properties["index"]) == 3
    assert properties["name"] == ["cell_a", "unmapped", "unmapped"]
    assert properties["cell_area"] == [4.0, None, None]


def test_find_layer_by_scope_matches_scope_not_key():
    cells = ("celltypes", "main", "cells")
    nuclei = ("celltypes", "main", "nuclei")
    pts = ("celltypes", "main", "points")

    main = SimpleNamespace(name="s-geneA (cells)", metadata={"display_scope": cells})
    outline = SimpleNamespace(name="s-geneA (cells) outline",
                              metadata={"display_scope": cells, "is_outline_layer": True})
    nuc = SimpleNamespace(name="s-geneA (nuclei)", metadata={"display_scope": nuclei})
    points = SimpleNamespace(name="s-geneA", metadata={"display_scope": pts})
    untagged = SimpleNamespace(name="Annotations", metadata={})
    viewer = SimpleNamespace(layers=[untagged, main, outline, nuc, points])

    assert _find_layer_by_scope(viewer, cells, outline=False) is main
    assert _find_layer_by_scope(viewer, cells, outline=True) is outline
    assert _find_layer_by_scope(viewer, nuclei, outline=True) is None
    assert _find_layer_by_scope(viewer, pts, outline=False) is points
    assert _find_layer_by_scope(viewer, ("x", "main", "cells"), outline=False) is None


def test_find_layer_by_scope_returns_last_match_for_duplicates():
    cells = ("celltypes", "main", "cells")
    first = SimpleNamespace(name="s-geneA (cells)", metadata={"display_scope": cells})
    second = SimpleNamespace(name="s-geneB (cells)", metadata={"display_scope": cells})
    viewer = SimpleNamespace(layers=[first, second])
    assert _find_layer_by_scope(viewer, cells, outline=False) is second


def test_create_points_layer_tags_display_scope():
    points_scope = ("main", "main", "points")
    layer = _create_points_layer(
        points=np.array([[0.0, 0.0], [1.0, 1.0]]),
        color_values=np.array([1.0, 2.0]),
        name="main-GeneA",
        point_names=np.array(["cell_a", "cell_b"]),
        display_scope=points_scope,
    )
    _, kwargs, layer_type = layer
    assert layer_type == "points"
    assert kwargs["metadata"]["display_scope"] == points_scope
    # min=0 disables napari's canvas-size floor, which otherwise forces a visible
    # border_color edge on zoom-out regardless of border_width=0 (see
    # .log/reports/260706/transcript-viewer-race-fix/report-transcript-viewer-race-fix.md,
    # Addendum B).
    assert kwargs["canvas_size_limits"] == (0, 10000)

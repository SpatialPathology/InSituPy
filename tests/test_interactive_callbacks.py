from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from magicgui import magicgui

from insitupy import WITH_NAPARI

pytestmark = pytest.mark.skipif(not WITH_NAPARI, reason="napari is required for interactive callback tests")


from insitupy.interactive._callbacks import (  # noqa: E402
    _refresh_widgets_after_data_change, _update_key_on_type_change)
from insitupy.interactive._widgets import \
    _build_labels_properties  # noqa: E402


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
        adata=SimpleNamespace(obs=pd.DataFrame({"cell_area": [1.0, 2.0]})),
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
        adata=SimpleNamespace(obs=pd.DataFrame({"cell_area": [4.0]})),
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
    assert properties["cell_area"] == [4.0, None, None]
    assert properties["cell_area"] == [4.0, None, None]
    assert properties["cell_area"] == [4.0, None, None]

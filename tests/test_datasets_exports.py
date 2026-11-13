"""Guard: every public dataset function must be reachable as ``insitupy.datasets.<name>``.

The MCP datasets guide and the generated skill reference list the functions defined in
``insitupy.datasets.datasets``; a function missing from ``insitupy/datasets/__init__.py``
is advertised but raises ``AttributeError`` for users.
"""
import inspect

import insitupy.datasets as ds_pkg
import insitupy.datasets.datasets as ds_mod


def test_public_dataset_functions_are_exported():
    public = [
        name for name, obj in inspect.getmembers(ds_mod, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == ds_mod.__name__
    ]
    assert public, "no public dataset functions found - filter is broken"
    missing = [name for name in public if not hasattr(ds_pkg, name)]
    assert not missing, f"not exported from insitupy.datasets: {missing}"

"""The short submodule aliases (pp, tl, pl, im) must be importable as modules.

Sphinx autodoc >= 9 imports the module path of an entry like ``pp.normalize_and_transform``
instead of walking attributes, so without the registration every ``pp.*`` / ``tl.*`` / ``im.*``
page of the API reference renders empty. Users also expect ``from insitupy.tl import dge``.
"""

import importlib

import pytest

import insitupy

ALIASES = {
    "pp": ("insitupy.preprocessing", "normalize_and_transform"),
    "tl": ("insitupy.tools", "pseudobulk_dge"),
    "pl": ("insitupy.plotting", "spatial"),
    "im": ("insitupy.images", "read_zarr"),
}


@pytest.mark.parametrize("alias, target_and_member", ALIASES.items())
def test_alias_is_importable_and_exposes_the_real_objects(alias, target_and_member):
    target, member = target_and_member
    module = importlib.import_module(f"insitupy.{alias}")
    real = importlib.import_module(target)
    assert getattr(module, member) is getattr(real, member)
    assert member in dir(module)
    # The package attribute stays the real submodule.
    assert getattr(insitupy, alias) is real


def test_from_alias_import_function():
    from insitupy.pp import normalize_and_transform
    from insitupy.tl import pseudobulk_dge

    assert normalize_and_transform is insitupy.preprocessing.normalize_and_transform
    assert pseudobulk_dge is insitupy.tools.pseudobulk_dge


def test_star_import_from_alias_matches_real_module():
    via_alias, via_real = {}, {}
    exec("from insitupy.pp import *", via_alias)
    exec("from insitupy.preprocessing import *", via_real)
    via_alias.pop("__builtins__")
    via_real.pop("__builtins__")
    assert via_alias.keys() == via_real.keys()
    assert "normalize_and_transform" in via_alias


def test_alias_submodule_import_does_not_shadow_functions():
    """Importing a submodule through an alias must not re-execute it: a re-import would set the
    module as an attribute of the real package and replace the function of the same name
    (``insitupy.plotting.spatial``), breaking ``isp.pl.spatial(...)`` for the rest of the session."""
    spatial_function = insitupy.plotting.spatial
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("insitupy.pl.spatial")
    assert insitupy.plotting.spatial is spatial_function
    assert callable(insitupy.pl.spatial)

import sys
import types


class _AliasModule(types.ModuleType):
    """Importable stand-in for a short submodule alias such as ``insitupy.pp``.

    Every attribute is looked up on the real submodule, so ``from insitupy.pp import x`` and
    Sphinx autodoc (which imports the module path of ``pp.x``) see the real objects. The stand-in
    deliberately has no ``__path__``: registering the real package under the alias name would let
    ``import insitupy.pl.spatial`` execute ``plotting/spatial.py`` a second time and overwrite the
    ``plotting.spatial`` function with that module.
    """

    def __init__(self, name: str, target: types.ModuleType):
        super().__init__(name, target.__doc__)
        self._target = target

    def __getattr__(self, attr):
        if attr == "__all__":
            # Star imports must see the same public names as the real module, which may not
            # define ``__all__`` (Python then uses its non-underscore names).
            return getattr(
                self._target, "__all__", [name for name in vars(self._target) if not name.startswith("_")]
            )
        # Other dunders (above all ``__path__``) are not forwarded, so the import system never
        # treats the alias as a package.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return getattr(self._target, attr)

    def __dir__(self):
        return dir(self._target)


def register_module_aliases(package: str, aliases: dict[str, types.ModuleType]) -> None:
    """Make ``<package>.<alias>`` importable for each alias, without shadowing anything real."""
    for alias, target in aliases.items():
        sys.modules.setdefault(f"{package}.{alias}", _AliasModule(f"{package}.{alias}", target))

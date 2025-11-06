from warnings import warn


def NoProjectLoadWarning():
    warn("Loading functions only work on a saved InSituPy project.", UserWarning)

# DEPRECATION WARNINGS
def plot_functions_deprecations_warning(name):
    warn(f"The naming of plotting functions has changed in v0.9.0 and the prefix 'plot_' has been removed. E.g. `insitupy.pl.plot_{name}()` became `insitupy.pl.{name}()`.", DeprecationWarning, stacklevel=3)



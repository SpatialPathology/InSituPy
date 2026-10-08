from warnings import warn


def NoProjectLoadWarning():
    """Issue a UserWarning that loading functions require a saved InSituPy project."""
    warn("Loading functions only work on a saved InSituPy project.", UserWarning)

# DEPRECATION WARNINGS
def plot_functions_deprecations_warning(name):
    """Issue a DeprecationWarning that the ``plot_`` prefix was removed from plotting functions in v0.9.0."""
    warn(f"The naming of plotting functions has changed in v0.9.0 and the prefix 'plot_' has been removed. E.g. `insitupy.pl.plot_{name}()` became `insitupy.pl.{name}()`.", DeprecationWarning, stacklevel=3)



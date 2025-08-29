from warnings import warn


def plot_overview(self, *args, **kwargs):
    warn("`plot_overview()` is deprecated. Use `insitupy.plotting.plot_overview()` instead.", DeprecationWarning, stacklevel=2)
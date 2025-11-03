from warnings import warn


def plot_overview(self, *args, **kwargs):
    warn("`plot_overview()` is deprecated. Use `insitupy.plotting.overview()` instead.", DeprecationWarning, stacklevel=2)

def collect_anndatas(self, *args, **kwargs):
    warn("`collect_anndatas()` is deprecated. Use `to_anndata()` instead.", DeprecationWarning, stacklevel=2)

def import_obs(self, *args, **kwargs):
    warn("`import_obs()` is deprecated. Use `import_from_anndata()` instead.", DeprecationWarning, stacklevel=2)
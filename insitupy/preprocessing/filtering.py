import warnings

from insitupy.experimental.filtering import (
    _compute_mad_threshold,  # noqa: F401 — re-export for plotting
)


def calculate_mad_thresholds(*args, **kwargs):
    """Deprecated. Use :func:`insitupy.experimental.calculate_mad_thresholds`."""
    warnings.warn(
        "insitupy.pp.calculate_mad_thresholds is deprecated; use "
        "insitupy.experimental.calculate_mad_thresholds instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from insitupy.experimental import calculate_mad_thresholds as _impl
    return _impl(*args, **kwargs)

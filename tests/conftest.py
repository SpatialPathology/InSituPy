import os
import sys


def _needs_offscreen_qt() -> bool:
    """Return True where Qt should run on the offscreen platform (no usable display)."""
    if os.environ.get("CI"):
        return True
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False  # Windows / macOS desktops always have a display


# Use an offscreen Qt platform only on headless Linux or in CI. On Windows and macOS
# desktops the offscreen platform has no usable OpenGL (GLError 1282 in napari tests),
# so the native platform is used there. An explicit QT_QPA_PLATFORM always wins.
# Must be set before any Qt import.
if _needs_offscreen_qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from insitupy import WITH_NAPARI  # noqa: E402

if WITH_NAPARI:
    from napari.utils._testsupport import make_napari_viewer  # noqa: F401

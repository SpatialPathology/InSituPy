import os

# Use an offscreen Qt platform so napari tests work without a display.
# Must be set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from insitupy import WITH_NAPARI  # noqa: E402

if WITH_NAPARI:
    from napari.utils._testsupport import make_napari_viewer  # noqa: F401

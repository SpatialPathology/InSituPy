from insitupy._constants import WITH_NAPARI

from ._transcript_viewer import (
                                 TranscriptViewerConfig,
                                 prepare_gene_colors,
                                 prepare_gene_data,
)

# Names that need napari. They are only defined when napari is installed
# (``pip install "insitupy-spatial[viewer]"``); otherwise ``__getattr__`` below raises a
# friendly ImportError instead of a bare "cannot import name".
_VIEWER_ONLY = ("TranscriptViewerWidget", "create_transcript_viewer_widget", "sync_geometries")

if WITH_NAPARI:
    from ._transcript_viewer import (
                                     TranscriptViewerWidget,
                                     create_transcript_viewer_widget,
    )
    from .viewer import sync_geometries


def __getattr__(name: str):
    if name in _VIEWER_ONLY:
        raise ImportError(
            f"`insitupy.interactive.{name}` needs napari. "
            "Install it with `pip install \"insitupy-spatial[viewer]\"`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "sync_geometries",
    "TranscriptViewerConfig",
    "TranscriptViewerWidget",
    "create_transcript_viewer_widget",
    "prepare_gene_colors",
    "prepare_gene_data",
]

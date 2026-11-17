"""Reader registry and format detection shared by the experiment-level entry points.

``InSituExperiment.from_config`` and ``InSituExperiment.add`` use this module to turn a
directory (plus an optional reader name) into an ``InSituData`` object.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from insitupy._constants import ISPY_METADATA_FILE
from insitupy._io._atera import ATERA_METADATA_FILE, ATERA_MORPHOLOGY_DIR

if TYPE_CHECKING:
    from insitupy._core.data import InSituData

# Keep in sync with the `mode` Literals of InSituExperiment.add / from_config.
READER_NAMES: tuple[str, ...] = ("insitupy", "xenium", "atera", "visium", "visium_hd")

XENIUM_METADATA_FILE = "experiment.xenium"
VISIUM_HD_BINNED_DIR = "binned_outputs"
VISIUM_HD_SEGMENTED_DIR = "segmented_outputs"
VISIUM_SCALEFACTORS_FILE = "spatial/scalefactors_json.json"

_CHECKED_MARKERS = (
    f"'{ISPY_METADATA_FILE}' (insitupy), "
    f"'{ATERA_METADATA_FILE}' or '{ATERA_MORPHOLOGY_DIR}/' (atera), "
    f"'{XENIUM_METADATA_FILE}' (xenium), "
    f"'{VISIUM_HD_BINNED_DIR}/' or '{VISIUM_HD_SEGMENTED_DIR}/' (visium_hd), "
    f"'{VISIUM_SCALEFACTORS_FILE}' (visium)"
)


def _get_reader(name: str) -> Callable[..., InSituData]:
    """Return the reader function for a reader name.

    The readers are looked up lazily to avoid an import cycle with ``insitupy.io.data``.
    """
    if name not in READER_NAMES:
        raise ValueError(f"Unknown reader '{name}'. Valid readers are: {list(READER_NAMES)}.")

    from insitupy._core.data import InSituData
    from insitupy.io import data as io_data

    if name == "insitupy":
        def _read_insitupy(path, **kwargs):
            return InSituData.read(path, load_all=False, **kwargs)
        return _read_insitupy
    return getattr(io_data, f"read_{name}")


def _detect_reader(path: Path) -> str:
    """Detect the reader of a directory from marker files.

    Markers are checked by family; precedence within a family is intentional (Atera over
    Xenium, Visium HD over Visium), a match in more than one family is an error.
    """
    path = Path(path)
    matches: dict[str, str] = {}

    # insitupy family
    if (path / ISPY_METADATA_FILE).exists():
        matches["insitupy"] = f"'{ISPY_METADATA_FILE}'"

    # xenium family
    if (path / ATERA_METADATA_FILE).exists() or (path / ATERA_MORPHOLOGY_DIR).is_dir():
        found = [m for m, ok in [
            (f"'{ATERA_METADATA_FILE}'", (path / ATERA_METADATA_FILE).exists()),
            (f"'{ATERA_MORPHOLOGY_DIR}/'", (path / ATERA_MORPHOLOGY_DIR).is_dir()),
        ] if ok]
        matches["atera"] = " and ".join(found)
    elif (path / XENIUM_METADATA_FILE).exists():
        matches["xenium"] = f"'{XENIUM_METADATA_FILE}'"

    # visium family
    if (path / VISIUM_HD_BINNED_DIR).is_dir() or (path / VISIUM_HD_SEGMENTED_DIR).is_dir():
        found = [f"'{d}/'" for d in (VISIUM_HD_BINNED_DIR, VISIUM_HD_SEGMENTED_DIR)
                 if (path / d).is_dir()]
        matches["visium_hd"] = " and ".join(found)
    elif (path / VISIUM_SCALEFACTORS_FILE).exists():
        matches["visium"] = f"'{VISIUM_SCALEFACTORS_FILE}'"

    if len(matches) == 1:
        return next(iter(matches))
    if not matches:
        raise ValueError(
            f"Cannot auto-detect format for '{path}': none of the markers {_CHECKED_MARKERS} "
            f"was found. Set 'mode' (or the 'reader' column) explicitly."
        )
    found_str = "; ".join(f"{r}: {m}" for r, m in matches.items())
    raise ValueError(
        f"Ambiguous format for '{path}': markers of several readers were found ({found_str}). "
        f"Set 'mode' (or the 'reader' column) explicitly."
    )


def _read_dataset(path: Path, reader: str, kwargs: dict | None = None) -> InSituData:
    """Read a directory with the named reader."""
    kwargs = dict(kwargs) if kwargs else {}
    if reader == "insitupy" and kwargs:
        raise ValueError(f"InSituData.read takes no reader arguments, got: {sorted(kwargs)}.")
    return _get_reader(reader)(path, **kwargs)

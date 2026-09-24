"""Commit-record helpers for the versioned modalities of an InSituPy project.

``cells``, ``annotations`` and ``regions`` are stored as time-stamped save
directories (``<modality>/<YYMMDD-HHMMSSffffff-hex8>/``). The ``data[<modality>]``
entry of the project's ``.ispy`` file names the committed one: ``.ispy`` is
written atomically as the last step of every save, so it is the commit record.
The newest directory name is only a fallback for stores that carry no pointer,
because wall-clock names are not ordered under clock skew (NTP steps, VM
restores, two machines on one synced project).
"""
import shutil
from pathlib import Path
from warnings import warn

from insitupy._constants import ISPY_METADATA_FILE
from insitupy._io.files import read_json, write_dict_to_json
from insitupy.utils._helpers import sort_paths_by_datetime

VERSIONED_MODALITIES = ("cells", "annotations", "regions")


def _pointer_to_dir(project: Path, modality: str, pointer) -> Path | None:
    """Resolve an ``.ispy`` pointer to an existing save directory of *modality*.

    Returns ``None`` if *pointer* is not a string, does not name a directory
    directly under ``<project>/<modality>``, or the directory does not exist.
    The containment check keeps a hostile or stale ``.ispy`` from pointing a
    loader (or a deletion) outside the project.
    """
    if not isinstance(pointer, str):
        return None
    candidate = (project / pointer).resolve()
    if candidate.parent != (project / modality).resolve():
        return None
    return candidate if candidate.is_dir() else None


def _read_ispy(project: Path) -> dict | None:
    """Read the project's ``.ispy`` file; ``None`` if missing or unreadable."""
    try:
        meta = read_json(project / ISPY_METADATA_FILE)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def resolve_committed_dir(project: str | Path, modality: str) -> Path | None:
    """Return the committed save directory of a versioned modality.

    1. The ``.ispy`` pointer ``data[modality]``, if it names an existing
       directory directly under ``<project>/<modality>``.
    2. If a pointer is present but rejected (dangling, or outside the modality
       directory), a warning names it.
    3. Fallback for a missing or rejected pointer: the newest directory by
       name among those that follow the save-directory naming pattern.

    Args:
        project: Project root (the directory holding ``.ispy``).
        modality: One of :data:`VERSIONED_MODALITIES`.

    Returns:
        The resolved directory, or ``None`` if the modality has no save on disk.
    """
    project = Path(project)

    meta = _read_ispy(project)
    data = meta.get("data") if meta is not None else None
    pointer = data.get(modality) if isinstance(data, dict) else None

    if pointer is not None:
        committed = _pointer_to_dir(project, modality, pointer)
        if committed is not None:
            return committed
        warn(
            f"The '{ISPY_METADATA_FILE}' pointer for '{modality}' ({pointer!r}) does not "
            f"name a save directory inside '{project / modality}'. Falling back to the "
            "newest save directory by name.",
            UserWarning,
            stacklevel=2,
        )

    root = project / modality
    if not root.is_dir():
        return None
    saves = sort_paths_by_datetime([p for p in root.glob("[!.]*") if p.is_dir()])
    return saves[0] if saves else None


def prune_uncommitted(project: str | Path, metadata: dict | None = None) -> dict[str, int]:
    """Delete every save directory that is not the committed one.

    For each versioned modality the committed directory (see
    :func:`resolve_committed_dir`) is kept and every other directory whose name
    follows the save-directory pattern is deleted. Folders that do not follow
    the pattern (e.g. a backup the user made) are never deleted.

    ``history`` entries whose directory no longer exists are dropped from
    *metadata* (if given) and from the project's ``.ispy`` file (only rewritten
    when something changed).

    Args:
        project: Project root (the directory holding ``.ispy``).
        metadata: In-memory metadata whose ``history`` should be cleaned too.

    Returns:
        Number of directories deleted per modality.
    """
    project = Path(project)
    removed: dict[str, int] = {}

    for modality in VERSIONED_MODALITIES:
        removed[modality] = 0
        keep = resolve_committed_dir(project, modality)
        if keep is None:
            continue
        root = project / modality
        keep = keep.resolve()
        saves = sort_paths_by_datetime([p for p in root.glob("[!.]*") if p.is_dir()])
        for d in saves:
            if d.resolve() != keep:
                shutil.rmtree(d)
                removed[modality] += 1

    def _drop_dangling(meta: dict) -> bool:
        history = meta.get("history")
        if not isinstance(history, dict):
            return False
        changed = False
        for modality in VERSIONED_MODALITIES:
            entries = history.get(modality)
            if not isinstance(entries, list):
                continue
            live = [e for e in entries if isinstance(e, str) and (project / e).is_dir()]
            if live != entries:
                history[modality] = live
                changed = True
        return changed

    if metadata is not None:
        _drop_dangling(metadata)
    on_disk = _read_ispy(project)
    if on_disk is not None and _drop_dangling(on_disk):
        write_dict_to_json(dictionary=on_disk, file=project / ISPY_METADATA_FILE)

    return removed


def discard_new_saves(project: str | Path, before: dict, after: dict) -> None:
    """Delete save directories written since *before* that were never committed.

    Used when a save fails before the ``.ispy`` commit: any versioned modality
    whose pointer differs between the *before* snapshot and the (partially
    updated) *after* metadata has a freshly written directory that no committed
    state refers to.

    Args:
        project: Project root (the directory holding ``.ispy``).
        before: Metadata snapshot taken before the save started.
        after: Metadata as mutated by the failed save.
    """
    project = Path(project)
    old_data = before.get("data") or {}
    new_data = after.get("data") or {}
    for modality in VERSIONED_MODALITIES:
        new = new_data.get(modality)
        if new is None or new == old_data.get(modality):
            continue
        d = _pointer_to_dir(project, modality, new)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

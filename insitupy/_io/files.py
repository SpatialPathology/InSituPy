import gzip
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from insitupy.utils.utils import nested_dict_numpy_to_list

logger = logging.getLogger(__name__)


def read_json(path: str | os.PathLike | Path) -> dict:
    '''
    Function to load json or json.gz files as dictionary.
    '''
    # Determine if the file is gzipped
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
    else:
        with open(path) as f:
            data = json.load(f)

    return data


def write_dict_to_json(
    dictionary: dict,
    file: str | os.PathLike | Path,
    ):
    """Serialise a dictionary to a JSON file, converting NumPy arrays to lists if needed.

    Writes atomically via a temporary file to avoid corrupting an existing
    file if serialisation fails.  Parent directories are created automatically.

    Args:
        dictionary: The dict to serialise.  Must be JSON-compatible after
            optional NumPy array conversion.
        file: Output file path.
    """
    # First, serialize to string (may raise TypeError — no file touched yet)
    try:
        dict_json = json.dumps(dictionary, indent=4)
    except TypeError:
        # one reason for this type error could be that there are ndarrays in the dict
        # convert them to lists
        nested_dict_numpy_to_list(dictionary)
        dict_json = json.dumps(dictionary, indent=4)

    # Write atomically via temp file to avoid corrupting existing file on failure
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=file.parent, delete=False, suffix='.tmp') as tmp:
        tmp.write(dict_json)
        tmp_path = tmp.name
    os.replace(tmp_path, str(file))


def check_overwrite_and_remove_if_true(
    path: str | os.PathLike | Path,
    overwrite: bool = False
    ):
    """Delete *path* if *overwrite* is True; raise :exc:`FileExistsError` otherwise.

    Args:
        path: File or directory to check.
        overwrite: If True and *path* exists, delete it (directory tree or
            file).  If False and *path* exists, raise.

    Raises:
        FileExistsError: If *path* exists and *overwrite* is False.
        ValueError: If *path* exists but is neither a file nor a directory.
    """
    path = Path(path)
    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path) # delete directory
            elif path.is_file():
                path.unlink() # delete file
            else:
                raise ValueError(f"Path is neither a directory nor a file. What is it? {str(path)}")
        else:
            raise FileExistsError(f"The output file already exists at {path}. To overwrite it, please set the `overwrite` parameter to True."
)


def atomic_replace_dir(staging: Path, destination: Path, *, what: str = "write") -> None:
    """Atomically replace directory *destination* with directory *staging*.

    Moves an existing *destination* aside to a backup, renames *staging* into
    place, and deletes the backup only once *destination* is confirmed
    present. On failure the original *destination* is restored and *staging*
    is removed; if neither the swap nor the restore succeeds, the backup is
    kept as the only surviving copy.

    A backup left by a previous interrupted swap is handled first: if
    *destination* is missing, the backup is the only surviving copy and is
    promoted back to *destination*; if *destination* is intact, the backup is
    redundant and dropped.

    Args:
        staging: Freshly written directory to move into place. Must exist.
        destination: Final path to replace.
        what: Verb used in the failure log messages.
    """
    staging = Path(staging)
    destination = Path(destination)
    backup = destination.parent / (destination.name + ".__ispy_bak__")

    if backup.exists():
        if destination.exists():
            # Destination is intact, so the backup is genuinely redundant: drop it.
            check_overwrite_and_remove_if_true(backup, overwrite=True)
        else:
            # Interrupted swap: `backup` is the ONLY surviving complete copy and
            # `destination` is missing. Promote it back instead of deleting it.
            os.rename(backup, destination)
            logger.warning(
                "Recovered '%s' from an interrupted write: restored it from backup '%s'.",
                destination, backup,
            )

    destination_backed_up = False
    try:
        if destination.exists():
            os.rename(destination, backup)
            destination_backed_up = True
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if destination_backed_up and not destination.exists() and backup.exists():
            try:
                os.rename(backup, destination)
            except Exception:
                logger.error(
                    "%s failed AND the previous data could not be restored "
                    "automatically. Your original data is preserved at '%s' - "
                    "rename it back to '%s' manually.", what, backup, destination,
                )
        if destination.exists():
            logger.error(
                "%s failed: '%s' was left unchanged. If a file inside it is still open "
                "(e.g. a viewer or a zipped store), close viewers/objects reading from "
                "it and retry.", what, destination,
            )
        raise
    finally:
        # Remove the backup only once the destination is confirmed in place.
        if backup.exists() and destination.exists():
            shutil.rmtree(backup, ignore_errors=True)


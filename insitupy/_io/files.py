import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

from insitupy.utils.utils import nested_dict_numpy_to_list


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


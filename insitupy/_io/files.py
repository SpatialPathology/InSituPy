import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union

from insitupy.utils.utils import nested_dict_numpy_to_list


def read_json(path: Union[str, os.PathLike, Path]) -> dict:
    '''
    Function to load json or json.gz files as dictionary.
    '''
    # Determine if the file is gzipped
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    return data


def write_dict_to_json(
    dictionary: dict,
    file: Union[str, os.PathLike, Path],
    ):
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
    path: Union[str, os.PathLike, Path],
    overwrite: bool = False
    ):
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


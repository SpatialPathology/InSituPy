import json
import os
import shutil
from pathlib import Path
from typing import Union

from insitupy.utils.utils import nested_dict_numpy_to_list


def read_json(
    file: Union[str, os.PathLike, Path],
    ) -> dict:
    '''
    Function to load json files as dictionary.
    '''
    # load metadata file
    with open(file, "r") as metafile:
        metadata = json.load(metafile)

    return metadata


def write_dict_to_json(
    dictionary: dict,
    file: Union[str, os.PathLike, Path],
    ):
    try:
        dict_json = json.dumps(dictionary, indent=4)
        with open(file, "w") as metafile:
                metafile.write(dict_json)
    except TypeError:
        # one reason for this type error could be that there are ndarrays in the dict
        # convert them to lists
        nested_dict_numpy_to_list(dictionary)

        dict_json = json.dumps(dictionary, indent=4)
        with open(file, "w") as metafile:
                metafile.write(dict_json)


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
import gzip
import json
import os
import shutil
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

def copy_files_from_xenium_output(
    source_dir,
    target_dir,
    filename,
    xenium_filename: str = "experiment.xenium"
    ):
    """
    Copies specified files from subdirectories within a source directory to a target directory,
    renaming them based on metadata found in a signature file.

    Args:
        source_dir (str): The path to the source directory containing subdirectories.
        target_dir (str): The path to the target directory where files will be copied.
        filename (str): The name of the file to be copied from each subdirectory.
        signature_filename (str, optional): The name of the signature file used to identify
                                            valid subdirectories and extract metadata. Defaults to "experiment.xenium".

    Raises:
        FileNotFoundError: If the specified file or signature file does not exist in a subdirectory.

    Example:
        copy_files_from_folder("/path/to/source", "/path/to/target", "data.txt")

    This function ensures the target directory exists, iterates through all subdirectories in the source directory,
    checks for the presence of a signature file, and copies the specified file to the target directory with a new name
    based on metadata from the signature file.
    """
    # Ensure the target directory exists
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all folders in the source directory
    for folder in Path(source_dir).glob('*'):
        if folder.is_dir():
            # check if it is a Xenium output directory
            xenium_file = folder / xenium_filename
            if xenium_file.exists():
                print(f"Found Xenium output directory: {folder}")
                # Check if the specified file exists in the current folder
                file_path = folder / filename

                # get metadata
                slide_id = read_json(xenium_file)["slide_id"]
                region_name = read_json(xenium_file)["region_name"]
                if file_path.exists():
                    # Copy the file to the target directory
                    shutil.copy(file_path, target_path / f"{slide_id}__{region_name}__{filename}")
                    print(f"\tCopied {file_path} to {target_path}")
                else:
                    print("\tFile not found in directory.")

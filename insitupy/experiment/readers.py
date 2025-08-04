import os
from numbers import Number
from pathlib import Path
from typing import Union

from insitupy._core._qupath import _list_insitupy_data_folders
from insitupy._core.readers import read_qupath
from insitupy.experiment.data import InSituExperiment


def read_qupath_project(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number,
    method_name: str = "mIF"
):
    """
    Load and process a full QuPath project directory into an `InSituExperiment` object.

    This function scans a QuPath project directory for sample subfolders containing
    exported spatial data. Each sample is processed using `read_qupath`, and the resulting
    `InSituData` objects are aggregated into a single `InSituExperiment` instance.

    Args:
        path (Union[str, os.PathLike, Path]): Path to the root directory of the QuPath project.
        pixel_size (Number): Pixel size used to scale coordinates from annotation geometry.
        method_name (str, optional): Name of the imaging method used. Defaults to multiplexed IF ("mIF").

    Returns:
        InSituExperiment: An object containing all samples and modalities from the project.

    Raises:
        FileNotFoundError: If any required files are missing in the sample directories.
        ValueError: If any annotation file contains more than one geometry.

    Notes:
        Each sample folder within the project directory is expected to follow the structure
        described in `read_qupath`, including:
            - `annotation.geojson`
            - `measurements.tsv`
            - `cells.geojson`
            - `image.ome.tif`

        To generate data in the correct format from QuPath, use the following export script:
        https://github.com/SpatialPathology/InSituPy-QuPath/blob/main/scripts/export_for_insitupy.groovy

    Example:
        >>> exp = read_qupath_project(
        ...     path="/data/qupath_project",
        ...     pixel_size=0.65
        ... )
    """

    data_dict = _list_insitupy_data_folders(project_path=path)
    #return data_dict

    exp = InSituExperiment()
    for dataset_name, path_list in data_dict.items():
        print(f"Reading '{dataset_name}'...")
        for path in path_list:
            sample_name = path.name

            data = read_qupath(
                path=path,
                pixel_size=pixel_size,
                dataset_name=dataset_name,
                sample_name=sample_name,
                method_name=method_name
            )

            # --- Add all modalities to InSituExperiment ---
            exp.add(data)

    return exp
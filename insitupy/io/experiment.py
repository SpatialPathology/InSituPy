import json
import logging
import os
from numbers import Number
from pathlib import Path

from insitupy._io._qupath import (
    _get_pixel_size_from_qupath_metadata,
    _list_insitupy_data_folders,
)
from insitupy.experiment.data import InSituExperiment
from insitupy.io.data import read_qupath

logger = logging.getLogger(__name__)


def read_qupath_project(
    path: str | os.PathLike | Path,
    pixel_size: Number | None = None,
    export_folder: str = "insitupy",
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
    # check if the path is a QuPath project or points directly to the dtasets
    path = Path(path)
    qp_project_file = path / "project.qpproj"
    if qp_project_file.exists():
        data_path = Path(path) / export_folder
        logger.info(f"QuPath project file 'project.qpproj' found in directory. Searching for data in:\n'{data_path}'")

        if pixel_size is None:
            logger.info("Will try to automatically infer pixel sizes.")

            # Replace 'your_file.json' with the path to your JSON file
            with open(qp_project_file) as file:
                metadata = json.load(file)
    else:
        if pixel_size is None:
            raise ValueError(f"QuPath project file 'project.qpproj' not found in '{path}'. Parameter `pixel_size` cannot be automatically inferred and must be provided.")
        data_path = path

    data_dict = _list_insitupy_data_folders(data_path=data_path)
    #return data_dict

    exp = InSituExperiment()
    for dataset_name, path_list in data_dict.items():
        logger.info(f"Reading '{dataset_name}'...")
        for p in path_list:
            sample_name = p.name

            if pixel_size is None:
                px = _get_pixel_size_from_qupath_metadata(metadata=metadata, name=dataset_name)
            else:
                px = pixel_size

            data = read_qupath(
                path=p,
                pixel_size=px,
                dataset_name=dataset_name,
                sample_name=sample_name,
                method_name=method_name
            )

            # --- Add all modalities to InSituExperiment ---
            exp.add(data)

    return exp

import os
from numbers import Number
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from uuid import uuid4

import anndata
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from parse import *
from PIL import Image

from insitupy import __version__
from insitupy._core.data import InSituData
from insitupy._exceptions import InvalidXeniumDirectory
from insitupy.dataclasses.dataclasses import CellData, ImageData, MultiCellData
from insitupy.io._qupath import (_read_boundaries_qupath,
                                 _read_measurements_qupath)
from insitupy.io._read import _read_boundaries, _read_measurements
from insitupy.io._xenium import (_read_boundaries_from_xenium,
                                 _read_matrix_from_xenium,
                                 _restructure_transcripts_dataframe)
from insitupy.io.files import read_json
from insitupy.io.geo import parse_geopandas


def read_xenium(
    path: Union[str, os.PathLike, Path],
    nuclei_type: Literal["focus", "mip", ""] = "mip",
    load_cell_segmentation_images: bool = True,
    verbose: bool = True,
    transcript_mode: Literal["pandas", "dask"] = "dask",
    restructure_transcripts: bool = False
    ) -> InSituData:
    """
    Reads `Xenium In Situ data <https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest>`__
    from the specified directory.

    Args:
        path (Union[str, os.PathLike, Path]): Path to the Xenium data bundle.
        nuclei_type (Literal["focus", "mip", ""], optional): Type of nuclei image to load. Defaults to "mip".
            If "mip" is unavailable, "focus" will be used as a fallback.
        load_cell_segmentation_images (bool, optional): Whether to load cell segmentation images. Defaults to True.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        transcript_mode (Literal["pandas", "dask"], optional): Mode to load transcript data. Defaults to "dask".
            - "pandas": Loads the data into a pandas DataFrame.
            - "dask": Loads the data into a Dask DataFrame for larger datasets.
        restructure_transcripts (bool, optional): Whether to restructure the transcript data. Defaults to False.

    Returns:
        InSituData: An object containing the processed Xenium experiment data, including metadata, cells, images, and transcripts.

    Raises:
        InvalidXeniumDirectory: If the specified directory does not contain the required Xenium metadata file.
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If an invalid value is provided for `transcript_mode`.

    Notes:
        - The function initializes an `InSituData` object with metadata and loads cell data, images, and transcripts.
        - For Xenium versions <2.0, the "mip" image is used if available; otherwise, the "focus" image is loaded.
        - Cell segmentation images are loaded if available and `load_cell_segmentation_images` is True.
        - Transcript data can be loaded using either pandas or Dask, depending on the `transcript_mode` parameter.
    """

    path = Path(path) # make sure the path is a pathlib path
    metadata_filename: str = "experiment.xenium"

    if not (path / metadata_filename).exists():
        raise InvalidXeniumDirectory(directory=path)

    # check if path exists
    if not path.is_dir():
        raise FileNotFoundError(f"No such directory found: {str(path)}")

    # read metadata
    xenium_metadata = read_json(path / metadata_filename)

    # get slide id and sample id from metadata
    slide_id = xenium_metadata["slide_id"]
    sample_id = xenium_metadata["region_name"]

    data = InSituData(
        path=path,
        metadata=None, # initializes new metadata
        slide_id=slide_id,
        sample_id=sample_id,
        method_name="Xenium",
        method_params=xenium_metadata,
        )

    # LOAD CELLS
    if verbose:
        print("Loading cells...", flush=True)

    pixel_size = xenium_metadata["pixel_size"]

    # read celldata
    matrix = _read_matrix_from_xenium(path=data.path)
    boundaries = _read_boundaries_from_xenium(path=data.path, pixel_size=pixel_size)
    data.cells = MultiCellData()
    cd = CellData(matrix=matrix, boundaries=boundaries)
    data.cells.add_celldata(cd=cd, key="main", is_main=True)

    # LOAD IMAGES
    if verbose:
        print("Loading images...", flush=True)
    nuclei_file_key = f"morphology_{nuclei_type}_filepath"

    # In v2.0 the "mip" image was removed due to better focusing of the machine.
    # For <v2.0 the function still tries to retrieve the "mip" image but in case this is not found
    # it will retrieve the "focus" image
    if nuclei_type == "mip" and nuclei_file_key not in data.metadata["method_params"]["images"].keys():

        nuclei_type = "focus"
        nuclei_file_key = f"morphology_{nuclei_type}_filepath"

    # if names == "nuclei":
    img_keys = [nuclei_file_key]
    img_names = ["nuclei"]

    # get path of image files
    img_files = [data.metadata["method_params"]["images"][k] for k in img_keys]

    if load_cell_segmentation_images:
        # get cell segmentation images if available
        if "morphology_focus/" in data.metadata["method_params"]["images"][nuclei_file_key]:
            seg_files = ["morphology_focus/morphology_focus_0001.ome.tif",
                            "morphology_focus/morphology_focus_0002.ome.tif",
                            "morphology_focus/morphology_focus_0003.ome.tif"
                            ]
            seg_names = ["cellseg1", "cellseg2", "cellseg3"]

            # check which segmentation files exist and append to image list
            seg_file_exists_list = [(data.path / f).is_file() for f in seg_files]
            #print(seg_file_exists_list)
            img_files += [f for f, exists in zip(seg_files, seg_file_exists_list) if exists]
            img_names += [n for n, exists in zip(seg_names, seg_file_exists_list) if exists]

    # create imageData object
    img_paths = [data.path / elem for elem in img_files]

    # if data.images is None:
    #     data.images = ImageData(img_paths, img_names)
    # else:
    for im, n in zip(img_paths, img_names):
        data.images.add_image(im, n, overwrite=False, verbose=True)

    # LOAD TRANSCRIPTS
    transcript_filename = "transcripts.parquet"
    if verbose:
        print("Loading transcripts...", flush=True)

    if transcript_mode == "pandas":
        transcript_dataframe = pd.read_parquet(data.path / transcript_filename)

        if restructure_transcripts:
            data.transcripts = _restructure_transcripts_dataframe(transcript_dataframe)
        else:
            data.transcripts = transcript_dataframe
    elif transcript_mode == "dask":
        # Load the transcript data using Dask
        data.transcripts = dd.read_parquet(data.path / transcript_filename)
    else:
        raise ValueError(f"Invalid value for `transcript_mode`: {transcript_mode}")

    return data

def read_any(
    cellular_measurements: Dict[str, Union[str, Path, os.PathLike]],
    cellular_coordinates: Union[str, Path],
    cellular_metadata: Optional[Union[str, Path]] = None,
    cell_boundaries: Optional[Union[str, Path, os.PathLike]] = None,
    nucleus_boundaries: Optional[Union[str, Path, os.PathLike]] = None,
    images: Optional[Dict[str, Union[str, Path, os.PathLike]]] = None,
    pixel_size: Optional[Number] = None,
    dataset_name: Optional[str] = "Data 1",
    sample_name: Optional[str] = "Sample 1",
    method_name: str = "Any",
    xshift: Number = 0,
    yshift: Number = 0,
):
    """
    Load and assemble spatial data from arbitrary sources into an `InSituData` object.

    This function reads cellular measurements, coordinates, optional metadata, boundaries,
    and images from user-specified paths. It integrates these components into a structured
    `InSituData` object for downstream spatial analysis.

    Args:
        cellular_measurements (Dict[str, Union[str, Path, os.PathLike]]):
            Dictionary mapping measurement names to file paths.
        cellular_coordinates (Union[str, Path]):
            Path to the file containing cellular coordinates.
        cellular_metadata (Optional[Union[str, Path]], optional):
            Path to optional metadata file. Defaults to None.
        cell_boundaries (Optional[Union[str, Path, os.PathLike]], optional):
            Path to cell boundary file. Required if nucleus_boundaries is provided. Defaults to None.
        nucleus_boundaries (Optional[Union[str, Path, os.PathLike]], optional):
            Path to nucleus boundary file. Defaults to None.
        images (Optional[Dict[str, Union[str, Path, os.PathLike]]], optional):
            Dictionary mapping image names to image file paths. Defaults to None.
        pixel_size (Optional[Number], optional):
            Pixel size used for scaling boundaries and images. Required if boundaries or images are provided. Defaults to None.
        dataset_name (Optional[str], optional):
            Identifier for the dataset or slide. Defaults to "Data 1".
        sample_name (Optional[str], optional):
            Identifier for the sample within the dataset. Defaults to "Sample 1".
        method_name (str, optional):
            Name of the imaging or data acquisition method. Defaults to "Any".
        xshift (Number, optional):
            Horizontal shift applied to coordinates. Defaults to 0.
        yshift (Number, optional):
            Vertical shift applied to coordinates. Defaults to 0.

    Returns:
        InSituData: A structured object containing cell measurements, boundaries, and images.

    Raises:
        FileNotFoundError: If any specified file does not exist.
        ValueError: If nucleus boundaries are provided without cell boundaries, or if pixel_size is missing when required.

    Notes:
        - All paths are validated and converted to `Path` objects.
        - Boundaries and images require `pixel_size` to be specified.
        - The function supports flexible input formats for integrating diverse spatial datasets.
    """


    # Validate and convert paths for cellular measurements
    for n, measurements_path in cellular_measurements.items():
        measurements_path = Path(measurements_path)
        if not measurements_path.exists():
            raise FileNotFoundError(f"No measurements file found at '{measurements_path}'.")
        cellular_measurements[n] = measurements_path

    # Convert coordinate path
    cellular_coordinates = Path(cellular_coordinates)

    # Convert metadata path if provided
    if cellular_metadata is not None:
        cellular_metadata = Path(cellular_metadata)

    # Read cellular measurements
    adata = _read_measurements(
        cellular_measurements,
        coordinates_path=cellular_coordinates,
        metadata_path=cellular_metadata,
        xshift=xshift,
        yshift=yshift
    )

    # Read boundaries if provided
    boundaries = None

    cell_boundaries = Path(cell_boundaries) if cell_boundaries is not None else cell_boundaries
    nucleus_boundaries = Path(nucleus_boundaries) if nucleus_boundaries is not None else nucleus_boundaries

    if nucleus_boundaries is not None and cell_boundaries is None:
        raise ValueError((
            f"If `nucleus_boundaries` is given, `cell_boundaries` must be given as well. "
            f"If you only have nucleus boundaries, add them as cell boundaries."
            ))

    if cell_boundaries is not None:
        if pixel_size is None:
            raise ValueError("If boundaries are given, `pixel_size` must not be None.")

        boundaries = _read_boundaries(
            cells_path=cell_boundaries,
            nuclei_path=nucleus_boundaries,
            xshift=xshift,
            yshift=yshift,
            pixel_size=pixel_size
        )

    # Create InSituData object
    data = InSituData(
        path=None,
        metadata=None, # initializes new metadata
        slide_id=dataset_name,
        sample_id=sample_name,
        method_name=method_name,
        method_params={}
    )

    # Add CellData
    cd = CellData(matrix=adata, boundaries=boundaries)
    data.cells.add_celldata(cd=cd, key="main", is_main=True)

    # Add ImageData if provided
    if images is not None:
        if pixel_size is None:
            raise ValueError("If `images` is given, `pixel_size` must not be None.")

        for img_name, image_path in images.items():
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"No image file found at '{image_path}'.")
            data.images.add_image(image=image_path, name=img_name)

    return data

def read_qupath(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number,
    dataset_name: str,
    sample_name: str,
    method_name: str = "mIF",
):
    """
    Load and process QuPath-exported spatial data into an `InSituData` object.

    This function reads annotation, cellular measurements, cell boundaries, and image data
    from a specified directory containing QuPath outputs. It applies coordinate shifts to
    align image data and transcriptomic data relative to the origin.
    All components are integrated into a structured `InSituData` object
    for downstream spatial analysis.

    Args:
        path (Union[str, os.PathLike, Path]): Path to the directory containing QuPath-exported files.
        pixel_size (Number): Pixel size used to scale coordinates from annotation geometry.
        dataset_name (str): Identifier for the dataset or slide.
        sample_name (str): Identifier for the sample within the dataset.
        method_name (str, optional): Name of the imaging method used. Defaults to multiplexed IF ("mIF").

    Returns:
        InSituData: A structured object containing image data, cell measurements, and boundaries.

    Raises:
        FileNotFoundError: If any of the required files (`annotation.geojson`, `measurements.tsv`,
            `cells.geojson`, or `image.ome.tif`) are missing in the specified path.
        ValueError: If more than one annotation geometry is found in the annotation file.

    Notes:
        The expected directory structure should include the following files:
            - `annotation.geojson`: A single annotation encircling the dataset.
              Contains information about the shift of the data relative to the origin.
            - `measurements.tsv`: Tabular data with cellular measurements. Measurements are divided
              into 'Cell', 'Nucleus', 'Cytoplasm', and 'Membrane'. These are saved as layers
              in the final `AnnData` object in `data.cells.matrix`.
            - `cells.geojson`: Nuclear and cellular boundaries of individual cells.
            - `image.ome.tif`: The corresponding image file.

        To generate data in the correct format from QuPath, use the following export script:
        https://github.com/SpatialPathology/InSituPy-QuPath/blob/main/scripts/export_for_insitupy.groovy

    Example:
        >>> data = read_qupath(
        ...     path="/data/qupath_export",
        ...     pixel_size=0.65,
        ...     dataset_name="Slide001",
        ...     sample_name="SampleA"
        ... )
    """
    # --- Check file paths ---
    path = Path(path)
    annot_path = path / "annotation.geojson"
    measurements_path = path / "measurements.tsv"
    bound_path = path / "cells.geojson"
    image_path = path / "image.ome.tif"

    if not annot_path.exists():
        raise FileNotFoundError(f"No annotation file found at '{annot_path}'.")

    if not measurements_path.exists():
        raise FileNotFoundError(f"No measurements file found at '{measurements_path}'.")

    if not bound_path.exists():
        raise FileNotFoundError(f"No boundaries file found at '{bound_path}'.")

    if not image_path.exists():
        raise FileNotFoundError(f"No image file found at '{image_path}'.")

    # --- Read annotation encircling the whole dataset to shift coordinates to origin ---
    annot = parse_geopandas(annot_path)

    if len(annot) > 1:
        raise ValueError(f"More than one annotation found in '{annot_path}'.")

    # determine the x and y shift of the data
    xmin = annot["geometry"].item().bounds[0] * pixel_size
    ymin = annot["geometry"].item().bounds[1] * pixel_size

    # move the annotation to the origin
    annot["geometry"] = annot["geometry"].translate(
        xoff=-xmin/pixel_size, yoff=-ymin/pixel_size
        )

    # --- Read cellular measurements ---
    adata = _read_measurements_qupath(
        measurements_path, xshift=xmin, yshift=ymin
        )

    # --- Read cellular boundaries ---
    print(bound_path)
    boundaries = _read_boundaries_qupath(
        bound_path,
        object_ids=adata.obs["Object ID"].values,
        cell_names=adata.obs_names,
        xshift=xmin, yshift=ymin,
        pixel_size=pixel_size
        )

    # --- Create InSituData object ---
    data = InSituData(
        path=None,
        metadata=None, # initializes new metadata
        slide_id=dataset_name,
        sample_id=sample_name,
        method_name=method_name,
        method_params={}
    )

    # --- Add CellData ---
    cd = CellData(matrix=adata, boundaries=boundaries)
    data.cells.add_celldata(
        cd=cd, key="main", is_main=True
    )

    # --- Add ImageData ---
    data.images.add_image(
        image=image_path,
        name=method_name,
    )

    # --- Add the data annotation as region ---
    data.regions.add_data(
        data=annot,
        key="data",
        scale_factor=pixel_size
    )

    return data


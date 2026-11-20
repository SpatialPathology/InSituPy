import logging
import os
import warnings
from numbers import Number
from pathlib import Path
from typing import Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd

from insitupy._core.data import InSituData
from insitupy._exceptions import (
    InSituPyDeprecationWarning,
    InvalidAteraDirectory,
    InvalidXeniumDirectory,
)
from insitupy._io._atera import (
    ATERA_CELL_FEATURE_MATRIX_FILE,
    ATERA_CELLS_FILE,
    ATERA_METADATA_FILE,
    ATERA_MORPHOLOGY_DIR,
    _read_pixel_size_from_ome,
    _read_table_from_atera,
    _read_transcripts_from_atera,
)
from insitupy._io._qupath import _read_boundaries_qupath, _read_measurements_qupath
from insitupy._io._read import _read_boundaries, _read_measurements
from insitupy._io._xenium import (
    _read_boundaries_from_xenium,
    _read_table_from_xenium,
    _restructure_transcripts_dataframe,
)
from insitupy._io.files import read_json
from insitupy._io.geo import parse_geopandas
from insitupy.containers.cell_data import CellData
from insitupy.containers.spatial_units_data import _transform_geoseries

logger = logging.getLogger(__name__)

# Highest "major_version" in experiment.xenium (Xenium Onboard Analysis 4.0) InSituPy was tested with.
_XENIUM_MAX_TESTED_MAJOR_VERSION = 6
# Highest "major_version" in experiment.spatial (Atera Onboard Analysis 1.0) InSituPy was tested with.
_ATERA_MAX_TESTED_MAJOR_VERSION = 1


def _warn_if_untested_major_version(metadata: dict, max_tested: int, product: str, metadata_file: str):
    """Log a warning when a bundle's ``major_version`` is newer than the highest tested one.

    A missing key (e.g. Xenium Onboard Analysis 1.x bundles) or a non-integer value is ignored.
    """
    found_major = metadata.get("major_version")
    if isinstance(found_major, int) and found_major > max_tested:
        logger.warning(
            "This %s bundle has major version %s (%s), but InSituPy was only "
            "tested up to major version %s. Reading may fail or be incomplete.",
            product, found_major, metadata_file, max_tested,
        )

CELLSEG_NAMES = ["atp1a1_cd45_e-cadherin", "18s", "alphasma_vimentin"]

def _handle_xenium_image_names(im_path):
    im_path = Path(im_path)
    if im_path.name.startswith("ch"):
        ch, ch_name = im_path.name.split(".")[0].split("_", maxsplit=1)

    elif im_path.name.startswith("morphology_"):
        _, _, ch = im_path.name.split(".")[0].split("_")
        ch_name = f"cellseg_{ch}"

    else:
        raise ValueError(
            f"Unexpected Xenium image filename '{im_path.name}'. "
            f"Expected filename starting with 'ch' or 'morphology_'. "
            f"If this is a valid Xenium image, please open an issue."
        )

    return ch, ch_name

def _add_morphology_images(
    data: InSituData,
    nuclei_file: str | os.PathLike | Path,
    image_dir: Path,
    load_cell_segmentation_images: bool,
    load_background_images: bool,
) -> None:
    """Add the nuclei image and, optionally, the other morphology channels of a 10x bundle.

    Shared by ``read_xenium`` (``morphology_focus/``) and ``read_atera`` (``morphology_2d/``),
    which use the same ``chNNNN_<name>.ome.tif`` naming.
    """
    img_names = ["nuclei"]
    img_files = [nuclei_file]

    # get cell segmentation images if available
    if image_dir.is_dir():
        for im_path in image_dir.glob("*.ome.tif"):
            ch, ch_name = _handle_xenium_image_names(im_path)

            if not load_cell_segmentation_images and (ch_name.startswith("cellseg") or ch_name in CELLSEG_NAMES):
                continue
            if not load_background_images and ch_name.endswith("_background"):
                continue
            if ch_name == "dapi":
                continue

            img_names.append(ch_name)
            img_files.append(im_path)

    img_paths = [data.path / elem for elem in img_files]
    for im, n in zip(img_paths, img_names, strict=True):
        data.images.add_image(im, n, overwrite=False, verbose=True)

def _warn_ignored_spatialdata_params(
    nuclei_type: str,
    load_cell_segmentation_images: bool,
    load_background_images: bool,
    transcript_mode: str,
    restructure_transcripts: bool,
) -> None:
    """Warn once, naming every ``read_xenium`` parameter the spatialdata backend ignores (SD-B18).

    The spatialdata backend delegates reading to ``spatialdata_io.xenium()``, which has no
    equivalent knob for any of these five insitupy-backend parameters - honoring them is
    deferred to a follow-up (0.12.x reader-audit item), so this only makes the gap loud
    instead of silently ignoring a value the caller explicitly asked for.
    """
    defaults = {
        "nuclei_type": "focus",
        "load_cell_segmentation_images": False,
        "load_background_images": False,
        "transcript_mode": "dask",
        "restructure_transcripts": False,
    }
    passed = {
        "nuclei_type": nuclei_type,
        "load_cell_segmentation_images": load_cell_segmentation_images,
        "load_background_images": load_background_images,
        "transcript_mode": transcript_mode,
        "restructure_transcripts": restructure_transcripts,
    }
    ignored = [name for name, value in passed.items() if value != defaults[name]]
    if ignored:
        warnings.warn(
            f"The spatialdata backend does not honor the following parameter(s), which were "
            f"passed with a non-default value: {', '.join(ignored)}. spatialdata_io.xenium() "
            "has no equivalent knobs for these - the read proceeds with its own defaults "
            "regardless of the value passed.",
            UserWarning, stacklevel=3
        )

def read_xenium(
    path: str | os.PathLike | Path,
    nuclei_type: Literal["focus", "mip", ""] = "focus",
    load_cell_segmentation_images: bool = False,
    load_background_images: bool = False,
    verbose: bool = True,
    transcript_mode: Literal["pandas", "dask"] = "dask",
    restructure_transcripts: bool = False,
    dataset_name: str | None = None,
    sample_name: str | None = None,
    slide_id: str | None = None,
    sample_id: str | None = None,
    backend: Literal["insitupy", "spatialdata"] = "insitupy",
    ) -> InSituData:
    """
    Reads `Xenium In Situ data <https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest>`__
    from the specified directory.

    Args:
        path (Union[str, os.PathLike, Path]): Path to the Xenium data bundle.
        nuclei_type (Literal["focus", "mip", ""], optional): Type of nuclei image to load. Defaults to "focus".
            If "mip" is unavailable, "focus" will be used as a fallback.
        load_cell_segmentation_images (bool, optional): Whether to load cell segmentation images. Defaults to False.
        load_background_images (bool, optional): Whether to load background images. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        transcript_mode (Literal["pandas", "dask"], optional): Mode to load transcript data. Defaults to "dask".
            - "pandas": Loads the data into a pandas DataFrame.
            - "dask": Loads the data into a Dask DataFrame for larger datasets.
        restructure_transcripts (bool, optional): Whether to restructure the transcript data. Defaults to False.

            .. deprecated:: 0.13
                Removed in 0.14. Only has an effect with ``transcript_mode='pandas'``.
        dataset_name (str, optional): Name for the overarching dataset/slide. For the insitupy
            backend this is overridden by the value in the Xenium metadata file. For the spatialdata
            backend it is not used (``spatialdata_io.xenium()`` has no equivalent parameter).
            Defaults to None.
        sample_name (str, optional): Name for the individual sample/ROI within the slide. For the
            insitupy backend this is overridden by the value in the Xenium metadata file. For the
            spatialdata backend it is not used. Defaults to None.
        slide_id (str, optional): Deprecated. Use ``dataset_name`` instead.
        sample_id (str, optional): Deprecated. Use ``sample_name`` instead.
        backend (Literal["insitupy", "spatialdata"], optional): Backend to use for loading data. Defaults to "insitupy".
            - "insitupy": Uses the native InSituPy loader.
            - "spatialdata": Uses spatialdata-io to load the data and converts to InSituData format.

    Returns:
        InSituData: An object containing the processed Xenium experiment data, including metadata, cells, images, and transcripts.

    Raises:
        InvalidXeniumDirectory: If the specified directory does not contain the required Xenium metadata file.
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If an invalid value is provided for `transcript_mode` or `backend`.

    Notes:
        - The function initializes an `InSituData` object with metadata and loads cell data, images, and transcripts.
        - For Xenium versions <2.0, the "mip" image is used if available; otherwise, the "focus" image is loaded.
        - Cell segmentation images are loaded if available and `load_cell_segmentation_images` is True.
        - Background images are loaded if available and `load_background_images` is True.
        - Transcript data can be loaded using either pandas or Dask, depending on the `transcript_mode` parameter.
        - The spatialdata backend provides interoperability with the SpatialData ecosystem.
    """
    if slide_id is not None:
        warnings.warn("'slide_id' is deprecated, use 'dataset_name' instead.",
                      DeprecationWarning, stacklevel=2)
        dataset_name = slide_id
    if sample_id is not None:
        warnings.warn("'sample_id' is deprecated, use 'sample_name' instead.",
                      DeprecationWarning, stacklevel=2)
        sample_name = sample_id
    if restructure_transcripts:
        warnings.warn(
            "`read_xenium(restructure_transcripts=True)` is deprecated and will be removed in "
            "InSituPy 0.14. The MultiIndex-column transcript table it builds cannot be saved. "
            "Use the default flat columns (`x_location`, `y_location`, ...) instead.",
            InSituPyDeprecationWarning, stacklevel=2,
        )

    path = Path(path) # make sure the path is a pathlib path

    metadata_filename: str = "experiment.xenium"

    # Check if the directory exists first
    if not path.exists():
        raise FileNotFoundError(f"The directory '{path}' does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    # Check if the required metadata file exists
    if not (path / metadata_filename).exists():
        # Atera bundles have no experiment.xenium but carry experiment.spatial and/or morphology_2d/
        hint = None
        if (path / ATERA_METADATA_FILE).exists() or (path / ATERA_MORPHOLOGY_DIR).is_dir():
            hint = "This looks like 10x Atera output - use `insitupy.io.read_atera` instead."
        raise InvalidXeniumDirectory(directory=path, hint=hint)

    # read metadata
    xenium_metadata = read_json(path / metadata_filename)
    pixel_size = xenium_metadata["pixel_size"]

    _warn_if_untested_major_version(
        xenium_metadata, _XENIUM_MAX_TESTED_MAJOR_VERSION, "Xenium", metadata_filename
    )

    if not path.is_dir():
        raise FileNotFoundError(f"No such directory found: {str(path)}")

    if backend == "spatialdata":
        from spatialdata_io import xenium

        from insitupy.spatialdata.convert import convert_from_foreign_spatialdata

        if dataset_name is not None or sample_name is not None:
            warnings.warn(
                "dataset_name and sample_name are not used by the spatialdata backend "
                "(spatialdata_io.xenium() has no equivalent parameters).",
                UserWarning, stacklevel=2
            )
        _warn_ignored_spatialdata_params(
            nuclei_type=nuclei_type,
            load_cell_segmentation_images=load_cell_segmentation_images,
            load_background_images=load_background_images,
            transcript_mode=transcript_mode,
            restructure_transcripts=restructure_transcripts,
        )
        if verbose:
            logger.info("Reading Xenium data with spatialdata-io backend...")
        sdata = xenium(path)
        data = convert_from_foreign_spatialdata(
            sdata=sdata,
            images={
                "nuclei": {"key": "morphology_focus", "pixel_size": pixel_size},
                "mip":    {"key": "morphology_mip",   "pixel_size": pixel_size},
            },
            cells={"main": {
                "table_key": "table",
                "cell_boundaries_data": ("cell_labels", pixel_size),
                "nucleus_boundaries_data": ("nucleus_labels", pixel_size),
                "label_map": "nucleus_boundaries",
            }},
            transcripts="transcripts",
            slide_id=dataset_name,
            sample_id=sample_name,
            method_name="Xenium"
            )

    elif backend == "insitupy":
        if verbose:
            logger.info("Reading Xenium data with InSituPy backend...")

        # get dataset_name and sample_name from metadata (overrides function parameters)
        _dataset_name = xenium_metadata["slide_id"]
        _sample_name = xenium_metadata["region_name"]

        data = InSituData(
            path=path,
            metadata=None, # initializes new metadata
            slide_id=_dataset_name,
            sample_id=_sample_name,
            method_name="Xenium",
            method_params=xenium_metadata,
            )

        # LOAD CELLS
        if verbose:
            logger.info("Loading cells...")

        # read celldata
        table = _read_table_from_xenium(path=data.path)
        boundaries = _read_boundaries_from_xenium(path=data.path, pixel_size=pixel_size)
        #data.cells = MultiCellData()
        cd = CellData(table=table, boundaries=boundaries)
        data.cells.add_celldata(cd=cd, key="main", is_main=True)

        # LOAD IMAGES
        if verbose:
            logger.info("Loading images...")
        nuclei_file_key = f"morphology_{nuclei_type}_filepath"

        # In v2.0 the "mip" image was removed due to better focusing of the machine.
        # For <v2.0 the function still tries to retrieve the "mip" image but in case this is not found
        # it will retrieve the "focus" image
        if nuclei_type == "mip" and nuclei_file_key not in data.metadata["method_params"]["images"].keys():
            nuclei_type = "focus"
            nuclei_file_key = f"morphology_{nuclei_type}_filepath"

        _add_morphology_images(
            data=data,
            nuclei_file=data.metadata["method_params"]["images"][nuclei_file_key],
            image_dir=path / "morphology_focus/",
            load_cell_segmentation_images=load_cell_segmentation_images,
            load_background_images=load_background_images,
        )

        # LOAD TRANSCRIPTS
        transcript_filename = "transcripts.parquet"
        if verbose:
            logger.info("Loading transcripts...")

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
    else:
        raise ValueError(f"Invalid backend: {backend}. Supported backends are 'insitupy' and 'spatialdata'.")

    return data


def read_atera(
    path: str | os.PathLike | Path,
    load_cell_segmentation_images: bool = False,
    load_background_images: bool = False,
    load_transcripts: bool = False,
    transcript_mode: Literal["pandas", "dask"] = "dask",
    dataset_name: str | None = None,
    sample_name: str | None = None,
    verbose: bool = True,
    ) -> InSituData:
    """
    Reads `10x Genomics Atera data <https://www.10xgenomics.com/support/software/atera-onboard-analysis/latest>`__
    from the specified directory.

    Supports output bundles of Atera Onboard Analysis 1.x (metadata in ``experiment.spatial``, cell
    table, segmentation and transcripts in zipped zarr stores, images in ``morphology_2d/``).
    Preproduction Atera data that was converted to the Xenium format (with an ``experiment.xenium``
    file) is read with :func:`read_xenium` and labelled as Atera.

    Args:
        path (Union[str, os.PathLike, Path]): Path to the Atera output bundle.
        load_cell_segmentation_images (bool, optional): Whether to load the cell segmentation stain
            images in addition to the nuclei (DAPI) image. Defaults to False.
        load_background_images (bool, optional): Whether to load background images. Defaults to False.
        load_transcripts (bool, optional): Whether to load the transcripts. Defaults to False, because
            whole-transcriptome bundles contain billions of transcripts and :meth:`InSituData.saveas`
            writes all loaded transcripts to disk. If True, the transcripts are loaded lazily.
            The same applies to preproduction data in Xenium format (``transcript_mode`` is only
            used if ``load_transcripts=True``).
        transcript_mode (Literal["pandas", "dask"], optional): Mode to load transcript data. Defaults to "dask".
            - "pandas": Computes all transcripts into memory.
            - "dask": Keeps the transcripts as a lazy Dask DataFrame.
        dataset_name (str, optional): Name for the overarching dataset/slide. Defaults to None, which
            uses ``slide_id`` from the Atera metadata. A value passed here takes precedence.
        sample_name (str, optional): Name for the individual sample/region. Defaults to None, which
            uses ``region_name`` from the Atera metadata. A value passed here takes precedence.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        InSituData: An object containing the Atera data, including metadata, cells, images and
        (optionally) transcripts.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the path is not a directory.
        InvalidAteraDirectory: If required Atera files are missing.
        ValueError: If an invalid value is provided for ``transcript_mode``, or if the cell ids of the
            cell table and the segmentation do not match.

    Notes:
        - The cell table contains only "Gene Expression" features (as in :func:`read_xenium`). The
          count matrix is loaded into memory. 10x's clusterings, cell type annotations and
          embeddings (``X_umap``, ``X_pca``) are kept. This includes the Azimuth cell type
          annotations (``azimuth_*`` columns in ``.obs``, labels and scores) of bundles that
          have them. 10x's ``highly_variable`` and ``filtered`` flags are renamed to
          ``atera_highly_variable`` and ``atera_filtered``.
        - ``experiment.spatial`` is optional. Without it the pixel size is read from the OME
          metadata of the first image in ``morphology_2d/``, ``slide_id`` and ``region_name`` are
          not set (pass ``dataset_name`` and ``sample_name``) and the default nuclei image
          ``morphology_2d/ch0000_dapi.ome.tif`` is used.
        - Transcript columns follow the Xenium naming (``x_location``, ``y_location``,
          ``z_location``, ``feature_name``, ``qv``, ``overlaps_nucleus``, ``cell_id``,
          ``codeword_index``). Transcripts outside of cells have ``cell_id == "UNASSIGNED"``.
        - 3D morphology images (``morphology_3d/``) are not loaded.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"The directory '{path}' does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    if transcript_mode not in ("pandas", "dask"):
        raise ValueError(f"Invalid value for `transcript_mode`: {transcript_mode}")

    # Format detection. Preproduction Atera data was converted to the Xenium output format, which
    # also has cells.zarr.zip and cell_feature_matrix.zarr.zip, so those files cannot tell the two
    # apart. Atera-specific markers are experiment.spatial and the morphology_2d/ directory.
    has_atera_markers = (path / ATERA_METADATA_FILE).exists() or (path / ATERA_MORPHOLOGY_DIR).is_dir()
    if (path / "experiment.xenium").exists() and not has_atera_markers:
        if verbose:
            logger.info("Xenium-formatted (preproduction) Atera data detected. Reading with read_xenium...")
        # read_xenium always attaches the transcripts (lazily in dask mode); keep them lazy and drop
        # them unless requested, so that saveas does not write them by accident
        data = read_xenium(
            path,
            load_cell_segmentation_images=load_cell_segmentation_images,
            load_background_images=load_background_images,
            verbose=verbose,
            transcript_mode=transcript_mode if load_transcripts else "dask",
        )
        if not load_transcripts:
            data._transcripts = None
        data._metadata["method"] = "Atera"
        if dataset_name is not None:
            data.slide_id = dataset_name
        if sample_name is not None:
            data.sample_id = sample_name
        return data

    required_files = [ATERA_CELL_FEATURE_MATRIX_FILE, ATERA_CELLS_FILE]
    missing_files = [f for f in required_files if not (path / f).exists()]
    if missing_files:
        raise InvalidAteraDirectory(directory=path, missing_files=missing_files)

    # read metadata; experiment.spatial is optional (some bundles omit it)
    if (path / ATERA_METADATA_FILE).exists():
        atera_metadata = read_json(path / ATERA_METADATA_FILE)
        _warn_if_untested_major_version(
            atera_metadata, _ATERA_MAX_TESTED_MAJOR_VERSION, "Atera", ATERA_METADATA_FILE
        )
    else:
        atera_metadata = {}
    if "pixel_size" in atera_metadata:
        pixel_size = float(atera_metadata["pixel_size"])
    else:
        pixel_size = _read_pixel_size_from_ome(path)
        atera_metadata["pixel_size"] = pixel_size

    if verbose:
        logger.info("Reading Atera data with InSituPy backend...")

    data = InSituData(
        path=path,
        metadata=None, # initializes new metadata
        slide_id=dataset_name if dataset_name is not None else atera_metadata.get("slide_id"),
        sample_id=sample_name if sample_name is not None else atera_metadata.get("region_name"),
        method_name="Atera",
        method_params=atera_metadata,
        )

    # LOAD CELLS
    if verbose:
        logger.info("Loading cells...")
    table = _read_table_from_atera(path=path)
    boundaries = _read_boundaries_from_xenium(path=path, pixel_size=pixel_size)

    # the cell table and the segmentation share the row order; a mismatch means the format changed
    if not table.obs_names.equals(pd.Index(boundaries.cell_names)):
        raise ValueError(
            f"Cell ids in '{ATERA_CELL_FEATURE_MATRIX_FILE}' and '{ATERA_CELLS_FILE}' do not match. "
            "This Atera output format is not supported yet - please open an issue."
        )

    cd = CellData(table=table, boundaries=boundaries)
    data.cells.add_celldata(cd=cd, key="main", is_main=True)

    # LOAD IMAGES
    if verbose:
        logger.info("Loading images...")
    _add_morphology_images(
        data=data,
        nuclei_file=atera_metadata.get("images", {}).get(
            "morphology_2d_filepath", f"{ATERA_MORPHOLOGY_DIR}/ch0000_dapi.ome.tif"
        ),
        image_dir=path / ATERA_MORPHOLOGY_DIR,
        load_cell_segmentation_images=load_cell_segmentation_images,
        load_background_images=load_background_images,
    )

    # LOAD TRANSCRIPTS
    if load_transcripts:
        if verbose:
            logger.info("Loading transcripts...")
        transcripts = _read_transcripts_from_atera(path)
        if transcript_mode == "pandas":
            transcripts = transcripts.compute()
        data.transcripts = transcripts

    return data


def read_visium(
    path: str | os.PathLike | Path,
    dataset_name: str | None = None,
    sample_name: str | None = None,
    slide_id: str | None = None,
    sample_id: str | None = None,
    verbose: bool = True,
    fullres_pixel_size: Number | None = None, # microns per pixel
    **kwargs,
) -> InSituData:
    """
    Reads `Visium spatial transcriptomics data <https://www.10xgenomics.com/products/spatial-gene-expression>`__
    from the specified directory using spatialdata-io.

    Args:
        path (Union[str, os.PathLike, Path]): Path to the Visium data bundle.
        dataset_name (Optional[str], optional): Name for the overarching dataset/slide. Also used
            internally as ``dataset_id`` for spatialdata-io element naming (shapes, images, table keys).
            Defaults to None (falls back to ``"visium"``).
        sample_name (Optional[str], optional): Name for the individual sample/ROI within the slide.
            Defaults to None, which uses the Space Ranger sample ID (``library_ids`` in the count
            matrix), else the folder name.
        slide_id (str, optional): Deprecated. Use ``dataset_name`` instead.
        sample_id (str, optional): Deprecated. Use ``sample_name`` instead.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        fullres_pixel_size (Number): Physical size of one full-resolution
            pixel in micrometers. Used to convert spot geometries to micron coordinates.
            Required - there is no reliable default; omitting it raises ``ValueError``.
        **kwargs: Additional keyword arguments forwarded to ``spatialdata_io.visium()``.
            Commonly used arguments include:

            - ``counts_file`` (str): Name of the counts file (e.g.
              ``'filtered_feature_bc_matrix.h5'``). Defaults to
              ``'filtered_feature_bc_matrix.h5'``.
            - ``fullres_image_file`` (Optional[str]): Path to the full-resolution
              tissue image. Auto-detected if None.

    Returns:
        InSituData: An object containing the processed Visium experiment data, including metadata, cells, and images.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If required Visium files are missing from the directory.

    Notes:
        - This function uses spatialdata-io to load Visium data and converts it to InSituData format.
        - The function loads spatial coordinates, gene expression counts, and optional histology images.
        - Spot positions are stored as features in the InSituData object.
    """
    if slide_id is not None:
        warnings.warn("'slide_id' is deprecated, use 'dataset_name' instead.",
                      DeprecationWarning, stacklevel=2)
        dataset_name = slide_id
    if sample_id is not None:
        warnings.warn("'sample_id' is deprecated, use 'sample_name' instead.",
                      DeprecationWarning, stacklevel=2)
        sample_name = sample_id

    from spatialdata_io import visium

    from insitupy.spatialdata.convert import convert_from_foreign_spatialdata

    path = Path(path)

    dataset_id = dataset_name if dataset_name is not None else "visium"

    if not path.is_dir():
        raise FileNotFoundError(f"No such directory found: {str(path)}")

    if sample_name is None:
        sample_name = _default_10x_sample_name(
            path,
            [path / kwargs.get("counts_file", "filtered_feature_bc_matrix.h5")]
            + sorted(path.glob("*filtered_feature_bc_matrix.h5")),
        )

    if verbose:
        logger.info("Reading Visium data with spatialdata-io...")

    if fullres_pixel_size is None:
        raise ValueError(
            "`fullres_pixel_size` is required (microns per full-resolution pixel). "
            "Without it, spot geometries would be left in pixel coordinates. "
            "Please determine the resolution of the full-resolution image and pass it explicitly."
        )

    sf_file = path / "spatial" / "scalefactors_json.json"
    scale_factors = read_json(sf_file)
    hires_pixel_size = fullres_pixel_size / scale_factors["tissue_hires_scalef"]
    lowres_pixel_size = fullres_pixel_size / scale_factors["tissue_lowres_scalef"]

    # Load Visium data using spatialdata-io
    sdata = visium(
        path=path,
        dataset_id=dataset_id,
        **kwargs
    )

    if fullres_pixel_size != 1.0:
        # convert spot geometries to microns
        sdata[dataset_id]['geometry'] = _transform_geoseries(
            sdata[dataset_id].geometry, lambda c: c * float(fullres_pixel_size)
        )

        # the units table must live in the same (micron) coordinates as the shapes
        for tkey in list(sdata.tables):
            tab = sdata.tables[tkey]
            if "spatial" in tab.obsm:
                tab.obsm["spatial"] = np.asarray(tab.obsm["spatial"], dtype=float) * float(fullres_pixel_size)

        if 'radius' in sdata[dataset_id].columns:
            sdata[dataset_id]['radius'] = sdata[dataset_id]['radius'] * fullres_pixel_size

    # Convert to InSituData format
    data = convert_from_foreign_spatialdata(
        sdata=sdata,
        images={
            "hires":  {"key": f"{dataset_id}_hires_image",  "pixel_size": hires_pixel_size,  "is_rgb": True},
            "lowres": {"key": f"{dataset_id}_lowres_image", "pixel_size": lowres_pixel_size, "is_rgb": True},
        },
        units={"visium": {"table_key": "table", "units_key": dataset_id, "unit_type": "visium"}},
        transcripts=None,           # Visium has no single-molecule transcripts
        slide_id=dataset_name,
        sample_id=sample_name,
        method_name="visium",
        verbose=verbose,
    )

    return data


def _default_10x_sample_name(path: Path, h5_candidates: list[Path]) -> str:
    """Default sample name for a Space Ranger output folder.

    Returns the single ``library_ids`` entry (the Space Ranger sample ID) of the first candidate
    count matrix that is readable HDF5 and carries exactly one id. Falls back to the folder name
    (the parent folder if *path* is Space Ranger's ``outs``). Never raises.
    """
    import h5py

    for f in h5_candidates:
        try:
            if not Path(f).is_file():
                continue
            with h5py.File(f, "r") as h:
                ids = h.attrs.get("library_ids")
            if ids is None:
                continue
            ids = np.atleast_1d(ids)
            if len(ids) != 1:
                continue
            val = ids[0]
            val = val.decode() if isinstance(val, bytes) else str(val)
            if val:
                return val
        except Exception:
            continue
    path = Path(path).resolve()  # a relative "." has an empty name
    return path.parent.name if path.name == "outs" else path.name


def _read_visium_hd_scalefactors(path: Path) -> dict:
    """Return scale factors from the first ``scalefactors_json.json`` found in a Visium HD bundle."""
    candidates = [path / "segmented_outputs" / "spatial" / "scalefactors_json.json"]
    candidates += sorted((path / "binned_outputs").glob("*/spatial/scalefactors_json.json"))
    for f in candidates:
        if f.is_file():
            return read_json(f)
    return {}


def _read_visium_hd_bin_mapping(path: Path, cell_ids: pd.Index) -> pd.DataFrame | None:
    """Read the 2 um bin -> cell mapping of a Visium HD bundle.

    Reads only the needed columns of ``barcode_mappings.parquet`` (bundle root) and keeps
    only 2 um bins that belong to a cell in ``cell_ids``.

    Returns:
        DataFrame with columns ``cell_id`` (categorical), ``array_row``, ``array_col`` (int32),
        ``in_nucleus``; or None if no mapping file exists.
    """
    file = path / "barcode_mappings.parquet"
    if not file.is_file():
        logger.info("No barcode_mappings.parquet found - bin -> cell mapping is not loaded.")
        return None

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = ["square_002um", "cell_id", "in_nucleus"]
    table = pq.read_table(file, columns=cols, filters=pc.field("cell_id").is_valid())
    df = table.to_pandas()
    n_before = len(df)

    parsed = df["square_002um"].astype(str).str.extract(r"^s_002um_(\d+)_(\d+)-\d+$")
    if parsed.isna().any(axis=None):
        bad = df["square_002um"][parsed.isna().any(axis=1)].iloc[0]
        raise ValueError(
            f"Could not parse array_row/array_col from 2 um barcode '{bad}' "
            "(expected 's_002um_<row>_<col>-<n>')."
        )
    df["array_row"] = parsed[0].astype("int32").to_numpy()
    df["array_col"] = parsed[1].astype("int32").to_numpy()

    df = df[df["cell_id"].isin(cell_ids)]
    n_dropped = n_before - len(df)
    logger.info(f"Bin mapping: kept {len(df)} 2 um bins, dropped {n_dropped} not in the cell layer.")
    if n_before > 0 and len(df) == 0:
        warnings.warn(
            "None of the cell ids in barcode_mappings.parquet matches the cell layer - "
            "the bin -> cell mapping is empty. The cell id format may differ between "
            "Space Ranger versions."
        )

    return pd.DataFrame({
        "cell_id": pd.Categorical(df["cell_id"].to_numpy()),
        "array_row": df["array_row"].to_numpy(),
        "array_col": df["array_col"].to_numpy(),
        "in_nucleus": df["in_nucleus"].to_numpy(),
    })


def read_visium_hd(
    path: str | os.PathLike | Path,
    bin_size: int | list[int] | None = None,
    load_segmentation: bool = True,
    load_nuclei: bool = False,
    segmentation_pixel_size: Number = 1.0,
    fullres_pixel_size: Number | None = None,
    load_bin_mapping: bool = True,
    dataset_name: str | None = None,
    sample_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> InSituData:
    """
    Reads a `10x Visium HD <https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression>`__
    Space Ranger output folder using spatialdata-io.

    Binned counts are imported as ``units`` layers (one per bin size, keys ``square_008um`` etc.,
    square polygons in micrometres), the Space Ranger >= 4 cell segmentation
    (``segmented_outputs/``) as the main cell layer with rasterized cell boundaries, and the
    hires/lowres (and full-resolution, if present) tissue images.

    Args:
        path: Path to the Space Ranger output folder.
        bin_size: Bin size(s) in micrometres to load, e.g. ``8`` or ``[2, 8, 16]``. Default
            (None): 8 and 16 um where available (2 um bins are large and therefore opt-in).
            ``[]`` loads the segmentation only. Loaded 2 um bins cost roughly 0.6-0.7 kB per bin
            in memory (about 6-8 GB for a full 6.5 mm slide), plus a higher transient peak while
            loading - crop early.
        load_segmentation: Load the cell segmentation, if present. Defaults to True.
        load_nuclei: Also import nucleus boundaries. Requires ``load_segmentation=True`` and is
            expensive on full slides (spatialdata-io builds nucleus counts from the 2 um matrix).
            Defaults to False.
        segmentation_pixel_size: Micrometres per pixel of the rasterized cell masks.
            Defaults to 1.0.
        fullres_pixel_size: Micrometres per full-resolution pixel. Defaults to
            ``microns_per_pixel`` from ``scalefactors_json.json``.
        load_bin_mapping: Store the 2 um bin -> cell mapping from ``barcode_mappings.parquet`` in
            ``cells["main"].table.uns["visium_hd_bin_mapping"]`` (columns ``cell_id``,
            ``array_row``, ``array_col``, ``in_nucleus``). Only effective if the segmentation is
            loaded. Used by :func:`insitupy.tl.transfer_obs`. The bin nesting (2 -> 8 -> 16 um) is
            not stored: it is derived from ``array_row``/``array_col`` by integer division.
            Defaults to True.
        dataset_name: Name for the overarching dataset/slide.
        sample_name: Name for the individual sample/ROI within the slide. Defaults to None, which
            uses the Space Ranger sample ID (``library_ids`` in the count matrix), else the
            folder name.
        verbose: Whether to print progress messages. Defaults to True.
        **kwargs: Forwarded to ``spatialdata_io.visium_hd()`` (e.g. ``dataset_id``,
            ``fullres_image_file``, ``filtered_counts_file``). ``bin_size``,
            ``load_segmentations_only``, ``load_nucleus_segmentations`` and ``bins_as_squares``
            are controlled by this reader and raise ``TypeError``.

    Returns:
        InSituData: Object with ``units`` (bins), ``cells["main"]`` (segmentation) and ``images``.

    Raises:
        FileNotFoundError: If ``path`` is not a directory.
        ValueError: If a requested bin size is unavailable, nothing would be loaded, or no pixel
            size can be determined.
        TypeError: If a reader-controlled keyword is passed.
    """
    import re

    from spatialdata_io import visium_hd

    from insitupy.spatialdata.convert import convert_from_foreign_spatialdata

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"No such directory found: {str(path)}")

    controlled = {"load_segmentations_only", "load_nucleus_segmentations", "bins_as_squares"}
    bad = sorted(controlled & set(kwargs))
    if bad:
        raise TypeError(f"read_visium_hd controls {bad} itself; do not pass them via kwargs.")

    # available bin sizes
    available = []
    for d in sorted((path / "binned_outputs").glob("square_*um")) if (path / "binned_outputs").is_dir() else []:
        m = re.fullmatch(r"square_0*(\d+)um", d.name)
        if m and d.is_dir():
            available.append(int(m.group(1)))
    available = sorted(set(available))

    if bin_size is None:
        bins = [b for b in (8, 16) if b in available]
        if not bins and available:
            non2 = [b for b in available if b != 2]
            bins = [non2[0]] if non2 else [available[0]]
            logger.info(f"Default bin sizes 8/16 um not available - using {bins[0]} um.")
    else:
        bins = sorted({int(b) for b in ([bin_size] if isinstance(bin_size, int) else bin_size)})
        missing = [b for b in bins if b not in available]
        if missing:
            raise ValueError(f"Bin size(s) {missing} um not available. Available: {available}.")

    seg_dir = path / "segmented_outputs"
    seg_present = (
        (seg_dir / "cell_segmentations.geojson").is_file()
        and (seg_dir / "filtered_feature_cell_matrix.h5").is_file()
    )
    if load_nuclei and not load_segmentation:
        raise ValueError("load_nuclei=True requires load_segmentation=True.")
    use_seg = load_segmentation and seg_present
    if load_segmentation and not seg_present and verbose:
        logger.info("No segmented_outputs/ found (Space Ranger < 4) - loading bins only.")
    use_nuclei = load_nuclei and use_seg
    if use_nuclei and not (seg_dir / "nucleus_segmentations.geojson").is_file():
        warnings.warn("nucleus_segmentations.geojson not found - nuclei are skipped.")
        use_nuclei = False

    if not bins and not use_seg:
        raise ValueError("Nothing to load: no bins selected/available and no segmentation present.")

    # pixel size
    sf = _read_visium_hd_scalefactors(path)
    ps = fullres_pixel_size if fullres_pixel_size is not None else sf.get("microns_per_pixel")
    if ps is None:
        raise ValueError(
            "No pixel size available: `microns_per_pixel` not found in scalefactors_json.json. "
            "Pass `fullres_pixel_size` (micrometres per full-resolution pixel)."
        )
    ps = float(ps)

    # units order: main first (8 um, else smallest), then ascending
    order = sorted(bins, key=lambda b: (b != 8, b))

    if sample_name is None:
        h5_candidates = [seg_dir / "filtered_feature_cell_matrix.h5"] if use_seg else []
        h5_candidates += [
            path / "binned_outputs" / f"square_{b:03d}um" / "filtered_feature_bc_matrix.h5"
            for b in order
        ]
        sample_name = _default_10x_sample_name(path, h5_candidates)

    if verbose:
        logger.info("Reading Visium HD data with spatialdata-io...")
    sdata = visium_hd(
        path,
        dataset_id=kwargs.pop("dataset_id", None),
        bin_size=bins,
        load_segmentations_only=(len(bins) == 0),
        load_nucleus_segmentations=use_nuclei,
        bins_as_squares=True,
        **kwargs,
    )

    # scale all shapes and bin centroids to micrometres
    if ps != 1.0:
        for key in list(sdata.shapes):
            gdf = sdata.shapes[key]
            gdf["geometry"] = _transform_geoseries(gdf.geometry, lambda c: c * ps)
        for tkey in list(sdata.tables):
            if "spatial" in sdata.tables[tkey].obsm:
                sdata.tables[tkey].obsm["spatial"] = sdata.tables[tkey].obsm["spatial"] * ps

    def _region(tkey):
        region = sdata.tables[tkey].uns["spatialdata_attrs"]["region"]
        return region[0] if isinstance(region, (list, tuple)) else region

    units = {}
    for b in order:
        tkey = f"square_{b:03d}um"
        if tkey not in sdata.tables:
            raise ValueError(f"spatialdata-io did not return table '{tkey}'.")
        sdata.tables[tkey].uns["bin_size_um"] = int(b)
        units[tkey] = {"table_key": tkey, "units_key": _region(tkey), "unit_type": "visium_hd_bin"}

    cells = None
    if use_seg:
        cell_spec = {
            "table_key": "cell_segmentations",
            "cells_key": _region("cell_segmentations"),
            "rasterize_shapes": True,
            "raster_pixel_size": float(segmentation_pixel_size),
        }
        if use_nuclei:
            cell_spec["nucleus_shapes_key"] = _region("nucleus_segmentations")
        cells = {"main": cell_spec}

    # images, resolved by suffix
    images = {}
    for name, suffix, scalef_key in (
        ("hires", "_hires_image", "tissue_hires_scalef"),
        ("lowres", "_lowres_image", "tissue_lowres_scalef"),
        ("fullres", "_full_image", None),
    ):
        keys = [k for k in sdata.images if k.endswith(suffix)]
        if len(keys) != 1:
            continue
        if scalef_key is None:
            pixel_size = ps
        elif scalef_key in sf:
            pixel_size = ps / sf[scalef_key]
        else:
            continue
        images[name] = {"key": keys[0], "pixel_size": pixel_size, "is_rgb": True}

    data = convert_from_foreign_spatialdata(
        sdata=sdata,
        images=images or None,
        cells=cells,
        units=units or None,
        transcripts=None,
        slide_id=dataset_name,
        sample_id=sample_name,
        method_name="visium_hd",
        verbose=verbose,
    )

    if use_seg and load_bin_mapping:
        celltable = data.cells["main"].table
        mapping = _read_visium_hd_bin_mapping(path, celltable.obs_names)
        if mapping is not None:
            celltable.uns["visium_hd_bin_mapping"] = mapping

    return data


def read_any(
    cellular_measurements: dict[str, str | Path | os.PathLike],
    cellular_coordinates: str | Path,
    cellular_metadata: str | Path | None = None,
    cell_boundaries: str | Path | os.PathLike | None = None,
    nucleus_boundaries: str | Path | os.PathLike | None = None,
    images: dict[str, str | Path | os.PathLike] | None = None,
    pixel_size: Number | None = None,
    dataset_name: str | None = "Data 1",
    sample_name: str | None = "Sample 1",
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
        raise ValueError(
            "If `nucleus_boundaries` is given, `cell_boundaries` must be given as well. "
            "If you only have nucleus boundaries, add them as cell boundaries."
            )

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
    cd = CellData(table=adata, boundaries=boundaries)
    data.cells.add_celldata(cd=cd, key="main", is_main=True)

    # Add ImageData if provided
    if images is not None:
        if pixel_size is None:
            raise ValueError("If `images` is given, `pixel_size` must not be None.")

        for img_name, image_path in images.items():
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"No image file found at '{image_path}'.")
            data.images.add_image(image=image_path, channel_names=img_name)

    return data


def read_qupath(
    path: str | os.PathLike | Path,
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
              in the final `AnnData` object in `data.cells.table`.
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
    cd = CellData(table=adata, boundaries=boundaries)
    data.cells.add_celldata(
        cd=cd, key="main", is_main=True
    )

    # --- Add ImageData ---
    data.images.add_image(
        image=image_path,
        channel_names=None,
    )

    # --- Add the data annotation as region ---
    data.regions.add_data(
        data=annot,
        key="data",
        scale_factor=pixel_size
    )

    return data

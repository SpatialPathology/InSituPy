import os
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import Optional, Union

from parse import *

from insitupy import __version__
from insitupy._core.dataclasses import ImageData, MultiCellData
from insitupy.utils.utils import _generate_time_based_uid


def _save_images(imagedata: ImageData,
                 path: Union[str ,os.PathLike],
                 metadata: Optional[dict] = None,
                 images_as_zarr: bool = True,
                 zipped: bool = False,
                 max_resolution: Optional[Number] = None, # in µm per pixel,
                 verbose: bool = False
                 ):
    img_path = (path / "images")

    savepaths = imagedata.save(
        output_folder=img_path,
        as_zarr=images_as_zarr,
        zipped=zipped,
        return_savepaths=True,
        max_resolution=max_resolution,
        verbose=verbose
        )

    if metadata is not None:
        metadata["data"]["images"] = {}
        for n in imagedata.metadata.keys():
            s = savepaths[n]
            # collect metadata
            metadata["data"]["images"][n] = Path(relpath(s, path)).as_posix()

def _save_cells(cells: MultiCellData,
                path,
                metadata,
                boundaries_zipped=False,
                max_resolution_boundaries: Optional[Number] = None, # in µm per pixel
                overwrite=False
                ):
    # create path for cells
    uid = _generate_time_based_uid()
    cells_path = path / "cells" / uid

    # save cells to path and write info to metadata
    cells.save(
        path=cells_path,
        boundaries_zipped=boundaries_zipped,
        max_resolution_boundaries=max_resolution_boundaries,
        overwrite=overwrite
        )

    if metadata is not None:
        try:
            # move old celldata paths to history
            old_path = metadata["data"]["cells"]
        except KeyError:
            pass
        else:
            metadata["history"]["cells"].append(old_path)

        # move new paths to data
        metadata["data"]["cells"] = Path(relpath(cells_path, path)).as_posix()


def _save_transcripts(transcripts, path, metadata):
    # create file path
    trans_path = path / "transcripts"
    trans_path.mkdir(parents=True, exist_ok=True) # create directory
    trans_file = trans_path / "transcripts.parquet"

    # save transcripts as parquet and modify metadata
    transcripts.to_parquet(trans_file)

    if metadata is not None:
        metadata["data"]["transcripts"] = Path(relpath(trans_file, path)).as_posix()

def _save_annotations(annotations, path, metadata):
    uid = _generate_time_based_uid()
    annot_path = path / "annotations" / uid

    # save annotations
    annotations.save(annot_path)

    if metadata is not None:
        try:
            # move old paths to history
            old_path = metadata["data"]["annotations"]
        except KeyError:
            pass
        else:
            metadata["history"]["annotations"].append(old_path)

        # add new paths
        metadata["data"]["annotations"] = Path(relpath(annot_path, path)).as_posix()

def _save_regions(regions, path, metadata):
    uid = _generate_time_based_uid()
    annot_path = path / "regions" / uid

    # save annotations
    regions.save(annot_path)

    if metadata is not None:
        try:
            # move old paths to history
            old_path = metadata["data"]["regions"]
        except KeyError:
            pass
        else:
            metadata["history"]["regions"].append(old_path)

        # add new paths
        metadata["data"]["regions"] = Path(relpath(annot_path, path)).as_posix()

import os
from numbers import Number
from pathlib import Path
from typing import Optional, Union

from insitupy._core._qupath import (_list_insitupy_data_folders,
                                    _read_boundaries_qupath,
                                    _read_measurements_qupath)
from insitupy._core.data import InSituData
from insitupy._core.dataclasses import CellData, ImageData, MultiCellData
from insitupy.experiment.data import InSituExperiment
from insitupy.io.geo import parse_geopandas


def read_qupath_project(
    path: Union[str, os.PathLike, Path],
    pixel_size: Optional[Number],
    method_name: str = "Multiplexed IF"
):
    data_dict = _list_insitupy_data_folders(project_path=path)
    #return data_dict

    exp = InSituExperiment()
    for dataset_name, path_list in data_dict.items():
        print(f"Reading '{dataset_name}'...")
        for p in path_list:
            sample_name = p.name
            annot_path = p / "annotation.geojson"
            measurements_path = p / "measurements.tsv"
            bound_path = p / "cells.geojson"
            image_path = p / "image.ome.tif"

            # read annotation of data to shift coordinates to origin
            if not annot_path.exists():
                raise FileNotFoundError(f"No annotation file found at '{annot_path}'.")

            annot = parse_geopandas(annot_path)

            if len(annot) > 1:
                raise ValueError(f"More than one annotation found in '{annot_path}'.")

            xmin = annot["geometry"].item().bounds[0] * pixel_size
            ymin = annot["geometry"].item().bounds[1] * pixel_size

            # --- Read cellular measurements ---
            if not measurements_path.exists():
                raise FileNotFoundError(f"No measurements file found at '{measurements_path}'.")

            adata = _read_measurements_qupath(
                measurements_path, xmin=xmin, ymin=ymin
                )

            # --- Read cellular boundaries ---
            if not bound_path.exists():
                raise FileNotFoundError(f"No boundaries file found at '{bound_path}'.")

            boundaries = _read_boundaries_qupath(
                bound_path,
                object_ids=adata.obs["Object ID"].values,
                cell_names=adata.obs_names,
                xmin=xmin, ymin=ymin,
                pixel_size=pixel_size
                )

            # --- Check image path ---
            if not image_path.exists():
                raise FileNotFoundError(f"No image file found at '{image_path}'.")

            # --- Create InSituData object ---
            data = InSituData(
                path=None,
                metadata={
                    "method": method_name,
                    "method_params": {
                        "pixel_size": pixel_size
                    }
                },
                slide_id=dataset_name,
                sample_id=sample_name
            )

            # --- Add CellData ---
            cd = CellData(matrix=adata, boundaries=boundaries)
            data.cells = MultiCellData()
            data.cells.add_celldata(
                cd=cd, key="main", is_main=True
            )

            data.images = ImageData()
            data.images.add_image(
                image=image_path,
                name="IF",
            )

            # --- Add to InSituExperiment ---
            exp.add(data)

    return exp
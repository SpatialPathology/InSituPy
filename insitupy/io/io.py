import os
from math import ceil
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Union

import dask.array as da
import pandas as pd
import scanpy as sc
import toml
import zarr
from rasterio.features import rasterize
from zarr.errors import ArrayNotFoundError

from insitupy._core.dataclasses import (AnnotationsData, BoundariesData,
                                        CellData, RegionsData, ShapesData, MultiCellData)
from insitupy.io.baysor import read_baysor_polygons
from insitupy.io.files import read_json
from insitupy.utils.utils import convert_to_list


def read_baysor_cells(
    baysor_output: Union[str, os.PathLike, Path],
    pixel_size: Number = 1 # the pixel size is usually 1 since baysor runs on the µm coordinates
    ) -> CellData:

    # convert to pathlib path
    baysor_output = Path(baysor_output)

    # read baysor metadata
    tomlfile = baysor_output / "segmentation_params.dump.toml"
    with open(tomlfile, 'r') as f:
        baysor_config = toml.load(f)

    # read matrix
    print("Parsing count matrix...", flush=True)
    loomfile = baysor_output / "segmentation_counts.loom"
    matrix = sc.read_loom(loomfile)

    # set indices for .obs and .var
    matrix.obs = matrix.obs.reset_index().set_index("Name")
    matrix.obs["CellID"] = matrix.obs["CellID"].astype(float).astype(int) # convert cell id to int
    matrix.var.set_index("Name", inplace=True)

    # remove unassigned codewords from genes and obs entries with an NaN in any column
    varmask = ~matrix.var_names.str.startswith("UnassignedCodeword")
    obsmask = ~matrix.obs.isna().any(axis=1)
    matrix = matrix[obsmask, varmask].copy()

    # set spatial coordinates
    matrix.obsm["spatial"] = matrix.obs[["x", "y"]].values
    matrix.obs.drop(["x", "y"], axis=1, inplace=True) # drop the coordinate columns

    # read polygons
    print("Reading segmentation masks", flush=True)
    print("\tRead polygons", flush=True)
    jsonfile = baysor_output / "segmentation_polygons.json"
    df = read_baysor_polygons(jsonfile)

    # remove polygons of cells that have been removed in the matrix
    df = df[df.cell.astype(int).isin(matrix.obs["CellID"])]

    # determine dimensions of dataset based on polygons
    polygon_bounds = df.geometry.bounds
    xmax = ceil(polygon_bounds.loc[:, "maxx"].max())
    ymax = ceil(polygon_bounds.loc[:, "maxy"].max())
    # xmax = ceil(matrix.obsm['spatial'][:, 0].max() + 15)
    # ymax = ceil(matrix.obsm['spatial'][:, 1].max() + 15)

    # generate a segmentation mask
    print("\tConvert polygons to segmentation mask", flush=True)
    img = rasterize(list(zip(df["geometry"], df["cell"])), out_shape=(ymax,xmax))

    # convert to dask array
    img = da.from_array(img)

    # create boundaries object
    cell_ids = da.from_array(matrix.obs["CellID"].values) # extract cell ids from adata
    seg_mask_value = da.from_array(sorted(df["cell"]))
    boundaries = BoundariesData(cell_ids=cell_ids, seg_mask_value=seg_mask_value)
    boundaries.add_boundaries(data={f"cellular": img}, pixel_size=pixel_size)

    celldata = CellData(matrix=matrix, boundaries=boundaries, config=baysor_config)

    return celldata


def read_baysor_transcripts(
    baysor_output: Union[str, os.PathLike, Path]
    ) -> pd.DataFrame:

    # convert to pathlib path
    baysor_output = Path(baysor_output)

    # read transcripts from Baysor results
    print("Parsing transcripts data...", flush=True)

    print("\tRead data", flush=True)
    segcsv_file = baysor_output / "segmentation.csv"
    baysor_transcript_dataframe = pd.read_csv(segcsv_file)

    # reshaping
    transcript_id_col = [elem for elem in ["transcript_id", "molecule_id"] if elem in baysor_transcript_dataframe.columns][0]
    baysor_transcript_dataframe = baysor_transcript_dataframe.set_index(transcript_id_col)
    return baysor_transcript_dataframe


def read_multicelldata(
        path: Union[str, os.PathLike, Path],
        old: Optional[bool] = False,
        path_upper: Optional[Union[str, os.PathLike, Path]] = None,
        alt_path_dict: Optional[dict] = None,
    ) -> MultiCellData:
    path = Path(path)
    mcd = MultiCellData()
    if not old:
        celldata_metadata = read_json(path / ".multicelldata")
        for key in celldata_metadata["all_keys"]:
            cd = read_celldata(path / key)
            mcd.add_celldata(cd=cd, key=key, is_main=(key == celldata_metadata["key_main"]))
    else:
        cd = read_celldata(path)
        mcd.add_celldata(cd=cd, key="main", is_main=True)
        if path_upper is not None and alt_path_dict is not None:
            path_upper = Path(path_upper)
            for k, p in alt_path_dict.items():
                cd = read_celldata(path=path_upper / p)
                mcd.add_celldata(cd=cd, key=k)
    return mcd

def read_celldata(
    path: Union[str, os.PathLike, Path],
    ) -> CellData:
    # read metadata
    path = Path(path)
    celldata_metadata = read_json(path / ".celldata")

    # read matrix data
    matrix = sc.read_h5ad(path / celldata_metadata["matrix"])

    # get path of boundaries data
    bound_path = path / celldata_metadata["boundaries"]

    # check whether it is zipped or not
    suffix = bound_path.name.split(".", maxsplit=1)[-1]

    # read cell ids and seg_mask_values
    cell_ids = da.from_zarr(bound_path, component="cell_id")

    try:
        # in older datasets sometimes seg_mask_value is missing
        seg_mask_value = da.from_zarr(bound_path, component="seg_mask_value")
    except ArrayNotFoundError:
        seg_mask_value = None

    # create boundaries data object
    boundaries = BoundariesData(cell_ids=cell_ids, seg_mask_value=seg_mask_value)

    # retrieve the boundaries data
    bound_data = {}
    meta = {}
    zipped = True if suffix == "zarr.zip" else False
    with zarr.ZipStore(bound_path, mode='r') if zipped else zarr.DirectoryStore(bound_path) as dirstore:
        for k in dirstore.listdir("masks"):
            if not k.startswith("."):
                # iterate through subresolutions
                subresolutions = dirstore.listdir(f"masks/{k}")

                if ".zarray" in subresolutions:
                    if zipped:
                        bound_data[k] = da.from_zarr(dirstore).persist() # persist is only needed in case of zipped zarrs
                    else:
                        bound_data[k] = da.from_zarr(dirstore)
                else:
                    # it is stored as pyramid -> initialize a list for the pyramid
                    bound_data[k] = []
                    for subres in subresolutions:
                        if not subres.startswith("."):
                            # append the pyramid to the list
                            if zipped:
                                bound_data[k].append(
                                    da.from_zarr(dirstore, component=f"masks/{k}/{subres}").persist()
                                    )
                            else:
                                bound_data[k].append(
                                    da.from_zarr(dirstore, component=f"masks/{k}/{subres}")
                                    )

                # retrieve boundaries metadata
                store = zarr.open(dirstore)
                meta[k] = store[f"masks/{k}"].attrs.asdict()

    # add boundaries
    boundaries.add_boundaries(data=bound_data,
                              pixel_size=meta[list(meta.keys())[0]]["pixel_size"]
                              )

    # try to extract configuration
    try:
        config = celldata_metadata["config"]
    except KeyError:
        config = {}

    # create CellData object
    celldata = CellData(matrix=matrix, boundaries=boundaries, config=config)

    return celldata


# def read_shapesdata(
#     path: Union[str, os.PathLike, Path],
# ):
#     path = Path(path)
#     metadata = read_json(path / "metadata.json")
#     keys = metadata.keys()
#     files = [path / f"{k}.geojson" for k in keys]
#     data = RegionsData(files, keys)

#     for k, f in zip(keys, files):
#         data.add_data(data=f, key=k)

#     # overwrite metadata
#     data.metadata = metadata
#     return data


def read_shapesdata(
    path: Union[str, os.PathLike, Path],
    mode: Literal["annotations", "regions", "shapes"],
    scale_factor: Optional[Number] = None
):
    path = Path(path)

    # e.g. when reading from a shapesdata object, it is assumed that it was saved as µm
    if scale_factor is None:
        scale_factor = 1

    # read metadata and retrieve keys and files from it
    metadata = read_json(path / "metadata.json")
    keys = metadata.keys()
    files = [path / f"{k}.geojson" for k in keys]

    # check which type of ShapesData is read here
    if mode == "annotations":
        data = AnnotationsData()
    elif mode == "regions":
        data = RegionsData()
    elif mode == "shapes":
        data = ShapesData()
    else:
        ValueError(f"Unknown `mode`: {mode}")

    # make sure files and keys are a list
    files = convert_to_list(files)
    keys = convert_to_list(keys)

    for k, f in zip(keys, files):
        data.add_data(data=f, key=k,
                      scale_factor=scale_factor
                      )

    # overwrite metadata
    data.metadata = metadata
    return data



import os
from math import ceil
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import Literal, Optional, Union
from warnings import warn

import dask.array as da
import numpy as np
import scanpy as sc
import toml
import zarr
from parse import *
from zarr.errors import ArrayNotFoundError

from insitupy import __version__
from insitupy._io.files import read_json
from insitupy.dataclasses._segmentations import _read_baysor_polygons
from insitupy.dataclasses.dataclasses import (AnnotationsData, BoundariesData,
                                              CellData, ImageData,
                                              MultiCellData, RegionsData,
                                              ShapesData)
from insitupy.utils.utils import (_generate_time_based_uid,
                                  convert_int_to_xenium_hex, convert_to_list)


def read_baysor_cells(
    baysor_output: Union[str, os.PathLike, Path],
    pixel_size: Number = 1 # the pixel size is usually 1 since baysor runs on the µm coordinates
    ) -> CellData:
    try:
        from rasterio.features import rasterize
    except ImportError:
        raise ImportError("This function requires the rasterio package, please install with `pip install rasterio`.")

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
    df = _read_baysor_polygons(jsonfile)

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

    try:
        # read cell ids and seg_mask_values
        cell_names = da.from_zarr(bound_path, component="cell_names")
    except ArrayNotFoundError:
        # if cell names is not found, the data might come from an older InSituPy version which contained a cell_id instead
        try:
            # read cell ids and seg_mask_values
            cell_ids = da.from_zarr(bound_path, component="cell_id").compute()
            cell_names = np.array([convert_int_to_xenium_hex(elem[0], elem[1]) for elem in cell_ids])
        except ArrayNotFoundError:
            # if no cell_id is present, this means that the data is from a new InSituPy version which is good.
            pass

    try:
        # in older datasets sometimes seg_mask_value is missing
        seg_mask_value = da.from_zarr(bound_path, component="seg_mask_value")
    except ArrayNotFoundError:
        warn("No `seg_mask_value` component found in boundaries zarr storage. This can lead to problems when syncing `.boundaries` and `.matrix`.")
        seg_mask_value = None

    # initialize boundaries data object
    boundaries = BoundariesData(cell_names=cell_names, seg_mask_value=seg_mask_value)

    # retrieve the boundaries data
    bound_data = {}
    meta = {}
    zipped = True if suffix == "zarr.zip" else False
    # with zarr.ZipStore(bound_path, mode='r') if zipped else zarr.DirectoryStore(bound_path) as dirstore:

    if zipped:
        dirstore = zarr.storage.ZipStore(bound_path, mode="r")
    else:
        dirstore = zarr.storage.LocalStore(bound_path)

    # open zarr group
    root = zarr.open_group(store=dirstore, mode='r')

    # for k in dirstore.listdir("masks"):
    #     if not k.startswith("."):
    for k in ["cells", "nuclei"]:
        #if (bound_path / "masks" / k).exists():
        comp = f"masks/{k}"
        if comp in root:
            # iterate through subresolutions
            # subresolutions = dirstore.listdir(f"masks/{k}")
            subresolutions = sorted(root[comp].keys())

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
                                da.from_zarr(dirstore, component=f"{comp}/{subres}").persist()
                                )
                        else:
                            bound_data[k].append(
                                da.from_zarr(dirstore, component=f"{comp}/{subres}")
                                )

            # retrieve boundaries metadata
            store = zarr.open(dirstore)
            meta[k] = store[f"masks/{k}"].attrs.asdict()

    cell_boundaries = bound_data["cells"]
    if "nuclei" in bound_data:
        nuclei_boundaries = bound_data["nuclei"]
    else:
        nuclei_boundaries = None

    # add boundaries
    boundaries.add_boundaries(
        #data=bound_data,
        cell_boundaries=cell_boundaries,
        nuclei_boundaries=nuclei_boundaries,
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
    # metadata = read_json(path / "metadata.json")
    # keys = metadata.keys()
    # files = [path / f"{k}.geojson" for k in keys]
    files_dict = {f.stem: f for f in path.glob("*.geojson") if f.stem != "metadata"}

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
    # files = convert_to_list(files)
    # keys = convert_to_list(keys)

    # for k, f in zip(keys, files):
    for k, f in files_dict.items():
        data.add_data(
            data=f, key=k,
            scale_factor=scale_factor
            )

    # overwrite metadata
    # data.metadata = metadata
    return data

def read_multicelldata(
        path: Union[str, os.PathLike, Path],
        path_upper: Optional[Union[str, os.PathLike, Path]] = None,
        alt_path_dict: Optional[dict] = None,
    ) -> MultiCellData:
    if os.path.exists(path / ".multicelldata"):
        old = False
    elif os.path.exists(path / ".celldata"):
        old = True
    else:
        raise FileNotFoundError(f"Metadata file for cells dimension in {path} was not found.")
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

    #if metadata is not None:
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

    #if metadata is not None:
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

    #if metadata is not None:
    metadata["data"]["transcripts"] = Path(relpath(trans_file, path)).as_posix()


def _save_features(features, path, metadata):
    # create file path
    features_path = path / "features"
    features_path.mkdir(parents=True, exist_ok=True) # create directory
    
    # save shapes as parquet
    shapes_file = features_path / "shapes.parquet"
    features.shapes.to_parquet(shapes_file)
    
    # save data as h5ad if present
    if features.data is not None:
        data_file = features_path / "data.h5ad"
        features.data.write_h5ad(data_file)
    
    # save metadata
    meta_dict = {
        "pixel_size": features.pixel_size,
        "feature_type": features.feature_type
    }
    meta_file = features_path / "metadata.json"
    import json
    with open(meta_file, 'w') as f:
        json.dump(meta_dict, f)

    #if metadata is not None:
    metadata["data"]["features"] = Path(relpath(features_path, path)).as_posix()


def _save_annotations(annotations, path, metadata):
    uid = _generate_time_based_uid()
    annot_path = path / "annotations" / uid

    # save annotations
    annotations.save(annot_path)

    #if metadata is not None:
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

    #if metadata is not None:
    try:
        # move old paths to history
        old_path = metadata["data"]["regions"]
    except KeyError:
        pass
    else:
        metadata["history"]["regions"].append(old_path)

    # add new paths
    metadata["data"]["regions"] = Path(relpath(annot_path, path)).as_posix()

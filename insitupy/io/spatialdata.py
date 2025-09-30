try:
    from spatialdata import SpatialData
except ImportError:
    raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")

from collections import defaultdict
from typing import List, Literal, Optional, Union
from warnings import warn

import dask.dataframe as dd
import numpy as np
from anndata import AnnData
from spatialdata._core.validation import check_valid_name
from spatialdata.models import (Image2DModel, Labels2DModel, PointsModel,
                                ShapesModel, TableModel)
from spatialdata.transformations.transformations import Identity, Scale
from xarray import DataArray

from insitupy._constants import (DEFAULT_CHUNK_SIZE_X, DEFAULT_CHUNK_SIZE_Y,
                                 MODALITIES, SAMPLE_STR)
from insitupy._core.data import InSituData
from insitupy.images.axes import ImageAxes
from insitupy.utils.utils import convert_to_list


def _generate_key(
    sample_id: str,
    modality: Literal[MODALITIES],
    locator: Optional[Union[str, tuple, List]]
    ):
    if not modality.lower() in MODALITIES:
        raise ValueError(f"Modality '{modality}' not recognized. Choose from {MODALITIES}.")

    if modality == "transcripts":
        assert locator is None, "Locator must be None for modality 'transcripts'."
    else:
        assert locator is not None, f"Locator cannot be None for modality '{modality}'."

    if sample_id is None:
        sample_part = ""
    else:
        sample_part = f"{SAMPLE_STR}.{sample_id}.."

    if locator is not None:
        locator = convert_to_list(locator)

        locator_checked = []
        for elem in locator:
            try:
                check_valid_name(elem)
            except ValueError as e:
                raise ValueError(f"Name '{elem}' does not meet the naming requirements of SpatialData. {e}")

            if "." in elem or "-" in elem:
                warn(f"Replacing '.' and '-' in '{elem}' with '_' to meet naming requirements.")
                elem = elem.replace(".", "_").replace("-", "_")

            locator_checked.append(elem)

        key = f"{sample_part}{modality.upper()}.{'.'.join(locator_checked)}"
    else:
        key = f"{sample_part}{modality.upper()}"

    # check key for validity
    check_valid_name(key)
    return key

def _transform_anndata(
    adata: AnnData,
    #cells_as_circles: bool = True
    cells_key: str,
    cell_area_key: Optional[str] = "cell_area"
    ):

    adata = adata.copy()
    region_str = "region"
    attrs = {}
    #attrs["instance_key"] = "cell_id" if cells_as_circles else "cell_labels"
    attrs["instance_key"] = "cell_id"
    adata.obs["cell_id"] = adata.obs.index
    #attrs[region_str] = "cell_circles" if cells_as_circles else "cell_labels"
    attrs[region_str] = cells_key
    adata.obs[region_str] = cells_key
    adata.obs[region_str] = adata.obs[region_str].astype("category")
    attrs["region_key"] = region_str
    adata.uns["spatialdata_attrs"] = attrs

    if cell_area_key is not None:
        try:
            cell_areas = adata.obs[cell_area_key].to_numpy()
        except KeyError:
            print(f"Key '{cell_area_key}' not found in AnnData. Skipped generation of sized circles.")
            circles_sized = None
        else:
            radius = np.sqrt(cell_areas / np.pi)
            circles_sized = ShapesModel.parse(
                    adata.obsm["spatial"].copy(),
                    geometry=0, # means "Circles" (3 is Polygon, 6 is MultiPolygon)
                    radius=radius,
                    index=adata.obs.index.copy(),
            )
    else:
        circles_sized = None

    circles = ShapesModel.parse(
        adata.obsm["spatial"].copy(),
        geometry=0,
        radius=5,
        index=adata.obs.index.copy()
        )

    return adata, circles_sized, circles



def _transform_images(
    xd: InSituData,
    levels: int = 5,
    sample_id: Optional[str] = None
    ):
    images = {}
    if xd.images is not None:
        for name in xd.images.names:
            image_list =  xd.images[name]
            meta = xd.images.metadata[name]
            pixel_size = meta["pixel_size"]
            transformations = {"global": Scale([pixel_size, pixel_size], axes=("x", "y"))}
            img = image_list[0]
            axes_isp = xd.images.metadata[name]["axes"]
            axes_config = ImageAxes(axes_isp)
            is_rgb = xd.images.metadata[name]["rgb"]

            dict_key = _generate_key(
                sample_id=sample_id,
                modality="images",
                locator=name
            )
            if is_rgb:
                axes = tuple(list(axes_isp.lower().replace("s", "c")))
                array = DataArray(
                    data=image_list[0],
                    name=name,
                    dims=axes
                    )
                images[dict_key] = Image2DModel.parse(
                    array,
                    rgb=True,
                    scale_factors=[2 for _ in range(levels)],
                    transformations=transformations,
                    chunks=(3, DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X)
                    )
            else:
                c_dim = axes_config.C if axes_config.C is not None else 1

                img_reshaped = img.reshape(
                    c_dim,
                    img.shape[axes_config.Y],
                    img.shape[axes_config.X]
                )
                array = DataArray(
                    data=img_reshaped,
                    name=name,
                    dims=("c", "y", "x")
                    )
                images[dict_key] = Image2DModel.parse(
                    array,
                    scale_factors=[2 for _ in range(levels)],
                    transformations=transformations,
                    chunks=(1, DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X)
                    )
    return images

def _transform_transcripts(
    xd: InSituData,
    sample_id: Optional[str] = None
    ):
    points = {}
    if xd.transcripts is not None:
        df = xd.transcripts
        #scale = Scale([1, 1, 1], axes=("x", "y", "z"))
        parsed_points = PointsModel.parse(
            df,
            coordinates={"x": "x_location", "y": "y_location", "z": "z_location"},
            feature_key="feature_name",
            instance_key="cell_id",
            #transformations={"global": scale},
            sort=True
            )

        dict_key = _generate_key(
            sample_id=sample_id,
            modality="transcripts",
            locator=None
        )

        points = {dict_key: parsed_points}
    return points


def _transform_matrix(
    xd: InSituData,
    sample_id: Optional[str] = None
    ):
    tables, cell_shapes = {}, {}
    #if xd.cells is not None and xd.cells.matrix is not None:
    if xd.cells is not None:
        for cell_key in xd.cells.keys():
            if xd.cells[cell_key].matrix is not None:
                tables_key = _generate_key(
                    sample_id=sample_id,
                    modality="cells",
                    locator=[cell_key, "matrix"]
                )

                circles_dict_key = _generate_key(
                    sample_id=sample_id,
                    modality="cells",
                    locator=[cell_key, "circles"]
                )

                adata, circles_sized, circles = _transform_anndata(
                    xd.cells[cell_key].matrix,
                    cells_key=circles_dict_key
                    )

                # see https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/tables.html#construct-a-table-annotating-1-or-more-spatialelements
                tables[tables_key] = TableModel.parse(adata)
                cell_shapes[circles_dict_key] = circles

                if circles_sized is not None:
                    circles_sized_dict_key = _generate_key(
                        sample_id=sample_id,
                        modality="cells",
                        locator=[cell_key, "circles_sized"]
                    )

                    # add sized circles
                    cell_shapes[circles_sized_dict_key] = circles_sized

    return tables, cell_shapes

def _transform_cell_boundaries(
    xd: InSituData,
    n_levels: int = 5,
    sample_id: Optional[str] = None
    ):
    cell_boundaries = {}
    #if xd.cells is not None and xd.cells.boundaries is not None:
    if xd.cells is not None:
        for cell_key in xd.cells.keys():
            if xd.cells[cell_key].boundaries is not None:
                celldata = xd.cells[cell_key]
                meta = celldata.boundaries.metadata[name]
                pixel_size = meta["pixel_size"]
                transformations = {"global": Scale([pixel_size, pixel_size], axes=("x", "y"))}
                for name in celldata.boundaries.metadata.keys():
                    # get list of available boundaries
                    labels_list =  celldata.boundaries[name]

                    if isinstance(labels_list, list):
                        top_array = labels_list[0]
                    array = DataArray(data=top_array, name=name, dims=("y", "x"))

                    dict_key = _generate_key(
                        sample_id=sample_id,
                        modality="cells",
                        locator=[cell_key, "boundaries", name]
                    )

                    cell_boundaries[dict_key] = Labels2DModel.parse(
                        array,
                        scale_factors=[2 for _ in range(n_levels)],
                        transformations=transformations,
                        chunks=(DEFAULT_CHUNK_SIZE_Y, DEFAULT_CHUNK_SIZE_X)
                        )
    return cell_boundaries

def _transform_annotations(
    xd: InSituData,
    sample_id: Optional[str] = None
    ):
    shapes = {}
    if xd.annotations is not None:
        for key in xd.annotations.metadata.keys():
            gdf = ShapesModel.parse(
                xd.annotations[key],
                )

            dict_key = _generate_key(
                sample_id=sample_id,
                modality="annotations",
                locator=key
            )

            shapes[dict_key] = gdf
    return shapes



def _transform_regions(
    xd: InSituData,
    sample_id: Optional[str] = None
    ):
    shapes = {}
    if xd.annotations is not None:
        for key in xd.regions.metadata.keys():
            gdf = ShapesModel.parse(
                xd.regions[key],
                )
            dict_key = _generate_key(
                sample_id=sample_id,
                modality="regions",
                locator=key
            )

            shapes[dict_key] = gdf
    return shapes

def _merge_dicts_with_warning(*dicts):
    merged = {}
    seen_keys = set()
    for d in dicts:
        for key in d:
            if key in seen_keys:
                print(f"Warning: Duplicate key detected - '{key}'")
            seen_keys.add(key)
        merged.update(d)
    return merged

def _check_case_insensitive_conflicts(keys):
    keys = convert_to_list(keys)
    grouped = defaultdict(list)

    for key in keys:
        grouped[key.lower()].append(key)

    conflicts = {k: v for k, v in grouped.items() if len(set(v)) > 1}

    if conflicts:
        message_lines = ["Case-insensitive key conflicts detected:"]
        for lower_key, variants in conflicts.items():
            message_lines.append(f"  - '{lower_key}': {variants}")
        message_lines.append(
            "\nThese conflicts can lead to problems when saving the SpatialData object, "
            "as some tools treat keys in a case-insensitive manner."
        )
        warn("\n".join(message_lines), category=UserWarning)
    else:
        print("No case-insensitive conflicts found.")

def convert_to_spatialdata_dict(data, levels: int = 5):

    """
    Converts an InSituData object to a dictionary for SpatialData object.

    This function integrates various data elements such as images, labels, transcripts, and annotations
    into a SpatialData object. It requires the spatialdata framework to be installed.

    Raises:
        ImportError: If the spatialdata framework is not installed.

    Returns:
        Dict: a dictionary with all modalities saved in SpatialData format.
    """
    # create SpatialData dictionary
    transcripts = _transform_transcripts(data)
    tables, cell_shapes = _transform_matrix(data)
    annotations = _transform_annotations(data)
    regions = _transform_regions(data)
    images = _transform_images(data, levels)
    labels = _transform_cell_boundaries(data)
    merged_dict = _merge_dicts_with_warning(transcripts, tables, cell_shapes, annotations, regions, images, labels)

    # check whether there are keys in the dictionary that could later lead to problems saving the data
    _check_case_insensitive_conflicts(merged_dict.keys())

    return merged_dict

def convert_to_spatialdata(data, levels: int = 5):

    """
    Converts an InSituData object to a SpatialData object.

    This function integrates various data elements such as images, labels, transcripts, and annotations
    into a SpatialData object. It requires the spatialdata framework to be installed.

    Returns:
        SpatialData: A SpatialData object containing the integrated data elements.

    """

    sd_dict = convert_to_spatialdata_dict(data, levels=levels)
    sdata = SpatialData.from_elements_dict(sd_dict)
    return sdata

def load_from_spatialdata(spatialdata_path, pixel_size):

    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import zarr
    from anndata import read_zarr
    from dask.array import transpose
    from dask.dataframe import read_parquet
    from ome_zarr.io import ZarrLocation
    from ome_zarr.reader import Label, Multiscales, Reader

    from insitupy.dataclasses import BoundariesData, CellData, ImageData


    def read_helper_images_labels(f_elem_store, type):
        nodes = []
        image_loc = ZarrLocation(f_elem_store)
        if image_loc.exists():
            image_reader = Reader(image_loc)()
            image_nodes = list(image_reader)
            if len(image_nodes):
                for node in image_nodes:
                    if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and (
                        type == "image"
                        and np.all([not isinstance(spec, Label) for spec in node.specs])
                        or type == "labels"
                        and np.any([isinstance(spec, Label) for spec in node.specs])
                    ):
                        nodes.append(node)
        assert len(nodes) == 1
        node = nodes[0]
        datasets = node.load(Multiscales).datasets
        multiscales = node.load(Multiscales).zarr.root_attrs["multiscales"]
        axes = [i["name"] for i in node.metadata["axes"]]
        assert len(multiscales) == 1
        if len(datasets) >= 1:
            multiscale_image = []
            for _, d in enumerate(datasets):
                data = node.load(Multiscales).array(resolution=d, version=None)
                if data.shape[0] == 1:
                    data = data.reshape(data.shape[1:])
                    axes = axes[1:]
                elif data.shape[0] == 3:
                    data = transpose(data, (1, 2, 0))
                multiscale_image.append(data)
            return multiscale_image, axes

    xd = InSituData(Path("./data1"), {"metadata_file": "meta.txt", "xenium":{"pixel_size": pixel_size}}, "", "", "")
    path = Path(spatialdata_path)
    f = zarr.open(path, mode="r")
    bd = BoundariesData(None, None)


    if "labels" in f:
        group = f["labels"]
        boundaries_dict = {}
        for name in group:
            if Path(name).name.startswith("."):
                continue
            f_elem = group[name]
            f_elem_store = os.path.join(f.store.path, f_elem.path)
            image, axes = read_helper_images_labels(f_elem_store, "labels")
            boundaries_dict[name] = image[0]
        bd.add_boundaries(boundaries_dict, pixel_size=pixel_size)

    if "tables" in f:
        group = f["tables"]
        i = 0
        for name in group:
            f_elem = group[name]
            f_elem_store = os.path.join(f.store.path, f_elem.path)
            cdata = CellData(read_zarr(f_elem_store), bd)
            if len(group) == 1 or i == 0:
                setattr(xd, "cells", cdata)
            else:
                xd.add_alt(cdata, key_to_add=name)
            i += 1

    if "points" in f:
        group = f["points"]
        for name in group:
            if name == "transcripts":
                f_elem = group[name]
                f_elem_store = os.path.join(f.store.path, f_elem.path)
                points = read_parquet(f_elem_store)
                pdf = points.compute()

                # Rename columns to match the new structure
                pdf = pdf.rename(columns={
                    'x': 'x',
                    'y': 'y',
                    'z': 'z',
                    'feature_name': 'gene',
                    'qv': 'qv',
                    'overlaps_nucleus': 'overlaps_nucleus',
                    'cell_id': 'xenium',
                    'transcript_id': 'transcript_id'
                })

                # Reorder columns to match the new structure
                pdf = pdf[['x', 'y', 'z', 'gene', 'qv', 'overlaps_nucleus', 'xenium', 'transcript_id']]

                # Set 'transcript_id' as the index
                pdf = pdf.set_index('transcript_id')

                # Set the MultiIndex for columns
                pdf.columns = pd.MultiIndex.from_tuples([
                    ('coordinates', 'x'),
                    ('coordinates', 'y'),
                    ('coordinates', 'z'),
                    ('properties', 'gene'),
                    ('properties', 'qv'),
                    ('properties', 'overlaps_nucleus'),
                    ('cell_id', 'xenium')
                ])
                setattr(xd, "transcripts", pdf)
    if "images" in f:
        group = f["images"]
        setattr(xd, "images", ImageData())
        for name in group:
            if Path(name).name.startswith("."):
                continue
            f_elem = group[name]
            f_elem_store = os.path.join(f.store.path, f_elem.path)
            image, axes = read_helper_images_labels(f_elem_store, "image")
            xd.images.add_image(image[0], name=name, axes=axes, pixel_size=pixel_size, ome_meta={'PhysicalSizeX': pixel_size})
    return xd
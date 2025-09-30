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
                for name in celldata.boundaries.metadata.keys():
                    meta = celldata.boundaries.metadata[name]
                    pixel_size = meta["pixel_size"]
                    transformations = {"global": Scale([pixel_size, pixel_size], axes=("x", "y"))}

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

    """
    Load an InSituData object from a SpatialData Zarr store, reversing the
    naming and partitioning logic used in `convert_to_spatialdata`.

    Args:
        spatialdata_path (str or Path): Path to the SpatialData .zarr directory
            (root store created by `SpatialData.write()`).
        pixel_size (float): Pixel size in the same units used during export.
            This value is applied to image and label metadata.

    Returns:
        InSituData: A reconstructed InSituData instance containing:
            - images (ImageData): Loaded from the "images" group.
            - cells (dict[str, CellData]): Each entry includes a cell matrix
            (AnnData) and associated boundaries (BoundariesData).
            - transcripts (pandas.DataFrame): Transcript coordinates and
            attributes, normalized to match the schema expected by
            `_transform_transcripts`.

    Raises:
        FileNotFoundError: If the specified SpatialData path does not exist.
        ValueError: If required coordinate columns are missing in the points
            tables or if the Zarr structure is invalid.

    Notes:
        - Supports keys generated by `_generate_key`, including optional
        sample prefixes (e.g., "sample.<id>..IMAGES.<name>").
        - Ignores derived elements such as "circles" and "circles_sized"
        because they can be regenerated from the cell matrix.
        - Multiple samples are merged into a single InSituData object. If
        per-sample separation is required, this function should be extended
        to return a mapping of sample IDs to InSituData instances.
    """

    import os
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import zarr
    from anndata import read_zarr
    from dask.array import transpose
    from dask.dataframe import read_parquet
    from ome_zarr.io import ZarrLocation
    from ome_zarr.reader import Label, Multiscales, Reader

    # Local imports (from insitupy)
    from insitupy.dataclasses import BoundariesData, CellData, ImageData

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _parse_key(name: str):
        """Parse element key created by `_generate_key`.

        Patterns:
          - "IMAGES.<image_name>"
          - "TRANSCRIPTS"
          - "CELLS.<cell_key>.matrix"
          - "CELLS.<cell_key>.circles" (ignored here)
          - "CELLS.<cell_key>.circles_sized" (ignored here)
          - "CELLS.<cell_key>.boundaries.<boundary_name>"
          - With sample: "sample.<id>..<MODALITY>.<...>"
        Returns: (sample_id or None, modality (upper), locator parts list)
        """
        parts = name.split('.')
        sample_id = None
        if len(parts) >= 4 and parts[0] == SAMPLE_STR and parts[2] == '':
            # sample.<id>..MODALITY[.locators]
            sample_id = parts[1]
            modality = parts[3]
            locators = parts[4:]
        else:
            modality = parts[0]
            locators = parts[1:]
        return sample_id, modality.upper(), locators

    def _read_multiscales(store_path: str, kind: str):
        """Read an image/labels multiscale stack and return [pyr_levels], axes.
        kind in {"image", "labels"}.
        """
        nodes = []
        loc = ZarrLocation(store_path)
        if not loc.exists():
            return None, None
        for node in Reader(loc)():
            has_ms = any(isinstance(spec, Multiscales) for spec in node.specs)
            is_label = any(isinstance(spec, Label) for spec in node.specs)
            if not has_ms:
                continue
            if kind == 'image' and not is_label:
                nodes.append(node)
            if kind == 'labels' and is_label:
                nodes.append(node)
        if not nodes:
            return None, None
        # There should be one node for each element
        node = nodes[0]
        datasets = node.load(Multiscales).datasets
        axes = [a['name'] for a in node.metadata['axes']]
        pyramid = []
        for d in datasets:
            arr = node.load(Multiscales).array(resolution=d, version=None)
            # If OME-NGFF stores as (C,Y,X) and C==1, drop C.
            if arr.shape[0] == 1:
                arr = arr.reshape(arr.shape[1:])
                # keep axes in sync
                if axes and axes[0] in ('c', 'C'):
                    axes_local = axes[1:]
                else:
                    axes_local = axes
            elif arr.shape[0] == 3:
                # likely RGB stored as (C,Y,X) -> (Y,X,C)
                arr = transpose(arr, (1, 2, 0))
                # and re-order axes accordingly
                if axes and axes[:3] == ['c', 'y', 'x']:
                    axes_local = ['y', 'x', 'c']
                elif axes and axes[:3] == ['C', 'Y', 'X']:
                    axes_local = ['y', 'x', 'c']
                else:
                    axes_local = axes
            else:
                axes_local = axes
            pyramid.append(arr)
        return pyramid, axes_local

    # ---------------------------------------------------------------------
    # Open store
    # ---------------------------------------------------------------------
    root = zarr.open(Path(spatialdata_path), mode='r')

    # Create target InSituData shell
    xd = InSituData()
        # Path("./data1"),
        #             {"metadata_file": "meta.txt", "xenium": {"pixel_size": pixel_size}},
        #             "", "", "")

    # ---------------------------------------------------------------------
    # IMAGES
    # ---------------------------------------------------------------------
    if 'images' in root:
        imgs_group = root['images']
        images_obj = ImageData()
        for elem_name in imgs_group:
            if Path(elem_name).name.startswith('.'):
                continue
            sample_id, modality, locs = _parse_key(elem_name)
            if modality != 'IMAGES':
                continue
            image_name = locs[0] if locs else elem_name
            elem = imgs_group[elem_name]
            elem_store = os.path.join(root.store.path, elem.path)
            pyramid, axes = _read_multiscales(elem_store, 'image')
            if pyramid is None:
                continue
            # Use highest resolution level (0)
            images_obj.add_image(
                pyramid[0], name=image_name, axes=axes, pixel_size=pixel_size,
                ome_meta={'PhysicalSizeX': pixel_size}
            )
        setattr(xd, 'images', images_obj)

    # ---------------------------------------------------------------------
    # LABELS (cell boundaries) -> group per cell_key
    # ---------------------------------------------------------------------
    cell_boundaries: dict[str, BoundariesData] = {}
    tmp_boundaries: dict[str, dict] = defaultdict(dict)
    if 'labels' in root:
        lbl_group = root['labels']
        for elem_name in lbl_group:
            if Path(elem_name).name.startswith('.'):
                continue
            sample_id, modality, locs = _parse_key(elem_name)
            if modality != 'CELLS':
                continue
            # expected: CELLS.<cell_key>.boundaries.<boundary_name>
            if len(locs) >= 3 and locs[1] == 'boundaries':
                cell_key = locs[0]
                boundary_name = '.'.join(locs[2:])  # in case boundary had dots replaced
                elem = lbl_group[elem_name]
                elem_store = os.path.join(root.store.path, elem.path)
                pyramid, _axes = _read_multiscales(elem_store, 'labels')
                if pyramid is None:
                    continue
                tmp_boundaries[cell_key][boundary_name] = pyramid[0]
        # Convert to BoundariesData objects
        for ck, bdict in tmp_boundaries.items():
            bd = BoundariesData(None, None)
            if bdict:
                bd.add_boundaries(bdict, pixel_size=pixel_size)
            cell_boundaries[ck] = bd

    # ---------------------------------------------------------------------
    # TABLES (cell matrices)
    # ---------------------------------------------------------------------
    cells_map: dict[str, CellData] = {}
    if 'tables' in root:
        tbl_group = root['tables']
        for elem_name in tbl_group:
            sample_id, modality, locs = _parse_key(elem_name)
            if modality != 'CELLS' or not locs or locs[-1] != 'matrix':
                continue
            cell_key = locs[0]
            elem = tbl_group[elem_name]
            elem_store = os.path.join(root.store.path, elem.path)
            matrix_adata = read_zarr(elem_store)
            bd = cell_boundaries.get(cell_key, BoundariesData(None, None))
            cells_map[cell_key] = CellData(matrix_adata, bd)

    if cells_map:
        # Expose as dict-like mapping so existing export code can iterate .keys()
        xd.cells = cells_map

    # ---------------------------------------------------------------------
    # POINTS (transcripts)
    # ---------------------------------------------------------------------
    if 'points' in root:
        pts_group = root['points']
        all_parts = []
        for elem_name in pts_group:
            sample_id, modality, locs = _parse_key(elem_name)
            if modality != 'TRANSCRIPTS':
                continue
            elem = pts_group[elem_name]
            elem_store = os.path.join(root.store.path, elem.path)
            ddf = read_parquet(elem_store)
            df = ddf.compute()
            # Align to the schema used by _transform_transcripts
            # SpatialData stores coordinates as columns 'x','y','z' (when present)
            rename = {}
            if 'x' in df.columns: rename['x'] = 'x_location'
            if 'y' in df.columns: rename['y'] = 'y_location'
            if 'z' in df.columns: rename['z'] = 'z_location'
            df = df.rename(columns=rename)
            # Ensure required columns exist
            for col in ['x_location', 'y_location']:
                if col not in df.columns:
                    raise ValueError(f"Missing required coordinate column '{col}' in points table '{elem_name}'.")
            if 'feature_name' not in df.columns and 'gene' in df.columns:
                df = df.rename(columns={'gene': 'feature_name'})
            if 'cell_id' not in df.columns and 'xenium' in df.columns:
                df = df.rename(columns={'xenium': 'cell_id'})
            all_parts.append(df)
        if all_parts:
            transcripts_df = pd.concat(all_parts, ignore_index=True)
            xd.transcripts = transcripts_df

    return xd


# def load_from_spatialdata(spatialdata_path, pixel_size):

#     import os
#     from pathlib import Path

#     import numpy as np
#     import pandas as pd
#     import zarr
#     from anndata import read_zarr
#     from dask.array import transpose
#     from dask.dataframe import read_parquet
#     from ome_zarr.io import ZarrLocation
#     from ome_zarr.reader import Label, Multiscales, Reader

#     from insitupy.dataclasses import BoundariesData, CellData, ImageData

#     def read_helper_images_labels(f_elem_store, type):
#         nodes = []
#         image_loc = ZarrLocation(f_elem_store)
#         if image_loc.exists():
#             image_reader = Reader(image_loc)()
#             image_nodes = list(image_reader)
#             if len(image_nodes):
#                 for node in image_nodes:
#                     if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and (
#                         type == "image"
#                         and np.all([not isinstance(spec, Label) for spec in node.specs])
#                         or type == "labels"
#                         and np.any([isinstance(spec, Label) for spec in node.specs])
#                     ):
#                         nodes.append(node)
#         assert len(nodes) == 1
#         node = nodes[0]
#         datasets = node.load(Multiscales).datasets
#         multiscales = node.load(Multiscales).zarr.root_attrs["multiscales"]
#         axes = [i["name"] for i in node.metadata["axes"]]
#         assert len(multiscales) == 1
#         if len(datasets) >= 1:
#             multiscale_image = []
#             for _, d in enumerate(datasets):
#                 data = node.load(Multiscales).array(resolution=d, version=None)
#                 if data.shape[0] == 1:
#                     data = data.reshape(data.shape[1:])
#                     axes = axes[1:]
#                 elif data.shape[0] == 3:
#                     data = transpose(data, (1, 2, 0))
#                 multiscale_image.append(data)
#             return multiscale_image, axes

#     xd = InSituData(Path("./data1"), {"metadata_file": "meta.txt", "xenium":{"pixel_size": pixel_size}}, "", "", "")
#     path = Path(spatialdata_path)
#     f = zarr.open(path, mode="r")
#     bd = BoundariesData(None, None)


#     if "labels" in f:
#         group = f["labels"]
#         boundaries_dict = {}
#         for name in group:
#             if Path(name).name.startswith("."):
#                 continue
#             f_elem = group[name]
#             f_elem_store = os.path.join(f.store.path, f_elem.path)
#             image, axes = read_helper_images_labels(f_elem_store, "labels")
#             boundaries_dict[name] = image[0]
#         bd.add_boundaries(boundaries_dict, pixel_size=pixel_size)

#     if "tables" in f:
#         group = f["tables"]
#         i = 0
#         for name in group:
#             f_elem = group[name]
#             f_elem_store = os.path.join(f.store.path, f_elem.path)
#             cdata = CellData(read_zarr(f_elem_store), bd)
#             if len(group) == 1 or i == 0:
#                 setattr(xd, "cells", cdata)
#             else:
#                 xd.add_alt(cdata, key_to_add=name)
#             i += 1

#     if "points" in f:
#         group = f["points"]
#         for name in group:
#             if name == "transcripts":
#                 f_elem = group[name]
#                 f_elem_store = os.path.join(f.store.path, f_elem.path)
#                 points = read_parquet(f_elem_store)
#                 pdf = points.compute()

#                 # Rename columns to match the new structure
#                 pdf = pdf.rename(columns={
#                     'x': 'x',
#                     'y': 'y',
#                     'z': 'z',
#                     'feature_name': 'gene',
#                     'qv': 'qv',
#                     'overlaps_nucleus': 'overlaps_nucleus',
#                     'cell_id': 'xenium',
#                     'transcript_id': 'transcript_id'
#                 })

#                 # Reorder columns to match the new structure
#                 pdf = pdf[['x', 'y', 'z', 'gene', 'qv', 'overlaps_nucleus', 'xenium', 'transcript_id']]

#                 # Set 'transcript_id' as the index
#                 pdf = pdf.set_index('transcript_id')

#                 # Set the MultiIndex for columns
#                 pdf.columns = pd.MultiIndex.from_tuples([
#                     ('coordinates', 'x'),
#                     ('coordinates', 'y'),
#                     ('coordinates', 'z'),
#                     ('properties', 'gene'),
#                     ('properties', 'qv'),
#                     ('properties', 'overlaps_nucleus'),
#                     ('cell_id', 'xenium')
#                 ])
#                 setattr(xd, "transcripts", pdf)
#     if "images" in f:
#         group = f["images"]
#         setattr(xd, "images", ImageData())
#         for name in group:
#             if Path(name).name.startswith("."):
#                 continue
#             f_elem = group[name]
#             f_elem_store = os.path.join(f.store.path, f_elem.path)
#             image, axes = read_helper_images_labels(f_elem_store, "image")
#             xd.images.add_image(image[0], name=name, axes=axes, pixel_size=pixel_size, ome_meta={'PhysicalSizeX': pixel_size})
#     return xd
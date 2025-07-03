# from typing import List

# import numpy as np
# from anndata import AnnData

# from insitupy._core.insitudata import InSituData


# def convert_to_spatialdata_dict(data, levels: int = 5):

#     """
#     Converts an InSituData object to a dictionary for SpatialData object.

#     This function integrates various data elements such as images, labels, transcripts, and annotations
#     into a SpatialData object. It requires the spatialdata framework to be installed.

#     Raises:
#         ImportError: If the spatialdata framework is not installed.

#     Returns:
#         Dict: a dictionary with all modalities saved in SpatialData format.
#     """

#     try:
#         import dask.dataframe as dd
#         from spatialdata.models import (Image2DModel, Labels2DModel,
#                                         PointsModel, ShapesModel, TableModel)
#         from spatialdata.transformations.transformations import Identity, Scale
#         from xarray import DataArray

#         from insitupy._deprecated._spatialdata import SpatialData
#     except ImportError:
#         raise ImportError("This function requires the spatialdata framework, please install it with `pip install spatialdata`.")

#     def transform_anndata(adata: AnnData, cells_as_circles: bool = True):

#         REGION = "region"
#         attrs = {}
#         attrs["instance_key"] = "cell_id" if cells_as_circles else "cell_labels"
#         adata.obs["cell_id"] = adata.obs.index
#         adata.obs.index = range(len(adata.obs))
#         adata.obs.index = adata.obs.index.astype(str)
#         attrs[REGION] = "cell_circles" if cells_as_circles else "cell_labels"
#         adata.obs[REGION] = attrs[REGION]
#         adata.obs[REGION] = adata.obs[REGION].astype("category")
#         attrs["region_key"] = REGION
#         adata.uns["spatialdata_attrs"] = attrs
#         if cells_as_circles:
#             transform = Scale([1.0 / data.metadata["method_params"]["pixel_size"], 1.0 / data.metadata["method_params"]["pixel_size"]], axes=("x", "y"))
#             radius = np.sqrt(data.cells.matrix.obs["cell_area"].to_numpy() / np.pi)
#             circles = ShapesModel.parse(
#                     data.cells.matrix.obsm["spatial"].copy(),
#                     geometry=0,
#                     radius=radius,
#                     transformations={"global": transform},
#                     index=data.cells.matrix.obs.index.copy(),
#             )
#         return adata, circles



#     def transform_images(xd: InSituData, levels: int = 5):
#         images = {}
#         image_types = {}
#         if xd.images is not None:
#             for name in xd.images.names:
#                 images_list =  xd.images[name]

#                 if len(xd.images.metadata[name]["shape"]) == 3:
#                     axes = ("y", "x", "c")
#                     array = DataArray(data=images_list[0], name="image", dims=axes)
#                     images[name] = Image2DModel.parse(array, dims=axes, c_coords=["r", "g", "b"], chunks=(1, 4096, 4096), scale_factors=[2 for _ in range(levels)])
#                     image_types[name] = "image"
#                 else:
#                     axes = ("c", "y", "x")
#                     array = DataArray(data=images_list[0].reshape(1, images_list[0].shape[0], images_list[0].shape[1]), name="image", dims=axes)
#                     images[name] = Image2DModel.parse(array, dims=("c", "y", "x"), chunks=(1, 4096, 4096), scale_factors=[2 for _ in range(levels)])
#                     image_types[name] = "image"
#         return images, image_types


#     def transform_labels(xd: InSituData):
#         labels = {}
#         label_types = {}
#         if xd.cells is not None and xd.cells.boundaries is not None:

#             for name in xd.cells.boundaries.metadata.keys():

#                 labels_list =  xd.cells.boundaries[name]
#                 if isinstance(labels_list, list):
#                     labels_list = labels_list[0]
#                 array = DataArray(data=labels_list, name="label", dims=("y", "x"))
#                 labels[name] = Labels2DModel.parse(array, dims=("y", "x"), chunks=(4096, 4096), scale_factors=[2 for _ in range(levels)])
#                 label_types[name] = "labels"
#         return labels, label_types


#     def transform_transcripts(xd: InSituData):
#         points = {}
#         point_types = {}
#         if xd.transcripts is not None:
#             df = dd.from_pandas(xd.transcripts, npartitions=1)
#             df.columns = df.columns.droplevel(0)
#             df['transcript_id'] = df.index
#             df = df.reset_index(drop=True)
#             rename_dict = {
#                             "xenium": "cell_id",
#                             "gene": "feature_name"
#                         }
#             df = df.rename(columns=rename_dict)
#             scale = Scale([1.0 / xd.metadata["method_params"]["pixel_size"], 1.0 / xd.metadata["method_params"]["pixel_size"], 1.0], axes=("x", "y", "z"))
#             parsed_points = PointsModel.parse(df,
#                                             coordinates={"x": "x", "y": "y", "z": "z"},
#                                             feature_key="feature_name",
#                                             instance_key="cell_id",
#                                             transformations={"global": scale},
#                                             sort=True)
#             points = {"transcripts": parsed_points}
#             point_types = {"transcripts": "points"}
#         return points, point_types

#     def transform_matrix_annotations(xd: InSituData):
#         tables, shapes, table_types, shape_type = {}, {}, {}, {}
#         if xd.cells is not None and xd.cells.matrix is not None:
#             adata, circles = transform_anndata(xd.cells.matrix.copy())
#             tables["table"] = TableModel.parse(adata)
#             shapes["cell_circles"] = circles
#             table_types["table"] = "tables"
#             shape_type["cell_circles"] = "shapes"

#         if xd.alt is not None:
#             for key in xd.alt.keys():
#                 if hasattr(xd.alt[key], "matrix"):
#                     adata, circles = transform_anndata(xd.alt[key].matrix.copy())
#                     tables[key] = TableModel.parse(adata)

#         if xd.annotations is not None:
#             for key in xd.annotations.metadata.keys():
#                 gdf = ShapesModel.parse(xd.annotations.get(key), transformations={"global": Identity()})
#                 shapes[key] = gdf
#                 shape_type[key] = "shapes"
#         return tables | shapes, table_types | shape_type


#     transcripts, points_names = transform_transcripts(data)
#     matrix_shapes, matrix_shapes_names = transform_matrix_annotations(data)
#     images, images_names = transform_images(data, levels)
#     labels, labels_names = transform_labels(data)
#     merged_dict = transcripts | matrix_shapes | images | labels
#     merged_dict_names = points_names | matrix_shapes_names | images_names | labels_names
#     return merged_dict, merged_dict_names

# def convert_to_spatialdata(data, levels: int = 5):

#     """
#     Converts an InSituData object to a SpatialData object.

#     This function integrates various data elements such as images, labels, transcripts, and annotations
#     into a SpatialData object. It requires the spatialdata framework to be installed.

#     Raises:
#         ImportError: If the spatialdata framework is not installed.

#     Returns:
#         SpatialData: A SpatialData object containing the integrated data elements.

#     """

#     try:
#         from insitupy._deprecated._spatialdata import SpatialData
#     except:
#         raise ImportError("This function requires spatialdata framework, please install with pip install spatialdata.")


#     dict, _ = convert_to_spatialdata_dict(data, levels=levels)
#     sdata = SpatialData.from_elements_dict(dict)
#     return sdata

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

#     from insitupy._core.dataclasses import BoundariesData, CellData, ImageData


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
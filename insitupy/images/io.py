import os
import warnings
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import List, Literal, Optional, Union

import dask.array as da
import numpy as np
import xmltodict
import zarr
from parse import *
from tifffile import TiffFile, TiffWriter, imread

from insitupy import __version__

from .._exceptions import InvalidFileTypeError
from ..images.utils import create_img_pyramid
from ..utils.utils import convert_to_list


def read_zarr(path):
    # load image from .zarr.zip
    #zipped = True if suffix == "zarr.zip" else False
    zipped = zipfile.is_zipfile(path)
    with zarr.ZipStore(path, mode="r") if zipped else zarr.DirectoryStore(path) as dirstore:

        # get components of zip store
        components = dirstore.listdir()

        if ".zarray" in components:
            # the store is an array which can be opened
            if zipped:
                img = da.from_zarr(dirstore).persist()
            else:
                img = da.from_zarr(dirstore)
        else:
            subres = [elem for elem in components if not elem.startswith(".")]
            img = []
            for s in subres:
                if zipped:
                    img.append(
                        da.from_zarr(dirstore, component=s).persist()
                                )
                else:
                    img.append(
                        da.from_zarr(dirstore, component=s)
                                )

        # retrieve OME metadata
        store = zarr.open(dirstore)
        meta = store.attrs.asdict()
        ome_meta = meta["OME"]
        axes = meta["axes"]

    return img, ome_meta, axes


def read_image(
    path
    ):
    path = Path(path)
    suffix = path.name.split(".", maxsplit=1)[-1]

    if "zarr" in suffix:
        img, ome_meta, axes = read_zarr(path)

    elif suffix in ["ome.tif", "ome.tiff"]:
        # load image from .ome.tiff
        img = read_ome_tiff(path=path, levels=None)
        # read ome metadata
        with TiffFile(path) as tif:
            axes = tif.series[0].axes # get axes
            ome_meta = tif.ome_metadata # read OME metadata
            ome_meta = xmltodict.parse(ome_meta, attr_prefix="")["OME"] # convert XML to dict

    else:
        raise InvalidFileTypeError(
            allowed_types=["zarr", "zarr.zip", "ome.tif", "ome.tiff"],
            received_type=suffix
            )

    return img, ome_meta, axes

def write_zarr(image, file,
               img_metadata: dict,
               save_pyramid: bool = True,
               axes: str = "YXS", # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
               overwrite: bool = False,
               verbose: bool = False
               ):
    if verbose:
        print(f"Saving image to {str(file)}")

    # get suffix
    file = Path(file)

    if file.exists():
        if overwrite:
            file.unlink() # delete file
        else:
            raise FileExistsError("Output file exists already ({}).\nFor overwriting it, select `overwrite=True`".format(file))

    suffix = file.name.split(".", 1)[-1]

    # check if the suffix contains zip
    zipped = "zip" in suffix

    # decide whether to save as pyramid or not
    if isinstance(image, list):
        if not save_pyramid:
            image_data = image[0]
        else:
            image_data = image
    else:
        if save_pyramid:
            # create img pyramid
            image_data = create_img_pyramid(img=image, nsubres=6, axes=axes)
        else:
            image_data = image

    with zarr.ZipStore(file, mode="w") if zipped else zarr.DirectoryStore(file) as dirstore:
        # check whether to save the image as pyramid or not
        if save_pyramid:
            for i, im in enumerate(image_data):
                im.to_zarr(dirstore, component=str(i))
        else:
            # save image data in zipstore without pyramid
            image_data.to_zarr(dirstore)

        # open zarr store save metadata in zarr store
        store = zarr.open(dirstore, mode="a")
        store.attrs.put(img_metadata)
        # for k,v in img_metadata.items():
        #     store.attrs[k] = v

def write_ome_tiff(
    image: Union[np.ndarray, da.core.Array, List[da.core.Array]],
    file: Union[str, os.PathLike, Path],
    axes: str = "YXS", # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
    metadata: dict = {},
    subresolutions = 6,
    subres_steps: int = 2,
    pixelsize: Optional[float] = 1, # defaults to Xenium settings.
    pixelunit: Optional[str] = None, # usually µm
    #significant_bits: Optional[int] = 16,
    photometric: Literal['rgb', 'minisblack', 'maxisblack'] = 'rgb', # before I had rgb here. Xenium doc says minisblack
    tile: tuple = (1024, 1024), # 1024 pixel is optimal for Xenium Explorer
    compression: Literal['jpeg', 'LZW', 'jpeg2000', "ZLIB", None] = 'ZLIB', # jpeg2000 or ZLIB are recommended in the Xenium documentation - ZLIB is faster
    overwrite: bool = False,
    verbose: bool = False
    ):

    '''
    Function to write (pyramidal) OME-TIFF files.
    Code adapted from: https://github.com/cgohlke/tifffile and Xenium docs (see below).

    For parameters optimal for Xenium see: https://www.10xgenomics.com/support/software/xenium-explorer/tutorials/xe-image-file-conversion
    '''
    if verbose:
        print(f"Saving image to {str(file)}")
    # check if the image is an image pyramid
    if isinstance(image, list):
        # if it is a pyramid, select only the highest resolution image
        first_image = image[0]
        image_pyramid = image
    elif isinstance(image, np.ndarray) or isinstance(image, da.core.Array):
        first_image = image
        image_pyramid = create_img_pyramid(img=image, nsubres=subresolutions, axes=axes)

    # determine significant bits variable - is important that Xenium explorer correctly distinguishes between 8 bit and 16 bit
    if first_image.dtype == np.dtype('uint8'):
        significant_bits = 8
    else:
        significant_bits = 16

    file = Path(file)
    if file.exists():
        if overwrite:
            file.unlink() # delete file
        else:
            raise FileExistsError("Output file exists already ({}).\nFor overwriting it, select `overwrite=True`".format(file))

    # create metadata
    if pixelsize != 1:
        metadata = {
            **metadata,
            **{
                'PhysicalSizeX': pixelsize,
                'PhysicalSizeY': pixelsize
            }
        }
    if pixelunit is not None:
        metadata = {
            **metadata,
            **{
                'PhysicalSizeXUnit': pixelunit,
                'PhysicalSizeYUnit': pixelunit
            }
        }
    if (significant_bits is not None) & ("SignificantBits" not in metadata.keys()):
        metadata = {
            **metadata,
            **{
                'SignificantBits': significant_bits
            }
        }


    with TiffWriter(file, bigtiff=True) as tif:
        options = dict(
            photometric=photometric,
            tile=tile,
            compression=compression,
            resolutionunit='CENTIMETER',
        )
        tif.write(
            image_pyramid[0],
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options
        )

        scale = 1
        for i in range(1, subresolutions+1):
            img = image_pyramid[i]
            #scale /= subres_steps
            #img = resize_image(img, scale_factor=1/subres_steps, axes=axes)
            tif.write(
                img,
                subfiletype=1,
                resolution=(1e4 / scale / pixelsize,1e4 / scale / pixelsize),
                **options
            )

def read_zarr_pyramid(dirstore, persist):
    # get components of zip store
    components = dirstore.listdir()

    if ".zarray" in components:
        # the store is an array which can be opened
        if persist:
            img = da.from_zarr(dirstore).persist()
        else:
            img = da.from_zarr(dirstore)
    else:
        subres = sorted([elem for elem in components if not elem.startswith(".")])
        img = []
        for s in subres:
            if persist:
                img.append(
                    da.from_zarr(dirstore, component=s).persist()
                            )
            else:
                img.append(
                    da.from_zarr(dirstore, component=s)
                            )

    return img

def read_ome_tiff(
    path,
    levels: Optional[Union[List[int], int]] = None,
    new_method: bool = True
    ):
    '''
    Function to load pyramid from `ome.tiff` file.
    From: https://www.youtube.com/watch?v=8TlAAZcJnvA
    Another good resource from 10x: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs

    Args:
        path (str): The file path to the `ome.tiff` file.
        levels (Optional[Union[List[int], int]]): A list of integers representing the levels of the pyramid to load. If None, all levels are loaded. Default is None.
        new_method (bool): Is now the default method and uses a strategy found here: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs.

    Returns:
        List[dask.array.Array] or dask.array.Array: The pyramid or a single level of the pyramid, represented as Dask arrays.

    '''
    if new_method:
        pyramid = []
        l = 0
        while True:
            try:
                store = imread(path, aszarr=True, level=l, is_ome=False)
                pyramid.append(da.from_zarr(store))
                l+=1 # count up
            except IndexError:
                break

    else:
        # read store
        store = imread(path, aszarr=True)

        # Open store (root group)
        grp = zarr.open(store, mode='r')

        # Read multiscale metadata
        datasets = grp.attrs["multiscales"][0]["datasets"]

        if levels is None:
            levels = range(0, len(datasets))
        # make sure level is a list
        levels = convert_to_list(levels)

        # extract images as pyramid list
        pyramid = [
            da.from_zarr(store, component=datasets[l]["path"])
            for l in levels
        ]

    # if pyramid has only one element, return only this image
    if len(pyramid) == 1:
        pyramid = pyramid[0]

    return pyramid
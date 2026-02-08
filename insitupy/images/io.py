import logging
import os
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import List, Literal, Optional, Union
from warnings import warn

import dask.array as da
import numpy as np
import xmltodict
import zarr
from parse import *
from tifffile import TiffFile, TiffWriter, imread

from insitupy import __version__
from insitupy._exceptions import InvalidFileTypeError
from insitupy.images.axes import ImageAxes
from insitupy.images.utils import _get_chunksize, create_img_pyramid
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)

# Detect Zarr version for compatibility
ZARR_V3 = hasattr(zarr.storage, 'LocalStore')

if ZARR_V3:
    logger.info("Using Zarr v3.")
else:
    logger.info("Using Zarr v2.")


def get_zarr_source_path(arr) -> Optional[Path]:
    """
    Extract the source Zarr store path from a dask array loaded via ``da.from_zarr()``.

    For image pyramids (list of dask arrays), inspects each level and returns
    the first path found.

    Args:
        arr: A dask array, list of dask arrays, or any other object.

    Returns:
        The resolved :class:`~pathlib.Path` to the source Zarr store, or
        ``None`` if the source cannot be determined (e.g. because the array
        was created from in-memory data or was ``.persist()``-ed).
    """
    # Handle list of dask arrays (pyramids)
    if isinstance(arr, list):
        for a in arr:
            result = get_zarr_source_path(a)
            if result is not None:
                return result
        return None

    if not isinstance(arr, da.Array):
        return None

    graph = arr.__dask_graph__()
    if not hasattr(graph, 'layers'):
        return None

    # Dask stores a zarr.Array object in a MaterializedLayer named
    # 'original-from-zarr-*'.  We look for zarr.Array values and
    # extract the store path from them.
    for layer_name, layer in graph.layers.items():
        mapping = getattr(layer, 'mapping', None)
        if mapping is None:
            if hasattr(layer, 'items'):
                mapping = layer
            else:
                continue

        try:
            items = list(mapping.items())
        except Exception:
            continue

        for _key, val in items:
            path = _extract_path_from_zarr_array(val)
            if path is not None:
                return path
            # Also check inside tuples (some dask versions)
            if isinstance(val, tuple):
                for v in val:
                    path = _extract_path_from_zarr_array(v)
                    if path is not None:
                        return path

    return None


def _extract_path_from_zarr_array(v) -> Optional[Path]:
    """Return the filesystem path of a ``zarr.Array``'s store, or ``None``."""
    if isinstance(v, zarr.Array):
        store = v.store
        return _extract_path_from_zarr_store(store)
    return None


def _extract_path_from_zarr_store(store) -> Optional[Path]:
    """Return the filesystem path backing a Zarr store, or ``None``."""
    if ZARR_V3:
        if isinstance(store, zarr.storage.LocalStore):
            root = getattr(store, 'root', None)
            if root is not None:
                return Path(str(root)).resolve()
            return Path(str(store)).resolve()
    else:
        if isinstance(store, zarr.storage.DirectoryStore):
            return Path(store.path).resolve()

    # Generic fallback
    if hasattr(store, 'path'):
        try:
            return Path(store.path).resolve()
        except Exception:
            pass
    if hasattr(store, 'root'):
        try:
            return Path(str(store.root)).resolve()
        except Exception:
            pass
    return None


def _get_zarr_store(path, mode: str = "r", zipped: bool = False):
    """
    Get a Zarr store compatible with both Zarr v2 and v3.

    Args:
        path: Path to the zarr store
        mode: Mode to open the store ('r', 'w', 'a')
        zipped: Whether the store is a ZipStore

    Returns:
        For Zarr v3: store object (no context manager needed)
        For Zarr v2: store object (should be used as context manager)
    """
    if ZARR_V3:
        # Zarr v3 API
        if zipped:
            return zarr.storage.ZipStore(path, mode=mode)
        else:
            return zarr.storage.LocalStore(path)
    else:
        # Zarr v2 API
        if zipped:
            return zarr.ZipStore(path, mode=mode)
        else:
            return zarr.DirectoryStore(path)


def read_zarr(path):
    # load image from .zarr.zip
    zipped = zipfile.is_zipfile(path)

    # Use ExitStack to handle context manager differences between Zarr v2 and v3
    with ExitStack() as stack:
        dirstore = _get_zarr_store(path, mode="r", zipped=zipped)

        # In Zarr v2, stores are context managers and need to be entered
        if not ZARR_V3:
            dirstore = stack.enter_context(dirstore)

        # open zarr group
        root = zarr.open_group(store=dirstore, mode='r')
        components = sorted(root.keys())

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
        pixel_size = meta["pixel_size"]

        if len(img) == 0:
            raise ValueError(f"No image data read from zarr file: {path}")

    return img, ome_meta, axes, pixel_size


def read_image(
    path
    ):
    path = Path(path)
    suffix = path.name.split(".", maxsplit=1)[-1]

    if "zarr" in suffix:
        img, ome_meta, axes, pixel_size = read_zarr(path)

    else:
        # non-ZARR files
        if suffix in ["ome.tif", "ome.tiff"]:
            # load image from .ome.tiff
            img = read_ome_tiff(path=path, levels=None)
        elif suffix in ["tif", "tiff"]:
            img = imread(path)
        else:
            raise InvalidFileTypeError(
                allowed_types=["zarr", "zarr.zip", "ome.tif", "ome.tiff"],
                received_type=suffix
                )

        # read ome metadata
        with TiffFile(path) as tif:
            # check whether the data is a multi-file OME-TIFF
            is_multifile = len(tif.series[0].levels[0].pages) != len(tif.pages)
            if is_multifile:
                axes = tif.pages[0].axes
                logger.warning(
                    f"'{Path(path).name}' is part of a multi-file OME-TIFF. "
                    "Axes are inferred from this file only and only data from this file will be returned.",
                )
            else:
                axes = tif.series[0].axes

            ome_meta = tif.ome_metadata # read OME metadata
            ome_meta = xmltodict.parse(ome_meta, attr_prefix="")["OME"] # convert XML to dict

            try:
                pixel_size = float(ome_meta['Image']['Pixels']['PhysicalSizeX'])
            except KeyError:
                try:
                    pixel_size = float(ome_meta['PhysicalSizeX'])
                except KeyError:
                    # in case of .tif image
                    pixel_size = float(ome_meta['OME:Image']['OME:Pixels']['PhysicalSizeX'])

        if axes == "CYX":
            if isinstance(img, list):
                shape = img[0].shape
            else:
                shape = img.shape
            if not len(shape) == 3:
                warn(f"Axes information ({axes}) and shape ({shape}) do not fit together. Assumed grayscale image with axes 'YX'.")
                axes = "YX"

    return img, ome_meta, axes, pixel_size

def write_zarr(image, file,
               img_metadata: dict,
               axes: str, # channels, e.g. "YXS" for RGB - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
               save_pyramid: bool = True,
               overwrite: bool = False,
               verbose: bool = False
               ):
    if verbose:
        print(f"Saving image to {str(file)}")

    # get suffix
    file = Path(file)

    if file.exists():
        if overwrite:
            if file.is_dir():
                import shutil
                shutil.rmtree(file)  # delete directory for .zarr folders
            else:
                file.unlink()  # delete file for .zarr.zip
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
            image_data = create_img_pyramid(img=image, axes=axes, nsubres=6, scale_steps=2)
        else:
            image_data = image

    # Use ExitStack to handle context manager differences between Zarr v2 and v3
    with ExitStack() as stack:
        dirstore = _get_zarr_store(file, mode="w", zipped=zipped)

        # In Zarr v2, stores are context managers and need to be entered
        if not ZARR_V3:
            dirstore = stack.enter_context(dirstore)

        # Parse axes configuration to determine proper chunk sizes
        axes_config = ImageAxes(axes)

        # check whether to save the image as pyramid or not
        if save_pyramid:
            for i, im in enumerate(image_data):
                chunksize = _get_chunksize(axes_config, im.ndim)
                im = im.rechunk(chunksize)
                im.to_zarr(dirstore, component=str(i))
        else:
            # save image data in zipstore without pyramid
            chunksize = _get_chunksize(axes_config, image_data.ndim)
            image_data = image_data.rechunk(chunksize)
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
    photometric: Literal['rgb', 'minisblack', 'maxisblack'] = 'rgb', # before I had rgb here. Xenium doc says minisblack
    tile: tuple = (1024, 1024), # 1024 pixel is optimal for Xenium Explorer
    compression: Literal['jpeg', 'LZW', 'jpeg2000', "ZLIB", None] = 'ZLIB', # jpeg2000 or ZLIB are recommended in the Xenium documentation - ZLIB is faster
    overwrite: bool = False,
    verbose: bool = False
    ):
    """Write image data to a pyramidal OME-TIFF file.

    Creates a multi-resolution pyramidal OME-TIFF file from an input image or
    image pyramid. Parameters are optimized for compatibility with Xenium Explorer.

    Code adapted from: https://github.com/cgohlke/tifffile and Xenium docs.
    For parameters optimal for Xenium see:
    https://www.10xgenomics.com/support/software/xenium-explorer/tutorials/xe-image-file-conversion

    Args:
        image: Input image as numpy array, dask array, or list of arrays
            representing an existing pyramid.
        file: Output file path for the OME-TIFF file.
        axes: String describing the axis configuration of the image.
            Examples: 'YX' for grayscale, 'YXS' for RGB, 'CYX' for
            multi-channel IF, 'TCYXS' for time-series RGB. Defaults to 'YXS'.
        metadata: Additional OME metadata to include in the file. Defaults to {}.
        subresolutions: Number of pyramid subresolution levels to create.
            Defaults to 6.
        subres_steps: Downsampling factor between consecutive pyramid levels.
            Defaults to 2.
        pixelsize: Physical pixel size in the specified unit. Defaults to 1.
        pixelunit: Unit for pixel size (e.g., 'µm'). Defaults to None.
        photometric: Photometric interpretation of the image data.
            Options: 'rgb', 'minisblack', 'maxisblack'. Xenium documentation
            recommends 'minisblack'. Defaults to 'rgb'.
        tile: Tile size for tiled TIFF writing. 1024x1024 is optimal for
            Xenium Explorer. Defaults to (1024, 1024).
        compression: Compression algorithm to use. Options: 'jpeg', 'LZW',
            'jpeg2000', 'ZLIB', None. JPEG2000 or ZLIB are recommended in
            Xenium documentation; ZLIB is faster. Defaults to 'ZLIB'.
        overwrite: If True, overwrite existing file. Defaults to False.
        verbose: If True, print progress information. Defaults to False.

    Raises:
        FileExistsError: If the output file already exists and overwrite is False.

    Example:
        >>> write_ome_tiff(
        ...     image=my_image,
        ...     file="output.ome.tiff",
        ...     axes="YXS",
        ...     pixelsize=0.2125,
        ...     pixelunit="µm"
        ... )
    """
    if verbose:
        print(f"Saving image to {str(file)}")
    # check if the image is an image pyramid
    if isinstance(image, list):
        # if it is a pyramid, select only the highest resolution image
        first_image = image[0]
        image_pyramid = image
    elif isinstance(image, np.ndarray) or isinstance(image, da.core.Array):
        first_image = image
        image_pyramid = create_img_pyramid(
            img=image, nsubres=subresolutions, axes=axes, scale_steps=subres_steps
            )

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
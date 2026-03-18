from __future__ import annotations

import logging
import os
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd

from insitupy._mixins import DeepCopyMixin
from insitupy._textformat import textformat as tf
from insitupy.images.axes import (ImageAxes, _transpose_to_standard_axes,
                                  get_height_and_width)
from insitupy.images.io import (get_zarr_source_path, is_from_zarr_disk,
                                read_image, write_ome_tiff, write_zarr)
from insitupy.images.utils import (create_img_pyramid,
                                   crop_dask_array_or_pyramid, resize_image)
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)


class ImageData(DeepCopyMixin):
    '''
    Object to read and load images.
    '''
    def __init__(self,
                 img_files: List[str] = None,
                 img_names: List[str] = None,
                 pixel_size: float = None,
                 ):

        # iterate through files and load them
        self._names = []
        self._metadata = {}
        self._data = {}

        # TODO: Add ImageData.read() — loading is complex (zarr pyramids, OME-TIFF)
        # and currently handled through InSituData.load_images().

        if img_files is not None:
            # convert arguments to lists
            img_files = convert_to_list(img_files)
            img_names = convert_to_list(img_names)

            for n, f in zip(img_names, img_files):
                self.add_image(
                    image=f,
                    channel_names=n,
                    axes=None,
                    pixel_size=pixel_size,
                    ome_meta=None,
                    )

    def __repr__(self):
        if len(self._data) > 0:
            # Calculate the maximum length of the key names for alignment
            max_key_len = max(len(n) for n in self._metadata.keys())
            pad = 3
            repr_strings = [f"{tf.Bold}'{n}':{tf.ResetAll}{' ' * (max_key_len - len(n) + pad)}{metadata['shape']}" for n,metadata in self._metadata.items()]
            s = "\n".join(repr_strings)
        else:
            s = "empty"
        return s

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data.get(key)

    def __delitem__(self, key: str):
        """Remove an image by name using del syntax.

        Args:
            key: Name of the image to remove.

        Raises:
            KeyError: If the image name is not found.

        Example:
            >>> del img_data['DAPI']
        """
        if key not in self._names:
            raise KeyError(f"Image '{key}' not found in ImageData. Available images: {self._names}")

        del self._data[key]
        self._names.remove(key)
        self._metadata.pop(key, None)

    def __contains__(self, key):
        return key in self.keys()

    def keys(self):
        """Return the keys of all stored images."""
        return self._data.keys()

    @property
    def metadata(self):
        """Dict of image metadata (pixel size, shape, axes, etc.) keyed by image name."""
        return self._metadata

    @property
    def names(self):
        """List of image names in insertion order."""
        return self._names

    @property
    def is_empty(self):
        """True if no images have been added yet."""
        return len(self._data) == 0

    def add_image(
        self,
        image: Union[da.core.Array, np.ndarray, str, os.PathLike, Path],
        channel_names: Optional[Union[str, List[str]]] = None,
        axes: Optional[str] = None,
        pixel_size: Optional[Number] = None,
        ome_meta: Optional[dict] = {},
        is_rgb: Optional[bool] = None,
        transformation_matrix: Optional[Union[np.ndarray, str, os.PathLike, Path]] = None,
        reference_image: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True
        ):
        """Add image data to the ImageData object.

        For multi-channel images ("CYX" axes), each channel is added as a separate entry.
        RGB images ("YXS" axes with 3 channels) are kept together as a single entry.

        Args:
            image: Either a dask/numpy array or a path to an image file.
            channel_names: Name identifier(s) for the image. For multi-channel images (CYX), provide
                a list of names (one per channel). For single-channel or RGB images, provide
                a string. If None, channel names are automatically extracted from OME metadata.
            axes: Axis specification (e.g., 'YX', 'CYX', 'YXS'). Required if image is an array.
            pixel_size: Physical pixel size in um/pixel. Required if image is an array.
            ome_meta: OME metadata dictionary.
            is_rgb: Whether the image is RGB. Auto-detected if None.
            transformation_matrix: Optional affine transformation matrix to apply to the image.
            reference_image: Name of the reference image in this ImageData object.
            overwrite: If True, overwrite existing image(s) with the same name(s).
            verbose: If True, print status messages.
        """
        # Load image data
        if isinstance(image, da.core.Array) or isinstance(image, np.ndarray):
            if axes is None:
                raise ValueError("If `image` is numpy or dask array, `axes` needs to be set.")
            if pixel_size is None:
                raise ValueError("If `image` is numpy or dask array, `pixel_size` needs to be set.")

            try:
                # convert to dask array before addition
                img = da.from_array(image)
            except ValueError:
                # in this case the array was already a dask array
                img = image
            filename = None

        elif isinstance(image, (str, os.PathLike, Path)):
            # read path
            image = Path(image)
            if not image.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            image = image.resolve() # resolve relative path
            filename = image.name
            img, ome_meta, axes, pixel_size = read_image(image) # returns image pyramid as list of dask arrays if possible
        else:
            raise TypeError(f"`image` must be a numpy/dask array or a file path. Got: {type(image)}")

        # Transpose image to standard axis order (YX, CYX, or YXS)
        img, axes = _transpose_to_standard_axes(img, axes)

        # Get image shape
        img_shape = img[0].shape if isinstance(img, list) else img.shape

        # Determine if this is a multi-channel image that should be split
        axes_config = ImageAxes(axes)
        is_multichannel = axes == "CYX" or (axes_config.C is not None and axes != "YXS")

        # Handle channel names for multi-channel images
        if is_multichannel:
            n_channels = img_shape[axes_config.C]

            # Extract or validate channel names
            if channel_names is None:
                # Try to extract from OME metadata
                if ome_meta and 'Image' in ome_meta:
                    try:
                        channels_info = ome_meta['Image']['Pixels']['Channel']
                        # Handle both single channel (dict) and multiple channels (list)
                        if isinstance(channels_info, dict):
                            channel_names = [channels_info.get('Name', f'Channel_0')]
                        else:
                            channel_names = [ch.get('Name', f'Channel_{i}') for i, ch in enumerate(channels_info)]

                        logger.info(f"Extracted channel names from OME metadata: {channel_names}")
                    except (KeyError, TypeError):
                        # Fallback to numbered channels
                        channel_names = [f'Channel_{i}' for i in range(n_channels)]
                        logger.info(f"Could not extract channel names from OME metadata. Using: {channel_names}")
                else:
                    # Fallback to numbered channels
                    channel_names = [f'Channel_{i}' for i in range(n_channels)]
                    logger.info(f"No OME metadata available. Using channel names: {channel_names}")

            elif isinstance(channel_names, str):
                raise ValueError(
                    f"Multi-channel image detected (axes='{axes}', {n_channels} channels) but `name` is a string. "
                    f"Please provide a list of channel names with length {n_channels}, or set name=None to "
                    f"automatically extract channel names from OME metadata."
                )

            elif isinstance(channel_names, list):
                if len(channel_names) != n_channels:
                    raise ValueError(
                        f"Length of `name` list ({len(channel_names)}) does not match number of channels ({n_channels}). "
                        f"Axes: '{axes}', Image shape: {img_shape}"
                    )
                channel_names = channel_names

            else:
                raise TypeError(f"`name` must be a string, list of strings, or None. Got: {type(channel_names)}")

            # Split channels and add each separately
            logger.info(f"Splitting multi-channel image into {n_channels} separate channels...")

            for i, ch_name in enumerate(channel_names):
                # Extract single channel
                if isinstance(img, list):
                    # Handle image pyramid
                    channel_img = [np.take(level, i, axis=axes_config.C) for level in img]
                else:
                    channel_img = np.take(img, i, axis=axes_config.C)

                # Determine new axes (remove channel dimension)
                channel_axes = "YX"

                # Add this channel as a separate image (recursive call for single channel)
                self._add_single_image(
                    img=channel_img,
                    name=ch_name,
                    axes=channel_axes,
                    pixel_size=pixel_size,
                    filename=filename,
                    ome_meta=ome_meta,
                    is_rgb=False,
                    transformation_matrix=transformation_matrix,
                    reference_image=reference_image,
                    overwrite=overwrite,
                    verbose=verbose
                )

        else:
            # Single-channel or RGB image - add as-is
            if isinstance(channel_names, list):
                raise ValueError(
                    f"Single-channel or RGB image (axes='{axes}') but `name` is a list. "
                    f"Please provide a single string name for this image."
                )

            if channel_names is None:
                # For single-channel images, try to extract name from OME or use default
                if ome_meta and 'Image' in ome_meta:
                    try:
                        channel_names = ome_meta['Image'].get('Name', 'Image_0')
                        logger.info(f"Extracted image name from OME metadata: {channel_names}")
                    except (KeyError, TypeError):
                        channel_names = 'Image_0'
                else:
                    channel_names = 'Image_0'

            # Add single image
            self._add_single_image(
                img=img,
                name=channel_names,
                axes=axes,
                pixel_size=pixel_size,
                filename=filename,
                ome_meta=ome_meta,
                is_rgb=is_rgb,
                transformation_matrix=transformation_matrix,
                reference_image=reference_image,
                overwrite=overwrite,
                verbose=verbose
            )

    def _add_single_image(
        self,
        img: Union[da.core.Array, np.ndarray, List],
        name: str,
        axes: str,
        pixel_size: Number,
        filename: Optional[str],
        ome_meta: dict,
        is_rgb: Optional[bool],
        transformation_matrix: Optional[Union[np.ndarray, str, os.PathLike, Path]],
        reference_image: Optional[str],
        overwrite: bool,
        verbose: bool
    ):
        """Internal method to add a single image (used by add_image after channel splitting)."""

        # Check if name already exists
        if name in self._names:
            if not overwrite:
                logger.info(f"`ImageData` object contains already an image with name '{name}'. Image is not added.")
                return
            else:
                # remove attribute with current name
                del self._data[name]
                # remove from name list and metadata
                self._names = [elem for elem in self._names if elem != name]
                self._metadata.pop(name, None)

        # Apply transformation if provided
        if transformation_matrix is not None:
            logger.info(f"Applying transformation to image '{name}'...")

            # Determine reference_pixel_size and output_size from reference_image if provided
            reference_pixel_size = None
            output_size = None

            if reference_image is not None:
                if reference_image not in self._names:
                    raise ValueError(
                        f"Reference image '{reference_image}' not found in ImageData. "
                        f"Available images: {self._names}"
                    )
                reference_pixel_size = self._metadata[reference_image]['pixel_size']

                # Get output_size from reference image
                ref_shape = self._metadata[reference_image]['shape']
                ref_axes = self._metadata[reference_image]['axes']
                ref_axes_config = ImageAxes(ref_axes)

                # Get height and width from reference image
                ref_height = ref_shape[ref_axes_config.Y]
                ref_width = ref_shape[ref_axes_config.X]

                # Convert to physical coordinates (um)
                output_size = (
                    ref_height * reference_pixel_size,
                    ref_width * reference_pixel_size
                )

                logger.info(f"Using reference image '{reference_image}' (pixel size: {reference_pixel_size} um/pixel, "
                            f"shape: {ref_height}x{ref_width} pixels = {output_size[0]:.1f}x{output_size[1]:.1f} um)")

            # Create a temporary ImageData object to use the transform method
            temp_img_data = ImageData()
            temp_img_data._data[name] = img
            temp_img_data._names = [name]
            temp_img_data._metadata[name] = {
                'pixel_size': pixel_size,
                'axes': axes
            }

            # Apply transformation
            temp_img_data.transform(
                transformation_matrix=transformation_matrix,
                source_pixel_size=pixel_size,
                reference_pixel_size=reference_pixel_size,
                output_size=output_size,
                inplace=True,
                verbose=verbose
            )

            # Get transformed image
            img = temp_img_data._data[name]

            # Update axes if needed (transform maintains axes)
            axes = temp_img_data._metadata[name]['axes']

        # set attribute and add names to object
        self._data[name] = img
        self._names.append(name)

        # retrieve metadata
        img_shape = img[0].shape if isinstance(img, list) else img.shape

        # save metadata
        self._metadata[name] = {}
        self._metadata[name]["filename"] = filename
        self._metadata[name]["shape"] = img_shape  # store shape
        self._metadata[name]["axes"] = axes
        self._metadata[name]["OME"] = ome_meta

        self._metadata[name]['pixel_size'] = pixel_size

        # check whether the image is RGB or not
        if is_rgb is None:
            if len(img_shape) == 3:
                channels = img_shape[2]
                if channels == 3:
                    self._metadata[name]["rgb"] = True
                else:
                    self._metadata[name]["rgb"] = False
            elif len(img_shape) == 2:
                self._metadata[name]["rgb"] = False
            else:
                raise ValueError(f"Unknown image shape: {img_shape}")
        else:
            self._metadata[name]["rgb"] = is_rgb

    def remove_image(
        self,
        names: Union[str, List[str]],
        verbose: bool = True
    ):
        """Remove one or more images from the ImageData object.

        Args:
            names: Name or list of names of images to remove.
            verbose: If True, print status messages for each removed image.

        Raises:
            KeyError: If any of the specified image names is not found.
        """
        names = convert_to_list(names)

        for name in names:
            if name not in self._names:
                raise KeyError(f"Image '{name}' not found in ImageData. Available images: {self._names}")

            del self[name]  # Uses __delitem__

            logger.info(f"Removed image '{name}'")

    def load(self,
             which: Union[List[str], str] = "all"
             ):
        '''
        Load images into memory.
        '''
        if which == "all":
            which = self._names

        # make sure which is a list
        which = convert_to_list(which)
        for n in which:
            img_loaded = self[n].compute()
            self._data[n] = img_loaded

    def crop(self,
             xlim: Optional[Tuple[int, int]],
             ylim: Optional[Tuple[int, int]],
             inplace: bool = False
             ):
        """Crop all images to a spatial bounding box.

        Slices each stored image (or pyramid) to the physical-unit region
        defined by *xlim* and *ylim* using
        :func:`~insitupy.images.utils.crop_dask_array_or_pyramid`, and
        records the crop coordinates in the metadata.

        Args:
            xlim: ``(x_min, x_max)`` in physical units (e.g. µm).
            ylim: ``(y_min, y_max)`` in physical units (e.g. µm).
            inplace: If True, modify this object in place; otherwise
                return a new cropped copy.

        Returns:
            ImageData or None: Cropped copy when ``inplace=False``,
            otherwise None.
        """
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()
        # extract names from metadata
        names = list(_self._metadata.keys())
        for n in names:
            # extract the image pyramid
            img_data = _self[n]

            # extract pixel size
            pixel_size = _self._metadata[n]['pixel_size']

            cropped_img_data = crop_dask_array_or_pyramid(
                data=img_data,
                xlim=xlim,
                ylim=ylim,
                pixel_size=pixel_size
            )

            # save cropping properties in metadata
            _self._metadata[n]["cropping_xlim"] = xlim
            _self._metadata[n]["cropping_ylim"] = ylim

            try:
                _self._metadata[n]["shape"] = cropped_img_data.shape
            except AttributeError:
                _self._metadata[n]["shape"] = cropped_img_data[0].shape

            # add cropped pyramid to object
            _self._data[n] = cropped_img_data

        if not inplace:
            return _self

    def save(self,
             path: Union[str, os.PathLike, Path],
             keys_to_save: Optional[str] = None,
             as_zarr: bool = True,
             zipped: bool = False,
             save_pyramid: bool = True,
             compression: Literal['jpeg', 'LZW', 'jpeg2000', 'ZLIB', None] = 'ZLIB',
             return_savepaths: bool = False,
             overwrite: bool = False,
             max_resolution: Optional[Number] = None,
             verbose: bool = False
             ):
        """
        Save images to the specified output folder in either Zarr or OME-TIFF format.

        Args:
            path (Union[str, os.PathLike, Path]): The directory where images will be saved.
            keys_to_save (Optional[str]): Specific keys of images to save. If None, all images are saved.
            as_zarr (bool): If True, save images in Zarr format. Otherwise, save as OME-TIFF.
            zipped (bool): If True and saving as Zarr, compress the Zarr files into zip archives.
            save_pyramid (bool): If True, save image pyramids (only applicable for Zarr format).
            compression (Literal['jpeg', 'LZW', 'jpeg2000', 'ZLIB', None]): Compression method for OME-TIFF files.
            return_savepaths (bool): If True, return the paths of the saved files.
            overwrite (bool): If True, overwrite existing files in the output folder.
            max_resolution (Optional[Number]): Maximum resolution for images in um per pixel.
            verbose (bool): If True, print status messages during saving.

        Returns:
            Optional[Dict[str, Path]]: A dictionary mapping image keys to their save paths if `return_savepaths` is True.
        """
        path = Path(path)

        if keys_to_save is None:
            keys_to_save = list(self._metadata.keys())
        else:
            keys_to_save = convert_to_list(keys_to_save)

        # create output directory (allow saving to existing directories)
        path.mkdir(parents=True, exist_ok=True)

        if return_savepaths:
            savepaths = {}

        for name, img_metadata in self._metadata.items():
            if name in keys_to_save:
                # extract image
                img = self[name]
                new_img_metadata = img_metadata.copy()

                axes = new_img_metadata['axes']
                pixel_size = new_img_metadata['pixel_size']

                if max_resolution is not None:
                    if max_resolution == pixel_size:
                        logger.warning(f"`max_pixel_size` ({max_resolution}) equal to `pixel_size` ({pixel_size}). Skipped resizing.")
                    elif max_resolution < pixel_size:
                        logger.warning(f"`max_pixel_size` ({max_resolution}) smaller than `pixel_size` ({pixel_size}). Skipped resizing.")
                    else:
                        # downscale image
                        if isinstance(img, list):
                            img = img[0]
                        downscale_factor = max_resolution / pixel_size

                        logger.info(f"Downscale image to {max_resolution} um per pixel by factor {downscale_factor}")
                        img = resize_image(img, scale_factor=1/downscale_factor, axes=axes)
                        img = da.from_array(img)

                        # change metadata
                        new_img_metadata['pixel_size'] = max_resolution
                        try:
                            new_img_metadata['OME']['Image']['Pixels']['PhysicalSizeX'] = str(max_resolution)
                        except KeyError:
                            new_img_metadata['OME']['PhysicalSizeX'] = str(max_resolution)

                        try:
                            new_img_metadata['OME']['Image']['Pixels']['PhysicalSizeY'] = str(max_resolution)
                        except KeyError:
                            new_img_metadata['OME']['PhysicalSizeY'] = str(max_resolution)

                if as_zarr:
                    # generate filename
                    if zipped:
                        filename = name + ".zarr.zip"
                    else:
                        filename = name + ".zarr"

                    # write to zarr
                    img_path = path / filename

                    # check if file exists and handle overwrite
                    if img_path.exists() and not overwrite:
                        logger.warning(f"Image '{name}' already exists at {img_path}. Skipping. Set `overwrite=True` to overwrite.")
                        continue

                    # Safety check: prevent overwriting a zarr store that the
                    # dask array is lazily reading from.
                    if overwrite and img_path.exists() and is_from_zarr_disk(img):
                        source_path = get_zarr_source_path(img)
                        target_path = img_path.resolve()
                        if source_path is not None and source_path == target_path:
                            logger.warning(
                                f"Skipping image '{name}': the dask array is lazily backed by the "
                                f"same Zarr store at {img_path}. Writing would destroy the source "
                                f"data before it is read. To update this image, first load it into "
                                f"memory (e.g. via `.persist()` or `.compute()`), or save it under "
                                f"a different name."
                            )
                            continue
                        elif source_path is None:
                            logger.warning(
                                f"Skipping image '{name}': the dask array appears to be backed "
                                f"by a Zarr store but the source path could not be determined. "
                                f"Cannot verify it differs from the target path {img_path}. "
                                f"Overwriting could destroy the source data. To update this image, "
                                f"first load it into memory (e.g. via `.persist()` or `.compute()`), "
                                f"or save it under a different name."
                            )
                            continue

                    write_zarr(image=img, file=img_path,
                               img_metadata=new_img_metadata,
                               save_pyramid=save_pyramid,
                               axes=axes, verbose=verbose,
                               overwrite=overwrite
                               )
                else:
                    # get file name for saving
                    filename = name + ".ome.tif"

                    # check if file exists and handle overwrite
                    img_path = path / filename
                    if img_path.exists() and not overwrite:
                        logger.warning(f"Image '{name}' already exists at {img_path}. Skipping. Set `overwrite=True` to overwrite.")
                        continue

                    # retrieve image metadata for saving
                    photometric = 'rgb' if new_img_metadata['rgb'] else 'minisblack'

                    # retrieve OME metadata
                    if 'OME' not in new_img_metadata or not new_img_metadata['OME']:
                        raise ValueError(
                            f"OME metadata is missing for image '{name}'. "
                            "OME-TIFF export requires OME metadata. Save as Zarr instead."
                        )

                    ome_meta_to_retrieve = ["SignificantBits", "PhysicalSizeX", "PhysicalSizeY",
                                            "PhysicalSizeXUnit", "PhysicalSizeYUnit"]

                    try:
                        pixel_meta = new_img_metadata["OME"]["Image"]["Pixels"]
                    except KeyError:
                        pixel_meta = new_img_metadata["OME"]

                    selected_metadata = {key: pixel_meta[key] for key in ome_meta_to_retrieve if key in pixel_meta}

                    # write images as OME-TIFF
                    write_ome_tiff(image=img, file=img_path,
                                photometric=photometric, axes=axes,
                                compression=compression,
                                metadata=selected_metadata, overwrite=overwrite,
                                verbose=verbose
                                )

                if return_savepaths:
                    # collect savepaths
                    savepaths[name] = path / filename

        if return_savepaths:
            return savepaths

    def transform(
        self,
        transformation_matrix: Union[np.ndarray, str, os.PathLike, Path],
        source_pixel_size: Optional[Number] = None,
        reference_pixel_size: Optional[Number] = None,
        output_size: Optional[Tuple[Number, Number]] = None,
        inplace: bool = False,
        verbose: bool = False
    ):
        """Apply an affine transformation to all images in the ImageData object.

        Args:
            transformation_matrix: Either a 2x3 or 3x3 numpy array or a path to a CSV/Excel file.
            source_pixel_size: Pixel size (in um/pixel) of the source image.
            reference_pixel_size: Pixel size (in um/pixel) of the reference image.
            output_size: Tuple of (height, width) in physical coordinates (um).
            inplace: If True, modify the object in place.
            verbose: If True, print status messages.

        Returns:
            ImageData: Transformed ImageData object if inplace=False, else None.
        """
        import cv2

        _self = self if inplace else self.copy()

        # Load transformation matrix if it's a file path
        if isinstance(transformation_matrix, (str, os.PathLike, Path)):
            transformation_matrix = Path(transformation_matrix)
            if not transformation_matrix.exists():
                raise FileNotFoundError(f"Transformation matrix file not found: {transformation_matrix}")

            # Read file based on extension
            if transformation_matrix.suffix.lower() in ['.csv', '.txt']:
                M = pd.read_csv(transformation_matrix, header=None).values
            elif transformation_matrix.suffix.lower() in ['.xlsx', '.xls']:
                M = pd.read_excel(transformation_matrix, header=None).values
            else:
                raise ValueError(f"Unsupported file format: {transformation_matrix.suffix}. Use .csv, .txt, .xlsx, or .xls")
        else:
            M = np.array(transformation_matrix)

        # Validate matrix dimensions
        if M.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Transformation matrix must be 2x3 or 3x3, got shape {M.shape}. "
                f"Expected format:\n"
                f"[[a, b, xoff],\n"
                f" [d, e, yoff]] or with [0, 0, 1] as third row."
            )

        # Extract transformation parameters
        if M.shape == (3, 3):
            # Validate that the third row is [0, 0, 1]
            if not np.allclose(M[2, :], [0, 0, 1]):
                raise ValueError("For 3x3 matrix, third row must be [0, 0, 1]")
            M = M[:2, :]

        # Convert pixel-based matrix to physical coordinates if reference_pixel_size is provided
        if reference_pixel_size is not None:
            M = M.copy().astype(np.float64)

            if source_pixel_size is not None:
                M[:2, :2] *= (reference_pixel_size / source_pixel_size)

            M[0, 2] *= reference_pixel_size  # Convert x offset: pixels -> um
            M[1, 2] *= reference_pixel_size  # Convert y offset: pixels -> um
            logger.info(f"Converted transformation matrix from pixel coordinates "
                        f"(reference: {reference_pixel_size} um/pixel) to physical coordinates.")

        logger.info(f"Applying transformation matrix (in physical coordinates):\n{M}")

        # Apply transformation to each image
        for name in list(_self._metadata.keys()):
            img = _self._data[name]
            pixel_size = _self._metadata[name]['pixel_size']
            axes = _self._metadata[name]['axes']

            # Handle image pyramids (list of arrays)
            if isinstance(img, list):
                img_to_transform = img[0]  # Use highest resolution
                is_pyramid = True
            else:
                img_to_transform = img
                is_pyramid = False

            # Convert dask array to numpy for transformation
            if isinstance(img_to_transform, da.Array):
                img_to_transform = img_to_transform.compute()

            # Scale transformation matrix based on pixel size
            scaled_M = M.copy().astype(np.float64)
            scaled_M[0, 2] /= pixel_size  # Scale x offset: um -> pixels
            scaled_M[1, 2] /= pixel_size  # Scale y offset: um -> pixels

            # Get image dimensions
            img_axes = ImageAxes(axes)
            if output_size is not None:
                # Convert physical output size (height, width) to pixels for this image
                h = int(round(output_size[0] / pixel_size))
                w = int(round(output_size[1] / pixel_size))
            else:
                # Use input image dimensions
                h = img_to_transform.shape[img_axes.Y]
                w = img_to_transform.shape[img_axes.X]

            logger.info(f"Transforming image '{name}' with shape {img_to_transform.shape} -> output size ({w}, {h})")

            # Apply transformation based on image type (grayscale, RGB, or multichannel)
            if len(img_to_transform.shape) == 2:
                # Grayscale image (YX)
                transformed = cv2.warpAffine(
                    img_to_transform,
                    scaled_M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            elif len(img_to_transform.shape) == 3:
                if img_axes.is_rgb:
                    # RGB image - transform directly
                    transformed = cv2.warpAffine(
                        img_to_transform,
                        scaled_M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                else:
                    # Multichannel image (CYX) - transform each channel
                    n_channels = img_to_transform.shape[img_axes.C]
                    transformed_channels = []
                    for c in range(n_channels):
                        channel = np.take(img_to_transform, c, axis=img_axes.C)
                        transformed_channel = cv2.warpAffine(
                            channel,
                            scaled_M,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        transformed_channels.append(transformed_channel)
                    # Stack channels back
                    transformed = np.stack(transformed_channels, axis=img_axes.C)
            else:
                raise ValueError(f"Unsupported image shape: {img_to_transform.shape}")

            # Convert back to dask array
            transformed = da.from_array(transformed)

            # Recreate pyramid if needed
            if is_pyramid:
                transformed = create_img_pyramid(transformed, axes=axes, nsubres=len(img))

            # Update data
            _self._data[name] = transformed

            # Update shape in metadata
            if isinstance(transformed, list):
                _self._metadata[name]["shape"] = transformed[0].shape
            else:
                _self._metadata[name]["shape"] = transformed.shape

            logger.info(f"Transformed image '{name}'")

        logger.info(f"Transformed {len(_self._metadata)} images.")

        if not inplace:
            return _self

import math
from numbers import Number
from typing import List, Literal, Tuple, Union
from warnings import warn

import cv2
import dask.array as da
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from skimage.color import hed2rgb, rgb2hed

from .._exceptions import InvalidDataTypeError
from .axes import ImageAxes, get_height_and_width


def _efficiently_resize_array(array, scale_factor):
    downscale_factor =  1/scale_factor

    # calculate factor for first part of the downscaling
    first_downscale_factor = math.floor(downscale_factor)

    # perform first downscaling
    im_ds = array[::first_downscale_factor, ::first_downscale_factor]

    # calculate scale factor for remaining downscaling
    second_sf = 1 / (downscale_factor / first_downscale_factor)

    # perform final downscaling
    resized_array = zoom(im_ds, zoom=second_sf, order=0)
    return resized_array

def _get_scale_factor_from_max_res(pixel_size, max_resolution):
    if max_resolution is not None:
        if (max_resolution == pixel_size) or (max_resolution < pixel_size):
            warn(f"`max_resolution` ({max_resolution}) equal as or smaller than `pixel_size` ({pixel_size}). Skipped resizing.")
            scale_factor = None
            pass
        else:
            scale_factor = 1 / (max_resolution / pixel_size)
    else:
        scale_factor = None

    return scale_factor


def resize_image(img: NDArray,
                 dim: Tuple[int, int] = None,
                 scale_factor: float = None,
                 axes = "YXS",
                 interpolation = cv2.INTER_LINEAR
                 ):
    '''
    Resize width and height of image by scale_factor. Resizing does not affect channels.
    So far the function assumes images to be either grayscale (axes="YX"), RGB (axes="YXS") or multi-channel IF (axes="CYX").
    Time-series images (e.g. "TCYX") are not supported yet.
    '''
    # read and interpret the image axes pattern
    image_axes = ImageAxes(pattern=axes)
    channel_axis = image_axes.C

    assert image_axes.T is None, "Time-series images are not supported in `resize_image`."

    if (channel_axis is not None) & (channel_axis != len(img.shape)-1):
        # move channel axis to last position if it is not there already
        img = np.moveaxis(img, channel_axis, -1)

    assert img.dtype in [np.dtype('uint16'), np.dtype('uint8')], \
        "Image must have one of the following numpy data types: `dtype('uint8)` or `dtype('uint16)`. \
        Otherwise cv2.resize shows an error."

    if isinstance(img, da.Array):
        img = img.compute() # load into memory

    # make sure the image is np.uint8
    #img = img.astype(np.uint8)

    if dim is None and scale_factor is not None:
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

    # do resizing
    img = cv2.resize(img, dim, interpolation = interpolation)

    if (channel_axis is not None) & (channel_axis != -1):
        # move channel axis back to original position
        img = np.moveaxis(img, -1, channel_axis)

    return img



def fit_image_to_size_limit(image: NDArray,
                            axes: str,  # description of axes, e.g. YXS for RGB, CYX for IF, TYXS for time-series RGB
                            size_limit: int,
                            return_scale_factor: bool = True
                            ):
    '''
    Function to resize image if necessary (warpAffine has a size limit for the image that is transformed).
    '''
    # retrieve image dimensions
    #orig_shape_image = image.shape
    height_width_image = get_height_and_width(image=image, axes_config=ImageAxes(axes))
    #xy_shape_image = orig_shape_image[:2]

    sf_image = (size_limit-1) / np.max(height_width_image)
    #new_shape = [int(elem * sf_image) for elem in height_width_image]

    # # if image has three dimensions (RGB) add third dimensions after resizing
    # if len(image.shape) == 3:
    #         new_shape += [image.shape[-1]]
    # new_shape = tuple(new_shape)

    # resize image
    #resized_image = resize_image(image, dim=(new_shape[1], new_shape[0]), axes=axes)
    resized_image = resize_image(image, scale_factor=sf_image, axes=axes)

    if return_scale_factor:
        return resized_image, sf_image
    else:
        return resized_image

def convert_to_8bit_func(img, save_mem=True, verbose=False):
    '''
    Convert numpy array image to 8bit.
    '''
    if not img.dtype == np.dtype('uint8'):
        if save_mem:
            # for a 16-bit image at least int32 is necessary for signed integers because the value range is [-65535,...,0,...,65535]
            # or uint16 can be used as unsigned integer with only positive values
            img = np.uint16(img)
        img = (img / img.max()) * 255
        img = np.uint8(img)
    else:
        if verbose:
            print("Image is already 8-bit. Not changed.", flush=True)
    return img

def scale_to_max_width(image: np.ndarray,
                       axes: str,  # description of axes, e.g. YXS for RGB, CYX for IF, TYXS for time-series RGB
                       max_width: int = 4000,
                       use_square_area: bool = False,
                       verbose: bool = True,
                       print_spacer: str = ""
                       ):
    '''
    Function to scale image to a maximum width or square area.
    '''
    image_axes = ImageAxes(pattern=axes)
    image_yx = (image.shape[image_axes.Y], image.shape[image_axes.X])

    if not use_square_area:
        # scale to the longest side of the image. Not good for very elongated images.
        if np.max(image_yx) > max_width:
            new_shape = tuple([int(elem / np.max(image_yx) * max_width) for elem in image_yx])
        else:
            new_shape = image.shape

    else:
        # use the square area of the maximum width as measure for rescaling. Better for elongated images.
        max_square_area = max_width ** 2

        # calculate new dimensions based on the maximum square area
        long_idx = np.argmax(image_yx)  # search for position of longest dimension
        short_idx = np.argmin(image_yx)  # same for shortest
        long_side = image_yx[long_idx]  # extract longest side
        short_side = image_yx[short_idx]  # extract shortest
        dim_ratio = short_side / long_side  # calculate ratio between the two sides.
        new_long_side = int(np.sqrt(max_square_area / dim_ratio))  # calculate the length of the new longer side based on area
        new_short_side = int(new_long_side * dim_ratio) # calculate length of new shorter side based on the longer one

        # create new shape
        new_shape = [None, None]
        new_shape[long_idx] = new_long_side
        new_shape[short_idx] = new_short_side
        new_shape = tuple(new_shape)

    # resizing - caution: order of dimensions is reversed in OpenCV compared to numpy
    image_scaled = resize_image(img=image, dim=(new_shape[1], new_shape[0]), axes=axes)
    print(f"{print_spacer}Rescaled from {image.shape} to following dimensions: {image_scaled.shape}") if verbose else None

    return image_scaled

def deconvolve_he(
    img: np.ndarray,
    return_type: Literal["grayscale", "greyscale", "rgb"] = "grayscale",
    convert: bool = True # convert to 8-bit
    ) -> np.ndarray:
    '''
    Deconvolves H&E image to separately extract hematoxylin and eosin stainings.

    from: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html

    --------------
    Returns:
    For return_type "grayscale": Numpy array with shape (h, w, 3) where the 3 channels correspond to
        hematoxylin, eosin and a third channel

    For return_type "rgb": Three separate RGB images as numpy array. Order: Hematoxylin, eosin, and third green channel.
    '''
    # perform deconvolution
    ihc_hed = rgb2hed(img)

    if return_type in ["grayscale", "greyscale"]:
        # extract hematoxylin channel and convert to 8-bit
        ihc_h = ihc_hed[:, :, 0] # hematoxylin
        ihc_e = ihc_hed[:, :, 1] # eosin
        ihc_d = ihc_hed[:, :, 2] # DAB

    elif return_type == "rgb":
        # Create an RGB image for each of the stains
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    else:
        raise ValueError('Unknown `return_type`. Possible values: "grayscale" or "rgb".')

    if convert:
        ihc_h = convert_to_8bit_func(ihc_h, save_mem=False)
        ihc_e = convert_to_8bit_func(ihc_e, save_mem=False)
        ihc_d = convert_to_8bit_func(ihc_d, save_mem=False)

    return ihc_h, ihc_e, ihc_d

def create_img_pyramid(img: Union[np.ndarray, da.core.Array],
                       nsubres: int = 6,
                       scale_steps: int = 2,
                       axes: str = "YXS" # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
                       ):
    # create subresolution pyramid from mask
    img_pyramid = [img]

    for n in range(nsubres):
        # create subresolution by scaling factor 2
        img = img[::scale_steps, ::scale_steps]
        #img = resize_image(img, scale_factor=1/scale_steps, axes=axes, interpolation=cv2.INTER_LINEAR)

        # # check dtype of image
        # if img.dtype not in [np.dtype('uint16'), np.dtype('uint8')]:
        #     warnings.warn("Image does not have dtype 'uint8' or 'uint16'. Is converted to 'uint16'.")

        #     if img.dtype == np.dtype('int8'):
        #         img = img.astype('uint8')
        #     else:
        #         img = img.astype('uint16')

        try:
            # rechunk to prevent dask errors
            img = img.rechunk()
        except AttributeError:
            # in case of numpy arrays a Attribute error is thrown
            pass

        # collect subresolution
        img_pyramid.append(img)

    return img_pyramid

def crop_dask_array_or_pyramid(
    data: Union[da.core.Array, List[da.core.Array]],
    xlim: Tuple[int, int],
    ylim: Tuple[int, int],
    pixel_size: Number
    ):
    # check if image data is one dask array or a pyramid of dask arrays
    if isinstance(data, list):
        if np.all([isinstance(elem, da.core.Array) for elem in data]):
            # get scale factors between the different pyramid levels
            scale_factors = [1] + [data[i].shape[0] / data[i+1].shape[0] for i in range(len(data)-1)]
            cropped_data = []
            xlim_scaled = (xlim[0] / pixel_size, xlim[1] / pixel_size) # convert to pixel unit
            ylim_scaled = (ylim[0] / pixel_size, ylim[1] / pixel_size) # convert to pixel unit
            for img, sf in zip(data, scale_factors):
                # do cropping while taking the scale factor into account
                # scale the x and y limits
                xlim_scaled = (int(xlim_scaled[0] / sf), int(xlim_scaled[1] / sf))
                ylim_scaled = (int(ylim_scaled[0] / sf), int(ylim_scaled[1] / sf))

                # do the cropping
                cdata = img[ylim_scaled[0]:ylim_scaled[1], xlim_scaled[0]:xlim_scaled[1]]

                # rechunk the array to prevent irregular chunking
                cdata = cdata.rechunk()

                # collect cropped data
                cropped_data.append(cdata)
    else:
        if isinstance(data, da.core.Array):
            # convert to metric unit
            xlim_um = tuple([int(elem / pixel_size) for elem in xlim])
            ylim_um = tuple([int(elem / pixel_size) for elem in ylim])

            cropped_data = data[ylim_um[0]:ylim_um[1], xlim_um[0]:xlim_um[1]]

            # rechunk the array to prevent irregular chunking
            cropped_data = cropped_data.rechunk()
        else:
            raise InvalidDataTypeError(
                allowed_types=[da.core.Array, List[da.core.Array]],
                received_type=type(data)
        )

    return cropped_data

def clip_image_histogram(
    image: np.ndarray,
    lower_perc: int = 2,
    upper_perc: int = 98
    ):
    # Define the min and max intensity values
    lp, up = np.percentile(image, (lower_perc, upper_perc))
    image = np.clip((image - lp) * 255.0 / (up - lp), 0, 255).astype(np.uint8)
    return image

def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    # Apply GaussianBlur to reduce image noise if necessary
    #blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply Otsu's thresholding
    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_image

def _get_contrast_limits(img):
    # retrieve metadata
    img_max = img[0].max() if isinstance(img, list) else img.max()
    try:
        img_max = img_max.compute()
    except AttributeError:
        img_max = img_max

    return (0, img_max)
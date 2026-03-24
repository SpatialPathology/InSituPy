import gc
import logging
import os
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import List, Optional, Union

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from insitupy._core.data import InSituData
from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy.images.axes import ImageAxes, get_height_and_width
from insitupy.images.io import read_image
from insitupy.images.registration import (_percentile_scale_for_saving,
                                          register_images_standalone,
                                          save_registered_image_tiff)
from insitupy.images.utils import (deconvolve_he, resize_image,
                                   scale_to_max_width)
from insitupy.images.warp import apply_warp
from insitupy.utils.utils import convert_to_list

logger = logging.getLogger(__name__)


class ImageRegistration:
    """Deprecated: Use ``insitupy.im.register_images_standalone()`` instead.

    This class will be removed in a future release.

    The typical workflow is:

    1. Instantiate with ``image`` (the image to align) and ``template`` (the fixed reference).
    2. Call :meth:`run` to execute the full pipeline (load, feature extraction, transformation
       matrix estimation, warping).
    3. Access results via instance attributes:

    Attributes:
        T (np.ndarray): Estimated transformation matrix. Shape ``(2, 3)`` for affine
            transforms or ``(3, 3)`` for perspective (homography) transforms.
        T_to_register (np.ndarray): Transformation matrix actually applied during warping
            (may differ from ``T`` when the image was resized before registration).
        registered (np.ndarray): The warped (registered) image array with shape matching
            the template dimensions.
        kpsA (list): Detected keypoints in the source image.
        kpsB (list): Detected keypoints in the template image.
        good_matches (list): Feature matches that passed the ratio test and RANSAC.
        inlier_mask (np.ndarray or None): Boolean mask of RANSAC inliers aligned with
            ``good_matches``.
    """

    def __init__(self,
                 image: Union[np.ndarray, da.Array],
                 template: Union[np.ndarray, da.Array],
                 **kwargs,
                 ):

        import warnings as _warnings
        _warnings.warn(
            "ImageRegistration is deprecated and will be removed in a future release. "
            "Use insitupy.im.register_images_standalone() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from insitupy.images.registration import RegistrationConfig as _RC
        self._config = _RC(
            axes_moving=kwargs.get("axes_image", "YXS"),
            axes_fixed=kwargs.get("axes_template", "YX"),
            max_width=kwargs.get("max_width", 4000),
            convert_to_grayscale=kwargs.get("convert_to_grayscale", False),
            deconvolve_moving=kwargs.get("deconvolve_image", False),
            deconvolve_fixed=kwargs.get("deconvolve_template", False),
            decon_scale_factor=kwargs.get("decon_scale_factor", 0.2),
            perspective_transform=kwargs.get("perspective_transform", False),
            feature_detection_method=kwargs.get("feature_detection_method", "sift"),
            flann=kwargs.get("flann", True),
            mutual_nearest_neighbor=kwargs.get("mutual_nearest_neighbor", False),
            ratio_test=kwargs.get("ratio_test", True),
            keep_fraction=kwargs.get("keepFraction", 0.2),
            max_features=kwargs.get("maxFeatures", 500),
            min_good_matches=kwargs.get("min_good_matches", 20),
            verbose=kwargs.get("verbose", True),
        )
        self._moving = image
        self._fixed = template

    def run(self):
        """Run the registration pipeline and store results on self.

        Delegates to :func:`insitupy.im.register_images_standalone`.
        Results are stored as ``self.registered`` (warped image array) and
        ``self.T`` (transformation matrix).
        """
        registered, T = register_images_standalone(
            self._moving, self._fixed,
            axes_moving=self._config.axes_moving,
            axes_fixed=self._config.axes_fixed,
            max_width=self._config.max_width,
            convert_to_grayscale=self._config.convert_to_grayscale,
            deconvolve_moving=self._config.deconvolve_moving,
            deconvolve_fixed=self._config.deconvolve_fixed,
            decon_scale_factor=self._config.decon_scale_factor,
            perspective_transform=self._config.perspective_transform,
            feature_detection_method=self._config.feature_detection_method,
            flann=self._config.flann,
            mutual_nearest_neighbor=self._config.mutual_nearest_neighbor,
            ratio_test=self._config.ratio_test,
            keep_fraction=self._config.keep_fraction,
            min_good_matches=self._config.min_good_matches,
            verbose=self._config.verbose,
        )
        self.registered = registered
        self.T = T


def register_images(
    data: InSituData,  # type: ignore
    image_path: Optional[Union[str, os.PathLike, Path]] = None,
    channel_names: Optional[Union[str, List[str]]] = None,
    channel_name_for_registration: Optional[str] = None,
    template_image_name: str = "nuclei",
    save_registered_images: bool = True,
    output_dir: Union[str, os.PathLike, Path] = None,
    min_good_matches_per_area: int = 5,  # unit: 1/mm²
    test_flipping: bool = True,
    decon_scale_factor: float = 0.2,
    deconvolve_template: bool = False,
    physicalsize: str = 'µm',
    debug: bool = False,
    rank_matches_for_qc: bool = True,
    identifier: Optional[str] = None,
    force_failure_qc: bool = False,
    *,
    image_to_be_registered: Optional[Union[str, os.PathLike, Path]] = None,
    ):
    """
    Register images stored in an InSituData object.

    Args:
        data (InSituData): The InSituData object containing the images.
        image_to_be_registered (Union[str, os.PathLike, Path]): Path to the image to be registered.
            Axes for this image are inferred from file metadata.
        image_path (Optional[Union[str, os.PathLike, Path]], optional): Alias for
            ``image_to_be_registered``. Prefer this name for new code.
        channel_names (Union[str, List[str]]): Names of the channels in the image.
        channel_name_for_registration (Optional[str], optional): Name of the channel used for registration. Required for IF images. Defaults to None.
        template_image_name (str, optional): Name of the template image. Defaults to "nuclei".
        save_registered_images (bool, optional): Whether to save the registered images. Defaults to True.
        output_dir (Union[str, os.PathLike, Path], optional): Directory where registered
            images and QC files are saved when ``save_registered_images=True``.
            Defaults to None (saves next to the InSituData project directory).
        min_good_matches_per_area (int, optional): Minimum number of good matches per mm² required for registration. Defaults to 5.
        test_flipping (bool): Whether to test flipping of images during registration. Defaults to True.
        decon_scale_factor (float, optional): Scale factor for deconvolution. Defaults to 0.2.
        deconvolve_template (bool, optional): Whether to apply HE color deconvolution to the template image.
            Set to True when the template is an H&E RGB image. Defaults to False.
        physicalsize (str, optional): Unit of physical size. Defaults to 'µm'.
        debug (bool, optional): If True, save registration QC/diagnostic files for successful runs.
            For histology registrations, also saves the deconvolved target image to
            ``registered_images/registration_qc``.
            If ``deconvolve_template=True``, also saves the deconvolved template image there.
            If False, skip routine QC file generation to speed up processing. Defaults to False.
        rank_matches_for_qc (bool, optional): If True, apply additional QC ranking before
            plotting matches. If False, keep original match order from feature extraction.
            Defaults to False.
        identifier (Optional[str], optional): An identifier string printed as a header to distinguish
            output when running in a loop. Defaults to None (auto-generated from slide/sample ID).
        force_failure_qc (bool, optional): If True, simulate a "not enough matches" failure even when
            sufficient matches are found. The QC images are saved and NotEnoughFeatureMatchesError is
            raised. Useful for testing the failure-path output without needing a bad image pair. Defaults to False.

    Raises:
        ValueError: If neither `image_to_be_registered` nor `image_path` is provided,
            or if both are provided at the same time.
        ValueError: If inferred `axes_image` is "CYX"/"YXC" and `channel_name_for_registration` is None.
        FileNotFoundError: If the image to be registered is not found.
        ValueError: If more than one image name is retrieved for histo images.
        ValueError: If no image name is found in the file.
        ValueError: If inferred `axes_image` has an unknown configuration.
        ValueError: If no channel indicator `C` is found in the image axes for IF images.
        ValueError: If inferred template axes metadata is missing.
        ValueError: If deconvolve_template is True but inferred `axes_template` is not RGB (YXS/SYX).
        ValueError: If IF channel metadata is inconsistent (channel count mismatch, duplicates,
            missing registration channel, or no channels left to register).
        ValueError: If decon_scale_factor is not strictly positive.

    Returns:
        None: The registered image(s) are added directly to ``data.images`` in place.
            If ``save_registered_images=True``, OME-TIFF files are also written to
            ``output_dir``.
    """
    # Tree drawing characters
    _TSIGN = "\u251c"   # ├
    _LSIGN = "\u2514"   # └
    _VLINE = "\u2502"   # │
    _HLINE = "\u2500"   # ─
    _SEP   = "\u2501"   # ━

    _t_start = time.time()
    tracemalloc.start()

    def _unwrap_first_level_image(img_obj, image_name: str):
        """Return the highest-resolution level when image-like input is nested as list/tuple."""
        if isinstance(img_obj, (list, tuple)):
            if len(img_obj) == 0:
                raise ValueError(f"Image '{image_name}' is empty.")
            level0 = img_obj[0]
            while isinstance(level0, (list, tuple)):
                if len(level0) == 0:
                    raise ValueError(f"Image '{image_name}' has an empty nested pyramid level.")
                level0 = level0[0]
            return level0
        return img_obj

    if decon_scale_factor <= 0:
        raise ValueError(
            f"`decon_scale_factor` must be > 0 (typically between 0.1 and 1.0), "
            f"got {decon_scale_factor}."
        )

    if image_path is not None and image_to_be_registered is not None:
        raise ValueError("Provide only one of `image_to_be_registered` or `image_path`, not both.")

    if image_path is not None:
        image_to_be_registered = image_path
    elif image_to_be_registered is not None:
        warnings.warn(
            "`image_to_be_registered` is deprecated and will be removed in a future release. "
            "Use `image_path` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        raise ValueError("Either `image_path` or `image_to_be_registered` must be provided.")

    if channel_names is None:
        raise ValueError("`channel_names` must be provided.")

    # make sure the given image names are in a list
    channel_names = convert_to_list(channel_names)

    if output_dir is None:
        output_dir = data.path.parent / "registered_images"
    else:
        output_dir = Path(output_dir) / "registered_images"
        output_dir.mkdir(parents=True, exist_ok=True)

    # check if image path exists
    image_to_be_registered = Path(image_to_be_registered)
    if not image_to_be_registered.is_file():
        raise FileNotFoundError(f"No such file found: {str(image_to_be_registered)}")

    # check that images are loaded
    if data.images.is_empty or template_image_name not in data.images:
        raise ValueError(
            f"Template image '{template_image_name}' not found in data.images. "
            f"Available images: {data.images.names}. "
            f"Use `data.load_images(names='{template_image_name}')` to load it first."
        )

    # read images
    logger.info("%s%s%s Loading images", _TSIGN, _HLINE, _HLINE)
    image, ome_meta, axes_image, pixel_size_image = read_image(image_to_be_registered)
    image = _unwrap_first_level_image(image, str(image_to_be_registered))

    # infer template axes from loaded image metadata
    axes_template = data.images.metadata[template_image_name].get("axes")
    if axes_template is None:
        raise ValueError(
            f"Template image '{template_image_name}' has no 'axes' metadata. "
            "Please ensure image metadata includes axes information."
        )

    if axes_image == "YXS":
        image_type = "histo"
    elif axes_image in ["CYX", "YXC"]:
        image_type = "IF"
    else:
        raise ValueError(
            f"Unknown inferred axes configuration '{axes_image}' for target image. "
            "Expected 'YXS' for histology RGB or 'CYX'/'YXC' for IF images."
        )

    # make sure channel naming is consistent with image type
    if image_type == "histo":
        if len(channel_names) > 1:
            raise ValueError(f"More than one image name retrieved ({channel_names})")
        if len(channel_names) == 0:
            raise ValueError(f"No image name found in file {image_to_be_registered}")

    # Print header
    _header_id = identifier if identifier is not None else f"{data.slide_id}__{data.sample_id}"
    _header_channels = ", ".join(channel_names)
    logger.info("%s", _SEP * 80)
    logger.info("Registration: %s %s%s %s (%s)", _header_id, _HLINE, _HLINE, _header_channels, image_type)
    logger.info("%s", _SEP * 80)

    # if image type is IF, the channel name for registration needs to be given
    if image_type == "IF" and channel_name_for_registration is None:
        raise ValueError("For IF images (`axes_image` in {'CYX', 'YXC'}), `channel_name_for_registration` must be provided.")

    # sometimes images are read with an empty time dimension in the first axis
    if len(image.shape) == 4:
        image = image[0]

    if image_type == "IF":
        channel_axis = axes_image.find("C")
        if channel_axis == -1:
            raise ValueError(f"No channel indicator `C` found in image axes ({axes_image})")

        n_image_channels = image.shape[channel_axis]
        n_named_channels = len(channel_names)
        if n_named_channels != n_image_channels:
            raise ValueError(
                "Mismatch between `channel_names` and image channels: "
                f"len(channel_names)={n_named_channels}, image channels={n_image_channels} "
                f"(axes_image='{axes_image}', image.shape={image.shape}, channel_axis={channel_axis})."
            )

        if len(set(channel_names)) != n_named_channels:
            raise ValueError(f"`channel_names` must be unique for IF images, got {channel_names}.")

        if channel_name_for_registration not in channel_names:
            raise ValueError(
                f"`channel_name_for_registration` ('{channel_name_for_registration}') "
                f"was not found in `channel_names`: {channel_names}."
            )

        if n_named_channels == 1 and channel_name_for_registration == channel_names[0]:
            raise ValueError(
                "No channels remain to register: `channel_names` only contains "
                "`channel_name_for_registration`. Provide at least one additional channel."
            )

    # Load template
    template = data.images[template_image_name][0]
    template = _unwrap_first_level_image(template, template_image_name)
    logger.info("%s     Image:    %s", _VLINE, image.shape)
    logger.info("%s     Template: %s", _VLINE, template.shape)

    # get pixel size from template image metadata
    pixel_size_template = data.images.metadata[template_image_name]["pixel_size"]

    # generate OME metadata for saving
    ome_metadata = {
        'SignificantBits': 8,
        'PhysicalSizeXUnit': physicalsize,
        'PhysicalSizeYUnit': physicalsize,
        'PhysicalSizeX': pixel_size_template,
        'PhysicalSizeY': pixel_size_template,
    }

    # determine minimum number of good matches required
    h, w = template.shape[:2]
    image_area = h * w * pixel_size_template ** 2 / 1000 ** 2  # in mm²
    min_good_matches = int(min_good_matches_per_area * image_area)

    # Validate deconvolve_template parameter
    if deconvolve_template and axes_template not in ["YXS", "SYX"]:
        raise ValueError(
            f"deconvolve_template=True requires RGB template with axes 'YXS' or 'SYX', "
            f"got '{axes_template}'"
        )

    # QC directory (used for both debug and failure QC)
    qc_dir_resolved = Path(output_dir) / "registration_qc"
    if debug:
        logger.info("%s%s%s QC directory: %s", _TSIGN, _HLINE, _HLINE, qc_dir_resolved)

    if image_type == "histo":
        save_identifier = identifier if identifier is not None else f"{data.slide_id}__{data.sample_id}__{channel_names[0]}"

        try:
            registered, T = register_images_standalone(
                moving=image,
                fixed=template,
                axes_moving=axes_image,
                axes_fixed=axes_template,
                deconvolve_moving=True,
                deconvolve_fixed=deconvolve_template,
                decon_scale_factor=decon_scale_factor,
                min_good_matches=min_good_matches,
                test_flipping=test_flipping,
                debug=debug,
                qc_dir=qc_dir_resolved,
                qc_identifier=save_identifier,
                rank_matches_for_qc=rank_matches_for_qc,
                pixel_size_moving=pixel_size_image,
                pixel_size_fixed=pixel_size_template,
                physical_size_unit=physicalsize,
                force_failure=force_failure_qc,
            )
        except NotEnoughFeatureMatchesError as exc:
            # Preserve failure QC even when debug=False (debug=True already handled by standalone)
            if not debug and output_dir is not None:
                _failed_identifier = f"{data.slide_id}__{data.sample_id}__{channel_names[0]}__FAILED"
                partial = exc.partial_result
                if partial is not None and partial.matchedVis is not None:
                    logger.info("%s%s%s Saving failure QC images", _TSIGN, _HLINE, _HLINE)
                    qc_dir_resolved.mkdir(parents=True, exist_ok=True)
                    import matplotlib.pyplot as plt
                    plt.imshow(partial.matchedVis)
                    plt.savefig(qc_dir_resolved / f"{_failed_identifier}__matches_overview.png", dpi=400)
                    plt.close()
            raise

        if save_registered_images:
            _outfile = save_registered_image_tiff(
                output_dir=output_dir,
                identifier=save_identifier,
                registered=registered,
                axes=axes_image,
                photometric='rgb',
                ome_metadata=ome_metadata,
            )
            logger.info("%s     Saved: %s", _VLINE, _outfile)

        data.images.add_image(
            image=registered,
            channel_names=channel_names[0],
            axes=axes_image,
            pixel_size=pixel_size_template,
            ome_meta=ome_metadata,
            overwrite=True,
        )

    else:
        # image_type is IF
        channel_id_for_registration = channel_names.index(channel_name_for_registration)
        logger.info("%s%s%s Selecting registration channel (index: %s)", _TSIGN, _HLINE, _HLINE, channel_id_for_registration)

        nuclei_img = np.take(image, channel_id_for_registration, channel_axis)
        if hasattr(nuclei_img, "compute"):
            nuclei_img = nuclei_img.compute()

        _qc_ref_name = channel_name_for_registration
        qc_identifier_if = f"{data.slide_id}__{data.sample_id}__{_qc_ref_name}"

        try:
            _, T = register_images_standalone(
                moving=nuclei_img,
                fixed=template,
                axes_moving="YX",
                axes_fixed=axes_template,
                deconvolve_fixed=deconvolve_template,
                decon_scale_factor=decon_scale_factor,
                min_good_matches=min_good_matches,
                test_flipping=test_flipping,
                debug=debug,
                qc_dir=qc_dir_resolved,
                qc_identifier=qc_identifier_if,
                rank_matches_for_qc=rank_matches_for_qc,
                pixel_size_moving=pixel_size_image,
                pixel_size_fixed=pixel_size_template,
                physical_size_unit=physicalsize,
                force_failure=force_failure_qc,
            )
        except NotEnoughFeatureMatchesError as exc:
            # Preserve failure QC even when debug=False
            if not debug and output_dir is not None:
                _failed_identifier = f"{data.slide_id}__{data.sample_id}__{_qc_ref_name}__FAILED"
                partial = exc.partial_result
                if partial is not None and partial.matchedVis is not None:
                    logger.info("%s%s%s Saving failure QC images", _TSIGN, _HLINE, _HLINE)
                    qc_dir_resolved.mkdir(parents=True, exist_ok=True)
                    import matplotlib.pyplot as plt
                    plt.imshow(partial.matchedVis)
                    plt.savefig(qc_dir_resolved / f"{_failed_identifier}__matches_overview.png", dpi=400)
                    plt.close()
            raise

        del nuclei_img

        # Compute output dimensions from template
        ref_h, ref_w = get_height_and_width(
            template if not hasattr(template, "compute") else template.compute(),
            ImageAxes(axes_template),
        )

        # Warp each non-registration channel using the shared transformation matrix
        for i, n in enumerate(channel_names):
            if n == channel_name_for_registration:
                continue

            logger.info("%s%s%s Registering channel: %s", _TSIGN, _HLINE, _HLINE, n)
            channel = np.take(image, i, channel_axis)
            if hasattr(channel, "compute"):
                channel = channel.compute()
            channel = np.asarray(channel)

            registered_channel = apply_warp(channel, T, (ref_w, ref_h), "YX")

            if save_registered_images:
                save_identifier = f"{data.slide_id}__{data.sample_id}__{n}"
                _outfile = save_registered_image_tiff(
                    output_dir=output_dir,
                    identifier=save_identifier,
                    registered=registered_channel,
                    axes='YX',
                    photometric='minisblack',
                    ome_metadata=ome_metadata,
                )
                logger.info("%s     Saved: %s", _VLINE, _outfile)

            data.images.add_image(
                image=registered_channel,
                channel_names=n,
                axes="YX",
                pixel_size=pixel_size_template,
                ome_meta=ome_metadata,
                overwrite=True,
            )

    _elapsed = time.time() - _t_start
    _, _peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_mem_str = f"{_peak_mem / 1024**3:.2f} GB" if _peak_mem >= 1024**3 else f"{_peak_mem / 1024**2:.1f} MB"
    logger.info("%s%s%s Done (%.1f s, peak memory: %s)", _LSIGN, _HLINE, _HLINE, _elapsed, _peak_mem_str)
    gc.collect()



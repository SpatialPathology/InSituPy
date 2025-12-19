import gc
import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from dask_image.imread import imread

from insitupy import __version__
from insitupy._constants import CACHE, SHRT_MAX
from insitupy._core.data import InSituData
from insitupy._exceptions import (NotEnoughFeatureMatchesError,
                                  UnknownOptionError)
from insitupy._textformat import textformat as tf
from insitupy.images.axes import ImageAxes, get_height_and_width
from insitupy.images.io import write_ome_tiff
from insitupy.images.utils import (clip_image_histogram, convert_to_8bit_func,
                                   deconvolve_he, fit_image_to_size_limit,
                                   otsu_thresholding, resize_image,
                                   scale_to_max_width)
from insitupy.utils.utils import convert_to_list, remove_last_line_from_csv


class ImageRegistration:
    '''
    Object to perform image registration.
    '''
    def __init__(self,
                 image: Union[np.ndarray, da.Array],
                 template: Union[np.ndarray, da.Array],
                 axes_image: str = "YXS", ## channel axes - other examples: 'TCYXS'. S for RGB channels.
                 axes_template: str = "YX",  # channel axes of template. Normally it is just a grayscale image - therefore YX.
                 max_width: Optional[int] = 4000,
                 convert_to_grayscale: bool = False,
                 deconvolve_image: bool = False,  # whether to apply HE deconvolution to the image
                 deconvolve_template: bool = False,  # whether to apply HE deconvolution to the template
                 decon_scale_factor: float = 0.2,  # scale factor for deconvolution to save memory
                 perspective_transform: bool = False,
                 feature_detection_method: Literal["sift", "surf"] = "sift",
                 flann: bool = True,
                 ratio_test: bool = True,
                 keepFraction: float = 0.2,
                 min_good_matches: int = 20,  # minimum number of good feature matches
                 maxFeatures: int = 500,
                 verbose: bool = True,
                 ):

        # check verbose mode
        self.verboseprint = print if verbose else lambda *a, **k: None

        # add arguments to object
        self.image = image
        self.template = template
        self.axes_image = axes_image
        self.axes_template = axes_template
        self.axes_config_image = ImageAxes(self.axes_image)
        self.axes_config_template = ImageAxes(self.axes_template)
        self.max_width = max_width
        self.convert_to_grayscale = convert_to_grayscale
        self.deconvolve_image = deconvolve_image
        self.deconvolve_template = deconvolve_template
        self.decon_scale_factor = decon_scale_factor
        self.perspective_transform = perspective_transform
        self.feature_detection_method = feature_detection_method
        self.flann = flann
        self.ratio_test = ratio_test
        self.keepFraction = keepFraction
        self.min_good_matches = min_good_matches
        self.maxFeatures = maxFeatures
        self.verbose = verbose

    def _deconvolve_he_image(self, img: np.ndarray, axes: str, name: str = "image") -> np.ndarray:
        """
        Apply HE color deconvolution to extract nuclei channel from an H&E stained image.

        Args:
            img: Input H&E RGB image
            axes: Axes configuration of the image (e.g., "YXS")
            name: Name for logging purposes ("image" or "template")

        Returns:
            Grayscale nuclei image after deconvolution
        """
        self.verboseprint(f"\t\tRun color deconvolution on {name}...", flush=True)

        # deconvolve HE - performed on resized image to save memory
        nuclei_img, _, _ = deconvolve_he(
            img=resize_image(img, scale_factor=self.decon_scale_factor, axes=axes),
            return_type="grayscale",
            convert=True
        )

        # bring back to original size
        nuclei_img = resize_image(nuclei_img, scale_factor=1/self.decon_scale_factor, axes="YX")

        return nuclei_img

    def load_and_scale_images(self):
        if not HAS_OPENCV:
            raise ImportError("OpenCV (cv2) is required for image registration. Install it with: pip install opencv-python")

        # load images into memory if they are dask arrays
        if isinstance(self.image, da.Array):
            self.verboseprint("\t\tLoad image into memory...", flush=True)
            self.image = self.image.compute()  # load into memory

        if isinstance(self.template, da.Array):
            self.verboseprint("\t\tLoad template into memory...", flush=True)
            self.template = self.template.compute()  # load into memory

        # Apply HE deconvolution if requested (before grayscale conversion)
        # Store original images for later use in registration
        self.image_original = self.image
        self.template_original = self.template

        if self.deconvolve_image:
            if self.axes_image not in ["YXS", "SYX"]:
                raise ValueError(f"HE deconvolution requires RGB image with axes 'YXS' or 'SYX', got '{self.axes_image}'")
            self.image = self._deconvolve_he_image(self.image, self.axes_image, "image")
            self.axes_image = "YX"  # update axes after deconvolution
            self.axes_config_image = ImageAxes(self.axes_image)

        if self.deconvolve_template:
            if self.axes_template not in ["YXS", "SYX"]:
                raise ValueError(f"HE deconvolution requires RGB template with axes 'YXS' or 'SYX', got '{self.axes_template}'")
            self.template = self._deconvolve_he_image(self.template, self.axes_template, "template")
            self.axes_template = "YX"  # update axes after deconvolution
            self.axes_config_template = ImageAxes(self.axes_template)

        if self.convert_to_grayscale:
            # check format
            if len(self.image.shape) == 3:
                self.verboseprint("\t\tConvert image to grayscale...")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if len(self.template.shape) == 3:
                self.verboseprint("\t\tConvert template to grayscale...")
                self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        if self.max_width is not None:
            self.verboseprint("\t\tRescale image and template to save memory.", flush=True)
            self.image_scaled = scale_to_max_width(self.image,
                                                   axes=self.axes_image,
                                                   max_width=self.max_width,
                                                   use_square_area=True,
                                                   verbose=self.verbose,
                                                   print_spacer="\t\t\t"
                                                   )
            self.template_scaled = scale_to_max_width(self.template,
                                                      axes=self.axes_template,
                                                      max_width=self.max_width,
                                                      use_square_area=True,
                                                      verbose=self.verbose,
                                                      print_spacer="\t\t\t"
                                                      )
        ##TODO: Should we delete the self.image after this step to free memory?
        else:
            self.image_scaled = self.image
            self.template_scaled = self.template

        # convert and normalize images to 8bit for registration
        self.verboseprint("\t\tConvert scaled images to 8 bit")
        self.image_scaled = convert_to_8bit_func(self.image_scaled)
        self.template_scaled = convert_to_8bit_func(self.template_scaled)

        # calculate scale factors for x and y dimension for image and template
        # TODO: Do we really nead to do this separately for both axes?
        self.x_sf_image = self.image_scaled.shape[1] / self.image.shape[1]
        self.y_sf_image = self.image_scaled.shape[0] / self.image.shape[0]
        self.x_sf_template = self.template_scaled.shape[1] / self.template.shape[1]
        self.y_sf_template = self.template_scaled.shape[0] / self.template.shape[0]

        # resize image if necessary (warpAffine has a size limit for the image that is transformed)
        # get width and height of image

        h_image, w_image = get_height_and_width(image=self.image, axes_config=self.axes_config_image)
        # if np.any([elem > SHRT_MAX for elem in self.image.shape[:2]]):
        if np.any([elem > SHRT_MAX for elem in (h_image, w_image)]):
            self.verboseprint(
                "\t\tWarning: Dimensions of image ({}) exceed C++ limit SHRT_MAX ({}). " \
                "Image dimensions are resized to meet requirements. " \
                "This leads to a loss of quality.".format(self.image.shape, SHRT_MAX))

            # fit image
            self.image_resized, self.resize_factor_image = fit_image_to_size_limit(
                self.image, size_limit=SHRT_MAX, return_scale_factor=True, axes=self.axes_image
                )
            print(f"Image dimensions after resizing: {self.image_resized.shape}. Resize factor: {self.resize_factor_image}")
        else:
            self.image_resized = None
            self.resize_factor_image = 1

    def extract_features(
        self,
        test_flipping: bool = True,
        adjust_contrast_method: Optional[Literal["otsu", "clip"]] = "clip",
        debugging: bool = False
        ):
        '''
        Function to extract paired features from image and template.
        '''

        self.verboseprint("\t\t{}: Get features...".format(f"{datetime.now():%Y-%m-%d %H:%M:%S}"))

        if test_flipping:
            # Test different flip transformations starting with no flip, then vertical, then horizontal.
            flip_axis_list = [None, 0] # before: [None, 0, 1]
        else:
            # do not test flipping of the axis
            flip_axis_list = [None]
        matches_list = [] # list to collect number of matches
        for flip_axis in flip_axis_list:
            flipped = False
            if flip_axis is not None:
                # flip image
                print(f"\t\t{'Vertical' if flip_axis == 0 else 'Horizontal'} flip is tested.", flush=True)
                self.image_scaled = np.flip(self.image_scaled, axis=flip_axis)
                flipped = True # set flipped flag to True

            # Get features
            # adjust contrast of both image and template
            if adjust_contrast_method is not None:
                self.verboseprint(f"\t\t\tAdjust contrast with {adjust_contrast_method} method...")
                if adjust_contrast_method == "otsu":
                    image_contrast_adj = otsu_thresholding(image=convert_to_8bit_func(self.image_scaled))
                    template_contrast_adj = otsu_thresholding(image=convert_to_8bit_func(self.template_scaled))
                elif adjust_contrast_method == "clip":
                    image_contrast_adj = clip_image_histogram(image=self.image_scaled, lower_perc=20, upper_perc=99)
                    template_contrast_adj = clip_image_histogram(image=self.template_scaled, lower_perc=20, upper_perc=99)
                else:
                    raise ValueError(f"Invalid method {adjust_contrast_method} for `adjust_contrast_method`.")
            else:
                image_contrast_adj = self.image_scaled
                template_contrast_adj = self.template_scaled

            if debugging:
                outpath = CACHE
                plt.imshow(self.image_scaled)
                plt.savefig(outpath / f"image.png")
                plt.close()

                plt.imshow(image_contrast_adj)
                plt.savefig(outpath / f"image_{adjust_contrast_method}.png")
                plt.close()

                plt.imshow(self.template_scaled)
                plt.savefig(outpath / f"template.png")
                plt.close()

                plt.imshow(template_contrast_adj)
                plt.savefig(outpath / f"template_{adjust_contrast_method}.png")
                plt.close()

            if self.feature_detection_method == "sift":
                self.verboseprint("\t\t\tMethod: SIFT...")
                # sift
                sift = cv2.SIFT_create()

                (kpsA, descsA) = sift.detectAndCompute(image_contrast_adj, None)
                (kpsB, descsB) = sift.detectAndCompute(template_contrast_adj, None)

            elif self.feature_detection_method == "surf":
                self.verboseprint("\t\t\tMethod: SURF...")
                surf = cv2.xfeatures2d.SURF_create(400)

                (kpsA, descsA) = surf.detectAndCompute(image_contrast_adj, None)
                (kpsB, descsB) = surf.detectAndCompute(template_contrast_adj, None)

            else:
                self.verboseprint("\t\t\tUnknown method. Aborted.")
                return

            if self.flann:
                self.verboseprint("\t\t{}: Compute matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)   # or pass empty dictionary

                # runn Flann matcher
                fl = cv2.FlannBasedMatcher(index_params, search_params)
                matches = fl.knnMatch(descsA, descsB, k=2)

            else:
                self.verboseprint("\t\t{}: Compute matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # feature matching
                #bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descsA, descsB, k=2)

            if self.ratio_test:
                self.verboseprint("\t\t{}: Filter matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # store all the good matches as per Lowe's ratio test.
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7*n.distance:
                        good_matches.append(m)
            else:
                self.verboseprint("\t\t{}: Filter matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
                matches = sorted(matches, key=lambda x: x.distance)
                # keep only the top matches
                keep = int(len(matches) * self.keepFraction)
                good_matches = matches[:keep][:self.maxFeatures]

                self.verboseprint("\t\t\tNumber of matches used: {}".format(len(good_matches)))

            # check if a sufficient number of good matches was found
            matches_list.append(len(good_matches))
            if len(good_matches) >= self.min_good_matches:
                print(f"\t\t\tSufficient number of good matches found ({len(good_matches)}/{self.min_good_matches}).")
                self.flip_axis = flip_axis
                break
            else:
                print(f"\t\t\tNumber of good matches ({len(good_matches)}) below threshold ({self.min_good_matches}). Flipping is tested.")
                if flipped:
                    # flip back
                    print("Flip back.", flush=True)
                    self.image_scaled = np.flip(self.image_scaled, axis=flip_axis)

        if not hasattr(self, "flip_axis"):
            raise NotEnoughFeatureMatchesError(number=np.max(matches_list), threshold=self.min_good_matches)

        # check to see if we should visualize the matched keypoints
        self.verboseprint("\t\t{}: Display matches...".format(f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        self.matchedVis = cv2.drawMatches(self.image_scaled, kpsA, self.template_scaled, kpsB,
                                        good_matches, None)

        # Get keypoints
        self.verboseprint("\t\t{}: Fetch keypoints...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        # allocate memory for the keypoints (x, y)-coordinates of the top matches
        self.ptsA = np.zeros((len(good_matches), 2), dtype="float")
        self.ptsB = np.zeros((len(good_matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(good_matches):
            # indicate that the two keypoints in the respective images map to each other
            self.ptsA[i] = kpsA[m.queryIdx].pt
            self.ptsB[i] = kpsB[m.trainIdx].pt

        # apply scale factors to points - separately for each dimension
        self.ptsA[:, 0] = self.ptsA[:, 0] / self.x_sf_image
        self.ptsA[:, 1] = self.ptsA[:, 1] / self.y_sf_image
        self.ptsB[:, 0] = self.ptsB[:, 0] / self.x_sf_template
        self.ptsB[:, 1] = self.ptsB[:, 1] / self.y_sf_template

    def calculate_transformation_matrix(self):
        '''
        Function to calculate the transformation matrix.
        '''
        if not HAS_OPENCV:
            raise ImportError("OpenCV (cv2) is required for calculating transformation matrix. Install it with: pip install opencv-python")

        if self.perspective_transform:
            # compute the homography matrix between the two sets of matched
            # points
            self.verboseprint(f"\t\t{datetime.now():%Y-%m-%d %H:%M:%S}: Compute homography matrix...")
            (self.T, mask) = cv2.findHomography(self.ptsA, self.ptsB, method=cv2.RANSAC)
        else:
            self.verboseprint(f"\t\t{datetime.now():%Y-%m-%d %H:%M:%S}: Estimate 2D affine transformation matrix...")
            (self.T, mask) = cv2.estimateAffine2D(self.ptsA, self.ptsB)

        if self.resize_factor_image != 1:
            if self.perspective_transform:
                self.verboseprint("\t\tEstimate perspective transformation matrix for resized image", flush=True)
                self.ptsA *= self.resize_factor_image # scale images features in case it was originally larger than the warpAffine limits
                (self.T_resized, mask) = cv2.findHomography(self.ptsA, self.ptsB, method=cv2.RANSAC)
            else:
                self.verboseprint("\t\tEstimate affine transformation matrix for resized image", flush=True)
                self.ptsA *= self.resize_factor_image # scale images features in case it was originally larger than the warpAffine limits
                (self.T_resized, mask) = cv2.estimateAffine2D(self.ptsA, self.ptsB)

    def perform_registration(self):

        # determine which image to be registered here
        if self.image_resized is None:
            self.image_to_register = self.image
            self.T_to_register = self.T
        else:
            self.image_to_register = self.image_resized
            self.T_to_register = self.T_resized

        # determine the kind of transformation
        warp_func, warp_name = (cv2.warpPerspective, "perspective") if self.perspective_transform else (cv2.warpAffine, "affine")

        if self.flip_axis is not None:
            print(f"\t\tImage is flipped {'vertically' if self.flip_axis == 0 else 'horizontally'}", flush=True)
            self.image_to_register = np.flip(self.image_to_register, axis=self.flip_axis)

        # use the transformation matrix to register the images
        # TODO: not very safe to use here "[:2]"
        (h, w) = self.template.shape[:2]
        # warping
        self.verboseprint(f"\t\t{datetime.now():%Y-%m-%d %H:%M:%S}: Register image by {warp_name} transformation...")
        self.registered = warp_func(self.image_to_register, self.T_to_register, (w, h))

    def run(self):
        '''
        Run the complete registration pipeline including following steps:
            1. Loading of images
            2. Feature extraction
            3. Calculation of transformation matrix
            4. Registration of images based on transformation matrix
        '''
        # load and scale images
        self.load_and_scale_images()

        # run feature extraction
        self.extract_features()

        # calculate transformation matrix
        self.calculate_transformation_matrix()

        # perform registration
        self.perform_registration()

    def save_registered_image(
        self,
        output_dir: Union[str, os.PathLike, Path],
        identifier: str,
        axes: str,  # string describing the channel axes, e.g. YXS or CYX
        photometric: Literal['rgb', 'minisblack', 'maxisblack'] = 'rgb',
        ome_metadata: dict = {},
        registered: Optional[np.ndarray] = None,  # registered image
        ):
        """
        Save the registered image as OME-TIFF.

        Args:
            output_dir: Directory to save the registered image.
            identifier: Identifier string for the output filename.
            axes: String describing the channel axes, e.g. 'YXS' or 'CYX'.
            photometric: Photometric interpretation ('rgb', 'minisblack', 'maxisblack').
            ome_metadata: OME metadata dictionary to include in the TIFF.
            registered: Registered image array. If None, uses self.registered.
        """
        if registered is None:
            registered = self.registered

        # save registered image as OME-TIFF
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.outfile = output_dir / f"{identifier}__registered.ome.tif"
        print(f"\t\tSave OME-TIFF to {self.outfile}", flush=True)
        write_ome_tiff(
            file=self.outfile,
            image=registered,
            axes=axes,
            photometric=photometric,
            overwrite=True,
            metadata=ome_metadata
            )

    def save_qc(
        self,
        output_dir: Union[str, os.PathLike, Path],
        identifier: str,
        _T: Optional[np.ndarray] = None,  # transformation matrix
        matchedVis: Optional[np.ndarray] = None  # image showing the matched visualization
        ):
        """
        Save registration QC files (transformation matrix and feature matches visualization).

        Args:
            output_dir: Directory to save QC files (will create 'registration_qc' subdirectory).
            identifier: Identifier string for the output filenames.
            _T: Transformation matrix. If None, uses the appropriate matrix from self.
            matchedVis: Image showing matched features. If None, uses self.matchedVis.
        """
        if _T is None:
            if self.resize_factor_image == 1:
                # if the image was not resized the transformation matrix to save is identical to the one used for registration
                T_to_save = self.T_to_register
            else:
                # if the image WAS resized the transformation matrix to save is not identical to the one used for registration
                # instead the transformation matrix before resizing needs to be used
                T_to_save = self.T
        else:
            T_to_save = _T

        if matchedVis is None:
            matchedVis = self.matchedVis

        # save registration QC files
        output_dir = Path(output_dir)
        reg_dir = output_dir / "registration_qc"
        reg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\t\tSave QC files to {reg_dir}", flush=True)

        # save transformation matrix
        T_to_save = np.vstack([T_to_save, [0,0,1]])  # add last line of affine transformation matrix
        T_csv = reg_dir / f"{identifier}__T.csv"
        np.savetxt(T_csv, T_to_save, delimiter=",")

        # remove last line break from csv since this gives error when importing to Xenium Explorer
        remove_last_line_from_csv(T_csv)

        # save image showing the number of key points found in both images during registration
        matchedVis_file = reg_dir / f"{identifier}__common_features.pdf"
        plt.imshow(matchedVis)
        plt.savefig(matchedVis_file, dpi=400)
        plt.close()

    def save(
        self,
        output_dir: Union[str, os.PathLike, Path],
        identifier: str,
        axes: str,  # string describing the channel axes, e.g. YXS or CYX
        photometric: Literal['rgb', 'minisblack', 'maxisblack'] = 'rgb',
        ome_metadata: dict = {},
        registered: Optional[np.ndarray] = None,  # registered image
        _T: Optional[np.ndarray] = None,  # transformation matrix
        matchedVis: Optional[np.ndarray] = None  # image showing the matched visualization
        ):
        """
        Save both the registered image and QC files.

        This is a convenience method that calls both save_registered_image() and save_qc().

        Args:
            output_dir: Directory to save output files.
            identifier: Identifier string for the output filenames.
            axes: String describing the channel axes, e.g. 'YXS' or 'CYX'.
            photometric: Photometric interpretation ('rgb', 'minisblack', 'maxisblack').
            ome_metadata: OME metadata dictionary to include in the TIFF.
            registered: Registered image array. If None, uses self.registered.
            _T: Transformation matrix. If None, uses the appropriate matrix from self.
            matchedVis: Image showing matched features. If None, uses self.matchedVis.
        """
        # Save registered image
        self.save_registered_image(
            output_dir=output_dir,
            identifier=identifier,
            axes=axes,
            photometric=photometric,
            ome_metadata=ome_metadata,
            registered=registered
        )

        # Save QC files
        self.save_qc(
            output_dir=output_dir,
            identifier=identifier,
            _T=_T,
            matchedVis=matchedVis
        )


def register_images(
    data: InSituData, # type: ignore
    image_to_be_registered: Union[str, os.PathLike, Path],
    axes_image: Literal["CYX", "YXS"],  # axes of the image to be registered, e.g. YXS for RGB images, CYX for IF images
    axes_template: Literal["YX", "CYX", "YXS"],  # axes of the template image, e.g. YX for grayscale images
    channel_names: Union[str, List[str]],
    channel_name_for_registration: Optional[str] = None,  # name used for the nuclei image. Only required for IF images.
    template_image_name: str = "nuclei",
    save_registered_images: bool = True,
    output_dir: Union[str, os.PathLike, Path] = None,
    min_good_matches_per_area: int = 5, # unit: 1/mm²
    test_flipping: bool = True,
    decon_scale_factor: float = 0.2,
    deconvolve_template: bool = False,  # whether to apply HE deconvolution to the template
    physicalsize: str = 'µm'
    ):
    """
    Register images stored in an InSituData object.

    Args:
        data (InSituData): The InSituData object containing the images.
        image_to_be_registered (Union[str, os.PathLike, Path]): Path to the image to be registered.
        axes_image (Literal["CYX", "YXS"]): Axes of the image to be registered, e.g. YXS for RGB images, CYX for IF images.
        axes_template (Literal["YX", "CYX", "YXS"]): Axes of the template image, e.g. YX for grayscale images, YXS for HE images.
        channel_names (Union[str, List[str]]): Names of the channels in the image.
        channel_name_for_registration (Optional[str], optional): Name of the channel used for registration. Required for IF images. Defaults to None.
        template_image_name (str, optional): Name of the template image. Defaults to "nuclei".
        save_registered_images (bool, optional): Whether to save the registered images. Defaults to True.
        min_good_matches_per_area (int, optional): Minimum number of good matches per mm² required for registration. Defaults to 5.
        test_flipping (bool): Whether to test flipping of images during registration. Defaults to True.
        decon_scale_factor (float, optional): Scale factor for deconvolution. Defaults to 0.2.
        deconvolve_template (bool, optional): Whether to apply HE color deconvolution to the template image.
            Set to True when the template is an H&E RGB image. Defaults to False.
        physicalsize (str, optional): Unit of physical size. Defaults to 'µm'.

    Raises:
        ValueError: If `axes_image` is "CYX"/"YXC" and `channel_name_for_registration` is None.
        FileNotFoundError: If the image to be registered is not found.
        ValueError: If more than one image name is retrieved for histo images.
        ValueError: If no image name is found in the file.
        ValueError: If an unknown axes configuration is provided.
        ValueError: If no channel indicator `C` is found in the image axes for IF images.
        ValueError: If deconvolve_template is True but axes_template is not RGB (YXS/SYX).

    Returns:
        None
    """
    # make sure the given image names are in a list
    channel_names = convert_to_list(channel_names)

    # determine the structure of the image axes and check other things
    if axes_image == "YXS":
        image_type = "histo"

        # make sure that there is only one image name given
        if len(channel_names) > 1:
            raise ValueError(f"More than one image name retrieved ({channel_names})")

        if len(channel_names) == 0:
            raise ValueError(f"No image name found in file {image_to_be_registered}")
    elif axes_image in ["CYX", "YXC"]:
        image_type = "IF"
    else:
        raise ValueError(f"Unknown axes configuration {axes_image} for target image. Please use 'YXS' for histo images or 'CYX'/'YXC' for IF images.")

    # if image type is IF, the channel name for registration needs to be given
    if image_type == "IF" and channel_name_for_registration is None:
        raise ValueError(f'If `image_type" is "IF", `channel_name_for_registration is not allowed to be `None`.')

    if output_dir is None:
        # define output directory
        output_dir = data.path.parent / "registered_images"
    else:
        output_dir = Path(output_dir) / "registered_images"
        output_dir.mkdir(parents=True, exist_ok=True)

    # if output_dir.is_dir() and not force:
    #     raise FileExistsError(f"Output directory {output_dir} exists already. If you still want to run the registration, set `force=True`.")

    # check if image path exists
    image_to_be_registered = Path(image_to_be_registered)
    if not image_to_be_registered.is_file():
        raise FileNotFoundError(f"No such file found: {str(image_to_be_registered)}")

    # axes_template = "YX"
    # if image_type == "histo":
    #     axes_image = "YXS"

    #     # make sure that there is only one image name given
    #     if len(channel_names) > 1:
    #         raise ValueError(f"More than one image name retrieved ({channel_names})")

    #     if len(channel_names) == 0:
    #         raise ValueError(f"No image name found in file {image_to_be_registered}")

    # elif image_type == "IF":
    #     axes_image = "CYX"
    # else:
    #     raise UnknownOptionError(image_type, available=["histo", "IF"])

    print(f'\tProcessing following {image_type} images: {tf.Bold}{", ".join(channel_names)}{tf.ResetAll}', flush=True)

    # read images
    print("\t\tLoading images to be registered...", flush=True)
    image = imread(image_to_be_registered) # e.g. HE image

    # sometimes images are read with an empty time dimension in the first axis.
    # If this is the case, it is removed here.
    if len(image.shape) == 4:
        image = image[0]

    # # read images in InSituData object
    template = data.images[template_image_name][0] # usually the nuclei/DAPI image is the template. Use highest resolution of pyramid.

    # extract OME metadata
    #ome_metadata_template = data.images.metadata[template_image_name]["OME"]

    # get pixel size from image metadata
    pixel_size = data.images.metadata[template_image_name]["pixel_size"]

    # extract pixel size for x and y from OME metadata
    #pixelsizes = {key: ome_metadata_template['Image']['Pixels'][key] for key in ['PhysicalSizeX', 'PhysicalSizeY']}

    # generate OME metadata for saving
    ome_metadata = {
        'SignificantBits': 8,
        'PhysicalSizeXUnit': physicalsize,
        'PhysicalSizeYUnit': physicalsize,
        'PhysicalSizeX': pixel_size,
        'PhysicalSizeY': pixel_size
        }

    # determine minimum number of good matches that are necessary for the registration to be performed
    h, w = template.shape[:2]
    image_area = h * w * pixel_size**2 / 1000**2 # in mm²
    min_good_matches = int(min_good_matches_per_area * image_area)

    # Validate deconvolve_template parameter
    if deconvolve_template and axes_template not in ["YXS", "SYX"]:
        raise ValueError(f"deconvolve_template=True requires RGB template with axes 'YXS' or 'SYX', got '{axes_template}'")

    # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
    if image_type == "histo":
        print("\t\tRun color deconvolution on image", flush=True)
        # deconvolve HE - performed on resized image to save memory
        # TODO: Scale to max width instead of using a fixed scale factor before deconvolution (`scale_to_max_width`)
        nuclei_img, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes="YXS"),
                                    return_type="grayscale", convert=True)

        # bring back to original size
        nuclei_img = resize_image(nuclei_img, scale_factor=1/decon_scale_factor, axes="YX")

        # set nuclei_channel and nuclei_axis to None
        channel_name_for_registration = channel_axis = None
    else:
        # image_type is "IF" then
        # get index of nuclei channel
        channel_id_for_registration = channel_names.index(channel_name_for_registration)
        channel_axis = axes_image.find("C")

        if channel_axis == -1:
            raise ValueError(f"No channel indicator `C` found in image axes ({axes_image})")

        print(f"\t\tSelect image with nuclei from IF image (channel index: {channel_id_for_registration})", flush=True)
        # # select nuclei channel from IF image
        # if channel_name_for_registration is None:
        #     raise TypeError("Argument `nuclei_channel` should be an integer and not NoneType.")

        # select dapi channel for registration and convert to numpy array
        nuclei_img = np.take(image, channel_id_for_registration, channel_axis).compute()

    # Setup image registration objects - is important to load and scale the images.
    # The reason for this are limits in C++, not allowing to perform certain OpenCV functions on big images.

    # First: Setup the ImageRegistration object for the whole image (before deconvolution in histo images and multi-channel in IF)
    imreg_complete = ImageRegistration(
        image=image,
        template=template,
        axes_image=axes_image,
        axes_template=axes_template,
        deconvolve_template=deconvolve_template,
        decon_scale_factor=decon_scale_factor,
        verbose=True
        )
    # load and scale the whole image
    print('Load and scale image data containing all channels.')
    imreg_complete.load_and_scale_images()

    # Determine the axes_template for the selected registration object
    # If template was deconvolved, it's now grayscale (YX)
    axes_template_selected = "YX" if deconvolve_template else axes_template

    # setup ImageRegistration object with the nucleus image (either from deconvolution or just selected from IF image)
    imreg_selected = ImageRegistration(
        image=nuclei_img,
        template=imreg_complete.template,  # use the (potentially deconvolved) template
        axes_image="YX", # at this point the nuclei image was extracted and therefore the axes are always "YX"
        axes_template=axes_template_selected,
        max_width=4000,
        convert_to_grayscale=False,
        perspective_transform=False,
        min_good_matches=min_good_matches
    )

    # run all steps to extract features and get transformation matrix
    print('Load and scale image data containing only the channels required for registration.')
    imreg_selected.load_and_scale_images()

    print("\t\tExtract common features from image and template", flush=True)
    # perform registration to extract the common features ptsA and ptsB
    imreg_selected.extract_features(test_flipping=test_flipping)
    imreg_selected.calculate_transformation_matrix()

    if image_type == "histo":
        # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
        if imreg_complete.image_resized is None:
            imreg_selected.image = imreg_complete.image  # use original image
        else:
            imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image

        # perform registration
        imreg_selected.perform_registration()

        if save_registered_images:
            # save files
            identifier = f"{data.slide_id}__{data.sample_id}__{channel_names[0]}"
            imreg_selected.save(
                output_dir=output_dir,
                identifier = identifier,
                axes=axes_image,
                photometric='rgb',
                ome_metadata=ome_metadata
                )

            # # save metadata
            # data.metadata["method_params"]['images'][f'registered_{channel_names[0]}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
            # write_dict_to_json(data.metadata["method_params"], data.path / "experiment_modified.xenium")
            # #self._save_metadata_after_registration()

        data.images.add_image(
            image=imreg_selected.registered,
            channel_names=channel_names[0],
            axes=axes_image,
            pixel_size=pixel_size,
            ome_meta=ome_metadata,
            overwrite=True
            )

        del imreg_complete, imreg_selected, image, template, nuclei_img, eo, dab
    else:
        # image_type is IF
        # In case of IF images the channels are normally in the first axis and each channel is registered separately
        # Further, each channel is then saved separately as grayscale image.

        # iterate over channels
        for i, n in enumerate(channel_names):
            # skip the DAPI image
            if n == channel_name_for_registration:
                continue

            if imreg_complete.image_resized is None:
                # select one channel from non-resized original image
                imreg_selected.image = np.take(imreg_complete.image, i, channel_axis)
            else:
                # select one channel from resized original image
                imreg_selected.image_resized = np.take(imreg_complete.image_resized, i, channel_axis)

            # perform registration
            imreg_selected.perform_registration()

            if save_registered_images:
                # save files
                identifier = f"{data.slide_id}__{data.sample_id}__{n}"

                imreg_selected.save(
                    output_dir=output_dir,
                    identifier=identifier,
                    axes='YX',
                    photometric='minisblack',
                    ome_metadata=ome_metadata
                    )

                # # save metadata
                # data.metadata["method_params"]['images'][f'registered_{n}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
                # write_dict_to_json(data.metadata["method_params"], data.path / "experiment_modified.xenium")
                # #self._save_metadata_after_registration()
            # if add_registered_image:
            data.images.add_image(
                image=imreg_selected.registered,
                channel_names=n,
                axes="YX", # currently the images are added channel wise and therefore it is always "YX"
                pixel_size=pixel_size,
                ome_meta=ome_metadata,
                overwrite=True
                )

        # free RAM
        del imreg_complete, imreg_selected, image, template, nuclei_img
    gc.collect()



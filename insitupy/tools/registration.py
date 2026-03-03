import gc
import os
import time
import tracemalloc
import warnings
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
# from dask_image.imread import imread
from matplotlib.patches import ConnectionPatch

from insitupy import __version__
from insitupy._constants import CACHE, SHRT_MAX
from insitupy._core.data import InSituData
from insitupy._exceptions import NotEnoughFeatureMatchesError
from insitupy._textformat import textformat as tf
from insitupy.images.axes import ImageAxes, get_height_and_width
from insitupy.images.io import read_image, write_ome_tiff
from insitupy.images.utils import (clip_image_histogram, convert_to_8bit_func,
                                   deconvolve_he, fit_image_to_size_limit,
                                   otsu_thresholding, resize_image,
                                   scale_to_max_width)
from insitupy.utils.utils import convert_to_list, remove_last_line_from_csv


def _percentile_scale_for_saving(img: np.ndarray, upper_percentile: float = 95.0) -> np.ndarray:
    """Clip to upper percentile and scale intensities to [0, 1] for visualization export."""
    arr = np.asarray(img)
    if arr.size == 0:
        return arr

    def _scale_channel(channel: np.ndarray) -> np.ndarray:
        c = channel.astype(np.float32, copy=False)
        c_min = np.nanmin(c)
        c_max = np.nanpercentile(c, upper_percentile)
        if not np.isfinite(c_min) or not np.isfinite(c_max) or c_max <= c_min:
            return np.zeros_like(c, dtype=np.float32)
        c = np.clip(c, c_min, c_max)
        return (c - c_min) / (c_max - c_min)

    if arr.ndim == 2:
        return _scale_channel(arr)

    if arr.ndim == 3:
        scaled = np.empty(arr.shape, dtype=np.float32)
        for channel_idx in range(arr.shape[2]):
            scaled[..., channel_idx] = _scale_channel(arr[..., channel_idx])
        return scaled

    return arr


class ImageRegistration:
    '''
    Object to perform image registration.
    '''
    # Tree drawing characters
    _TSIGN = "\u251c"   # ├
    _LSIGN = "\u2514"   # └
    _VLINE = "\u2502"   # │
    _HLINE = "\u2500"   # ─
    _TICK  = "\u2714"   # ✔

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
                 mutual_nearest_neighbor: bool = False,
                 ratio_test: bool = True,
                 keepFraction: float = 0.2,
                 min_good_matches: int = 20,  # minimum number of good feature matches
                 maxFeatures: int = 500,
                 verbose: bool = True,
                 print_prefix: str = "  ",
                 ):

        # check verbose mode
        self.verboseprint = print if verbose else lambda *a, **k: None
        self.print_prefix = print_prefix

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
        self.mutual_nearest_neighbor = mutual_nearest_neighbor
        self.ratio_test = ratio_test
        self.keepFraction = keepFraction
        self.min_good_matches = min_good_matches
        self.maxFeatures = maxFeatures
        self.verbose = verbose

    def _log(self, message: str, is_last: bool = False, detail: bool = False, flush: bool = True):
        """Print a log message with tree-style prefix."""
        if detail:
            prefix = f"{self.print_prefix}{self._VLINE}     "
        elif is_last:
            prefix = f"{self.print_prefix}{self._LSIGN}{self._HLINE}{self._HLINE} "
        else:
            prefix = f"{self.print_prefix}{self._TSIGN}{self._HLINE}{self._HLINE} "
        self.verboseprint(f"{prefix}{message}", flush=flush)

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
        self._log(f"Color deconvolution ({name}, scale factor: {self.decon_scale_factor})")

        # deconvolve HE - performed on resized image to save memory
        nuclei_img, _, _ = deconvolve_he(
            img=resize_image(img, scale_factor=self.decon_scale_factor, axes=axes),
            return_type="grayscale",
            convert=True
        )

        # bring back to original size
        nuclei_img = resize_image(nuclei_img, scale_factor=1/self.decon_scale_factor, axes="YX")

        return nuclei_img

    def load_and_scale_images(self, scaling_log_label: str = "Scaling"):
        if not HAS_OPENCV:
            raise ImportError("OpenCV (cv2) is required for image registration. Install it with: pip install opencv-python")

        detail_prefix = f"{self.print_prefix}{self._VLINE}     "

        # load images into memory if they are dask arrays
        if isinstance(self.image, da.Array):
            self._log("Loading images into memory")
            self.image = self.image.compute()  # load into memory

        if isinstance(self.template, da.Array):
            self.template = self.template.compute()  # load into memory

        # Apply HE deconvolution if requested (before grayscale conversion)
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
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if len(self.template.shape) == 3:
                self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        if self.max_width is not None:
            self._log(scaling_log_label)
            self.image_scaled = scale_to_max_width(self.image,
                                                   axes=self.axes_image,
                                                   max_width=self.max_width,
                                                   use_square_area=True,
                                                   verbose=self.verbose,
                                                   print_spacer=f"{detail_prefix}Image:    "
                                                   )
            self.template_scaled = scale_to_max_width(self.template,
                                                      axes=self.axes_template,
                                                      max_width=self.max_width,
                                                      use_square_area=True,
                                                      verbose=self.verbose,
                                                      print_spacer=f"{detail_prefix}Template: "
                                                      )
        else:
            self.image_scaled = self.image
            self.template_scaled = self.template

        # convert and normalize images to 8bit for registration
        self.image_scaled = convert_to_8bit_func(self.image_scaled)
        self.template_scaled = convert_to_8bit_func(self.template_scaled)

        # calculate scale factors for x and y dimension for image and template
        self.x_sf_image = self.image_scaled.shape[1] / self.image.shape[1]
        self.y_sf_image = self.image_scaled.shape[0] / self.image.shape[0]
        self.x_sf_template = self.template_scaled.shape[1] / self.template.shape[1]
        self.y_sf_template = self.template_scaled.shape[0] / self.template.shape[0]

        # Store shapes and template dimensions for later use
        self.image_shape = self.image.shape
        self.template_shape = self.template.shape
        self.template_h, self.template_w = get_height_and_width(image=self.template, axes_config=self.axes_config_template)

        # resize image if necessary (warpAffine has a size limit for the image that is transformed)
        # get width and height of image
        h_image, w_image = get_height_and_width(image=self.image, axes_config=self.axes_config_image)
        if np.any([elem > SHRT_MAX for elem in (h_image, w_image)]):
            self._log(
                f"Warning: Dimensions {self.image_shape} exceed SHRT_MAX ({SHRT_MAX}). Resizing.")

            # fit image
            self.image_resized, self.resize_factor_image = fit_image_to_size_limit(
                self.image, size_limit=SHRT_MAX, return_scale_factor=True, axes=self.axes_image
                )
            self._log(f"Resized to {self.image_resized.shape} (factor: {self.resize_factor_image:.3f})", detail=True)
        else:
            self.image_resized = None
            self.resize_factor_image = 1

        # free memory - delete full-resolution image if resized version exists
        if self.image_resized is not None:
            del self.image  # free memory - keep only resized version

    def extract_features(
        self,
        test_flipping: bool = True,
        adjust_contrast_method: Optional[Literal["otsu", "clip"]] = "clip",
        debugging: bool = False,
        save_matched_vis: bool = True,
        force_failure: bool = False,
        ):
        '''
        Function to extract paired features from image and template.

        Args:
            test_flipping: Whether to test vertical flipping of the image during feature matching.
            adjust_contrast_method: Contrast adjustment applied before feature detection ("otsu", "clip", or None).
            debugging: If True, saves intermediate contrast-adjusted images to the cache directory.
            save_matched_vis: If True, stores a visualisation of matched keypoints in self.matchedVis.
            force_failure: If True, simulate a "not enough matches" failure even when sufficient matches
                are found. Useful for testing the failure-path QC output without needing a bad image pair.
            Mutual nearest-neighbor filtering is applied when ``self.mutual_nearest_neighbor`` is True.
        '''

        method_name = self.feature_detection_method.upper()
        contrast_info = f", contrast: {adjust_contrast_method}" if adjust_contrast_method else ""
        self._log(f"Feature extraction ({method_name}{contrast_info})")

        if test_flipping:
            # Test different flip transformations starting with no flip, then vertical, then horizontal.
            flip_axis_list = [None, 0] # before: [None, 0, 1]
        else:
            # do not test flipping of the axis
            flip_axis_list = [None]
        matches_list = [] # list to collect number of matches
        best_good_matches, best_kpsA, best_kpsB = [], None, None  # track best attempt across flips for failure diagnostics
        best_flip_axis = None  # flip axis that produced the best match count (None = no flip)
        for flip_axis in flip_axis_list:
            flipped = False
            if flip_axis is not None:
                # flip image
                flip_dir = 'vertical' if flip_axis == 0 else 'horizontal'
                self._log(f"Testing {flip_dir} flip", detail=True)
                self.image_scaled = np.flip(self.image_scaled, axis=flip_axis)
                flipped = True # set flipped flag to True

            # Get features
            # adjust contrast of both image and template
            if adjust_contrast_method is not None:
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
                # sift
                sift = cv2.SIFT_create()

                (kpsA, descsA) = sift.detectAndCompute(image_contrast_adj, None)
                (kpsB, descsB) = sift.detectAndCompute(template_contrast_adj, None)

            elif self.feature_detection_method == "surf":
                surf = cv2.xfeatures2d.SURF_create(400)

                (kpsA, descsA) = surf.detectAndCompute(image_contrast_adj, None)
                (kpsB, descsB) = surf.detectAndCompute(template_contrast_adj, None)

            else:
                self._log(f"Unknown method '{self.feature_detection_method}'. Aborted.", detail=True)
                return

            if self.flann:
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)   # or pass empty dictionary

                # runn Flann matcher
                fl = cv2.FlannBasedMatcher(index_params, search_params)
                matches = fl.knnMatch(descsA, descsB, k=2)

                # reverse matching for optional mutual nearest-neighbor filter (B -> A)
                if self.mutual_nearest_neighbor:
                    fl_rev = cv2.FlannBasedMatcher(index_params, search_params)
                    reverse_matches = fl_rev.knnMatch(descsB, descsA, k=1)

            else:
                # feature matching
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descsA, descsB, k=2)

                # reverse matching for optional mutual nearest-neighbor filter (B -> A)
                if self.mutual_nearest_neighbor:
                    reverse_matches = bf.knnMatch(descsB, descsA, k=1)

            if self.ratio_test:
                # store all the good matches as per Lowe's ratio test.
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7*n.distance:
                        good_matches.append(m)
            else:
                # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
                matches = sorted(matches, key=lambda x: x.distance)
                # keep only the top matches
                keep = int(len(matches) * self.keepFraction)
                good_matches = matches[:keep][:self.maxFeatures]

            if self.mutual_nearest_neighbor:
                reverse_best = {}
                for rev in reverse_matches:
                    if len(rev) == 0:
                        continue
                    m_rev = rev[0]
                    # queryIdx in reverse pass refers to descsB index, trainIdx refers to descsA index
                    reverse_best[m_rev.queryIdx] = m_rev.trainIdx

                n_before = len(good_matches)
                good_matches = [
                    m for m in good_matches
                    if reverse_best.get(m.trainIdx, None) == m.queryIdx
                ]
                self._log(f"Mutual NN filter: {len(good_matches)} / {n_before} kept", detail=True)

            # check if a sufficient number of good matches was found
            matches_list.append(len(good_matches))
            # track best result across all flip variants for failure diagnostics
            if len(good_matches) > len(best_good_matches):
                best_good_matches, best_kpsA, best_kpsB = good_matches, kpsA, kpsB
                best_flip_axis = flip_axis  # remember which orientation gave the best result
            if len(good_matches) >= self.min_good_matches and not force_failure:
                self._log(f"Good matches: {len(good_matches)} / {self.min_good_matches} required  {self._TICK}", detail=True)
                self.flip_axis = flip_axis
                break
            else:
                if force_failure and len(good_matches) >= self.min_good_matches:
                    self._log(f"Good matches: {len(good_matches)} / {self.min_good_matches} required  {self._TICK} (force_failure=True, simulating failure)", detail=True)
                else:
                    self._log(f"Good matches: {len(good_matches)} / {self.min_good_matches} required (insufficient, testing flip)", detail=True)
                if flipped:
                    # flip back
                    self.image_scaled = np.flip(self.image_scaled, axis=flip_axis)

        if not hasattr(self, "flip_axis"):
            # Re-apply the flip that produced the best result so that self.image_scaled and
            # best_kpsA are always in the same coordinate space. After the loop, image_scaled
            # has been flipped back to its original orientation, but best_kpsA coordinates
            # may belong to the flipped state — this would cause mismatches in all QC figures.
            if best_flip_axis is not None:
                self.image_scaled = np.flip(self.image_scaled, axis=best_flip_axis)
            # Save best available matchedVis and keypoints for failure diagnostics before raising
            if save_matched_vis and best_kpsA is not None:
                self.matchedVis = cv2.drawMatches(self.image_scaled, best_kpsA,
                                                  self.template_scaled, best_kpsB,
                                                  best_good_matches, None)
            self.kpsA = best_kpsA
            self.kpsB = best_kpsB
            self.good_matches = best_good_matches
            raise NotEnoughFeatureMatchesError(number=np.max(matches_list), threshold=self.min_good_matches)

        # check to see if we should visualize the matched keypoints
        if save_matched_vis:
            self.matchedVis = cv2.drawMatches(self.image_scaled, kpsA, self.template_scaled, kpsB,
                                            good_matches, None)

        # store keypoints and matches on self for detailed QC visualization
        self.kpsA = kpsA
        self.kpsB = kpsB
        self.good_matches = good_matches

        # Get keypoints
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

        transform_type = "perspective" if self.perspective_transform else "affine"
        self._log(f"Transformation matrix ({transform_type})")

        if self.perspective_transform:
            (self.T, mask) = cv2.findHomography(self.ptsA, self.ptsB, method=cv2.RANSAC)
        else:
            (self.T, mask) = cv2.estimateAffine2D(self.ptsA, self.ptsB)

        # Store inlier mask from RANSAC for QC (aligned with self.good_matches order).
        if mask is not None:
            self.inlier_mask = mask.ravel().astype(bool)
        else:
            self.inlier_mask = None

        if self.resize_factor_image != 1:
            self.ptsA *= self.resize_factor_image # scale images features in case it was originally larger than the warpAffine limits
            if self.perspective_transform:
                (self.T_resized, mask) = cv2.findHomography(self.ptsA, self.ptsB, method=cv2.RANSAC)
            else:
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
            flip_dir = 'vertically' if self.flip_axis == 0 else 'horizontally'
            self._log(f"Applying {flip_dir} flip", detail=True)
            self.image_to_register = np.flip(self.image_to_register, axis=self.flip_axis)

        # use the transformation matrix to register the images
        (h, w) = (self.template_h, self.template_w)
        # warping
        self._log("Registration")
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

    def _create_match_figure(
        self,
        topn: int,
        rank_matches_for_qc: bool = True,
        ranked_idx: Optional[List[int]] = None,
        figsize: tuple = (16, 8),
        create_detail_figure: bool = True,
        detail_window_um: float = 200.0,
    ) -> tuple[plt.Figure, Optional[plt.Figure]]:
        """
        Create a figure showing top-N matched keypoints between image and template.

        Matches are ranked by a composite quality score:
        1) RANSAC inlier status (if available),
        2) reprojection error in original space (if available),
        3) local patch similarity (ZNCC),
        4) descriptor distance.
        A one-to-one uniqueness constraint (queryIdx/trainIdx) is applied to reduce
        ambiguous duplicates. Labels indicate global quality rank (1 = best match).

        Args:
            topn: Maximum number of matches to display.
            figsize: Figure size as (width, height) tuple.

        Returns:
            Tuple of:
                - Main matplotlib Figure object (top-N overview).
                - Optional detail figure (5 rows x 2 columns) showing top-5 ranked matches
                  in 200 µm-wide windows around each feature center.
        """
        kpsA = self.kpsA
        kpsB = self.kpsB
        good_matches = self.good_matches
        n_total = len(good_matches)
        n_display = min(topn, n_total)

        if ranked_idx is None:
            if rank_matches_for_qc:
                ranked_idx = self._rank_match_indices_for_qc()
            else:
                ranked_idx = list(range(n_total))
        rank_by_match_idx = {match_idx: rank for rank, match_idx in enumerate(ranked_idx, start=1)}

        def _minmax_scale_for_display(img: np.ndarray) -> np.ndarray:
            """Scale image intensities to [0, 1] for visualization only."""
            arr = np.asarray(img)
            if arr.size == 0:
                return arr

            def _scale_channel(channel: np.ndarray) -> np.ndarray:
                c = channel.astype(np.float32, copy=False)
                c_min = np.nanmin(c)
                c_max = np.nanmax(c)
                if not np.isfinite(c_min) or not np.isfinite(c_max) or c_max <= c_min:
                    return np.zeros_like(c, dtype=np.float32)
                return (c - c_min) / (c_max - c_min)

            if arr.ndim == 2:
                return _scale_channel(arr)

            if arr.ndim == 3:
                scaled = np.empty(arr.shape, dtype=np.float32)
                for channel_idx in range(arr.shape[2]):
                    scaled[..., channel_idx] = _scale_channel(arr[..., channel_idx])
                return scaled

            return arr

        def _create_top5_detail_figure() -> plt.Figure:
            """Create 5x2 zoomed view for globally ranked matches 1-5."""
            n_rows = 5
            fig_detail, axes = plt.subplots(n_rows, 2, figsize=(12, 3.0 * n_rows))

            unit = str(getattr(self, "physical_size_unit", "µm")).strip().lower().replace("μ", "µ")
            is_um_unit = unit in {"µm", "um", "micrometer", "micrometre", "micron", "microns"}

            px_um_img = getattr(self, "pixel_size_image", None)
            px_um_tmpl = getattr(self, "pixel_size_template", None)
            if px_um_img is None:
                px_um_img = px_um_tmpl
            if px_um_tmpl is None:
                px_um_tmpl = px_um_img

            def _half_window_scaled(sf: float, pixel_size_um: Optional[float]) -> int:
                fallback_half_original = detail_window_um / 2.0
                if is_um_unit and pixel_size_um is not None and float(pixel_size_um) > 0:
                    half_window_original = (detail_window_um / float(pixel_size_um)) / 2.0
                else:
                    half_window_original = fallback_half_original
                return max(1, int(np.ceil(half_window_original * sf)))

            def _fixed_window_bounds(center: float, half_window: int, max_size: int) -> tuple[float, float]:
                """Return bounds with fixed width (2*half_window) whenever possible within image limits."""
                if max_size <= 1:
                    return 0.0, 0.0

                desired_width = float(2 * half_window)
                available_width = float(max_size - 1)
                width = min(desired_width, available_width)

                x0 = float(center) - width / 2.0
                x1 = float(center) + width / 2.0

                if x0 < 0.0:
                    x1 -= x0
                    x0 = 0.0
                if x1 > available_width:
                    x0 -= (x1 - available_width)
                    x1 = available_width
                x0 = max(0.0, x0)
                x1 = min(available_width, x1)
                return x0, x1

            top5_idx = ranked_idx[:min(5, n_total)]
            img_h, img_w = self.image_scaled.shape[:2]
            tmpl_h, tmpl_w = self.template_scaled.shape[:2]
            image_scaled_disp = _minmax_scale_for_display(self.image_scaled)
            template_scaled_disp = _minmax_scale_for_display(self.template_scaled)

            for row in range(n_rows):
                ax_img_row, ax_tmpl_row = axes[row, 0], axes[row, 1]
                ax_img_row.imshow(image_scaled_disp, cmap="gray" if self.image_scaled.ndim == 2 else None)
                ax_tmpl_row.imshow(template_scaled_disp, cmap="gray" if self.template_scaled.ndim == 2 else None)
                ax_img_row.axis("off")
                ax_tmpl_row.axis("off")

                if row >= len(top5_idx):
                    ax_img_row.set_title(f"Target - rank {row + 1} (not available)")
                    ax_tmpl_row.set_title(f"Template - rank {row + 1} (not available)")
                    continue

                match_i = top5_idx[row]
                m = good_matches[match_i]
                ptA = kpsA[m.queryIdx].pt
                ptB = kpsB[m.trainIdx].pt
                rank_label = rank_by_match_idx[match_i]

                half_w_img = _half_window_scaled(self.x_sf_image, px_um_img)
                half_h_img = _half_window_scaled(self.y_sf_image, px_um_img)
                half_w_tmpl = _half_window_scaled(self.x_sf_template, px_um_tmpl)
                half_h_tmpl = _half_window_scaled(self.y_sf_template, px_um_tmpl)

                x0_img, x1_img = _fixed_window_bounds(ptA[0], half_w_img, img_w)
                y0_img, y1_img = _fixed_window_bounds(ptA[1], half_h_img, img_h)

                x0_tmpl, x1_tmpl = _fixed_window_bounds(ptB[0], half_w_tmpl, tmpl_w)
                y0_tmpl, y1_tmpl = _fixed_window_bounds(ptB[1], half_h_tmpl, tmpl_h)

                ax_img_row.set_xlim(x0_img, x1_img)
                ax_img_row.set_ylim(y1_img, y0_img)
                ax_tmpl_row.set_xlim(x0_tmpl, x1_tmpl)
                ax_tmpl_row.set_ylim(y1_tmpl, y0_tmpl)

                ax_img_row.plot(ptA[0], ptA[1], marker="o", markersize=10,
                                markerfacecolor="none", markeredgecolor="yellow", markeredgewidth=1.1)
                ax_tmpl_row.plot(ptB[0], ptB[1], marker="o", markersize=10,
                                 markerfacecolor="none", markeredgecolor="yellow", markeredgewidth=1.1)

                ax_img_row.set_title(f"Target - rank {rank_label}")
                ax_tmpl_row.set_title(f"Template - rank {rank_label}")

            fig_detail.suptitle("Top-5 matches detailed view (200 µm window)", y=0.995)
            fig_detail.tight_layout(rect=[0, 0, 1, 0.985])
            return fig_detail

        if n_display == 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, "No matches available", ha="center", va="center")
            ax.axis("off")
            plt.tight_layout()
            detail_fig = _create_top5_detail_figure() if create_detail_figure else None
            return fig, detail_fig

        subset_idx = ranked_idx[:n_display]

        fig, (ax_img, ax_tmpl) = plt.subplots(1, 2, figsize=figsize)

        image_scaled_disp = _minmax_scale_for_display(self.image_scaled)
        template_scaled_disp = _minmax_scale_for_display(self.template_scaled)
        ax_img.imshow(image_scaled_disp, cmap="gray" if self.image_scaled.ndim == 2 else None)
        ax_tmpl.imshow(template_scaled_disp, cmap="gray" if self.template_scaled.ndim == 2 else None)
        has_transform = hasattr(self, "T") and self.T is not None and hasattr(self, "ptsA") and hasattr(self, "ptsB")
        if not rank_matches_for_qc:
            ranking_text = "original order"
        elif has_transform:
            ranking_text = "ranked by inlier+reprojection"
        else:
            ranking_text = "ranked by NCC+distance"
        ax_img.set_title(f"Target \u2013 top {n_display} of {n_total} matches ({ranking_text})")
        ax_tmpl.set_title("Template")
        ax_img.axis("off")
        ax_tmpl.axis("off")

        cmap = plt.colormaps.get_cmap("tab20")
        n_colors = 20
        for plot_i, match_i in enumerate(subset_idx):
            m = good_matches[match_i]
            ptA = kpsA[m.queryIdx].pt
            ptB = kpsB[m.trainIdx].pt
            color = cmap(plot_i % n_colors)
            rank_label = rank_by_match_idx[match_i]

            ax_img.plot(*ptA, "o", color=color, markersize=5,
                        markeredgewidth=0.5, markeredgecolor="white")
            ax_img.annotate(str(rank_label), ptA, color=color, fontsize=5,
                            xytext=(3, 3), textcoords="offset points")
            ax_tmpl.plot(*ptB, "o", color=color, markersize=5,
                         markeredgewidth=0.5, markeredgecolor="white")
            ax_tmpl.annotate(str(rank_label), ptB, color=color, fontsize=5,
                             xytext=(3, 3), textcoords="offset points")

            con = ConnectionPatch(
                xyA=ptA, xyB=ptB,
                coordsA="data", coordsB="data",
                axesA=ax_img, axesB=ax_tmpl,
                color=color, linewidth=0.5, alpha=0.6
            )
            con.set_clip_on(True)
            fig.add_artist(con)

        plt.tight_layout()
        detail_fig = _create_top5_detail_figure() if create_detail_figure else None
        return fig, detail_fig

    def _enforce_unique_match_pairs(self, ordered_match_indices: List[int]) -> List[int]:
        """Keep one-to-one unique queryIdx/trainIdx pairs while preserving input order."""
        selected = []
        used_query = set()
        used_train = set()
        for idx in ordered_match_indices:
            m = self.good_matches[idx]
            if m.queryIdx in used_query or m.trainIdx in used_train:
                continue
            selected.append(idx)
            used_query.add(m.queryIdx)
            used_train.add(m.trainIdx)
        return selected

    def _rank_match_indices_for_qc(self, patch_half_size: int = 16) -> List[int]:
        """Rank good match indices by robust QC quality score."""
        good_matches = self.good_matches
        n_total = len(good_matches)

        if n_total == 0:
            return []

        def _to_gray(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return img.astype(np.float32)
            if img.ndim == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return img.mean(axis=-1).astype(np.float32)

        def _extract_patch(img: np.ndarray, x: float, y: float, half: int) -> Optional[np.ndarray]:
            cx = int(round(x))
            cy = int(round(y))
            x0, x1 = cx - half, cx + half + 1
            y0, y1 = cy - half, cy + half + 1
            h, w = img.shape[:2]
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                return None
            return img[y0:y1, x0:x1]

        def _zncc(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
            if a is None or b is None:
                return -2.0
            aa = a.astype(np.float32)
            bb = b.astype(np.float32)
            aa = aa - np.mean(aa)
            bb = bb - np.mean(bb)
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom <= 1e-8:
                return -2.0
            return float(np.sum(aa * bb) / denom)

        has_transform = hasattr(self, "T") and self.T is not None and hasattr(self, "ptsA") and hasattr(self, "ptsB")
        has_inlier_mask = hasattr(self, "inlier_mask") and self.inlier_mask is not None and len(self.inlier_mask) == n_total

        reproj_error_by_idx = {}
        if has_transform and len(self.ptsA) == n_total and len(self.ptsB) == n_total:
            ptsA = self.ptsA.astype(np.float32)
            if self.perspective_transform:
                projected = cv2.perspectiveTransform(ptsA.reshape(-1, 1, 2), self.T).reshape(-1, 2)
            else:
                projected = cv2.transform(ptsA.reshape(-1, 1, 2), self.T).reshape(-1, 2)
            errors = np.linalg.norm(projected - self.ptsB, axis=1)
            reproj_error_by_idx = {idx: float(err) for idx, err in enumerate(errors)}

        img_gray = _to_gray(self.image_scaled)
        tmpl_gray = _to_gray(self.template_scaled)
        ncc_by_idx = {}
        for idx, match in enumerate(good_matches):
            ptA = self.kpsA[match.queryIdx].pt
            ptB = self.kpsB[match.trainIdx].pt
            patch_a = _extract_patch(img_gray, ptA[0], ptA[1], patch_half_size)
            patch_b = _extract_patch(tmpl_gray, ptB[0], ptB[1], patch_half_size)
            ncc_by_idx[idx] = _zncc(patch_a, patch_b)

        def _rank_key(idx: int) -> tuple:
            inlier_rank = 0 if (has_inlier_mask and bool(self.inlier_mask[idx])) else (1 if has_inlier_mask else 0)
            reproj_rank = reproj_error_by_idx.get(idx, np.inf)
            ncc_rank = -ncc_by_idx.get(idx, -2.0)
            dist_rank = float(good_matches[idx].distance)
            if has_transform:
                return (inlier_rank, reproj_rank, ncc_rank, dist_rank)
            return (ncc_rank, dist_rank)

        ranked = sorted(range(n_total), key=_rank_key)
        selected = self._enforce_unique_match_pairs(ranked)

        # Fallback if uniqueness filtering became too strict.
        if len(selected) == 0:
            selected = ranked

        return selected

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
        self._log(f"Saving")
        self._log(f"Image: {self.outfile}", detail=True)
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
        matchedVis: Optional[np.ndarray] = None,  # image showing the matched visualization
        rank_matches_for_qc: bool = True,
        ):
        """
        Save registration QC files (transformation matrix and feature matches visualization).

        Args:
            output_dir: Directory to save QC files (will create 'registration_qc' subdirectory).
            identifier: Identifier string for the output filenames.
            _T: Transformation matrix. If None, uses the appropriate matrix from self.
            matchedVis: Image showing matched features. If None, uses self.matchedVis.
            rank_matches_for_qc: If True, re-rank matches for QC using robust scoring.
                If False, keep the original matching order from feature extraction.
        """
        if _T is None:
            if hasattr(self, "resize_factor_image") and hasattr(self, "T_to_register"):
                if self.resize_factor_image == 1:
                    # if the image was not resized the transformation matrix to save is identical to the one used for registration
                    T_to_save = self.T_to_register
                else:
                    # if the image WAS resized the transformation matrix to save is not identical to the one used for registration
                    # instead the transformation matrix before resizing needs to be used
                    T_to_save = self.T
            else:
                T_to_save = None  # no transformation matrix available (e.g. failure before matrix calculation)
        else:
            T_to_save = _T

        if matchedVis is None and hasattr(self, "matchedVis"):
            matchedVis = self.matchedVis

        # save registration QC files
        output_dir = Path(output_dir)
        reg_dir = output_dir / "registration_qc"
        reg_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"QC:    {reg_dir}", detail=True)

        if T_to_save is not None:
            # save transformation matrix
            T_to_save = np.vstack([T_to_save, [0,0,1]])  # add last line of affine transformation matrix
            T_csv = reg_dir / f"{identifier}__transform.csv"
            np.savetxt(T_csv, T_to_save, delimiter=",")

            # remove last line break from csv since this gives error when importing to Xenium Explorer
            remove_last_line_from_csv(T_csv)

        if hasattr(self, "kpsA") and self.kpsA is not None and hasattr(self, "good_matches") and self.good_matches:
            if rank_matches_for_qc:
                ranked_idx_cache = self._rank_match_indices_for_qc()
            else:
                ranked_idx_cache = list(range(len(self.good_matches)))

            # __matches_overview: subset up to 100 rendered with matplotlib
            overview_fig, top5_fig = self._create_match_figure(
                topn=100,
                rank_matches_for_qc=rank_matches_for_qc,
                ranked_idx=ranked_idx_cache,
            )
            overview_fig.savefig(reg_dir / f"{identifier}__matches_overview.png", dpi=150, bbox_inches="tight")
            plt.close(overview_fig)

            # __matches_detail: detailed 5x2 zoom panel for globally best matches
            if top5_fig is not None:
                top5_fig.savefig(reg_dir / f"{identifier}__matches_detail.png", dpi=150, bbox_inches="tight")
                plt.close(top5_fig)
        elif matchedVis is not None:
            # fallback: save the cv2.drawMatches visualisation if no keypoints stored
            plt.imshow(matchedVis)
            plt.savefig(reg_dir / f"{identifier}__matches_overview.png", dpi=400)
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
    image_path: Optional[Union[str, os.PathLike, Path]] = None,
    channel_names: Optional[Union[str, List[str]]] = None,
    channel_name_for_registration: Optional[str] = None,  # name used for the nuclei image. Only required for IF images.
    template_image_name: str = "nuclei",
    save_registered_images: bool = True,
    output_dir: Union[str, os.PathLike, Path] = None,
    min_good_matches_per_area: int = 5, # unit: 1/mm²
    test_flipping: bool = True,
    decon_scale_factor: float = 0.2,
    deconvolve_template: bool = False,  # whether to apply HE deconvolution to the template
    physicalsize: str = 'µm',
    debug: bool = False,
    rank_matches_for_qc: bool = True,
    identifier: Optional[str] = None,
    force_failure_qc: bool = False,  # if True, simulate a failure even when enough matches are found (for QC testing)
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
        None
    """
    # Tree drawing characters
    _TSIGN = "\u251c"   # ├
    _LSIGN = "\u2514"   # └
    _VLINE = "\u2502"   # │
    _HLINE = "\u2500"   # ─
    _TICK  = "\u2714"   # ✔
    _SEP   = "\u2501"   # ━
    _prefix = "  "  # consistent print prefix

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
        # define output directory
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
    print(f"{_prefix}{_TSIGN}{_HLINE}{_HLINE} Loading images", flush=True)
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
    print(f"{_SEP * 80}", flush=True)
    print(f"Registration: {tf.Bold}{_header_id}{tf.ResetAll} {_HLINE}{_HLINE} {_header_channels} ({image_type})", flush=True)
    print(f"{_SEP * 80}", flush=True)

    # if image type is IF, the channel name for registration needs to be given
    if image_type == "IF" and channel_name_for_registration is None:
        raise ValueError("For IF images (`axes_image` in {'CYX', 'YXC'}), `channel_name_for_registration` must be provided.")

    # sometimes images are read with an empty time dimension in the first axis.
    # If this is the case, it is removed here.
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

    # # read images in InSituData object
    template = data.images[template_image_name][0] # usually the nuclei/DAPI image is the template. Use highest resolution of pyramid.
    template = _unwrap_first_level_image(template, template_image_name)
    print(f"{_prefix}{_VLINE}     Image:    {image.shape}", flush=True)
    print(f"{_prefix}{_VLINE}     Template: {template.shape}", flush=True)

    # extract OME metadata
    #ome_metadata_template = data.images.metadata[template_image_name]["OME"]

    # get pixel size from template image metadata
    pixel_size_template = data.images.metadata[template_image_name]["pixel_size"]

    # extract pixel size for x and y from OME metadata
    #pixelsizes = {key: ome_metadata_template['Image']['Pixels'][key] for key in ['PhysicalSizeX', 'PhysicalSizeY']}

    # generate OME metadata for saving
    ome_metadata = {
        'SignificantBits': 8,
        'PhysicalSizeXUnit': physicalsize,
        'PhysicalSizeYUnit': physicalsize,
        'PhysicalSizeX': pixel_size_template,
        'PhysicalSizeY': pixel_size_template
        }

    # determine minimum number of good matches that are necessary for the registration to be performed
    h, w = template.shape[:2]
    image_area = h * w * pixel_size_template**2 / 1000**2 # in mm²
    min_good_matches = int(min_good_matches_per_area * image_area)

    # Validate deconvolve_template parameter
    if deconvolve_template and axes_template not in ["YXS", "SYX"]:
        raise ValueError(f"deconvolve_template=True requires RGB template with axes 'YXS' or 'SYX', got '{axes_template}'")

    # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
    if image_type == "histo":
        print(f"{_prefix}{_TSIGN}{_HLINE}{_HLINE} Color deconvolution (scale factor: {decon_scale_factor})", flush=True)
        # deconvolve HE - performed on resized image to save memory
        # TODO: Scale to max width instead of using a fixed scale factor before deconvolution (`scale_to_max_width`)
        nuclei_img, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes="YXS"),
                                    return_type="grayscale", convert=True)

        # bring back to original size
        nuclei_img = resize_image(nuclei_img, scale_factor=1/decon_scale_factor, axes="YX")
        del eo, dab  # free memory - deconvolution intermediates no longer needed

        if debug:
            debug_id = identifier if identifier is not None else f"{data.slide_id}__{data.sample_id}"
            reg_qc_dir = Path(output_dir) / "registration_qc"
            reg_qc_dir.mkdir(parents=True, exist_ok=True)
            debug_decon_qc_path = reg_qc_dir / f"{debug_id}__deconvolved_target.png"
            nuclei_img_scaled = _percentile_scale_for_saving(nuclei_img, upper_percentile=95.0)
            plt.imsave(debug_decon_qc_path, nuclei_img_scaled, cmap="gray")
            print(f"{_prefix}{_VLINE}     Debug: saved deconvolved target -> {debug_decon_qc_path}", flush=True)

        # set nuclei_channel and nuclei_axis to None
        channel_name_for_registration = channel_axis = None
    else:
        # image_type is "IF" then
        # get index of nuclei channel
        channel_id_for_registration = channel_names.index(channel_name_for_registration)

        print(f"{_prefix}{_TSIGN}{_HLINE}{_HLINE} Selecting nuclei channel (index: {channel_id_for_registration})", flush=True)
        # # select nuclei channel from IF image
        # if channel_name_for_registration is None:
        #     raise TypeError("Argument `nuclei_channel` should be an integer and not NoneType.")

        # select dapi channel for registration and convert to numpy array
        nuclei_img = np.take(image, channel_id_for_registration, channel_axis)
        if hasattr(nuclei_img, "compute"):
            nuclei_img = nuclei_img.compute()

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
        verbose=True,
        print_prefix=_prefix
        )
    # load and scale the whole image
    imreg_complete.load_and_scale_images(scaling_log_label="Scaling (full image prep)")

    if debug and deconvolve_template:
        debug_id = identifier if identifier is not None else f"{data.slide_id}__{data.sample_id}"
        reg_qc_dir = Path(output_dir) / "registration_qc"
        reg_qc_dir.mkdir(parents=True, exist_ok=True)
        debug_decon_template_qc_path = reg_qc_dir / f"{debug_id}__deconvolved_template.png"
        template_scaled_for_save = _percentile_scale_for_saving(imreg_complete.template, upper_percentile=95.0)
        plt.imsave(debug_decon_template_qc_path, template_scaled_for_save, cmap="gray")
        print(f"{_prefix}{_VLINE}     Debug: saved deconvolved template -> {debug_decon_template_qc_path}", flush=True)

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
        min_good_matches=min_good_matches,
        print_prefix=_prefix
    )
    imreg_selected.pixel_size_image = pixel_size_image
    imreg_selected.pixel_size_template = pixel_size_template
    imreg_selected.physical_size_unit = physicalsize

    # run all steps to extract features and get transformation matrix
    imreg_selected.load_and_scale_images(scaling_log_label="Scaling (registration channel)")
    del imreg_selected.template  # free memory - h/w already stored
    del imreg_complete.image_scaled, imreg_complete.template_scaled  # free memory

    # perform registration to extract the common features ptsA and ptsB
    try:
        imreg_selected.extract_features(test_flipping=test_flipping, force_failure=force_failure_qc)
    except NotEnoughFeatureMatchesError:
        if output_dir is not None:
            if image_type == "IF":
                _qc_ref_name = channel_name_for_registration if channel_name_for_registration is not None else "registration_reference"
                _failed_identifier = f"{data.slide_id}__{data.sample_id}__{_qc_ref_name}__FAILED"
            else:
                _failed_identifier = f"{data.slide_id}__{data.sample_id}__{channel_names[0]}__FAILED"
            if hasattr(imreg_selected, "matchedVis"):
                print(f"{_prefix}{_TSIGN}{_HLINE}{_HLINE} Saving failure QC images", flush=True)
                imreg_selected.save_qc(
                    output_dir=output_dir,
                    identifier=_failed_identifier,
                    rank_matches_for_qc=rank_matches_for_qc,
                )
        raise
    imreg_selected.calculate_transformation_matrix()
    del nuclei_img  # free memory - features and transformation matrix already extracted

    if image_type == "histo":
        # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
        if imreg_complete.image_resized is None:
            imreg_selected.image = imreg_complete.image  # use original image
            del imreg_complete.image  # free memory - avoid holding two references
        else:
            imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image
            del imreg_complete.image_resized  # free memory - avoid holding two references

        # perform registration
        imreg_selected.perform_registration()

        if save_registered_images:
            # save files
            save_identifier = f"{data.slide_id}__{data.sample_id}__{channel_names[0]}"
            imreg_selected.save_registered_image(
                output_dir=output_dir,
                identifier=save_identifier,
                axes=axes_image,
                photometric='rgb',
                ome_metadata=ome_metadata
            )
            if debug:
                imreg_selected.save_qc(
                    output_dir=output_dir,
                    identifier=save_identifier,
                    rank_matches_for_qc=rank_matches_for_qc,
                )

            # # save metadata
            # data.metadata["method_params"]['images'][f'registered_{channel_names[0]}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
            # write_dict_to_json(data.metadata["method_params"], data.path / "experiment_modified.xenium")
            # #self._save_metadata_after_registration()

        data.images.add_image(
            image=imreg_selected.registered,
            channel_names=channel_names[0],
            axes=axes_image,
            pixel_size=pixel_size_template,
            ome_meta=ome_metadata,
            overwrite=True
            )

        del imreg_complete, imreg_selected, image, template
    else:
        # image_type is IF
        # In case of IF images the channels are normally in the first axis and each channel is registered separately
        # Further, each channel is then saved separately as grayscale image.

        # iterate over channels
        for i, n in enumerate(channel_names):
            # skip the DAPI image
            if n == channel_name_for_registration:
                continue

            print(f"{_prefix}{_TSIGN}{_HLINE}{_HLINE} Registering channel: {n}", flush=True)
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
                save_identifier = f"{data.slide_id}__{data.sample_id}__{n}"

                imreg_selected.save_registered_image(
                    output_dir=output_dir,
                    identifier=save_identifier,
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
                pixel_size=pixel_size_template,
                ome_meta=ome_metadata,
                overwrite=True
                )

        if save_registered_images and debug:
            _qc_ref_name = channel_name_for_registration if channel_name_for_registration is not None else "registration_reference"
            qc_identifier = f"{data.slide_id}__{data.sample_id}__{_qc_ref_name}"
            imreg_selected.save_qc(
                output_dir=output_dir,
                identifier=qc_identifier,
                rank_matches_for_qc=rank_matches_for_qc,
            )

        # free RAM
        del imreg_complete, imreg_selected, image, template

    _elapsed = time.time() - _t_start
    _, _peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _peak_mem_str = f"{_peak_mem / 1024**3:.2f} GB" if _peak_mem >= 1024**3 else f"{_peak_mem / 1024**2:.1f} MB"
    print(f"{_prefix}{_LSIGN}{_HLINE}{_HLINE} Done ({_elapsed:.1f} s, peak memory: {_peak_mem_str})", flush=True)
    gc.collect()



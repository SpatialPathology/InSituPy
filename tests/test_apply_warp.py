"""Tests for insitupy.images.warp.apply_warp - the SHRT_MAX source-resize guard (F1).

Covers the fix for a bug where the guard rescaled the translation column of the
warp matrix instead of the linear block, collapsing over-SHRT_MAX warp output
toward the origin by the resize scale factor.
"""

import cv2
import numpy as np

from insitupy.images import warp as warp_module
from insitupy.images.warp import apply_warp

# ── Pure-math pin of the corrected rescale formula ────────────────────────────

class TestShrtMaxRescaleFormula:
    def test_matches_diag_scale_for_2x3_matrix(self):
        sf = 0.4
        M = np.array([[1.5, 0.2, 10.0], [-0.3, 2.0, -5.0]], dtype=np.float64)
        expected = M @ np.diag([1 / sf, 1 / sf, 1.0])

        actual = M.copy()
        actual[:, :2] /= sf

        np.testing.assert_allclose(actual, expected)

    def test_matches_diag_scale_for_3x3_matrix(self):
        sf = 0.6
        M = np.array(
            [
                [1.2, -0.1, 3.0],
                [0.05, 0.9, -2.0],
                [0.001, 0.002, 1.0],
            ],
            dtype=np.float64,
        )
        expected = M @ np.diag([1 / sf, 1 / sf, 1.0])

        actual = M.copy()
        actual[:, :2] /= sf

        np.testing.assert_allclose(actual, expected)


# ── Regression guard: guard rescales the linear block, not the translation ───

class TestShrtMaxGuardRescalesLinearBlock:
    def test_translation_unchanged_linear_block_scaled(self, monkeypatch):
        captured = {}

        def _fake_warp_affine(img, M, dsize, **kwargs):
            captured["M"] = M.copy()
            return np.zeros((dsize[1], dsize[0]), dtype=img.dtype)

        monkeypatch.setattr(warp_module, "SHRT_MAX", 20)
        monkeypatch.setattr(cv2, "warpAffine", _fake_warp_affine)

        image = np.zeros((30, 30), dtype=np.uint8)
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float64)

        apply_warp(image, M, output_size=(30, 30), axes="YX")

        assert "M" in captured
        sf = (20 - 1) / 30  # matches fit_image_to_size_limit's formula
        # translation column (anti-pattern would scale this by `sf`)
        np.testing.assert_allclose(captured["M"][:, 2], [5.0, 3.0])
        # linear block scaled by 1/sf (identity here since M's linear block is identity)
        np.testing.assert_allclose(captured["M"][:, :2], np.eye(2) / sf, rtol=1e-6)

    def test_output_size_untouched_by_guard(self, monkeypatch):
        captured = {}

        def _fake_warp_affine(img, M, dsize, **kwargs):
            captured["dsize"] = dsize
            return np.zeros((dsize[1], dsize[0]), dtype=img.dtype)

        monkeypatch.setattr(warp_module, "SHRT_MAX", 20)
        monkeypatch.setattr(cv2, "warpAffine", _fake_warp_affine)

        image = np.zeros((30, 30), dtype=np.uint8)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        apply_warp(image, M, output_size=(45, 50), axes="YX")

        assert captured["dsize"] == (45, 50)


# ── End-to-end: content lands at the analytically-expected destination ───────

class TestShrtMaxGuardEndToEnd:
    def test_content_lands_at_expected_location_after_guard_fires(self, monkeypatch):
        # Inject a small SHRT_MAX so the guard fires on a small, fast image
        # instead of requiring a real >32767px array.
        monkeypatch.setattr(warp_module, "SHRT_MAX", 40)

        size = 50
        image = np.zeros((size, size), dtype=np.uint8)
        # small marker blob with a known center at (x=10, y=10)
        image[9:12, 9:12] = 255

        # scale-by-2 plus translation exercises both the linear block and the
        # translation column of the guard's matrix rescale.
        M = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 3.0]], dtype=np.float64)
        output_size = (60, 60)  # (width, height)

        assert size > warp_module.SHRT_MAX  # sanity: guard actually fires

        warped = apply_warp(image, M, output_size=output_size, axes="YX")

        # analytic expected destination of the marker's center: dst = A @ p + t
        expected_x = 2.0 * 10 + 5.0
        expected_y = 2.0 * 10 + 3.0

        peak_y, peak_x = np.unravel_index(np.argmax(warped), warped.shape)
        # tolerance accounts for interpolation loss from the mandatory downscale;
        # the pre-fix bug would collapse the peak toward the origin by ~sf
        # (roughly 5-6px away here), well outside this tolerance.
        assert abs(int(peak_x) - expected_x) <= 3
        assert abs(int(peak_y) - expected_y) <= 3

    def test_guard_inert_when_image_within_limit(self, monkeypatch):
        monkeypatch.setattr(warp_module, "SHRT_MAX", 1000)

        size = 40
        image = np.zeros((size, size), dtype=np.uint8)
        image[19:22, 19:22] = 255

        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float64)
        warped = apply_warp(image, M, output_size=(size, size), axes="YX")

        peak_y, peak_x = np.unravel_index(np.argmax(warped), warped.shape)
        assert abs(int(peak_x) - 25) <= 1
        assert abs(int(peak_y) - 23) <= 1

"""Tests for insitupy.utils._regression — LOWESS/confidence interval helpers."""

import numpy as np
import pytest

from insitupy.utils._regression import confidence_intervals, lowess, lowess_prediction


class TestConfidenceIntervals:
    def test_stores_lower_and_upper(self):
        # Must store both bounds exactly as provided
        lower = np.array([1.0, 2.0, 3.0])
        upper = np.array([4.0, 5.0, 6.0])
        ci = confidence_intervals(lower, upper)
        np.testing.assert_array_equal(ci.lower, lower)
        np.testing.assert_array_equal(ci.upper, upper)


class TestLowessPrediction:
    def test_stores_values_stderr_smooths(self):
        # All three attributes must be accessible after construction
        values = np.array([1.0, 2.0, 3.0])
        stderr = np.array([0.1, 0.1, 0.1])
        smooths = np.ones((3, 10))
        pred = lowess_prediction(values, stderr, smooths)
        np.testing.assert_array_equal(pred.values, values)
        np.testing.assert_array_equal(pred.stderr, stderr)
        np.testing.assert_array_equal(pred.smooths, smooths)

    def test_confidence_normal_approx_shape(self):
        # Normal-approximation CI must return lower/upper with same length as values
        n_points = 5
        values = np.linspace(0, 1, n_points)
        stderr = np.full(n_points, 0.1)
        rng = np.random.default_rng(0)
        smooths = rng.normal(0, 0.1, (n_points, 50))
        pred = lowess_prediction(values, stderr, smooths)
        ci = pred.confidence(alpha=0.05, percentile_method=False)
        assert isinstance(ci, confidence_intervals)
        assert ci.lower.shape == (n_points,)
        assert ci.upper.shape == (n_points,)

    def test_confidence_lower_le_upper(self):
        # Lower bound must not exceed upper bound at any point
        n_points = 5
        rng = np.random.default_rng(1)
        smooths = rng.normal(0, 0.5, (n_points, 100))
        stderr = np.nanstd(smooths, axis=1)
        pred = lowess_prediction(np.zeros(n_points), stderr, smooths)
        ci = pred.confidence(alpha=0.05)
        assert np.all(ci.lower <= ci.upper)

    def test_confidence_percentile_method(self):
        # Percentile method must also return lower <= upper
        n_points = 4
        rng = np.random.default_rng(2)
        smooths = rng.normal(1.0, 0.2, (n_points, 100))
        pred = lowess_prediction(np.ones(n_points), np.zeros(n_points), smooths)
        ci = pred.confidence(alpha=0.05, percentile_method=True)
        assert np.all(ci.lower <= ci.upper)


class TestLowessClass:
    def _make_linear_data(self, n=50, seed=42):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 10, n)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.5, n)
        return x, y

    def test_predict_raises_if_not_fitted(self):
        # Calling predict before fit must raise RuntimeError
        x, y = self._make_linear_data()
        model = lowess(x, y)
        with pytest.raises(RuntimeError):
            model.predict(np.linspace(0, 10, 10))

    def test_fit_sets_fitted_flag(self):
        # After fit(), self.fitted must be True
        x, y = self._make_linear_data()
        model = lowess(x, y)
        model.fit()
        assert model.fitted is True

    def test_predict_returns_lowess_prediction(self):
        # predict() must return a lowess_prediction instance
        x, y = self._make_linear_data()
        model = lowess(x, y)
        model.fit()
        result = model.predict(np.linspace(1, 9, 20), stderror=False)
        assert isinstance(result, lowess_prediction)

    def test_predict_output_length_matches_newdata(self):
        # Output values must have the same length as the prediction grid
        x, y = self._make_linear_data()
        model = lowess(x, y)
        model.fit()
        newdata = np.linspace(1, 9, 15)
        result = model.predict(newdata, stderror=False)
        assert len(result.values) == len(newdata)

    def test_predict_with_stderror_returns_smooths(self):
        # When stderror=True, the smooths array must have shape (n_pred, K)
        x, y = self._make_linear_data()
        model = lowess(x, y)
        model.fit()
        K = 10
        newdata = np.linspace(1, 9, 8)
        result = model.predict(newdata, stderror=True, K=K)
        assert result.smooths is not None
        assert result.smooths.shape[0] == len(newdata)
        assert result.smooths.shape[1] == K

    def test_lowess_tracks_linear_trend(self):
        # LOWESS on a clean linear signal must approximate the line closely
        x = np.linspace(0, 10, 60)
        y = 3.0 * x  # perfect linear signal
        model = lowess(x, y)
        model.fit()
        newdata = np.linspace(1, 9, 9)
        result = model.predict(newdata, stderror=False)
        expected = 3.0 * newdata
        # Allow 20% relative error for LOWESS approximation
        np.testing.assert_allclose(result.values, expected, rtol=0.2)

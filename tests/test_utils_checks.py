"""Tests for insitupy.utils._checks — validation helper functions."""

from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from insitupy.utils._checks import (
    _is_list_of_dask_arrays,
    _is_list_unique,
    check_adata,
    check_batch,
    check_hvg,
    check_integer_counts,
    check_raw,
    check_sanity,
    check_zip,
    is_integer_counts,
    is_valid_rgb_tuple,
)


def _make_adata(n_cells=5, n_genes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 30, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {"batch": rng.choice(["A", "B"], size=n_cells)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"Gene{j}" for j in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestCheckAdata:
    def test_valid_anndata_does_not_raise(self):
        # Valid AnnData must pass without exception; catches false rejection
        adata = _make_adata()
        check_adata(adata)  # should not raise

    def test_raises_type_error_for_dataframe(self):
        # A DataFrame is not AnnData and must be rejected
        with pytest.raises(TypeError):
            check_adata(pd.DataFrame({"a": [1, 2]}))

    def test_raises_type_error_for_dict(self):
        # A plain dict must also raise TypeError
        with pytest.raises(TypeError):
            check_adata({"X": np.ones((3, 3))})

    def test_raises_type_error_for_none(self):
        with pytest.raises(TypeError):
            check_adata(None)


class TestCheckBatch:
    def test_valid_batch_column_does_not_raise(self):
        # An existing column must pass silently
        obs = pd.DataFrame({"batch": ["A", "B"]}, index=["c0", "c1"])
        check_batch("batch", obs)

    def test_raises_for_missing_column(self):
        # A missing column must raise ValueError, catches silent downstream failure
        obs = pd.DataFrame({"other": [1, 2]}, index=["c0", "c1"])
        with pytest.raises(ValueError, match="not in obs"):
            check_batch("batch", obs)

    def test_verbose_mode_logs_without_raising(self):
        # verbose=True should log but not raise
        obs = pd.DataFrame({"batch": ["A", "B", "A"]}, index=["c0", "c1", "c2"])
        check_batch("batch", obs, verbose=True)  # should not raise


class TestCheckHvg:
    def test_valid_hvg_list_and_key_do_not_raise(self):
        # All genes in hvg must exist in var and hvg_key must be a column
        var = pd.DataFrame(
            {"highly_variable": [True, False, True]},
            index=["Gene0", "Gene1", "Gene2"],
        )
        check_hvg(["Gene0", "Gene2"], "highly_variable", var)

    def test_raises_type_error_for_non_list_hvg(self):
        # HVG must be a list; tuple or set must be rejected
        var = pd.DataFrame({"hv": [True]}, index=["Gene0"])
        with pytest.raises(TypeError, match="not a list"):
            check_hvg(("Gene0",), "hv", var)

    def test_raises_value_error_for_missing_gene(self):
        # Gene not in var must raise ValueError, catches silent wrong gene names
        var = pd.DataFrame({"hv": [True]}, index=["Gene0"])
        with pytest.raises(ValueError):
            check_hvg(["Gene0", "GeneXXX"], "hv", var)

    def test_raises_key_error_for_missing_hvg_key(self):
        # hvg_key not in var.columns must raise KeyError
        var = pd.DataFrame(index=["Gene0", "Gene1"])
        with pytest.raises(KeyError):
            check_hvg(["Gene0"], "highly_variable", var)


class TestCheckSanity:
    def test_valid_adata_passes_all_checks(self):
        # Well-formed AnnData with valid batch and HVG should pass
        adata = _make_adata()
        var = pd.DataFrame({"hv": [True, False, True, False]}, index=adata.var_names)
        adata.var = var
        check_sanity(adata, "batch", ["Gene0"], "hv")

    def test_skips_hvg_check_when_hvg_is_none(self):
        # When hvg is None, check_hvg must not run
        adata = _make_adata()
        check_sanity(adata, "batch", None, "hv")  # no KeyError for missing hv

    def test_propagates_check_adata_error(self):
        with pytest.raises(TypeError):
            check_sanity("not_adata", "batch", None, "hv")


class TestCheckIntegerCounts:
    def test_integer_matrix_passes(self):
        # Integer-valued floats must not raise
        X = np.array([[1.0, 2.0], [3.0, 0.0]])
        check_integer_counts(X)  # should not raise

    def test_float_matrix_raises(self):
        # Normalized floats must raise ValueError, catches accidental double-normalization
        X = np.array([[0.5, 1.2], [3.0, 0.0]])
        with pytest.raises(ValueError, match="raw counts"):
            check_integer_counts(X)

    def test_sparse_integer_matrix_passes(self):
        # Sparse matrices with integer values must also pass
        X = csr_matrix(np.array([[1.0, 0.0], [0.0, 3.0]]))
        check_integer_counts(X)

    def test_sparse_float_matrix_raises(self):
        X = csr_matrix(np.array([[0.3, 0.0], [0.0, 1.5]]))
        with pytest.raises(ValueError):
            check_integer_counts(X)


class TestIsIntegerCounts:
    def test_integer_values_return_true(self):
        # Pure integer matrix → True
        assert is_integer_counts(np.array([[1.0, 2.0], [3.0, 4.0]])) == True

    def test_float_values_return_false(self):
        # Normalized values → False
        assert is_integer_counts(np.array([[0.5, 1.2]])) == False

    def test_zeros_return_true(self):
        assert is_integer_counts(np.zeros((3, 3))) == True


class TestCheckRaw:
    def test_use_raw_false_returns_X(self):
        # Without use_raw, should return adata.X
        adata = _make_adata()
        X, var, var_names = check_raw(adata, use_raw=False)
        np.testing.assert_array_equal(X, adata.X)
        assert list(var_names) == list(adata.var_names)

    def test_use_raw_false_with_layer_returns_layer(self):
        # Should return the specified layer, not adata.X
        adata = _make_adata()
        layer_data = np.ones_like(adata.X) * 5
        adata.layers["counts"] = layer_data
        X, _, _ = check_raw(adata, use_raw=False, layer="counts")
        np.testing.assert_array_equal(X, layer_data)

    def test_use_raw_true_returns_raw_data(self):
        # With use_raw=True, must pull from adata.raw
        adata = _make_adata()
        adata.raw = adata  # simplest way to set raw
        X, var, var_names = check_raw(adata, use_raw=True)
        np.testing.assert_array_equal(X, adata.raw.X)


class TestCheckZip:
    def test_zip_suffix_returns_true(self):
        result = check_zip(Path("output.zip"))
        assert result is True

    def test_no_suffix_returns_false(self):
        result = check_zip(Path("output"))
        assert result is False

    def test_invalid_suffix_raises_value_error(self):
        # Extensions other than .zip or empty must be rejected
        with pytest.raises(ValueError):
            check_zip(Path("output.tar.gz"))


class TestIsListUnique:
    def test_unique_list_returns_true(self):
        assert _is_list_unique([1, 2, 3]) is True

    def test_duplicate_list_returns_false(self):
        # Duplicates must return False, catches silent data corruption
        assert _is_list_unique([1, 2, 2]) is False

    def test_empty_list_returns_true(self):
        # Empty list has no duplicates by definition
        assert _is_list_unique([]) is True

    def test_single_element_returns_true(self):
        assert _is_list_unique(["a"]) is True


class TestIsListOfDaskArrays:
    def test_list_of_dask_arrays_returns_true(self):
        arr = da.from_array(np.ones((3, 3)))
        assert _is_list_of_dask_arrays([arr, arr]) is True

    def test_empty_list_returns_true(self):
        # Empty list trivially satisfies "all elements are dask arrays"
        assert _is_list_of_dask_arrays([]) is True

    def test_non_list_returns_false(self):
        arr = da.from_array(np.ones((3,)))
        assert _is_list_of_dask_arrays(arr) is False

    def test_mixed_list_returns_false(self):
        arr = da.from_array(np.ones((3,)))
        assert _is_list_of_dask_arrays([arr, np.ones((3,))]) is False


class TestIsValidRgbTuple:
    def test_valid_tuple_returns_true(self):
        assert is_valid_rgb_tuple((255, 128, 0)) is True

    def test_valid_list_returns_true(self):
        assert is_valid_rgb_tuple([0, 0, 0]) is True

    def test_out_of_range_returns_false(self):
        # 256 > 255, invalid RGB
        assert is_valid_rgb_tuple((256, 0, 0)) is False

    def test_negative_value_returns_false(self):
        assert is_valid_rgb_tuple((-1, 0, 0)) is False

    def test_wrong_length_returns_false(self):
        # RGBA (4-element) is not a valid RGB triple
        assert is_valid_rgb_tuple((255, 0, 0, 1)) is False

    def test_non_sequence_returns_false(self):
        assert is_valid_rgb_tuple(42) is False

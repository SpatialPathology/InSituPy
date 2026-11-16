"""Tests for the tqdm output redirect in InSituExperiment.iterdata(progress=True)."""

import gc
import logging
import warnings

import pandas as pd
import pytest

from insitupy import _logging
from insitupy.experiment.data import InSituExperiment
from tests.test_experiment_table import _make_insitudata


def _make_experiment(n_samples=2):
    exp = InSituExperiment()
    for i in range(n_samples):
        exp._data.append(_make_insitudata(n_cells=5, n_genes=3, seed=i, cell_prefix=f"s{i}cell"))
    exp._metadata = pd.DataFrame({"uid": [f"sample_{i}" for i in range(n_samples)]})
    return exp


@pytest.fixture
def global_state():
    """Record the global state the redirect touches and check it is restored afterwards."""
    orig = warnings.showwarning
    handlers = list(logging.getLogger().handlers)
    yield orig, handlers
    # make sure a failing test cannot leak the patch into other tests
    warnings.showwarning = orig
    assert _logging._redirect_depth == 0


def _assert_restored(global_state):
    orig, handlers = global_state
    assert warnings.showwarning is orig
    assert list(logging.getLogger().handlers) == handlers
    assert _logging._redirect_depth == 0


@pytest.mark.parametrize("exhaust_first", ["g1", "g2"])
def test_interleaved_generators_restore_state(global_state, exhaust_first):
    exp = _make_experiment()
    g1 = exp.iterdata(progress=True)
    g2 = exp.iterdata(progress=True)
    next(g1)
    next(g2)

    first, second = (g1, g2) if exhaust_first == "g1" else (g2, g1)
    list(first)
    # the other loop is still running, so the redirect must still be active
    assert warnings.showwarning is not global_state[0]
    list(second)

    _assert_restored(global_state)


def test_abandoned_generator_restores_state(global_state):
    exp = _make_experiment()
    for _ in exp.iterdata(progress=True):
        break
    gc.collect()

    _assert_restored(global_state)


def test_warning_inside_loop_is_routed(global_state):
    exp = _make_experiment()
    for _ in exp.iterdata(progress=True):
        assert warnings.showwarning is not global_state[0]
        warnings.warn("inside the loop", UserWarning, stacklevel=2)

    _assert_restored(global_state)


def test_foreign_showwarning_is_not_overwritten(global_state):
    exp = _make_experiment()

    def foreign(*args, **kwargs):
        pass

    for _ in exp.iterdata(progress=True):
        warnings.showwarning = foreign

    assert warnings.showwarning is foreign

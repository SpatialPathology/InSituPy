from warnings import warn


def NoProjectLoadWarning():
    warn("Loading functions only work on a saved InSituPy project.", UserWarning)

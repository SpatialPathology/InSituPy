import pandas as pd

from insitupy._exceptions import InSituPyError

_MUTATORS_MSG = (
    "The metadata returned by `InSituExperiment.metadata` is read-only. "
    "Modify it through `add_metadata_column()`, `append_metadata()`, or "
    "`set_metadata_values()`. To get an editable copy, call `.copy()`."
)


class _GuardedMetadataFrame(pd.DataFrame):
    """Read-only view of an experiment's metadata.

    Reads behave exactly like a DataFrame; column/attribute assignment raises
    `InSituPyError` naming the validated mutators. Derived frames (slices, `.copy()`)
    are plain, editable `pandas.DataFrame` objects.
    """

    @property
    def _constructor(self):
        # Operations that build a new frame (slicing, arithmetic, reindex) return a
        # plain, editable DataFrame - the guard applies only to the object handed out
        # by the getter.
        return pd.DataFrame

    def __setitem__(self, key, value):
        raise InSituPyError(_MUTATORS_MSG)

    def __setattr__(self, name, value):
        # pandas sets underscore-prefixed internals (_mgr, _item_cache, _flags,
        # _attrs, ...) during construction and normal ops - allow those through.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            raise InSituPyError(_MUTATORS_MSG)

    def copy(self, deep=True):
        # Hand back a normal editable frame, never another guarded one.
        return pd.DataFrame(self).copy(deep=deep)

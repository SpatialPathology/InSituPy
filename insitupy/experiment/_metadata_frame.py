import pandas as pd

from insitupy._exceptions import InSituPyError

_MUTATORS_MSG = (
    "The metadata returned by `InSituExperiment.metadata` is read-only. "
    "Modify it through `add_metadata_column()`, `append_metadata()`, or "
    "`set_metadata_values()`. To get an editable copy, call `.copy()`."
)


class _ReadOnlyIndexer:
    """Read-only proxy around a pandas indexer (`.loc`, `.iloc`, `.at`, `.iat`).

    Reads (`__getitem__`) are forwarded to the wrapped indexer unchanged. Writes
    (`__setitem__`) raise `InSituPyError` instead of silently mutating the disposable
    copy behind the guarded frame. Does not subclass the private pandas indexer
    classes (`_LocIndexer`/`_iLocIndexer`/...), so it depends on no pandas internals.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getitem__(self, key):
        return self._inner[key]

    def __setitem__(self, key, value):
        raise InSituPyError(_MUTATORS_MSG)

    def __call__(self, *args, **kwargs):
        # Supports call-then-subscript usage such as `df.loc(axis=1)[...]`.
        return _ReadOnlyIndexer(self._inner(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _GuardedMetadataFrame(pd.DataFrame):
    """Read-only view of an experiment's metadata.

    Reads behave exactly like a DataFrame. The following mutation paths raise
    `InSituPyError` naming the validated mutators: column/attribute assignment,
    `.loc`/`.iloc`/`.at`/`.iat` writes, `inplace=True` methods, `pop`, `del`,
    `insert`, and `update`. Derived frames (slices, `.copy()`) are plain, editable
    `pandas.DataFrame` objects.

    Not caught: chained assignment (e.g. `df["c"][0] = v`) and writes through
    `.values`/`.to_numpy()`, since those operate on a plain `Series`/`ndarray` and
    cannot be guarded cheaply.
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

    @property
    def loc(self):
        return _ReadOnlyIndexer(super().loc)

    @property
    def iloc(self):
        return _ReadOnlyIndexer(super().iloc)

    @property
    def at(self):
        return _ReadOnlyIndexer(super().at)

    @property
    def iat(self):
        return _ReadOnlyIndexer(super().iat)

    def _update_inplace(self, *args, **kwargs):
        # Private pandas hook that every `inplace=True` method (fillna, dropna,
        # drop, sort_values, sort_index, replace, where, mask, drop_duplicates,
        # query, interpolate, clip, rename, ...) and `eval(..., inplace=True)`
        # funnel through to swap in the new block manager. It is marked
        # `@typing.final` in pandas, which is not enforced at runtime, so
        # overriding it here is the only cheap way to catch all of them at once.
        # The signature differs between pandas 1.5 and 2.x, hence *args/**kwargs.
        raise InSituPyError(_MUTATORS_MSG)

    def __delitem__(self, key):
        raise InSituPyError(_MUTATORS_MSG)

    def insert(self, *args, **kwargs):
        raise InSituPyError(_MUTATORS_MSG)

    def update(self, *args, **kwargs):
        raise InSituPyError(_MUTATORS_MSG)

    def copy(self, deep=True):
        # Hand back a normal editable frame, never another guarded one.
        return pd.DataFrame(self).copy(deep=deep)

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from insitupy.utils.utils import convert_to_list

if TYPE_CHECKING:
    from insitupy.experiment.data import InSituExperiment


@dataclass
class FilterSpec:
    """Structured filter specification."""
    key: str
    mask: list[bool]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise this filter spec to a JSON-compatible dict."""
        return {
            "mask": np.asarray(self.mask, dtype=bool).tolist(),
            "note": self.note,
        }

    @classmethod
    def from_entry(cls, key: str, entry: dict[str, Any]) -> "FilterSpec":
        """Deserialise a :class:`FilterSpec` from a stored dict entry.

        Args:
            key: The filter key.
            entry: Dict with at least a ``"mask"`` key (boolean list) and
                an optional ``"note"`` key.

        Returns:
            A new :class:`FilterSpec` instance.

        Raises:
            ValueError: If *entry* is not a dict or is missing ``"mask"``.
        """
        if not isinstance(entry, dict):
            raise ValueError(
                "Invalid filter entry: expected a dictionary with keys 'mask' and optional 'note'."
            )
        if "mask" not in entry:
            raise ValueError("Invalid filter entry: missing required key 'mask'.")

        mask = entry.get("mask", None)
        try:
            mask_list = list(np.asarray(mask, dtype=bool))
        except Exception as err:
            raise ValueError(
                f"Invalid filter entry: 'mask' must be convertible to a boolean list. Got {type(mask).__name__}."
            ) from err

        note = entry.get("note", None)
        if note is not None and not isinstance(note, str):
            note = str(note)

        return cls(key=key, mask=mask_list, note=note)


@dataclass
class CompositeFilterSpec:
    """Structured specification for a composite filter combining base filters.

    Composite filters are lazily evaluated: the mask is computed from the
    current state of the source base filters at call time, so they always
    reflect the latest changes to those filters.
    """
    key: str
    operation: Literal["and", "or"]
    source_keys: list[str]
    negated: list[str]
    note: str | None = None

    def evaluate(self, masks: dict[str, list[bool]]) -> list[bool]:
        """Compute the composite boolean mask from resolved source masks.

        Args:
            masks: Mapping of filter key to its boolean mask list.  Every key
                in :attr:`source_keys` must be present.

        Returns:
            A plain Python list of booleans.
        """
        arrays = []
        for k in self.source_keys:
            arr = np.asarray(masks[k], dtype=bool)
            if k in self.negated:
                arr = ~arr
            arrays.append(arr)
        if self.operation == "and":
            result = np.logical_and.reduce(arrays)
        else:
            result = np.logical_or.reduce(arrays)
        return result.tolist()

    def formula_str(self) -> str:
        """Return a human-readable formula string, e.g. ``'a AND NOT b'``."""
        parts = [f"NOT {k}" if k in self.negated else k for k in self.source_keys]
        op = " AND " if self.operation == "and" else " OR "
        return op.join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "operation": self.operation,
            "source_keys": list(self.source_keys),
            "negated": list(self.negated),
            "note": self.note,
        }

    @classmethod
    def from_entry(cls, key: str, entry: dict[str, Any]) -> "CompositeFilterSpec":
        """Deserialise from a stored dict entry.

        Args:
            key: The filter key.
            entry: Dict with keys ``"operation"``, ``"source_keys"``, and
                optional ``"negated"`` and ``"note"``.

        Returns:
            A new :class:`CompositeFilterSpec` instance.

        Raises:
            ValueError: If *entry* is not a dict, is missing required keys, or
                *operation* is not ``"and"`` or ``"or"``.
        """
        if not isinstance(entry, dict):
            raise ValueError(
                "Invalid composite filter entry: expected a dictionary."
            )
        for required in ("operation", "source_keys"):
            if required not in entry:
                raise ValueError(
                    f"Invalid composite filter entry: missing required key '{required}'."
                )

        operation = entry["operation"]
        if operation not in ("and", "or"):
            raise ValueError(
                f"Invalid composite filter operation: '{operation}'. Must be 'and' or 'or'."
            )

        source_keys = list(entry["source_keys"])
        negated = list(entry.get("negated", []))
        note = entry.get("note", None)
        if note is not None and not isinstance(note, str):
            note = str(note)

        return cls(
            key=key,
            operation=operation,
            source_keys=source_keys,
            negated=negated,
            note=note,
        )


class FilterManager:
    """Manager object exposing filter operations for an InSituExperiment."""

    def __init__(self, experiment: "InSituExperiment"):
        self._experiment = experiment

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _composites(self) -> dict:
        return self._experiment._composites

    def _resolve_mask(self, key: str) -> np.ndarray:
        """Return the evaluated boolean mask for *key* (base or composite)."""
        if key in self._experiment._filters:
            spec = FilterSpec.from_entry(key, self._experiment._filters[key])
            return np.asarray(spec.mask, dtype=bool)
        if key in self._composites:
            comp = CompositeFilterSpec.from_entry(key, self._composites[key])
            source_masks: dict[str, list[bool]] = {}
            for sk in comp.source_keys:
                if sk not in self._experiment._filters:
                    raise KeyError(
                        f"Composite filter '{key}' references missing base filter '{sk}'."
                    )
                src_spec = FilterSpec.from_entry(sk, self._experiment._filters[sk])
                source_masks[sk] = src_spec.mask
            return np.asarray(comp.evaluate(source_masks), dtype=bool)
        raise KeyError(
            f"Filter '{key}' not found. Available filters: {self.keys()}"
        )

    def _check_key_free(self, key: str, overwrite: bool) -> None:
        """Raise if *key* already exists and *overwrite* is ``False``."""
        if (key in self._experiment._filters or key in self._composites) and not overwrite:
            raise ValueError(
                f"Filter '{key}' already exists. Set overwrite=True to replace it."
            )

    def _composites_referencing(self, base_key: str) -> list[str]:
        """Return names of composite filters that reference *base_key*."""
        return [
            ck for ck, entry in self._composites.items()
            if base_key in CompositeFilterSpec.from_entry(ck, entry).source_keys
        ]

    # ── Key access ────────────────────────────────────────────────────────────

    def keys(self) -> list[str]:
        """Return the names of all stored filters (base and composite)."""
        return list(self._experiment._filters.keys()) + list(self._composites.keys())

    def base_keys(self) -> list[str]:
        """Return the names of all stored base filters."""
        return list(self._experiment._filters.keys())

    def composite_keys(self) -> list[str]:
        """Return the names of all stored composite filters."""
        return list(self._composites.keys())

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all filters.

        Returns:
            A :class:`pandas.DataFrame` with columns ``filter_key``, ``type``,
            ``formula``, ``n_selected``, ``n_total``, ``n_excluded``,
            ``selected_fraction``, and ``note``.  Base filters have
            ``formula=None``; composite filters carry a human-readable formula
            string (e.g. ``'a AND NOT b'``).
        """
        columns = [
            "filter_key", "type", "formula",
            "n_selected", "n_total", "n_excluded", "selected_fraction", "note",
        ]
        if not self._experiment._filters and not self._composites:
            return pd.DataFrame(columns=columns)

        rows = []
        n_total = len(self._experiment._metadata)

        for filter_key, entry in self._experiment._filters.items():
            spec = FilterSpec.from_entry(filter_key, entry)
            mask_arr = np.asarray(spec.mask, dtype=bool)
            if len(mask_arr) != n_total:
                warnings.warn(
                    f"Filter '{filter_key}' length ({len(mask_arr)}) does not match metadata length "
                    f"({n_total}). Skipping from filter overview.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            n_selected = int(mask_arr.sum())
            rows.append({
                "filter_key": filter_key,
                "type": "base",
                "formula": None,
                "n_selected": n_selected,
                "n_total": n_total,
                "n_excluded": int(n_total - n_selected),
                "selected_fraction": float(n_selected / n_total) if n_total > 0 else np.nan,
                "note": spec.note,
            })

        for filter_key, entry in self._composites.items():
            comp = CompositeFilterSpec.from_entry(filter_key, entry)
            try:
                mask_arr = self._resolve_mask(filter_key)
                if len(mask_arr) != n_total:
                    raise ValueError("Mask length mismatch.")
                n_selected = int(mask_arr.sum())
                rows.append({
                    "filter_key": filter_key,
                    "type": "composite",
                    "formula": comp.formula_str(),
                    "n_selected": n_selected,
                    "n_total": n_total,
                    "n_excluded": int(n_total - n_selected),
                    "selected_fraction": float(n_selected / n_total) if n_total > 0 else np.nan,
                    "note": comp.note,
                })
            except (KeyError, ValueError) as err:
                warnings.warn(
                    f"Composite filter '{filter_key}' could not be evaluated: {err}. "
                    "Skipping from summary.",
                    UserWarning,
                    stacklevel=2,
                )

        return pd.DataFrame(rows, columns=columns)

    # ── Masks ─────────────────────────────────────────────────────────────────

    def masks(self) -> dict[str, list[bool]]:
        """Return a dict mapping each filter key to its evaluated boolean mask list."""
        out: dict[str, list[bool]] = {}
        for key, entry in self._experiment._filters.items():
            spec = FilterSpec.from_entry(key, entry)
            out[key] = list(np.asarray(spec.mask, dtype=bool))
        for key in self._composites:
            try:
                out[key] = self._resolve_mask(key).tolist()
            except KeyError:
                pass
        return out

    # ── Create (base) ─────────────────────────────────────────────────────────

    def create(
        self,
        by: str,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        key: str | None = None,
        note: str | None = None,
        overwrite: bool = False,
        in_: list[str] | str | None = None,
        out: list[str] | str | None = None,
    ) -> str:
        """Create and store a new base filter based on a metadata column.

        Builds a boolean mask by matching values in the experiment metadata
        column *by* against the *include* or *exclude* list, then stores it
        under *key*.

        Args:
            by: Name of the metadata column to filter on.
            include: Value(s) to include.  Exactly one of *include* or
                *exclude* must be provided.
            exclude: Value(s) to exclude.  Exactly one of *include* or
                *exclude* must be provided.
            key: Name under which the filter is stored. Required.
            note: Optional free-text description of the filter.
            overwrite: If ``True``, replace an existing filter with the same
                key. Defaults to ``False``.
            in_: Deprecated alias for *include*.
            out: Deprecated alias for *exclude*.

        Returns:
            A confirmation string ``"Added filter '<key>' (n/N selected)."``.

        Raises:
            ValueError: If *key* is not provided, both or neither of
                *include*/*exclude* are given, or *by* is not a valid column.
            KeyError: If *by* is not found in the experiment metadata.
        """
        if key is None:
            raise ValueError("`key` must be provided to store the filter.")

        if in_ is not None:
            warnings.warn(
                "`in_` is deprecated and will be removed in a future version. "
                "Use `include` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if include is not None:
                raise ValueError("Specify only one of `include` or deprecated `in_`, not both.")
            include = in_

        if out is not None:
            warnings.warn(
                "`out` is deprecated and will be removed in a future version. "
                "Use `exclude` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if exclude is not None:
                raise ValueError("Specify only one of `exclude` or deprecated `out`, not both.")
            exclude = out

        if include is not None and exclude is not None:
            raise ValueError("Specify only one of `include` or `exclude`, not both.")
        if include is None and exclude is None:
            raise ValueError("Specify either `include` or `exclude`.")

        self._check_key_free(key, overwrite)

        metadata = self._experiment._metadata
        if by not in metadata.columns:
            raise KeyError(f"Column '{by}' not found in metadata.")

        if include is not None:
            values = convert_to_list(include)
            mask = metadata[by].isin(values)
        else:
            values = convert_to_list(exclude)
            mask = ~metadata[by].isin(values)

        mask_arr = mask.astype(bool).to_numpy()
        spec = FilterSpec(key=key, mask=mask_arr.tolist(), note=note)
        self._experiment._filters[key] = spec.to_dict()
        n_selected = int(mask_arr.sum())
        n_total = int(len(mask_arr))
        return f"Added filter '{key}' ({n_selected}/{n_total} selected)."

    # ── Combine (composite) ───────────────────────────────────────────────────

    def combine(
        self,
        keys: list[str],
        operation: Literal["and", "or"],
        key: str,
        negate: list[str] | str | None = None,
        note: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Create a composite filter by combining base filters with a boolean operation.

        Composite filters are lazily evaluated: the mask is recomputed from the
        current state of the source filters every time the composite is applied,
        so it always reflects the latest changes.

        Args:
            keys: Base filter keys to combine.  All must exist and be base
                (not composite) — composite-of-composites is not supported.
            operation: ``"and"`` or ``"or"`` — how to combine the masks.
            key: Name under which the composite filter is stored.
            negate: Filter key(s) within *keys* whose masks are inverted before
                combining (logical NOT).  Defaults to ``None`` (no negation).
            note: Optional free-text description.
            overwrite: If ``True``, replace an existing filter with the same
                key. Defaults to ``False``.

        Returns:
            A confirmation string.

        Raises:
            ValueError: If *keys* is empty, *operation* is invalid, any key in
                *keys* is a composite filter, or a key in *negate* is not in *keys*.
            KeyError: If any key in *keys* does not exist.
        """
        if not keys:
            raise ValueError("`keys` must not be empty.")
        if operation not in ("and", "or"):
            raise ValueError(f"`operation` must be 'and' or 'or', got '{operation}'.")

        negated = list(convert_to_list(negate)) if negate is not None else []

        for k in keys:
            if k in self._composites:
                raise ValueError(
                    f"'{k}' is a composite filter. Only base filters may be referenced in v1."
                )
            if k not in self._experiment._filters:
                raise KeyError(
                    f"Base filter '{k}' not found. Available base filters: {self.base_keys()}"
                )

        for k in negated:
            if k not in keys:
                raise ValueError(
                    f"Negated key '{k}' is not in `keys`. Only keys listed in `keys` can be negated."
                )

        self._check_key_free(key, overwrite)
        if key in self._experiment._filters:
            del self._experiment._filters[key]

        comp = CompositeFilterSpec(
            key=key,
            operation=operation,
            source_keys=list(keys),
            negated=negated,
            note=note,
        )
        self._composites[key] = comp.to_dict()

        mask = self._resolve_mask(key)
        n_selected = int(mask.sum())
        n_total = len(self._experiment._metadata)
        return f"Added composite filter '{key}' ({n_selected}/{n_total} selected)."

    # ── Invert ────────────────────────────────────────────────────────────────

    def invert(
        self,
        key: str,
        new_key: str,
        note: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Create a new base filter that is the logical NOT of an existing base filter.

        The result is a materialized snapshot — it does not track future
        changes to *key*.  For a lazily evaluated NOT, use
        ``combine([key], "and", ..., negate=[key])`` instead.

        Args:
            key: Existing base filter key to invert.
            new_key: Name for the resulting filter.
            note: Optional description.  Defaults to ``"NOT <key>"``.
            overwrite: If ``True``, replace an existing filter with the same
                *new_key*.

        Returns:
            A confirmation string.

        Raises:
            KeyError: If *key* is not a stored base filter.
            ValueError: If *key* is a composite filter, or *new_key* already
                exists and *overwrite* is ``False``.
        """
        if key not in self._experiment._filters:
            if key in self._composites:
                raise ValueError(
                    f"'{key}' is a composite filter. "
                    f"Use combine(['{key}'], 'and', ..., negate=['{key}']) for a lazy NOT."
                )
            raise KeyError(
                f"Base filter '{key}' not found. Available base filters: {self.base_keys()}"
            )
        self._check_key_free(new_key, overwrite)

        spec = FilterSpec.from_entry(key, self._experiment._filters[key])
        inverted = (~np.asarray(spec.mask, dtype=bool)).tolist()
        new_spec = FilterSpec(
            key=new_key,
            mask=inverted,
            note=note if note is not None else f"NOT {key}",
        )
        self._experiment._filters[new_key] = new_spec.to_dict()
        n_selected = int(sum(inverted))
        n_total = len(inverted)
        return f"Added filter '{new_key}' ({n_selected}/{n_total} selected)."

    # ── Materialize ───────────────────────────────────────────────────────────

    def materialize(
        self,
        key: str,
        new_key: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Convert a composite filter to a base filter by materializing its mask.

        The resulting base filter is a snapshot of the composite's current
        evaluated mask and no longer tracks changes to source filters.

        Args:
            key: Composite filter key to materialize.
            new_key: Name for the resulting base filter.  If ``None``, the
                composite is replaced in-place under the same key.
            overwrite: If ``True``, replace an existing filter with the same
                *new_key*.  Defaults to ``False``.

        Returns:
            A confirmation string.

        Raises:
            KeyError: If *key* is not a stored composite filter.
            ValueError: If *key* is already a base filter.
        """
        if key not in self._composites:
            if key in self._experiment._filters:
                raise ValueError(f"'{key}' is already a base filter.")
            raise KeyError(
                f"Composite filter '{key}' not found. "
                f"Available composite filters: {self.composite_keys()}"
            )

        target_key = new_key if new_key is not None else key
        if new_key is not None:
            self._check_key_free(new_key, overwrite)

        comp = CompositeFilterSpec.from_entry(key, self._composites[key])
        mask = self._resolve_mask(key)
        note = comp.note if comp.note is not None else f"Materialized: {comp.formula_str()}"
        spec = FilterSpec(key=target_key, mask=mask.tolist(), note=note)

        if new_key is None:
            del self._composites[key]
        self._experiment._filters[target_key] = spec.to_dict()

        n_selected = int(mask.sum())
        n_total = len(mask)
        return f"Added filter '{target_key}' ({n_selected}/{n_total} selected)."

    # ── Apply / view ──────────────────────────────────────────────────────────

    def apply(self, key: str) -> "InSituExperiment":
        """Return a new :class:`InSituExperiment` containing only the filtered samples.

        Creates a permanent (non-view) subset of the experiment using the
        boolean mask for *key*.  For composite filters the mask is evaluated
        from the current state of the source filters at call time.

        Args:
            key: Name of the filter to apply (base or composite).

        Returns:
            A new :class:`InSituExperiment` with only the selected samples.

        Raises:
            KeyError: If *key* is not a stored filter.
            ValueError: If the mask length does not match the metadata length.
        """
        if key not in self._experiment._filters and key not in self._composites:
            raise KeyError(
                f"Filter '{key}' not found. Available filters: {self.keys()}"
            )
        mask = self._resolve_mask(key)
        if len(mask) != len(self._experiment._metadata):
            raise ValueError(
                f"Filter '{key}' length ({len(mask)}) does not match metadata length "
                f"({len(self._experiment._metadata)})."
            )
        return self._experiment._subset(
            pd.Series(mask, index=self._experiment._metadata.index),
            as_view=False,
        )

    def view(self, key: str) -> "InSituExperiment":
        """Return a lazy view of the experiment containing only the filtered samples.

        Unlike :meth:`apply`, this returns a *view* that remembers which filter
        was used, without creating a full copy of the data.  For composite
        filters the mask is evaluated from the current state of the source
        filters at call time.

        Args:
            key: Name of the filter to apply as a view (base or composite).

        Returns:
            A view :class:`InSituExperiment` with only the selected samples.

        Raises:
            KeyError: If *key* is not a stored filter.
            ValueError: If the mask length does not match the metadata length.
        """
        if key not in self._experiment._filters and key not in self._composites:
            raise KeyError(
                f"Filter '{key}' not found. Available filters: {self.keys()}"
            )
        mask = self._resolve_mask(key)
        if len(mask) != len(self._experiment._metadata):
            raise ValueError(
                f"Filter '{key}' length ({len(mask)}) does not match metadata length "
                f"({len(self._experiment._metadata)})."
            )
        return self._experiment._subset(
            pd.Series(mask, index=self._experiment._metadata.index),
            as_view=True,
            added_filter=key,
        )

    # ── Remove / rename / clear ───────────────────────────────────────────────

    def remove(self, key: str):
        """Delete a stored filter by name.

        Args:
            key: Name of the filter to remove (base or composite).

        Raises:
            KeyError: If *key* is not a stored filter.
            ValueError: If *key* is a base filter referenced by one or more
                composite filters (remove those composites first).
        """
        if key in self._experiment._filters:
            refs = self._composites_referencing(key)
            if refs:
                raise ValueError(
                    f"Cannot remove base filter '{key}': referenced by composite filter(s) "
                    f"{refs}. Remove or update those composites first."
                )
            del self._experiment._filters[key]
        elif key in self._composites:
            del self._composites[key]
        else:
            raise KeyError(
                f"Filter '{key}' not found. Available filters: {self.keys()}"
            )

    def rename(self, old_key: str, new_key: str, overwrite: bool = False):
        """Rename a stored filter.

        Args:
            old_key: Current name of the filter.
            new_key: New name for the filter.
            overwrite: If ``True``, replace an existing filter with name
                *new_key*. Defaults to ``False``.

        Raises:
            KeyError: If *old_key* is not a stored filter.
            ValueError: If *new_key* already exists and *overwrite* is
                ``False``, or if *old_key* is a base filter referenced by
                composite filters.
        """
        if old_key not in self._experiment._filters and old_key not in self._composites:
            raise KeyError(
                f"Filter '{old_key}' not found. Available filters: {self.keys()}"
            )
        if old_key == new_key:
            return
        self._check_key_free(new_key, overwrite)

        if old_key in self._experiment._filters:
            refs = self._composites_referencing(old_key)
            if refs:
                raise ValueError(
                    f"Cannot rename base filter '{old_key}': referenced by composite filter(s) "
                    f"{refs}. Remove or update those composites first."
                )
            spec = FilterSpec.from_entry(old_key, self._experiment._filters[old_key])
            spec.key = new_key
            self._experiment._filters[new_key] = spec.to_dict()
            del self._experiment._filters[old_key]
        else:
            comp = CompositeFilterSpec.from_entry(old_key, self._composites[old_key])
            comp.key = new_key
            self._composites[new_key] = comp.to_dict()
            del self._composites[old_key]

    def clear(self):
        """Remove all stored filters (base and composite)."""
        self._experiment._filters = {}
        self._experiment._composites = {}

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        df = self.summary()
        base = df[df["type"] == "base"].drop(columns=["type", "formula"])
        composite = df[df["type"] == "composite"].drop(columns=["type"])

        _tab = "    "  # one indent level for table rows under each subsection header

        parts = []
        if not base.empty:
            table_str = base.to_string(index=False).replace("\n", f"\n{_tab}")
            parts.append(f"Base filters ({len(base)}):\n{_tab}{table_str}")
        else:
            parts.append("Base filters: none")

        if not composite.empty:
            table_str = composite.to_string(index=False).replace("\n", f"\n{_tab}")
            parts.append(f"Composite filters ({len(composite)}):\n{_tab}{table_str}")
        else:
            parts.append("Composite filters: none")

        return "\n\n".join(parts)

    def _repr_html_(self):
        """HTML representation for notebook display."""
        df = self.summary()
        base = df[df["type"] == "base"].drop(columns=["type", "formula"])
        composite = df[df["type"] == "composite"].drop(columns=["type"])

        def _section(title: str, sub_df: pd.DataFrame) -> str:
            body = (
                "<div style='overflow-x:auto; max-width:100%; padding-left:1em'>"
                + sub_df.to_html(index=False)
                + "</div>"
            ) if not sub_df.empty else "<p style='padding-left:1em'><i>none</i></p>"
            return f"<b>{title}</b>{body}"

        return (
            _section(f"Base filters ({len(base)})", base)
            + _section(f"Composite filters ({len(composite)})", composite)
        )

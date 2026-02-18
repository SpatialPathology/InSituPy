import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from insitupy.utils.utils import convert_to_list

if TYPE_CHECKING:
    from insitupy.experiment.data import InSituExperiment


@dataclass
class FilterSpec:
    """Structured filter specification."""
    key: str
    mask: List[bool]
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask": list(np.asarray(self.mask, dtype=bool)),
            "note": self.note,
        }

    @classmethod
    def from_entry(cls, key: str, entry: Dict[str, Any]) -> "FilterSpec":
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


class FilterManager:
    """Manager object exposing filter operations for an InSituExperiment."""
    def __init__(self, experiment: "InSituExperiment"):
        self._experiment = experiment

    def keys(self) -> List[str]:
        return list(self._experiment._filters.keys())

    def summary(self) -> pd.DataFrame:
        if not self._experiment._filters:
            return pd.DataFrame(columns=["filter_key", "n_selected", "n_total", "n_excluded", "selected_fraction", "note"])

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
                    stacklevel=2
                )
                continue

            n_selected = int(mask_arr.sum())
            rows.append({
                "filter_key": filter_key,
                "n_selected": n_selected,
                "n_total": n_total,
                "n_excluded": int(n_total - n_selected),
                "selected_fraction": float(n_selected / n_total) if n_total > 0 else np.nan,
                "note": spec.note,
            })

        return pd.DataFrame(rows)

    def masks(self) -> Dict[str, List[bool]]:
        out = {}
        for key, entry in self._experiment._filters.items():
            spec = FilterSpec.from_entry(key, entry)
            out[key] = list(np.asarray(spec.mask, dtype=bool))
        return out

    def create(
        self,
        by: str,
        include: Optional[Union[List[str], str]] = None,
        exclude: Optional[Union[List[str], str]] = None,
        key: Optional[str] = None,
        note: Optional[str] = None,
        overwrite: bool = False,
        in_: Optional[Union[List[str], str]] = None,
        out: Optional[Union[List[str], str]] = None,
    ) -> str:
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

        metadata = self._experiment._metadata
        if by not in metadata.columns:
            raise KeyError(f"Column '{by}' not found in metadata.")
        if key in self._experiment._filters and not overwrite:
            raise ValueError(
                f"Filter '{key}' already exists. Set overwrite=True to replace it."
            )

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

    def apply(self, key: str) -> "InSituExperiment":
        if key not in self._experiment._filters:
            raise KeyError(
                f"Filter '{key}' not found. Available filters: {self.keys()}"
            )
        spec = FilterSpec.from_entry(key, self._experiment._filters[key])
        mask = np.asarray(spec.mask, dtype=bool)
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
        if key not in self._experiment._filters:
            raise KeyError(
                f"Filter '{key}' not found. Available filters: {self.keys()}"
            )
        spec = FilterSpec.from_entry(key, self._experiment._filters[key])
        mask = np.asarray(spec.mask, dtype=bool)
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

    def remove(self, key: str):
        if key not in self._experiment._filters:
            raise KeyError(
                f"Filter '{key}' not found. Available filters: {self.keys()}"
            )
        del self._experiment._filters[key]

    def clear(self):
        self._experiment._filters = {}

    def rename(self, old_key: str, new_key: str, overwrite: bool = False):
        if old_key not in self._experiment._filters:
            raise KeyError(
                f"Filter '{old_key}' not found. Available filters: {self.keys()}"
            )
        if old_key == new_key:
            return
        if new_key in self._experiment._filters and not overwrite:
            raise ValueError(
                f"Filter '{new_key}' already exists. Set overwrite=True to replace it."
            )

        spec = FilterSpec.from_entry(old_key, self._experiment._filters[old_key])
        spec.key = new_key
        self._experiment._filters[new_key] = spec.to_dict()
        del self._experiment._filters[old_key]

    def __repr__(self):
        return self.summary().__repr__()

    def _repr_html_(self):
        """HTML representation for notebook display with horizontal scrolling."""
        df = self.summary()
        table_html = df.to_html(index=False)
        return (
            "<div style='overflow-x:auto; max-width:100%;'>"
            + table_html +
            "</div>"
        )

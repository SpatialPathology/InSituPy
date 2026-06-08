import contextlib
import gc
import json
import logging
import os
import shutil
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from tqdm import tqdm

from insitupy._constants import (
    DEFAULT_CATEGORICAL_CMAP,
    ISPY_METADATA_FILE,
    LOAD_FUNCS,
    MODALITIES,
    MODALITIES_ABBR,
    SAMPLE_STR,
    with_insitupy_style,
)
from insitupy._core.data import InSituData
from insitupy._exceptions import ModalityNotFoundError
from insitupy._io.files import check_overwrite_and_remove_if_true, read_json, write_dict_to_json
from insitupy._logging import WarningCollector, collect_warnings
from insitupy._textformat import textformat as tf
from insitupy.containers._utils import _get_cell_layer
from insitupy.experiment.filters import CompositeFilterSpec, FilterManager, FilterSpec
from insitupy.io.data import read_xenium
from insitupy.palettes import map_to_colors
from insitupy.utils._adata import _select_anndata_elements
from insitupy.utils.utils import (
    _crop_transcripts,
    convert_to_list,
    get_nrows_maxcols,
    remove_empty_subplots,
)

logger = logging.getLogger(__name__)

# Feature flag for SpatialData mode
# Set to True to enable spatialdata mode functionality
# Currently disabled while the feature is under development
_SPATIALDATA_MODE_ENABLED = False
_FILTERS_SCHEMA_VERSION = 2
_SUPPORTED_FILTER_VERSIONS = {1, 2}
_METADATA_SCHEMA_VERSION = 1
_METADATA_SCHEMA_FILENAME = "metadata.schema.json"
_METADATA_PARQUET_FILENAME = "metadata.parquet"

# Sentinel value to detect when 'by' is not explicitly provided
_UNSET = object()


class TableAccessor:
    """Dict-like accessor for per-cells-layer concatenated tables.

    Returned by :attr:`InSituExperiment.table`. Use bracket notation to load
    the AnnData for a specific layer::

        exp.table["main"]    # AnnData for the "main" segmentation
        exp.table["proseg"]  # AnnData for the "proseg" segmentation
        exp.table[None]      # auto-select when only one table exists
        exp.table.keys()     # list available layer names

    .. note::
        This feature is experimental and may change in future versions.
    """

    def __init__(self, experiment: "InSituExperiment") -> None:
        self._exp = experiment

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return the layer names for which a table has been built."""
        if self._exp.path is None:
            return []
        tables_dir = Path(self._exp.path) / "tables"
        if not tables_dir.exists():
            return []
        return sorted(
            p.stem for p in tables_dir.iterdir()
            if p.suffix == ".zarr" and p.is_dir() and p.name != "concat.zarr"
        )

    def __getitem__(self, cells_layer: str | None) -> AnnData | None:
        return self._load(cells_layer)

    def __repr__(self) -> str:
        keys = self.keys()
        if not keys:
            return "TableAccessor(no tables built yet)"
        return f"TableAccessor(layers={keys})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, cells_layer: str | None) -> AnnData | None:
        try:
            table_path = self._exp._get_table_path(cells_layer)
        except ValueError as exc:
            warnings.warn(str(exc), UserWarning, stacklevel=3)
            return None
        if table_path is None or not table_path.exists():
            warnings.warn(
                "No concatenated table found. Call `build_table()` to create one.",
                UserWarning,
                stacklevel=3,
            )
            return None
        try:
            return anndata.experimental.read_lazy(table_path)
        except ImportError:
            return anndata.read_zarr(table_path)


class ViewTableAccessor(TableAccessor):
    """Table accessor for :class:`InSituExperimentView`.

    Behaves identically to :class:`TableAccessor` but row-filters the loaded
    AnnData to only the samples present in the view.
    """

    def _load(self, cells_layer: str | None) -> AnnData | None:
        full_table = super()._load(cells_layer)
        if full_table is None:
            return None

        label_col = self._exp._read_build_params(cells_layer).get("label_col", "uid")

        if label_col not in self._exp._metadata.columns:
            warnings.warn(
                f"Column '{label_col}' not found in view metadata. "
                "Cannot filter table by view samples.",
                UserWarning,
                stacklevel=3,
            )
            return full_table

        view_sample_ids = set(self._exp._metadata[label_col].values)

        if label_col not in full_table.obs.columns:
            warnings.warn(
                f"Column '{label_col}' not found in concatenated table obs. "
                "Cannot filter by view samples.",
                UserWarning,
                stacklevel=3,
            )
            return full_table

        import numpy as np
        obs_values = np.asarray(full_table.obs[label_col])
        mask = np.isin(obs_values, list(view_sample_ids))
        return full_table[mask]


class InSituExperiment:
    """
    A class to manage and analyze multiple spatially resolved single-cell transcriptomics experiments.

    .. figure:: ../../_static/img/insituexperiment_overview.svg
       :width: 400px
       :align: right
       :class: dark-light

    This class provides functionality for managing datasets, performing differential gene expression analysis,
    querying metadata, visualizing data, and saving/loading experiments. It operates on multiple datasets, each
    represented as an :class:`~insitupy._core.data.InSituData` object, and maintains associated metadata in a
    `pandas.DataFrame`.

    Supports two modes:
    1. InSituPy mode (default): Stores InSituData objects
    2. SpatialData mode: Stores StructuredSpatialData objects from SpatialData zarr stores

    Examples:
        >>> # Create an InSituExperiment object
        >>> experiment = InSituExperiment()

        >>> # Add a dataset
        >>> experiment.add(data="path/to/dataset", mode="insitupy", metadata={"experiment": "test"})

        >>> # Read from SpatialData
        >>> exp = InSituExperiment.read_spatialdata("path/to/data.zarr")

        >>> # Perform differential gene expression analysis
        >>> experiment.dge(target_id=0, ref_id=1, target_annotation_tuple=("cell_type", "neuron"))

        >>> # Save the experiment
        >>> experiment.saveas("path/to/save", overwrite=True)

        >>> # Query the experiment
        >>> subset = experiment.query({"experiment": ["test"]})

        >>> # Plot UMAPs
        >>> experiment.plot_umaps(color="cell_type", title_column="experiment")
    """

    from ._deprecated import collect_anndatas, import_obs, plot_overview

    def __init__(self, data_type: Literal["insitupy", "spatialdata"] = "insitupy"):
        """
        Initialize an InSituExperiment object.

        Args:
            data_type: The type of data to store. Either "insitupy" (default) or "spatialdata".
                       Note: "spatialdata" mode is currently disabled and will be enabled in a future release.
        """
        # Check if spatialdata mode is requested but disabled
        if data_type == "spatialdata" and not _SPATIALDATA_MODE_ENABLED:
            raise NotImplementedError(
                "SpatialData mode is currently disabled and under development. "
                "It will be enabled in a future release. "
                "Please use data_type='insitupy' for now."
            )

        self._metadata = pd.DataFrame(columns=['uid'])
        self._data = []  # Can hold either InSituData or StructuredSpatialData
        self._path = None
        self._colors = {}
        self._filters = {}
        self._composites: dict = {}
        self._applied_filters: list[str] = []
        self._parent_indices: list[int] | None = None
        self._data_type = data_type

    def _modality_counts(self):
        """Return (n_total, [(name, count), ...]) for modalities present in at least one dataset."""
        n = len(self._data)
        checks = {
            "cells":       lambda xd: not xd._cells.is_empty,
            "images":      lambda xd: not xd._images.is_empty,
            "transcripts": lambda xd: xd._transcripts is not None,
            "annotations": lambda xd: not xd._annotations.is_empty,
            "regions":     lambda xd: not xd._regions.is_empty,
            "units":       lambda xd: xd._units is not None,
        }
        result = []
        for name, check in checks.items():
            count = sum(1 for xd in self._data if check(xd))
            if count > 0:
                result.append((name, count))
        return n, result

    def __repr__(self):
        n_samples = len(self._metadata)
        object_name = "InSituExperimentView" if self.is_view else "InSituExperiment"
        mode_str = f" ({tf.Bold}{self._data_type}{tf.ResetAll} mode)"

        header = f"{tf.Bold}{object_name}{tf.ResetAll}{mode_str}\n"
        header += f"{tf.Bold}Path:{tf.ResetAll}\t\t{self._path}"
        if self.applied_filters:
            header += f"\n{tf.Bold}Applied filters:{tf.ResetAll} {' -> '.join(self.applied_filters)}"

        arrow = f"\n{tf.SPACER}{tf.RARROWHEAD}{tf.Bold}"
        indent = f"\n{tf.SPACER}{tf.SPACER}"  # two SPACERs: one for arrow level, one for content
        sub_indent = indent + tf.SPACER       # three SPACERs: one extra for sub-items

        # data section — sample count + wrapped quoted column names
        cols = list(self._metadata.columns)
        cols_quoted = [f'"{c}"' for c in cols]
        col_lines, current = [], ""
        max_col_width = 80 - 2 * len(tf.SPACER)
        for q in cols_quoted:
            candidate = current + (", " if current else "") + q
            if len(candidate) > max_col_width and current:
                col_lines.append(current)
                current = q
            else:
                current = candidate
        col_lines.append(current)
        col_display = indent.join(col_lines)

        n_total, modality_counts = self._modality_counts()
        modality_label = indent + f"{tf.Bold}Loaded modalities{tf.ResetAll}"
        if modality_counts:
            modality_block = modality_label + sub_indent.join(
                [""] + [f"{name}: {count}/{n_total}" for name, count in modality_counts]
            )
        else:
            modality_block = modality_label + sub_indent + "None."

        data_section = (
            f"{arrow} data{tf.ResetAll}"
            + indent + f"{n_samples} samples"
            + indent + f"{len(cols)} metadata columns:"
            + indent + col_display
            + modality_block
        )

        # filters section
        filters_repr = self.filters.__repr__()
        filters_section = (
            f"{arrow} filters{tf.ResetAll}"
            + indent + filters_repr.replace("\n", indent)
        )

        # table section
        table_keys = self.table.keys()
        n_layers = len(table_keys)
        if n_layers == 0:
            table_str = "no tables built"
        elif n_layers == 1:
            table_str = f"1 layer: {table_keys[0]}"
        else:
            table_str = f"{n_layers} layers: {', '.join(table_keys)}"
        table_section = f"{arrow} table{tf.ResetAll}" + indent + table_str

        return header + data_section + filters_section + table_section

    def _repr_html_(self):
        n_samples = len(self._metadata)
        object_name = "InSituExperimentView" if self.is_view else "InSituExperiment"

        parts = [
            f"<b>{object_name}</b> <i>({self._data_type} mode)</i><br>",
            f"<b>Path:</b> {self._path}<br>",
        ]
        if self.applied_filters:
            parts.append(f"<b>Applied filters:</b> {' → '.join(self.applied_filters)}<br>")

        # data section — metadata column summary with quoted names
        cols = list(self._metadata.columns)
        cols_str = ", ".join(f'"{c}"' for c in cols)

        n_total, modality_counts = self._modality_counts()
        if modality_counts:
            mod_lines = "<br>".join(f"{name}: {count}/{n_total}" for name, count in modality_counts)
        else:
            mod_lines = "None."
        modality_html = f"<b>Loaded modalities</b><br><div style='padding-left:1em'>{mod_lines}</div>"

        parts.append(
            f"<b>▶ data</b><br>"
            f"<div style='padding-left:1em'>{n_samples} samples<br>"
            f"{len(cols)} metadata columns:<br>"
            f"{cols_str}<br>"
            f"{modality_html}</div>"
        )

        # filters section
        parts.append(
            f"<b>▶ filters</b><br>"
            f"<div style='padding-left:1em'>"
            + self.filters._repr_html_()
            + "</div>"
        )

        # table section
        table_keys = self.table.keys()
        n_layers = len(table_keys)
        if n_layers == 0:
            table_content = "no tables built"
        elif n_layers == 1:
            table_content = f"1 layer: {table_keys[0]}"
        else:
            table_content = f"{n_layers} layers: {', '.join(table_keys)}"
        parts.append(f"<b>▶ table</b><br><div style='padding-left:1em'>{table_content}</div>")

        return "".join(parts)

    @property
    def is_view(self) -> bool:
        """Return False; this is the base experiment, not a view."""
        return False

    @property
    def applied_filters(self) -> list[str]:
        """Return the list of filter labels that have been applied to this experiment."""
        return list(self._applied_filters)

    def _subset(
        self,
        key,
        as_view: bool = False,
        added_filter: str | None = None,
    ):
        """
        Internal helper to subset experiment data and metadata.

        Args:
            key: Subsetting key (same accepted types as ``__getitem__``).
            as_view: If True, keep path linkage and return an InSituExperimentView.
            added_filter: Optional filter key to append to applied filter history.
        """
        if isinstance(key, int):
            n = len(self)
            if key < -n or key >= n:
                raise IndexError(f"Index ({key}) is out of range {n}.")
            if key < 0:
                key += n
            key = slice(key, key + 1)

        elif isinstance(key, list):
            if all(isinstance(i, bool) for i in key):
                key = pd.Series(key)

        elif isinstance(key, pd.Series):
            if key.dtype != bool:
                key = key.tolist()

        subset_cls = InSituExperimentView if as_view else InSituExperiment

        # Handle boolean mask
        if isinstance(key, pd.Series) and key.dtype == bool:
            if not key.index.equals(self._metadata.index):
                key = key.reset_index(drop=True)
            selected_indices = list(self._metadata.index[key])
            new_experiment = subset_cls(data_type=self._data_type)
            new_experiment._data = [d for d, k in zip(self._data, key) if k]
            new_experiment._metadata = self._metadata[key].reset_index(drop=True)

        # Handle slices, list of ints, ndarray, or Series of ints
        else:
            selected_indices = list(self._metadata.iloc[key].index)
            new_experiment = subset_cls(data_type=self._data_type)
            new_experiment._data = [self._data[i] for i in self._metadata.iloc[key].index]
            new_experiment._metadata = self._metadata.iloc[key].reset_index(drop=True)

        # Carry over colors and filters, and subset filters to the new metadata
        new_experiment._colors = deepcopy(self._colors)
        new_experiment._filters = {}
        if self._filters:
            for name, entry in self._filters.items():
                spec = FilterSpec.from_entry(name, entry)
                mask = spec.mask
                note = spec.note
                mask_arr = np.asarray(mask, dtype=bool)
                if len(mask_arr) != len(self._metadata):
                    warnings.warn(
                        f"Filter '{name}' length ({len(mask_arr)}) does not match metadata length "
                        f"({len(self._metadata)}). Skipping filter in subset.",
                        UserWarning,
                        stacklevel=2
                    )
                    continue
                new_experiment._filters[name] = {
                    "mask": mask_arr[selected_indices].tolist(),
                    "note": note,
                }

        new_experiment._composites = deepcopy(self._composites) if self._composites else {}

        # Keep linkage only for view objects
        if as_view:
            new_experiment._path = self._path
            if getattr(self, "_parent_indices", None) is not None:
                new_experiment._parent_indices = [self._parent_indices[i] for i in selected_indices]
            else:
                new_experiment._parent_indices = list(selected_indices)
        else:
            new_experiment._path = None

        if as_view:
            new_experiment._applied_filters = list(self._applied_filters)
            if added_filter is not None:
                new_experiment._applied_filters.append(added_filter)
        else:
            new_experiment._applied_filters = []

        return new_experiment

    def __getitem__(self, key):
        """
        Retrieve a subset of the experiment.

        Args:
            key (int, slice, list, np.ndarray, pd.Series): The index, slice, list of indices, boolean mask,
                or Series to retrieve.

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the key is invalid.
        """
        return self._subset(key, as_view=self.is_view)

    def __len__(self):
        """Returns the number of datasets in the experiment.

        Returns:
            int: The number of datasets.
        """
        return len(self._data)

    @property
    def data_type(self):
        """The type of data stored in this experiment ('insitupy' or 'spatialdata')."""
        return self._data_type

    @property
    def cells(self):
        """
        Displays a summary of :attr:`~insitupy._core.data.InSituData.cells` for all datasets.
        """
        self.show_modality("cells")

    @property
    def images(self):
        """
        Displays a summary of the 'images' modality for all datasets.
        """
        self.show_modality("images")

    @property
    def transcripts(self):
        """
        Displays a summary of the 'transcripts' modality for all datasets.
        """
        self.show_modality("transcripts")

    @property
    def annotations(self):
        """
        Displays a summary of the 'annotations' modality for all datasets.
        """
        self.show_modality("annotations")

    @property
    def regions(self):
        """
        Displays a summary of the 'regions' modality for all datasets.
        """
        self.show_modality("regions")

    @property
    def colors(self):
        """
        Color dictionaries created by :meth:`~insitupy.experiment.data.InSituExperiment.sync_colors`.

        Returns:
            dict: A dictionary mapping metadata keys to color dictionaries.
        """
        return self._colors

    @property
    def filters(self):
        """
        Filter manager exposing filter operations (create, remove, clear, apply, view, rename).

        Returns:
            FilterManager: Manager object for filter operations and summaries.
        """
        return FilterManager(self)

    @property
    def filter_masks(self):
        """
        Raw filter dictionary mapping keys to boolean masks.

        Returns:
            dict: A dictionary mapping filter keys to boolean-mask lists.
        """
        return self.filters.masks()

    @property
    def data(self):
        """
        List of datasets as :class:`~insitupy._core.data.InSituData` or StructuredSpatialData objects.

        Returns:
            list: A list of data objects.
        """
        return self._data

    @property
    def metadata(self):
        """
        Returns the experiment-level metadata as a pandas DataFrame.

        Returns a copy of the metadata DataFrame. For interactive display, use :attr:`imetadata`.

        Note:
            This returns a **copy** of the internal metadata :class:`pandas.DataFrame`. Any modifications
            to this copy will **not** affect the actual metadata. To modify metadata, use
            `add_metadata_column()` or `append_metadata()`.

        Returns:
            pandas.DataFrame: A copy of the metadata DataFrame.
        """
        logger.warning(
            "You are accessing a copy of the metadata. Changes to this DataFrame will not affect the internal metadata. "
            "Use `add_metadata_column()` or `append_metadata()` to add new information to the metadata."
        )
        return self._metadata.copy()

    # @property
    def imetadata(self, fixed=None):
        """
        Displays the experiment-level metadata as an interactive table using itables.

        This method provides an interactive view of the metadata with search and filter capabilities.
        Requires the `itables` package to be installed.

        Parameters:
            fixed: str, list of str, or None
                Column name(s) to fix/freeze on the left side when scrolling horizontally.
                These columns will be reordered to the left of the table.
                The index column is always included as the first fixed column.
                If None, no columns are fixed.

        Returns:
            None: Displays the interactive table in the output.

        Raises:
            ImportError: If the `itables` package is not installed.
            ValueError: If any specified fixed column is not found in the metadata.
        """
        try:
            from itables import show

            df = self._metadata.reset_index()
            index_col = df.columns[0]  # Get the name of the index column

            # Determine columns to fix and reorder
            if fixed is not None:
                if isinstance(fixed, str):
                    fixed = [fixed]

                # Validate that all fixed columns exist
                missing = [col for col in fixed if col not in df.columns]
                if missing:
                    raise ValueError(f"Fixed column(s) not found in metadata: {missing}")

                # Ensure index column is first, then other fixed columns (avoid duplicates)
                fixed = [index_col] + [col for col in fixed if col != index_col]

                # Reorder columns: fixed columns first, then the rest
                other_cols = [col for col in df.columns if col not in fixed]
                df = df[fixed + other_cols]

                fixed_columns = {"start": len(fixed)}
            else:
                fixed_columns = None

            show_kwargs = {
                "layout": {"bottom1": "searchBuilder"},
                "column_filters": "footer",
            }

            if fixed_columns:
                show_kwargs["scrollX"] = True
                show_kwargs["fixedColumns"] = fixed_columns

            show(df, **show_kwargs)
            return None

        except ImportError:
            logger.warning(
                f"Package `itables` not installed. Install with `pip install itables` for interactive display. "
                f"Falling back to static display.{tf.ResetAll}"
            )
            return self._metadata.copy()

    @property
    def path(self):
        """
        Save path of the InSituExperiment object.

        Returns:
            str or None: The save path of the object, or None if not set.
        """
        return self._path

    def add(self,
            data: str | os.PathLike | Path | InSituData,
            mode: Literal["insitupy", "xenium"] = "insitupy",
            metadata: dict = {}
            ):
        """
        Add a dataset to the experiment and update metadata.

        Args:
            data (Union[str, os.PathLike, Path, InSituData]): The dataset to add. Can be a path or an InSituData object.
            mode (Literal["insitupy", "xenium"], optional): The mode for loading the dataset. Defaults to "insitupy".
            metadata (dict, optional): Additional metadata to associate with the dataset. Defaults to an empty dictionary.

        Raises:
            ValueError: If the mode is invalid.
            AssertionError: If the loaded dataset is not an InSituData object.
        """
        # Check if we're in spatialdata mode
        if self._data_type == "spatialdata":
            raise ValueError(
                "Cannot add individual datasets in SpatialData mode. "
                "Use InSituExperiment.read_spatialdata() to load SpatialData experiments."
            )

        # Check if the dataset is of the correct type
        try:
            data = Path(data)
        except TypeError:
            dataset = data
        else:
            if mode == "insitupy":
                dataset = InSituData.read(data, load_all=False)
            elif mode == "xenium":
                dataset = read_xenium(data)
            else:
                raise ValueError("Invalid mode. Supported modes are 'insitupy' and 'xenium'.")

        # checks whether dataset is an instance of InSituData or any subclass of it, and avoids issues with direct object identity comparison
        if dataset.__class__ is not InSituData:
            raise TypeError(f"Loaded dataset is not an InSituData object. Instead: '{dataset.__class__}'")

        # Resolve the UID for this slot
        existing_uids = list(self._metadata["uid"]) if "uid" in self._metadata.columns else []
        if dataset.uid is not None and dataset.uid in existing_uids:
            # idempotent re-add: dataset already belongs to this experiment slot
            slot_uid = dataset.uid
            idx = existing_uids.index(slot_uid)
            self._data[idx] = dataset
            return
        else:
            # assign a fresh UID for this experiment context
            slot_uid = str(uuid4()).split("-")[0]

        dataset._uid = slot_uid

        # Add the dataset to the data collection
        self._data.append(dataset)

        # Create a new DataFrame for the new metadata
        new_metadata = {
            'uid': slot_uid,
        }

        # add information from metadata argument
        new_metadata.update(metadata)

        # convert to dataframe
        new_metadata = pd.DataFrame([new_metadata])

        # Concatenate the new metadata with the existing metadata
        self._metadata = pd.concat([self._metadata, new_metadata], axis=0, ignore_index=True)

        for entry in self._filters.values():
            entry["mask"].append(False)


    def add_metadata_column(
        self,
        column_name: str,
        values: list | str | pd.Series | np.ndarray,
        overwrite: bool = False
        ):
        """
        Add a metadata column to the experiment.

        Args:
            column_name (str): Name of the column to add.
            values (Union[List, str, pd.Series, np.ndarray]): Values for the new column.
            overwrite (bool, optional): Whether to overwrite the column if it already exists.
                Defaults to False.

        Warns:
            UserWarning: If the column already exists and overwrite is False.
        """
        if column_name in ("slide_id", "sample_id"):
            warnings.warn(
                f'"{column_name}" is also an intrinsic attribute of InSituData objects. '
                f"This column is independent — changes to InSituData.{column_name} will not be reflected here automatically.",
                UserWarning,
                stacklevel=2,
            )

        if column_name in self._metadata.columns and not overwrite:
            warnings.warn(
                f"Column '{column_name}' already exists in metadata. "
                f"Set overwrite=True to replace it.",
                UserWarning,
                stacklevel=2
            )
            return

        self._metadata[column_name] = values

    def append_metadata(self,
                        new_metadata: pd.DataFrame | dict | str | os.PathLike | Path,
                        by: str | None = _UNSET,
                        overwrite: bool = False
                        ):
        """
        Append metadata to the existing InSituExperiment object.

        Args:
            new_metadata (Union[pd.DataFrame, dict, str, os.PathLike, Path]): The new metadata to be added.
                Can be a DataFrame, a dictionary, or a path to a CSV/Excel file.
            by (str, optional): The column name to use for pairing metadata. If not provided, a warning is
                raised prompting the user to specify a column. Set `by=None` explicitly to pair by row order.
            overwrite (bool, optional): Whether to overwrite existing columns in the metadata. Defaults to False.

        Raises:
            TypeError: If new_metadata is not a supported type.
            ValueError: If the 'by' column is not unique or missing in either the existing or new metadata.
        """
        # If 'by' was not explicitly provided, warn and default to None (order-based)
        if by is _UNSET:
            warnings.warn(
                "'by' was not specified. Metadata will be paired by row order, which may lead to "
                "incorrect alignment if the rows are not in the same order. Pass `by='column_name'` "
                "to merge on a key column, or `by=None` to explicitly confirm order-based pairing.",
                UserWarning,
                stacklevel=2
            )
            by = None
        # Convert new_metadata to a DataFrame if it is not already one
        if isinstance(new_metadata, pd.DataFrame):
            pass  # already a DataFrame
        elif isinstance(new_metadata, dict):
            new_metadata = pd.DataFrame(new_metadata)
        elif isinstance(new_metadata, (str, os.PathLike, Path)):
            new_metadata = Path(new_metadata)
            if new_metadata.suffix == '.csv':
                new_metadata = pd.read_csv(new_metadata)
            elif new_metadata.suffix in ['.xlsx', '.xls']:
                new_metadata = pd.read_excel(new_metadata)
            else:
                raise ValueError("Unsupported file format. Please provide a path to a CSV or Excel file.")
        else:
            raise TypeError(
                f"new_metadata must be a DataFrame, dict, or file path, got {type(new_metadata).__name__}."
            )

        # Create a copy of the existing metadata
        old_metadata = self._metadata.copy()

        if by is not None:
            if by not in new_metadata.columns or by not in old_metadata.columns:
                raise ValueError(
                    f"Column '{by}' must be present in both existing and new metadata. "
                    f"If you want to append metadata by order, set `by=None`."
                )

            # Validate that the 'by' column has no NaN values
            if new_metadata[by].isna().any():
                raise ValueError(f"Column '{by}' in new_metadata contains NaN values.")
            if old_metadata[by].isna().any():
                raise ValueError(f"Column '{by}' in existing metadata contains NaN values.")

            # Validate uniqueness of the 'by' column
            if not old_metadata[by].is_unique or not new_metadata[by].is_unique:
                raise ValueError(f"Column '{by}' must be unique in both existing and new metadata.")

        if overwrite:
            # preserve only the columns of the old metadata that are not in the new metadata
            cols_to_use = list(old_metadata.columns.difference(new_metadata.columns))

            if by is not None:
                cols_to_use = [by] + cols_to_use

                # sort them by the original order
                cols_to_use = [elem for elem in old_metadata.columns if elem in cols_to_use]

            # Warn if new_metadata has no overlapping columns to overwrite
            overlapping = list(set(old_metadata.columns) & set(new_metadata.columns) - ({by} if by is not None else set()))
            if len(overlapping) == 0:
                warnings.warn(
                    "No overlapping columns found to overwrite. No changes will be made.",
                    UserWarning,
                    stacklevel=2
                )
                return

            old_metadata = old_metadata[cols_to_use]
        else:
            # preserve only such columns of the new metadata that are not yet in the old metadata
            cols_to_use = list(new_metadata.columns.difference(old_metadata.columns))

            # Warn if new_metadata has no new columns to add
            if len(cols_to_use) == 0:
                warnings.warn(
                    "All columns in new_metadata already exist in the current metadata. "
                    "No new columns to add. Set `overwrite=True` to replace existing columns.",
                    UserWarning,
                    stacklevel=2
                )
                return

            if by is not None:
                cols_to_use = [by] + cols_to_use

            new_metadata = new_metadata[cols_to_use]

        if by is None:
            if len(new_metadata) != len(old_metadata):
                raise ValueError("Length of new metadata does not match the existing metadata.")
            warnings.warn(
                "No 'by' column provided. Metadata will be paired by order.",
                UserWarning,
                stacklevel=2
            )
            updated_metadata = pd.merge(left=old_metadata, right=new_metadata,
                                        left_index=True, right_index=True, how="left")
        else:
            # Warn about unmatched rows in new_metadata
            unmatched = set(new_metadata[by]) - set(old_metadata[by])
            if unmatched:
                warnings.warn(
                    f"{len(unmatched)} entries in new_metadata['{by}'] have no match in existing metadata "
                    f"and will be ignored.",
                    UserWarning,
                    stacklevel=2
                )

            updated_metadata = pd.merge(left=old_metadata, right=new_metadata,
                                        on=by, how="left")

        # Ensure the metadata row count is preserved
        if len(updated_metadata) != len(self._metadata):
            raise ValueError(
                f"Merge changed the number of metadata rows from {len(self._metadata)} to "
                f"{len(updated_metadata)}. This indicates a problem with the merge key."
            )

        # Update the object's metadata only if the check passes
        self._metadata = updated_metadata

    def set_metadata_values(
        self,
        index: int | list[int] | list[bool] | slice | range | np.ndarray | pd.Series,
        column_name: str,
        values: Any | list | pd.Series | np.ndarray
    ):
        """
        Set metadata values for one or more indices.

        Args:
            index: Row index/indices to update. Can be:
                - int: Single row index (e.g., 0)
                - List[int]: Multiple specific indices (e.g., [0, 2, 5])
                - List[bool]: Boolean mask (e.g., [True, False, True])
                - slice: Range of indices (e.g., slice(0, 5))
                - range: Range object (e.g., range(0, 5))
                - np.ndarray: Array of indices or boolean mask
                - pd.Series: Boolean Series for filtering
            column_name: Name of the metadata column to update
            values: Value(s) to set. Can be:
                - Single value: Applied to all specified indices (broadcast)
                - List/Series/array: Must have same length as number of indices

        Raises:
            ValueError: If values is a sequence with mismatched length to indices,
                or if boolean mask length doesn't match metadata length
            KeyError: If column_name doesn't exist in metadata
            IndexError: If any index is out of bounds

        Examples:
            >>> # Single value at single index
            >>> exp.set_metadata_values(0, "type", "tumor")

            >>> # Same value for multiple indices (broadcast)
            >>> exp.set_metadata_values([0, 1, 2], "type", "tumor")

            >>> # Different values for multiple indices
            >>> exp.set_metadata_values([0, 1, 2], "type", ["tumor", "normal", "tumor"])

            >>> # Using slice notation
            >>> exp.set_metadata_values(slice(0, 5), "Localisation", "head")

            >>> # Using range
            >>> exp.set_metadata_values(range(0, 3), "type", ["tumor", "normal", "tumor"])

            >>> # Using boolean mask
            >>> exp.set_metadata_values(exp.metadata["type"] == "normal", "Localisation", "body")

            >>> # Using boolean list
            >>> exp.set_metadata_values([True, False, True, False], "type", "tumor")
        """
        # Normalize index to list
        if isinstance(index, int):
            indices = [index]
        elif isinstance(index, pd.Series):
            # Handle pandas Series (typically boolean masks)
            if index.dtype == bool:
                if len(index) != len(self._metadata):
                    raise ValueError(
                        f"Boolean mask length ({len(index)}) must match metadata length ({len(self._metadata)})"
                    )
                indices = list(self._metadata.index[index])
            else:
                indices = index.tolist()
        elif isinstance(index, slice):
            indices = list(range(*index.indices(len(self._metadata))))
        elif isinstance(index, range):
            indices = list(index)
        elif isinstance(index, np.ndarray):
            # Handle numpy arrays (could be indices or boolean masks)
            if index.dtype == bool:
                if len(index) != len(self._metadata):
                    raise ValueError(
                        f"Boolean mask length ({len(index)}) must match metadata length ({len(self._metadata)})"
                    )
                indices = list(np.where(index)[0])
            else:
                indices = index.tolist()
        elif isinstance(index, list):
            # Check if it's a boolean list
            if index and isinstance(index[0], (bool, np.bool_)):
                if len(index) != len(self._metadata):
                    raise ValueError(
                        f"Boolean mask length ({len(index)}) must match metadata length ({len(self._metadata)})"
                    )
                indices = [i for i, mask in enumerate(index) if mask]
            else:
                indices = list(index)
        else:
            indices = list(index)

        # Validate column exists
        if column_name not in self._metadata.columns:
            raise KeyError(f"Column '{column_name}' does not exist in metadata")

        # Validate indices are in bounds
        max_idx = len(self._metadata) - 1
        out_of_bounds = [i for i in indices if i > max_idx or i < -len(self._metadata)]
        if out_of_bounds:
            raise IndexError(
                f"Indices out of bounds: {out_of_bounds}. "
                f"Valid range: 0 to {max_idx}"
            )

        # Handle values: check if sequence and validate length
        is_sequence = isinstance(values, (list, pd.Series, np.ndarray, tuple))

        if is_sequence:
            values_list = list(values) if not isinstance(values, list) else values
            if len(values_list) != len(indices):
                raise ValueError(
                    f"Length mismatch: {len(indices)} index/indices specified but "
                    f"{len(values_list)} value(s) provided. They must match."
                )
            self._metadata.loc[indices, column_name] = values_list
        else:
            # Single value - broadcast to all indices
            self._metadata.loc[indices, column_name] = values

    def update_metadata(self):
        """Sync ``slide_id`` and ``sample_id`` from child datasets into the experiment metadata.

        .. deprecated::
            ``slide_id`` and ``sample_id`` are no longer stored in the experiment metadata.
            This method is a no-op and will be removed in a future release.
            Access these values directly via ``exp.data[i].slide_id`` and ``exp.data[i].sample_id``.
        """
        warnings.warn(
            "update_metadata() is deprecated and has no effect. "
            "slide_id and sample_id are no longer stored in InSituExperiment metadata. "
            "Access them directly via exp.data[i].slide_id and exp.data[i].sample_id.",
            FutureWarning,
            stacklevel=2,
        )

    def rename_metadata_column(self, old_name: str, new_name: str):
        """Rename a column in the experiment metadata.

        Args:
            old_name (str): Current column name.
            new_name (str): New column name.

        Raises:
            KeyError: If ``old_name`` does not exist in the metadata.
            ValueError: If ``new_name`` already exists in the metadata.

        Examples:
            >>> exp.rename_metadata_column("old_col", "new_col")
        """
        if old_name not in self._metadata.columns:
            raise KeyError(f"Column '{old_name}' not found in metadata.")
        if new_name in self._metadata.columns:
            raise ValueError(f"Column '{new_name}' already exists in metadata.")
        self._metadata.rename(columns={old_name: new_name}, inplace=True)

    def remove_metadata_columns(self, columns):
        """
        Remove specified columns from the internal metadata.

        Args:
            columns (list or str): The column(s) to remove from the metadata.
        """
        self._metadata.drop(columns=columns, inplace=True, errors='ignore')

    def copy(self):
        """
        Create a deep copy of the InSituExperiment object.

        Returns:
            InSituExperiment: A new InSituExperiment object that is a deep copy of the current object.
        """
        return deepcopy(self)

    def dge(
        self,
        target_id: int,
        ref_id: int | list[int] | Literal["rest"] | None = None,
        target_annotation_tuple: tuple[str, str] | None = None,
        target_cell_type_tuple: tuple[str, str] | None = None,
        target_region_tuple: tuple[str, str] | None = None,
        ref_annotation_tuple: Literal["rest", "same"] | tuple[str, str] | None = "same",
        ref_cell_type_tuple: Literal["rest", "same"] | tuple[str, str] | None = "same",
        ref_region_tuple: Literal["rest", "same"] | tuple[str, str] | None = "same",
        method: Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var'] | None = 't-test',
        exclude_ambiguous_assignments: bool = False,
        force_assignment: bool = False,
        name_col: str | None = "uid",
        ):
        """
        Wrapper function for performing differential gene expression analysis within an `InSituExperiment` object.

        This function serves as a wrapper around the `differential_gene_expression` function,
        facilitating the retrieval of data and metadata, and the generation of a plot title
        if not provided. It compares gene expression between specified annotations within
        a single InSituData object or between two InSituData objects.

        Args:
            target_id (int): Index for the target dataset in the `InSituExperiment` object.
            ref_id (Optional[Union[int, List[int], Literal["rest"]]]): Index or list of indices for the reference dataset.
            target_annotation_tuple (Optional[Tuple[str, str]]): Tuple containing the annotation key and name for the primary data.
            target_cell_type_tuple (Optional[Tuple[str, str]]): Tuple specifying an observation key and value to filter the primary data.
            target_region_tuple (Optional[Tuple[str, str]]): Tuple specifying a region key and name to restrict the analysis.
            ref_annotation_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Reference annotation. Defaults to "same".
            ref_cell_type_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Reference cell type. Defaults to "same".
            ref_region_tuple (Optional[Union[Literal["rest", "same"], Tuple[str, str]]]): Reference region. Defaults to "same".
            method (Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']], optional): Statistical method. Defaults to 't-test'.
            exclude_ambiguous_assignments (bool, optional): Whether to exclude ambiguous assignments. Defaults to False.
            force_assignment (bool, optional): Whether to force assignment of annotations and regions. Defaults to False.
            name_col (str, optional): Column name in metadata to use for naming samples. Defaults to "uid".

        Returns:
            DGE results object
        """
        self._check_mode_compatibility("dge")

        from insitupy.tools.dge import dge

        # get data and extract information about experiment
        target = self.data[target_id]
        target_name = self._metadata.loc[target_id, name_col]
        target_metadata = self._metadata.loc[target_id].to_dict()

        if ref_id is not None:
            if ref_id == "rest":
                ref = [d for i, (m, d) in enumerate(self.iterdata()) if i != target_id]
                ref_name = "rest"
                ref_metadata = self._metadata.loc[[i for i in self._metadata.index if i != target_id]].to_dict(orient="list")

            elif isinstance(ref_id, int):
                ref = self.data[ref_id]
                ref_name = self._metadata.loc[ref_id, name_col]
                ref_metadata = self._metadata.loc[ref_id].to_dict()
            elif isinstance(ref_id, list):
                ref = [self.data[i] for i in ref_id]
                ref_name = [self._metadata.iloc[i][name_col] for i in ref_id]
                ref_name = ", ".join(ref_name)
                ref_metadata = self._metadata.loc[ref_id].to_dict(orient="list")
            else:
                raise ValueError(f"Argument `ref_id` has to be either int, list of int or 'rest'. Instead: {ref_id}")

        else:
            ref = None
            ref_name = target_name
            ref_metadata = None

        dge_res = dge(
            target=target,
            ref=ref,
            target_annotation_tuple=target_annotation_tuple,
            target_cell_type_tuple=target_cell_type_tuple,
            target_region_tuple=target_region_tuple,
            target_name=target_name,
            target_metadata=target_metadata,
            ref_annotation_tuple=ref_annotation_tuple,
            ref_cell_type_tuple=ref_cell_type_tuple,
            ref_region_tuple=ref_region_tuple,
            ref_name=ref_name,
            ref_metadata=ref_metadata,
            method=method,
            exclude_ambiguous_assignments=exclude_ambiguous_assignments,
            force_assignment=force_assignment,
        )

        return dge_res

    def get_n_cells(
        self,
        cells_layer: str | None = None
        ):
        """
        Get the total number of cells across all datasets.

        Args:
            cells_layer (Optional[str], optional): The layer to access. Defaults to None.

        Returns:
            int: The total number of cells.
        """
        self._check_mode_compatibility("get_n_cells")

        n_cells = 0
        for _, d in self.iterdata():
            if not d.cells.is_empty:
                celldata = _get_cell_layer(cells=d.cells, cells_layer=cells_layer)
                n_cells += len(celldata.table)

        return n_cells


    def _transfer_to_samples(
        self,
        adata: AnnData,
        uid_column: str,
        uid_column_adata: str,
        obs_columns_to_transfer: list[str] | None = None,
        obsm_keys_to_transfer: list[str] | None = None,
        cells_layer: str | None = None,
        overwrite: bool = False,
        strip_uid_prefix: bool = True,
        fill_missing: bool = True,
    ) -> "InSituExperiment":
        """Transfer obs columns and obsm keys from an AnnData to per-sample AnnData objects.

        Datasets are matched using unique identifiers in the metadata and
        ``adata.obs``. Both ``.obs`` annotations and ``.obsm`` embeddings can
        be transferred.

        Args:
            adata: Source AnnData object.
            uid_column: Column in the InSituExperiment metadata that identifies
                each sample.
            uid_column_adata: Column in ``adata.obs`` that identifies each
                sample.
            obs_columns_to_transfer: ``adata.obs`` columns to copy into each
                per-sample AnnData.
            obsm_keys_to_transfer: ``adata.obsm`` keys to copy.
            cells_layer: Cell layer to receive the transferred data.
            overwrite: If True, overwrite existing columns/keys.
            strip_uid_prefix: If True, strip the ``"{index}-"`` prefix from
                obs_names before matching.
            fill_missing: If True, allow partial matches (missing cells filled
                with NaN).

        Returns:
            Self, for method chaining.
        """
        for meta, xd in self.iterdata():
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            current_uid = meta[uid_column]
            mask = adata.obs[uid_column_adata] == current_uid
            subset = adata[mask].copy()

            if len(subset) == 0:
                warnings.warn(
                    f"No matching data found in `adata` for ID '{current_uid}'. "
                    f"Skipping this dataset.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            # Handle cell name matching
            if strip_uid_prefix:
                if len(subset.obs_names) > 0:
                    sample_name = str(subset.obs_names[0])
                    if '-' in sample_name:
                        subset.obs_names = pd.Index([name.split('-', 1)[1] if '-' in name else name
                                                    for name in subset.obs_names])

            # Check for cell name matches
            matching_cells = celldata.table.obs_names.isin(subset.obs_names)
            n_matching = matching_cells.sum()
            n_total = len(celldata.table)

            if n_matching == 0:
                raise ValueError(
                    f"No matching cell names found for dataset '{current_uid}'. "
                    f"Ensure cell names match between adata and InSituData."
                )

            if n_matching < n_total:
                if not fill_missing:
                    raise ValueError(
                        f"Cell name mismatch for dataset '{current_uid}': "
                        f"Only {n_matching}/{n_total} cells found. "
                        f"Set `fill_missing=True` to allow partial matches."
                    )
                else:
                    warnings.warn(
                        f"Partial match for dataset '{current_uid}': "
                        f"Only {n_matching}/{n_total} cells found. "
                        f"Missing cells will be filled with NaN.",
                        UserWarning,
                        stacklevel=3,
                    )

            # Transfer obs columns
            if obs_columns_to_transfer:
                for col in obs_columns_to_transfer:
                    if col in celldata.table.obs.columns and not overwrite:
                        raise ValueError(
                            f"Column '{col}' already exists for dataset '{current_uid}'. "
                            f"Set `overwrite=True` to overwrite."
                        )
                    celldata.table.obs[col] = subset.obs[col]

            # Transfer obsm keys
            if obsm_keys_to_transfer:
                for key in obsm_keys_to_transfer:
                    if key in celldata.table.obsm.keys() and not overwrite:
                        raise ValueError(
                            f"Key '{key}' already exists for dataset '{current_uid}'. "
                            f"Set `overwrite=True` to overwrite."
                        )

                    # Create empty array with NaN
                    n_cells_target = len(celldata.table)
                    n_features = subset.obsm[key].shape[1]
                    target_array = np.full((n_cells_target, n_features), np.nan)

                    # Fill with matching values
                    subset_index_map = {name: idx for idx, name in enumerate(subset.obs_names)}
                    for target_idx, cell_name in enumerate(celldata.table.obs_names):
                        if cell_name in subset_index_map:
                            subset_idx = subset_index_map[cell_name]
                            target_array[target_idx, :] = subset.obsm[key][subset_idx, :]

                    if np.isnan(target_array).any() and not fill_missing:
                        raise ValueError(
                            f"Cannot transfer obsm key '{key}' for dataset '{current_uid}': "
                            f"Missing values. Set `fill_missing=True`."
                        )

                    celldata.table.obsm[key] = target_array

        return self

    def import_from_anndata(
        self,
        adata: AnnData,
        uid_column: str,
        uid_column_adata: str,
        obs_columns_to_transfer: list[str] | None = None,
        obsm_keys_to_transfer: list[str] | None = None,
        cells_layer: str | None = None,
        overwrite: bool = False,
        strip_uid_prefix: bool = True,
        fill_missing: bool = True
    ) -> "InSituExperiment":
        """
        Import observation and observation matrix data from an AnnData object into the experiment.

        This function transfers data from an AnnData object to the InSituExperiment's
        InSituData objects. Datasets are matched using unique identifiers specified
        in the metadata and AnnData.obs. Data can be transferred from both `.obs`
        (cell-level annotations) and `.obsm` (dimensionality reductions, embeddings).

        Args:
            adata: The AnnData object from which to transfer data.
            uid_column: Column name in the InSituExperiment metadata containing
                unique identifiers for matching datasets.
            uid_column_adata: Column name in `adata.obs` containing unique
                identifiers for matching datasets.
            obs_columns_to_transfer: List of column names in `adata.obs` to transfer.
            obsm_keys_to_transfer: List of keys in `adata.obsm` to transfer.
            cells_layer: The layer in `InSituData.cells` to which data should be added.
            overwrite: If True, overwrites existing columns/keys. Defaults to False.
            strip_uid_prefix: If True, strips the "{index}-" prefix from obs_names. Defaults to True.
            fill_missing: If True, allows partial matches with NaN filling. Defaults to True.

        Returns:
            InSituExperiment: Returns self to allow method chaining.
        """
        self._check_mode_compatibility("import_from_anndata")

        # Validate inputs
        if obs_columns_to_transfer is None and obsm_keys_to_transfer is None:
            raise ValueError(
                "Both `obs_columns_to_transfer` and `obsm_keys_to_transfer` are None. "
                "At least one must be provided."
            )

        # Validate uid_column exists in metadata
        if uid_column not in self._metadata.columns:
            raise ValueError(
                f"Column '{uid_column}' not found in metadata. "
                f"Available columns: {list(self._metadata.columns)}"
            )

        # Validate uid_column_adata exists in adata.obs
        if uid_column_adata not in adata.obs.columns:
            raise ValueError(
                f"Column '{uid_column_adata}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        return self._transfer_to_samples(
            adata=adata,
            uid_column=uid_column,
            uid_column_adata=uid_column_adata,
            obs_columns_to_transfer=obs_columns_to_transfer,
            obsm_keys_to_transfer=obsm_keys_to_transfer,
            cells_layer=cells_layer,
            overwrite=overwrite,
            strip_uid_prefix=strip_uid_prefix,
            fill_missing=fill_missing,
        )

    def import_from_table(
        self,
        obs_columns: list[str] | None = None,
        obsm_keys: list[str] | None = None,
        cells_layer: str | None = None,
        overwrite: bool = False,
    ) -> "InSituExperiment":
        """Import data from the concatenated table back into per-sample AnnData objects.

        Transfers selected obs columns and obsm keys from :attr:`table` back to
        the individual per-sample AnnData objects. This is the reverse operation
        of :meth:`build_table`.

        .. note::
            This feature is experimental and may change in future versions.

        Args:
            obs_columns: Column names in ``table.obs`` to transfer.
            obsm_keys: Keys in ``table.obsm`` to transfer.
            cells_layer: Cell layer to transfer data into.
            overwrite: If True, overwrite existing columns/keys.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If no table has been built yet, or if both ``obs_columns``
                and ``obsm_keys`` are None.
        """
        self._check_mode_compatibility("import_from_table")

        if obs_columns is None and obsm_keys is None:
            raise ValueError(
                "Both `obs_columns` and `obsm_keys` are None. "
                "At least one must be provided."
            )

        try:
            table_path = self._get_table_path(cells_layer)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if table_path is None or not table_path.exists():
            raise ValueError(
                "No concatenated table found. Call `build_table()` first."
            )

        # Load the full table into memory for cell matching
        table_adata = anndata.read_zarr(table_path)

        # Retrieve the label column used during build_table (sidecar JSON takes priority)
        label_col = self._read_build_params(cells_layer).get("label_col", "uid")

        if label_col not in table_adata.obs.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in table.obs. "
                "The table may have been built with a different label_col."
            )

        return self._transfer_to_samples(
            adata=table_adata,
            uid_column=label_col,
            uid_column_adata=label_col,
            obs_columns_to_transfer=obs_columns,
            obsm_keys_to_transfer=obsm_keys,
            cells_layer=cells_layer,
            overwrite=overwrite,
            strip_uid_prefix=True,
            fill_missing=True,
        )

    def iterdata(self, progress: bool = False, desc: str = "Samples"):
        """
        Iterate over the metadata rows and corresponding data.

        Args:
            progress: If True, display a tqdm progress bar. Logging messages and
                Python warnings are routed through :func:`tqdm.write` so they do
                not corrupt the bar.
            desc: Label shown on the progress bar. Defaults to ``"Samples"``.

        Yields:
            tuple: A tuple containing the metadata row as a Series and the corresponding data.
        """
        it = self._metadata.iterrows()
        if not progress:
            for idx, row in it:
                yield row, self._data[idx]
            return

        import warnings

        from tqdm.contrib.logging import logging_redirect_tqdm

        bar = tqdm(it, total=len(self._metadata), desc=desc, dynamic_ncols=True)

        # Patch warnings.showwarning so warning text goes via tqdm.write()
        # instead of straight to stderr, which would corrupt the bar.
        _orig_showwarning = warnings.showwarning

        def _tqdm_showwarning(message, category, filename, lineno, file=None, line=None):
            tqdm.write(
                warnings.formatwarning(message, category, filename, lineno, line).rstrip()
            )

        warnings.showwarning = _tqdm_showwarning
        try:
            with logging_redirect_tqdm():
                for idx, row in bar:
                    yield row, self._data[idx]
        finally:
            warnings.showwarning = _orig_showwarning


    def _concatenate_samples(
        self,
        cells_layer: str | None = None,
        label_col: str = "uid",
        obs_keys: list[str] | str | Literal["all"] | None = None,
        var_keys: list[str] | str | Literal["all"] | None = None,
        obsm_keys: list[str] | str | Literal["all"] | None = "spatial",
        varm_keys: list[str] | str | Literal["all"] | None = None,
        uns_keys: list[str] | str | Literal["all"] | None = None,
        layer_keys: list[str] | str | Literal["all"] | None = None,
        metadata_keys: list[str] | str | Literal["all"] | None = None,
        make_obs_names_unique: bool = True,
        join: Literal["inner", "outer"] = "inner",
        min_shared_genes: int | None = None,
    ) -> anndata.AnnData:
        """Concatenate all sample AnnData objects into a single AnnData.

        Args:
            cells_layer: The layer name to extract cell data from.
            label_col: Column name in metadata to use as labels. Defaults to "uid".
            obs_keys: Keys to select from obs dataframe.
            var_keys: Keys to select from var dataframe.
            obsm_keys: Keys to select from obsm dictionary.
            varm_keys: Keys to select from varm dictionary.
            uns_keys: Keys to select from uns dictionary.
            layer_keys: Keys to select from layers dictionary.
            metadata_keys: Metadata columns to add to obs dataframe.
            make_obs_names_unique: If True, prepends dataset index to obs names.
            join: How to join variables. ``"inner"`` keeps only shared genes;
                ``"outer"`` keeps all genes with fill values.
            min_shared_genes: If set and ``join="inner"``, warn when the
                number of shared genes falls below this threshold.

        Returns:
            AnnData: A concatenated AnnData object.
        """
        # Validate label_col exists in metadata
        if label_col not in self._metadata.columns:
            raise ValueError(
                f"Column '{label_col}' not found in metadata. "
                f"Available columns: {list(self._metadata.columns)}"
            )

        self._assert_cells_loaded(cells_layer)

        adatas: dict[Any, anndata.AnnData] = {}

        for i, (meta, xd) in enumerate(self.iterdata()):
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            adata = celldata.table

            # Filter adata
            adata = _select_anndata_elements(
                adata=adata,
                obs_keys=obs_keys,
                var_keys=var_keys,
                obsm_keys=obsm_keys,
                varm_keys=varm_keys,
                uns_keys=uns_keys,
                layer_keys=layer_keys
            )

            # Add metadata columns to obs
            if metadata_keys is not None:
                if metadata_keys == "all":
                    keys_to_add = list(meta.index)
                else:
                    # make sure keys are a list
                    keys_to_add = convert_to_list(metadata_keys)

                for key in keys_to_add:
                    if key in meta.index:
                        adata.obs[key] = meta[key]
                    else:
                        raise ValueError(
                            f"Column '{key}' not found in metadata. "
                            f"Available columns: {list(self._metadata.columns)}"
                        )

            if make_obs_names_unique:
                adata.obs_names = f"{str(i)}-" + adata.obs_names

            adatas[meta[label_col]] = adata

        adata_concat = anndata.concat(
            adatas,
            axis='obs',
            join=join,
            label=label_col,
            merge="unique"
        )

        # Warn if inner join result has fewer shared genes than threshold
        if join == "inner" and min_shared_genes is not None:
            n_genes = adata_concat.n_vars
            if n_genes < min_shared_genes:
                warnings.warn(
                    f"Only {n_genes} shared genes after inner join "
                    f"(threshold: {min_shared_genes}). Consider using join='outer' "
                    "to retain all genes.",
                    UserWarning,
                    stacklevel=3,
                )

        # Move label_col to first position in obs columns
        if label_col in adata_concat.obs.columns:
            cols = [label_col] + [col for col in adata_concat.obs.columns if col != label_col]
            adata_concat.obs = adata_concat.obs[cols]

        return adata_concat

    def to_anndata(
        self,
        cells_layer: str | None = None,
        label_col: str = "uid",
        obs_keys: list[str] | str | Literal["all"] | None = None,
        var_keys: list[str] | str | Literal["all"] | None = None,
        obsm_keys: list[str] | str | Literal["all"] | None = "spatial",
        varm_keys: list[str] | str | Literal["all"] | None = None,
        uns_keys: list[str] | str | Literal["all"] | None = None,
        layer_keys: list[str] | str | Literal["all"] | None = None,
        metadata_keys: list[str] | str | Literal["all"] | None = None,
        make_obs_names_unique: bool = True,
    ) -> anndata.AnnData:
        """
        Concatenate all datasets into a single AnnData object.

        Args:
            cells_layer: The layer name to extract cell data from.
            label_col: Column name in metadata to use as labels. Defaults to "uid".
            obs_keys: Keys to select from obs dataframe.
            var_keys: Keys to select from var dataframe.
            obsm_keys: Keys to select from obsm dictionary.
            varm_keys: Keys to select from varm dictionary.
            uns_keys: Keys to select from uns dictionary.
            layer_keys: Keys to select from layers dictionary.
            metadata_keys: Metadata columns to add to obs dataframe. Can be a list of column names, a single column name, or "all" for all columns.
            make_obs_names_unique: If True, prepends dataset index to obs names. Defaults to True.

        Returns:
            AnnData: A concatenated AnnData object.
        """
        self._check_mode_compatibility("to_anndata")
        return self._concatenate_samples(
            cells_layer=cells_layer,
            label_col=label_col,
            obs_keys=obs_keys,
            var_keys=var_keys,
            obsm_keys=obsm_keys,
            varm_keys=varm_keys,
            uns_keys=uns_keys,
            layer_keys=layer_keys,
            metadata_keys=metadata_keys,
            make_obs_names_unique=make_obs_names_unique,
            join="inner",
        )

    # ── Table (cross-sample concatenated AnnData) ─────────────────────────────

    def _get_table_path(self, cells_layer: str | None = None) -> Path | None:
        """Return the path to the per-layer zarr table, or None if no experiment path is set.

        When *cells_layer* is given, returns ``tables/{cells_layer}.zarr``.
        When *cells_layer* is ``None``, scans ``tables/`` for ``*.zarr`` directories:

        - Exactly one found → returns it (auto-resolve).
        - Zero found → falls back to the legacy ``concat.zarr`` if present, otherwise
          returns a ``tables/main.zarr`` placeholder so callers can check existence.
        - More than one found → raises :exc:`ValueError` asking for an explicit layer.
        """
        if self.path is None:
            return None
        if cells_layer is not None:
            return Path(self.path) / "tables" / f"{cells_layer}.zarr"
        # Auto-resolve when cells_layer is None
        tables_dir = Path(self.path) / "tables"
        if not tables_dir.exists():
            return tables_dir / "main.zarr"
        candidates = [
            p for p in tables_dir.iterdir()
            if p.suffix == ".zarr" and p.is_dir() and p.name != "concat.zarr"
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            legacy = tables_dir / "concat.zarr"
            return legacy if legacy.exists() else tables_dir / "main.zarr"
        raise ValueError(
            f"Multiple tables found in '{tables_dir}': "
            f"{sorted(p.name for p in candidates)}. "
            "Pass cells_layer= explicitly to select one."
        )

    def _get_build_params_path(self, cells_layer: str | None = None) -> Path | None:
        """Return the path to the per-layer sidecar JSON, or None if no experiment path.

        Mirrors :meth:`_get_table_path`: when *cells_layer* is given returns
        ``tables/{cells_layer}.json``; otherwise derives the stem from the resolved
        table path.
        """
        if self.path is None:
            return None
        if cells_layer is not None:
            return Path(self.path) / "tables" / f"{cells_layer}.json"
        table_path = self._get_table_path()
        if table_path is not None:
            return table_path.with_suffix(".json")
        return Path(self.path) / "tables" / "build_params.json"

    def _read_build_params(self, cells_layer: str | None = None) -> dict:
        """Read build parameters from the per-layer sidecar JSON, with sensible defaults.

        Falls back to the legacy ``tables/build_params.json`` for datasets built
        before the per-layer convention was introduced.

        Returns:
            dict with at least ``label_col``, ``method``, and ``cells_layer`` keys.
        """
        params_path = self._get_build_params_path(cells_layer)
        if params_path is not None and params_path.exists():
            return read_json(params_path)
        # Legacy fallback: shared build_params.json from earlier versions
        if self.path is not None:
            legacy = Path(self.path) / "tables" / "build_params.json"
            if legacy.exists():
                return read_json(legacy)
        return {"label_col": "uid"}

    def _latest_cells_save_dir(self, xd: "InSituData", *, label: str | None = None) -> Path:
        """Return the most recent timestamped cells save directory for *xd*.

        Locates ``<xd path>/cells`` and returns its newest (by timestamp) layer
        directory.  Shared by :meth:`_resolve_cell_layer_from_disk` and
        :meth:`_resolve_per_sample_h5ad_paths`; callers read the
        ``.multicelldata`` sidecar inside the returned directory as needed.

        Args:
            xd: Dataset whose saved cells directory should be located.
            label: Optional dataset identifier (e.g. a uid) interpolated into
                error messages to keep them actionable.

        Returns:
            Path: The most recent timestamped directory under ``cells/``.

        Raises:
            ValueError: If *xd* has no save path, no ``cells`` directory, or no
                saved cells timestamp directory.
        """
        from insitupy.utils._helpers import sort_paths_by_datetime

        desc = f" for dataset '{label}'" if label is not None else ""
        if xd._path is None:
            raise ValueError(
                f"Cannot locate saved cells{desc}: dataset has no save path. "
                "Save the dataset(s) to disk (e.g. via saveas()) first."
            )
        cells_dir = xd._path / "cells"
        if not cells_dir.exists():
            raise ValueError(
                f"No cells directory found{desc} at '{cells_dir}'. "
                "Ensure the dataset has been saved with cell data."
            )
        timestamp_dirs = [p for p in cells_dir.glob("[!.]*") if p.is_dir()]
        if not timestamp_dirs:
            raise ValueError(f"No saved cells found{desc} in '{cells_dir}'.")
        return sort_paths_by_datetime(timestamp_dirs)[0]

    def _resolve_per_sample_h5ad_paths(
        self,
        cells_layer: str | None = None,
        label_col: str = "uid",
    ) -> list[tuple]:
        """Resolve on-disk h5ad paths for each sample.

        Args:
            cells_layer: Cell layer key. ``None`` uses the main layer.
            label_col: Metadata column whose value is used as the sample label.

        Returns:
            List of ``(label_value, h5ad_path)`` pairs, one per sample.

        Raises:
            ValueError: If any dataset has no save path, has no saved cells
                directory, or if the h5ad file cannot be located.
        """
        results = []
        for meta, xd in self.iterdata():
            uid = meta["uid"]
            label_value = meta[label_col]

            most_recent = self._latest_cells_save_dir(xd, label=uid)

            # Determine the layer directory name
            if cells_layer is None:
                mc_meta = read_json(most_recent / ".multicelldata")
                layer_key = mc_meta["key_main"]
            else:
                layer_key = cells_layer

            h5ad_path = most_recent / layer_key / "table.h5ad"
            if not h5ad_path.exists():
                raise ValueError(
                    f"h5ad file not found for dataset '{uid}': '{h5ad_path}'. "
                    "Ensure the dataset has been saved."
                )

            results.append((label_value, h5ad_path))

        return results

    def _concat_samples_on_disk(
        self,
        output_path: Path,
        cells_layer: str | None = None,
        label_col: str = "uid",
        join: Literal["inner", "outer"] = "inner",
        min_shared_genes: int | None = None,
        make_obs_names_unique: bool = True,
    ) -> None:
        """Concatenate per-sample h5ad files on disk without loading all into RAM.

        Uses :func:`anndata.experimental.concat_on_disk` to stream each
        sample's h5ad directly to the output zarr store.

        .. note::
            Requires all datasets to have been saved to disk (i.e. each
            :class:`~insitupy._core.data.InSituData` must have a ``_path``).
            Obs/var key filtering and experiment-metadata columns are not
            supported in this mode.

        Args:
            output_path: Destination zarr path.
            cells_layer: Cell layer to use. ``None`` selects the main layer.
            label_col: Metadata column used as sample identifier label.
            join: ``"inner"`` or ``"outer"`` variable join.
            min_shared_genes: Warn if fewer shared genes after inner join.
            make_obs_names_unique: If True, prepend label value to obs names
                using a ``"-"`` separator.

        Raises:
            ValueError: If any dataset cannot be located on disk.
        """
        from anndata.experimental import concat_on_disk

        sample_paths = self._resolve_per_sample_h5ad_paths(
            cells_layer=cells_layer,
            label_col=label_col,
        )

        # Build ordered mapping: {label_value → h5ad_path}
        in_files: dict[str, Path] = {label: path for label, path in sample_paths}

        index_unique = "-" if make_obs_names_unique else None

        # When in_files is a Mapping, the dict keys serve as the label values;
        # passing keys= separately would be redundant and raises TypeError.
        concat_on_disk(
            in_files=in_files,
            out_file=output_path,
            join=join,
            label=label_col,
            index_unique=index_unique,
            merge="unique",
        )

        if min_shared_genes is not None and join == "inner":
            import zarr
            try:
                z = zarr.open_group(str(output_path), mode="r")
                # var index is stored as a zarr array; shape gives gene count
                var_group = z["var"]
                # anndata stores the index under "_index"
                n_genes = var_group["_index"].shape[0]
            except Exception:
                n_genes = None

            if n_genes is not None and n_genes < min_shared_genes:
                warnings.warn(
                    f"Only {n_genes} shared genes after inner join "
                    f"(threshold: {min_shared_genes}). Consider using join='outer' "
                    "to retain all genes.",
                    UserWarning,
                    stacklevel=4,
                )

        logger.info(
            "Built concatenated table (concat_on_disk) at '%s'.", output_path
        )

    @property
    def table(self) -> "TableAccessor":
        """Dict-like accessor for per-cells-layer concatenated tables.

        Use bracket notation to load the AnnData for a specific layer::

            exp.table["main"]    # AnnData for the "main" segmentation
            exp.table["proseg"]  # AnnData for the "proseg" segmentation
            exp.table[None]      # auto-select when only one table exists
            exp.table.keys()     # list available layer names

        Call :meth:`build_table` first to create a table.

        .. note::
            This feature is experimental and may change in future versions.
        """
        return TableAccessor(self)

    def _resolve_cell_layer_from_disk(self, xd: "InSituData") -> str:
        """Return the main cell layer name for *xd* by reading .multicelldata from disk.

        Used by ``build_table(method='concat_on_disk')`` to resolve
        ``cells_layer=None`` without requiring cells to be in memory.

        Raises:
            ValueError: If *xd* has no save path or no saved cells directory.
        """
        most_recent = self._latest_cells_save_dir(xd)
        mc_meta = read_json(most_recent / ".multicelldata")
        return mc_meta["key_main"]

    def _assert_cells_loaded(self, cells_layer: str | None) -> None:
        """Raise a clear ValueError if any dataset is missing *cells_layer*.

        Args:
            cells_layer: Layer name to check, or None to check for any loaded layer.
        """
        missing = []
        for i, (meta, xd) in enumerate(self.iterdata()):
            keys = xd.cells.keys()
            if cells_layer is None:
                if xd.cells.main_key is None:
                    missing.append((i, meta.get("uid", i)))
            else:
                if cells_layer not in keys:
                    missing.append((i, meta.get("uid", i)))
        if missing:
            raise ValueError(
                f"Cells not loaded for {len(missing)} dataset(s) (index/uid: {missing}). "
                f"Call load_cells() on the experiment before build_table()/to_anndata()."
            )

    def build_table(
        self,
        cells_layer: str | None = None,
        label_col: str = "uid",
        obs_keys: list[str] | str | Literal["all"] | None = None,
        var_keys: list[str] | str | Literal["all"] | None = None,
        obsm_keys: list[str] | str | Literal["all"] | None = "spatial",
        varm_keys: list[str] | str | Literal["all"] | None = None,
        uns_keys: list[str] | str | Literal["all"] | None = None,
        layer_keys: list[str] | str | Literal["all"] | None = None,
        metadata_keys: list[str] | str | Literal["all"] | None = None,
        make_obs_names_unique: bool = True,
        join: Literal["inner", "outer"] = "inner",
        min_shared_genes: int | None = None,
        overwrite: bool = False,
        method: Literal["in_memory", "concat_on_disk"] = "in_memory",
    ) -> None:
        """Build a zarr-backed concatenated AnnData across all samples.

        Concatenates all per-sample AnnData objects and writes the result to
        ``{experiment_path}/tables/concat.zarr``. After building, access the
        result via :attr:`table`.

        Two concatenation strategies are available via ``method``:

        - ``"in_memory"`` *(default)*: loads every sample's AnnData into RAM,
          concatenates with :func:`anndata.concat`, then writes zarr.
          Supports all filtering and metadata options.
        - ``"concat_on_disk"``: streams each sample's saved ``table.h5ad``
          file directly to the output zarr store using
          :func:`anndata.experimental.concat_on_disk`.
          Requires all datasets to be saved on disk. Does **not** support
          ``obs_keys``, ``var_keys``, ``obsm_keys``, ``varm_keys``,
          ``uns_keys``, ``layer_keys``, or ``metadata_keys``. Obs name
          prefixes use the label value (e.g. ``"uid-cell_0"``) rather than
          the numeric index (``"0-cell_0"``).

        .. note::
            This feature is experimental and may change in future versions.

        Args:
            cells_layer: Cell layer to extract from each sample.
            label_col: Metadata column used as the sample identifier label.
                Defaults to ``"uid"``.
            obs_keys: Obs columns to retain (``in_memory`` only).
            var_keys: Var columns to retain (``in_memory`` only).
            obsm_keys: Obsm keys to retain (``in_memory`` only). Defaults to
                ``"spatial"``.
            varm_keys: Varm keys to retain (``in_memory`` only).
            uns_keys: Uns keys to retain (``in_memory`` only).
            layer_keys: Layer keys to retain (``in_memory`` only).
            metadata_keys: Experiment metadata columns to add to obs
                (``in_memory`` only).
            make_obs_names_unique: Prepend a prefix to obs names to guarantee
                uniqueness across samples.
            join: How to join variables. ``"inner"`` (default) keeps only
                shared genes; ``"outer"`` keeps all genes with fill values.
            min_shared_genes: Warn when fewer than this many genes remain after
                an inner join.
            overwrite: If True, overwrite an existing table.
            method: Concatenation strategy. ``"in_memory"`` (default) or
                ``"concat_on_disk"`` for memory-efficient on-disk streaming.

        Raises:
            ValueError: If the experiment has no save path, or if
                ``method="concat_on_disk"`` is used with unsupported filter
                arguments.
            FileExistsError: If a table already exists and ``overwrite=False``.
        """
        self._check_mode_compatibility("build_table")

        if self.path is None:
            raise ValueError(
                "Cannot build table: experiment has no save path. "
                "Call `saveas()` first to give the experiment a path."
            )

        # concat_on_disk never touches in-memory cells; in_memory does.
        if method != "concat_on_disk":
            self._assert_cells_loaded(cells_layer)

        # Validate concat_on_disk restrictions
        if method == "concat_on_disk":
            unsupported = {
                "obs_keys": obs_keys,
                "var_keys": var_keys,
                "obsm_keys": obsm_keys if obsm_keys != "spatial" else None,
                "varm_keys": varm_keys,
                "uns_keys": uns_keys,
                "layer_keys": layer_keys,
                "metadata_keys": metadata_keys,
            }
            active = [k for k, v in unsupported.items() if v is not None]
            if active:
                raise ValueError(
                    f"method='concat_on_disk' does not support these arguments: "
                    f"{active}. Use method='in_memory' to apply filtering."
                )

        # Resolve cells_layer=None to the actual layer key so the output filename
        # is always explicit (e.g. "main" rather than the ambiguous None).
        # concat_on_disk reads from .multicelldata on disk; in_memory reads cells in memory.
        if cells_layer is None:
            if method == "concat_on_disk":
                cells_layer = self._resolve_cell_layer_from_disk(
                    next(self.iterdata())[1]
                )
            else:
                _, cells_layer = _get_cell_layer(
                    next(self.iterdata())[1].cells, cells_layer=None, return_layer_name=True
                )

        output_path = self._get_table_path(cells_layer)
        check_overwrite_and_remove_if_true(path=output_path, overwrite=overwrite)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        build_params = {"label_col": label_col, "method": method, "cells_layer": cells_layer}

        if method == "in_memory":
            adata = self._concatenate_samples(
                cells_layer=cells_layer,
                label_col=label_col,
                obs_keys=obs_keys,
                var_keys=var_keys,
                obsm_keys=obsm_keys,
                varm_keys=varm_keys,
                uns_keys=uns_keys,
                layer_keys=layer_keys,
                metadata_keys=metadata_keys,
                make_obs_names_unique=make_obs_names_unique,
                join=join,
                min_shared_genes=min_shared_genes,
            )
            adata.uns["_insitupy_build_params"] = build_params
            adata.write_zarr(output_path)
            logger.info(
                "Built concatenated table at '%s' (%d cells, %d genes).",
                output_path,
                adata.n_obs,
                adata.n_vars,
            )

        elif method == "concat_on_disk":
            self._concat_samples_on_disk(
                output_path=output_path,
                cells_layer=cells_layer,
                label_col=label_col,
                join=join,
                min_shared_genes=min_shared_genes,
                make_obs_names_unique=make_obs_names_unique,
            )

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'in_memory' or 'concat_on_disk'."
            )

        # Write per-layer sidecar build params (readable by both methods without loading zarr)
        import json as _json
        self._get_build_params_path(cells_layer).write_text(_json.dumps(build_params))

    def load_all(self,
                 skip: str | None = None,
                 ):
        """
        Load all data modalities for all datasets.

        Args:
            skip (Optional[str], optional): A modality to skip during loading. Defaults to None.
        """
        self._check_mode_compatibility("load_all")

        for xd in tqdm(self._data):
            for f in LOAD_FUNCS:
                if skip is None or skip not in f:
                    func = getattr(xd, f)
                    try:
                        func()
                    except ModalityNotFoundError as err:
                        logger.warning(str(err))

    def load_annotations(self):
        """Load annotations for all datasets."""
        self._check_mode_compatibility("load_annotations")
        for xd in tqdm(self._data):
            xd.load_annotations()

    def load_cells(self):
        """Load cells for all datasets."""
        self._check_mode_compatibility("load_cells")
        for xd in tqdm(self._data):
            xd.load_cells()

    def load_images(self,
                    names: Literal["all", "nuclei"] | str = "all",
                    overwrite: bool = False,
                    verbose: bool = False
                    ):
        """Load images for all datasets."""
        self._check_mode_compatibility("load_images")

        for xd in tqdm(self._data):
            xd.load_images(
                names=names,
                overwrite=overwrite,
                verbose=verbose
                )

    def load_regions(self):
        """Load regions for all datasets."""
        self._check_mode_compatibility("load_regions")
        for xd in tqdm(self._data):
            xd.load_regions()

    def load_transcripts(self,
                        transcript_filename: str = "transcripts.parquet"
                        ):
        """Load transcripts for all datasets."""
        self._check_mode_compatibility("load_transcripts")
        for xd in tqdm(self._data):
            xd.load_transcripts()

    @with_insitupy_style
    def plot_embedding(
        self,
        basis: str,
        cells_layer: str | None = None,
        color: str | None = None,
        title_column: str | None = None,
        title_size: int = 24,
        max_cols: int = 4,
        figsize: tuple[int, int] = (8,6),
        savepath: str | os.PathLike | Path | None = None,
        save_only: bool = False,
        show: bool = True,
        fig: Figure | None = None,
        dpi_save: int = 300,
        **kwargs
        ):
        """Create a plot with embeddings of all datasets as subplots."""
        self._check_mode_compatibility("plot_embedding")

        from insitupy.plotting.save import save_and_show_figure

        num_datasets = len(self._data)
        n_plots, n_rows, max_cols = get_nrows_maxcols(len(self._data), max_cols)
        fig, axes = plt.subplots(n_rows, max_cols, figsize=(figsize[0]*max_cols, figsize[1]*n_rows))
        if n_plots > 1:
            axes = axes.ravel()

        # make sure title_columns is a list
        if title_column is not None:
            title_columns = self._metadata[title_column].tolist()
        else:
            title_columns = [f"Sample {idx + 1}" for idx in range(len(self))]

        for idx, (metadata_row, dataset) in enumerate(self.iterdata()):
            ax = axes[idx] if num_datasets > 1 else axes

            # Get data from MultiCellData
            celldata = _get_cell_layer(cells=dataset.cells, cells_layer=cells_layer)
            adata = celldata.table

            # plot UMAP and add to axis
            sc.pl.embedding(
                adata=adata,
                basis=basis,
                color=color,
                ax=ax,
                show=False,
                **kwargs
            )

            ax.set_title(title_columns[idx],
                         fontdict={"fontsize": title_size},
                         pad=10
                         )

        remove_empty_subplots(axes, n_plots, n_rows, max_cols)
        if show:
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)
        else:
            return fig, axes


    @with_insitupy_style
    def plot_umaps(
        self,
        cells_layer: str | None = None,
        color: str | None = None,
        title_column: str | None = None,
        title_size: int = 20,
        max_cols: int = 4,
        figsize: tuple[int, int] = (8, 6),
        savepath: str | os.PathLike | Path | None = None,
        save_only: bool = False,
        show: bool = True,
        fig: Figure | None = None,
        dpi_save: int = 300,
        **kwargs
    ):
        """Create a plot with UMAPs of all datasets as subplots."""
        return self.plot_embedding(
            basis='X_umap',
            cells_layer=cells_layer,
            color=color,
            title_column=title_column,
            title_size=title_size,
            max_cols=max_cols,
            figsize=figsize,
            savepath=savepath,
            save_only=save_only,
            show=show,
            fig=fig,
            dpi_save=dpi_save,
            **kwargs
        )

    def query(self, criteria):
        """Query the experiment based on metadata criteria.

        Args:
            criteria (dict or str):
                - A dictionary where keys are column names and values are lists of categories to select.
                - A string expression to evaluate using pandas.DataFrame.query().

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        if isinstance(criteria, dict):
            mask = pd.Series([True] * len(self._metadata), index=self._metadata.index)
            for column, values in criteria.items():
                values = convert_to_list(values)
                if column in self._metadata.columns:
                    mask &= self._metadata[column].isin(values)
                else:
                    raise KeyError(f"Column '{column}' not found in metadata.")
            return self[mask]

        elif isinstance(criteria, str):
            try:
                result_df = self._metadata.query(criteria)
                return self[result_df.index]
            except Exception as e:
                raise ValueError(f"Failed to evaluate query expression: {e}")

        else:
            raise TypeError("Criteria must be either a dictionary or a string.")



    def remove_history(self):
        """Remove history from all datasets."""
        self._check_mode_compatibility("remove_history")
        for xd in tqdm(self._data):
            xd.remove_history(verbose=False)

    def reload(
        self,
        skip: list | None = None,
        verbose: bool = True,
    ):
        """Reload all datasets and experiment-level files from disk.

        Calls :meth:`~insitupy._core.data.InSituData.reload` on every child
        dataset, then re-reads ``metadata.csv``, ``colors.json``, and
        ``filters.json`` from the experiment's save path.  Useful after a
        :meth:`save` call to replace in-memory data with fresh on-disk state.

        Args:
            skip: Modality name(s) forwarded to each dataset's
                :meth:`~insitupy._core.data.InSituData.reload` (e.g.
                ``["images"]``).  Defaults to ``None``.
            verbose: If ``True``, log progress.  Defaults to ``True``.

        Raises:
            ValueError: If no experiment save path is set.
        """
        self._check_mode_compatibility("reload")

        if self.path is None:
            raise ValueError(
                "No save path available. Cannot reload without a save path. "
                "Save the experiment with `saveas()` first."
            )

        path = Path(self.path)

        # Collect the union of loaded modalities across all datasets for the summary log
        if verbose:
            all_modalities: set = set()
            for xd in self._data:
                all_modalities.update(xd.get_loaded_modalities())
            skip_set = set(skip) if skip else set()
            active_modalities = [m for m in all_modalities if m not in skip_set]
            logger.info(
                "Reloading %d dataset(s) [modalities: %s]...",
                len(self._data),
                ", ".join(active_modalities) if active_modalities else "none",
            )

        # Reload each child dataset (suppress per-dataset messages to keep progress bar clean)
        for xd in tqdm(self._data):
            xd.reload(skip=skip, verbose=False)

        # Reload experiment metadata (Parquet canonical or legacy CSV).
        if (path / _METADATA_PARQUET_FILENAME).exists() or (path / "metadata.csv").exists():
            self._metadata = self._read_metadata_with_schema(path)
            if verbose:
                logger.info("Reloaded metadata, colors, and filters from disk.")

        # Reload colors
        colors_path = path / "colors.json"
        if colors_path.exists():
            with open(colors_path) as f:
                self._colors = json.load(f)
        else:
            self._colors = {}

        # Reload filters
        self._filters = {}
        self._composites = {}
        self._applied_filters = []
        filters_path = path / "filters.json"
        if filters_path.exists():
            try:
                with open(filters_path) as f:
                    filters_payload = json.load(f)
                version = filters_payload.get("version", None)
                filters = filters_payload.get("filters", None)
                if version not in _SUPPORTED_FILTER_VERSIONS:
                    raise ValueError(
                        f"Unsupported filters schema version: {version}. "
                        f"Supported versions: {sorted(_SUPPORTED_FILTER_VERSIONS)}."
                    )
                if isinstance(filters, dict):
                    for name, entry in filters.items():
                        spec = FilterSpec.from_entry(name, entry)
                        mask_arr = np.asarray(spec.mask, dtype=bool)
                        if len(mask_arr) != len(self._metadata):
                            warnings.warn(
                                f"Filter '{name}' length ({len(mask_arr)}) does not match "
                                f"metadata length ({len(self._metadata)}). Skipping.",
                                UserWarning,
                                stacklevel=2,
                            )
                            continue
                        self._filters[name] = {"mask": mask_arr.tolist(), "note": spec.note}
                if version == 2:
                    composites = filters_payload.get("composites", {})
                    if isinstance(composites, dict):
                        for name, entry in composites.items():
                            try:
                                comp = CompositeFilterSpec.from_entry(name, entry)
                                self._composites[name] = comp.to_dict()
                            except (ValueError, KeyError) as err:
                                warnings.warn(
                                    f"Could not load composite filter '{name}': {err}. Skipping.",
                                    UserWarning,
                                    stacklevel=2,
                                )
            except Exception as err:
                warnings.warn(f"Could not reload filters.json: {err}", UserWarning, stacklevel=2)

    def unload(self, modalities: list | None = None):
        """Unload modality data from memory for every dataset in this experiment.

        Calls :meth:`~insitupy._core.data.InSituData.unload` on every child
        dataset.  The experiment object itself (metadata, colors, filters) is
        unaffected.  Call the corresponding ``load_*()`` methods to bring
        modalities back into memory.

        Args:
            modalities: Modality name(s) to unload (e.g. ``["cells", "images"]``).
                Defaults to ``None``, which unloads all modalities.

        Raises:
            ValueError: If any dataset has no save path set and has loaded
                modalities, because unloading would make that data unrecoverable.
        """
        self._check_mode_compatibility("unload")

        target = set(convert_to_list(modalities)) if modalities is not None else set(MODALITIES)
        missing = [
            i for i, xd in enumerate(self._data)
            if xd._path is None and bool(set(xd.get_loaded_modalities()) & target)
        ]
        if missing:
            raise ValueError(
                f"Cannot unload: dataset(s) at index {missing} have no save "
                f"path. Call saveas() first to avoid permanent data loss."
            )

        all_modalities = list(MODALITIES)
        active = modalities if modalities is not None else all_modalities
        logger.info(
            "Unloading %d dataset(s) [modalities: %s]...",
            len(self._data),
            ", ".join(active) if active else "none",
        )
        for xd in tqdm(self._data):
            xd.unload(modalities=modalities, verbose=False)

    def replace(self, idx: int | str, new_data: "InSituData", *, confirm: bool = True) -> None:
        """Replace a dataset in this experiment with *new_data*.

        The slot's UID and on-disk path are inherited by *new_data* so the experiment
        metadata row remains consistent.  The in-memory swap happens unconditionally;
        the disk write only occurs after optional confirmation.

        Args:
            idx: Integer position or UID string of the slot to replace.
            new_data: The replacement :class:`~insitupy._core.data.InSituData` object.
            confirm: If ``True`` (default), prompt before overwriting the directory on disk.
                Set to ``False`` for scripted use.

        Raises:
            IndexError: If *idx* is an integer outside the valid range.
            KeyError: If *idx* is a UID string not present in the experiment.
            ValueError: If this experiment has no path set (cannot write to disk).
        """
        if self.is_view:
            raise ValueError(
                "replace() is not supported on an InSituExperimentView. "
                "Call replace() on the parent experiment and re-create the view."
            )
        # Resolve idx to an integer position
        if isinstance(idx, str):
            uid_series = self._metadata["uid"]
            matches = uid_series[uid_series == idx].index.tolist()
            if not matches:
                raise KeyError(f"No dataset with UID '{idx}' found in this experiment.")
            pos = matches[0]
        else:
            pos = idx
            if pos < 0 or pos >= len(self._data):
                raise IndexError(
                    f"Index {pos} out of range. Valid range: 0 to {len(self._data) - 1}."
                )

        bad_path = self._data[pos].path
        slot_uid = self._metadata.loc[pos, "uid"]

        if new_data.uid is not None:
            warnings.warn(
                f"new_data already has uid='{new_data.uid}'. "
                f"It will be overwritten with the slot uid='{slot_uid}'.",
                UserWarning,
                stacklevel=2,
            )

        # Memory swap (non-destructive, unconditional)
        new_data._uid = slot_uid
        new_data._path = bad_path
        self._data[pos] = new_data

        if confirm:
            print(f"The following directory will be permanently overwritten: {bad_path}")
            answer = input("Proceed? [Y/n]: ").strip()
            if answer.lower() not in ("", "y"):
                print(
                    "Disk write cancelled. The in-memory swap is active but the directory on disk is unchanged."
                )
                return

        if bad_path is None:
            raise ValueError(
                "Cannot write to disk: the replaced slot has no path. "
                "Use confirm=False only after verifying the path is set."
            )

        new_data.saveas(bad_path, overwrite=True)

    def remove(self, idx: int | str, *, confirm: bool = True, delete_from_disk: bool = False) -> None:
        """Remove a dataset from this experiment.

        Drops the dataset at the given position from the in-memory list and
        experiment metadata.  Filter masks are truncated to match the new
        dataset count.  The on-disk directory is left untouched unless
        ``delete_from_disk=True``.

        Args:
            idx: Integer position or UID string of the dataset to remove.
            confirm: If ``True`` (default), print a summary and prompt for
                confirmation before proceeding.  Set to ``False`` for scripted use.
            delete_from_disk: If ``True``, permanently delete the dataset
                directory from disk using :func:`shutil.rmtree`.  Skipped
                silently when the dataset has no path set.  Default ``False``.

        Raises:
            IndexError: If *idx* is an integer outside the valid range.
            KeyError: If *idx* is a UID string not present in the experiment.

        Note:
            Filter masks are truncated to match the new dataset count after
            removal.  ``delete_from_disk=False`` (default) leaves the dataset
            directory untouched on disk.

            This method does **not** automatically persist the updated experiment
            to disk after removal.  Call ``self.save()`` explicitly afterwards
            if you want the change to be durable.

            When called on an :class:`InSituExperimentView` with
            ``delete_from_disk=False``, only the view's ``_data`` and
            ``_metadata`` are updated; the parent experiment is unaffected.
            Use ``delete_from_disk=True`` on the parent experiment to
            permanently delete a dataset from disk.
        """
        if self.is_view and delete_from_disk:
            raise ValueError(
                "delete_from_disk=True is not allowed on an InSituExperimentView "
                "because the parent experiment still references this dataset. "
                "Call remove() on the parent experiment instead."
            )
        # Resolve idx to an integer position
        if isinstance(idx, str):
            uid_series = self._metadata["uid"]
            matches = uid_series[uid_series == idx].index.tolist()
            if not matches:
                raise KeyError(f"No dataset with UID '{idx}' found in this experiment.")
            pos = matches[0]
        else:
            pos = idx
            if pos < 0 or pos >= len(self._data):
                raise IndexError(
                    f"Index {pos} out of range. Valid range: 0 to {len(self._data) - 1}."
                )

        path = self._data[pos].path
        uid = self._metadata.loc[pos, "uid"]

        if confirm:
            print(
                f"Dataset at position {pos} (uid='{uid}', path={path}) will be removed "
                "from this experiment."
            )
            if delete_from_disk and path is not None:
                print(
                    f"The dataset directory will also be permanently deleted from disk: {path}"
                )
            answer = input("Proceed? [y/N]: ").strip()
            if answer.lower() != "y":
                print("Removal cancelled.")
                return

        # Memory removal
        del self._data[pos]
        self._metadata = self._metadata.drop(index=pos).reset_index(drop=True)
        for entry in self._filters.values():
            entry["mask"].pop(pos)

        # Disk deletion
        if delete_from_disk and path is not None:
            shutil.rmtree(path)

        print("Experiment updated in memory but not yet saved to disk. Call .save() to persist the change.")

    def save(self,
             verbose: bool = False,
             collect_warnings_mode: bool = True,
             **kwargs
             ):
        """Save the full experiment to its existing project path.

        This method has a single responsibility: perform a full project save
        (datasets + metadata + colors + filters).

        For partial save workflows, use dedicated methods:
        ``save_metadata()``, ``save_colors()``, ``save_images()``, and ``save_filters()``.

        When called on an :class:`InSituExperimentView`, ``save()`` only writes
        each dataset's per-dataset state. It deliberately does **not** write
        experiment-level files (``metadata.parquet``, ``colors.json``,
        ``filters.json``) — those would otherwise corrupt the parent. To export
        a subset as a standalone experiment, call ``view.saveas(path)``. To
        update individual columns of the parent's metadata from a view, call
        ``view.save_metadata()``.

        Args:
            verbose: If True, print verbose output for dataset-level save operations.
            collect_warnings_mode: If True, collect warnings and print a summary at end
                instead of displaying them inline (prevents progress bar disruption).
            **kwargs: Additional keyword arguments passed to ``InSituData.save()``.

        Raises:
            ValueError: If no experiment save path is available or dataset paths are inconsistent.
            RuntimeError: If one or more datasets fail to save. All datasets are attempted
                regardless of individual failures; experiment-level files (metadata, colors,
                filters) are written only when all datasets succeed.
        """
        self._check_mode_compatibility("save")

        if self.is_view:
            if collect_warnings_mode:
                with collect_warnings() as collector:
                    for xd in tqdm(self._data):
                        xd.save(verbose=verbose, **kwargs)
                collector.print_summary()
            else:
                for xd in tqdm(self._data):
                    xd.save(verbose=verbose, **kwargs)
            return

        if self.path is None:
            raise ValueError(
                "No save path available. First save the InSituExperiment using `saveas()` "
                "or set `self.path` by reading an existing experiment."
            )

        parent_path_identical = [
            (d.path is not None) and (Path(d.path).parent == self.path)
            for d in self.data
        ]
        if not np.all(parent_path_identical):
            invalid_uids = self._metadata.loc[~np.array(parent_path_identical), "uid"].tolist()
            raise ValueError(
                "Saving failed: save path of some InSituData objects does not lie inside "
                f"the InSituExperiment save path. Affected uids: {invalid_uids}"
            )

        failures = []

        if collect_warnings_mode:
            with collect_warnings() as collector:
                for xd in tqdm(self._data):
                    try:
                        xd.save(verbose=verbose, **kwargs)
                    except Exception as exc:
                        failures.append((xd.uid, exc))
            collector.print_summary()
        else:
            for xd in tqdm(self._data):
                try:
                    xd.save(verbose=verbose, **kwargs)
                except Exception as exc:
                    failures.append((xd.uid, exc))

        if failures:
            summary = "\n".join(
                f"  - {uid}: {type(exc).__name__}: {exc}"
                for uid, exc in failures
            )
            raise RuntimeError(
                f"save() failed for {len(failures)}/{len(self._data)} dataset(s):\n{summary}\n"
                "Experiment-level files (metadata, colors, filters) were NOT updated. "
                "Fix the failing datasets and call save() again."
            )

        self.save_metadata(overwrite=True)
        self.save_colors(overwrite=True)
        self.save_filters(path=self.path)

    def save_metadata(
        self,
        path: str | os.PathLike | Path | None = None,
        overwrite: bool = True,
    ):
        """Save experiment metadata to ``metadata.parquet`` (canonical) and ``metadata.csv`` (export).

        The Parquet file is the authoritative store; ``metadata.csv`` is regenerated on every save
        as a human-readable reference and should not be edited directly.

        Args:
            path: Directory where metadata files should be written.
                If None, uses ``self.path``.
            overwrite: If True, overwrite existing metadata files.

        Raises:
            ValueError: If neither ``path`` nor ``self.path`` is set.
            FileExistsError: If metadata files exist and ``overwrite`` is False.
        """
        if path is None:
            if self.path is None:
                raise ValueError(
                    "No save path available. Provide `path` or set `self.path` first (e.g. via `saveas`)."
                )
            path = self.path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        parquet_path = path / _METADATA_PARQUET_FILENAME
        csv_path = path / "metadata.csv"

        if (parquet_path.exists() or csv_path.exists()) and not overwrite:
            raise FileExistsError(
                "Metadata file(s) already exist. "
                f"Found: {parquet_path} and/or {csv_path}. "
                "Set `overwrite=True` to replace them."
            )

        # Atomic Parquet write: write to a temp file first, then replace.
        # Path.replace() uses os.replace(), which overwrites the target atomically on all platforms.
        tmp_path = path / "metadata.parquet.tmp"
        self._metadata.to_parquet(tmp_path, index=False)
        tmp_path.replace(parquet_path)

        # Regenerate CSV as a human-readable export (not the canonical source).
        # Write to a tmp file first, then replace atomically.
        tmp_csv_path = path / "metadata.csv.tmp"
        with open(tmp_csv_path, "w", newline="") as f:
            f.write("# AUTO-GENERATED — human-readable export only; edits are ignored (canonical data is in metadata.parquet)\n")
            self._metadata.to_csv(f, index=True)
        tmp_csv_path.replace(csv_path)

        # Remove stale schema sidecar written by older versions.
        stale_schema = self._metadata_schema_path(path)
        if stale_schema.exists():
            stale_schema.unlink()

    @staticmethod
    def _metadata_schema_path(path: Path) -> Path:
        """Return the path to metadata schema sidecar file."""
        return path / _METADATA_SCHEMA_FILENAME

    @staticmethod
    def _metadata_dtype_map(metadata: pd.DataFrame) -> dict[str, str]:
        """Serialize DataFrame dtypes to JSON-compatible strings."""
        return {str(column): str(dtype) for column, dtype in metadata.dtypes.items()}

    @classmethod
    def _save_metadata_schema(cls, path: Path, metadata: pd.DataFrame) -> None:
        """Write metadata dtype schema used for safe round-trip casting on reload."""
        schema_path = cls._metadata_schema_path(path)
        schema_payload = {
            "version": _METADATA_SCHEMA_VERSION,
            "column_dtypes": cls._metadata_dtype_map(metadata),
        }
        with open(schema_path, 'w') as f:
            json.dump(schema_payload, f, indent=2, sort_keys=True)

    @classmethod
    def _load_metadata_dtype_map(cls, path: Path) -> dict[str, str] | None:
        """Load metadata dtype map if available and valid, else return None."""
        schema_path = cls._metadata_schema_path(path)
        if not schema_path.exists():
            return None

        try:
            with open(schema_path) as f:
                schema_payload = json.load(f)
        except Exception as err:
            logger.warning(
                "Could not parse metadata schema at '%s': %s. Falling back to CSV dtype inference.",
                schema_path,
                err,
            )
            return None

        if not isinstance(schema_payload, dict):
            logger.warning(
                "Invalid metadata schema at '%s': expected a JSON object. Falling back to CSV dtype inference.",
                schema_path,
            )
            return None

        version = schema_payload.get("version", None)
        if version != _METADATA_SCHEMA_VERSION:
            logger.warning(
                "Unsupported metadata schema version at '%s': %s (expected %s). "
                "Falling back to CSV dtype inference.",
                schema_path,
                version,
                _METADATA_SCHEMA_VERSION,
            )
            return None

        column_dtypes = schema_payload.get("column_dtypes", None)
        if not isinstance(column_dtypes, dict):
            logger.warning(
                "Invalid metadata schema at '%s': 'column_dtypes' must be a dictionary. "
                "Falling back to CSV dtype inference.",
                schema_path,
            )
            return None

        return {
            str(column): str(dtype)
            for column, dtype in column_dtypes.items()
        }

    @classmethod
    def _read_metadata_with_schema(cls, path: Path) -> pd.DataFrame:
        """Read experiment metadata, preferring the Parquet canonical store.

        Falls back to ``metadata.csv`` + optional schema sidecar for legacy directories
        that predate the Parquet format.
        """
        parquet_path = path / _METADATA_PARQUET_FILENAME
        if parquet_path.exists():
            metadata = pd.read_parquet(parquet_path)
            return cls._migrate_legacy_metadata(metadata)

        # Legacy path: CSV + optional dtype schema sidecar.
        metadata_path = path / "metadata.csv"
        dtype_map = cls._load_metadata_dtype_map(path)

        # Force string-like columns at CSV parse time to preserve values such as
        # leading-zero IDs before post-load casting is applied.
        read_csv_kwargs: dict[str, Any] = {"comment": "#"}
        if dtype_map:
            string_like_dtypes = {"str", "string", "object", "category"}
            csv_dtypes = {
                column: "string"
                for column, dtype in dtype_map.items()
                if dtype.lower() in string_like_dtypes
            }
            if csv_dtypes:
                read_csv_kwargs["dtype"] = csv_dtypes

        metadata = pd.read_csv(metadata_path, index_col=0, **read_csv_kwargs)

        if not dtype_map:
            return cls._migrate_legacy_metadata(metadata)

        for column, dtype in dtype_map.items():
            if column not in metadata.columns:
                continue
            try:
                metadata[column] = metadata[column].astype(dtype)
            except Exception as err:
                logger.warning(
                    "Could not cast metadata column '%s' to dtype '%s': %s. Keeping inferred dtype.",
                    column,
                    dtype,
                    err,
                )

        return cls._migrate_legacy_metadata(metadata)

    @staticmethod
    def _migrate_legacy_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """Discard legacy slide_id/sample_id columns from loaded metadata CSVs."""
        legacy_cols = [c for c in ("slide_id", "sample_id") if c in df.columns]
        if legacy_cols:
            df = df.drop(columns=legacy_cols)
        return df

    def save_colors(
        self,
        path: str | os.PathLike | Path | None = None,
        overwrite: bool = True,
    ):
        """Save only experiment colors to ``colors.json``.

        Args:
            path: Directory where ``colors.json`` should be written.
                If None, uses ``self.path``.
            overwrite: If True, overwrite an existing ``colors.json``.

        Raises:
            ValueError: If neither ``path`` nor ``self.path`` is set.
            FileExistsError: If ``colors.json`` exists and ``overwrite`` is False.
        """
        if path is None:
            if self.path is None:
                raise ValueError(
                    "No save path available. Provide `path` or set `self.path` first (e.g. via `saveas`)."
                )
            path = self.path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        colors_path = path / "colors.json"

        if colors_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {colors_path}. Set `overwrite=True` to replace it."
            )

        write_dict_to_json(self.colors, colors_path)

    def save_images(
        self,
        overwrite: bool = False,
        collect_warnings_mode: bool = True,
        **kwargs,
    ):
        """Save only image data for datasets in the experiment.

        This method syncs images only by forwarding ``sync_images=True`` and
        ``images_only=True`` to each dataset save call.

        Args:
            overwrite: If True, overwrite existing images on disk.
                Default is False (skip images that already exist).
            collect_warnings_mode: If True, collect warnings and print summary at end
                instead of displaying them inline (prevents progress bar disruption).
            **kwargs: Additional keyword arguments passed to ``InSituData.save()``.

        Raises:
            ValueError: If no experiment save path is available or dataset paths are inconsistent.
        """
        self._check_mode_compatibility("save_images")

        dataset_verbose = kwargs.pop("verbose", False)

        if not self.is_view:
            if self.path is None:
                raise ValueError(
                    "No save path available. First save the InSituExperiment using `saveas()` "
                    "or set `self.path` by reading an existing experiment."
                )

            parent_path_identical = [
                (d.path is not None) and (Path(d.path).parent == self.path)
                for d in self.data
            ]
            if not np.all(parent_path_identical):
                invalid_uids = self._metadata.loc[~np.array(parent_path_identical), "uid"].tolist()
                raise ValueError(
                    "Saving images failed: save path of some InSituData objects does not lie inside "
                    f"the InSituExperiment save path. Affected uids: {invalid_uids}"
                )

        if collect_warnings_mode:
            with collect_warnings() as collector:
                for xd in tqdm(self._data):
                    xd.save_images(overwrite=overwrite, verbose=dataset_verbose, **kwargs)
            collector.print_summary()
        else:
            for xd in tqdm(self._data):
                xd.save_images(overwrite=overwrite, verbose=dataset_verbose, **kwargs)

    def save_geometries(
        self,
        collect_warnings_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        """Save only annotation and region geometries for all datasets.

        Iterates all datasets and calls
        :meth:`~insitupy.InSituData.save_geometries` on each.  All other
        modalities are left untouched on disk.

        Args:
            collect_warnings_mode: If ``True``, collect warnings and print a
                summary at the end instead of displaying them inline.
            verbose: If ``True``, log per-dataset progress messages.

        Raises:
            ValueError: If no experiment save path is available or dataset
                paths are inconsistent.
        """
        self._check_mode_compatibility("save_geometries")

        dataset_verbose = verbose

        if not self.is_view:
            if self.path is None:
                raise ValueError(
                    "No save path available. First save the InSituExperiment using `saveas()` "
                    "or set `self.path` by reading an existing experiment."
                )

            parent_path_identical = [
                (d.path is not None) and (Path(d.path).parent == self.path)
                for d in self.data
            ]
            if not np.all(parent_path_identical):
                invalid_uids = self._metadata.loc[~np.array(parent_path_identical), "uid"].tolist()
                raise ValueError(
                    "Saving geometries failed: save path of some InSituData objects does not lie "
                    f"inside the InSituExperiment save path. Affected uids: {invalid_uids}"
                )

        if collect_warnings_mode:
            with collect_warnings() as collector:
                for xd in tqdm(self._data):
                    xd.save_geometries(verbose=dataset_verbose)
            collector.print_summary()
        else:
            for xd in tqdm(self._data):
                xd.save_geometries(verbose=dataset_verbose)

    def save_cells(
        self,
        collect_warnings_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        """Save only cell data (expression table and boundaries) for all datasets.

        Iterates all datasets and calls
        :meth:`~insitupy.InSituData.save_cells` on each.  All other
        modalities are left untouched on disk.

        Args:
            collect_warnings_mode: If ``True``, collect warnings and print a
                summary at the end instead of displaying them inline.
            verbose: If ``True``, log per-dataset progress messages.

        Raises:
            ValueError: If no experiment save path is available or dataset
                paths are inconsistent.
        """
        self._check_mode_compatibility("save_cells")

        dataset_verbose = verbose

        if not self.is_view:
            if self.path is None:
                raise ValueError(
                    "No save path available. First save the InSituExperiment using `saveas()` "
                    "or set `self.path` by reading an existing experiment."
                )

            parent_path_identical = [
                (d.path is not None) and (Path(d.path).parent == self.path)
                for d in self.data
            ]
            if not np.all(parent_path_identical):
                invalid_uids = self._metadata.loc[~np.array(parent_path_identical), "uid"].tolist()
                raise ValueError(
                    "Saving cells failed: save path of some InSituData objects does not lie "
                    f"inside the InSituExperiment save path. Affected uids: {invalid_uids}"
                )

        if collect_warnings_mode:
            with collect_warnings() as collector:
                for xd in tqdm(self._data):
                    xd.save_cells(verbose=dataset_verbose)
            collector.print_summary()
        else:
            for xd in tqdm(self._data):
                xd.save_cells(verbose=dataset_verbose)

    def saveas(
        self,
        path: str | os.PathLike | Path,
        overwrite: bool = False,
        verbose: bool = False,
        collect_warnings_mode: bool = True,
        free_after_save: bool = False,
        **kwargs):
        """Save experiment to a new location (initial full write).

        This method writes all datasets to ``path`` and then saves metadata,
        colors, and filters using dedicated helper methods.

        Args:
            path: Path to save the InSituExperiment.
            overwrite: If True, overwrite existing files.
            verbose: If True, print verbose output.
            collect_warnings_mode: If True, collect warnings and print summary at end
                instead of displaying them inline (prevents progress bar disruption).
            free_after_save: If True, every dataset's in-memory modality data is
                released **after the full save completes** (all datasets written
                and the destination swapped into place).  This frees RAM once the
                experiment is safely on disk; it does **not** reduce peak RAM
                *during* the write, since all datasets remain in memory until the
                save finishes.  Afterwards the ``InSituExperiment`` object has
                empty data containers and should be reloaded from disk for further
                use.  Defaults to False.
            **kwargs: Additional keyword arguments passed to dataset.saveas().
        """
        self._check_mode_compatibility("saveas")

        path = Path(path)
        staging = path.parent / (path.name + ".__ispy_tmp__")

        # clean any stale staging dir left by a previous failed write
        check_overwrite_and_remove_if_true(staging, overwrite=True)
        # check overwrite flag on the final target (delete deferred to inside try)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"The output file already exists at {path}. "
                "To overwrite it, please set the `overwrite` parameter to True."
            )

        backup = path.parent / (path.name + ".__ispy_bak__")
        # clean any stale backup left by a previous failed write
        check_overwrite_and_remove_if_true(backup, overwrite=True)

        logger.info(f"Saving InSituExperiment to {str(path)}") if verbose else None

        destination_backed_up = False
        try:
            if collect_warnings_mode:
                with collect_warnings() as collector:
                    for index, dataset in enumerate(tqdm(self._data)):
                        subfolder_path = staging / f"data-{str(index).zfill(3)}"
                        dataset.saveas(subfolder_path, verbose=False, **kwargs)
                collector.print_summary()
            else:
                for index, dataset in enumerate(tqdm(self._data)):
                    subfolder_path = staging / f"data-{str(index).zfill(3)}"
                    dataset.saveas(subfolder_path, verbose=False, **kwargs)

            self.save_metadata(path=staging, overwrite=True)
            self.save_colors(path=staging, overwrite=True)
            self.save_filters(path=staging)

            # Atomic swap: move the old destination aside so it can be restored if
            # the final rename fails, then move staging into place.
            if path.exists():
                os.rename(path, backup)
                destination_backed_up = True
            os.rename(staging, path)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if destination_backed_up and not path.exists() and backup.exists():
                try:
                    os.rename(backup, path)
                except Exception:
                    logger.error(
                        "saveas failed AND the previous experiment could not be restored "
                        "automatically. Your original data is preserved at '%s' — rename it "
                        "back to '%s' manually.", backup, path,
                    )
            raise
        finally:
            # Remove the backup only once the destination is confirmed in place.
            # If neither swap nor restore succeeded, keep it — it is the only surviving copy.
            if backup.exists() and path.exists():
                shutil.rmtree(backup, ignore_errors=True)

        self._path = path.resolve()

        for index, dataset in enumerate(self._data):
            dataset._path = (path / f"data-{str(index).zfill(3)}").resolve()

        if free_after_save:
            for dataset in self._data:
                dataset._release_data()
            gc.collect()

        logger.info("Saved.") if verbose else None

    def save_filters(
        self,
        path: str | os.PathLike | Path | None = None,
        overwrite: bool = True,
    ):
        """
        Save only experiment filters to ``filters.json``.

        Args:
            path: Directory where ``filters.json`` should be written.
                If None, uses ``self.path``.
            overwrite: If True, overwrite an existing ``filters.json``.

        Raises:
            ValueError: If neither ``path`` nor ``self.path`` is set.
            FileExistsError: If ``filters.json`` exists and ``overwrite`` is False.
        """
        if path is None:
            if self.path is None:
                raise ValueError(
                    "No save path available. Provide `path` or set `self.path` first (e.g. via `saveas`)."
                )
            path = self.path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        filters_path = path / "filters.json"
        if filters_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {filters_path}. Set `overwrite=True` to replace it."
            )

        filters_payload = self._build_filters_payload()
        write_dict_to_json(filters_payload, filters_path)

    def show(
        self,
        index: int,
        verbose: bool = False
        ):
        """
        Displays the dataset at the specified index.

        Args:
            index (int): The index of the dataset to display.
            verbose (bool, optional): If True, show verbose output. Defaults to False.
        """
        dataset = self.data[index]
        dataset.show(verbose=verbose)

    def show_modality(self, modality, uid_column: str = "uid"):
        """Show a modality for all datasets."""
        repr_string = ""
        for meta, data in self.iterdata():
            repr_string += f"{meta.name}: {tf.Bold+tf.Red}{meta[uid_column]}{tf.ResetAll}\n"

            if self._data_type == "insitupy":
                repr_string += f"{tf.SPACER}   " + data.get_modality(modality).__repr__().replace("\n", f"\n{tf.SPACER}   ") + "\n"
            else:
                # For spatialdata mode, get modality directly
                if modality == "cells":
                    repr_string += f"{tf.SPACER}   " + data._cells.__repr__().replace("\n", f"\n{tf.SPACER}   ") + "\n"
                elif modality == "images":
                    repr_string += f"{tf.SPACER}   " + data._images.__repr__().replace("\n", f"\n{tf.SPACER}   ") + "\n"
                elif modality == "transcripts":
                    repr_string += f"{tf.SPACER}   " + str(data._transcripts) + "\n"
                elif modality == "annotations":
                    repr_string += f"{tf.SPACER}   " + data._annotations.__repr__().replace("\n", f"\n{tf.SPACER}   ") + "\n"
                elif modality == "regions":
                    repr_string += f"{tf.SPACER}   " + data._regions.__repr__().replace("\n", f"\n{tf.SPACER}   ") + "\n"

        logger.info(repr_string)

    def sync_colors(
        self,
        keys: str | list[str],
        cells_layer: str | None = None,
        palette: ListedColormap = DEFAULT_CATEGORICAL_CMAP,
        overwrite: bool = False,
        verbose: bool = False
    ):
        """
        Synchronize color dictionaries for categorical metadata across datasets.

        Args:
            keys (Union[str, List[str]]): The metadata keys to synchronize colors for.
            cells_layer (Optional[str], optional): The layer to access. Defaults to None.
            palette (ListedColormap, optional): The color palette to use.
            overwrite (bool, optional): Whether to overwrite existing color
                dictionaries. Defaults to False.
            verbose (bool, optional): Whether to print status messages. Defaults to True.
        """
        self._check_mode_compatibility("sync_colors")

        # Make sure obs_cols is a list
        keys = convert_to_list(keys)

        for obs_col in keys:
            # Skip numeric columns to avoid treating continuous values as categorical
            is_numeric = False
            for _, xd in self.iterdata():
                celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
                if obs_col in celldata.table.obs.columns:
                    series = celldata.table.obs[obs_col]
                    if is_numeric_dtype(series) and not is_bool_dtype(series):
                        is_numeric = True
                        break

            if is_numeric:
                if verbose:
                    logger.info(f"Skipping sync_colors for numeric column '{obs_col}'.")
                continue

            if obs_col not in self.colors or overwrite:
                # create a color dictionary with all categories
                color_dict = self._create_categorical_color_dict(
                    obs_col=obs_col,
                    cells_layer=cells_layer,
                    palette=palette
                )

                if color_dict is not None:
                    # iterate over all datasets and set the colors in .uns
                    uns_key = f"{obs_col}_colors"
                    for _, xd in self.iterdata():
                        celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

                        try:
                            # try to retrieve categories
                            cats = celldata.table.obs[obs_col].cat.categories.values
                        except AttributeError:
                            # convert to categorical
                            celldata.table.obs[obs_col] = celldata.table.obs[obs_col].astype("category")

                            # retrieve categories
                            cats = celldata.table.obs[obs_col].cat.categories.values
                            cats = np.unique(celldata.table.obs[obs_col])
                        celldata.table.uns[uns_key] = [color_dict[c] for c in cats]

                    # save color dict in InSituExperiment
                    self.colors[obs_col] = color_dict

                    if verbose:
                        logger.info(f"Synchronized colors for key '{obs_col}' and palette '{palette.name}'.")
            else:
                logger.info(f"Key '{obs_col}' found already in `exp.colors`. To overwrite it, run `sync_colors` with `overwrite=True`.")


    @classmethod
    def concat(
        cls,
        objs,
        new_col_name=None,
        path: str | os.PathLike | Path | None = None,
        mode: Literal["copy", "move"] = "copy",
    ):
        """Concatenate multiple InSituExperiment objects.

        Args:
            objs (Union[List[InSituExperiment], Dict[str, InSituExperiment]]):
                A list of InSituExperiment objects or a dictionary where keys
                are added as a new column.
            new_col_name (str, optional):
                The name of the new column to add when objs is a dictionary.
                Defaults to None.
            path (Union[str, os.PathLike, Path], optional):
                Destination directory for the concatenated experiment.
                Required when ``mode="move"``.
            mode (str):
                ``"copy"`` (default) — datasets remain at their original paths
                and the new experiment object is in-memory only (no save path
                is set unless you call :meth:`saveas` afterwards).

                ``"move"`` — each dataset directory is moved (not copied) to
                ``path``, the experiment-level files (``metadata.parquet``,
                ``metadata.csv``, ``colors.json``, ``filters.json``) are written
                there, and the
                original experiment root directories are removed.  This is
                disk-efficient because no data is duplicated.

                .. warning::
                    ``mode="move"`` is **destructive and irreversible**.
                    All source experiments must reside on the same filesystem
                    as ``path``.  Subsetted experiments (created via ``[]``
                    indexing) are supported: their datasets are moved normally,
                    but the original experiment root directory is **not** removed
                    automatically — a :class:`UserWarning` is emitted and the
                    caller is responsible for cleaning up the remainder.

        Returns:
            InSituExperiment: A new InSituExperiment object.

        Raises:
            ValueError: For invalid arguments or precondition violations in
                ``mode="move"``.

        Notes:
            Existing filter masks are **always dropped** after concatenation
            because their lengths no longer match the merged metadata.
            Colors from all source experiments are merged with a first-wins
            strategy: the first experiment whose color dict contains a given
            key takes precedence.
        """
        if isinstance(objs, dict):
            if new_col_name is None:
                raise ValueError("new_col_name must be provided when objs is a dictionary.")
            keys, objs = zip(*objs.items())
        else:
            keys = [None] * len(objs)

        objs = list(objs)

        # Validate mode
        if mode not in ("copy", "move"):
            raise ValueError(f"mode must be 'copy' or 'move', got '{mode}'.")

        if mode == "move" and path is None:
            raise ValueError("path must be provided when mode='move'.")

        # Check that all objects have the same data type
        data_types = [obj._data_type for obj in objs]
        if len(set(data_types)) > 1:
            raise ValueError(
                f"Cannot concatenate InSituExperiment objects with different data types: {set(data_types)}"
            )
        data_type = data_types[0]

        # --- mode="move" precondition checks ---
        if mode == "move":
            for obj in objs:
                if getattr(obj, "is_view", False):
                    raise ValueError(
                        "concat(..., mode='move') cannot operate on an InSituExperimentView. "
                        "A view shares its datasets with the parent experiment; moving them would "
                        "relocate the parent's data and then delete the parent directory. "
                        "Materialise the view first (view.saveas(path); "
                        "InSituExperiment.read(path)) or use mode='copy'."
                    )
            for i, obj in enumerate(objs):
                if not isinstance(obj, InSituExperiment):
                    raise TypeError("All objects must be instances of InSituExperiment.")
                for xd in obj._data:
                    if xd._path is None:
                        raise ValueError(
                            f"mode='move' requires all datasets to have a save path set, "
                            f"but dataset '{xd.slide_id}' in experiment at index {i} has no path. "
                            f"Call saveas() on the experiment first."
                        )

            path = Path(path)

            # Verify same filesystem per dataset (covers subset experiments whose
            # obj._path is None): compare device of destination against each source.
            path.mkdir(parents=True, exist_ok=True)
            dst_dev = os.stat(path).st_dev
            for obj in objs:
                for xd in obj._data:
                    src_dev = os.stat(xd._path).st_dev
                    if src_dev != dst_dev:
                        raise ValueError(
                            f"mode='move' requires source and destination to be on the "
                            f"same filesystem. Source '{xd._path}' and destination '{path}' "
                            f"are on different devices. Use mode='copy' + saveas() instead."
                        )

            return cls._concat_move(
                objs=objs,
                keys=list(keys),
                new_col_name=new_col_name,
                data_type=data_type,
                path=path,
            )

        # --- mode="copy" (original behaviour + colors merge) ---
        new_experiment = cls(data_type=data_type)

        new_data = []
        new_metadata = []
        merged_colors: dict = {}

        for key, obj in zip(keys, objs):
            if not isinstance(obj, InSituExperiment):
                raise TypeError("All objects must be instances of InSituExperiment.")
            new_data.extend(obj._data)
            metadata = obj._metadata.copy()
            if key is not None:
                metadata[new_col_name] = key
            new_metadata.append(metadata)
            # Merge colors: first-wins per key
            for k, v in obj._colors.items():
                if k not in merged_colors:
                    merged_colors[k] = v

        new_experiment._data = new_data
        new_experiment._metadata = pd.concat(new_metadata, ignore_index=True)
        new_experiment._colors = merged_colors
        # Filters are intentionally dropped: masks are sized to individual
        # experiment metadata and cannot be meaningfully merged.
        new_experiment._path = None

        # check if observation names are unique (only for insitupy mode)
        if data_type == "insitupy":
            new_experiment._check_obs_uniqueness()

        return new_experiment

    @classmethod
    def _concat_move(
        cls,
        objs: list,
        keys: list,
        new_col_name,
        data_type: str,
        path: Path,
    ) -> "InSituExperiment":
        """Back-end for ``concat(..., mode='move')``.

        Moves each source dataset directory to ``path/data-NNN``, updates
        ``InSituData._path``, detaches in-memory data, writes experiment-level
        files to ``path``, and removes the now-empty source experiment roots.
        Subset experiments (``obj._path is None``) are supported: their datasets
        are moved normally but the original experiment root is not removed.
        """
        new_experiment = cls(data_type=data_type)
        new_metadata: list = []
        merged_colors: dict = {}
        global_idx = 0

        # Warn upfront about subset experiments whose roots will not be cleaned up.
        for i, obj in enumerate(objs):
            if getattr(obj, "is_view", False):
                continue
            if obj._path is None and obj._data:
                inferred_root = obj._data[0]._path.parent
                warnings.warn(
                    f"Experiment at index {i} is a subset (no experiment root path set). "
                    f"{len(obj._data)} dataset(s) will be moved out of '{inferred_root}', "
                    f"but that directory will NOT be removed automatically — any remaining "
                    f"datasets and experiment-level files there must be cleaned up manually.",
                    UserWarning,
                    stacklevel=3,
                )

        for key, obj in zip(keys, objs):
            # Release in-memory data before moving so the move loop stays clean
            obj.unload()

            desc = f"Moving datasets from {obj._path.name}" if obj._path else "Moving datasets"
            for xd in tqdm(obj._data, desc=desc):
                dst = path / f"data-{str(global_idx).zfill(3)}"
                shutil.move(str(xd._path), str(dst))
                xd._path = dst
                new_experiment._data.append(xd)
                global_idx += 1

            metadata = obj._metadata.copy()
            if key is not None:
                metadata[new_col_name] = key
            new_metadata.append(metadata)

            # Merge colors first-wins
            for k, v in obj._colors.items():
                if k not in merged_colors:
                    merged_colors[k] = v

        new_experiment._metadata = pd.concat(new_metadata, ignore_index=True)
        new_experiment._colors = merged_colors
        new_experiment._path = path

        # Write experiment-level files
        new_experiment.save_metadata(path=path, overwrite=True)
        new_experiment.save_colors(path=path, overwrite=True)
        new_experiment.save_filters(path=path)

        # Remove source experiment roots (datasets have already been moved out).
        # Subset experiments have no root path and are skipped.
        _EXPECTED_ROOT_FILES = {
            "metadata.csv",
            _METADATA_PARQUET_FILENAME,
            _METADATA_SCHEMA_FILENAME,
            "colors.json",
            "filters.json",
        }
        for obj in objs:
            if getattr(obj, "is_view", False):
                continue
            if obj._path is None:
                continue
            remaining = {p.name for p in obj._path.iterdir()}
            unexpected = remaining - _EXPECTED_ROOT_FILES
            if unexpected:
                warnings.warn(
                    f"Removing source experiment root '{obj._path}' which still contains "
                    f"{len(unexpected)} unexpected item(s): {sorted(unexpected)}. "
                    f"These will be permanently deleted.",
                    UserWarning,
                    stacklevel=3,
                )
            shutil.rmtree(str(obj._path))
            obj._path = None

        if data_type == "insitupy":
            new_experiment._check_obs_uniqueness()

        return new_experiment

    @classmethod
    def from_config(cls,
                    config: str | os.PathLike | Path | pd.DataFrame,
                    mode: Literal["insitupy", "xenium", "auto"] = "auto",
                    collect_warnings_mode: bool = True,
                    **kwargs
                    ):
        """Create an InSituExperiment object from a configuration file or DataFrame.

        Args:
            config (Union[str, os.PathLike, Path, pd.DataFrame]): Configuration specifying the
                datasets to load. Either a path to a CSV or Excel file, or a :class:`pandas.DataFrame`
                directly. Must contain a ``'directory'`` column with the path to each dataset.
                When passing a DataFrame the index is ignored.
            mode (Literal["insitupy", "xenium", "auto"]): The mode to use for loading the datasets.
                - "auto": Automatically detect the format of each directory by looking for ``.ispy``
                  (InSituPy project) or ``experiment.xenium`` (Xenium output bundle). Raises a
                  ``ValueError`` if neither marker file is found. Defaults to ``"auto"``.
                - "insitupy": Load previously saved InSituPy projects using :meth:`~insitupy._core.data.InSituData.read`.
                - "xenium": Load Xenium data bundles directly using :func:`~insitupy.io.read_xenium`.
            collect_warnings_mode (bool): If True, collect warnings during loading and print a summary at the end.
                This keeps the progress bar clean while still showing important warnings. Defaults to True.
        """
        if isinstance(config, pd.DataFrame):
            config = config.reset_index(drop=True)
        else:
            config_path = Path(config)
            if config_path.suffix == '.csv':
                config = pd.read_csv(config_path, dtype=str)
            elif config_path.suffix in ('.xlsx', '.xls'):
                config = pd.read_excel(config_path, dtype=str)
            else:
                raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")

        # Ensure the 'directory' column exists
        if 'directory' not in config.columns:
            raise ValueError("The configuration file must contain a 'directory' column.")

        # Get the current working directory
        current_path = Path.cwd()

        # Initialize a new InSituExperiment object
        experiment = cls(data_type="insitupy")

        # Create a warning collector if collect_warnings_mode is enabled
        warning_collector = WarningCollector() if collect_warnings_mode else None

        def _resolve_mode(path: Path) -> str:
            if (path / ISPY_METADATA_FILE).exists():
                return "insitupy"
            elif (path / "experiment.xenium").exists():
                return "xenium"
            else:
                raise ValueError(
                    f"Cannot auto-detect format for '{path}': neither '{ISPY_METADATA_FILE}' "
                    f"nor 'experiment.xenium' was found. Set 'mode' explicitly."
                )

        # Iterate over each row in the configuration file
        for i in tqdm(range(len(config))):
            row = config.iloc[i, :]
            dataset_path = Path(row['directory'])

            # Check if the path is relative and if so, append the current path
            if not dataset_path.is_absolute():
                dataset_path = current_path / dataset_path

            # Check if the directory exists
            if not dataset_path.exists():
                raise FileNotFoundError(f"No such directory found: {str(dataset_path)}")

            resolved_mode = _resolve_mode(dataset_path) if mode == "auto" else mode

            ctx = collect_warnings(warning_collector) if collect_warnings_mode else contextlib.nullcontext()
            with ctx:
                if resolved_mode == "insitupy":
                    dataset = InSituData.read(dataset_path, load_all=False)
                elif resolved_mode == "xenium":
                    dataset = read_xenium(dataset_path, verbose=False, **kwargs)
                else:
                    raise ValueError(f"Invalid mode '{resolved_mode}'. Supported modes are 'insitupy', 'xenium', and 'auto'.")

            experiment._data.append(dataset)

            # Extract metadata from the row, excluding the 'directory' column
            metadata = row.drop(labels=['directory']).to_dict()
            slot_uid = str(uuid4()).split("-")[0]
            metadata['uid'] = slot_uid
            dataset._uid = slot_uid

            # Append the metadata to the experiment's metadata DataFrame
            experiment._metadata = pd.concat([experiment._metadata, pd.DataFrame([metadata])], ignore_index=True)

        # Print collected warnings summary at the end
        if warning_collector and len(warning_collector) > 0:
            warning_collector.print_summary()

        return experiment

    @classmethod
    def from_regions(cls,
                    data: InSituData,
                    region_key: str,
                    region_names: list[str] | str | None = None,
                    lazy: bool = False,
                    detach_transcripts: bool = True
                    ):
        """Creates an `InSituExperiment` object from specified regions in the given `InSituData`.

        Args:
            data (InSituData): The input data containing regions to extract.
            region_key (str): The key identifying the region of interest in `data.regions`.
            region_names (Optional[Union[List[str], str]]): Region names to include.
            lazy (bool): If ``False`` (default), transcript data is loaded into memory
                once before cropping, making per-region crops fast (pandas boolean mask
                instead of repeated Dask task-graph traversal). If ``True``, transcripts
                remain lazy; each crop reads from disk. Slower but uses less peak RAM.
            detach_transcripts (bool): If ``True`` (default), transcripts are detached
                from ``data`` before the loop so that ``deepcopy()`` inside ``crop()``
                does not copy the full transcript array on every iteration. Cropping is
                then applied directly on the shared pandas DataFrame. Set to ``False``
                to let ``crop()`` handle transcripts normally (useful for benchmarking
                or if detaching causes unexpected side-effects).

        Returns:
            InSituExperiment: An instance containing the cropped data and metadata for the specified regions.
        """
        # Retrieve the regions dataframe
        region_df = data.regions[region_key]

        # check which region names are allowed
        if region_names is None:
            region_names = region_df["name"].tolist()
        else:
            # make sure region_names is a list
            region_names = convert_to_list(region_names)

        # When lazy=False, load transcripts into memory.
        # When detach_transcripts=True: compute directly to pandas and detach from
        # data so deepcopy() inside crop() does not copy the array each iteration.
        # Transcripts are restored to their original state in the finally-block.
        # When detach_transcripts=False: materialize into an in-memory Dask DF so
        # deepcopy is cheaper; data._transcripts is mutated (in-memory after return).
        transcripts_pdf = None
        original_transcripts = None
        if not lazy and data.transcripts is not None:
            warnings.warn(
                "Transcript data will be loaded into memory to speed up region cropping. "
                "This may require substantial RAM for large datasets.",
                stacklevel=2
            )
            if detach_transcripts:
                logger.info("Loading transcripts into memory...")
                transcripts_pdf = data._transcripts.compute()
                logger.info(f"Transcripts loaded: {len(transcripts_pdf):,} rows.")
                original_transcripts = data._transcripts
                data._transcripts = None                     # detach — crop() skips it
            else:
                data.materialize(layers=["transcripts"], verbose=True)

        # Initialize a new InSituExperiment object
        experiment = cls(data_type="insitupy")

        try:
            for n in tqdm(sorted(region_df["name"].tolist()), desc="Iterating regions"):
                if n in region_names:
                    cropped_data = data.crop(
                        region_tuple=(region_key, n),
                        materialize_transcripts=not detach_transcripts
                    )

                    # when detached, apply transcript crop separately on the shared pandas DF
                    if detach_transcripts and transcripts_pdf is not None:
                        shape = region_df[region_df["name"] == n]["geometry"].item()
                        cropped_data.transcripts = _crop_transcripts(
                            transcript_df=transcripts_pdf,
                            shape=shape,
                        )

                    # skip regions with no cells
                    if not cropped_data.cells.is_empty and cropped_data.cells.table.n_obs == 0:
                        logger.warning(f"Region '{n}' contains no cells and will be skipped.")
                        continue

                    # create metadata
                    metadata = {"region_key": region_key, "region_name": n}

                    # add to InSituExperiment
                    experiment.add(data=cropped_data, metadata=metadata)
        finally:
            # Restore detached transcripts so the caller's object is unchanged.
            if original_transcripts is not None:
                data._transcripts = original_transcripts

        return experiment

    @classmethod
    def read(cls,
             path: str | os.PathLike | Path,
               mode: Literal["insitupy", "spatialdata"] = "insitupy",
               filter_key: str | None = None) -> "InSituExperiment":
        """
        Read an InSituExperiment object from a specified folder.

        Args:
            path: Path to the experiment directory or SpatialData zarr store
            mode: Read mode - either "insitupy" (default) or "spatialdata"
                  Note: "spatialdata" mode is currently disabled and will be enabled in a future release.

        Returns:
            InSituExperiment object in the specified mode
        """
        if mode == "spatialdata":
            if not _SPATIALDATA_MODE_ENABLED:
                raise NotImplementedError(
                    "SpatialData mode is currently disabled and under development. "
                    "It will be enabled in a future release. "
                    "Please use mode='insitupy' for now."
                )
            return cls._read_spatialdata(path)
        elif mode == "insitupy":
            return cls._read_insitupy(path, filter_key=filter_key)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'insitupy' or 'spatialdata'")

    # ==================== SPATIALDATA MODE METHODS ====================

    @classmethod
    def _read_spatialdata(cls, path: str | os.PathLike | Path) -> "InSituExperiment":
        """
        Read an InSituExperiment from a SpatialData zarr store.

        This method reads a SpatialData zarr directory and creates an InSituExperiment
        containing StructuredSpatialData objects. It handles both single-sample and
        multi-sample SpatialData stores.

        Args:
            path: Path to the SpatialData .zarr directory

        Returns:
            InSituExperiment in SpatialData mode

        Raises:
            ImportError: If spatialdata is not installed
            FileNotFoundError: If the path does not exist
        """
        # Import for SpatialData mode
        try:
            import spatialdata

        except ImportError:
            raise ImportError(
                "This function requires the spatialdata-wrapper package. "
                "Install it with: pip install insitupy[spatialdata]"
            )
        else:
            from spatialdata_wrapper._io import silent_read_zarr as _silent_read_zarr

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SpatialData path not found: {path}")

        # Initialize experiment in spatialdata mode
        experiment = cls(data_type="spatialdata")
        experiment._path = path

        # Read the SpatialData zarr store
        logger.info(f"Reading SpatialData from {path}...")
        # sdata = read_zarr(path)
        sdata = _silent_read_zarr(path)

        # Extract samples from the SpatialData object
        samples = cls._extract_samples_from_spatialdata(sdata)

        logger.info(f"Found {len(samples)} sample(s)")

        # Create StructuredSpatialData for each sample
        for sample_id, sample_elements in tqdm(samples.items(), desc="Loading samples"):
            # Import from external package
            from spatialdata_wrapper import StructuredSpatialData

            struct_data = StructuredSpatialData()
            struct_data._path = path

            # Populate the StructuredSpatialData from elements
            cls._populate_structured_data(struct_data, sample_elements, sample_id)

            # Add to experiment
            experiment._data.append(struct_data)

            # Create metadata entry
            metadata_entry = {
                'uid': sample_id if sample_id != 'single' else str(uuid4()).split("-")[0],
            }
            experiment._metadata = pd.concat([
                experiment._metadata,
                pd.DataFrame([metadata_entry])
            ], ignore_index=True)

        # Try to load colors if they exist
        colors_path = path / "colors.json"
        if colors_path.exists():
            try:
                with open(colors_path) as f:
                    experiment._colors = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load colors.json: {e}")

        return experiment

    @staticmethod
    def _extract_samples_from_spatialdata(sdata) -> dict[str, dict]:
        """
        Group SpatialData elements by sample ID.

        Elements with keys like 'sample.<id>..MODALITY.name' are grouped by sample_id.
        Elements without sample prefix are treated as a single sample.

        Args:
            sdata: SpatialData object

        Returns:
            Dictionary mapping sample_id to dictionary of elements
        """
        samples = defaultdict(dict)
        has_multi_sample = False

        for elem_type, key, elem in sdata.gen_elements():
            if key.startswith(SAMPLE_STR):
                # Multi-sample format: 'sample.<id>..MODALITY...'
                parts = key.split('.')
                sample_id = parts[1]  # Extract sample ID
                samples[sample_id][key] = (elem_type, elem)
                has_multi_sample = True
            else:
                # Single-sample format or element without sample prefix
                samples['single'][key] = (elem_type, elem)

        # If we have multi-sample data, remove the 'single' key
        if has_multi_sample and 'single' in samples:
            if len(samples['single']) > 0:
                logger.warning(
                    "Found both multi-sample (with 'sample.' prefix) and single-sample elements. "
                    "Single-sample elements will be ignored."
                )
            del samples['single']

        return dict(samples)

    @staticmethod
    def _populate_structured_data(struct_data, sample_elements: dict, sample_id: str):
        """
        Populate a StructuredSpatialData object from a dictionary of elements.

        Args:
            struct_data: StructuredSpatialData object to populate
            sample_elements: Dictionary mapping element keys to (elem_type, elem) tuples
            sample_id: ID of the sample (used for filtering)
        """
        for key, (elem_type, elem) in sample_elements.items():
            # Parse the key to determine where to place the element
            # Remove sample prefix if present
            if key.startswith(SAMPLE_STR):
                # Format: 'sample.<id>..MODALITY.locators...'
                parts = key.split('.')
                # Remove 'sample', '<id>', and empty string
                parts = [p for p in parts[3:] if p]
            else:
                # Format: 'MODALITY.locators...'
                parts = key.split('.')

            if len(parts) == 0:
                logger.warning(f"Could not parse key: {key}")
                continue

            modality = parts[0]

            # Route to appropriate structure based on modality
            if modality == "IMAGES":
                if len(parts) >= 2:
                    image_name = parts[1]
                    # Get transformation for pixel size
                    try:
                        from spatialdata.transformations import get_transformation
                        scale_obj = get_transformation(elem)
                        struct_data._images.add_image(image_name, elem, scale_obj=scale_obj)
                    except Exception as e:
                        logger.warning(f"Could not add image {image_name}: {e}")

            elif modality == "CELLS":
                if len(parts) >= 2:
                    cell_key = parts[1]

                    # Initialize CellData if not exists
                    if cell_key not in struct_data._cells._layers:
                        from spatialdata_wrapper import StructuredCellData
                        struct_data._cells[cell_key] = StructuredCellData()

                    if len(parts) >= 3:
                        if parts[2] == "table":
                            struct_data._cells[cell_key].table = elem
                        elif parts[2] == "boundaries" and len(parts) >= 4:
                            boundary_name = parts[3]
                            struct_data._cells[cell_key].boundaries[boundary_name] = elem
                        elif parts[2] in ["circles", "circles_sized"]:
                            # Skip circles representations (derived from table)
                            pass

            elif modality == "TRANSCRIPTS":
                struct_data._transcripts = elem

            elif modality == "ANNOTATIONS":
                if len(parts) >= 2:
                    annotation_name = parts[1]
                    struct_data._annotations[annotation_name] = elem

            elif modality == "REGIONS":
                if len(parts) >= 2:
                    region_name = parts[1]
                    struct_data._regions[region_name] = elem

            else:
                logger.warning(f"Unknown modality in key: {key}")

    @staticmethod
    def _get_loaded_modalities_spatialdata(data) -> list[str]:
        """
        Get list of loaded modalities from a StructuredSpatialData object.

        Args:
            data: StructuredSpatialData object

        Returns:
            List of modality names that have data
        """
        loaded = []

        if not data._images.is_empty:
            loaded.append("images")
        if not data._cells.is_empty:
            loaded.append("cells")
        if data._transcripts is not None:
            loaded.append("transcripts")
        if not data._annotations.is_empty:
            loaded.append("annotations")
        if not data._regions.is_empty:
            loaded.append("regions")

        return loaded

    @classmethod
    def _read_insitupy(cls, path: str | os.PathLike | Path,
                       filter_key: str | None = None) -> "InSituExperiment":
        """
        Read an InSituExperiment in InSituPy format (original implementation).

        Args:
            path: Path to the InSituExperiment directory

        Returns:
            InSituExperiment in insitupy mode
        """
        path = Path(path)

        # Load metadata
        metadata = cls._read_metadata_with_schema(path)

        try:
            # load colors
            with open(path / "colors.json") as f:
                colors = json.load(f)
        except FileNotFoundError:
            colors = {}

        # Load filters (optional)
        filters = {}
        raw_composites: dict = {}
        filters_path = path / "filters.json"
        if filters_path.exists():
            try:
                with open(filters_path) as f:
                    filters_payload = json.load(f)
            except Exception as err:
                raise ValueError(
                    f"Could not load filters.json at '{filters_path}': {err}"
                ) from err

            if not isinstance(filters_payload, dict):
                raise ValueError(
                    "Invalid filters schema: expected a JSON object with keys 'version' and 'filters'."
                )

            version = filters_payload.get("version", None)
            filters = filters_payload.get("filters", None)

            if version not in _SUPPORTED_FILTER_VERSIONS:
                raise ValueError(
                    f"Unsupported filters schema version: {version}. "
                    f"Supported versions: {sorted(_SUPPORTED_FILTER_VERSIONS)}."
                )

            if not isinstance(filters, dict):
                raise ValueError(
                    "Invalid filters schema: 'filters' must be a dictionary mapping filter keys to filter entries."
                )

            if version == 2:
                raw_composites = filters_payload.get("composites", {}) or {}

        # Load each dataset
        data = []
        dataset_paths = sorted([elem for elem in path.glob("data-*") if elem.is_dir()])
        for dataset_path in tqdm(dataset_paths):
            dataset = InSituData.read(dataset_path, load_all=False)
            data.append(dataset)

        # Create a new InSituExperiment object
        experiment = cls(data_type="insitupy")
        experiment._metadata = metadata
        experiment._data = data
        experiment._path = path
        experiment._colors = colors
        experiment._filters = {}
        experiment._composites = {}

        # Backfill _uid from experiment metadata for legacy datasets (saved before uid feature)
        if "uid" in metadata.columns:
            backfilled = []
            for i, dataset in enumerate(data):
                if dataset._uid is None and i < len(metadata):
                    uid_val = metadata.iloc[i]["uid"]
                    if pd.notna(uid_val):
                        dataset._uid = uid_val
                        backfilled.append(i)
            if backfilled:
                warnings.warn(
                    f"{len(backfilled)} dataset(s) had no uid stored on disk and were "
                    f"backfilled from the experiment metadata. Call .save_geometries() on each dataset to persist the uids.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate and store filters
        if filters:
            for name, entry in filters.items():
                spec = FilterSpec.from_entry(name, entry)
                mask = spec.mask
                note = spec.note
                mask_arr = np.asarray(mask, dtype=bool)
                if len(mask_arr) != len(metadata):
                    warnings.warn(
                        f"Filter '{name}' length ({len(mask_arr)}) does not match metadata length "
                        f"({len(metadata)}). Skipping this filter.",
                        UserWarning,
                        stacklevel=2
                    )
                    continue
                experiment._filters[name] = {
                    "mask": mask_arr.tolist(),
                    "note": note,
                }

        for name, entry in raw_composites.items():
            try:
                comp = CompositeFilterSpec.from_entry(name, entry)
                experiment._composites[name] = comp.to_dict()
            except (ValueError, KeyError) as err:
                warnings.warn(
                    f"Could not load composite filter '{name}': {err}. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )

        if filter_key is not None:
            if filter_key not in experiment._filters and filter_key not in experiment._composites:
                raise KeyError(
                    f"Filter '{filter_key}' not found. "
                    f"Available filters: {list(experiment.filters.keys())}"
                )
            experiment = experiment.filters.apply(filter_key)

        return experiment

    def _build_filters_payload(self) -> dict[str, Any]:
        """Build versioned JSON payload for ``filters.json``."""
        payload: dict[str, Any] = {
            "version": _FILTERS_SCHEMA_VERSION,
            "filters": {},
            "composites": {},
        }

        for key, entry in self._filters.items():
            spec = FilterSpec.from_entry(key, entry)
            payload["filters"][key] = spec.to_dict()

        for key, entry in self._composites.items():
            comp = CompositeFilterSpec.from_entry(key, entry)
            payload["composites"][key] = comp.to_dict()

        return payload

    def _check_mode_compatibility(self, method_name: str):
        """
        Check if the current mode is compatible with a method.
        Raises NotImplementedError for spatialdata mode (for now).
        """
        if self._data_type == "spatialdata":
            raise NotImplementedError(
                f"Method '{method_name}' is not yet implemented for SpatialData mode. "
                f"This will be added in a future update."
            )
    def _check_obs_uniqueness(
        self,
        cells_layer: str | None = None
        ):
        """
        Check if the observation names are unique across all datasets.

        Args:
            cells_layer (Optional[str]): The layer in `xd.cells` to access. Defaults to None.

        Raises:
            Warning: If observation names are not unique across all datasets.
        """
        # get obs dataframes
        obs_list = []
        for _, d in self.iterdata():
            if not d.cells.is_empty:
                celldata = _get_cell_layer(cells=d.cells, cells_layer=cells_layer)
                obs_list.append(celldata.table.obs)

        # concatenate the obs dataframes
        if not obs_list:
            return
        all_obs = pd.concat(obs_list, axis=0, ignore_index=False)
        if not all_obs.index.is_unique:
            logger.warning("Observation names are not unique across all datasets.")

    def _create_categorical_color_dict(
        self,
        obs_col: str,
        cells_layer: str | None = None,
        palette: ListedColormap = DEFAULT_CATEGORICAL_CMAP
        ) -> dict:
        """Create a color dictionary for categorical data."""
        cols = []
        for _, xd in self.iterdata():
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            if obs_col in celldata.table.obs.columns:
                if celldata.table.obs[obs_col].isna().all():
                    raise ValueError(f"Column '{obs_col}' in obs contains only NaNs. Cannot create color dictionary.")
                cols.append(np.unique(celldata.table.obs[obs_col]))

        if len(cols) > 0:
            all_cats = np.sort(np.unique(np.concatenate(cols)))

            # create color dict
            color_dict = map_to_colors(all_cats, palette=palette)
            return color_dict
        else:
            return None

    def calculate_qc_metrics(
        self,
        cells_layer: str | None = None,
        layer: str = None,
        force_layer: bool = False,
        add_to_metadata: bool = True,
        return_metrics: bool = False,
    ) -> dict | None:
        """
        Calculate quality control metrics for the InSituExperiment.

        Args:
            cells_layer: The layer of cells to use. Defaults to None.
            layer: The layer of the AnnData object to use for calculations.
                If None, uses adata.X or 'counts' layer if X is not integer counts.
            force_layer: Whether to use specified layer even if not integer counts.
            add_to_metadata: Whether to add metrics to exp._metadata. Default True.
            return_metrics: Whether to return metrics as dict. Default False.

        Returns:
            If return_metrics is True, returns dict with 'median_genes_per_cell'
            and 'median_transcripts_per_cell' lists. Otherwise returns None.
        """
        from insitupy.utils._checks import _calculate_single_metrics

        median_genes = []
        median_transcripts = []
        num_cells = []

        for _, dataset in self.iterdata():
            if dataset.cells.is_empty:
                logger.warning("Cells were not loaded. Loading cells.")
                dataset.load_cells()

            celldata = _get_cell_layer(cells=dataset.cells, cells_layer=cells_layer)
            m_genes, m_transcripts = _calculate_single_metrics(
                celldata.table, layer=layer, force_layer=force_layer
            )
            median_genes.append(m_genes)
            median_transcripts.append(m_transcripts)
            num_cells.append(celldata.table.n_obs)

        # Create column names with optional cells_layer suffix
        suffix = f" ('{cells_layer}')" if cells_layer else ""
        genes_col = f"median_genes_per_cell{suffix}"
        transcripts_col = f"median_transcripts_per_cell{suffix}"
        cells_col = f"num_cells{suffix}"

        if add_to_metadata:
            self._metadata[genes_col] = median_genes
            self._metadata[transcripts_col] = median_transcripts
            self._metadata[cells_col] = num_cells

        if return_metrics:
            return {
                genes_col: median_genes,
                transcripts_col: median_transcripts,
                cells_col: num_cells,
            }


class InSituExperimentView(InSituExperiment):
    """Lightweight linked view of an InSituExperiment subset.

    Dataset references are shared with the parent experiment. Mutating
    ``view._data[i]`` (or any attribute reached through it, such as
    ``view.cells[...]``, ``xd.cells.table``, or annotation/region containers)
    mutates the parent's state in place. Methods that mutate dataset internals
    on a view — for example ``sync_colors``, ``import_from_anndata``,
    ``import_from_table``, ``calculate_metrics``, ``save_cells`` — propagate to
    the parent. This is deliberate (a view is a lightweight filter, not a copy);
    use ``view.saveas(path)`` to materialise an independent copy.
    """

    @property
    def is_view(self) -> bool:
        """Return True; this object is a linked view of a parent experiment."""
        return True

    def build_table(self, *args, **kwargs):
        """Raise NotImplementedError — views cannot build their own table.

        ``build_table`` writes to the experiment's save directory.  A view
        shares its save path with the parent, so writing would corrupt the
        parent's table.  Build the table on the parent experiment and access
        the view-filtered result via ``view.table[<layer>]``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "build_table() is not supported on an InSituExperimentView. "
            "Call build_table() on the parent experiment and read the result "
            "via view.table[<layer>]."
        )

    def add(self, *args, **kwargs):
        """Raise NotImplementedError — datasets cannot be added to a view.

        A view is a linked subset of the parent experiment.  Adding a dataset
        would extend the parent's ``_data`` without updating the parent's
        ``_metadata`` and ``_filters`` in a consistent way.  Call ``add()``
        on the parent experiment directly.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "add() is not supported on an InSituExperimentView. "
            "Call add() on the parent experiment instead."
        )

    def reload(self, skip=None, verbose=True):
        """Reload only the datasets included in this view from disk.

        Unlike the base ``reload``, this override does **not** re-read the
        experiment-level metadata, colors, or filters from the parent path.
        Those files describe the full experiment; overwriting the view's
        in-memory slices with the full-experiment data would desync the view.

        Args:
            skip: Modality name(s) forwarded to each dataset's reload.
            verbose: If True, log progress.
        """
        if verbose:
            logger.info(
                "Reloading %d dataset(s) in view (skipping experiment-level files).",
                len(self._data),
            )
        for xd in self._data:
            xd.reload(skip=skip, verbose=False)

    def copy(self):
        """Return a standalone deep copy of this view as a plain InSituExperiment.

        Unlike the base ``copy`` (which returns another view), this override
        returns an independent ``InSituExperiment`` with deep-copied datasets,
        metadata, colors, and filters.  Mutating the returned object does not
        affect the parent experiment.

        Returns:
            InSituExperiment: A deep copy of this view's content.
        """
        new = InSituExperiment(data_type=self._data_type)
        new._data = [deepcopy(d) for d in self._data]
        new._metadata = self._metadata.reset_index(drop=True).copy()
        new._colors = deepcopy(self._colors)
        new._filters = deepcopy(self._filters)
        new._composites = deepcopy(self._composites)
        new._applied_filters = []
        new._parent_indices = None
        new._path = None
        return new

    def save_metadata(
        self,
        path: str | os.PathLike | Path | None = None,
        overwrite: bool = True,
    ):
        """Save view metadata by merging changes back into the full on-disk metadata.

        Rather than writing only the view's subset rows (which would silently drop all
        non-selected datasets from the file), this method:

        1. Loads the full metadata from the parent experiment path (``self.path``).
        2. Updates matching rows using the ``uid`` column.
        3. Adds any new columns that exist in the view but not on disk, filling
           non-view rows with ``pd.NA``.
        4. Writes the complete merged metadata to ``path`` (or ``self.path`` if
           ``path`` is None).

        Args:
            path: Directory where metadata files should be written.
                If None, uses ``self.path`` (the parent experiment path).
            overwrite: If True, overwrite existing metadata files.

        Raises:
            ValueError: If neither ``path`` nor ``self.path`` is set, or if the
                on-disk metadata has no ``uid`` column (legacy format).
            FileExistsError: If metadata files exist and ``overwrite`` is False.
        """
        if self.path is None:
            raise ValueError(
                "Cannot save view metadata: the parent experiment path is not set. "
                "Load the experiment from disk before calling save_metadata() on a view."
            )

        # Load the authoritative full metadata from the parent path.
        on_disk = self._read_metadata_with_schema(self.path)

        if "uid" not in on_disk.columns:
            raise ValueError(
                "The on-disk metadata has no 'uid' column. "
                "This experiment was saved with an older version of InSituPy that did not "
                "assign UIDs. Re-add all datasets and save the full experiment first."
            )
        if "uid" not in self._metadata.columns:
            raise ValueError(
                "The view metadata has no 'uid' column and cannot be safely merged."
            )

        # Merge: update only the rows present in this view, keyed by uid.
        on_disk_idx = on_disk.set_index("uid")
        view_idx = self._metadata.set_index("uid")

        # Add columns that are new in the view (e.g., from add_metadata_column).
        for col in view_idx.columns:
            if col not in on_disk_idx.columns:
                on_disk_idx[col] = pd.NA

        on_disk_idx.update(view_idx)
        merged = on_disk_idx.reset_index()

        # Resolve write path.
        write_path = Path(path) if path is not None else self.path
        write_path.mkdir(parents=True, exist_ok=True)

        parquet_path = write_path / _METADATA_PARQUET_FILENAME
        csv_path = write_path / "metadata.csv"

        if (parquet_path.exists() or csv_path.exists()) and not overwrite:
            raise FileExistsError(
                "Metadata file(s) already exist. "
                f"Found: {parquet_path} and/or {csv_path}. "
                "Set `overwrite=True` to replace them."
            )

        tmp_path = write_path / "metadata.parquet.tmp"
        merged.to_parquet(tmp_path, index=False)
        tmp_path.replace(parquet_path)

        tmp_csv_path = write_path / "metadata.csv.tmp"
        with open(tmp_csv_path, "w", newline="") as f:
            f.write("# AUTO-GENERATED — human-readable export only; edits are ignored (canonical data is in metadata.parquet)\n")
            merged.to_csv(f, index=True)
        tmp_csv_path.replace(csv_path)

        stale_schema = self._metadata_schema_path(write_path)
        if stale_schema.exists():
            stale_schema.unlink()

    def save_colors(
        self,
        path: str | os.PathLike | Path | None = None,
        overwrite: bool = True,
    ):
        """Save view colors by merging with the parent's on-disk ``colors.json``.

        Loads the parent's existing color dict from ``self.path``, merges the
        view's colors on top (view wins on shared keys), and writes the result.
        This prevents view-only ``sync_colors`` calls from silently deleting
        color entries for datasets not included in the view.

        Args:
            path: Directory where ``colors.json`` should be written.
                If None, uses ``self.path``.
            overwrite: If True, overwrite an existing ``colors.json``.

        Raises:
            ValueError: If ``self.path`` is not set.
            FileExistsError: If ``colors.json`` exists and ``overwrite`` is False.
        """
        if self.path is None:
            raise ValueError(
                "Cannot save view colors: parent experiment path is not set."
            )

        on_disk_path = self.path / "colors.json"
        if on_disk_path.exists():
            on_disk = read_json(on_disk_path)
        else:
            on_disk = {}

        merged = {**on_disk, **self._colors}

        write_path = Path(path) if path is not None else self.path
        write_path.mkdir(parents=True, exist_ok=True)
        colors_path = write_path / "colors.json"

        if colors_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {colors_path}. Set `overwrite=True` to replace it."
            )

        with open(colors_path, "w") as f:
            json.dump(merged, f)

    def save_filters(
        self,
        path: str | os.PathLike | Path | None = None,
        overwrite: bool = True,
    ):
        """Save view filters by merging masks back into the parent's ``filters.json``.

        Loads the full filter payload from the parent path (``self.path``),
        splices the view's per-filter masks into the full-length parent masks at
        the positions recorded in ``_parent_indices``, and writes the merged
        result. This mirrors the behaviour of :meth:`save_metadata` and prevents
        view-sliced (shorter) masks from being written back to the parent file.

        New filters created on the view are added with ``False`` for all rows
        outside the view. Composite filters are merged by key.

        Args:
            path: Directory where ``filters.json`` should be written.
                If None, uses ``self.path``.
            overwrite: If True, overwrite an existing ``filters.json``.

        Raises:
            ValueError: If ``self.path`` is not set, or if ``_parent_indices``
                is missing (view was not created through ``_subset``).
            FileExistsError: If ``filters.json`` exists and ``overwrite`` is False.
        """
        if self.path is None:
            raise ValueError(
                "Cannot save view filters: parent experiment path is not set."
            )

        parent_idx = getattr(self, "_parent_indices", None)
        if parent_idx is None:
            raise ValueError(
                "Cannot save view filters: _parent_indices is not set. "
                "Recreate the view using exp[...] or exp.filters.view(...) "
                "to enable filter merge-back."
            )

        write_path = Path(path) if path is not None else self.path
        write_path.mkdir(parents=True, exist_ok=True)

        filters_json_path = write_path / "filters.json"
        if filters_json_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {filters_json_path}. Set `overwrite=True` to replace it."
            )

        # Load on-disk filter payload from parent path.
        on_disk_path = self.path / "filters.json"
        if on_disk_path.exists():
            payload = read_json(on_disk_path)
        else:
            payload = {
                "version": _FILTERS_SCHEMA_VERSION,
                "filters": {},
                "composites": {},
            }

        if "filters" not in payload:
            payload["filters"] = {}
        if "composites" not in payload:
            payload["composites"] = {}
        payload["version"] = _FILTERS_SCHEMA_VERSION

        n_parent = len(self._read_metadata_with_schema(self.path))

        for key, entry in self._filters.items():
            view_mask = np.asarray(FilterSpec.from_entry(key, entry).mask, dtype=bool)
            existing = payload["filters"].get(key)
            if existing is not None:
                existing_arr = np.asarray(existing["mask"], dtype=bool)
                full = existing_arr.copy() if len(existing_arr) == n_parent else np.zeros(n_parent, dtype=bool)
            else:
                full = np.zeros(n_parent, dtype=bool)
            full[parent_idx] = view_mask
            payload["filters"][key] = {
                "mask": full.tolist(),
                "note": entry.get("note"),
            }

        for key, entry in self._composites.items():
            payload["composites"][key] = CompositeFilterSpec.from_entry(key, entry).to_dict()

        with open(filters_json_path, "w") as f:
            json.dump(payload, f)

    def saveas(
        self,
        path: str | os.PathLike | Path,
        overwrite: bool = False,
        verbose: bool = False,
        collect_warnings_mode: bool = True,
        free_after_save: bool = False,
        **kwargs,
    ):
        """Export this view to a standalone InSituExperiment at *path*.

        Materialises the view into a plain :class:`InSituExperiment` (copying
        the view's datasets, metadata, colors, and filters) and delegates to the
        base :meth:`~InSituExperiment.saveas`. The resulting directory is a
        self-contained experiment with ``len(view)`` datasets and
        correctly-sized filter masks.

        ``self._path`` is **not** mutated. To work with the exported experiment,
        re-read it from *path*::

            view.saveas("/path/to/export")
            exported = InSituExperiment.read("/path/to/export")

        Args:
            path: Destination directory.
            overwrite: If True, overwrite an existing directory at *path*.
            verbose: If True, print verbose output.
            collect_warnings_mode: Collect and print warnings after save.
            free_after_save: **Not supported on a view** — must be False. A view
                shares its datasets with the parent experiment, so releasing
                their in-memory data would also empty the parent. Passing True
                raises ``ValueError``; materialise first
                (``view.copy().saveas(path, free_after_save=True)``) instead.
            **kwargs: Forwarded to :meth:`InSituData.saveas` for each dataset.

        Raises:
            ValueError: If ``free_after_save=True`` (unsupported on a view).
        """
        if free_after_save:
            raise ValueError(
                "free_after_save=True is not supported on an InSituExperimentView. "
                "A view shares its datasets with the parent experiment, so releasing "
                "their in-memory data would also empty the parent. To export with "
                "memory release, materialise the view first: "
                "view.copy().saveas(path, free_after_save=True)."
            )

        materialised = InSituExperiment(data_type=self._data_type)
        materialised._data = list(self._data)
        materialised._metadata = self._metadata.reset_index(drop=True).copy()
        materialised._colors = deepcopy(self._colors)
        materialised._filters = deepcopy(self._filters)
        materialised._composites = deepcopy(self._composites)
        materialised._applied_filters = []
        materialised._parent_indices = None

        # Snapshot child _paths: the base saveas (with R8) repaths each dataset
        # object's _path to the export location. Because materialised shares
        # those objects with the parent, we must restore the originals so the
        # parent's save()/path guard keeps working.
        original_child_paths = [xd._path for xd in self._data]
        try:
            materialised.saveas(
                path,
                overwrite=overwrite,
                verbose=verbose,
                collect_warnings_mode=collect_warnings_mode,
                free_after_save=free_after_save,
                **kwargs,
            )
        finally:
            for xd, original in zip(self._data, original_child_paths):
                xd._path = original

    @property
    def table(self) -> "ViewTableAccessor":
        """Dict-like accessor for per-cells-layer concatenated tables (view-filtered).

        Returns an accessor that loads the AnnData for a specific layer and
        row-filters it to only the samples present in this view::

            view.table["main"]   # AnnData filtered to this view's samples
            view.table.keys()    # available layer names (from parent path)

        Requires the parent experiment to have called :meth:`build_table` first.

        .. note::
            This feature is experimental and may change in future versions.
        """
        return ViewTableAccessor(self)

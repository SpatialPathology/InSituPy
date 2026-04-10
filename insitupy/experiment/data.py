import json
import logging
import os
import shutil
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
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

from insitupy._constants import (DEFAULT_CATEGORICAL_CMAP, LOAD_FUNCS,
                                 MODALITIES, MODALITIES_ABBR, SAMPLE_STR,
                                 with_insitupy_style)
from insitupy._core.data import InSituData
from insitupy._exceptions import ModalityNotFoundError
from insitupy._io.files import check_overwrite_and_remove_if_true
from insitupy._logging import WarningCollector, collect_warnings
from insitupy._textformat import textformat as tf
from insitupy.containers._utils import _get_cell_layer
from insitupy.experiment.filters import FilterManager, FilterSpec
from insitupy.io.data import read_xenium
from insitupy.palettes import map_to_colors
from insitupy.utils._adata import _select_anndata_elements
from insitupy.utils.utils import (_crop_transcripts, convert_to_list,
                                  get_nrows_maxcols, remove_empty_subplots)

logger = logging.getLogger(__name__)

# Feature flag for SpatialData mode
# Set to True to enable spatialdata mode functionality
# Currently disabled while the feature is under development
_SPATIALDATA_MODE_ENABLED = False
_FILTERS_SCHEMA_VERSION = 1
_METADATA_SCHEMA_VERSION = 1
_METADATA_SCHEMA_FILENAME = "metadata.schema.json"

# Sentinel value to detect when 'by' is not explicitly provided
_UNSET = object()


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

        self._metadata = pd.DataFrame(columns=['uid', 'slide_id', 'sample_id'])
        self._data = []  # Can hold either InSituData or StructuredSpatialData
        self._path = None
        self._colors = {}
        self._filters = {}
        self._applied_filters: List[str] = []
        self._data_type = data_type

    def __repr__(self):
        """
        Provide a string representation of the InSituExperiment object.

        Returns:
            str: A string summarizing the InSituExperiment object, including the number of samples
            and a table of metadata with loaded modalities.
        """
        # extract metadata
        mdf = self._metadata.copy()
        num_samples = len(mdf)

        # Add data type indicator
        mode_str = f" ({tf.Bold}{self._data_type}{tf.ResetAll} mode)"

        # check which modalities are loaded and add information as string to the copied metadata dataframe
        loaded_list = []
        for _, data in self.iterdata():
            if self._data_type == "insitupy":
                loaded_modalities = data.get_loaded_modalities()
            else:  # spatialdata mode
                loaded_modalities = self._get_loaded_modalities_spatialdata(data)
            loaded_string = "".join(["+" if m in loaded_modalities else "-" for m in MODALITIES])
            loaded_list.append(loaded_string)
        mdf.insert(1, MODALITIES_ABBR, loaded_list)

        # generate string summary
        sample_summary = mdf.to_string(index=True, col_space=4, max_colwidth=15, max_cols=10)
        object_name = "InSituExperimentView" if self.is_view else "InSituExperiment"
        filters_info = ""
        if self.applied_filters:
            filters_info = f"\nApplied filters: {' -> '.join(self.applied_filters)}"

        return (f"{tf.Bold}{object_name}{tf.ResetAll}{mode_str} with {num_samples} samples:{filters_info}\n"
                f"{sample_summary}")

    @property
    def is_view(self) -> bool:
        """Return False; this is the base experiment, not a view."""
        return False

    @property
    def applied_filters(self) -> List[str]:
        """Return the list of filter labels that have been applied to this experiment."""
        return list(self._applied_filters)

    def _subset(
        self,
        key,
        as_view: bool = False,
        added_filter: Optional[str] = None,
    ):
        """
        Internal helper to subset experiment data and metadata.

        Args:
            key: Subsetting key (same accepted types as ``__getitem__``).
            as_view: If True, keep path linkage and return an InSituExperimentView.
            added_filter: Optional filter key to append to applied filter history.
        """
        if isinstance(key, int):
            if key > (len(self) - 1):
                raise IndexError(f"Index ({key}) is out of range {len(self)}.")
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

        # Keep linkage only for view objects
        if as_view:
            new_experiment._path = self._path
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
            data: Union[str, os.PathLike, Path, InSituData],
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
                dataset = InSituData.read(data)
            elif mode == "xenium":
                dataset = read_xenium(data)
            else:
                raise ValueError("Invalid mode. Supported modes are 'insitupy' and 'xenium'.")

        # checks whether dataset is an instance of InSituData or any subclass of it, and avoids issues with direct object identity comparison
        if dataset.__class__ is not InSituData:
            raise TypeError(f"Loaded dataset is not an InSituData object. Instead: '{dataset.__class__}'")

        # Add the dataset to the data collection
        self._data.append(dataset)

        # Create a new DataFrame for the new metadata
        new_metadata = {
            'uid': str(uuid4()).split("-")[0],
            'slide_id': dataset.slide_id,
            'sample_id': dataset.sample_id
        }

        # add information from metadata argument
        new_metadata.update(metadata)

        # convert to dataframe
        new_metadata = pd.DataFrame([new_metadata])

        # Concatenate the new metadata with the existing metadata
        self._metadata = pd.concat([self._metadata, new_metadata], axis=0, ignore_index=True)


    def add_metadata_column(
        self,
        column_name: str,
        values: Union[List, str, pd.Series, np.ndarray],
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
                        new_metadata: Union[pd.DataFrame, dict, str, os.PathLike, Path],
                        by: Optional[str] = _UNSET,
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
        index: Union[int, List[int], List[bool], slice, range, np.ndarray, pd.Series],
        column_name: str,
        values: Union[Any, List, pd.Series, np.ndarray]
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

        Reads :attr:`~insitupy._core.data.InSituData.slide_id` and
        :attr:`~insitupy._core.data.InSituData.sample_id` from every dataset in
        :attr:`data` and overwrites the corresponding values in :attr:`metadata`.
        All other metadata columns are left untouched.

        Call this method after changing ``slide_id`` or ``sample_id`` on one or more
        :class:`~insitupy._core.data.InSituData` objects that belong to this experiment.

        Examples:
            >>> exp.data[0].slide_id = '0005405'
            >>> exp.update_metadata()
        """
        changed = 0
        for i, dataset in enumerate(self._data):
            old_slide = self._metadata.at[i, 'slide_id']
            old_sample = self._metadata.at[i, 'sample_id']
            new_slide = dataset.slide_id
            new_sample = dataset.sample_id
            if old_slide != new_slide or old_sample != new_sample:
                self._metadata.at[i, 'slide_id'] = new_slide
                self._metadata.at[i, 'sample_id'] = new_sample
                changed += 1
        if changed:
            logger.info("update_metadata: synced slide_id/sample_id for %d dataset(s).", changed)
        else:
            logger.info("update_metadata: all slide_id/sample_id values already in sync.")

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
        ref_id: Optional[Union[int, List[int], Literal["rest"]]] = None,
        target_annotation_tuple: Optional[Tuple[str, str]] = None,
        target_cell_type_tuple: Optional[Tuple[str, str]] = None,
        target_region_tuple: Optional[Tuple[str, str]] = None,
        ref_annotation_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
        ref_cell_type_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
        ref_region_tuple: Optional[Union[Literal["rest", "same"], Tuple[str, str]]] = "same",
        method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 't-test',
        exclude_ambiguous_assignments: bool = False,
        force_assignment: bool = False,
        name_col: Optional[str] = "uid",
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
        cells_layer: Optional[str] = None
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


    def import_from_anndata(
        self,
        adata: AnnData,
        uid_column: str,
        uid_column_adata: str,
        obs_columns_to_transfer: Optional[List[str]] = None,
        obsm_keys_to_transfer: Optional[List[str]] = None,
        cells_layer: Optional[str] = None,
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
                    stacklevel=2
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
                        stacklevel=2
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


    def iterdata(self):
        """
        Iterate over the metadata rows and corresponding data.

        Yields:
            tuple: A tuple containing the metadata row as a Series and the corresponding data.
        """
        for idx, row in self._metadata.iterrows():
            yield row, self._data[idx]


    def to_anndata(
        self,
        cells_layer: Optional[str] = None,
        label_col: str = "uid",
        obs_keys: Optional[Union[List[str], str, Literal["all"]]] = None,
        var_keys: Optional[Union[List[str], str, Literal["all"]]] = None,
        obsm_keys: Optional[Union[List[str], str, Literal["all"]]] = "spatial",
        varm_keys: Optional[Union[List[str], str, Literal["all"]]] = None,
        uns_keys: Optional[Union[List[str], str, Literal["all"]]] = None,
        layer_keys: Optional[Union[List[str], str, Literal["all"]]] = None,
        metadata_keys: Optional[Union[List[str], str, Literal["all"]]] = None,
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

        # Validate label_col exists in metadata
        if label_col not in self._metadata.columns:
            raise ValueError(
                f"Column '{label_col}' not found in metadata. "
                f"Available columns: {list(self._metadata.columns)}"
            )

        adatas: Dict[Any, anndata.AnnData] = {}

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
            join='inner',
            label=label_col,
            merge="unique"
        )

        # Move label_col to first position in obs columns
        if label_col in adata_concat.obs.columns:
            cols = [label_col] + [col for col in adata_concat.obs.columns if col != label_col]
            adata_concat.obs = adata_concat.obs[cols]

        return adata_concat


    def load_all(self,
                 skip: Optional[str] = None,
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
                    names: Union[Literal["all", "nuclei"], str] = "all",
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
        cells_layer: Optional[str] = None,
        color: Optional[str] = None,
        title_column: Optional[str] = None,
        title_size: int = 24,
        max_cols: int = 4,
        figsize: Tuple[int, int] = (8,6),
        savepath: Optional[Union[str, os.PathLike, Path]] = None,
        save_only: bool = False,
        show: bool = True,
        fig: Optional[Figure] = None,
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
        cells_layer: Optional[str] = None,
        color: Optional[str] = None,
        title_column: Optional[str] = None,
        title_size: int = 20,
        max_cols: int = 4,
        figsize: Tuple[int, int] = (8, 6),
        savepath: Optional[Union[str, os.PathLike, Path]] = None,
        save_only: bool = False,
        show: bool = True,
        fig: Optional[Figure] = None,
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
        skip: Optional[List] = None,
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

        # Reload experiment metadata
        metadata_path = path / "metadata.csv"
        if metadata_path.exists():
            self._metadata = self._read_metadata_with_schema(path)
            if verbose:
                logger.info("Reloaded metadata, colors, and filters from disk.")

        # Reload colors
        colors_path = path / "colors.json"
        if colors_path.exists():
            with open(colors_path, 'r') as f:
                self._colors = json.load(f)
        else:
            self._colors = {}

        # Reload filters
        self._filters = {}
        self._applied_filters = []
        filters_path = path / "filters.json"
        if filters_path.exists():
            try:
                with open(filters_path, 'r') as f:
                    filters_payload = json.load(f)
                version = filters_payload.get("version", None)
                filters = filters_payload.get("filters", None)
                if version != _FILTERS_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported filters schema version: {version}. "
                        f"Expected version {_FILTERS_SCHEMA_VERSION}."
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
            except Exception as err:
                warnings.warn(f"Could not reload filters.json: {err}", UserWarning, stacklevel=2)

    def unload(self, modalities: Optional[List] = None):
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

        Args:
            verbose: If True, print verbose output for dataset-level save operations.
            collect_warnings_mode: If True, collect warnings and print a summary at end
                instead of displaying them inline (prevents progress bar disruption).
            **kwargs: Additional keyword arguments passed to ``InSituData.save()``.

        Raises:
            ValueError: If no experiment save path is available or dataset paths are inconsistent.
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

        if collect_warnings_mode:
            with collect_warnings() as collector:
                for xd in tqdm(self._data):
                    xd.save(verbose=verbose, **kwargs)
            collector.print_summary()
        else:
            for xd in tqdm(self._data):
                xd.save(verbose=verbose, **kwargs)

        self.save_metadata(overwrite=True)
        self.save_colors(overwrite=True)
        self.save_filters(path=self.path)

    def save_metadata(
        self,
        path: Optional[Union[str, os.PathLike, Path]] = None,
        overwrite: bool = True,
    ):
        """Save experiment metadata to ``metadata.csv`` and ``metadata.schema.json``.

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
        metadata_path = path / "metadata.csv"
        metadata_schema_path = self._metadata_schema_path(path)

        if (metadata_path.exists() or metadata_schema_path.exists()) and not overwrite:
            raise FileExistsError(
                "Metadata file(s) already exist. "
                f"Found: {metadata_path} and/or {metadata_schema_path}. "
                "Set `overwrite=True` to replace them."
            )

        self._metadata.to_csv(metadata_path, index=True)
        self._save_metadata_schema(path, self._metadata)

    @staticmethod
    def _metadata_schema_path(path: Path) -> Path:
        """Return the path to metadata schema sidecar file."""
        return path / _METADATA_SCHEMA_FILENAME

    @staticmethod
    def _metadata_dtype_map(metadata: pd.DataFrame) -> Dict[str, str]:
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
    def _load_metadata_dtype_map(cls, path: Path) -> Optional[Dict[str, str]]:
        """Load metadata dtype map if available and valid, else return None."""
        schema_path = cls._metadata_schema_path(path)
        if not schema_path.exists():
            return None

        try:
            with open(schema_path, 'r') as f:
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
        """Read metadata.csv and restore dtypes from optional schema sidecar."""
        metadata_path = path / "metadata.csv"
        dtype_map = cls._load_metadata_dtype_map(path)

        # Force string-like columns at CSV parse time to preserve values such as
        # leading-zero IDs before post-load casting is applied.
        read_csv_kwargs: Dict[str, Any] = {}
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
            return metadata

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

        return metadata

    def save_colors(
        self,
        path: Optional[Union[str, os.PathLike, Path]] = None,
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

        with open(colors_path, 'w') as f:
            json.dump(self.colors, f)

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
                    xd.save(
                        verbose=dataset_verbose,
                        sync_images=True,
                        images_only=True,
                        overwrite_images=overwrite,
                        **kwargs,
                    )
            collector.print_summary()
        else:
            for xd in tqdm(self._data):
                xd.save(
                    verbose=dataset_verbose,
                    sync_images=True,
                    images_only=True,
                    overwrite_images=overwrite,
                    **kwargs,
                )



    def saveas(
        self,
        path: Union[str, os.PathLike, Path],
        overwrite: bool = False,
        verbose: bool = False,
        collect_warnings_mode: bool = True,
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
            **kwargs: Additional keyword arguments passed to dataset.saveas().
        """
        self._check_mode_compatibility("saveas")

        # Create the main directory if it doesn't exist
        path = Path(path)

        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)

        logger.info(f"Saving InSituExperiment to {str(path)}") if verbose else None

        if collect_warnings_mode:
            with collect_warnings() as collector:
                # Iterate over the datasets and save each one in a numbered subfolder
                for index, dataset in enumerate(tqdm(self._data)):
                    subfolder_path = path / f"data-{str(index).zfill(3)}"
                    dataset.saveas(subfolder_path, verbose=False, **kwargs)

            # Print collected warnings at the end
            collector.print_summary()
        else:
            # Original behavior - warnings shown inline
            for index, dataset in enumerate(tqdm(self._data)):
                subfolder_path = path / f"data-{str(index).zfill(3)}"
                dataset.saveas(subfolder_path, verbose=False, **kwargs)

        self._path = path
        self.save_metadata(path=path, overwrite=True)
        self.save_colors(path=path, overwrite=True)
        self.save_filters(path=path)

        logger.info("Saved.") if verbose else None

    def save_filters(
        self,
        path: Optional[Union[str, os.PathLike, Path]] = None,
    ):
        """
        Save only experiment filters to ``filters.json``.

        Args:
            path: Directory where ``filters.json`` should be written.
                If None, uses ``self.path``.

        Raises:
            ValueError: If neither ``path`` nor ``self.path`` is set.
        """
        if path is None:
            if self.path is None:
                raise ValueError(
                    "No save path available. Provide `path` or set `self.path` first (e.g. via `saveas`)."
                )
            path = self.path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        filters_payload = self._build_filters_payload()
        with open(path / "filters.json", 'w') as f:
            json.dump(filters_payload, f)

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

    def show_modality(self, modality, uid_column: str = "sample_id"):
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
        keys: Union[str, List[str]],
        cells_layer: Optional[str] = None,
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
            overwrite (bool, optional): Whether to overwrite existing color dictionaries. Defaults to False.
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
        path: Optional[Union[str, os.PathLike, Path]] = None,
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
                ``path``, the experiment-level files (``metadata.csv``,
                ``colors.json``, ``filters.json``) are written there, and the
                original experiment root directories are removed.  This is
                disk-efficient because no data is duplicated.

                .. warning::
                    ``mode="move"`` is **destructive and irreversible**.
                    All source experiments must have a save path set and must
                    reside on the same filesystem as ``path``.

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
            for i, obj in enumerate(objs):
                if not isinstance(obj, InSituExperiment):
                    raise TypeError("All objects must be instances of InSituExperiment.")
                if obj._path is None:
                    raise ValueError(
                        f"mode='move' requires all experiments to have a save path set, "
                        f"but experiment at index {i} has no path. "
                        f"Call saveas() first."
                    )

            path = Path(path)

            # Verify same filesystem: compare the device of the destination
            # parent (creating it if needed) against each source dataset.
            path.mkdir(parents=True, exist_ok=True)
            dst_dev = os.stat(path).st_dev
            for obj in objs:
                src_dev = os.stat(obj._path).st_dev
                if src_dev != dst_dev:
                    raise ValueError(
                        f"mode='move' requires source and destination to be on the "
                        f"same filesystem. Source '{obj._path}' and destination '{path}' "
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
        """
        new_experiment = cls(data_type=data_type)
        new_metadata: list = []
        merged_colors: dict = {}
        global_idx = 0

        for key, obj in zip(keys, objs):
            # Release in-memory data before moving so the move loop stays clean
            obj.unload()

            for xd in tqdm(obj._data, desc=f"Moving datasets from {obj._path.name}"):
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

        # Remove source experiment roots (datasets have already been moved out)
        for obj in objs:
            remaining = list(obj._path.iterdir())
            if remaining:
                logger.info(
                    "Removing source experiment root '%s' (%d item(s) remaining).",
                    obj._path, len(remaining),
                )
            shutil.rmtree(str(obj._path))
            obj._path = None

        if data_type == "insitupy":
            new_experiment._check_obs_uniqueness()

        return new_experiment

    @classmethod
    def from_config(cls,
                    config_path: Union[str, os.PathLike, Path],
                    mode: Literal["insitupy", "xenium"],
                    collect_warnings_mode: bool = True,
                    **kwargs
                    ):
        """Create an InSituExperiment object from a configuration file.

        Args:
            config_path (Union[str, os.PathLike, Path]): The path to the configuration CSV or Excel file.
            mode (Literal["insitupy", "xenium"]): The mode to use for loading the datasets.
                - "insitupy": Load previously saved InSituPy projects using :meth:`~insitupy._core.data.InSituData.read`.
                - "xenium": Load Xenium data bundles directly using :func:`~insitupy.io.read_xenium`.
            collect_warnings_mode (bool): If True, collect warnings during loading and print a summary at the end.
                This keeps the progress bar clean while still showing important warnings. Defaults to True.
        """
        config_path = Path(config_path)

        # Determine file type and read the configuration file
        if config_path.suffix in ['.csv']:
            config = pd.read_csv(config_path, dtype=str)
        elif config_path.suffix in ['.xlsx', '.xls']:
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

            # Use collect_warnings context manager to capture warnings without disrupting progress bar
            if collect_warnings_mode:
                with collect_warnings(warning_collector):
                    if mode == "insitupy":
                        dataset = InSituData.read(dataset_path)
                    elif mode == "xenium":
                        dataset = read_xenium(dataset_path, verbose=False, **kwargs)
                    else:
                        raise ValueError("Invalid mode. Supported modes are 'insitupy' and 'xenium'.")
            else:
                if mode == "insitupy":
                    dataset = InSituData.read(dataset_path)
                elif mode == "xenium":
                    dataset = read_xenium(dataset_path, verbose=False, **kwargs)
                else:
                    raise ValueError("Invalid mode. Supported modes are 'insitupy' and 'xenium'.")

            experiment._data.append(dataset)

            # Extract metadata from the row, excluding the 'directory' column
            metadata = row.drop(labels=['directory']).to_dict()
            metadata['uid'] = str(uuid4()).split("-")[0]
            metadata['slide_id'] = dataset.slide_id
            metadata['sample_id'] = dataset.sample_id

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
                    region_names: Optional[Union[List[str], str]] = None,
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
             path: Union[str, os.PathLike, Path],
               mode: Literal["insitupy", "spatialdata"] = "insitupy",
               filter_key: Optional[str] = None) -> "InSituExperiment":
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
    def _read_spatialdata(cls, path: Union[str, os.PathLike, Path]) -> "InSituExperiment":
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
            from spatialdata_wrapper._io import \
                silent_read_zarr as _silent_read_zarr

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
                'slide_id': sample_id if sample_id != 'single' else 'unknown',
                'sample_id': sample_id if sample_id != 'single' else 'unknown',
            }
            experiment._metadata = pd.concat([
                experiment._metadata,
                pd.DataFrame([metadata_entry])
            ], ignore_index=True)

        # Try to load colors if they exist
        colors_path = path / "colors.json"
        if colors_path.exists():
            try:
                with open(colors_path, 'r') as f:
                    experiment._colors = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load colors.json: {e}")

        return experiment

    @staticmethod
    def _extract_samples_from_spatialdata(sdata) -> Dict[str, Dict]:
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
    def _populate_structured_data(struct_data, sample_elements: Dict, sample_id: str):
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
                        from spatialdata.transformations import \
                            get_transformation
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
    def _get_loaded_modalities_spatialdata(data) -> List[str]:
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
    def _read_insitupy(cls, path: Union[str, os.PathLike, Path],
                       filter_key: Optional[str] = None) -> "InSituExperiment":
        """
        Read an InSituExperiment in InSituPy format (original implementation).

        Args:
            path: Path to the InSituExperiment directory

        Returns:
            InSituExperiment in insitupy mode
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.csv"
        metadata = cls._read_metadata_with_schema(path)

        try:
            # load colors
            with open(path / "colors.json", 'r') as f:
                colors = json.load(f)
        except FileNotFoundError:
            colors = {}

        # Load filters (optional)
        filters = {}
        filters_path = path / "filters.json"
        if filters_path.exists():
            try:
                with open(filters_path, 'r') as f:
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

            if version != _FILTERS_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported filters schema version: {version}. "
                    f"Expected version {_FILTERS_SCHEMA_VERSION}."
                )

            if not isinstance(filters, dict):
                raise ValueError(
                    "Invalid filters schema: 'filters' must be a dictionary mapping filter keys to filter entries."
                )

        # Load each dataset
        data = []
        dataset_paths = sorted([elem for elem in path.glob("data-*") if elem.is_dir()])
        for dataset_path in tqdm(dataset_paths):
            dataset = InSituData.read(dataset_path)
            data.append(dataset)

        # Create a new InSituExperiment object
        experiment = cls(data_type="insitupy")
        experiment._metadata = metadata
        experiment._data = data
        experiment._path = path
        experiment._colors = colors
        experiment._filters = {}

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

        if filter_key is not None:
            if filter_key not in experiment._filters:
                raise KeyError(
                    f"Filter '{filter_key}' not found. Available filters: {list(experiment._filters.keys())}"
                )
            experiment = experiment.filters.apply(filter_key)

        return experiment

    def _build_filters_payload(self) -> Dict[str, Any]:
        """Build versioned JSON payload for ``filters.json``."""
        payload: Dict[str, Any] = {
            "version": _FILTERS_SCHEMA_VERSION,
            "filters": {},
        }

        for key, entry in self._filters.items():
            spec = FilterSpec.from_entry(key, entry)
            payload["filters"][key] = spec.to_dict()

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
        cells_layer: Optional[str] = None
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
        cells_layer: Optional[str] = None,
        palette: ListedColormap = DEFAULT_CATEGORICAL_CMAP
        ) -> Dict:
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
        cells_layer: Optional[str] = None,
        layer: str = None,
        force_layer: bool = False,
        add_to_metadata: bool = True,
        return_metrics: bool = False,
    ) -> Optional[Dict]:
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
    """Lightweight linked view of an InSituExperiment subset."""

    @property
    def is_view(self) -> bool:
        """Return True; this object is a linked view of a parent experiment."""
        return True
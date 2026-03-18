import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pandas as pd
import toml

logger = logging.getLogger(__name__)


@dataclass
class DiffExprConfigCollector:
    """Dataclass collecting all parameters that define a differential expression run.

    Stores target and reference group specifications alongside the analysis
    mode (``"single-cell"`` or ``"pseudobulk"``) and method parameters, so
    that results can be faithfully reproduced and provenance is recorded.
    Serialisable to/from TOML via :meth:`save_as_toml` / :meth:`read_from_toml`.
    """
    # General
    mode: Literal["single-cell", "pseudobulk"]
    method_params: dict
    cells_layer: Optional[str] = None
    exclude_ambiguous_assignments: Optional[bool] = None

    # Target
    target_annotation: Optional[str] = None
    target_cell_type: Optional[str] = None
    target_region: Optional[str] = None
    target_cell_number: Optional[int] = None
    target_name: Optional[str] = None
    target_metadata: Dict[str, Any] = None

    # Reference
    ref_annotation: Optional[str] = None
    ref_cell_type: Optional[str] = None
    ref_region: Optional[str] = None
    ref_cell_number: Optional[int] = None
    ref_name: Optional[str] = None
    ref_metadata: Dict[str, Any] = None

    # class variables
    GENERAL_FIELDS = ["mode", "method_params", "cells_layer", "exclude_ambiguous_assignments"]
    TARGET_FIELDS = ["target_annotation", "target_cell_type", "target_region",
                     "target_cell_number", "target_name", "target_metadata"]
    REFERENCE_FIELDS = ["ref_annotation", "ref_cell_type", "ref_region",
                        "ref_cell_number", "ref_name", "ref_metadata"]

    def __post_init__(self):
        # Validate string fields
        for field_name in ["target_annotation", "target_cell_type", "target_region", "target_name",
                           "ref_annotation", "ref_cell_type", "ref_region", "ref_name",
                           "mode", "cells_layer"]:
            field = getattr(self, field_name)
            if not isinstance(field, str) and field is not None:
                raise TypeError(f"{field_name} must be a string. Instead got {type(getattr(self, field_name))}.")

        # Validate integer fields
        for field_name in ["target_cell_number", "ref_cell_number"]:
            field = getattr(self, field_name)
            if field is not None:
                if not isinstance(field, Integral) or field < 0:
                    raise ValueError(f"{field_name} must be a non-negative integer. Instead: {field} with type {type(field)}.")

        # Validate boolean fields
        for field_name in ["exclude_ambiguous_assignments"]:
            field = getattr(self, field_name)
            if not isinstance(field, bool) and field is not None:
                raise TypeError(f"{field_name} must be a boolean. Instead got {type(getattr(self, field_name))}.")

        # Validate dict fields
        for field_name in ["method_params", "target_metadata", "ref_metadata"]:
            field = getattr(self, field_name)
            if not isinstance(field, dict) and field is not None:
                raise TypeError(f"{field_name} must be a dictionary. Instead got {type(getattr(self, field_name))}.")


    def __repr__(self):
        config = self.to_dict()
        lines = ["DiffExprConfigCollector("]
        for section, values in config.items():
            lines.append(f"  {section}:")
            for key, value in values.items():
                if isinstance(value, dict):
                    value = f"Dictionary with following keys: {value.keys()}"
                lines.append(f"    {key}: {value}")
        lines.append(")")
        return "\n".join(lines)

    def to_dict(self):
        """Serialise config to a nested dict with General / Target / Reference sections.

        Returns:
            A ``dict`` with three top-level keys (``"General"``, ``"Target"``,
            ``"Reference"``), each mapping field names to their values.  NumPy
            scalar types are converted to native Python types.
        """
        def convert(value):
            """Convert NumPy scalars to native Python types."""
            return value.item() if hasattr(value, "item") else value

        config_dict = {
            "General": {
                field: convert(getattr(self, field)) for field in self.GENERAL_FIELDS
            },
            "Target": {
                field.replace("target_", ""): convert(getattr(self, field)) for field in self.TARGET_FIELDS
            },
            "Reference": {
                field.replace("ref_", ""): convert(getattr(self, field)) for field in self.REFERENCE_FIELDS
            }
        }

        return config_dict

    def save_as_toml(self, filepath: Union[str, os.PathLike, Path]):
        """Serialise this config to a TOML file.

        Args:
            filepath: Output file path.  The file is created or overwritten.
        """
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            toml.dump(config_dict, f)

    @classmethod
    def read_from_toml(cls, filepath: Union[str, os.PathLike, Path]) -> "DiffExprConfigCollector":
        """Load a :class:`DiffExprConfigCollector` from a TOML file.

        Args:
            filepath: Path to a TOML file previously written by
                :meth:`save_as_toml`.

        Returns:
            A new :class:`DiffExprConfigCollector` instance populated with
            the values from the file.
        """
        with open(filepath, 'r') as f:
            config = toml.load(f)

        # Flatten the nested config dictionary
        flat_config = {}

        # General section
        for field in cls.GENERAL_FIELDS:
            try:
                flat_config[field] = config["General"][field]
            except KeyError:
                flat_config[field] = None

        # Target section
        for field in cls.TARGET_FIELDS:
            key = field.replace("target_", "")
            try:
                flat_config[field] = config["Target"][key]
            except KeyError:
                flat_config[field] = None

        # Reference section
        for field in cls.REFERENCE_FIELDS:
            key = field.replace("ref_", "")
            try:
                flat_config[field] = config["Reference"][key]
            except KeyError:
                flat_config[field] = None

        return cls(**flat_config)


class DiffExprResults:
    """
    Container for pseudobulk differential gene expression (DGE) results.

    Attributes
    ----------
    main : pd.DataFrame
        DGE results comparing condition A vs. condition B for the selected cell type.
    target_neighborhood : Optional[pd.DataFrame]
        DGE results comparing condition A cells vs. their neighboring cells (if neighborhood data used).
    ref_neighborhood : Optional[pd.DataFrame]
        DGE results comparing condition B cells vs. their neighboring cells (if neighborhood data used).
    config : dict
        Optional metadata about the analysis (e.g., cell type, setup tuple, parameters).
    """

    def __init__(
        self,
        main: pd.DataFrame,
        config: DiffExprConfigCollector,
        target_neighborhood: Optional[pd.DataFrame] = None,
        ref_neighborhood: Optional[pd.DataFrame] = None,
    ):
        self.main = main
        self.target_neighborhood = target_neighborhood
        self.ref_neighborhood = ref_neighborhood
        self.config = config

        # check columns
        required_columns = {"log2foldchange", "padj"}
        self._validate_df(self.main, "main", required_columns)
        if self.target_neighborhood is not None:
            self._validate_df(self.target_neighborhood, "target_neighborhood", required_columns)
        if self.ref_neighborhood is not None:
            self._validate_df(self.ref_neighborhood, "ref_neighborhood", required_columns)

    def __repr__(self):
        return f"<DiffExprResults main={len(self.main)} genes, neighbors={self.has_neighbors()}>"

    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """Return all results in a dictionary for easy iteration."""
        results = {"main": self.main}
        if self.target_neighborhood is not None:
            results["target_neighborhood"] = self.target_neighborhood
        if self.ref_neighborhood is not None:
            results["ref_neighborhood"] = self.ref_neighborhood
        return results

    def has_neighbors(self) -> bool:
        """Return True if neighborhood results are available."""
        return self.target_neighborhood is not None and self.ref_neighborhood is not None


    @classmethod
    def read(cls, directory: Union[str, os.PathLike, Path]) -> "DiffExprResults":
        """
        Read saved differential expression results and metadata from a directory.

        Parameters
        ----------
        directory : str
            Path to the directory containing saved results.

        Returns
        -------
        DiffExprResults
            Reconstructed instance from saved files.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")

        # Load main results
        main_path = os.path.join(directory, "main.csv")
        if not os.path.isfile(main_path):
            raise FileNotFoundError(f"Main results file '{main_path}' not found.")
        main = pd.read_csv(main_path, index_col=0)

        # Load neighbor results if available
        target_nb_path = os.path.join(directory, "target_neighborhood.csv")
        ref_nb_path = os.path.join(directory, "ref_neighborhood.csv")
        target_neighborhood = pd.read_csv(target_nb_path, index_col=0) if os.path.isfile(target_nb_path) else None
        ref_neighborhood = pd.read_csv(ref_nb_path, index_col=0) if os.path.isfile(ref_nb_path) else None

        # Load metadata
        metadata_path = Path(directory) / "config.toml"
        config = {}
        if metadata_path.is_file():
            config = DiffExprConfigCollector.read_from_toml(metadata_path)

        return cls(
            main=main,
            target_neighborhood=target_neighborhood,
            ref_neighborhood=ref_neighborhood,
            config=config
        )


    def save(
        self,
        directory: Union[str, os.PathLike, Path],
        overwrite: bool = False):
        """
        Save all results and metadata to the specified directory.

        Parameters
        ----------
        directory : str
            Path to the directory where results should be saved.
        overwrite : bool
            If True, overwrite the directory if it already exists.
        """
        directory = Path(directory)
        if directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory '{directory}' already exists. "
                    "Set `overwrite=True` to overwrite its contents."
                )
            else:
                logger.warning(f"Overwriting existing directory '{directory}'.")
                shutil.rmtree(directory)

        directory.mkdir(exist_ok=True)

        # Save main results
        self.main.to_csv(directory / "main.csv", index=True)

        # Save neighbor results if available
        if self.target_neighborhood is not None:
            self.target_neighborhood.to_csv(directory / "target_neighborhood.csv", index=True)
        if self.ref_neighborhood is not None:
            self.ref_neighborhood.to_csv(directory / "ref_neighborhood.csv", index=True)

        # Save metadata
        self.config.save_as_toml(directory / "config.toml")

    def summary(self) -> str:
        """Return a quick summary of available results."""
        lines = [f"Main DGE results: {len(self.main)} genes"]
        if self.has_neighbors():
            lines.append(f"Neighbor comparison (A): {len(self.target_neighborhood)} genes")
            lines.append(f"Neighbor comparison (B): {len(self.ref_neighborhood)} genes")
        if self.config:
            lines.append(f"Configuration:\n{self.config.__repr__}")
        return "\n".join(lines)

    def _validate_df(self, df: pd.DataFrame, name: str, required: set):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"The '{name}' DataFrame is missing following mandatory columns: {', '.join(missing)}. "
                f"Expected at least following columns: {', '.join(required)}."
            )



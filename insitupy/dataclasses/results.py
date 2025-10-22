import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class DiffExprResults:
    """
    Container for pseudobulk differential gene expression (DGE) results.

    Attributes
    ----------
    main : pd.DataFrame
        DGE results comparing condition A vs. condition B for the selected cell type.
    nb_condition_a : Optional[pd.DataFrame]
        DGE results comparing condition A cells vs. their neighboring cells (if neighborhood data used).
    nb_condition_b : Optional[pd.DataFrame]
        DGE results comparing condition B cells vs. their neighboring cells (if neighborhood data used).
    metadata : dict
        Optional metadata about the analysis (e.g., cell type, setup tuple, parameters).
    """
    main: pd.DataFrame
    nb_condition_a: Optional[pd.DataFrame] = None
    nb_condition_b: Optional[pd.DataFrame] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        required_columns = {"log2FoldChange", "pvalue"}
        self._validate_df(self.main, "main", required_columns)
        if self.nb_condition_a is not None:
            self._validate_df(self.nb_condition_a, "nb_condition_a", required_columns)
        if self.nb_condition_b is not None:
            self._validate_df(self.nb_condition_b, "nb_condition_b", required_columns)

    def __repr__(self):
        return f"<DiffExprResults main={len(self.main)} genes, neighbors={self.has_neighbors()}>"

    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """Return all results in a dictionary for easy iteration."""
        results = {"main": self.main}
        if self.nb_condition_a is not None:
            results["nb_condition_a"] = self.nb_condition_a
        if self.nb_condition_b is not None:
            results["nb_condition_b"] = self.nb_condition_b
        return results

    def has_neighbors(self) -> bool:
        """Return True if neighborhood results are available."""
        return self.nb_condition_a is not None and self.nb_condition_b is not None


    @classmethod
    def read(cls, directory: str) -> "DiffExprResults":
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
        nb_a_path = os.path.join(directory, "nb_condition_a.csv")
        nb_b_path = os.path.join(directory, "nb_condition_b.csv")
        nb_condition_a = pd.read_csv(nb_a_path, index_col=0) if os.path.isfile(nb_a_path) else None
        nb_condition_b = pd.read_csv(nb_b_path, index_col=0) if os.path.isfile(nb_b_path) else None

        # Load metadata
        metadata_path = os.path.join(directory, "metadata.json")
        metadata = {}
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        return cls(
            main=main,
            nb_condition_a=nb_condition_a,
            nb_condition_b=nb_condition_b,
            metadata=metadata
        )


    def save(self, path: str, overwrite: bool = False):
        """
        Save all results and metadata to the specified directory.

        Parameters
        ----------
        path : str
            Path to the directory where results should be saved.
        overwrite : bool
            If True, overwrite the directory if it already exists.
        """
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(
                    f"Directory '{path}' already exists. "
                    "Set `overwrite=True` to overwrite its contents."
                )
            else:
                print(f"Warning: Overwriting existing directory '{path}'.")
                shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)

        # Save main results
        self.main.to_csv(os.path.join(path, "main.csv"), index=True)

        # Save neighbor results if available
        if self.nb_condition_a is not None:
            self.nb_condition_a.to_csv(os.path.join(path, "nb_condition_a.csv"), index=True)
        if self.nb_condition_b is not None:
            self.nb_condition_b.to_csv(os.path.join(path, "nb_condition_b.csv"), index=True)

        # Save metadata
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

    def summary(self) -> str:
        """Return a quick summary of available results."""
        lines = [f"Main DGE results: {len(self.main)} genes"]
        if self.has_neighbors():
            lines.append(f"Neighbor comparison (A): {len(self.nb_condition_a)} genes")
            lines.append(f"Neighbor comparison (B): {len(self.nb_condition_b)} genes")
        if self.metadata:
            lines.append(f"Metadata: {self.metadata}")
        return "\n".join(lines)

    def _validate_df(self, df: pd.DataFrame, name: str, required: set):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"The '{name}' DataFrame is missing following mandatory columns: {', '.join(missing)}. "
                f"Expected at least following columns: {', '.join(required)}."
            )




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

    def has_neighbors(self) -> bool:
        """Return True if neighborhood results are available."""
        return self.nb_condition_a is not None and self.nb_condition_b is not None

    def summary(self) -> str:
        """Return a quick summary of available results."""
        lines = [f"Main DGE results: {len(self.main)} genes"]
        if self.has_neighbors():
            lines.append(f"Neighbor comparison (A): {len(self.nb_condition_a)} genes")
            lines.append(f"Neighbor comparison (B): {len(self.nb_condition_b)} genes")
        if self.metadata:
            lines.append(f"Metadata: {self.metadata}")
        return "\n".join(lines)

    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """Return all results in a dictionary for easy iteration."""
        results = {"main": self.main}
        if self.nb_condition_a is not None:
            results["nb_condition_a"] = self.nb_condition_a
        if self.nb_condition_b is not None:
            results["nb_condition_b"] = self.nb_condition_b
        return results

    def __repr__(self):
        return f"<DiffExprResults main={len(self.main)} genes, neighbors={self.has_neighbors()}>"

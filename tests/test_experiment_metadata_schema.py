import json

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype

from insitupy.experiment.data import InSituExperiment


def test_metadata_schema_preserves_string_ids(tmp_path):
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "patient_id": pd.Series(["P0001", "P0456"], dtype="string"),
            "n_cells": pd.Series([10, 20], dtype="Int64"),
        }
    )

    exp.save_metadata(path=tmp_path)

    schema_path = tmp_path / "metadata.schema.json"
    assert schema_path.exists()

    payload = json.loads(schema_path.read_text())
    assert payload["column_dtypes"]["patient_id"] == "string"

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    assert out["patient_id"].tolist() == ["P0001", "P0456"]
    assert is_string_dtype(out["patient_id"])
    assert is_integer_dtype(out["n_cells"])


def test_metadata_csv_only_remains_loadable(tmp_path):
    metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "n_cells": [10, 20],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv")

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    # Legacy CSV-only folders are still accepted and loaded via pandas inference.
    assert is_integer_dtype(out["n_cells"])


def test_legacy_metadata_slide_sample_id_discarded(tmp_path):
    """Legacy metadata.csv with slide_id/sample_id loads without error; those columns are dropped."""
    metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "slide_id": ["0000001", "0000456"],
            "sample_id": ["s1", "s2"],
            "n_cells": [10, 20],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv")

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    assert "slide_id" not in out.columns
    assert "sample_id" not in out.columns
    assert "n_cells" in out.columns
    assert "uid" in out.columns

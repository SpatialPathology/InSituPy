import json

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype

from insitupy.experiment.data import InSituExperiment


def test_metadata_schema_preserves_string_ids(tmp_path):
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "slide_id": pd.Series(["0000001", "0000456"], dtype="string"),
            "n_cells": pd.Series([10, 20], dtype="Int64"),
        }
    )

    exp.save_metadata(path=tmp_path)

    schema_path = tmp_path / "metadata.schema.json"
    assert schema_path.exists()

    payload = json.loads(schema_path.read_text())
    assert payload["column_dtypes"]["slide_id"] == "string"

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    assert out["slide_id"].tolist() == ["0000001", "0000456"]
    assert is_string_dtype(out["slide_id"])
    assert is_integer_dtype(out["n_cells"])


def test_metadata_csv_only_remains_loadable(tmp_path):
    metadata = pd.DataFrame(
        {
            "uid": ["sample-1", "sample-2"],
            "slide_id": ["0000001", "0000456"],
            "n_cells": [10, 20],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv")

    reloaded = InSituExperiment._read_insitupy(tmp_path)
    out = reloaded.metadata

    # Legacy CSV-only folders are still accepted and loaded via pandas inference.
    assert out["slide_id"].tolist() == [1, 456]
    assert is_integer_dtype(out["slide_id"])
    assert is_integer_dtype(out["n_cells"])

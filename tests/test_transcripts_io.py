"""Transcripts are saved as one Parquet file with fixed-size row groups (format v2, viewer
milestone M3) and read back with one dask partition per row group."""

import math

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import insitupy.containers.io as cio
from insitupy import InSituData
from insitupy._io._xenium import _restructure_transcripts_dataframe
from tests.spatialdata_fixtures import make_insitudata


def _frame(n, n_genes, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x_location": rng.random(n),
        "y_location": rng.random(n),
        "z_location": rng.random(n),
        "feature_name": pd.Categorical([f"gene_{i % n_genes}" for i in range(n)]),
        "cell_id": [f"cell_{i}" for i in range(n)],
    })


def _two_partition_transcripts(tmp_path):
    """Two independently written parts: the categorical gets an int8 index in the first
    (5 categories) and an int16 index in the second (300 categories)."""
    parts = tmp_path / "parts"
    parts.mkdir()
    frames = [_frame(5, 5, 0), _frame(300, 300, 1)]
    for i, df in enumerate(frames):
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parts / f"part.{i}.parquet")
    return dd.read_parquet(parts), frames


def _saveas_and_reload(xd, path, mode="dask"):
    xd.saveas(path, overwrite=True)
    back = InSituData.read(path)
    back.load_transcripts(mode=mode)
    return back


def _sorted(df):
    df = df.copy()
    df["feature_name"] = df["feature_name"].astype(str)
    return df.sort_values("x_location").reset_index(drop=True)


def test_single_file_with_row_groups_reads_one_partition_per_group(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "_TRANSCRIPT_ROW_GROUP_SIZE", 7)
    xd = make_insitudata(n_cells=4)
    xd.transcripts, frames = _two_partition_transcripts(tmp_path)

    back = _saveas_and_reload(xd, tmp_path / "proj")

    file = tmp_path / "proj" / "transcripts" / "transcripts.parquet"
    assert file.is_file()
    assert not (file.parent / "transcripts.parquet.__ispy_tmp__").exists()
    n_groups = pq.ParquetFile(file).metadata.num_row_groups
    assert n_groups == sum(math.ceil(len(f) / 7) for f in frames)
    # without split_row_groups dask reads the whole file as one partition
    assert back.transcripts.npartitions == n_groups

    got = back.transcripts.compute()
    assert isinstance(got["feature_name"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(
        _sorted(got), _sorted(pd.concat(frames)), check_dtype=False, check_categorical=False,
    )


def test_partitions_are_streamed_and_empty_ones_skipped(tmp_path, monkeypatch):
    df = _frame(30, 4, 2)
    df["x_location"] = np.arange(30, dtype=float)
    ddf = dd.from_pandas(df, npartitions=3)
    # drop every row of the last partition
    ddf = ddf[ddf.x_location < 20]
    n_nonempty = sum(len(p) > 0 for p in ddf.partitions)

    calls = []
    original = pq.ParquetWriter.write_table

    def counting(self, table, *args, **kwargs):
        calls.append(table.num_rows)
        return original(self, table, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", counting)
    xd = make_insitudata(n_cells=4)
    xd.transcripts = ddf
    back = _saveas_and_reload(xd, tmp_path / "proj")

    # one write per non-empty partition: never the whole frame at once
    assert len(calls) == n_nonempty == 2
    assert sum(calls) == 20
    assert len(back.transcripts.compute()) == 20


def test_all_partitions_empty_writes_a_loadable_file(tmp_path):
    ddf = dd.from_pandas(_frame(10, 3, 3), npartitions=2)
    xd = make_insitudata(n_cells=4)
    xd.transcripts = ddf[ddf.x_location > 2]  # no rows

    back = _saveas_and_reload(xd, tmp_path / "proj")

    got = back.transcripts.compute()
    assert len(got) == 0
    assert list(got.columns) == list(ddf.columns)


def test_object_columns_take_their_type_from_the_data(tmp_path):
    # object columns (dask string conversion off): the meta only holds placeholder strings
    df = pd.DataFrame({"x_location": [1.0, 2.0, 3.0, 4.0], "raw": [b"a", b"b", b"c", b"d"],
                       "note": [None, None, "q", "r"]})
    with dask.config.set({"dataframe.convert-string": False}):
        ddf = dd.from_pandas(df, npartitions=2)
    file = tmp_path / "t.parquet"
    cio._write_dask_transcripts(ddf, file)

    schema = pq.read_schema(file)
    assert schema.field("raw").type == pa.binary()
    # all-null in the first partition: keeps the string type of the meta
    assert schema.field("note").type == pa.string()
    pd.testing.assert_frame_equal(pd.read_parquet(file), df)


def test_named_dask_index_is_kept(tmp_path):
    df = _frame(12, 3, 4)
    df.index = pd.Index([f"t{i}" for i in range(12)], name="transcript_id")
    xd = make_insitudata(n_cells=4)
    xd.transcripts = dd.from_pandas(df, npartitions=2)

    back = _saveas_and_reload(xd, tmp_path / "proj")

    got = back.transcripts.compute()
    assert got.index.name == "transcript_id"
    assert sorted(got.index) == sorted(df.index)


def test_multiindex_columns_are_refused_before_writing(tmp_path):
    # the restructured pandas-mode Xenium layout: dask could not read such a file back,
    # so the saved project would not open
    raw = _frame(20, 4, 5)
    raw["transcript_id"] = np.arange(20, dtype=np.uint64) + 1000
    xd = make_insitudata(n_cells=4)
    xd.transcripts = _restructure_transcripts_dataframe(raw)

    with pytest.raises(ValueError, match="MultiIndex columns"):
        xd.saveas(tmp_path / "proj")
    assert not (tmp_path / "proj").exists()
    assert not (tmp_path / "proj.__ispy_tmp__").exists()


def test_pandas_transcripts_keep_named_index(tmp_path, monkeypatch):
    df = _frame(10, 3, 5)
    df.index = pd.Index(np.arange(10, dtype=np.int64) + 1000, name="transcript_id")
    xd = make_insitudata(n_cells=4)
    xd.transcripts = df
    first = _saveas_and_reload(xd, tmp_path / "proj", mode="pandas")
    assert isinstance(first.transcripts, pd.DataFrame)

    # the pandas branch of the writer
    monkeypatch.setattr(cio, "_TRANSCRIPT_ROW_GROUP_SIZE", 4)
    back = _saveas_and_reload(first, tmp_path / "proj2", mode="pandas")

    file = tmp_path / "proj2" / "transcripts" / "transcripts.parquet"
    assert pq.ParquetFile(file).metadata.num_row_groups == 3
    got = back.transcripts.sort_index()
    assert got.index.name == "transcript_id"
    np.testing.assert_array_equal(got.index, df.index)
    np.testing.assert_allclose(got["x_location"], df["x_location"])
    assert list(got["feature_name"].astype(str)) == list(df["feature_name"].astype(str))

"""Tests for the dataset download/unpack logic (no network, no real cache)."""
import shutil
import zipfile
from pathlib import Path

import pytest

from insitupy.datasets import datasets
from insitupy.datasets.datasets import _data_check_and_download, _md5sum


def _make_zip(path: Path, files: dict) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return _md5sum(path)


@pytest.fixture
def source_zip(tmp_path):
    """Zip with an `outs/` layout, as the 10x bundles have."""
    src = tmp_path / "source" / "bundle.zip"
    src.parent.mkdir()
    md5 = _make_zip(src, {"outs/a.txt": "A", "outs/sub/b.txt": "B"})
    return src, md5


@pytest.fixture
def paths(tmp_path):
    named = tmp_path / "cache" / "dataset"
    named.mkdir(parents=True)
    return named, named / "output-data", named / "bundle.zip"


@pytest.fixture
def fake_download(monkeypatch, source_zip):
    """Replace download_url by a copy of the local zip; records the calls."""
    src, _ = source_zip
    calls = []

    def _fake(url, out_dir=".", file_name=None, chunk_size=65536, overwrite=False):
        calls.append(url)
        shutil.copy(src, Path(out_dir) / Path(url).name)

    monkeypatch.setattr(datasets, "download_url", _fake)
    return calls


def test_existing_valid_zip_without_data_dir_is_unpacked(paths, source_zip, fake_download):
    named, data_dir, zip_file = paths
    src, md5 = source_zip
    shutil.copy(src, zip_file)

    _data_check_and_download(data_dir, zip_file, md5, False, "http://x/bundle.zip", named)

    assert fake_download == []  # zip reused, no download
    assert (data_dir / "a.txt").read_text() == "A"
    assert (data_dir / "sub" / "b.txt").read_text() == "B"
    assert not (data_dir / "outs").exists()
    assert not zip_file.exists()


def test_leftover_partial_dir_is_cleaned_up(paths, source_zip, fake_download):
    named, data_dir, zip_file = paths
    _, md5 = source_zip
    partial = named / f".{data_dir.name}.partial"
    partial.mkdir()
    (partial / "stale.txt").write_text("stale")

    _data_check_and_download(data_dir, zip_file, md5, False, "http://x/bundle.zip", named)

    assert (data_dir / "a.txt").exists()
    assert not (data_dir / "stale.txt").exists()
    assert not partial.exists()


def test_overwrite_replaces_existing_data_dir(paths, source_zip, fake_download):
    named, data_dir, zip_file = paths
    _, md5 = source_zip
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("old")
    (data_dir / "old_only.txt").write_text("old")

    _data_check_and_download(data_dir, zip_file, md5, True, "http://x/bundle.zip", named)

    assert (data_dir / "a.txt").read_text() == "A"
    assert not (data_dir / "old_only.txt").exists()


def test_md5_mismatch_after_download_raises_and_cleans_up(paths, fake_download):
    named, data_dir, zip_file = paths

    with pytest.raises(ValueError, match="md5sum"):
        _data_check_and_download(data_dir, zip_file, "0" * 32, False, "http://x/bundle.zip", named)

    assert not data_dir.exists()
    assert not zip_file.exists()
    assert not (named / f".{data_dir.name}.partial").exists()


def test_existing_data_dir_skips_download(paths, source_zip, fake_download):
    named, data_dir, zip_file = paths
    _, md5 = source_zip
    data_dir.mkdir()
    (data_dir / "keep.txt").write_text("keep")

    _data_check_and_download(data_dir, zip_file, md5, False, "http://x/bundle.zip", named)

    assert fake_download == []
    assert (data_dir / "keep.txt").exists()

"""Tests for InSituExperiment.__repr__ and InSituData.__repr__."""
import pandas as pd
from insitupy.experiment.data import InSituExperiment


def _make_exp(n=25):
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({
        "uid": [f"s{i}" for i in range(n)],
        "condition": ["A"] * (n // 2) + ["B"] * (n - n // 2),
    })
    return exp


def test_empty():
    exp = InSituExperiment()
    r = repr(exp)
    assert "InSituExperiment" in r
    assert "Path:" in r
    assert "data" in r
    assert "0 samples" in r
    assert "filters" in r
    assert "table" in r


def test_with_samples():
    exp = _make_exp(25)
    r = repr(exp)
    # sample count and column names on separate lines
    lines = r.splitlines()
    assert any("25 samples" in l and "metadata" not in l for l in lines), "samples line should be standalone"
    assert any("metadata columns:" in l for l in lines)
    assert '"uid"' in r
    assert '"condition"' in r


def test_column_wrapping():
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({f"very_long_column_name_{i}": [1] for i in range(15)})
    r = repr(exp)
    assert "15 metadata columns:" in r
    lines_with_quotes = [l for l in r.splitlines() if '"very_long' in l]
    assert len(lines_with_quotes) > 1, "Expected column names to wrap to multiple lines"


def test_filter_indentation():
    exp = InSituExperiment()
    exp._metadata = pd.DataFrame({"uid": ["a", "b", "c"], "qc": ["pass", "pass", "fail"]})
    exp._filters["qc_pass"] = {"mask": [True, True, False], "note": None}
    r = repr(exp)
    lines = r.splitlines()
    assert any("filters" in l and "➤" in l for l in lines)
    base_header_idx = next(i for i, l in enumerate(lines) if "Base filters" in l)
    table_line_idx = next(i for i, l in enumerate(lines) if "qc_pass" in l)
    # table row should be indented more than "Base filters:" header
    base_indent = len(lines[base_header_idx]) - len(lines[base_header_idx].lstrip())
    table_indent = len(lines[table_line_idx]) - len(lines[table_line_idx].lstrip())
    assert table_indent > base_indent, "Table rows should be indented more than subsection header"


def test_table_human_readable():
    exp = _make_exp(3)
    r = repr(exp)
    assert "no tables built" in r


def test_html():
    exp = _make_exp(25)
    h = exp._repr_html_()
    assert "<b>InSituExperiment</b>" in h
    assert "25 samples" in h
    assert "metadata columns:" in h
    assert '"uid"' in h
    assert "<b>▶ filters</b><br>" in h
    assert "padding-left:1em" in h  # indentation applied
    assert "table" in h
    assert "no tables built" in h

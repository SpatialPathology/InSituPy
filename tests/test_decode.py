import pandas as pd

from insitupy.utils.utils import decode_robust_series

s = pd.Series(list("ABCD"))  # string series
b = pd.Series([b"A", b"B", b"C", b"D"])  # byte series
n = pd.Series(range(0, 4))  # numeric series
a = pd.Series(["\xa1", "\xa2", "\xa5", "\xa3"])  # ascii series
m = pd.Series(["A", b"c", 1, "\xa5"]) # mixed series
mt = pd.Series(["A", "c", 1, "\xa5"]) # test result for mixed series

def test_string_decode():
    assert decode_robust_series(s).equals(s)

def test_byte_decode():
    assert decode_robust_series(b).equals(s)

def test_numeric_decode():
    assert decode_robust_series(n).equals(n)

def test_ascii_decode():
    assert decode_robust_series(a).equals(a)

def test_mixed_series_decode():
    assert decode_robust_series(m).equals(mt)

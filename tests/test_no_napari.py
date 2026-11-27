"""InSituPy must work without the napari viewer stack (``pip install insitupy-spatial``).

Each test runs in a subprocess with napari and its GUI dependencies masked via
``sys.modules[name] = None`` (importing a masked module raises ImportError), so the checks are
independent of whether napari is installed in the test environment and the masking cannot leak
into other tests.
"""
import subprocess
import sys
import textwrap

_PRELUDE = """
import sys
for _name in ("napari", "magicgui", "qtpy", "vispy", "superqt"):
    sys.modules[_name] = None
import warnings
warnings.simplefilter("ignore", ImportWarning)
import insitupy
assert insitupy.WITH_NAPARI is False
"""


def _run(body: str) -> subprocess.CompletedProcess:
    code = _PRELUDE + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)


def _assert_ok(proc: subprocess.CompletedProcess):
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_public_submodules_import_without_napari():
    proc = _run("""
        import insitupy.io, insitupy.pp, insitupy.tl, insitupy.pl, insitupy.im, insitupy.utils
        import insitupy.datasets
        import insitupy.interactive._label_alignment
        from insitupy.interactive import TranscriptViewerConfig, prepare_gene_data
    """)
    _assert_ok(proc)


def test_viewer_only_names_raise_friendly_import_error():
    proc = _run("""
        try:
            from insitupy.interactive import sync_geometries
        except ImportError as err:
            assert "insitupy-spatial[viewer]" in str(err), str(err)
        else:
            raise AssertionError("expected ImportError")
    """)
    _assert_ok(proc)


def test_show_raises_friendly_import_error():
    proc = _run("""
        from insitupy import InSituData
        try:
            InSituData.show(object())
        except ImportError as err:
            assert "insitupy-spatial[viewer]" in str(err), str(err)
        else:
            raise AssertionError("expected ImportError")
    """)
    _assert_ok(proc)


def test_shapesdata_skips_non_polygons_with_warning_not_nameerror():
    proc = _run("""
        import geopandas as gpd
        from shapely import Point, Polygon
        from insitupy.containers.shapes_data import ShapesData

        gdf = gpd.GeoDataFrame(
            {"id": ["u1", "u2"], "name": ["a", "b"], "color": [[255, 0, 0]] * 2},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1)]), Point(5, 5)],
        )
        shapes = ShapesData(polygons_only=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shapes.add_data(gdf, key="k", scale_factor=1.0, verbose=True)
        assert any("Polygon" in str(w.message) for w in caught), [str(w.message) for w in caught]
        assert len(shapes["k"]) == 1
    """)
    _assert_ok(proc)

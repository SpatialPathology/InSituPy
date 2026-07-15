import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import Point
from shapely.geometry import Polygon

from insitupy._io.geo import parse_geopandas, read_qupath_geojson, write_qupath_geojson

d = {'id': ['i'],
    'objectType': ['o'],
    'name': ['n'],
    'color': ['c'],
    'geometry': [Point(0, 1)],
    'origin': ['manual']
    }

pdf = pd.DataFrame(d)

result_df = GeoDataFrame(d, geometry=d["geometry"])
result_df = result_df.set_index("id")

def test_dict_geopandas():
    assert parse_geopandas(d).equals(result_df)

def test_pandas_geopandas():
    assert parse_geopandas(pdf).equals(result_df)


def test_read_qupath_geojson_handles_non_literal_classification(monkeypatch):
    data = pd.DataFrame(
        {
            "classification": [
                '{"name": "tumor", "color": [1, 2, 3], "scale": [1, 1]}',
                "QColor(255, 0, 0)",
            ]
        }
    )

    def fake_read_file(file, engine):
        return data.copy()

    monkeypatch.setattr("insitupy._io.geo.geopandas.read_file", fake_read_file)

    result = read_qupath_geojson("dummy.geojson")

    assert result["name"].tolist() == ["tumor", "unclassified"]
    assert result["color"].tolist() == [[1, 2, 3], [0, 0, 0]]
    assert result["scale"].tolist() == [[1, 1], (1, 1)]
    assert "classification" not in result.columns


def test_write_qupath_geojson_preserves_numpy_colors(tmp_path):
    def _poly(x=0):
        return Polygon([(x, 0), (x + 10, 0), (x + 10, 10), (x, 10)])

    gdf = GeoDataFrame(
        {"name": ["A", "B"],
         "color": [np.array([250, 62, 62]), np.array([112, 112, 225])],  # numpy, like the zarr round trip
         "scale": [(1, 1), (1, 1)],
         "geometry": [_poly(0), _poly(20)]},
        geometry="geometry",
    )
    gdf.index = ["id0", "id1"]
    gdf.index.name = "id"
    gdf = gdf.set_crs(4326)

    out = tmp_path / "shapes.geojson"
    write_qupath_geojson(gdf.copy(), out)
    back = read_qupath_geojson(out)

    assert [list(c) for c in back["color"]] == [[250, 62, 62], [112, 112, 225]]

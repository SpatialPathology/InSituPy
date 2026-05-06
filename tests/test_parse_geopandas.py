import pandas as pd
from geopandas import GeoDataFrame
from shapely import Point

from insitupy._io.geo import parse_geopandas, read_qupath_geojson

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

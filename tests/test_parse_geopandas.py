from shapely import Point
from geopandas import GeoDataFrame
import pandas as pd
from insitupy.io.geo import parse_geopandas

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
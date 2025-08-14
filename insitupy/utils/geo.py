import math

import geopandas as gpd
import shapely
from shapely import MultiPolygon, Polygon
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

# def calculate_optimal_grid_size(polygon, num_points):
#     # Get the bounds of the polygon
#     minx, miny, maxx, maxy = polygon.bounds

#     # Calculate the area of the bounding box
#     bbox_area = (maxx - minx) * (maxy - miny)

#     # Calculate a factor based on the number of points
#     factor = max(1, num_points / 1000)  # Adjust the divisor as needed

#     # Calculate the optimal grid size
#     optimal_grid_size = (bbox_area / factor) ** 0.5
#     #print(optimal_grid_size)

#     return optimal_grid_size

def create_strtree_from_polygon(polygon, max_grid_width):
    # Calculate the optimal grid size
    #grid_size = calculate_optimal_grid_size(polygon, num_points)

    # Create a GeoDataFrame from the polygon
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})

    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    if max_grid_width < max(maxx - minx, maxy - miny):
        # Create a grid of boxes
        rows = math.ceil((maxy - miny) / max_grid_width)
        cols = math.ceil((maxx - minx) / max_grid_width)
        boxes = []
        for i in range(rows):
            for j in range(cols):
                boxes.append(box(minx + j * max_grid_width,
                                miny + i * max_grid_width,
                                minx + (j + 1) * max_grid_width,
                                miny + (i + 1) * max_grid_width))

        # Create a GeoDataFrame from the boxes
        grid = gpd.GeoDataFrame({'geometry': boxes})

        # Overlay to split the original polygon
        split_gdf = gpd.overlay(gdf, grid, how='intersection')

        # Create an STRtree index
        strtree = STRtree(split_gdf.geometry)
    else:
        strtree = None

    return strtree

def fast_query_points_within_polygon(polygon: Polygon, points: gpd.GeoSeries):
    # create STRtree
    #tree = create_strtree_from_polygon(polygon=polygon, num_points=len(points))
    tree = create_strtree_from_polygon(polygon=polygon, max_grid_width=150)
    #print(len(tree.geometries))

    if tree is None:
        mask = polygon.contains(points).values
    else:
        # query tree
        res = tree.query(points, predicate="within")
        point_ids = res[0]
        mask = points.index.isin(point_ids)
    return mask


def _parse_geometries(geometry):
    coords = geometry["coordinates"]
    if geometry['type'] == 'MultiPolygon':
        try:
            multipolygons = MultiPolygon(coords)
        except TypeError:
            multipolygons = MultiPolygon([coords])
        polygon = multipolygons.convex_hull
        parsed_type = "polygon"

    elif geometry['type'] == 'Polygon':
        if len(coords) > 2:

            merged_geometry = Polygon(geometry['coordinates'][0])
            parsed_type = "polygon"

        # # check if there are enough coordinates for a Polygon (some segmented cells are very small in Baysor)
        # if len(coords) > 2:
        #     p = shapely.Polygon(coords[0])
        #     results_dict["geometry"].append(p)
        #     results_dict["type"].append("polygon")

        # else:
        #     p = shapely.LineString(coords)
        #     results_dict["geometry"].append(p)
        #     results_dict["type"].append("line")
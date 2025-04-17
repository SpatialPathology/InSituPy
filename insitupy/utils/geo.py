import math

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree


def calculate_optimal_grid_size(polygon, num_points):
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate the area of the bounding box
    bbox_area = (maxx - minx) * (maxy - miny)

    # Calculate a factor based on the number of points
    factor = max(1, num_points / 1000)  # Adjust the divisor as needed

    # Calculate the optimal grid size
    optimal_grid_size = (bbox_area / factor) ** 0.5

    return optimal_grid_size

def create_strtree_from_polygon(polygon, num_points):
    # Calculate the optimal grid size
    grid_size = calculate_optimal_grid_size(polygon, num_points)

    # Create a GeoDataFrame from the polygon
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})

    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Create a grid of boxes
    rows = math.ceil((maxy - miny) / grid_size)
    cols = math.ceil((maxx - minx) / grid_size)
    boxes = []
    for i in range(rows):
        for j in range(cols):
            boxes.append(box(minx + j * grid_size,
                             miny + i * grid_size,
                             minx + (j + 1) * grid_size,
                             miny + (i + 1) * grid_size))

    # Create a GeoDataFrame from the boxes
    grid = gpd.GeoDataFrame({'geometry': boxes})

    # Overlay to split the original polygon
    split_gdf = gpd.overlay(gdf, grid, how='intersection')

    # Create an STRtree index
    strtree = STRtree(split_gdf.geometry)

    return strtree

def fast_query_points_within_polygon(polygon: Polygon, points: gpd.GeoSeries):
    # create STRtree
    tree = create_strtree_from_polygon(polygon=polygon, num_points=len(points))

    # query tree
    res = tree.query(points, predicate="within")
    point_ids = res[0]
    mask = points.index.isin(point_ids)
    return mask
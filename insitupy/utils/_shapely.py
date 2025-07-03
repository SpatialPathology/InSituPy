from shapely import affinity, wkt


def scale_polygon(polygon, constant):
    """
        Scales the polygon by a given constant.

        Args:
            polygon_wkt (str): Polygon in shapely.geometry.polygon.Polygon format.
            constant (float): Scaling constant.

        Returns:
            Polygon: Scaled polygon.
    """
    divided_polygon = affinity.scale(polygon, xfact=1/constant, yfact=1/constant, origin=(0, 0))
    return divided_polygon
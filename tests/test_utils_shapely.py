"""Tests for insitupy.utils._shapely — scale_polygon."""

from shapely.geometry import Polygon

from insitupy.utils._shapely import scale_polygon


class TestScalePolygon:
    def test_scale_by_one_is_identity(self):
        # Scaling by 1 must not change polygon coordinates
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        result = scale_polygon(poly, 1)
        assert result.equals(poly)

    def test_scale_by_two_halves_coordinates(self):
        # scale_polygon divides by the constant, so factor=2 → coords halved
        # Polygon at (0,0)-(2,0)-(2,2)-(0,2) → after /2 → (0,0)-(1,0)-(1,1)-(0,1)
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        result = scale_polygon(poly, 2)
        expected = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert result.equals(expected)

    def test_scale_preserves_polygon_type(self):
        # The return type must still be a shapely Polygon
        poly = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        result = scale_polygon(poly, 3)
        assert isinstance(result, Polygon)

    def test_scale_by_constant_reduces_area_proportionally(self):
        # Area should be reduced by constant^2 since both x and y are divided
        # Square 4x4 has area 16; scaling by 2 → 2x2 → area 4 = 16/4
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        result = scale_polygon(poly, 2)
        assert abs(result.area - 4.0) < 1e-10

"""ASC-FITS-REGION polygon membership helpers.

ASC-FITS-REGION includes polygon boundary lines in the region element. The
minimum supported ``regions`` release exposes strict-interior ``contains()`` but
not the newer boundary-inclusive ``covers()`` API, so the package overlays an
explicit point-on-segment test without raising the dependency floor.
"""

from __future__ import annotations

import numpy as np
from regions import PixCoord, PolygonPixelRegion

_BOUNDARY_ULP_FACTOR = 64.0


def _coordinate_tolerance(
    x: np.ndarray,
    y: np.ndarray,
    *coordinates: float,
) -> np.ndarray:
    """Return a floating-roundoff tolerance in native coordinate units.

    The tolerance scales with coordinate magnitude and IEEE-754 double epsilon;
    it is not a physical detector tolerance. At typical EPIC DET coordinates
    (order 1e4--1e5) it is order 1e-10--1e-9 DET, far below one native DET unit.
    """
    scale = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1.0)
    for value in coordinates:
        scale = np.maximum(scale, abs(float(value)))
    return _BOUNDARY_ULP_FACTOR * np.finfo(float).eps * scale


def _points_on_segment(
    x: np.ndarray,
    y: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    dx = float(bx - ax)
    dy = float(by - ay)
    px = x - ax
    py = y - ay
    tolerance = _coordinate_tolerance(x, y, ax, ay, bx, by)
    segment_length = float(np.hypot(dx, dy))

    if segment_length == 0.0:
        distance = np.hypot(px, py)
    else:
        # |cross| / |segment| is the perpendicular distance in DET units.
        distance = np.abs(px * dy - py * dx) / segment_length

    within_bounds = (
        (x >= min(ax, bx) - tolerance)
        & (x <= max(ax, bx) + tolerance)
        & (y >= min(ay, by) - tolerance)
        & (y <= max(ay, by) + tolerance)
    )
    return within_bounds & (distance <= tolerance)


def points_on_polygon_boundary(
    polygon: PolygonPixelRegion,
    coordinates: PixCoord,
) -> np.ndarray:
    """Return points lying on any polygon edge within floating roundoff."""
    x = np.asarray(coordinates.x, dtype=float)
    y = np.asarray(coordinates.y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    vertices_x = np.asarray(polygon.vertices.x, dtype=float)
    vertices_y = np.asarray(polygon.vertices.y, dtype=float)
    if vertices_x.ndim != 1 or vertices_y.ndim != 1 or len(vertices_x) != len(vertices_y):
        raise ValueError("polygon vertices must be one-dimensional paired coordinate arrays")
    if len(vertices_x) < 3:
        raise ValueError("polygon must contain at least three vertices")

    boundary = np.zeros(x.shape, dtype=bool)
    for index in range(len(vertices_x)):
        following = (index + 1) % len(vertices_x)
        boundary |= _points_on_segment(
            x,
            y,
            float(vertices_x[index]),
            float(vertices_y[index]),
            float(vertices_x[following]),
            float(vertices_y[following]),
        )
    return boundary


def polygon_covers(
    polygon: PolygonPixelRegion,
    coordinates: PixCoord,
) -> np.ndarray:
    """Evaluate ASC polygon membership: strict interior OR polygon boundary."""
    interior = np.asarray(polygon.contains(coordinates), dtype=bool)
    return interior | points_on_polygon_boundary(polygon, coordinates)


__all__ = ["points_on_polygon_boundary", "polygon_covers"]

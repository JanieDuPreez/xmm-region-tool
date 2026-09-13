"""Reusable detector-space topology validation.

This module validates the structural meaning of ``DetectorRegion``: one simple
outer component boundary followed by zero or more simple subtractive holes.
Top-level included/excluded regions are independent Boolean components; no
containment relationship is imposed between separate regions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .limits import detector_region_table_budget, detector_topology_work_budget
from .model import DetectorRegion


class DetectorTopologyError(ValueError):
    """Raised when detector polygons do not form canonical component/hole topology."""


_EPS = 1e-9
_FLOAT_EPS = np.finfo(float).eps
_NUMERIC_SAFETY = 64.0


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(
    a: np.ndarray,
    b: np.ndarray,
    point: np.ndarray,
    *,
    eps: float = _EPS,
) -> bool:
    return (
        min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, point)) <= eps
    )


def segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = _EPS,
) -> bool:
    """Return whether two closed line segments cross or touch."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return (
        (abs(o1) <= eps and _on_segment(a, b, c, eps=eps))
        or (abs(o2) <= eps and _on_segment(a, b, d, eps=eps))
        or (abs(o3) <= eps and _on_segment(c, d, a, eps=eps))
        or (abs(o4) <= eps and _on_segment(c, d, b, eps=eps))
    )


def _extent_scale(vertices: np.ndarray) -> float:
    """Return a translation-invariant characteristic polygon length scale."""
    spans = np.ptp(vertices, axis=0)
    return float(max(spans[0], spans[1]))


def _minimum_numeric_edge(scale: float) -> float:
    return _NUMERIC_SAFETY * _FLOAT_EPS * scale


def _twice_polygon_area(vertices: np.ndarray) -> float:
    """Return absolute twice-area after centring to reduce cancellation."""
    centred = vertices - vertices[0]
    x = centred[:, 0]
    y = centred[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _minimum_numeric_twice_area(scale: float, vertex_count: int) -> float:
    return _NUMERIC_SAFETY * _FLOAT_EPS * max(1, vertex_count) * scale * scale


def validate_detector_boundary(vertices: np.ndarray) -> None:
    """Require one finite simple closed polygon with non-degenerate enclosed area.

    Degeneracy tolerances scale with the polygon's own detector-space extent.
    There is intentionally no fixed minimum science-region size: an ordinary
    tiny polygon can remain valid as long as its edges/area are numerically
    resolved relative to its own scale.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise DetectorTopologyError("projected detector boundary has fewer than three vertices")
    if not np.isfinite(vertices).all():
        raise DetectorTopologyError("projected detector boundary contains non-finite coordinates")

    scale = _extent_scale(vertices)
    if scale == 0.0:
        raise DetectorTopologyError("projected detector boundary has zero spatial extent")
    edge_tolerance = _minimum_numeric_edge(scale)
    n = len(vertices)
    for index in range(n):
        if np.linalg.norm(vertices[(index + 1) % n] - vertices[index]) <= edge_tolerance:
            raise DetectorTopologyError("projected detector boundary contains a degenerate edge")

    # Diagnose actual crossings before the area test. A bow-tie has zero signed
    # shoelace area too, but "self-intersects" is the scientifically more
    # specific failure. Purely collinear/overlapping-adjacent cases then fall
    # through to the non-degenerate-area check below.
    for first in range(n):
        a = vertices[first]
        b = vertices[(first + 1) % n]
        for second in range(first + 1, n):
            if second in {first, (first + 1) % n}:
                continue
            if first == 0 and second == n - 1:
                continue
            c = vertices[second]
            d = vertices[(second + 1) % n]
            if segments_intersect(a, b, c, d):
                raise DetectorTopologyError(
                    "projected detector boundary self-intersects; refusing ambiguous topology"
                )

    twice_area = _twice_polygon_area(vertices)
    area_tolerance = _minimum_numeric_twice_area(scale, n)
    if twice_area <= area_tolerance:
        raise DetectorTopologyError(
            "projected detector boundary has zero/near-zero enclosed area"
        )


def _point_location(point: np.ndarray, polygon: np.ndarray) -> str:
    """Return ``inside``, ``outside`` or ``boundary`` for a simple polygon."""
    point = np.asarray(point, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    inside = False
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    for index in range(n):
        a = polygon[index]
        b = polygon[(index + 1) % n]
        if _on_segment(a, b, point):
            return "boundary"
        ay, by = float(a[1]), float(b[1])
        if (ay > y) == (by > y):
            continue
        ax, bx = float(a[0]), float(b[0])
        x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
        if abs(x_cross - x) <= _EPS:
            return "boundary"
        if x_cross > x:
            inside = not inside
    return "inside" if inside else "outside"


def _boundaries_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    for first_index in range(len(first)):
        a0 = first[first_index]
        a1 = first[(first_index + 1) % len(first)]
        for second_index in range(len(second)):
            b0 = second[second_index]
            b1 = second[(second_index + 1) % len(second)]
            if segments_intersect(a0, a1, b0, b1):
                return True
    return False


def validate_detector_regions(regions: Sequence[DetectorRegion]) -> None:
    """Validate bounded simple component/hole topology for detector regions.

    The serializer's detector row/vertex/aggregate-coordinate envelope and an
    explicit quadratic-work budget are applied before any pairwise topology
    traversal. Within each component, every subtractive boundary must be strictly
    contained by the outer boundary. Distinct holes must be disjoint and
    non-nested. The same rule applies to an excluded top-level component because
    its internal subtractive boundaries are still holes of that component.

    Separate top-level regions are intentionally not compared: an exclusion mask
    may overlap or cross an included science component as ordinary Boolean
    geometry.
    """
    try:
        detector_region_table_budget(regions)
        detector_topology_work_budget(regions)
    except ValueError as exc:
        raise DetectorTopologyError(str(exc)) from exc

    for region_index, region in enumerate(regions):
        boundaries = region.boundaries
        for boundary in boundaries:
            validate_detector_boundary(boundary.vertices)

        for first_index, first in enumerate(boundaries):
            for second_index, second in enumerate(
                boundaries[first_index + 1 :],
                start=first_index + 1,
            ):
                if _boundaries_intersect(first.vertices, second.vertices):
                    raise DetectorTopologyError(
                        "projected detector boundaries intersect/touch within "
                        f"component {region_index} (boundaries {first_index} and "
                        f"{second_index}); refusing ambiguous component/hole topology"
                    )

        outer = boundaries[0].vertices
        for hole_index, hole in enumerate(boundaries[1:], start=1):
            location = _point_location(hole.vertices[0], outer)
            if location != "inside":
                raise DetectorTopologyError(
                    f"subtractive hole {hole_index} in component {region_index} is not "
                    "strictly contained by its outer detector boundary"
                )

        holes = boundaries[1:]
        for first_index, first in enumerate(holes, start=1):
            for second_index, second in enumerate(
                holes[first_index:],
                start=first_index + 1,
            ):
                second_in_first = _point_location(second.vertices[0], first.vertices)
                first_in_second = _point_location(first.vertices[0], second.vertices)
                if second_in_first != "outside" or first_in_second != "outside":
                    raise DetectorTopologyError(
                        "subtractive holes must be disjoint and non-nested within one "
                        f"component; component {region_index} holes {first_index} and "
                        f"{second_index} are redundant/ambiguous"
                    )

"""Numerical validity checks for shortest-geodesic celestial polygons."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from .model import SelectionGeometryError
from .limits import MAX_GEODESIC_PAIR_CHECKS


# An angular roundoff allowance, not a minimum astronomical region size.
_ANGLE_TOL = 32 * np.finfo(float).eps


def validate_geodesic_vertices(vertices: SkyCoord) -> None:
    """Reject undefined edges and positive-length retracing, retaining crossings.

    Angular endpoint comparisons allow 32 double-precision epsilons in radians.
    Plane tests scale with chord lengths rather than imposing a minimum area,
    so genuinely two-dimensional small polygons remain supported.
    """
    if vertices.ndim != 1 or len(vertices) < 3:
        raise SelectionGeometryError("a geodesic polygon requires a 1-D sequence of at least three vertices")
    count = len(vertices)
    if count * (count - 1) // 2 > MAX_GEODESIC_PAIR_CHECKS:
        raise SelectionGeometryError(
            f"geodesic pair-check work budget exceeded ({MAX_GEODESIC_PAIR_CHECKS}); "
            "simplify the authored polygon to at most 256 vertices"
        )
    coords = vertices.icrs
    if not np.isfinite(coords.ra.rad).all() or not np.isfinite(coords.dec.rad).all():
        raise SelectionGeometryError("celestial boundary vertices must be finite")
    lengths = coords.separation(coords[np.roll(np.arange(count), -1)]).to_value(u.rad)
    for index, length in enumerate(lengths):
        if length <= _ANGLE_TOL:
            raise SelectionGeometryError(f"geodesic edge {index} has duplicate or numerically coincident vertices")
        if np.pi - length <= _ANGLE_TOL:
            raise SelectionGeometryError(f"geodesic edge {index} is antipodal; its shortest arc is not unique")

    # Unit-spherical directions: source distance has no bearing on sky arcs.
    points = coords.represent_as("unitspherical").to_cartesian().xyz.value.T
    for first in range(count):
        a = points[first]
        b = points[(first + 1) % count]
        chord = b - a
        chord_length = np.linalg.norm(chord)
        normal = np.cross(a, chord)
        normal_length = np.linalg.norm(normal)
        # The normal gives a tangent pointing from a towards b.
        tangent = np.cross(normal / normal_length, a)
        for second in range(first + 1, count):
            c = points[second]
            d = points[(second + 1) % count]
            if any(
                abs(np.dot(normal, point - a))
                > _ANGLE_TOL * (chord_length + np.linalg.norm(point - a))
                for point in (c, d)
            ):
                continue
            start = np.arctan2(np.dot(c, tangent), np.dot(c, a))
            stop = np.arctan2(np.dot(d, tangent), np.dot(d, a))
            sweep = (stop - start + np.pi) % (2 * np.pi) - np.pi
            low, high = sorted((start, start + sweep))
            for shift in (-2 * np.pi, 0.0, 2 * np.pi):
                overlap = min(lengths[first], high + shift) - max(0.0, low + shift)
                if overlap > _ANGLE_TOL:
                    raise SelectionGeometryError(
                        f"geodesic edges {first} and {second} overlap or retrace; "
                        "provide a non-degenerate polygon boundary"
                    )

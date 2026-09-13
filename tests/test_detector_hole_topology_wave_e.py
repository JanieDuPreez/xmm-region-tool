from __future__ import annotations

import numpy as np
import pytest

from xmm_region_tool.model import DetectorBoundary, DetectorRegion
from xmm_region_tool.topology import (
    DetectorTopologyError,
    validate_detector_boundary,
    validate_detector_regions,
)


def _square(x0: float, y0: float, x1: float, y1: float, *, subtract: bool = False):
    return DetectorBoundary(
        np.asarray(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=float,
        ),
        subtract=subtract,
    )


def test_collinear_triangle_is_rejected_as_zero_area():
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)

    with pytest.raises(DetectorTopologyError, match="zero/near-zero enclosed area"):
        validate_detector_boundary(vertices)


def test_very_small_but_resolved_triangle_is_accepted():
    # There is no fixed minimum detector-region size. Degeneracy is assessed
    # relative to this polygon's own scale, so an ordinary tiny triangle remains
    # valid when its area and edges are numerically resolved.
    vertices = np.asarray(
        [[0.0, 0.0], [1.0e-12, 0.0], [0.0, 1.0e-12]],
        dtype=float,
    )

    validate_detector_boundary(vertices)


def test_valid_internal_hole_is_accepted():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(0, 0, 10, 10),
            _square(2, 2, 4, 4, subtract=True),
        ),
    )

    validate_detector_regions((region,))


def test_hole_fully_outside_outer_fails_with_containment_error():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(0, 0, 10, 10),
            _square(20, 20, 30, 30, subtract=True),
        ),
    )

    with pytest.raises(DetectorTopologyError, match="hole 1.*not strictly contained"):
        validate_detector_regions((region,))


def test_hole_containing_outer_fails_with_containment_error():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(2, 2, 4, 4),
            _square(0, 0, 10, 10, subtract=True),
        ),
    )

    with pytest.raises(DetectorTopologyError, match="hole 1.*not strictly contained"):
        validate_detector_regions((region,))


def test_nested_internal_holes_are_rejected_as_noncanonical():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(0, 0, 20, 20),
            _square(2, 2, 10, 10, subtract=True),
            _square(4, 4, 6, 6, subtract=True),
        ),
    )

    with pytest.raises(DetectorTopologyError, match="disjoint and non-nested"):
        validate_detector_regions((region,))


def test_touching_hole_outer_boundary_remains_rejected_as_intersection():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(0, 0, 10, 10),
            _square(0, 2, 4, 4, subtract=True),
        ),
    )

    with pytest.raises(DetectorTopologyError, match="intersect/touch"):
        validate_detector_regions((region,))


def test_internal_hole_rule_applies_inside_excluded_component():
    excluded_annulus = DetectorRegion(
        include=False,
        boundaries=(
            _square(0, 0, 10, 10),
            _square(2, 2, 4, 4, subtract=True),
        ),
    )

    validate_detector_regions((excluded_annulus,))


def test_separate_top_level_exclusion_may_cross_positive_component():
    positive = DetectorRegion(include=True, boundaries=(_square(0, 0, 10, 10),))
    crossing_mask = DetectorRegion(include=False, boundaries=(_square(5, -2, 12, 5),))

    # Top-level Boolean components are independent; #52/#59 must not turn an
    # ordinary crossing exclusion mask into an invalid internal-hole topology.
    validate_detector_regions((positive, crossing_mask))

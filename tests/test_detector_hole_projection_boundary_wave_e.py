from __future__ import annotations

import numpy as np
import pytest

from xmm_region_tool.model import DetectorBoundary, DetectorRegion
from xmm_region_tool.sas import SasConversionError, _validate_region_topology


def _square(x0, y0, x1, y1, *, subtract=False):
    return DetectorBoundary(
        np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float),
        subtract=subtract,
    )


def test_projection_boundary_reports_invalid_hole_containment():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(0, 0, 10, 10),
            _square(20, 20, 30, 30, subtract=True),
        ),
    )

    with pytest.raises(SasConversionError, match="subtractive hole 1.*not strictly contained"):
        _validate_region_topology((region,))


def test_projection_boundary_accepts_valid_internal_hole():
    region = DetectorRegion(
        include=True,
        boundaries=(
            _square(0, 0, 10, 10),
            _square(2, 2, 4, 4, subtract=True),
        ),
    )

    _validate_region_topology((region,))

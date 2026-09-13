from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.output import write_fits_region
from xmm_region_tool.sas import DetectorBoundary, DetectorRegion
from xmm_region_tool.sas_validation import (
    _MAX_VALIDATION_EVENT_ROWS,
    SasRegionValidationError,
    _validation_event_copy,
    _validation_event_table_preflight,
    fits_region_membership,
)


def boundary(*points, subtract: bool = False) -> DetectorBoundary:
    return DetectorBoundary(np.asarray(points, dtype=float), subtract=subtract)


def test_fits_region_membership_respects_include_and_exclude_components(tmp_path):
    regions = [
        DetectorRegion(
            True,
            (boundary((0, 0), (4, 0), (4, 4), (0, 4)),),
        ),
        DetectorRegion(
            False,
            (boundary((1, 1), (3, 1), (1, 3)),),
        ),
    ]
    path = write_fits_region(tmp_path / "region.fits", regions)

    mask = fits_region_membership(
        path,
        np.asarray([0.5, 1.5, 5.0]),
        np.asarray([0.5, 1.5, 5.0]),
    )

    assert mask.tolist() == [True, False, False]


def test_fits_region_membership_respects_annulus_hole(tmp_path):
    annulus = DetectorRegion(
        True,
        (
            boundary((0, 0), (6, 0), (6, 6), (0, 6)),
            boundary((2, 2), (4, 2), (4, 4), (2, 4), subtract=True),
        ),
    )
    path = write_fits_region(tmp_path / "annulus.fits", [annulus])

    mask = fits_region_membership(
        path,
        np.asarray([1.0, 3.0, 7.0]),
        np.asarray([1.0, 3.0, 7.0]),
    )

    assert mask.tolist() == [True, False, False]


def test_fits_region_membership_includes_positive_polygon_edge_and_vertex(tmp_path):
    region = DetectorRegion(
        True,
        (boundary((0, 0), (4, 0), (0, 4)),),
    )
    path = write_fits_region(tmp_path / "positive-boundary.fits", [region])

    mask = fits_region_membership(
        path,
        np.asarray([1.0, 3.0, 2.0, 0.0, 2.0]),
        np.asarray([1.0, 3.0, 0.0, 0.0, -1.0e-6]),
    )

    # inside, outside, exact edge, exact vertex, deliberately just outside.
    assert mask.tolist() == [True, False, True, True, False]


def test_fits_region_membership_excludes_negated_polygon_edge_and_vertex(tmp_path):
    region = DetectorRegion(
        True,
        (
            boundary((0, 0), (6, 0), (6, 6), (0, 6)),
            boundary((2, 2), (4, 2), (4, 4), (2, 4), subtract=True),
        ),
    )
    path = write_fits_region(tmp_path / "negative-boundary.fits", [region])

    mask = fits_region_membership(
        path,
        np.asarray([1.0, 3.0, 2.0, 2.0]),
        np.asarray([1.0, 3.0, 2.0, 3.0]),
    )

    # shell interior is selected; hole interior, exact hole vertex and exact hole edge are excluded.
    assert mask.tolist() == [True, False, False, False]


def test_validation_event_copy_injects_unique_row_ids_without_mutating_source(tmp_path):
    source = tmp_path / "events.fits"
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 3.0, 4.0])),
            fits.Column(name="PI", format="J", array=np.asarray([500, 500, 600], dtype=np.int32)),
        ],
        name="EVENTS",
    )
    fits.HDUList([fits.PrimaryHDU(), events]).writeto(source)
    original_bytes = source.read_bytes()

    target = _validation_event_copy(source, tmp_path / "validation.fits")

    assert source.read_bytes() == original_bytes
    with fits.open(target) as hdus:
        data = hdus["EVENTS"].data
        assert data is not None
        assert "XMMRGROW" in hdus["EVENTS"].columns.names
        assert hdus["EVENTS"].columns["XMMRGROW"].format == "J"
        assert np.asarray(data["XMMRGROW"], dtype=np.int64).tolist() == [0, 1, 2]
        assert np.asarray(data["PI"]).tolist() == [500, 500, 600]


class _HeaderOnlyEventHDU:
    def __init__(self, **header_values):
        self.header = fits.Header(header_values)
        self.columns = SimpleNamespace(names=("DETX", "DETY"))

    @property
    def data(self):
        raise AssertionError("EVENTS data was materialised before metadata preflight")


def test_validation_event_preflight_rejects_int32_overflow_before_data_access():
    event_hdu = _HeaderOnlyEventHDU(
        NAXIS1=45,
        NAXIS2=np.iinfo(np.int32).max + 1,
        PCOUNT=0,
    )

    with pytest.raises(SasRegionValidationError, match="int32 validation ids"):
        _validation_event_table_preflight(event_hdu)


def test_validation_event_preflight_rejects_workload_rows_before_data_access():
    event_hdu = _HeaderOnlyEventHDU(
        NAXIS1=45,
        NAXIS2=_MAX_VALIDATION_EVENT_ROWS + 1,
        PCOUNT=0,
    )

    with pytest.raises(SasRegionValidationError, match="row workload limit"):
        _validation_event_table_preflight(event_hdu)


@pytest.mark.parametrize(
    ("header_values", "message"),
    [
        ({"NAXIS1": -1, "NAXIS2": 10, "PCOUNT": 0}, "NAXIS1"),
        ({"NAXIS1": 45, "NAXIS2": -1, "PCOUNT": 0}, "NAXIS2"),
        ({"NAXIS1": 45, "NAXIS2": 10, "PCOUNT": -1}, "PCOUNT"),
        (
            {"NAXIS1": 45, "NAXIS2": 10, "PCOUNT": 0, "THEAP": 449},
            "THEAP points inside",
        ),
        (
            {"NAXIS1": 45, "NAXIS2": 10, "PCOUNT": 5, "THEAP": 456},
            "THEAP lies beyond",
        ),
    ],
)
def test_validation_event_preflight_rejects_malformed_layout_before_data_access(
    header_values,
    message,
):
    event_hdu = _HeaderOnlyEventHDU(**header_values)

    with pytest.raises(SasRegionValidationError, match=message):
        _validation_event_table_preflight(event_hdu)

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import CircleSkyRegion

from xmm_region_tool.sas import DetectorBoundary, DetectorRegion
from xmm_region_tool.validation import compare_event_membership, event_xy_wcs


def write_event(path):
    x = np.asarray([10.5, 11.5, 14.5], dtype=np.float64)
    y = np.asarray([10.5, 10.5, 10.5], dtype=np.float64)
    detx = x - 1.0
    dety = y - 1.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="X", format="D", array=x),
            fits.Column(name="Y", format="D", array=y),
            fits.Column(name="DETX", format="D", array=detx),
            fits.Column(name="DETY", format="D", array=dety),
        ],
        name="EVENTS",
    )
    scale = 1.0 / 3600.0
    events.header["TCTYP1"] = "RA---TAN"
    events.header["TCUNI1"] = "deg"
    events.header["TCRVL1"] = 10.0
    events.header["TCRPX1"] = 10.5
    events.header["TCDLT1"] = -scale
    events.header["TCTYP2"] = "DEC--TAN"
    events.header["TCUNI2"] = "deg"
    events.header["TCRVL2"] = -9.0
    events.header["TCRPX2"] = 10.5
    events.header["TCDLT2"] = scale
    fits.HDUList([fits.PrimaryHDU(), events]).writeto(path)
    return path


def detector_circle(*, centre_x: float = 9.5) -> DetectorRegion:
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    vertices = np.column_stack(
        (
            centre_x + 2.0 * np.cos(theta),
            9.5 + 2.0 * np.sin(theta),
        )
    )
    return DetectorRegion(True, (DetectorBoundary(vertices),))


def test_event_xy_wcs_maps_reference_sky_position_to_fits_origin_one(tmp_path):
    event = write_event(tmp_path / "events.fits")
    wcs = event_xy_wcs(event)

    x, y = wcs.all_world2pix([[10.0, -9.0]], 1)[0]

    assert x == pytest.approx(10.5, abs=1e-10)
    assert y == pytest.approx(10.5, abs=1e-10)


def test_independent_sky_and_detector_membership_agree(tmp_path):
    event = write_event(tmp_path / "events.fits")
    source = [CircleSkyRegion(SkyCoord(10.0, -9.0, unit="deg"), 2.0 * u.arcsec)]

    report = compare_event_membership(event, source, [detector_circle()])

    assert report.total_events == 3
    assert report.sky_selected == 2
    assert report.detector_selected == 2
    assert report.disagreements == 0
    assert report.first_mismatch_rows == ()


def test_membership_validation_reports_shifted_detector_geometry(tmp_path):
    event = write_event(tmp_path / "events.fits")
    source = [CircleSkyRegion(SkyCoord(10.0, -9.0, unit="deg"), 2.0 * u.arcsec)]

    report = compare_event_membership(event, source, [detector_circle(centre_x=13.5)])

    assert report.disagreements > 0
    assert report.sky_only > 0 or report.detector_only > 0
    assert report.first_mismatch_rows

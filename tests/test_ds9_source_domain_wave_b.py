from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from xmm_region_tool.geometry import load_ds9_selection


DET_ARCSEC = 0.05


def _tan_wcs(*, crval=(90.0, -40.0), fk4=False) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [4096.0, 4096.0]
    wcs.wcs.cdelt = np.array([1.0 / 3600.0, 1.0 / 3600.0])
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    if fk4:
        wcs.wcs.radesys = "FK4"
        wcs.wcs.equinox = 1950.0
    return wcs


def _pixel_xy(coords: SkyCoord, wcs: WCS) -> np.ndarray:
    x, y = skycoord_to_pixel(coords, wcs, origin=0)
    return np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))


def _center_pixel(center: SkyCoord, wcs: WCS) -> np.ndarray:
    x, y = skycoord_to_pixel(center, wcs, origin=0)
    return np.asarray([float(x), float(y)])


def test_off_reference_30_arcmin_circle_exceeds_centred_tan_budget(tmp_path):
    """Pin why a radius-only DS9 display-equivalence cutoff is not truthful."""
    center = SkyCoord(90 * u.deg, (-40 + 5 / 60) * u.deg, frame="icrs")
    region = tmp_path / "off-crval-circle.reg"
    region.write_text(
        f"icrs\ncircle({center.ra.deg:.15g},{center.dec.deg:.15g},1800\")\n"
    )
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = _tan_wcs()

    projected = _pixel_xy(path.sample(1440), wcs)
    radius_pixels = np.linalg.norm(projected - _center_pixel(center, wcs), axis=1)
    centred_scale_reference_error = float(np.max(np.abs(radius_pixels - 1800.0)))

    # The same 30-arcmin size passes the historical test at CRVAL, but not
    # after moving the marker centre. A bare region file contains no source
    # image CRVAL/WCS with which to decide this condition at parse time.
    assert centred_scale_reference_error > DET_ARCSEC


def test_rotated_fk4_shape_is_covered_inside_controlled_centred_tan_case(tmp_path):
    """Exercise the accepted FK4/B1950 source-frame path independently."""
    region = tmp_path / "fk4-ellipse.reg"
    region.write_text('fk4\nellipse(90,-40,1200\",600\",37)\n')
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path

    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="fk4", equinox="B1950")
    wcs = _tan_wcs(fk4=True)
    count = 1440
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local_x = 1200.0 * np.cos(theta)
    local_y = 600.0 * np.sin(theta)
    rotation = np.deg2rad(-37.0)
    expected = np.column_stack(
        (
            local_x * np.cos(rotation) - local_y * np.sin(rotation),
            local_x * np.sin(rotation) + local_y * np.cos(rotation),
        )
    ) + _center_pixel(center, wcs)

    projected = _pixel_xy(path.sample(count), wcs)
    if np.linalg.norm(projected[0] - expected[count // 2]) < np.linalg.norm(
        projected[0] - expected[0]
    ):
        expected = np.roll(expected, count // 2, axis=0)

    assert float(np.max(np.linalg.norm(projected - expected, axis=1))) < DET_ARCSEC

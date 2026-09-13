from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from regions import EllipseSkyRegion

from xmm_region_tool.geometry import (
    UnsupportedRegionError,
    load_ds9_selection,
    selection_from_sky_regions,
)


DET_ARCSEC = 0.05


def tan_wcs(*, crval=(90.0, -40.0), galactic=False) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [4096.0, 4096.0]
    wcs.wcs.cdelt = np.array([1.0 / 3600.0, 1.0 / 3600.0])
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["GLON-TAN", "GLAT-TAN"] if galactic else ["RA---TAN", "DEC--TAN"]
    return wcs


def pixel_xy(coords: SkyCoord, wcs: WCS) -> np.ndarray:
    x, y = skycoord_to_pixel(coords, wcs, origin=0)
    return np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))


def center_pixel(center: SkyCoord, wcs: WCS) -> np.ndarray:
    x, y = skycoord_to_pixel(center, wcs, origin=0)
    return np.asarray([float(x), float(y)])


def test_xmm_sized_ds9_circle_matches_local_wcs_circle_within_one_det(tmp_path):
    region = tmp_path / "circle.reg"
    region.write_text("icrs\ncircle(90,-40,1800\")\n")
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs()
    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="icrs")

    projected = pixel_xy(path.sample(1440), wcs)
    radius_pixels = np.linalg.norm(projected - center_pixel(center, wcs), axis=1)
    max_source_semantic_error_arcsec = float(np.max(np.abs(radius_pixels - 1800.0)))

    assert max_source_semantic_error_arcsec < DET_ARCSEC


def test_large_ds9_circle_stress_case_detects_planar_vs_spherical_difference(tmp_path):
    region = tmp_path / "large-circle.reg"
    region.write_text("icrs\ncircle(90,-40,5d)\n")
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs()
    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="icrs")

    projected = pixel_xy(path.sample(1440), wcs)
    radius_pixels = np.linalg.norm(projected - center_pixel(center, wcs), axis=1)
    max_source_semantic_error_arcsec = float(np.max(np.abs(radius_pixels - 5 * 3600.0)))

    # This is a deliberately large non-XMM stress shape.  The assertion makes
    # the independent cross-check sensitive to the planar-vs-spherical choice
    # rather than merely checking that both calculations return finite points.
    assert max_source_semantic_error_arcsec > 1.0


def test_xmm_sized_rotated_ds9_ellipse_matches_ds9_clockwise_sky_construction(tmp_path):
    region = tmp_path / "ellipse.reg"
    region.write_text("icrs\nellipse(90,-40,1800\",900\",37)\n")
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs()
    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="icrs")

    count = 1440
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    # Real DS9/SAS visual validation established that a celestial DS9 marker
    # written with +37 deg corresponds to -37 deg in our positive local
    # longitude-axis convention.  DS9 ellipse arguments are semiaxis radii.
    local_x = 1800.0 * np.cos(theta)
    local_y = 900.0 * np.sin(theta)
    angle = np.deg2rad(-37.0)
    expected = np.column_stack(
        (
            local_x * np.cos(angle) - local_y * np.sin(angle),
            local_x * np.sin(angle) + local_y * np.cos(angle),
        )
    ) + center_pixel(center, wcs)

    projected = pixel_xy(path.sample(count), wcs)
    # An undirected axis canonicalized modulo 180 may start at the opposite end.
    if np.linalg.norm(projected[0] - expected[count // 2]) < np.linalg.norm(projected[0] - expected[0]):
        expected = np.roll(expected, count // 2, axis=0)
    max_source_semantic_error_arcsec = float(
        np.max(np.linalg.norm(projected - expected, axis=1))
    )

    assert max_source_semantic_error_arcsec < DET_ARCSEC


def test_near_30_arcmin_rounded_ellipse_stays_within_total_source_budget(
    tmp_path,
):
    region = tmp_path / "near-limit-rounded-ellipse.reg"
    region.write_text(
        "icrs\n"
        'ellipse(90,-40,900",450",1800",900.01",0)\n'
    )

    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs()
    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="icrs")

    count = 1440
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)

    # Compare the final canonical semantic outer boundary directly with the
    # originally authored DS9/reference-plane ellipse, so this measures the
    # combined representation + serialization-repair error.
    expected = np.column_stack(
        (
            1800.0 * np.cos(theta),
            900.01 * np.sin(theta),
        )
    ) + center_pixel(center, wcs)

    projected = pixel_xy(path.sample(count), wcs)
    total_error_arcsec = float(
        np.max(np.linalg.norm(projected - expected, axis=1))
    )

    assert total_error_arcsec < DET_ARCSEC


def test_near_30_arcmin_rounding_rejects_repair_that_double_spends_budget(
    tmp_path,
):
    region = tmp_path / "near-limit-excess-rounding.reg"
    region.write_text(
        "icrs\n"
        'ellipse(90,-40,900",450",1800",900.02",0)\n'
    )

    # A standalone 0.05-arcsec repair allowance would accept this geometry.
    # Near 30 arcmin, however, the TAN-vs-spherical representation already
    # consumes about 0.0457 arcsec of the total 0.05-arcsec source budget.
    with pytest.raises(
        UnsupportedRegionError,
        match="residual of the 0.05 arcsec total source-semantic budget",
    ):
        load_ds9_selection(region)


def test_programmatic_regions_ellipse_keeps_native_positive_rotation_convention():
    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="icrs")
    source = EllipseSkyRegion(
        center=center,
        width=3600 * u.arcsec,
        height=1800 * u.arcsec,
        angle=37 * u.deg,
    )
    selection = selection_from_sky_regions([source])
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs()

    count = 1440
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local_x = 1800.0 * np.cos(theta)
    local_y = 900.0 * np.sin(theta)
    angle = np.deg2rad(37.0)
    expected = np.column_stack(
        (
            local_x * np.cos(angle) - local_y * np.sin(angle),
            local_x * np.sin(angle) + local_y * np.cos(angle),
        )
    ) + center_pixel(center, wcs)

    projected = pixel_xy(path.sample(count), wcs)
    assert float(np.max(np.linalg.norm(projected - expected, axis=1))) < DET_ARCSEC


def test_xmm_sized_rotated_ds9_box_vertices_match_ds9_clockwise_sky_construction(tmp_path):
    region = tmp_path / "box.reg"
    region.write_text("icrs\nbox(90,-40,2400\",1200\",37)\n")
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs()
    center = SkyCoord(90 * u.deg, -40 * u.deg, frame="icrs")

    # DS9 box arguments are full width/height, unlike ellipse semiaxis radii.
    local_x = np.asarray([-1200.0, 1200.0, 1200.0, -1200.0])
    local_y = np.asarray([-600.0, -600.0, 600.0, 600.0])
    angle = np.deg2rad(-37.0)
    expected = np.column_stack(
        (
            local_x * np.cos(angle) - local_y * np.sin(angle),
            local_x * np.sin(angle) + local_y * np.cos(angle),
        )
    ) + center_pixel(center, wcs)

    projected = pixel_xy(path.vertices, wcs)
    # Modulo-180 box orientation changes only the cyclic starting corner.
    if np.linalg.norm(projected[0] - expected[2]) < np.linalg.norm(projected[0] - expected[0]):
        expected = np.roll(expected, 2, axis=0)
    max_source_semantic_error_arcsec = float(
        np.max(np.linalg.norm(projected - expected, axis=1))
    )

    assert max_source_semantic_error_arcsec < DET_ARCSEC


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, stop: np.ndarray) -> float:
    direction = stop - start
    fraction = float(np.dot(point - start, direction) / np.dot(direction, direction))
    closest = start + np.clip(fraction, 0.0, 1.0) * direction
    return float(np.linalg.norm(point - closest))


def test_fk5_polygon_geodesic_edges_are_straight_on_tan_projection(tmp_path):
    region = tmp_path / "polygon.reg"
    region.write_text(
        "fk5\n"
        "polygon(89.8,-40.1,90.2,-40.1,90.25,-39.9,89.75,-39.9)\n"
    )
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path

    # Centre the TAN WCS on the physical FK5/J2000 centre transformed to ICRS.
    fk5_center = SkyCoord(90 * u.deg, -40 * u.deg, frame="fk5")
    physical_center = fk5_center.icrs
    wcs = tan_wcs(crval=(physical_center.ra.deg, physical_center.dec.deg))
    projected_vertices = pixel_xy(path.vertices, wcs)

    distances = []
    intervals = path.initial_intervals()
    for index, (start_parameter, stop_parameter) in enumerate(intervals):
        midpoint = pixel_xy(path.point((start_parameter + stop_parameter) / 2.0), wcs)[0]
        distances.append(
            point_to_segment_distance(
                midpoint,
                projected_vertices[index],
                projected_vertices[(index + 1) % len(projected_vertices)],
            )
        )

    # A gnomonic/TAN projection maps great circles to straight lines.  This
    # independently checks the package's explicit polygon-edge convention
    # against DS9's straight reference-image polygon edges.
    assert max(distances) < 1e-6


def test_galactic_ds9_ellipse_preserves_declared_frame_with_ds9_rotation_sign(tmp_path):
    region = tmp_path / "galactic.reg"
    region.write_text("galactic\nellipse(120,30,1200\",600\",37)\n")
    selection = load_ds9_selection(region)
    path = selection.regions[0].boundaries[0].path
    wcs = tan_wcs(crval=(120.0, 30.0), galactic=True)
    center = SkyCoord(120 * u.deg, 30 * u.deg, frame="galactic")

    count = 720
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local_x = 1200.0 * np.cos(theta)
    local_y = 600.0 * np.sin(theta)
    angle = np.deg2rad(-37.0)
    expected = np.column_stack(
        (
            local_x * np.cos(angle) - local_y * np.sin(angle),
            local_x * np.sin(angle) + local_y * np.cos(angle),
        )
    ) + center_pixel(center, wcs)
    projected = pixel_xy(path.sample(count), wcs)

    max_source_semantic_error_arcsec = float(
        np.max(np.linalg.norm(projected - expected, axis=1))
    )
    assert max_source_semantic_error_arcsec < DET_ARCSEC


def test_epanda_accepts_ds9_rounded_common_axis_ratio(tmp_path):
    region = tmp_path / "rounded-epanda.reg"
    region.write_text(
        "# Region file format: DS9\n"
        "fk5\n"
        'epanda(15.6735138,-21.8814583,252.46293,345.38594,1,'
        '27.648",54.289",64.032",125.729",1,241.46119)\n'
    )

    first = load_ds9_selection(region)
    second = load_ds9_selection(region)

    assert len(first.regions) == 1
    assert first.geometry_sha256 == second.geometry_sha256

    model = first.regions[0].boundaries[0].path

    # DS9 reconciliation preserves the authored first semiaxes exactly. This
    # real saved-region fixture needs only about 0.001 arcsec of repair, well
    # inside the scale-dependent residual of the total source-semantic budget.
    assert model.inner_semi_major.to_value(u.arcsec) == pytest.approx(
        27.648,
        abs=1e-9,
    )
    assert model.outer_semi_major.to_value(u.arcsec) == pytest.approx(
        64.032,
        abs=1e-9,
    )
    assert abs(model.inner_semi_minor.to_value(u.arcsec) - 54.289) < 0.002
    assert abs(model.outer_semi_minor.to_value(u.arcsec) - 125.729) < 0.002

    inner_ratio = (
        model.inner_semi_minor.to_value(u.arcsec)
        / model.inner_semi_major.to_value(u.arcsec)
    )
    outer_ratio = (
        model.outer_semi_minor.to_value(u.arcsec)
        / model.outer_semi_major.to_value(u.arcsec)
    )
    assert inner_ratio == pytest.approx(outer_ratio, abs=1e-12)


def test_epanda_rejects_materially_different_axis_ratios(tmp_path):
    region = tmp_path / "invalid-epanda.reg"
    region.write_text(
        "# Region file format: DS9\n"
        "fk5\n"
        'epanda(15.6735138,-21.8814583,252.46293,345.38594,1,'
        '27.648",54.289",64.032",130.000",1,241.46119)\n'
    )

    with pytest.raises(
        UnsupportedRegionError,
        match="nested ellipses must have the same major/minor axis ratio",
    ):
        load_ds9_selection(region)

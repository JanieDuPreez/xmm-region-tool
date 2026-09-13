from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from xmm_region_tool import (
    ellipse_local_sector_annulus_selection,
    elliptical_annulus_selection,
    elliptical_sector_annulus_selection,
)
from xmm_region_tool.elliptical_sector import EllipticalSectorAnnulusPath
from xmm_region_tool.geometry import UnsupportedRegionError, load_ds9_selection
from xmm_region_tool.model import EllipsePath, SelectionGeometryError, local_angle_to_icrs
from xmm_region_tool.workflow import extraction_cells


def _unique_ellipse_semiaxes(selection):
    edges = {}
    for region in selection.regions:
        for boundary in region.boundaries:
            path = boundary.path
            if not isinstance(path, EllipsePath):
                continue
            first = float(path.width.to_value(u.arcsec) / 2.0)
            second = float(path.height.to_value(u.arcsec) / 2.0)
            edges[round(first, 9)] = (first, second)
    return [edges[key] for key in sorted(edges)]


def test_repeated_ds9_ellipse_annulus_pairs_expand_to_independent_shells(tmp_path):
    path = tmp_path / "elliptical-profile.reg"
    path.write_text(
        "icrs\n"
        "ellipse(10,-9,60\",30\",120\",60\",180\",90\",25)\n"
    )

    selection = load_ds9_selection(path)
    cells = extraction_cells(selection)

    assert len(selection.regions) == 2
    assert len(cells) == 2
    assert all(len(cell.selection.regions[0].boundaries) == 2 for cell in cells)
    outer_widths = [
        cell.selection.regions[0].boundaries[0].path.width.to_value(u.arcsec)
        for cell in cells
    ]
    assert outer_widths == pytest.approx([240.0, 360.0])


def test_ds9_ellipse_annulus_n_subdivision_expands_to_independent_shells(tmp_path):
    path = tmp_path / "elliptical-profile-n.reg"
    path.write_text(
        "fk5\n"
        "ellipse(10,-9,60\",30\",240\",120\",n=3,17)\n"
    )

    selection = load_ds9_selection(path)

    assert len(selection.regions) == 3
    assert len(extraction_cells(selection)) == 3
    outer_semimajor = [
        region.boundaries[0].path.width.to_value(u.arcsec) / 2.0
        for region in selection.regions
    ]
    assert outer_semimajor == pytest.approx([120.0, 180.0, 240.0])


def test_rounded_repeated_ds9_ellipse_annulus_is_geometry_bounded(tmp_path):
    path = tmp_path / "rounded-elliptical-profile.reg"
    path.write_text(
        "icrs\n"
        'ellipse(10,-9,27.648",54.289",64.032",125.729",'
        '96.048",188.594",25)\n'
    )

    coarse = load_ds9_selection(path, samples=32)
    dense = load_ds9_selection(path, samples=512)

    assert coarse.geometry_sha256 == dense.geometry_sha256

    canonical = _unique_ellipse_semiaxes(coarse)
    declared = [
        (27.648, 54.289),
        (64.032, 125.729),
        (96.048, 188.594),
    ]

    assert [first for first, _ in canonical] == pytest.approx(
        [first for first, _ in declared],
        abs=1e-9,
    )

    corrections = [
        abs(canonical_second - declared_second)
        for (_, canonical_second), (_, declared_second) in zip(
            canonical,
            declared,
            strict=True,
        )
    ]
    assert max(corrections) < 0.002

    ratios = [second / first for first, second in canonical]
    assert max(ratios) - min(ratios) < 1e-12


def test_rounded_ds9_ellipse_annulus_n_is_geometry_bounded(tmp_path):
    path = tmp_path / "rounded-elliptical-profile-n.reg"
    path.write_text(
        "fk5\n"
        'ellipse(10,-9,27.648",54.289",64.032",125.729",n=3,17)\n'
    )

    coarse = load_ds9_selection(path, samples=32)
    dense = load_ds9_selection(path, samples=512)

    assert coarse.geometry_sha256 == dense.geometry_sha256
    assert len(coarse.regions) == 3

    canonical = _unique_ellipse_semiaxes(coarse)
    assert len(canonical) == 4

    first_edge = canonical[0]
    last_edge = canonical[-1]

    assert first_edge[0] == pytest.approx(27.648, abs=1e-9)
    assert last_edge[0] == pytest.approx(64.032, abs=1e-9)
    assert abs(first_edge[1] - 54.289) < 0.002
    assert abs(last_edge[1] - 125.729) < 0.002

    ratios = [second / first for first, second in canonical]
    assert max(ratios) - min(ratios) < 1e-12


def test_ds9_ellipse_rounding_rejects_scale_dependent_excess(tmp_path):
    path = tmp_path / "too-large-rounding-repair.reg"
    path.write_text(
        "icrs\n"
        'ellipse(10,-9,450",900",900",1800.16",n=2,17)\n'
    )

    # The ratio discrepancy is below 1e-4 in relative terms, but reconciling
    # these XMM-scale serialized ellipses cannot fit inside the residual of
    # the total 0.05-arcsec source-semantic budget.
    with pytest.raises(
        UnsupportedRegionError,
        match="residual of the 0.05 arcsec total source-semantic budget",
    ):
        load_ds9_selection(path)


def test_ds9_ellipse_annulus_n_rejects_material_axis_ratio_change(tmp_path):
    path = tmp_path / "bad-elliptical-profile-n.reg"
    path.write_text(
        "icrs\n"
        'ellipse(10,-9,60",30",240",100",n=3,17)\n'
    )

    with pytest.raises(
        UnsupportedRegionError,
        match="same major/minor axis ratio",
    ):
        load_ds9_selection(path)


def test_ellipse_annulus_rejects_axis_ratio_change(tmp_path):
    path = tmp_path / "bad-elliptical-profile.reg"
    path.write_text("icrs\nellipse(10,-9,60\",30\",120\",50\",180\",90\")\n")

    with pytest.raises(UnsupportedRegionError, match="same major/minor axis ratio"):
        load_ds9_selection(path)


def test_epanda_grid_expands_to_angle_times_radius_cells(tmp_path):
    path = tmp_path / "epanda-grid.reg"
    path.write_text(
        "icrs\n"
        "epanda(10,-9,330,30,2,60\",30\",180\",90\",3,27)\n"
    )

    selection = load_ds9_selection(path)
    cells = extraction_cells(selection)

    assert len(selection.regions) == 6
    assert len(cells) == 6
    assert all(
        isinstance(region.boundaries[0].path, EllipticalSectorAnnulusPath)
        for region in selection.regions
    )
    starts = [
        region.boundaries[0].path.start_angle.to_value(u.deg)
        for region in selection.regions
    ]
    assert starts[0:3] == pytest.approx([starts[0]] * 3)
    assert starts[3:6] == pytest.approx([starts[3]] * 3)
    # DS9 epanda local angles increase opposite to this path's positive
    # longitude-axis sweep, so successive cells move by -angle_step here.
    assert np.mod(starts[0] - starts[3], 360.0) == pytest.approx(30.0, abs=1e-8)


def test_epanda_zero_inner_axes_are_supported(tmp_path):
    path = tmp_path / "epanda-zero.reg"
    path.write_text("icrs\nepanda(10,-9,0,90,1,0\",0\",120\",60\",1,20)\n")

    selection = load_ds9_selection(path)
    model = selection.regions[0].boundaries[0].path

    assert isinstance(model, EllipticalSectorAnnulusPath)
    assert model.segment_count == 3
    assert model.inner_semi_major.to_value(u.arcsec) == pytest.approx(0.0)
    assert model.inner_semi_minor.to_value(u.arcsec) == pytest.approx(0.0)


def test_epanda_rejects_axis_ratio_change(tmp_path):
    path = tmp_path / "bad-epanda.reg"
    path.write_text("icrs\nepanda(10,-9,0,90,1,60\",30\",180\",70\",2,20)\n")

    with pytest.raises(UnsupportedRegionError, match="same major/minor axis ratio"):
        load_ds9_selection(path)


def test_epanda_full_circle_reuses_elliptical_annulus_semantics(tmp_path):
    path = tmp_path / "full-epanda.reg"
    path.write_text("icrs\nepanda(10,-9,45,45,1,60\",30\",120\",60\",1,35)\n")

    selection = load_ds9_selection(path)
    boundaries = selection.regions[0].boundaries

    assert len(boundaries) == 2
    assert all(isinstance(boundary.path, EllipsePath) for boundary in boundaries)
    assert boundaries[1].subtract is True


def test_epanda_ds9_nonfull_cell_uses_directed_ellipse_axis(tmp_path):
    path = tmp_path / "epanda.reg"
    path.write_text("icrs\nepanda(10,-9,20,140,1,60\",30\",180\",90\",1,27)\n")
    from_ds9 = load_ds9_selection(path)

    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")
    direct = ellipse_local_sector_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        20 * u.deg,
        140 * u.deg,
        153 * u.deg,
    )

    # An ordinary ellipse cannot distinguish -27 from 153 because its axis
    # has 180-degree symmetry. A non-full epanda can because its local cuts
    # attach to a directed end of that axis.
    assert from_ds9.geometry_sha256 == direct.geometry_sha256


def test_full_epanda_matches_programmatic_elliptical_annulus(tmp_path):
    path = tmp_path / "full-epanda.reg"
    path.write_text("icrs\nepanda(10,-9,0,360,1,60\",30\",180\",90\",1,27)\n")
    from_ds9 = load_ds9_selection(path)

    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")
    direct = elliptical_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        -27 * u.deg,
    )

    assert from_ds9.geometry_sha256 == direct.geometry_sha256


def test_epanda_sampling_hint_does_not_change_semantic_identity(tmp_path):
    path = tmp_path / "epanda.reg"
    path.write_text("icrs\nepanda(10,-9,20,140,1,60\",30\",180\",90\",1,27)\n")

    coarse = load_ds9_selection(path, samples=32)
    dense = load_ds9_selection(path, samples=512)

    assert coarse.geometry_sha256 == dense.geometry_sha256
    assert len(coarse.regions[0].boundaries[0].vertices) == 32
    assert len(dense.regions[0].boundaries[0].vertices) == 512


def test_rotated_epanda_zero_local_angle_hits_rotated_semimajor_axis():
    """A local panda angle of zero lies on the programmatic rotated major axis.

    ``ellipse_local_sector_annulus_selection`` is the explicit programmatic
    ellipse-local constructor. The DS9 adapter performs its separate directed
    angle conversion before calling it.
    """
    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")
    selection = ellipse_local_sector_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        0 * u.deg,
        90 * u.deg,
        30 * u.deg,
    )
    path = selection.regions[0].boundaries[0].path
    assert isinstance(path, EllipticalSectorAnnulusPath)

    ds9_start_on_outer_arc = path.point(1.0 / path.segment_count)
    assert center.separation(ds9_start_on_outer_arc).to_value(u.arcsec) == pytest.approx(
        180.0, rel=0, abs=1e-8
    )
    expected_angle = local_angle_to_icrs(center, 30 * u.deg)
    actual_angle = 90 * u.deg - center.icrs.position_angle(ds9_start_on_outer_arc)
    assert np.mod(actual_angle.to_value(u.deg), 360.0) == pytest.approx(
        expected_angle.to_value(u.deg), abs=1e-10
    )


def test_rotated_epanda_nonzero_local_angle_uses_negative_local_direction():
    """Programmatic physical ray direction is ellipse_angle - local_angle."""
    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")
    selection = ellipse_local_sector_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        30 * u.deg,
        90 * u.deg,
        20 * u.deg,
    )
    path = selection.regions[0].boundaries[0].path
    assert isinstance(path, EllipticalSectorAnnulusPath)

    # Positive path traversal begins on the local stop cut: 20 - 90 = -70 deg.
    expected_path_start = local_angle_to_icrs(center, -70 * u.deg)
    assert path.start_angle.to_value(u.deg) == pytest.approx(
        expected_path_start.to_value(u.deg), abs=1e-10
    )
    assert center.separation(path.point(0.0)).to_value(u.arcsec) == pytest.approx(
        90.0, rel=0, abs=1e-8
    )

    # The end of the outer arc is the local start cut: 20 - 30 = -10 deg.
    local_start = path.point(1.0 / path.segment_count)
    expected_local_start = local_angle_to_icrs(center, -10 * u.deg)
    actual_local_start = 90 * u.deg - center.icrs.position_angle(local_start)
    assert np.mod(actual_local_start.to_value(u.deg), 360.0) == pytest.approx(
        expected_local_start.to_value(u.deg), abs=1e-9
    )
    expected_radius = 1.0 / np.sqrt(
        (np.cos(np.deg2rad(30.0)) / 180.0) ** 2
        + (np.sin(np.deg2rad(30.0)) / 90.0) ** 2
    )
    assert center.separation(local_start).to_value(u.arcsec) == pytest.approx(
        expected_radius, rel=0, abs=1e-8
    )


def test_elliptical_sector_model_rejects_mismatched_axes():
    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")
    with pytest.raises(SelectionGeometryError, match="same axis ratio"):
        EllipticalSectorAnnulusPath(
            center=center,
            inner_radius=60 * u.arcsec,
            outer_radius=180 * u.arcsec,
            start_angle=0 * u.deg,
            sweep=90 * u.deg,
            inner_semi_minor=30 * u.arcsec,
            outer_semi_minor=70 * u.arcsec,
            ellipse_angle=20 * u.deg,
        )


def test_bpanda_remains_explicitly_unsupported(tmp_path):
    path = tmp_path / "bpanda.reg"
    path.write_text("icrs\nbpanda(10,-9,0,90,1,60\",30\",180\",90\",2,20)\n")

    with pytest.raises(UnsupportedRegionError, match="bpanda is not supported"):
        load_ds9_selection(path)


def test_absolute_and_ellipse_local_sector_angles_are_distinct():
    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")

    absolute = elliptical_sector_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        30 * u.deg,
        90 * u.deg,
        20 * u.deg,
    )
    local = ellipse_local_sector_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        30 * u.deg,
        90 * u.deg,
        20 * u.deg,
    )

    absolute_path = absolute.regions[0].boundaries[0].path
    local_path = local.regions[0].boundaries[0].path
    assert isinstance(absolute_path, EllipticalSectorAnnulusPath)
    assert isinstance(local_path, EllipticalSectorAnnulusPath)

    # Absolute API begins on the physical 30-degree ray.
    expected_absolute_start = local_angle_to_icrs(center, 30 * u.deg)
    assert absolute_path.start_angle.to_value(u.deg) == pytest.approx(
        expected_absolute_start.to_value(u.deg),
        abs=1e-10,
    )

    # Ellipse-local API begins on the local stop cut:
    # physical direction = ellipse_angle - 90 deg = -70 deg.
    expected_local_start = local_angle_to_icrs(center, -70 * u.deg)
    assert local_path.start_angle.to_value(u.deg) == pytest.approx(
        expected_local_start.to_value(u.deg),
        abs=1e-10,
    )

    assert absolute.geometry_sha256 != local.geometry_sha256


def test_absolute_elliptical_sector_matches_circular_angle_convention():
    center = SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs")
    selection = elliptical_sector_annulus_selection(
        center,
        60 * u.arcsec,
        30 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        25 * u.deg,
        100 * u.deg,
        40 * u.deg,
    )
    path = selection.regions[0].boundaries[0].path
    assert isinstance(path, EllipticalSectorAnnulusPath)

    expected_start = local_angle_to_icrs(center, 25 * u.deg)
    assert path.start_angle.to_value(u.deg) == pytest.approx(
        expected_start.to_value(u.deg),
        abs=1e-10,
    )
    assert path.sweep.to_value(u.deg) == pytest.approx(75.0)

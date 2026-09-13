from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from xmm_region_tool import (
    box_annulus_selection,
    box_selection,
    circle_selection,
    circular_annulus_selection,
    ellipse_selection,
    ellipse_local_sector_annulus_selection,
    elliptical_sector_annulus_selection,
    elliptical_annulus_selection,
    load_ds9_selection,
    polygon_selection,
    sector_annulus_selection,
    SelectionGeometryError,
)


CENTER = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")


def ds9_selection(tmp_path, geometry: str):
    path = tmp_path / "source.reg"
    path.write_text(f"icrs\n{geometry}\n")
    return load_ds9_selection(path)


def assert_same_geometry(ds9, in_memory):
    assert in_memory.geometry_sha256 == ds9.geometry_sha256


def test_circle_constructor_matches_ds9(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'circle(10,-9,120")'),
        circle_selection(CENTER, 120 * u.arcsec),
    )


def test_ellipse_constructor_matches_ds9_after_adapter_sign_conversion(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'ellipse(10,-9,120",60",37)'),
        ellipse_selection(CENTER, 120 * u.arcsec, 60 * u.arcsec, -37 * u.deg),
    )


def test_box_constructor_matches_ds9_after_adapter_sign_conversion(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'box(10,-9,240",120",33)'),
        box_selection(CENTER, 240 * u.arcsec, 120 * u.arcsec, -33 * u.deg),
    )


def test_polygon_constructor_matches_ds9(tmp_path):
    vertices = SkyCoord(
        [9.98, 10.02, 10.03, 9.99] * u.deg,
        [-9.01, -9.02, -8.98, -8.97] * u.deg,
        frame="icrs",
    )
    assert_same_geometry(
        ds9_selection(tmp_path, "polygon(9.98,-9.01,10.02,-9.02,10.03,-8.98,9.99,-8.97)"),
        polygon_selection(vertices),
    )


def test_circular_annulus_constructor_matches_ds9(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'annulus(10,-9,60",180")'),
        circular_annulus_selection(CENTER, 60 * u.arcsec, 180 * u.arcsec),
    )


def test_elliptical_annulus_constructor_matches_ds9_after_adapter_sign_conversion(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'ellipse(10,-9,60",30",180",90",32)'),
        elliptical_annulus_selection(
            CENTER,
            60 * u.arcsec,
            30 * u.arcsec,
            180 * u.arcsec,
            90 * u.arcsec,
            -32 * u.deg,
        ),
    )


def test_box_annulus_constructor_matches_ds9_after_adapter_sign_conversion(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'box(10,-9,120",60",300",150",27)'),
        box_annulus_selection(
            CENTER,
            120 * u.arcsec,
            60 * u.arcsec,
            300 * u.arcsec,
            150 * u.arcsec,
            -27 * u.deg,
        ),
    )


def test_sector_annulus_constructor_matches_ds9_panda(tmp_path):
    assert_same_geometry(
        ds9_selection(tmp_path, 'panda(10,-9,330,30,1,60",180",1)'),
        sector_annulus_selection(
            CENTER,
            60 * u.arcsec,
            180 * u.arcsec,
            150 * u.deg,
            210 * u.deg,
        ),
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: circle_selection(CENTER, 120 * u.arcsec, include=False),
        lambda: ellipse_selection(CENTER, 120 * u.arcsec, 60 * u.arcsec, include=False),
        lambda: box_selection(CENTER, 240 * u.arcsec, 120 * u.arcsec, include=False),
        lambda: circular_annulus_selection(
            CENTER, 60 * u.arcsec, 180 * u.arcsec, include=False
        ),
        lambda: sector_annulus_selection(
            CENTER, 60 * u.arcsec, 180 * u.arcsec, 20 * u.deg, 140 * u.deg, include=False
        ),
    ],
)
def test_public_constructors_preserve_explicit_exclusion(factory):
    selection = factory()
    assert all(region.include is False for region in selection.regions)


def test_external_metadata_does_not_change_constructor_geometry_identity():
    plain = circle_selection(CENTER, 120 * u.arcsec)
    managed = circle_selection(
        CENTER,
        120 * u.arcsec,
        external_identity="science-region-17",
        external_provenance={"owner": "caller"},
    )
    assert plain.geometry_sha256 == managed.geometry_sha256
    assert managed.external_identity == "science-region-17"
    assert managed.external_provenance == {"owner": "caller"}


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0 * u.arcsec, 120 * u.arcsec),
        (-1 * u.arcsec, 120 * u.arcsec),
        (120 * u.arcsec, 0 * u.arcsec),
        (120 * u.arcsec, -1 * u.arcsec),
        (float("nan") * u.arcsec, 120 * u.arcsec),
        (120 * u.arcsec, float("inf") * u.arcsec),
    ],
)
def test_box_constructor_rejects_invalid_dimensions(width, height):
    with pytest.raises(SelectionGeometryError):
        box_selection(CENTER, width, height)


@pytest.mark.parametrize(
    ("inner", "outer"),
    [
        (-1 * u.arcsec, 120 * u.arcsec),
        (120 * u.arcsec, 120 * u.arcsec),
        (180 * u.arcsec, 120 * u.arcsec),
        (float("nan") * u.arcsec, 120 * u.arcsec),
        (60 * u.arcsec, float("inf") * u.arcsec),
    ],
)
def test_circular_annulus_rejects_invalid_radii(inner, outer):
    with pytest.raises(SelectionGeometryError):
        circular_annulus_selection(CENTER, inner, outer)


def test_circular_annulus_zero_inner_normalises_to_circle():
    limiting = circular_annulus_selection(CENTER, 0 * u.arcsec, 180 * u.arcsec)
    circle = circle_selection(CENTER, 180 * u.arcsec)
    assert limiting.geometry_sha256 == circle.geometry_sha256


@pytest.mark.parametrize(
    ("inner_major", "inner_minor", "outer_major", "outer_minor"),
    [
        (60, 30, 60, 30),
        (180, 90, 120, 60),
        (-1, 30, 180, 90),
        (60, -1, 180, 90),
        (60, 30, float("nan"), 90),
        (60, 30, 180, float("inf")),
        (60, 20, 180, 90),
    ],
)
def test_elliptical_annulus_rejects_invalid_axes(
    inner_major,
    inner_minor,
    outer_major,
    outer_minor,
):
    with pytest.raises(SelectionGeometryError):
        elliptical_annulus_selection(
            CENTER,
            inner_major * u.arcsec,
            inner_minor * u.arcsec,
            outer_major * u.arcsec,
            outer_minor * u.arcsec,
        )


def test_elliptical_annulus_zero_inner_normalises_to_ellipse():
    limiting = elliptical_annulus_selection(
        CENTER,
        0 * u.arcsec,
        0 * u.arcsec,
        180 * u.arcsec,
        90 * u.arcsec,
        17 * u.deg,
    )
    ellipse = ellipse_selection(
        CENTER,
        180 * u.arcsec,
        90 * u.arcsec,
        17 * u.deg,
    )
    assert limiting.geometry_sha256 == ellipse.geometry_sha256


@pytest.mark.parametrize(
    ("inner_width", "inner_height", "outer_width", "outer_height"),
    [
        (120, 60, 120, 180),
        (120, 60, 300, 60),
        (300, 180, 120, 60),
        (-1, 60, 300, 180),
        (120, -1, 300, 180),
        (0, 60, 300, 180),
        (60, 0, 300, 180),
        (120, 60, float("nan"), 180),
        (120, 60, 300, float("inf")),
    ],
)
def test_box_annulus_rejects_invalid_dimensions(
    inner_width,
    inner_height,
    outer_width,
    outer_height,
):
    with pytest.raises(SelectionGeometryError):
        box_annulus_selection(
            CENTER,
            inner_width * u.arcsec,
            inner_height * u.arcsec,
            outer_width * u.arcsec,
            outer_height * u.arcsec,
        )


def test_box_annulus_zero_inner_normalises_to_box():
    limiting = box_annulus_selection(
        CENTER,
        0 * u.arcsec,
        0 * u.arcsec,
        300 * u.arcsec,
        180 * u.arcsec,
        23 * u.deg,
    )
    box = box_selection(
        CENTER,
        300 * u.arcsec,
        180 * u.arcsec,
        23 * u.deg,
    )
    assert limiting.geometry_sha256 == box.geometry_sha256


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("constructor,kwargs", [
    (box_selection, {"width": 2 * u.arcmin, "height": 1 * u.arcmin}),
    (box_annulus_selection, {"inner_width": 1 * u.arcmin, "inner_height": 1 * u.arcmin,
                             "outer_width": 2 * u.arcmin, "outer_height": 2 * u.arcmin}),
])
def test_boxes_reject_array_centres(count, constructor, kwargs):
    center = SkyCoord(np.arange(count) * u.deg, np.zeros(count) * u.deg)
    with pytest.raises(SelectionGeometryError, match="scalar"):
        constructor(center, **kwargs)


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("constructor,kwargs", [
    (circle_selection, {"radius": 1 * u.arcmin}),
    (ellipse_selection, {"semi_major": 2 * u.arcmin, "semi_minor": 1 * u.arcmin,
                         "angle": 0 * u.deg}),
    (box_selection, {"width": 2 * u.arcmin, "height": 1 * u.arcmin, "angle": 0 * u.deg}),
    (box_annulus_selection, {"inner_width": 1 * u.arcmin, "inner_height": 1 * u.arcmin,
                             "outer_width": 2 * u.arcmin, "outer_height": 2 * u.arcmin,
                             "angle": 0 * u.deg}),
    (elliptical_annulus_selection, {"inner_semi_major": 1 * u.arcmin,
        "inner_semi_minor": 0.5 * u.arcmin, "outer_semi_major": 2 * u.arcmin,
        "outer_semi_minor": 1 * u.arcmin, "angle": 0 * u.deg}),
    (sector_annulus_selection, {"inner_radius": 1 * u.arcmin, "outer_radius": 2 * u.arcmin,
                               "start_angle": 0 * u.deg, "stop_angle": 90 * u.deg}),
    (elliptical_sector_annulus_selection, {"inner_semi_major": 1 * u.arcmin,
        "inner_semi_minor": 0.5 * u.arcmin, "outer_semi_major": 2 * u.arcmin,
        "outer_semi_minor": 1 * u.arcmin, "start_angle": 0 * u.deg,
        "stop_angle": 90 * u.deg, "ellipse_angle": 0 * u.deg}),
    (ellipse_local_sector_annulus_selection, {"inner_semi_major": 1 * u.arcmin,
        "inner_semi_minor": 0.5 * u.arcmin, "outer_semi_major": 2 * u.arcmin,
        "outer_semi_minor": 1 * u.arcmin, "local_start_angle": 0 * u.deg,
        "local_stop_angle": 90 * u.deg, "ellipse_angle": 0 * u.deg}),
])
def test_public_angular_fields_reject_arrays(count, constructor, kwargs):
    center = SkyCoord(10 * u.deg, -9 * u.deg)
    for field, scalar in kwargs.items():
        values = dict(kwargs, **{field: np.repeat(scalar.value, count) * scalar.unit})
        with pytest.raises(SelectionGeometryError, match="scalar"):
            constructor(center, **values)


@pytest.mark.parametrize("extent", [180, 200, 360])
def test_radial_offsets_reject_antipodal_domain(extent):
    with pytest.raises(SelectionGeometryError, match="180"):
        circle_selection(CENTER, extent * u.deg)
    with pytest.raises(SelectionGeometryError, match="180"):
        sector_annulus_selection(CENTER, 0 * u.deg, extent * u.deg, 0 * u.deg, 90 * u.deg)
    with pytest.raises(SelectionGeometryError, match="180"):
        ellipse_selection(CENTER, extent * u.deg, 1 * u.deg)
    with pytest.raises(SelectionGeometryError, match="180"):
        box_selection(CENTER, 2 * extent * u.deg, 1 * u.deg)


def test_box_validates_corner_extent_not_only_dimensions():
    with pytest.raises(SelectionGeometryError, match="180"):
        box_selection(CENTER, 260 * u.deg, 260 * u.deg)


@pytest.mark.parametrize("frame", ["icrs", "galactic"])
@pytest.mark.parametrize("latitude", [-90, 90])
def test_oriented_shapes_reject_source_frame_poles(frame, latitude):
    center = SkyCoord(10 * u.deg, latitude * u.deg, frame=frame)
    for constructor, args in [
        (ellipse_selection, [2 * u.arcmin, 1 * u.arcmin]),
        (box_selection, [2 * u.arcmin, 1 * u.arcmin]),
        (sector_annulus_selection, [0 * u.arcmin, 2 * u.arcmin, 0 * u.deg, 90 * u.deg]),
    ]:
        with pytest.raises(SelectionGeometryError, match="local-angle basis"):
            constructor(center, *args)
    assert circle_selection(center, 1 * u.arcmin).geometry_sha256


@pytest.mark.parametrize("sweep", [359.999, 359.997, 1e-12])
def test_nonfull_sectors_preserve_radial_edges(sweep):
    from xmm_region_tool.model import SectorAnnulusPath

    circular = sector_annulus_selection(CENTER, 1 * u.arcmin, 2 * u.arcmin,
                                         0 * u.deg, sweep * u.deg)
    absolute = elliptical_sector_annulus_selection(
        CENTER, 1 * u.arcmin, 0.5 * u.arcmin, 2 * u.arcmin, 1 * u.arcmin,
        0 * u.deg, sweep * u.deg)
    local = ellipse_local_sector_annulus_selection(
        CENTER, 1 * u.arcmin, 0.5 * u.arcmin, 2 * u.arcmin, 1 * u.arcmin,
        0 * u.deg, sweep * u.deg)
    for selection in (circular, absolute, local):
        path = selection.regions[0].boundaries[0].path
        assert isinstance(path, SectorAnnulusPath)
        assert path.sweep.to_value(u.deg) == sweep
        assert path.segment_count == 4


def test_positive_inner_extent_retains_hole_topology():
    tiny = 1e-16 * u.deg
    circular = sector_annulus_selection(CENTER, tiny, 1 * u.arcmin, 0 * u.deg, 90 * u.deg)
    elliptical = elliptical_sector_annulus_selection(
        CENTER, tiny, tiny / 2, 2 * u.arcmin, 1 * u.arcmin, 0 * u.deg, 90 * u.deg)
    for selection in (circular, elliptical):
        assert selection.regions[0].boundaries[0].path.segment_count == 4
    selection = elliptical_annulus_selection(CENTER, tiny, tiny, 1 * u.arcmin, 1 * u.arcmin)
    assert len(selection.regions[0].boundaries) == 2
    # The positive box hole must be retained or fail closed if its corners collapse
    # at machine precision; it must never silently become a filled outer box.
    with pytest.raises(SelectionGeometryError, match="coincident"):
        box_annulus_selection(CENTER, tiny, tiny, 1 * u.arcmin, 1 * u.arcmin)


def test_tiny_positive_elliptical_extent_is_not_collapsed():
    from xmm_region_tool.elliptical_sector import EllipticalSectorAnnulusPath

    path = EllipticalSectorAnnulusPath(CENTER, 0 * u.deg, 1e-16 * u.deg,
        0 * u.deg, 90 * u.deg, outer_semi_minor=5e-17 * u.deg)
    assert path._radial_extent(path.outer_radius, path.outer_semi_minor, 0 * u.deg) > 0 * u.deg


@pytest.mark.parametrize("angles", [(0, 180, 360), (-37, 143, 323)])
@pytest.mark.parametrize("frame", ["icrs", "galactic"])
def test_undirected_axis_spellings_have_identical_hashes(angles, frame):
    center = SkyCoord(10 * u.deg, -9 * u.deg, frame=frame)
    for constructor, dimensions in (
        (ellipse_selection, [2 * u.arcmin, 1 * u.arcmin]),
        (box_selection, [2 * u.arcmin, 1 * u.arcmin]),
        (elliptical_annulus_selection, [1 * u.arcmin, 0.5 * u.arcmin, 2 * u.arcmin, 1 * u.arcmin]),
        (box_annulus_selection, [1 * u.arcmin, 0.5 * u.arcmin, 2 * u.arcmin, 1 * u.arcmin]),
    ):
        hashes = {constructor(center, *dimensions, angle * u.deg).geometry_sha256 for angle in angles}
        assert len(hashes) == 1


def test_ds9_equivalent_undirected_axes(tmp_path):
    for shape, sizes in (("ellipse", '120",60"'), ("box", '240",120"')):
        hashes = {ds9_selection(tmp_path, f'{shape}(10,-9,{sizes},{angle})').geometry_sha256
                  for angle in (-37, 143, 323)}
        assert len(hashes) == 1


def test_named_major_minor_axes_are_truthful():
    calls = [
        lambda: ellipse_selection(CENTER, 1*u.arcmin, 2*u.arcmin),
        lambda: elliptical_annulus_selection(CENTER, 1*u.arcmin, 2*u.arcmin, 2*u.arcmin, 4*u.arcmin),
        lambda: elliptical_sector_annulus_selection(CENTER, 1*u.arcmin, 2*u.arcmin,
            2*u.arcmin, 4*u.arcmin, 0*u.deg, 90*u.deg),
        lambda: ellipse_local_sector_annulus_selection(CENTER, 1*u.arcmin, 2*u.arcmin,
            2*u.arcmin, 4*u.arcmin, 0*u.deg, 90*u.deg),
    ]
    for call in calls:
        with pytest.raises(SelectionGeometryError, match="semi_major"):
            call()
    assert ellipse_selection(CENTER, 1*u.arcmin, 1*u.arcmin).geometry_sha256


def test_ds9_ordered_axes_remain_supported(tmp_path):
    assert ds9_selection(tmp_path, 'ellipse(10,-9,60",120",37)').geometry_sha256
    assert ds9_selection(tmp_path, 'epanda(10,-9,0,90,1,30",60",60",120",1,37)').geometry_sha256


@pytest.mark.parametrize("frame", ["icrs", "galactic"])
@pytest.mark.parametrize("angle", [0, -37, 143])
def test_absolute_sector_axis_equivalence_preserves_points(frame, angle):
    center = SkyCoord(10*u.deg, -9*u.deg, frame=frame)
    dimensions = (1*u.arcmin, 0.5*u.arcmin, 2*u.arcmin, 1*u.arcmin, 20*u.deg, 100*u.deg)
    first = elliptical_sector_annulus_selection(center, *dimensions, angle*u.deg)
    second = elliptical_sector_annulus_selection(center, *dimensions, (angle+180)*u.deg)
    assert first.canonical_geometry_record() == second.canonical_geometry_record()
    assert first.geometry_sha256 == second.geometry_sha256
    first_path = first.regions[0].boundaries[0].path
    second_path = second.regions[0].boundaries[0].path
    for parameter in np.linspace(0, 1, 33):
        assert first_path.point(parameter).separation(second_path.point(parameter)).deg == 0
    local_first = ellipse_local_sector_annulus_selection(center, *dimensions, angle*u.deg)
    local_second = ellipse_local_sector_annulus_selection(center, *dimensions, (angle+180)*u.deg)
    assert local_first.geometry_sha256 != local_second.geometry_sha256
    assert local_first.regions[0].boundaries[0].path.point(0).separation(
        local_second.regions[0].boundaries[0].path.point(0)).arcsec > 1

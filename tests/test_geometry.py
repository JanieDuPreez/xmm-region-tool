from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from xmm_region_tool.geometry import (
    load_ds9_selection,
    load_ds9_sky_regions,
    read_ds9_sky_regions,
)


def test_circle_and_exclusion_are_parsed_as_sky_boundaries(tmp_path):
    path = tmp_path / "regions.reg"
    path.write_text(
        "# Region file format: DS9 version 4.1\n"
        "fk5\n"
        "circle(10.0,-9.0,2')\n"
        "-circle(10.01,-9.0,30\")\n"
    )

    regions = load_ds9_sky_regions(path, samples=32)

    assert len(regions) == 2
    assert regions[0].include is True
    assert regions[1].include is False
    assert len(regions[0].boundaries[0].vertices) == 32
    assert len(regions[1].boundaries[0].vertices) == 32


def test_annulus_becomes_outer_boundary_minus_inner_boundary(tmp_path):
    path = tmp_path / "annulus.reg"
    path.write_text("fk5\nannulus(10.0,-9.0,1',2')\n")

    regions = load_ds9_sky_regions(path, samples=24)

    assert len(regions) == 1
    assert len(regions[0].boundaries) == 2
    assert regions[0].boundaries[0].subtract is False
    assert regions[0].boundaries[1].subtract is True
    assert len(regions[0].boundaries[0].vertices) == 24


def test_ds9_multi_annulus_expands_to_independent_adjacent_regions(tmp_path):
    path = tmp_path / "multi-annulus.reg"
    path.write_text("fk5\nannulus(10.0,-9.0,1',2',3',5')\n")

    regions = load_ds9_sky_regions(path, samples=24)

    assert len(regions) == 3
    radii = []
    for region in regions:
        assert region.include is True
        assert len(region.boundaries) == 2
        outer = region.boundaries[0].path.canonical_record()["radius_deg"] * 60
        inner = region.boundaries[1].path.canonical_record()["radius_deg"] * 60
        radii.append((inner, outer))
    assert radii == pytest.approx([(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)])


def test_ds9_multi_annulus_n_subdivision_expands_to_independent_regions(tmp_path):
    path = tmp_path / "multi-annulus-n.reg"
    path.write_text("fk5\nannulus(10.0,-9.0,1',4',n=3)\n")

    regions = load_ds9_sky_regions(path, samples=24)

    assert len(regions) == 3
    radii = []
    for region in regions:
        outer = region.boundaries[0].path.canonical_record()["radius_deg"] * 60
        inner = region.boundaries[1].path.canonical_record()["radius_deg"] * 60
        radii.append((inner, outer))
    assert radii == pytest.approx([(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])


def test_box_annulus_becomes_outer_rectangle_minus_inner_rectangle(tmp_path):
    path = tmp_path / "box-annulus.reg"
    path.write_text("fk5\nbox(10.0,-9.0,1',2',3',6',30)\n")

    regions = load_ds9_sky_regions(path)

    assert len(regions) == 1
    assert len(regions[0].boundaries) == 2
    assert regions[0].boundaries[0].subtract is False
    assert regions[0].boundaries[1].subtract is True
    assert len(regions[0].boundaries[0].vertices) == 4
    assert len(regions[0].boundaries[1].vertices) == 4


def test_rotated_ds9_ellipse_uses_opposite_sign_from_regions_angle(tmp_path):
    path = tmp_path / "rotated.reg"
    path.write_text("fk5\nellipse(10.0,-9.0,2',1',37)\n")

    source = read_ds9_sky_regions(path)[0]
    sampled = load_ds9_sky_regions(path, samples=32)[0]
    first = sampled.boundaries[0].vertices[0]

    # DS9 handedness is reflected; the undirected axis has PA 90+37 modulo 180.
    position_angle = source.center.position_angle(first).wrap_at(180 * u.deg)
    separation = source.center.separation(first)
    assert position_angle.to_value(u.deg) % 180 == pytest.approx(127.0, abs=1e-8)
    assert separation.to_value(u.arcmin) == pytest.approx(2.0, abs=1e-8)


def test_ds9_panda_becomes_one_semantic_sector_cell(tmp_path):
    path = tmp_path / "sector.reg"
    path.write_text("fk5\npanda(10.0,-9.0,0,90,1,60\",120\",1)\n")

    selection = load_ds9_selection(path)

    assert len(selection.regions) == 1
    boundary = selection.regions[0].boundaries[0]
    record = boundary.path.canonical_record()
    assert record["kind"] == "sector-annulus"
    assert record["inner_radius_deg"] == pytest.approx(1.0 / 60.0)
    assert record["outer_radius_deg"] == pytest.approx(2.0 / 60.0)
    assert record["sweep_deg"] == pytest.approx(90.0)


def test_ds9_extensions_expand_in_source_order(tmp_path):
    path = tmp_path / "interleaved.reg"
    path.write_text(
        "icrs\n"
        'panda(10.0,-9.0,0,90,2,60",120",1)\n'
        'circle(10.1,-9.0,30")\n'
        'annulus(10.2,-9.0,30",90",n=2)\n'
        'epanda(10.3,-9.0,30,90,1,60",30",120",60",1,27)\n'
        'circle(10.4,-9.0,30")\n'
    )

    selection = load_ds9_selection(path)

    # Multi-cell constructs expand exactly where they appear in the authored
    # DS9 stream:
    #
    # panda A -> 2 cells
    # circle B -> 1
    # annulus C -> 2
    # epanda D -> 1
    # circle E -> 1
    assert len(selection.regions) == 7

    centres = [
        region.boundaries[0].path.center.ra.to_value(u.deg)
        for region in selection.regions
    ]
    assert centres == pytest.approx(
        [10.0, 10.0, 10.1, 10.2, 10.2, 10.3, 10.4],
        abs=1e-10,
    )


def test_ds9_panda_directed_angles_reflect_and_reverse(tmp_path):
    path = tmp_path / "asymmetric-sector.reg"
    path.write_text("icrs\npanda(10.0,-9.0,30,110,1,60\",180\",1)\n")

    selection = load_ds9_selection(path)
    model = selection.regions[0].boundaries[0].path

    # Real A133/DS9/MOS1 visual validation established:
    #
    #   DS9 30 -> 110
    #   ray reflection: 150, 70
    #   traversal reversal: native 70 -> 150
    #
    # The physical 80-degree wedge is preserved.
    assert model.start_angle.to_value(u.deg) == pytest.approx(70.0, abs=1e-9)
    assert model.sweep.to_value(u.deg) == pytest.approx(80.0, abs=1e-9)


def test_ds9_panda_grid_expands_to_angle_times_radius_cells(tmp_path):
    path = tmp_path / "grid.reg"
    path.write_text("icrs\npanda(10.0,-9.0,0,180,2,0\",180\",3)\n")

    selection = load_ds9_selection(path)

    assert len(selection.regions) == 6
    records = [region.boundaries[0].path.canonical_record() for region in selection.regions]
    assert [record["sweep_deg"] for record in records] == pytest.approx([90.0] * 6)
    assert [record["inner_radius_deg"] * 60 for record in records] == pytest.approx(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    )
    assert [record["outer_radius_deg"] * 60 for record in records] == pytest.approx(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    )


def test_ds9_panda_wraps_stop_angle_through_zero(tmp_path):
    path = tmp_path / "wrap.reg"
    path.write_text("icrs\npanda(10.0,-9.0,330,30,1,60\",120\",1)\n")

    selection = load_ds9_selection(path)
    path_model = selection.regions[0].boundaries[0].path

    assert path_model.sweep.to_value(u.deg) == pytest.approx(60.0)
    # DS9 330 -> 30 reflects to native 150 -> 210 while preserving
    # the physical 60-degree wedge.
    assert path_model.start_angle.to_value(u.deg) == pytest.approx(150.0, abs=1e-9)


def test_ds9_full_circle_panda_reuses_annulus_semantics(tmp_path):
    path = tmp_path / "full.reg"
    path.write_text("icrs\npanda(10.0,-9.0,0,360,1,60\",120\",1)\n")

    selection = load_ds9_selection(path)

    assert len(selection.regions) == 1
    boundaries = selection.regions[0].boundaries
    assert len(boundaries) == 2
    assert boundaries[0].path.canonical_record()["kind"] == "circle"
    assert boundaries[1].path.canonical_record()["kind"] == "circle"
    assert boundaries[1].subtract is True


def test_ds9_equal_panda_angles_mean_full_circle(tmp_path):
    path = tmp_path / "full-equal.reg"
    path.write_text("icrs\npanda(10.0,-9.0,45,45,1,0\",120\",1)\n")

    selection = load_ds9_selection(path)

    assert len(selection.regions[0].boundaries) == 1
    assert selection.regions[0].boundaries[0].path.canonical_record()["kind"] == "circle"


def test_excluded_panda_preserves_exclusion_semantics(tmp_path):
    path = tmp_path / "excluded.reg"
    path.write_text("icrs\n-panda(10.0,-9.0,20,140,2,60\",180\",2)\n")

    selection = load_ds9_selection(path)

    assert len(selection.regions) == 4
    assert all(region.include is False for region in selection.regions)


def test_mixed_file_keeps_ordinary_shape_and_panda(tmp_path):
    path = tmp_path / "mixed.reg"
    path.write_text(
        "fk5\n"
        "circle(10.0,-9.0,2')\n"
        "panda(10.0,-9.0,0,180,1,30\",2',1)\n"
    )

    selection = load_ds9_selection(path)

    assert len(selection.regions) == 2
    kinds = [region.boundaries[0].path.canonical_record()["kind"] for region in selection.regions]
    assert kinds == ["circle", "sector-annulus"]


def test_panda_semantic_identity_does_not_depend_on_sampling_hint(tmp_path):
    path = tmp_path / "sector.reg"
    path.write_text("icrs\npanda(10.0,-9.0,15,195,1,60\",180\",1)\n")

    coarse = load_ds9_selection(path, samples=32)
    dense = load_ds9_selection(path, samples=512)

    assert coarse.geometry_sha256 == dense.geometry_sha256
    assert len(coarse.regions[0].boundaries[0].vertices) == 32
    assert len(dense.regions[0].boundaries[0].vertices) == 512


def test_pixel_coordinate_region_is_rejected(tmp_path):
    path = tmp_path / "pixels.reg"
    path.write_text("image\ncircle(100,100,20)\n")

    with pytest.raises(TypeError, match="sky coordinates"):
        load_ds9_sky_regions(path)


def test_pixel_coordinate_panda_is_rejected(tmp_path):
    path = tmp_path / "pixels-panda.reg"
    path.write_text("image\npanda(100,100,0,90,1,20,40,1)\n")

    with pytest.raises(TypeError, match="sky coordinates"):
        load_ds9_selection(path)


def test_panda_rejects_invalid_division_counts(tmp_path):
    path = tmp_path / "bad.reg"
    path.write_text("icrs\npanda(10,-9,0,90,0,1',2',1)\n")

    with pytest.raises(ValueError, match="nangle"):
        load_ds9_selection(path)


def test_panda_cell_order_is_preserved_under_directed_angle_reflection(tmp_path):
    path = tmp_path / "wrap-grid.reg"
    path.write_text("icrs\npanda(10,-9,330,30,3,1',2',1)\n")

    selection = load_ds9_selection(path)
    starts = np.array(
        [region.boundaries[0].path.start_angle.to_value(u.deg) for region in selection.regions]
    )

    # Source cell ordering is preserved while angular handedness is reflected.
    assert starts == pytest.approx([190.0, 170.0, 150.0], abs=1e-9)

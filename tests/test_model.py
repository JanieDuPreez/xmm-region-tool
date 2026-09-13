from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool import CelestialSelection, load_ds9_selection
from xmm_region_tool.sas import project_with_esky2det


def polygon_vertices() -> SkyCoord:
    return SkyCoord(
        [10.0, 10.001, 10.0] * u.deg,
        [-9.0, -9.0, -8.999] * u.deg,
        frame="icrs",
    )


def test_ds9_and_in_memory_polygon_have_same_geometry_identity(tmp_path):
    path = tmp_path / "polygon.reg"
    path.write_text("icrs\npolygon(10,-9,10.001,-9,10,-8.999)\n")

    from_ds9 = load_ds9_selection(path)
    in_memory = CelestialSelection.polygon(
        polygon_vertices(),
        external_identity="caller-region-17",
        external_provenance={"owner": "caller"},
    )

    assert from_ds9.geometry_sha256 == in_memory.geometry_sha256
    assert in_memory.external_identity == "caller-region-17"
    assert in_memory.external_provenance == {"owner": "caller"}


def test_ds9_and_native_sector_annulus_match_after_angle_adapter(tmp_path):
    path = tmp_path / "sector.reg"
    path.write_text("icrs\npanda(10,-9,330,30,1,60\",120\",1)\n")

    from_ds9 = load_ds9_selection(path)
    in_memory = CelestialSelection.sector_annulus(
        SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs"),
        60 * u.arcsec,
        120 * u.arcsec,
        150 * u.deg,
        210 * u.deg,
    )

    # DS9 330 -> 30 reflects and reverses to native 150 -> 210.
    assert from_ds9.geometry_sha256 == in_memory.geometry_sha256


def test_external_metadata_does_not_change_geometry_identity():
    first = CelestialSelection.polygon(polygon_vertices(), external_identity="science-a")
    second = CelestialSelection.polygon(
        polygon_vertices(),
        external_identity="science-b",
        external_provenance={"different": True},
    )

    assert first.geometry_sha256 == second.geometry_sha256


def test_circle_semantic_identity_does_not_depend_on_sampling_hint(tmp_path):
    path = tmp_path / "circle.reg"
    path.write_text("icrs\ncircle(10,-9,2')\n")

    coarse = load_ds9_selection(path, samples=32)
    dense = load_ds9_selection(path, samples=512)

    assert len(coarse.regions[0].boundaries[0].vertices) == 32
    assert len(dense.regions[0].boundaries[0].vertices) == 512
    assert coarse.geometry_sha256 == dense.geometry_sha256


def test_ellipse_semantic_identity_does_not_depend_on_sampling_hint(tmp_path):
    path = tmp_path / "ellipse.reg"
    path.write_text("icrs\nellipse(10,-9,2',1',35)\n")

    coarse = load_ds9_selection(path, samples=32)
    dense = load_ds9_selection(path, samples=256)

    assert coarse.geometry_sha256 == dense.geometry_sha256


def test_sector_annulus_has_piecewise_semantic_boundary():
    center = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")
    selection = CelestialSelection.sector_annulus(
        center,
        1.0 * u.arcmin,
        2.0 * u.arcmin,
        0.0 * u.deg,
        90.0 * u.deg,
    )

    boundary = selection.regions[0].boundaries[0]
    path = boundary.path
    assert path.canonical_record()["kind"] == "sector-annulus"
    assert path.segment_count == 4
    assert np.isclose(path.sweep.to_value(u.deg), 90.0)

    outer_start = path.point(0.0)
    outer_stop = path.point(0.25)
    inner_stop = path.point(0.5)
    inner_start = path.point(0.75)

    assert np.isclose(center.separation(outer_start).to_value(u.arcmin), 2.0, atol=1e-8)
    assert np.isclose(center.separation(outer_stop).to_value(u.arcmin), 2.0, atol=1e-8)
    assert np.isclose(center.separation(inner_stop).to_value(u.arcmin), 1.0, atol=1e-8)
    assert np.isclose(center.separation(inner_start).to_value(u.arcmin), 1.0, atol=1e-8)
    assert np.isclose(center.position_angle(outer_start).to_value(u.deg), 90.0, atol=1e-7)
    assert np.isclose(center.position_angle(outer_stop).to_value(u.deg), 0.0, atol=1e-7)


def test_sector_annulus_wraps_across_zero_degrees():
    center = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")
    selection = CelestialSelection.sector_annulus(
        center,
        1.0 * u.arcmin,
        2.0 * u.arcmin,
        330.0 * u.deg,
        30.0 * u.deg,
    )

    path = selection.regions[0].boundaries[0].path
    assert np.isclose(path.start_angle.to_value(u.deg), 330.0)
    assert np.isclose(path.sweep.to_value(u.deg), 60.0)


def test_sector_annulus_supports_sweeps_greater_than_180_degrees():
    center = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")
    selection = CelestialSelection.sector_annulus(
        center,
        1.0 * u.arcmin,
        2.0 * u.arcmin,
        20.0 * u.deg,
        290.0 * u.deg,
    )

    path = selection.regions[0].boundaries[0].path
    assert np.isclose(path.start_angle.to_value(u.deg), 20.0)
    assert np.isclose(path.sweep.to_value(u.deg), 270.0)


def test_sector_wedge_with_zero_inner_radius_has_three_segments():
    center = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")
    selection = CelestialSelection.sector_annulus(
        center,
        0.0 * u.arcmin,
        2.0 * u.arcmin,
        20.0 * u.deg,
        140.0 * u.deg,
    )

    path = selection.regions[0].boundaries[0].path
    assert path.segment_count == 3
    assert path.point(2.0 / 3.0).separation(center).to_value(u.arcsec) < 1e-8


def test_full_circle_sector_uses_existing_annulus_semantics():
    center = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")
    selection = CelestialSelection.sector_annulus(
        center,
        1.0 * u.arcmin,
        2.0 * u.arcmin,
        0.0 * u.deg,
        360.0 * u.deg,
    )

    boundaries = selection.regions[0].boundaries
    assert len(boundaries) == 2
    assert boundaries[0].path.canonical_record()["kind"] == "circle"
    assert boundaries[1].path.canonical_record()["kind"] == "circle"
    assert boundaries[1].subtract is True


def test_equal_sector_angles_mean_full_circle_like_ds9_panda():
    center = SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs")
    selection = CelestialSelection.sector_annulus(
        center,
        0.0 * u.arcmin,
        2.0 * u.arcmin,
        45.0 * u.deg,
        45.0 * u.deg,
    )

    boundaries = selection.regions[0].boundaries
    assert len(boundaries) == 1
    assert boundaries[0].path.canonical_record()["kind"] == "circle"


def test_in_memory_selection_projects_without_ds9_round_trip(monkeypatch, tmp_path):
    event = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event)
    selection = CelestialSelection.polygon(polygon_vertices())

    monkeypatch.setattr(
        "xmm_region_tool.sas.shutil.which",
        lambda *args, **kwargs: "/sas/bin/esky2det",
    )

    def fake_run(command, **kwargs):
        intab = next(value for value in command if value.startswith("intab="))
        path = Path(intab.removeprefix("intab=").split(":", maxsplit=1)[0])
        with fits.open(path) as hdus:
            source = hdus["INPUT"].data
            output = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="D", array=source["RA"]),
                    fits.Column(name="DEC", format="D", array=source["DEC"]),
                    fits.Column(name="ROW_ID", format="J", array=source["ROW_ID"]),
                    fits.Column(name="DETX", format="D", array=source["RA"] * 1000.0),
                    fits.Column(name="DETY", format="D", array=source["DEC"] * 1000.0),
                ],
                name="INPUT",
            )
        fits.HDUList([fits.PrimaryHDU(), output]).writeto(path, overwrite=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    detector = project_with_esky2det(selection, calinfoset=event)

    assert detector.source_geometry_sha256 == selection.geometry_sha256
    assert len(detector.regions) == 1
    assert np.allclose(
        detector.regions[0].boundaries[0].vertices[:, 0],
        [10000.0, 10001.0, 10000.0],
    )


def test_external_metadata_is_an_immutable_nonsemantic_snapshot():
    import pytest
    from xmm_region_tool.workflow import extraction_cells

    metadata = {"nested": [{"owner": "original"}]}
    selection = CelestialSelection.polygon(polygon_vertices(), external_provenance=metadata)
    baseline = selection.geometry_sha256
    metadata["nested"][0]["owner"] = "changed"
    assert selection.external_provenance["nested"][0]["owner"] == "original"
    with pytest.raises(TypeError):
        selection.external_provenance["new"] = 1
    with pytest.raises(TypeError):
        extraction_cells(selection)[0].selection.external_provenance["nested"][0]["owner"] = "changed"
    record = selection.external_provenance_record()
    record["nested"][0]["owner"] = "changed"
    assert selection.geometry_sha256 == baseline
    assert baseline == CelestialSelection.polygon(polygon_vertices()).geometry_sha256


def test_external_metadata_rejects_mutable_unrecognized_values():
    import pytest
    from xmm_region_tool import SelectionGeometryError

    for kwargs in ({"external_identity": []}, {"external_provenance": {"bad": np.zeros(2)}}):
        with pytest.raises(SelectionGeometryError, match="external"):
            CelestialSelection.polygon(polygon_vertices(), **kwargs)


def test_detector_buffer_cannot_be_made_writeable():
    import pytest
    from xmm_region_tool import DetectorBoundary, DetectorRegion, DetectorSelection

    supplied = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    boundary = DetectorBoundary(supplied)
    selection = DetectorSelection((DetectorRegion(True, (boundary,)),))
    baseline = selection.geometry_sha256
    supplied[0] = [3, 3]
    with pytest.raises(ValueError):
        boundary.vertices.setflags(write=True)
    with pytest.raises(ValueError):
        boundary.vertices.base.setflags(write=True)
    exposed = boundary.vertices
    exposed.shape = (6,)
    assert selection.geometry_sha256 == baseline


def test_science_flags_reject_truthiness_coercion():
    import pytest
    from xmm_region_tool import (
        CelestialBoundary, CelestialRegion, DetectorBoundary, DetectorRegion,
        SelectionGeometryError, circle_selection,
    )
    from xmm_region_tool.geometry import selection_from_sky_regions
    from regions import CircleSkyRegion

    celestial = CelestialBoundary(polygon_vertices())
    xy = np.array([[0, 0], [1, 0], [0, 1]])
    detector = DetectorBoundary(xy)
    for bad in ("false", "0", "exclude", 0, 1, [], {}, object()):
        for field, construct in (
            ("include", lambda: CelestialRegion(bad, (celestial,))),
            ("include", lambda: DetectorRegion(bad, (detector,))),
            ("subtract", lambda: CelestialBoundary(polygon_vertices(), subtract=bad)),
            ("subtract", lambda: DetectorBoundary(xy, subtract=bad)),
            ("include", lambda: circle_selection(polygon_vertices()[0], 1 * u.arcmin, include=bad)),
        ):
            with pytest.raises(SelectionGeometryError, match=field):
                construct()
    for bad in ("false", "0", 2, object()):
        region = CircleSkyRegion(polygon_vertices()[0], 1 * u.arcmin, meta={"include": bad})
        with pytest.raises(ValueError, match="include"):
            selection_from_sky_regions([region])


def test_all_celestial_paths_are_detached_immutable_snapshots():
    from xmm_region_tool.model import CirclePath, EllipsePath, SectorAnnulusPath, GeodesicPolygonPath
    from xmm_region_tool.elliptical_sector import EllipticalSectorAnnulusPath

    original = polygon_vertices()
    center = original[0]
    radius = 1 * u.arcmin
    paths = [
        GeodesicPolygonPath(original), CirclePath(center, radius),
        EllipsePath(center, 2 * radius, radius, 20 * u.deg),
        SectorAnnulusPath(center, radius / 2, radius, 20 * u.deg, 60 * u.deg),
        EllipticalSectorAnnulusPath(center, radius / 2, radius, 20 * u.deg, 60 * u.deg,
            inner_semi_minor=radius / 4, outer_semi_minor=radius / 2, ellipse_angle=10 * u.deg),
    ]
    records = [path.canonical_record() for path in paths]
    points = [path.point(0.37) for path in paths]
    original[0] = SkyCoord(40 * u.deg, 10 * u.deg)
    radius[...] = 8 * u.arcmin
    for path, record, point in zip(paths, records, points):
        if isinstance(path, GeodesicPolygonPath):
            path.vertices[0] = SkyCoord(20 * u.deg, 0 * u.deg)
        else:
            path.center.data.lon[...] = 30 * u.deg
            for name in ("radius", "width", "height", "angle", "inner_radius", "outer_radius",
                         "start_angle", "sweep", "inner_semi_minor", "outer_semi_minor", "ellipse_angle"):
                if hasattr(path, name):
                    getattr(path, name)[...] = 3 * u.deg
        assert path.canonical_record() == record
        assert path.point(0.37).separation(point).to_value(u.deg) == 0

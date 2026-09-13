from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.output import (
    RegionSerializationError,
    detector_region_components,
    selection_expression,
    write_esas_regionfile,
    write_fits_region,
)
from xmm_region_tool.provenance import EventIdentity, file_sha256
from xmm_region_tool.sas import DetectorBoundary, DetectorRegion


def boundary(*points: tuple[float, float], subtract: bool = False) -> DetectorBoundary:
    return DetectorBoundary(np.asarray(points, dtype=float), subtract=subtract)


def triangle(x: float = 0.0, y: float = 0.0, *, subtract: bool = False) -> DetectorBoundary:
    return boundary((x, y), (x + 1, y), (x, y + 1), subtract=subtract)


def event_identity(tmp_path) -> EventIdentity:
    event_path = Path(tmp_path) / "events.fits"
    event_path.write_bytes(b"synthetic exact event artifact bytes")
    return EventIdentity(
        path=event_path,
        instrument="mos1",
        instrument_header="EMOS1",
        obs_id="0144310101",
        exposure_id="S001",
        telescope="XMM",
        date_obs="2003-06-22T00:00:00",
        ra_pnt=15.673,
        dec_pnt=-21.88,
        pa_pnt=72.0,
    )


def calibration_identity(tmp_path) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=Path(tmp_path) / "ccf.cif",
        cif_file_sha256="b" * 64,
        calindex_sha256="c" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def test_direct_expression_unions_inclusions_and_subtracts_exclusions():
    regions = [
        DetectorRegion(True, (triangle(),)),
        DetectorRegion(True, (triangle(10, 10),)),
        DetectorRegion(False, (triangle(0.1, 0.1),)),
    ]

    expression = selection_expression(regions)

    assert " || " in expression
    assert " && !" in expression
    assert expression.count("polygon2(") == 3


def test_direct_expression_supports_exclusion_only_ds9_files():
    expression = selection_expression(
        [DetectorRegion(False, (triangle(),)), DetectorRegion(False, (triangle(10, 10),))]
    )

    assert expression.startswith("!")
    assert " && !" in expression
    assert expression.count("polygon2(") == 2


def test_included_annulus_is_one_component_with_negative_inner_polygon():
    annulus = DetectorRegion(True, (triangle(), triangle(0.2, 0.2, subtract=True)))

    components = detector_region_components([annulus])

    assert len(components) == 1
    assert [element.negate for element in components[0]] == [False, True]


def test_excluded_annulus_expands_to_outside_outer_or_inside_inner():
    annulus = DetectorRegion(False, (triangle(), triangle(0.2, 0.2, subtract=True)))

    components = detector_region_components([annulus])

    assert len(components) == 2
    assert [element.negate for element in components[0]] == [True]
    assert [element.negate for element in components[1]] == [False]


def test_boolean_component_expansion_has_explicit_safety_limit():
    regions = [
        DetectorRegion(False, (triangle(), triangle(0.2, 0.2, subtract=True))),
        DetectorRegion(False, (triangle(10, 10), triangle(10.2, 10.2, subtract=True))),
    ]

    with pytest.raises(RegionSerializationError, match="4 FITS REGION components"):
        detector_region_components(regions, max_components=3)


def test_fits_region_uses_standard_headers_components_and_polygon_closure(tmp_path):
    regions = [
        DetectorRegion(True, (triangle(), triangle(0.2, 0.2, subtract=True))),
        DetectorRegion(True, (triangle(10, 10),)),
    ]
    path = write_fits_region(tmp_path / "detector.fits", regions)

    with fits.open(path) as hdus:
        table = hdus["REGION"]
        assert table.header["HDUCLAS1"] == "REGION"
        assert table.header["HDUCLAS2"] == "STANDARD"
        assert table.header["MFORM1"] == "X,Y"
        assert table.columns["COMPONENT"].format == "J"
        assert list(table.data["COMPONENT"]) == [1, 1, 2]
        shapes = [value.strip() for value in table.data["SHAPE"]]
        assert shapes == ["POLYGON", "!POLYGON", "POLYGON"]
        assert table.data["X"][0][3] == table.data["X"][0][0]
        assert table.data["Y"][0][3] == table.data["Y"][0][0]


def test_fits_region_densifies_shorter_polygons_without_repeated_padding(tmp_path):
    outer = boundary((0, 0), (4, 0), (4, 4), (0, 4), (0, 2), (0, 1))
    inner = boundary((1, 1), (3, 1), (2, 3), subtract=True)
    path = write_fits_region(
        tmp_path / "annulus.fits",
        [DetectorRegion(True, (outer, inner))],
    )

    with fits.open(path) as hdus:
        table = hdus["REGION"].data
        assert table is not None
        assert len(table) == 2
        assert [value.strip() for value in table["SHAPE"]] == ["POLYGON", "!POLYGON"]
        for row in table:
            coords = np.column_stack((row["X"], row["Y"]))
            assert np.array_equal(coords[0], coords[-1])
            open_vertices = coords[:-1]
            assert len(open_vertices) == 6
            successive = np.roll(open_vertices, -1, axis=0) - open_vertices
            assert np.all(np.linalg.norm(successive, axis=1) > 0)


def test_fits_region_records_event_and_source_identity(tmp_path):
    identity = event_identity(tmp_path)
    source_sha = "a" * 64
    path = write_fits_region(
        tmp_path / "detector.fits",
        [DetectorRegion(True, (triangle(),))],
        event_identity=identity,
        source_region_sha256=source_sha,
    )

    with fits.open(path) as hdus:
        header = hdus["REGION"].header
        assert header["TELESCOP"] == "XMM"
        assert header["INSTRUME"] == "EMOS1"
        assert header["OBS_ID"] == "0144310101"
        assert header["EXP_ID"] == "S001"
        assert header["XMMRGID"] == identity.identity_sha256
        assert header["XMRGEVT"] == file_sha256(identity.path)
        assert header["XMRGSHA"] == source_sha


def test_fits_region_records_calibration_and_sas_producer_identity(tmp_path):
    calibration = calibration_identity(tmp_path)
    producer = SasProducerIdentity(
        esky2det_version="esky2det-2.0",
        sas_version="xmmsas_20250127_1200-22.0.0",
    )
    path = write_fits_region(
        tmp_path / "detector.fits",
        [DetectorRegion(True, (triangle(),))],
        calibration_identity=calibration,
        sas_producer=producer,
    )

    with fits.open(path) as hdus:
        header = hdus["REGION"].header
        assert header["XMRGCCF"] == calibration.identity_sha256
        assert header["XMRGCIF"] == calibration.cif_file_sha256
        assert header["XMRGCAL"] == calibration.calindex_sha256
        assert header["XMRGREP"] == 0
        assert header["XMRGPRD"] == producer.identity_sha256
        assert header["ESKYVER"] == producer.esky2det_version
        assert header["SASVERS"] == producer.sas_version


def test_invalid_source_digest_is_rejected(tmp_path):
    with pytest.raises(RegionSerializationError, match="SHA256"):
        write_fits_region(
            tmp_path / "detector.fits",
            [DetectorRegion(True, (triangle(),))],
            source_region_sha256="not-a-digest",
        )


def test_default_esas_output_is_short_wrapper_plus_companion_fits(tmp_path):
    written = write_esas_regionfile(
        tmp_path / "mos1-region.txt",
        [DetectorRegion(True, (triangle(),))],
    )

    assert written.representation == "fits"
    assert written.fits_region is not None
    assert written.fits_region.is_file()
    text = written.regionfile.read_text()
    assert text.startswith("&&region(")
    assert text.endswith(",DETX,DETY)\n")
    assert "polygon" not in text


def test_auto_uses_expression_only_when_it_fits_explicit_limit(tmp_path):
    region = [DetectorRegion(True, (triangle(),))]

    direct = write_esas_regionfile(
        tmp_path / "short.txt", region, representation="auto", inline_limit=1000
    )
    delegated = write_esas_regionfile(
        tmp_path / "long.txt", region, representation="auto", inline_limit=10
    )

    assert direct.representation == "expression"
    assert "polygon2(" in direct.regionfile.read_text()
    assert delegated.representation == "fits"
    assert delegated.fits_region is not None

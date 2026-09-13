from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.output import write_fits_region
from xmm_region_tool.provenance import (
    EventIdentityError,
    check_region_binding,
    file_sha256,
    read_event_identity,
)
from xmm_region_tool.sas import DetectorBoundary, DetectorRegion


def write_event(
    path,
    *,
    instrument: str = "EMOS1",
    telescope: str = "XMM",
    obs_id: str = "0144310101",
    detx: tuple[int, int] = (1, 2),
):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = telescope
    primary.header["INSTRUME"] = instrument
    primary.header["OBS_ID"] = obs_id
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="J", array=np.array(detx, dtype=np.int32)),
            fits.Column(name="DETY", format="J", array=np.array([3, 4], dtype=np.int32)),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def triangle() -> DetectorRegion:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return DetectorRegion(True, (DetectorBoundary(vertices),))


def test_event_identity_binds_observation_camera_and_pointing(tmp_path):
    path = write_event(tmp_path / "events.fits")

    identity = read_event_identity(path)

    assert identity.obs_id == "0144310101"
    assert identity.instrument == "mos1"
    assert identity.instrument_header == "EMOS1"
    assert identity.exposure_id == "S001"
    assert identity.ra_pnt == pytest.approx(15.673)
    assert len(identity.identity_sha256) == 64


def test_event_identity_digest_is_path_independent(tmp_path):
    first = write_event(tmp_path / "first.fits")
    second = write_event(tmp_path / "second.fits")

    assert read_event_identity(first).identity_sha256 == read_event_identity(second).identity_sha256


@pytest.mark.parametrize("keyword", ["RA_PNT", "DEC_PNT", "PA_PNT"])
@pytest.mark.parametrize("value", ["NaN", "+inf", "-inf", "1e999"])
def test_event_identity_rejects_nonfinite_pointing(tmp_path, keyword, value):
    path = write_event(tmp_path / "events.fits")
    # FITS disallows native float NaN/inf cards, but accepts numeric text.
    with fits.open(path, mode="update") as hdus:
        hdus[0].header[keyword] = value

    with pytest.raises(EventIdentityError, match=rf"{keyword}.*finite"):
        read_event_identity(path)


@pytest.mark.parametrize("keyword", ["RA_PNT", "DEC_PNT", "PA_PNT"])
@pytest.mark.parametrize("value", [None, -999.0, 999.0, "15.673"])
def test_event_identity_preserves_optional_finite_pointing(tmp_path, keyword, value):
    path = write_event(tmp_path / "events.fits")
    with fits.open(path, mode="update") as hdus:
        if value is None:
            del hdus[0].header[keyword]
        else:
            hdus[0].header[keyword] = value

    identity = read_event_identity(path)

    assert getattr(identity, keyword.lower()) == (None if value is None else float(value))
    assert len(identity.identity_sha256) == 64
    assert identity.identity_sha256 == read_event_identity(path).identity_sha256


def test_non_epic_instrument_is_rejected(tmp_path):
    path = write_event(tmp_path / "events.fits", instrument="RGS1")

    with pytest.raises(EventIdentityError, match="unsupported XMM instrument"):
        read_event_identity(path)


def test_non_xmm_product_is_rejected(tmp_path):
    path = write_event(tmp_path / "events.fits", telescope="CHANDRA")

    with pytest.raises(EventIdentityError, match="does not identify an XMM"):
        read_event_identity(path)


def test_source_region_sha_changes_with_content(tmp_path):
    path = tmp_path / "source.reg"
    path.write_text("fk5\ncircle(10,-9,1')\n")
    first = file_sha256(path)
    path.write_text("fk5\ncircle(10,-9,2')\n")

    assert len(first) == 64
    assert file_sha256(path) != first


def test_generated_region_binding_accepts_exact_event_and_source(tmp_path):
    event = write_event(tmp_path / "events.fits")
    source = tmp_path / "source.reg"
    source.write_text("fk5\ncircle(10,-9,1')\n")
    identity = read_event_identity(event)
    region = write_fits_region(
        tmp_path / "detector.fits",
        [triangle()],
        event_identity=identity,
        source_region_sha256=file_sha256(source),
    )

    report = check_region_binding(region, event, source_region=source)

    assert report.compatible
    assert report.mismatches == ()
    assert report.binding.event_file_sha256 == file_sha256(event)


def test_generated_region_binding_rejects_same_headers_different_event_bytes(tmp_path):
    original = write_event(tmp_path / "original.fits", detx=(1, 2))
    changed = write_event(tmp_path / "changed.fits", detx=(1, 99))
    assert read_event_identity(original).identity_sha256 == read_event_identity(changed).identity_sha256
    assert file_sha256(original) != file_sha256(changed)

    region = write_fits_region(
        tmp_path / "detector.fits",
        [triangle()],
        event_identity=read_event_identity(original),
    )

    report = check_region_binding(region, changed)

    assert not report.compatible
    assert "exact event-file SHA256 differs" in report.mismatches
    assert "event identity SHA256 differs" not in report.mismatches


def test_generated_region_binding_rejects_other_observation(tmp_path):
    original = write_event(tmp_path / "original.fits")
    other = write_event(tmp_path / "other.fits", obs_id="9999999999")
    identity = read_event_identity(original)
    region = write_fits_region(
        tmp_path / "detector.fits",
        [triangle()],
        event_identity=identity,
    )

    report = check_region_binding(region, other)

    assert not report.compatible
    assert "event identity SHA256 differs" in report.mismatches
    assert "exact event-file SHA256 differs" in report.mismatches
    assert any("ObsID differs" in mismatch for mismatch in report.mismatches)


def test_generated_region_binding_rejects_edited_source_ds9(tmp_path):
    event = write_event(tmp_path / "events.fits")
    source = tmp_path / "source.reg"
    source.write_text("fk5\ncircle(10,-9,1')\n")
    region = write_fits_region(
        tmp_path / "detector.fits",
        [triangle()],
        event_identity=read_event_identity(event),
        source_region_sha256=file_sha256(source),
    )
    source.write_text("fk5\ncircle(10,-9,2')\n")

    report = check_region_binding(region, event, source_region=source)

    assert not report.compatible
    assert "source DS9 SHA256 differs" in report.mismatches


def test_binding_checks_expected_celestial_and_projection_identity(tmp_path):
    event = tmp_path / "events.fits"
    write_event(event)

    region = tmp_path / "region.fits"
    event_identity = read_event_identity(event)
    write_fits_region(
        region,
        [triangle()],
        event_identity=event_identity,
        celestial_geometry_sha256="a" * 64,
        projection_identity_sha256="b" * 64,
    )

    report = check_region_binding(
        region,
        event,
        expected_celestial_geometry_sha256="a" * 64,
        expected_projection_identity_sha256="b" * 64,
    )
    assert report.compatible


def test_binding_rejects_wrong_expected_science_cell(tmp_path):
    event = tmp_path / "events.fits"
    write_event(event)

    region = tmp_path / "region.fits"
    event_identity = read_event_identity(event)
    write_fits_region(
        region,
        [triangle()],
        event_identity=event_identity,
        celestial_geometry_sha256="a" * 64,
        projection_identity_sha256="b" * 64,
    )

    report = check_region_binding(
        region,
        event,
        expected_celestial_geometry_sha256="c" * 64,
        expected_projection_identity_sha256="b" * 64,
    )
    assert not report.compatible
    assert "celestial geometry SHA256 differs" in report.mismatches

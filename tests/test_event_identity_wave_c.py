from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.provenance import EventIdentityError, read_event_identity
from xmm_region_tool.sas_validation import (
    SasRegionValidationError,
    _event_hdu,
    _validation_event_copy,
)


def _table(name: str = "EVENTS", *, dety: bool = True, rows=(1.0, 2.0)):
    columns = [fits.Column(name="DETX", format="D", array=np.asarray(rows, dtype=float))]
    if dety:
        columns.append(
            fits.Column(name="DETY", format="D", array=np.asarray(rows, dtype=float) + 10.0)
        )
    return fits.BinTableHDU.from_columns(columns, name=name)


def _write_event(path, *, primary_updates=None, event_updates=None, prefix_hdus=(), suffix_hdus=()):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    for key, value in (primary_updates or {}).items():
        if value is None:
            primary.header.remove(key, ignore_missing=True, remove_all=True)
        else:
            primary.header[key] = value

    events = _table()
    for key, value in (event_updates or {}).items():
        events.header[key] = value
    fits.HDUList([primary, *prefix_hdus, events, *suffix_hdus]).writeto(path)
    return path


def test_identity_uses_named_events_even_with_decoy_detector_table_first(tmp_path):
    decoy = _table("DECOY", rows=(100.0, 200.0))
    path = _write_event(tmp_path / "events.fits", prefix_hdus=(decoy,))

    identity = read_event_identity(path)

    assert identity.instrument == "mos1"
    assert identity.obs_id == "0144310101"

    hdus, event_hdu = _event_hdu(path)
    try:
        assert event_hdu.name == "EVENTS"
        assert np.asarray(event_hdu.data["DETX"]).tolist() == [1.0, 2.0]
    finally:
        hdus.close()


def test_validation_copy_modifies_named_events_not_decoy_table(tmp_path):
    decoy = _table("DECOY", rows=(100.0, 200.0))
    source = _write_event(tmp_path / "source.fits", prefix_hdus=(decoy,))

    target = _validation_event_copy(source, tmp_path / "copy.fits")

    with fits.open(target) as hdus:
        assert "XMMRGROW" not in hdus["DECOY"].columns.names
        assert "XMMRGROW" in hdus["EVENTS"].columns.names
        assert np.asarray(hdus["EVENTS"].data["XMMRGROW"]).tolist() == [0, 1]


def test_identity_requires_named_events_extension(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    path = tmp_path / "not-events.fits"
    fits.HDUList([primary, _table("SCIENCE")]).writeto(path)

    with pytest.raises(EventIdentityError, match="no canonical EVENTS"):
        read_event_identity(path)
    with pytest.raises(SasRegionValidationError, match="no canonical EVENTS"):
        _event_hdu(path)


def test_duplicate_events_extensions_fail_closed(tmp_path):
    path = _write_event(tmp_path / "duplicate.fits", suffix_hdus=(_table("EVENTS"),))

    with pytest.raises(EventIdentityError, match="2 EVENTS extensions"):
        read_event_identity(path)
    with pytest.raises(SasRegionValidationError, match="2 EVENTS extensions"):
        _validation_event_copy(path, tmp_path / "copy.fits")


def test_named_events_must_own_required_detector_columns(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    events = _table("EVENTS", dety=False)
    decoy = _table("DECOY")
    path = tmp_path / "missing-dety.fits"
    fits.HDUList([primary, decoy, events]).writeto(path)

    with pytest.raises(EventIdentityError, match="EVENTS.*DETY"):
        read_event_identity(path)


def test_conflicting_obs_id_across_headers_fails(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"OBS_ID": "9999999999"})

    with pytest.raises(EventIdentityError, match="conflicting event header OBS_ID"):
        read_event_identity(path)


def test_conflicting_obs_id_aliases_fail(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"OBSID": "9999999999"})

    with pytest.raises(EventIdentityError, match="conflicting event header OBS_ID"):
        read_event_identity(path)


def test_instrument_aliases_normalize_to_one_camera_identity(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"INSTRUME": "MOS1"})

    identity = read_event_identity(path)

    assert identity.instrument == "mos1"
    assert identity.instrument_header == "EMOS1"


def test_conflicting_instruments_fail(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"INSTRUME": "EPN"})

    with pytest.raises(EventIdentityError, match="conflicting event header INSTRUME"):
        read_event_identity(path)


def test_conflicting_exposure_aliases_fail(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"EXPIDSTR": "S999"})

    with pytest.raises(EventIdentityError, match="conflicting event header EXP_ID"):
        read_event_identity(path)


@pytest.mark.parametrize("keyword", ["RA_PNT", "DEC_PNT", "PA_PNT"])
def test_conflicting_duplicate_pointing_fails(tmp_path, keyword):
    path = _write_event(tmp_path / "events.fits", event_updates={keyword: 123.0})

    with pytest.raises(EventIdentityError, match=rf"conflicting event header {keyword}"):
        read_event_identity(path)


def test_numeric_pointing_spellings_compare_after_finite_parsing(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"RA_PNT": "15.6730"})

    assert read_event_identity(path).ra_pnt == pytest.approx(15.673)


def test_signed_zero_pointing_spellings_canonicalize(tmp_path):
    path = _write_event(
        tmp_path / "events.fits",
        primary_updates={"RA_PNT": 0.0},
        event_updates={"RA_PNT": "-0.0"},
    )

    value = read_event_identity(path).ra_pnt
    assert value == 0.0
    assert not np.signbit(value)


def test_conflicting_date_aliases_fail(tmp_path):
    path = _write_event(
        tmp_path / "events.fits",
        event_updates={"DATE_OBS": "2003-06-23T00:00:00"},
    )

    with pytest.raises(EventIdentityError, match="conflicting event header DATE-OBS"):
        read_event_identity(path)


def test_compatible_xmm_telescope_spellings_normalize(tmp_path):
    path = _write_event(tmp_path / "events.fits", event_updates={"TELESCOP": "XMM-Newton"})

    assert read_event_identity(path).telescope == "XMM"

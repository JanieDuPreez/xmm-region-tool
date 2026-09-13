from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.provenance import EventIdentityError, read_event_identity


def _write_event(path, *, primary_updates=None, event_updates=None):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0723802001"
    primary.header["EXP_ID"] = "0723802001001"
    primary.header["EXPIDSTR"] = "S001"
    primary.header["DATE-OBS"] = "2013-12-18T12:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    for key, value in (primary_updates or {}).items():
        if value is None:
            primary.header.remove(key, ignore_missing=True, remove_all=True)
        else:
            primary.header[key] = value

    columns = [
        fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
        fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
    ]
    events = fits.BinTableHDU.from_columns(columns, name="EVENTS")
    for key, value in (event_updates or {}).items():
        events.header[key] = value
    fits.HDUList([primary, events]).writeto(path)
    return path


def test_real_sas_long_exp_id_and_short_expidstr_are_not_alias_conflicts(tmp_path):
    path = _write_event(tmp_path / "mos1S001-allevc.fits")

    identity = read_event_identity(path)

    assert identity.obs_id == "0723802001"
    assert identity.exposure_id == "S001"


def test_real_sas_expidstr_is_preferred_when_long_exp_id_is_also_present(tmp_path):
    path = _write_event(
        tmp_path / "mos1S001-allevc.fits",
        event_updates={"EXP_ID": "0723802001001", "EXPIDSTR": "S001"},
    )

    assert read_event_identity(path).exposure_id == "S001"


def test_conflicting_long_exp_id_duplicates_still_fail_closed(tmp_path):
    path = _write_event(
        tmp_path / "mos1S001-allevc.fits",
        event_updates={"EXP_ID": "9999999999999", "EXPIDSTR": "S001"},
    )

    with pytest.raises(EventIdentityError, match="conflicting event header EXP_ID"):
        read_event_identity(path)


def test_conflicting_short_exposure_duplicates_still_fail_closed(tmp_path):
    path = _write_event(
        tmp_path / "mos1S001-allevc.fits",
        event_updates={"EXP_ID": "0723802001001", "EXPIDSTR": "S999"},
    )

    with pytest.raises(EventIdentityError, match="conflicting event header EXPIDSTR"):
        read_event_identity(path)

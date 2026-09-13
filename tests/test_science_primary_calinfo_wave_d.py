from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.event_roles import read_science_calinfoset_identity
from xmm_region_tool.provenance import EventIdentityError, read_event_identity


def test_science_calinfo_uses_primary_when_events_timing_and_pointing_differ(tmp_path):
    path = tmp_path / "events.fits"

    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0723802001"
    primary.header["EXPIDSTR"] = "S001"
    primary.header["DATE-OBS"] = "2013-06-08T18:19:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0

    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    events.header["TELESCOP"] = "XMM"
    events.header["INSTRUME"] = "EMOS1"
    events.header["OBS_ID"] = "0723802001"
    events.header["EXPIDSTR"] = "S001"
    events.header["DATE-OBS"] = "2013-06-08T15:09:56.000"
    events.header["RA_PNT"] = 15.7
    events.header["DEC_PNT"] = -21.9
    events.header["PA_PNT"] = 71.0

    fits.HDUList([primary, events]).writeto(path)

    # Generic inspection deliberately remains conservative about conflicting
    # cross-HDU metadata.
    with pytest.raises(EventIdentityError, match="conflicting event header DATE-OBS"):
        read_event_identity(path)

    # The science calinfostyle=set role follows SAS header ownership instead:
    # the material date/pointing values are those in PRIMARY.
    identity = read_science_calinfoset_identity(path)
    assert identity.date_obs == "2013-06-08T18:19:00"
    assert identity.ra_pnt == pytest.approx(15.673)
    assert identity.dec_pnt == pytest.approx(-21.88)
    assert identity.pa_pnt == pytest.approx(72.0)

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.event_roles import read_science_calinfoset_identity
from xmm_region_tool.provenance import EventIdentityError, read_event_identity


def _event(path, *, instrument="EPN"):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = instrument
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S003"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
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
    fits.HDUList([primary, events]).writeto(path)
    return path


@pytest.mark.parametrize("name", ["pnS003-allevcoot.fits", "pnS003-allevc-oot.fits"])
def test_known_pn_oot_names_remain_generically_inspectable_but_not_science_events(tmp_path, name):
    path = _event(tmp_path / name)

    assert read_event_identity(path).instrument == "pn"
    with pytest.raises(EventIdentityError, match="pn OOT product.*normal science event"):
        read_science_calinfoset_identity(path)


def test_normal_pn_allevc_remains_valid_science_event(tmp_path):
    path = _event(tmp_path / "pnS003-allevc.fits")

    assert read_science_calinfoset_identity(path).instrument == "pn"


def test_oot_like_basename_does_not_reclassify_mos_product(tmp_path):
    path = _event(tmp_path / "mos1S001-allevcoot.fits", instrument="EMOS1")

    assert read_science_calinfoset_identity(path).instrument == "mos1"


def test_arbitrarily_renamed_pn_product_cannot_be_classified_from_identity_alone(tmp_path):
    path = _event(tmp_path / "renamed-event.fits")

    # The role guard is deliberately a deterministic known-product-name check,
    # not a claim that generic EPIC headers cryptographically identify OOT data.
    assert read_science_calinfoset_identity(path).instrument == "pn"

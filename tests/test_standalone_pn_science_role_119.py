from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import xmm_region_tool.cli as cli
from xmm_region_tool.provenance import EventIdentityError, read_event_identity
from xmm_region_tool.workflow import (
    WorkflowError,
    discover_all_event_files,
    discover_event_files,
    pn_oot_event_file,
)


def _legacy_pn_science_event(path, *, marker=None):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EPN"
    primary.header["OBS_ID"] = "0693010301"
    primary.header["EXP_ID"] = "0693010301003"
    primary.header["EXPIDSTR"] = "S003"
    primary.header["DATE-OBS"] = "2012-06-08T03:46:58"
    primary.header["DATE_OBS"] = "2012-06-08T02:45:44.000"
    primary.header["RA_PNT"] = 4.9095747
    primary.header["DEC_PNT"] = 3.6033689
    primary.header["PA_PNT"] = 0.0
    if marker is not None:
        primary.header["TESTMARK"] = marker

    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    events.header["TELESCOP"] = "XMM"
    events.header["INSTRUME"] = "EPN"
    events.header["OBS_ID"] = "0693010301"
    events.header["EXP_ID"] = "0693010301003"
    events.header["EXPIDSTR"] = "S003"
    events.header["DATE-OBS"] = "2012-06-08T03:46:58"
    events.header["DATE_OBS"] = "2012-06-08T02:45:44.000"

    fits.HDUList([primary, events]).writeto(path)
    return path


def test_generic_identity_remains_strict_for_legacy_pn_timing_aliases(tmp_path):
    pn = _legacy_pn_science_event(tmp_path / "pnS003-allevc.fits")

    with pytest.raises(EventIdentityError, match="conflicting event header DATE-OBS"):
        read_event_identity(pn)


def test_canonical_discovery_uses_science_calinfoset_role_for_legacy_pn(tmp_path):
    pn = _legacy_pn_science_event(tmp_path / "pnS003-allevc.fits")

    assert discover_event_files(tmp_path, ("pn",)) == (pn.resolve(),)
    assert discover_all_event_files(tmp_path) == (pn.resolve(),)


def test_cli_event_set_preflight_uses_science_calinfoset_role_for_legacy_pn(tmp_path):
    pn = _legacy_pn_science_event(tmp_path / "pnS003-allevc.fits")

    ((path, identity),) = cli._validate_event_set((pn.resolve(),))

    assert path == pn.resolve()
    assert identity.instrument == "pn"
    assert identity.obs_id == "0693010301"
    assert identity.exposure_id == "S003"
    assert identity.date_obs == "2012-06-08T03:46:58"


def test_canonical_looking_invalid_candidate_is_not_silently_omitted(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    pn.write_bytes(b"not a fits event")

    with pytest.raises(
        WorkflowError,
        match=r"canonical-looking pn ESAS event file cannot be used.*pnS003-allevc\.fits",
    ):
        discover_all_event_files(tmp_path)


def test_pn_oot_pairing_ignores_nonmaterial_legacy_timing_alias_conflicts(tmp_path):
    pn = _legacy_pn_science_event(tmp_path / "pnS003-allevc.fits", marker="science")
    oot = _legacy_pn_science_event(tmp_path / "pnS003-allevcoot.fits", marker="oot")

    assert pn_oot_event_file(pn) == oot.resolve()

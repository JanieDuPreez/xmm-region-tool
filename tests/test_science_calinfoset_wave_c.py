from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

import xmm_region_tool.event_roles as event_roles
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.event_roles import (
    read_science_calinfoset_identity,
    read_science_calinfoset_snapshot,
)
from xmm_region_tool.execution import SasProjectionContext
from xmm_region_tool.model import CelestialSelection
from xmm_region_tool.provenance import EventIdentityError, file_sha256, read_event_identity
from xmm_region_tool.sas import _project_skycoords, project_selection


def _write_event(path, *, remove=(), updates=None, event_updates=None):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    for key in remove:
        primary.header.remove(key, ignore_missing=True, remove_all=True)
    for key, value in (updates or {}).items():
        primary.header[key] = value

    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    for key, value in (event_updates or {}).items():
        events.header[key] = value
    fits.HDUList([primary, events]).writeto(path)
    return path


def _context(tmp_path: Path) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_text("fake")
    cif = tmp_path / "ccf.cif"
    cif.write_text("fake")
    return SasProjectionContext(
        environment={"PATH": str(tmp_path), "SAS_CCF": str(cif)},
        calibration=CalibrationIdentity(
            cif_path=cif,
            cif_file_sha256="1" * 64,
            calindex_sha256="2" * 64,
            replacements=(),
            ccf_search_path=(),
        ),
        producer=SasProducerIdentity(
            esky2det_version="22",
            sas_version="22",
            esky2det_sha256="3" * 64,
        ),
        esky2det_path=executable,
    )


def _selection() -> CelestialSelection:
    return CelestialSelection.polygon(
        SkyCoord(
            [15.67, 15.68, 15.67] * u.deg,
            [-21.88, -21.88, -21.87] * u.deg,
            frame="icrs",
        )
    )


def test_nominal_science_calinfoset_is_accepted(tmp_path):
    path = _write_event(tmp_path / "events.fits")

    identity = read_science_calinfoset_identity(path)

    assert identity.instrument == "mos1"
    assert identity.date_obs == "2003-06-22T00:00:00"
    assert identity.ra_pnt == pytest.approx(15.673)
    assert identity.dec_pnt == pytest.approx(-21.88)
    assert identity.pa_pnt == pytest.approx(72.0)


@pytest.mark.parametrize(
    "keyword",
    ["TELESCOP", "INSTRUME", "DATE-OBS", "RA_PNT", "DEC_PNT", "PA_PNT"],
)
def test_science_calinfoset_requires_complete_primary_header(tmp_path, keyword):
    event_updates = {}
    if keyword != "TELESCOP":
        fallback = {
            "INSTRUME": "EMOS1",
            "DATE-OBS": "2003-06-22T00:00:00",
            "RA_PNT": 15.673,
            "DEC_PNT": -21.88,
            "PA_PNT": 72.0,
        }
        event_updates[keyword] = fallback[keyword]
    path = _write_event(
        tmp_path / "events.fits",
        remove=(keyword,),
        event_updates=event_updates,
    )

    # Generic identity inspection may still use a coherent secondary copy.
    if keyword != "TELESCOP":
        assert read_event_identity(path)

    with pytest.raises(EventIdentityError, match=rf"primary header.*{keyword}"):
        read_science_calinfoset_identity(path)


def test_science_calinfoset_rejects_malformed_date(tmp_path):
    path = _write_event(tmp_path / "events.fits", updates={"DATE-OBS": "not-a-date"})

    with pytest.raises(EventIdentityError, match="DATE-OBS.*valid FITS"):
        read_science_calinfoset_identity(path)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("RA_PNT", -0.1, r"RA_PNT.*\[0, 360\)"),
        ("RA_PNT", 360.0, r"RA_PNT.*\[0, 360\)"),
        ("DEC_PNT", -90.1, r"DEC_PNT.*\[-90, 90\]"),
        ("DEC_PNT", 90.1, r"DEC_PNT.*\[-90, 90\]"),
        ("PA_PNT", -0.1, r"PA_PNT.*\[0, 360\)"),
        ("PA_PNT", 360.0, r"PA_PNT.*\[0, 360\)"),
    ],
)
def test_science_calinfoset_rejects_out_of_domain_pointing(tmp_path, keyword, value, message):
    path = _write_event(tmp_path / "events.fits", updates={keyword: value})

    with pytest.raises(EventIdentityError, match=message):
        read_science_calinfoset_identity(path)


def test_science_snapshot_rejects_persistent_replacement_during_identity_read(monkeypatch, tmp_path):
    event = _write_event(tmp_path / "events.fits")
    replacement = _write_event(
        tmp_path / "replacement.fits",
        updates={"OBS_ID": "9999999999"},
    )
    generation_a_sha = file_sha256(event)
    generation_b_sha = file_sha256(replacement)
    assert generation_a_sha != generation_b_sha

    real_hash = file_sha256
    calls = 0

    def replace_after_first_hash(path):
        nonlocal calls
        calls += 1
        digest = real_hash(path)
        if calls == 1:
            replacement.replace(event)
        return digest

    monkeypatch.setattr(event_roles, "file_sha256", replace_after_first_hash)

    with pytest.raises(EventIdentityError, match="bytes changed.*incoherent event snapshot"):
        read_science_calinfoset_snapshot(event)

    assert calls == 2
    assert read_event_identity(event).obs_id == "9999999999"
    assert real_hash(event) == generation_b_sha


def test_science_snapshot_accepts_stable_new_bytes_with_same_semantic_identity(tmp_path):
    event = _write_event(tmp_path / "events.fits")
    replacement = _write_event(
        tmp_path / "replacement.fits",
        updates={"TESTTAG": "different exact bytes, same semantic identity"},
    )
    original_identity = read_science_calinfoset_identity(event)
    replacement_identity = read_science_calinfoset_identity(replacement)
    replacement_sha = file_sha256(replacement)
    assert original_identity.identity_sha256 == replacement_identity.identity_sha256
    assert file_sha256(event) != replacement_sha

    replacement.replace(event)
    identity, exact_sha = read_science_calinfoset_snapshot(event)

    assert identity.identity_sha256 == replacement_identity.identity_sha256
    assert exact_sha == replacement_sha


def test_project_selection_rejects_incomplete_calinfo_before_esky2det(monkeypatch, tmp_path):
    event = _write_event(tmp_path / "events.fits", remove=("RA_PNT",))
    monkeypatch.setattr(SasProjectionContext, "validate", lambda self: None)

    def must_not_run(*args, **kwargs):
        pytest.fail("esky2det subprocess was reached before science-calinfoset preflight")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", must_not_run)

    with pytest.raises(EventIdentityError, match="RA_PNT"):
        project_selection(_selection(), calinfoset=event, context=_context(tmp_path))


def test_esky2det_command_explicitly_requests_calinfostyle_set(monkeypatch, tmp_path):
    calinfo = _write_event(tmp_path / "events.fits")
    executable = tmp_path / "esky2det"
    executable.write_text("fake")
    captured = []

    def fake_run(command, **kwargs):
        captured.append(tuple(command))
        intab = next(value for value in command if value.startswith("intab="))
        path = Path(intab.removeprefix("intab=").split(":", maxsplit=1)[0])
        with fits.open(path) as hdus:
            source = hdus["INPUT"].data
            output = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="D", array=source["RA"]),
                    fits.Column(name="DEC", format="D", array=source["DEC"]),
                    fits.Column(name="ROW_ID", format="J", array=source["ROW_ID"]),
                    fits.Column(name="DETX", format="D", array=np.asarray([1.0])),
                    fits.Column(name="DETY", format="D", array=np.asarray([2.0])),
                ],
                name="INPUT",
            )
        fits.HDUList([fits.PrimaryHDU(), output]).writeto(path, overwrite=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)
    _project_skycoords(
        SkyCoord([15.673] * u.deg, [-21.88] * u.deg, frame="icrs"),
        calinfoset=calinfo,
        esky2det=executable,
        environment={"PATH": str(tmp_path)},
    )

    assert "calinfostyle=set" in captured[0]

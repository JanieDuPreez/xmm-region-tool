from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import ProjectionRule, SasProjectionContext
from xmm_region_tool.model import CelestialSelection
from xmm_region_tool.sas import SasConversionError, project_selection


def _write_event(path: Path) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0723802001"
    primary.header["EXP_ID"] = "0723802001001"
    primary.header["EXPIDSTR"] = "S001"
    primary.header["DATE-OBS"] = "2013-06-08T15:10:41"
    primary.header["RA_PNT"] = 15.645
    primary.header["DEC_PNT"] = -21.8721388888889
    primary.header["PA_PNT"] = 55.8575019836426
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([2.0])),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def _write_cif(path: Path, *, obsvdate: str) -> Path:
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="FNAME", format="32A", array=["A.CCF"])],
        name="CALINDEX",
    )
    table.header["OBSVDATE"] = obsvdate
    table.header["ANALDATE"] = "2026-07-13T13:45:48"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    return path


def _observation_block(obs_id: str, start: str, end: str) -> str:
    return "\n".join(
        ["OBSERVATION", f"{obs_id:<10} 2472", "unused", start, end]
    ) + "\n"


def _context(tmp_path: Path, cif: Path, sosf: Path) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_text("fake executable")
    return SasProjectionContext(
        environment={
            "PATH": str(tmp_path),
            "SAS_CCF": str(cif),
            "SAS_ODF": str(sosf),
        },
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
    vertices = SkyCoord(
        [15.64, 15.65, 15.645] * u.deg,
        [-21.87, -21.87, -21.86] * u.deg,
        frame="icrs",
    )
    return CelestialSelection.polygon(vertices)


def test_stale_cif_fails_before_any_projection_call(monkeypatch, tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    cif = _write_cif(tmp_path / "ccf.cif", obsvdate="2003-04-01T00:00:00")
    odf_dir = tmp_path / "odf"
    odf_dir.mkdir()
    (odf_dir / "2472_0723802001_SCX00000SUM.ASC").write_text(
        _observation_block(
            "0723802001",
            "2013-06-08T14:52:32",
            "2013-06-09T16:34:12",
        )
    )
    sosf = tmp_path / "2472_0723802001_SCX00000SUM.SAS"
    sosf.write_text(
        _observation_block(
            "0723802001",
            "2013-06-08T15:09:56",
            "2013-06-09T15:44:18",
        )
        + f"PATH {odf_dir}\n"
    )
    ctx = _context(tmp_path, cif, sosf)

    monkeypatch.setattr(SasProjectionContext, "validate", lambda self: None)

    def unexpected_projection(*args, **kwargs):
        pytest.fail("stale CIF reached detector projection")

    monkeypatch.setattr(
        "xmm_region_tool.sas._convert_regions_with_esky2det",
        unexpected_projection,
    )

    with pytest.raises(SasConversionError, match="CIF OBSVDATE does not match"):
        project_selection(
            _selection(),
            calinfoset=event,
            context=ctx,
            rule=ProjectionRule(samples=16),
        )


def test_sosf_obsid_mismatch_fails_before_any_projection_call(monkeypatch, tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    cif = _write_cif(tmp_path / "ccf.cif", obsvdate="2013-06-08T14:52:32")
    odf_dir = tmp_path / "odf"
    odf_dir.mkdir()
    (odf_dir / "2472_0723802001_SCX00000SUM.ASC").write_text(
        _observation_block(
            "0723802001",
            "2013-06-08T14:52:32",
            "2013-06-09T16:34:12",
        )
    )
    sosf = tmp_path / "other.SAS"
    sosf.write_text(
        _observation_block(
            "0144310101",
            "2013-06-08T15:09:56",
            "2013-06-09T15:44:18",
        )
        + f"PATH {odf_dir}\n"
    )
    ctx = _context(tmp_path, cif, sosf)

    monkeypatch.setattr(SasProjectionContext, "validate", lambda self: None)

    def unexpected_projection(*args, **kwargs):
        pytest.fail("wrong-observation SOSF reached detector projection")

    monkeypatch.setattr(
        "xmm_region_tool.sas._convert_regions_with_esky2det",
        unexpected_projection,
    )

    with pytest.raises(SasConversionError, match="active SAS ODF summary belongs"):
        project_selection(
            _selection(),
            calinfoset=event,
            context=ctx,
            rule=ProjectionRule(samples=16),
        )

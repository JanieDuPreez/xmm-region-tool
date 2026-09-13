from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import xmm_region_tool.sas_validation as sas_validation
from xmm_region_tool.output import write_fits_region
from xmm_region_tool.provenance import (
    check_region_binding,
    file_sha256,
    read_event_identity,
)
from xmm_region_tool.sas import DetectorBoundary, DetectorRegion
from xmm_region_tool.sas_validation import SasRegionValidationError


def _write_event(path: Path, *, marker: str | None = None) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    if marker is not None:
        primary.header["TESTTAG"] = marker
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([0.25, 0.75, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([0.25, 0.75, 2.0])),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def _region(path: Path, event: Path) -> Path:
    detector = DetectorRegion(
        True,
        (
            DetectorBoundary(
                np.asarray(
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    dtype=float,
                )
            ),
        ),
    )
    return write_fits_region(
        path,
        [detector],
        event_identity=read_event_identity(event),
        event_file_sha256=file_sha256(event),
    )


def test_validator_rejects_event_replacement_before_snapshot(monkeypatch, tmp_path):
    event = _write_event(tmp_path / "events.fits")
    replacement = _write_event(tmp_path / "replacement.fits", marker="generation-b")
    region = _region(tmp_path / "region.fits", event)
    sha_a = file_sha256(event)
    sha_b = file_sha256(replacement)
    assert sha_a != sha_b
    assert read_event_identity(event).identity_sha256 == read_event_identity(replacement).identity_sha256

    def binding_then_replace(region_fits, event_file, *, source_region=None):
        report = check_region_binding(
            region_fits,
            event_file,
            source_region=source_region,
        )
        assert report.compatible
        assert report.event_file_sha256 == sha_a
        replacement.replace(event)
        return report

    monkeypatch.setattr(sas_validation, "check_region_binding", binding_then_replace)
    monkeypatch.setattr(
        sas_validation,
        "_producer_capture",
        lambda value: SimpleNamespace(executable_path=Path("/sas/bin/evselect")),
    )

    def must_not_run(*args, **kwargs):
        pytest.fail("evselect ran after event generation drift")

    monkeypatch.setattr(sas_validation, "run_bounded", must_not_run)

    with pytest.raises(
        SasRegionValidationError,
        match="changed while the real-SAS validation snapshot was being captured",
    ):
        sas_validation.validate_fits_region_with_evselect(region, event)

    assert file_sha256(event) == sha_b

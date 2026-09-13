from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.geometry import ParsedSkyRegion, SkyBoundary
from xmm_region_tool.sas import SasConversionError, convert_with_esky2det


def sample_region() -> list[ParsedSkyRegion]:
    vertices = SkyCoord(
        [10.0, 10.001, 10.0] * u.deg,
        [-9.0, -9.0, -8.999] * u.deg,
        frame="icrs",
    )
    return [ParsedSkyRegion(True, (SkyBoundary(vertices),))]


def test_missing_esky2det_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr("xmm_region_tool.sas.shutil.which", lambda _: None)

    with pytest.raises(SasConversionError, match="esky2det.*PATH"):
        convert_with_esky2det(sample_region(), calinfoset=tmp_path / "events.fits")


def test_missing_event_file_is_rejected_before_sas(monkeypatch, tmp_path):
    monkeypatch.setattr("xmm_region_tool.sas.shutil.which", lambda _: "/sas/bin/esky2det")
    missing = Path(tmp_path) / "missing-events.fits"

    with pytest.raises(SasConversionError, match="does not exist"):
        convert_with_esky2det(sample_region(), calinfoset=missing)


def write_calinfo(path):
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    return path


def test_bulk_conversion_uses_in_place_table_and_preserves_row_id(monkeypatch, tmp_path):
    monkeypatch.setattr("xmm_region_tool.sas.shutil.which", lambda _: "/sas/bin/esky2det")
    event = write_calinfo(tmp_path / "events.fits")
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        intab = next(value for value in command if value.startswith("intab="))
        path = Path(intab.removeprefix("intab=").split(":", maxsplit=1)[0])
        with fits.open(path) as hdus:
            source = hdus["INPUT"].data
            assert hdus["INPUT"].columns["ROW_ID"].format == "J"
            output = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="D", array=source["RA"]),
                    fits.Column(name="DEC", format="D", array=source["DEC"]),
                    fits.Column(name="ROW_ID", format="J", array=source["ROW_ID"]),
                    fits.Column(
                        name="DETX",
                        format="E",
                        array=np.asarray([100.0, 200.0, 300.0], dtype=np.float32),
                    ),
                    fits.Column(
                        name="DETY",
                        format="E",
                        array=np.asarray([400.0, 500.0, 550.0], dtype=np.float32),
                    ),
                ],
                name="INPUT",
            )
        fits.HDUList([fits.PrimaryHDU(), output]).writeto(path, overwrite=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    converted = convert_with_esky2det(sample_region(), calinfoset=event)

    assert "withouttab=no" in captured[0]
    assert not any(value.startswith("outtab=") for value in captured[0])
    assert "checkfov=no" in captured[0]
    assert converted[0].boundaries[0].vertices.tolist() == [
        [100.0, 400.0],
        [200.0, 500.0],
        [300.0, 550.0],
    ]


def test_bulk_conversion_rejects_reordered_row_identity(monkeypatch, tmp_path):
    monkeypatch.setattr("xmm_region_tool.sas.shutil.which", lambda _: "/sas/bin/esky2det")
    event = write_calinfo(tmp_path / "events.fits")

    def fake_run(command, **kwargs):
        intab = next(value for value in command if value.startswith("intab="))
        path = Path(intab.removeprefix("intab=").split(":", maxsplit=1)[0])
        with fits.open(path) as hdus:
            source = hdus["INPUT"].data
            output = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="D", array=source["RA"]),
                    fits.Column(name="DEC", format="D", array=source["DEC"]),
                    fits.Column(
                        name="ROW_ID",
                        format="J",
                        array=np.asarray([1, 0, 2], dtype=np.int32),
                    ),
                    fits.Column(name="DETX", format="E", array=np.ones(3, dtype=np.float32)),
                    fits.Column(name="DETY", format="E", array=np.ones(3, dtype=np.float32)),
                ],
                name="INPUT",
            )
        fits.HDUList([fits.PrimaryHDU(), output]).writeto(path, overwrite=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    with pytest.raises(SasConversionError, match="ROW_ID sequence"):
        convert_with_esky2det(sample_region(), calinfoset=event)


def test_sas_failure_exposes_structured_execution_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr("xmm_region_tool.sas.shutil.which", lambda _: "/sas/bin/esky2det")
    event = write_calinfo(tmp_path / "events.fits")

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=17,
            stdout="synthetic stdout",
            stderr="synthetic SAS failure",
        )

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    with pytest.raises(SasConversionError, match="exit code 17") as caught:
        convert_with_esky2det(sample_region(), calinfoset=event)

    error = caught.value
    assert error.returncode == 17
    assert error.stdout == "synthetic stdout"
    assert error.stderr == "synthetic SAS failure"
    assert error.command is not None
    assert error.command[0] == "/sas/bin/esky2det"
    assert any(value.startswith("calinfoset=") for value in error.command)

    evidence = error.evidence_record()
    assert evidence["returncode"] == 17
    assert evidence["stdout"] == "synthetic stdout"
    assert evidence["stderr"] == "synthetic SAS failure"
    assert evidence["command"] == list(error.command)

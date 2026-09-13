from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool import batch, calibration
from xmm_region_tool.batch import convert_selection_batch
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import ProjectionRule, SasProjectionContext
from xmm_region_tool.model import CelestialSelection
from xmm_region_tool.sas import SasConversionError
from xmm_region_tool.subprocess_capture import (
    DIAGNOSTIC_STREAM_LIMIT,
    run_bounded,
)


def _write_event(path: Path) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "001"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2000-01-01T00:00:00"
    primary.header["RA_PNT"] = 10.0
    primary.header["DEC_PNT"] = -9.0
    primary.header["PA_PNT"] = 0.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="J", array=[1]),
            fits.Column(name="DETY", format="J", array=[1]),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def _context(tmp_path: Path) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_bytes(b"synthetic executable")
    cif = tmp_path / "ccf.cif"
    cif.write_bytes(b"synthetic cif")
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
            esky2det_version="esky2det-test",
            sas_version="SAS-test",
            esky2det_sha256="3" * 64,
        ),
        esky2det_path=executable,
    )


def _selection() -> CelestialSelection:
    vertices = SkyCoord(
        [10.0, 10.001, 10.0] * u.deg,
        [-9.0, -9.0, -8.999] * u.deg,
        frame="icrs",
    )
    return CelestialSelection.polygon(vertices)


def test_run_bounded_real_child_retains_only_head_and_tail():
    payload_size = DIAGNOSTIC_STREAM_LIMIT * 3
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.stdout.write('HEAD-' + 'x' * "
            f"{payload_size} + '-TAIL'); "
            "sys.stderr.write('short stderr')"
        ),
    ]

    result = run_bounded(command)

    assert result.returncode == 0
    assert result.stdout_capture.truncated is True
    assert result.stdout_capture.total_bytes > DIAGNOSTIC_STREAM_LIMIT
    assert result.stdout_capture.retained_bytes == DIAGNOSTIC_STREAM_LIMIT
    assert result.stdout.startswith("HEAD-")
    assert result.stdout.endswith("-TAIL")
    assert "xmm-region-tool truncated" in result.stdout
    assert len(result.stdout.encode("utf-8")) < DIAGNOSTIC_STREAM_LIMIT + 100
    assert result.stderr == "short stderr"
    assert result.stderr_capture.truncated is False


def test_run_bounded_fake_runner_cannot_return_unbounded_diagnostics(monkeypatch):
    noisy_stdout = "HEAD-" + "x" * (DIAGNOSTIC_STREAM_LIMIT * 2) + "-TAIL"
    noisy_stderr = "ERRHEAD-" + "y" * (DIAGNOSTIC_STREAM_LIMIT * 2) + "-ERRTAIL"

    def fake_run(command, **kwargs):
        assert hasattr(kwargs["stdout"], "write")
        assert hasattr(kwargs["stderr"], "write")
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=17, stdout=noisy_stdout, stderr=noisy_stderr)

    monkeypatch.setattr("xmm_region_tool.subprocess_capture.subprocess.run", fake_run)

    result = run_bounded(["synthetic-sas"])

    assert result.returncode == 17
    assert result.stdout_capture.truncated is True
    assert result.stderr_capture.truncated is True
    assert result.stdout.startswith("HEAD-") and result.stdout.endswith("-TAIL")
    assert result.stderr.startswith("ERRHEAD-") and result.stderr.endswith("-ERRTAIL")
    assert result.stdout_capture.total_bytes == len(noisy_stdout.encode())
    assert result.stderr_capture.total_bytes == len(noisy_stderr.encode())


def test_noisy_projection_failure_manifest_is_bounded(monkeypatch, tmp_path):
    event = _write_event(tmp_path / "events.fits")
    context = _context(tmp_path)
    noisy = "HEAD-" + "z" * (DIAGNOSTIC_STREAM_LIMIT * 3) + "-TAIL"

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=23, stdout=noisy, stderr="short failure")

    monkeypatch.setattr("xmm_region_tool.subprocess_capture.subprocess.run", fake_run)
    captured = run_bounded(["synthetic-esky2det"])

    def fail_projection(*args, **kwargs):
        raise SasConversionError(
            "esky2det failed with exit code 23: short failure",
            command=("synthetic-esky2det",),
            returncode=23,
            stdout=captured.stdout,
            stderr=captured.stderr,
            stdout_capture=captured.stdout_capture,
            stderr_capture=captured.stderr_capture,
        )

    monkeypatch.setattr(batch, "project_selection", fail_projection)

    result = convert_selection_batch(
        _selection(),
        [event],
        tmp_path / "out",
        rule=ProjectionRule(samples=128),
        context=context,
    )

    assert result.successful is False
    manifest_text = result.manifest.read_text()
    manifest = json.loads(manifest_text)
    failure = manifest["items"][0]["failure_evidence"]
    assert failure["returncode"] == 23
    assert failure["stdout_evidence"]["truncated"] is True
    assert failure["stdout_evidence"]["total_bytes"] == len(noisy.encode())
    assert failure["stdout_evidence"]["retained_bytes"] == DIAGNOSTIC_STREAM_LIMIT
    assert failure["stdout"].startswith("HEAD-")
    assert failure["stdout"].endswith("-TAIL")
    assert "xmm-region-tool truncated" in failure["stdout"]
    assert len(manifest_text.encode()) < DIAGNOSTIC_STREAM_LIMIT + 20_000


def test_esky2det_version_probe_rejects_truncated_output(monkeypatch, tmp_path):
    executable = tmp_path / "esky2det"
    executable.write_bytes(b"synthetic executable")
    noisy = (
        "esky2det (esky2det-1.20) [22.1.0-test]\n"
        + "x" * (DIAGNOSTIC_STREAM_LIMIT * 2)
    )

    monkeypatch.setattr(
        calibration,
        "resolve_executable",
        lambda name, environment=None: executable,
    )
    monkeypatch.setattr(
        "xmm_region_tool.subprocess_capture.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout=noisy, stderr=""),
    )

    with pytest.raises(calibration.CalibrationIdentityError, match="truncated diagnostic output"):
        calibration._task_version("esky2det", environment={"PATH": str(tmp_path)})


def test_sasversion_probe_rejects_truncated_output(monkeypatch, tmp_path):
    esky2det = tmp_path / "esky2det"
    sasversion = tmp_path / "sasversion"
    esky2det.write_bytes(b"synthetic executable")
    sasversion.write_bytes(b"synthetic sasversion")
    calls = {"count": 0}

    monkeypatch.setattr(
        calibration,
        "resolve_executable",
        lambda name, environment=None: {"esky2det": esky2det, "sasversion": sasversion}[name],
    )

    def fake_run(command, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(
                returncode=0,
                stdout="esky2det (esky2det-1.20) [22.1.0-test]\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout="SAS release: 22.1.0-test\n" + "y" * (DIAGNOSTIC_STREAM_LIMIT * 2),
            stderr="",
        )

    monkeypatch.setattr("xmm_region_tool.subprocess_capture.subprocess.run", fake_run)

    with pytest.raises(calibration.CalibrationIdentityError, match="truncated sasversion output"):
        calibration.read_sas_producer_identity({"PATH": str(tmp_path)})

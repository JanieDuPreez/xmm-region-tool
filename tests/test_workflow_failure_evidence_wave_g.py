from __future__ import annotations

import json
from types import SimpleNamespace

from xmm_region_tool import cli
from xmm_region_tool.sas import SasConversionError


def _install_cli_failure_fixture(monkeypatch, tmp_path, exc):
    source = tmp_path / "source.reg"
    source.write_text('icrs\ncircle(10,-9,30")\n')
    event = tmp_path / "events.fits"
    event.write_bytes(b"synthetic event bytes")

    event_identity = SimpleNamespace(
        obs_id="001",
        instrument="mos1",
        exposure_id="S001",
    )
    context = SimpleNamespace(
        identity_sha256="b" * 64,
        calibration=SimpleNamespace(
            identity_sha256="c" * 64,
            cif_path=tmp_path / "ccf.cif",
        ),
        producer=SimpleNamespace(identity_sha256="d" * 64),
    )
    selection = SimpleNamespace(
        geometry_sha256="a" * 64,
        canonical_geometry_record=lambda: {
            "schema": "synthetic-test-selection/v1",
            "regions": [],
        },
    )

    monkeypatch.setattr(cli, "_resolve_events", lambda args: (event,))
    monkeypatch.setattr(cli, "_validate_event_set", lambda paths: ((event, event_identity),))
    monkeypatch.setattr(cli.SasProjectionContext, "from_environment", lambda: context)
    monkeypatch.setattr(
        cli,
        "load_ds9_selection_snapshot",
        lambda path, samples: (selection, cli.file_sha256(path)),
    )
    monkeypatch.setattr(
        cli,
        "extraction_cells",
        lambda value: (cli.ExtractionCell(index=1, label="r001", selection=selection),),
    )

    def fail_projection(*args, **kwargs):
        raise exc

    monkeypatch.setattr(cli, "project_selection", fail_projection)
    return source, event


def test_main_workflow_manifest_preserves_structured_sas_failure_evidence(
    monkeypatch,
    tmp_path,
):
    exc = SasConversionError(
        "esky2det failed with exit code 17: synthetic SAS failure",
        command=("/sas/bin/esky2det", "checkfov=no"),
        returncode=17,
        stdout="synthetic stdout",
        stderr="synthetic SAS failure",
    )
    source, event = _install_cli_failure_fixture(monkeypatch, tmp_path, exc)

    status = cli.main(
        [
            str(source),
            "--event-file",
            str(event),
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert status == 1
    manifest = json.loads((tmp_path / "source-xmm-region-manifest.json").read_text())
    assert manifest["successful"] is False
    product = manifest["products"][0]
    assert product["status"] == "failed"
    assert product["error_type"] == "SasConversionError"
    evidence = product["failure_evidence"]
    assert evidence["command"] == ["/sas/bin/esky2det", "checkfov=no"]
    assert evidence["returncode"] == 17
    assert evidence["stdout"] == "synthetic stdout"
    assert evidence["stderr"] == "synthetic SAS failure"
    assert evidence["stdout_evidence"] == {
        "total_bytes": len("synthetic stdout"),
        "retained_bytes": len("synthetic stdout"),
        "truncated": False,
    }
    assert evidence["stderr_evidence"] == {
        "total_bytes": len("synthetic SAS failure"),
        "retained_bytes": len("synthetic SAS failure"),
        "truncated": False,
    }


def test_main_workflow_non_sas_failure_does_not_invent_execution_evidence(
    monkeypatch,
    tmp_path,
):
    source, event = _install_cli_failure_fixture(
        monkeypatch,
        tmp_path,
        ValueError("synthetic geometry failure"),
    )

    status = cli.main(
        [
            str(source),
            "--event-file",
            str(event),
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert status == 1
    manifest = json.loads((tmp_path / "source-xmm-region-manifest.json").read_text())
    product = manifest["products"][0]
    assert product["status"] == "failed"
    assert product["error_type"] == "ValueError"
    assert product["error_message"] == "synthetic geometry failure"
    assert product["failure_evidence"] is None

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from xmm_region_tool import check_cli, sas_validate_cli, science_batch_cli, science_cli, validate_cli
from xmm_region_tool.execution import SasProjectionContextError
from xmm_region_tool.sas_validation import SasRegionValidationError


def _install_main_context_failure(monkeypatch, tmp_path):
    implementation = science_cli._implementation_cli
    source = tmp_path / "source.reg"
    source.write_text('icrs\ncircle(10,-9,30")\n')
    event = tmp_path / "events.fits"
    event.write_bytes(b"synthetic event bytes")

    identity = SimpleNamespace(obs_id="001", instrument="mos1", exposure_id="S001")
    context = SimpleNamespace(
        producer=SimpleNamespace(),
        identity_sha256="b" * 64,
        calibration=SimpleNamespace(identity_sha256="c" * 64, cif_path=tmp_path / "ccf.cif"),
    )
    selection = SimpleNamespace(
        geometry_sha256="a" * 64,
        canonical_geometry_record=lambda: {"schema": "synthetic/v1", "regions": []},
    )

    monkeypatch.setattr(
        implementation.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(science_cli, "validate_durable_sas_producer", lambda value: None)
    monkeypatch.setattr(implementation, "_resolve_events", lambda args: (event,))
    monkeypatch.setattr(implementation, "_validate_event_set", lambda paths: ((event, identity),))
    monkeypatch.setattr(
        implementation,
        "load_ds9_selection_snapshot",
        lambda path, samples: (selection, implementation.file_sha256(path)),
    )
    monkeypatch.setattr(
        implementation,
        "extraction_cells",
        lambda value: (
            implementation.ExtractionCell(index=1, label="r001", selection=selection),
        ),
    )

    def stale_context(*args, **kwargs):
        raise SasProjectionContextError("active SAS_CCF bytes changed")

    monkeypatch.setattr(implementation, "project_selection", stale_context)
    return source, event


def test_main_release_cli_stale_context_aborts_invocation_without_manifest(
    monkeypatch,
    tmp_path,
    capsys,
):
    source, event = _install_main_context_failure(monkeypatch, tmp_path)

    status = science_cli.main(
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
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "active SAS_CCF bytes changed" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "source-xmm-region-manifest.json").exists()


def test_release_batch_context_capture_failure_is_concise_stderr(monkeypatch, capsys):
    def fail_context():
        raise SasProjectionContextError("captured esky2det producer is stale")

    monkeypatch.setattr(
        science_batch_cli._implementation_batch_cli.SasProjectionContext,
        "from_environment",
        fail_context,
    )

    status = science_batch_cli.main(
        [
            "source.reg",
            "--event-file",
            "events.fits",
            "--output-dir",
            "out",
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "xmm-region-batch: error:" in captured.err
    assert "captured esky2det producer is stale" in captured.err
    assert "Traceback" not in captured.err


def test_release_batch_failed_items_use_stderr_but_manifest_stays_stdout(monkeypatch, capsys):
    context = SimpleNamespace(producer=SimpleNamespace())
    monkeypatch.setattr(
        science_batch_cli._implementation_batch_cli.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(science_batch_cli, "validate_durable_sas_producer", lambda value: None)
    result = SimpleNamespace(
        items=(
            SimpleNamespace(
                status="success",
                event_file=Path("ok.fits"),
                label="ok-label",
                refinement_diagnostics={},
            ),
            SimpleNamespace(
                status="failed",
                event_file=Path("bad.fits"),
                error_type="SasConversionError",
                error_message="projection failed",
            ),
        ),
        manifest=Path("out/xmm-region-manifest.json"),
        successful=False,
    )
    monkeypatch.setattr(
        science_batch_cli._implementation_batch_cli,
        "convert_batch",
        lambda *args, **kwargs: result,
    )

    status = science_batch_cli.main(
        [
            "source.reg",
            "--event-file",
            "events.fits",
            "--output-dir",
            "out",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "OK ok.fits -> ok-label" in captured.out
    assert "out/xmm-region-manifest.json" in captured.out
    assert "FAILED" not in captured.out
    assert "FAILED bad.fits: SasConversionError: projection failed" in captured.err


def test_checker_missing_region_is_operational_error_on_stderr(tmp_path, capsys):
    status = check_cli.main(
        [
            str(tmp_path / "missing-region.fits"),
            "--event-file",
            str(tmp_path / "missing-event.fits"),
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "validation could not be completed" in captured.err
    assert "Traceback" not in captured.err


def test_real_sas_validator_expected_failure_has_no_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        sas_validate_cli,
        "validate_fits_region_with_evselect",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SasRegionValidationError("evselect is unavailable")
        ),
    )

    status = sas_validate_cli.main(["region.fits", "--event-file", "events.fits"])

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evselect is unavailable" in captured.err
    assert "Traceback" not in captured.err


def test_real_sas_validator_completed_mismatch_remains_status_one(monkeypatch, capsys):
    report = SimpleNamespace(compatible=False, format_text=lambda: "compatible: NO")
    monkeypatch.setattr(
        sas_validate_cli,
        "validate_fits_region_with_evselect",
        lambda *args, **kwargs: report,
    )

    status = sas_validate_cli.main(["region.fits", "--event-file", "events.fits"])

    assert status == 1
    captured = capsys.readouterr()
    assert "compatible: NO" in captured.out
    assert captured.err == ""


def test_legacy_validator_expected_input_failure_has_no_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        validate_cli,
        "read_ds9_sky_regions",
        lambda path: (_ for _ in ()).throw(ValueError("unsupported DS9 region")),
    )

    status = validate_cli.main(["source.reg", "--event-file", "events.fits"])

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported DS9 region" in captured.err
    assert "Traceback" not in captured.err


def test_legacy_validator_completed_disagreement_remains_status_one(monkeypatch, capsys):
    monkeypatch.setattr(validate_cli, "read_ds9_sky_regions", lambda path: ("source",))
    monkeypatch.setattr(
        validate_cli,
        "load_ds9_sky_regions",
        lambda path, samples: ("sampled",),
    )
    monkeypatch.setattr(
        validate_cli,
        "convert_with_esky2det",
        lambda regions, calinfoset: ("detector",),
    )
    report = SimpleNamespace(disagreements=1, format_text=lambda: "disagreements: 1")
    monkeypatch.setattr(
        validate_cli,
        "compare_event_membership",
        lambda *args, **kwargs: report,
    )

    status = validate_cli.main(["source.reg", "--event-file", "events.fits"])

    assert status == 1
    captured = capsys.readouterr()
    assert "disagreements: 1" in captured.out
    assert "allowed disagreements: 0" in captured.out
    assert captured.err == ""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from xmm_region_tool import sas_validate_cli, science_batch_cli, science_cli
from xmm_region_tool.sas_validation import SasRegionValidationError
from xmm_region_tool.terminal import SafeTerminalWriter, safe_terminal_text


def _assert_no_raw_controls(text: str) -> None:
    assert "\x1b" not in text
    assert "\r" not in text
    assert "\t" not in text
    assert "\a" not in text
    assert "\x85" not in text


def test_safe_terminal_text_escapes_controls_and_preserves_ordinary_text():
    ordinary = "/data/A133/mos1S001-allevc.fits: ordinary failure"
    assert safe_terminal_text(ordinary) == ordinary

    rendered = safe_terminal_text("bad\x1b[31m\nFAKE\rLINE\tEND\x85")
    assert rendered == "bad\\x1b[31m\\nFAKE\\rLINE\\tEND\\x85"
    _assert_no_raw_controls(rendered)
    assert "\n" not in rendered


def test_safe_terminal_writer_preserves_only_structural_print_newline():
    target = io.StringIO()
    writer = SafeTerminalWriter(target)

    print("real\nFAKE\x1b[2J\rline", file=writer)

    rendered = target.getvalue()
    assert rendered == "real\\nFAKE\\x1b[2J\\rline\n"
    _assert_no_raw_controls(rendered)
    assert rendered.count("\n") == 1


def test_release_batch_cli_sanitizes_paths_and_external_failure_text(monkeypatch, capsys):
    producer = SimpleNamespace()
    context = SimpleNamespace(producer=producer)
    monkeypatch.setattr(
        science_batch_cli._implementation_batch_cli.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(science_batch_cli, "validate_durable_sas_producer", lambda value: None)

    success = SimpleNamespace(
        status="success",
        event_file=Path("event\x1b[31m\nFAKE.fits"),
        label="label\rspoof",
        refinement_diagnostics={},
    )
    failure = SimpleNamespace(
        status="failed",
        event_file=Path("failed\x1b]0;title\a.fits"),
        error_type="SasConversionError",
        error_message="bad SAS\nFAKE STATUS\r\x1b[2J",
    )
    result = SimpleNamespace(
        items=(success, failure),
        manifest=Path("manifest\x1b[5m.json"),
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
            "event.fits",
            "--output-dir",
            "out",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    _assert_no_raw_controls(captured.out)
    _assert_no_raw_controls(captured.err)
    assert "event\\x1b[31m\\nFAKE.fits" in captured.out
    assert "label\\rspoof" in captured.out
    assert "manifest\\x1b[5m.json" in captured.out
    assert "bad SAS\\nFAKE STATUS\\r\\x1b[2J" in captured.err
    assert "failed\\x1b]0;title\\x07.fits" in captured.err
    assert captured.out.count("\n") == 2
    assert captured.err.count("\n") == 1


def test_release_cli_sanitizes_delegated_progress_and_error_streams(monkeypatch, capsys):
    producer = SimpleNamespace()
    context = SimpleNamespace(producer=producer)
    monkeypatch.setattr(
        science_cli._implementation_cli.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(science_cli, "validate_durable_sas_producer", lambda value: None)

    def fake_run(args, *, context):
        print("OK /tmp/event\x1b[31m\nFAKE.fits")
        print("SAS error\rspoof\x1b[2J", file=__import__("sys").stderr)
        return 0

    monkeypatch.setattr(science_cli._implementation_cli, "_run", fake_run)

    status = science_cli.main(["source.reg", "--event-file", "event.fits"])

    assert status == 0
    captured = capsys.readouterr()
    _assert_no_raw_controls(captured.out)
    _assert_no_raw_controls(captured.err)
    assert captured.out == "OK /tmp/event\\x1b[31m\\nFAKE.fits\n"
    assert captured.err == "SAS error\\rspoof\\x1b[2J\n"


def test_real_sas_validation_cli_sanitizes_external_error(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise SasRegionValidationError("evselect bad\nFAKE\r\x1b[2J")

    monkeypatch.setattr(sas_validate_cli, "validate_fits_region_with_evselect", fail)

    status = sas_validate_cli.main(
        ["region.fits", "--event-file", "events.fits"]
    )

    assert status == 2
    error = capsys.readouterr().err
    _assert_no_raw_controls(error)
    assert "evselect bad\\nFAKE\\r\\x1b[2J" in error
    assert error.count("\n") == 1

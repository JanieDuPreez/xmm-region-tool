from __future__ import annotations

from importlib.metadata import distribution
from types import SimpleNamespace

import pytest

from xmm_region_tool import batch_cli, cli, science_batch_cli, science_cli


def test_release_cli_rejects_legacy_samples_but_internal_parser_retains_it():
    parser = science_cli.build_parser()
    assert "--samples" not in parser.format_help()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["source.reg", "--samples", "32"])
    assert excinfo.value.code == 2

    internal = cli.build_parser().parse_args(["source.reg", "--samples", "32"])
    assert internal.samples == 32
    release = science_cli.build_parser().parse_args(["source.reg"])
    assert release.samples is None


def test_release_batch_rejects_legacy_samples_but_internal_parser_retains_it():
    args = ["source.reg", "--event-file", "event.fits", "--output-dir", "out"]
    parser = science_batch_cli.build_parser()
    assert "--samples" not in parser.format_help()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([*args, "--samples", "32"])
    assert excinfo.value.code == 2

    internal = batch_cli.build_parser().parse_args([*args, "--samples", "32"])
    assert internal.samples == 32
    release = science_batch_cli.build_parser().parse_args(args)
    assert release.samples is None


def test_release_cli_invocation_supplies_adaptive_samples_none(monkeypatch):
    context = SimpleNamespace(producer=object())
    captured = {}

    monkeypatch.setattr(
        science_cli._implementation_cli.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(science_cli, "validate_durable_sas_producer", lambda producer: None)

    def run(args, *, context=None):
        captured["samples"] = args.samples
        captured["context"] = context
        return 0

    monkeypatch.setattr(science_cli._implementation_cli, "_run", run)

    assert science_cli.main(["source.reg", "--event-file", "event.fits"]) == 0
    assert captured == {"samples": None, "context": context}


def test_release_cli_without_selector_uses_all_unambiguous_cameras(monkeypatch, tmp_path):
    mos1 = (tmp_path / "mos1S001-allevc.fits").resolve()
    pn = (tmp_path / "pnS003-allevc.fits").resolve()
    captured = {}

    def discover(search_dir):
        captured["search_dir"] = search_dir
        return (mos1, pn)

    monkeypatch.setattr(cli, "discover_all_event_files", discover)

    args = science_cli.build_parser().parse_args(
        ["source.reg", "--search-dir", str(tmp_path)]
    )
    assert cli._resolve_events(args) == (mos1, pn)
    assert captured["search_dir"] == tmp_path.resolve()
    assert args.instrument is None
    assert "all unambiguous current-ESAS" in science_cli.build_parser().format_help()


def test_release_batch_invocation_supplies_adaptive_samples_none(monkeypatch):
    context = SimpleNamespace(producer=object())
    captured = {}
    result = SimpleNamespace(items=(), manifest="manifest.json", successful=True)

    monkeypatch.setattr(
        science_batch_cli._implementation_batch_cli.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(science_batch_cli, "validate_durable_sas_producer", lambda producer: None)

    def convert_batch(region, event_files, output_dir, **kwargs):
        captured["region"] = region
        captured["event_files"] = event_files
        captured["output_dir"] = output_dir
        captured["samples"] = kwargs["samples"]
        captured["context"] = context
        return result

    monkeypatch.setattr(science_batch_cli._implementation_batch_cli, "convert_batch", convert_batch)

    argv = ["source.reg", "--event-file", "event.fits", "--output-dir", "out"]
    assert science_batch_cli.main(argv) == 0
    assert captured["samples"] is None
    assert captured["context"] is context


def test_legacy_xy_validator_is_not_an_installed_console_script():
    scripts = {
        entry.name: entry.value
        for entry in distribution("xmm-region-tool").entry_points
        if entry.group == "console_scripts"
    }
    assert "xmm-region-validate" not in scripts
    assert scripts["xmm-region-sas-validate"] == "xmm_region_tool.sas_validate_cli:main"

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xmm_region_tool.calibration import SasProducerIdentity


def _complete_context():
    producer = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256="a" * 64,
        sas_version="xmmsas_20250304_1200",
    )
    return SimpleNamespace(producer=producer)


def _incomplete_context():
    producer = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256=None,
        sas_version="xmmsas_20250304_1200",
    )
    return SimpleNamespace(producer=producer)


def test_release_cli_validates_and_runs_with_one_exact_context(monkeypatch, tmp_path) -> None:
    from xmm_region_tool import science_cli

    contexts = [_complete_context(), _incomplete_context()]
    captures = 0
    received = []

    def capture_once():
        nonlocal captures
        value = contexts[captures]
        captures += 1
        return value

    def runner(args, *, context=None):
        received.append(context)
        return 0

    monkeypatch.setattr(
        science_cli._implementation_cli.SasProjectionContext,
        "from_environment",
        capture_once,
    )
    monkeypatch.setattr(science_cli._implementation_cli, "_run", runner)

    assert science_cli.main([str(tmp_path / "source.reg")]) == 0
    assert captures == 1
    assert received == [contexts[0]]


def test_implementation_runner_does_not_recapture_injected_context(monkeypatch, tmp_path) -> None:
    from xmm_region_tool import cli

    context = _complete_context()
    args = cli.build_parser().parse_args([str(tmp_path / "source.reg")])

    monkeypatch.setattr(cli, "_resolve_events", lambda args: ())
    monkeypatch.setattr(cli, "_validate_event_set", lambda paths: ())
    monkeypatch.setattr(
        cli.SasProjectionContext,
        "from_environment",
        lambda: pytest.fail("injected context must not be recaptured"),
    )
    monkeypatch.setattr(
        cli,
        "load_ds9_selection_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("past context selection")),
    )

    with pytest.raises(RuntimeError, match="past context selection"):
        cli._run(args, context=context)

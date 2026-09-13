from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from xmm_region_tool import cli, geometry
from xmm_region_tool.batch import convert_batch
from xmm_region_tool.geometry import UnsupportedRegionError, load_ds9_selection_snapshot
from xmm_region_tool.provenance import file_sha256


def _replacement_hash(monkeypatch, source: Path, replacement: Path) -> None:
    real_hash = file_sha256
    calls = 0

    def hash_then_replace(path):
        nonlocal calls
        calls += 1
        digest = real_hash(path)
        if calls == 1:
            replacement.replace(source)
        return digest

    monkeypatch.setattr(geometry, "file_sha256", hash_then_replace)


def test_stable_ds9_snapshot_returns_hash_of_parsed_generation(tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("icrs\ncircle(10,-9,1')\n")

    selection, source_sha = load_ds9_selection_snapshot(source)

    assert selection.regions
    assert source_sha == file_sha256(source)


def test_ds9_snapshot_rejects_persistent_replacement_during_parse(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("icrs\ncircle(10,-9,1')\n")
    replacement = tmp_path / "replacement.reg"
    replacement.write_text("icrs\ncircle(10,-9,2')\n")
    sha_a = file_sha256(source)
    sha_b = file_sha256(replacement)
    assert sha_a != sha_b
    _replacement_hash(monkeypatch, source, replacement)

    with pytest.raises(UnsupportedRegionError, match="changed while it was being parsed"):
        load_ds9_selection_snapshot(source)

    assert file_sha256(source) == sha_b


def test_cli_fails_before_projection_when_ds9_source_changes(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("icrs\ncircle(10,-9,1')\n")
    replacement = tmp_path / "replacement.reg"
    replacement.write_text("icrs\ncircle(10,-9,2')\n")
    event = tmp_path / "events.fits"
    event.write_bytes(b"synthetic event bytes")
    _replacement_hash(monkeypatch, source, replacement)

    identity = SimpleNamespace(obs_id="001", instrument="mos1", exposure_id="S001")
    monkeypatch.setattr(cli, "_validate_event_set", lambda paths: ((event.resolve(), identity),))
    monkeypatch.setattr(
        cli.SasProjectionContext,
        "from_environment",
        lambda: SimpleNamespace(
            identity_sha256="a" * 64,
            calibration=SimpleNamespace(identity_sha256="b" * 64),
            producer=SimpleNamespace(identity_sha256="c" * 64),
        ),
    )

    def must_not_project(*args, **kwargs):
        pytest.fail("projection ran after DS9 source generation drift")

    monkeypatch.setattr(cli, "project_selection", must_not_project)

    output_dir = tmp_path / "out"
    status = cli.main(
        [
            str(source),
            "--event-file",
            str(event),
            "--output-dir",
            str(output_dir),
            "--quiet",
        ]
    )

    assert status == 1
    assert not output_dir.exists()


def test_ds9_batch_adapter_fails_before_batch_work_when_source_changes(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("icrs\ncircle(10,-9,1')\n")
    replacement = tmp_path / "replacement.reg"
    replacement.write_text("icrs\ncircle(10,-9,2')\n")
    event = tmp_path / "event.fits"
    event.write_bytes(b"unused because source snapshot must fail first")
    _replacement_hash(monkeypatch, source, replacement)

    with pytest.raises(UnsupportedRegionError, match="changed while it was being parsed"):
        convert_batch(source, [event], tmp_path / "out")

    assert not (tmp_path / "out").exists()

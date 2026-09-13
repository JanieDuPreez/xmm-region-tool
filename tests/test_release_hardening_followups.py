from __future__ import annotations

from pathlib import Path

import pytest

from xmm_region_tool import sas_validate_cli
from xmm_region_tool.path_safety import PathSafetyError, validate_output_namespace
from xmm_region_tool.sas_task_producer import SasTaskProducerIdentity
from xmm_region_tool.sas_validation import SasRegionValidationReport


def _assert_no_raw_controls(text: str) -> None:
    for character in text:
        codepoint = ord(character)
        if character == "\n":
            continue
        assert not (codepoint < 0x20 or 0x7F <= codepoint <= 0x9F), repr(text)


def test_successful_real_sas_report_sanitizes_dynamic_fields(monkeypatch, capsys):
    producer = SasTaskProducerIdentity(
        task_name="evselect",
        task_version="evselect-3.71.3\x1b[31m\nFAKE\rVERSION",
        executable_sha256="a" * 64,
        sas_version="22.1.0\x85\nFAKE-SAS",
    )
    report = SasRegionValidationReport(
        expected_selected=3,
        sas_selected=3,
        compared_columns=("TIME", "DETX", "DETY"),
        row_sequence_match=True,
        command=("evselect",),
        evselect_executable=Path("/tmp/evselect\x1b[2J\nFAKE-PATH"),
        evselect_producer=producer,
    )
    original_record = producer.canonical_record()
    original_path = report.evselect_executable
    monkeypatch.setattr(
        sas_validate_cli,
        "validate_fits_region_with_evselect",
        lambda *args, **kwargs: report,
    )

    status = sas_validate_cli.main(["region.fits", "--event-file", "events.fits"])

    assert status == 0
    stdout = capsys.readouterr().out
    _assert_no_raw_controls(stdout)
    assert "evselect\\x1b[2J\\nFAKE-PATH" in stdout
    assert "evselect-3.71.3\\x1b[31m\\nFAKE\\rVERSION" in stdout
    assert "22.1.0\\x85\\nFAKE-SAS" in stdout
    assert stdout.count("\n") == 12
    assert producer.canonical_record() == original_record
    assert report.evselect_executable == original_path


@pytest.mark.parametrize(
    ("first_role", "first_name", "second_role", "second_name"),
    [
        ("region wrapper A", "profile-mos1-S001.txt", "region wrapper B", "profile-mos1-s001.txt"),
        ("detector FITS A", "profile-mos1-S001.fits", "detector FITS B", "profile-mos1-s001.fits"),
        (
            "projection sidecar A",
            "profile-mos1-S001.provenance.json",
            "projection sidecar B",
            "profile-mos1-s001.provenance.json",
        ),
        (
            "workflow manifest A",
            "Profile-xmm-region-manifest.json",
            "workflow manifest B",
            "profile-xmm-region-manifest.json",
        ),
    ],
)
def test_generated_namespace_rejects_casefold_collisions(
    tmp_path,
    first_role,
    first_name,
    second_role,
    second_name,
):
    with pytest.raises(PathSafetyError, match="case-insensitive filesystem semantics"):
        validate_output_namespace(
            (
                (first_role, tmp_path / first_name),
                (second_role, tmp_path / second_name),
            ),
            (),
        )


def test_generated_namespace_rejects_casefold_collision_with_protected_input(tmp_path):
    with pytest.raises(PathSafetyError, match="protected input.*case-insensitive"):
        validate_output_namespace(
            (("detector FITS", tmp_path / "EVENT.fits"),),
            (("science event", tmp_path / "event.fits"),),
        )

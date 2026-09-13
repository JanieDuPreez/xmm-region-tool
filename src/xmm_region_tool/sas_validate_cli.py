"""CLI for validating a generated FITS region through real SAS evselect."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .sas_validation import SasRegionValidationError, validate_fits_region_with_evselect
from .terminal import safe_terminal_lines, safe_terminal_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmm-region-sas-validate",
        description=(
            "Run real SAS evselect on a generated FITS detector region and compare "
            "the selected event rows with the FITS REGION boolean algebra."
        ),
    )
    parser.add_argument("region_fits", type=Path, help="generated FITS REGION file")
    parser.add_argument(
        "--event-file",
        required=True,
        type=Path,
        help="EPIC event file bound to the generated detector region",
    )
    parser.add_argument(
        "--source-region",
        type=Path,
        help="optional original DS9 file whose stored SHA256 must also match",
    )
    return parser


def _terminal_report_lines(report) -> tuple[str, ...]:
    """Build report lines while keeping dynamic values inside trusted line boundaries."""
    return (
        "XMM FITS REGION real-SAS validation",
        f"expected selected rows: {report.expected_selected}",
        f"SAS selected rows: {report.sas_selected}",
        f"row sequence match: {'yes' if report.row_sequence_match else 'NO'}",
        "row identity column: XMMRGROW (temporary unique int32 row id)",
        "compared columns: " + ", ".join(report.compared_columns),
        f"evselect executable: {report.evselect_executable}",
        f"evselect task version: {report.evselect_producer.task_version}",
        f"evselect executable SHA256: {report.evselect_producer.executable_sha256}",
        f"evselect producer identity SHA256: {report.evselect_producer.identity_sha256}",
        "SAS release: " + (report.evselect_producer.sas_version or "unavailable"),
        f"compatible: {'yes' if report.compatible else 'NO'}",
    )


def _terminal_report_text(report) -> str:
    """Render a validation report without trusting dynamic text as line structure."""
    structured_fields = (
        "expected_selected",
        "sas_selected",
        "row_sequence_match",
        "compared_columns",
        "evselect_executable",
        "evselect_producer",
    )
    if all(hasattr(report, name) for name in structured_fields):
        return safe_terminal_lines(_terminal_report_lines(report))
    return safe_terminal_text(report.format_text())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_fits_region_with_evselect(
            args.region_fits,
            args.event_file,
            source_region=args.source_region,
        )
    except (SasRegionValidationError, ValueError, OSError) as exc:
        print(safe_terminal_text(f"xmm-region-sas-validate: error: {exc}"), file=sys.stderr)
        return 2
    print(_terminal_report_text(report))
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())

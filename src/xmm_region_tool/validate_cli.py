"""CLI for independently validating sky-to-detector event membership."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .geometry import load_ds9_sky_regions, read_ds9_sky_regions
from .sas import SasConversionError, convert_with_esky2det
from .terminal import safe_terminal_text
from .validation import compare_event_membership


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmm-region-validate",
        description=(
            "Convert a DS9 sky region with esky2det, then independently compare "
            "event membership against the event-list X/Y celestial WCS."
        ),
    )
    parser.add_argument("region", type=Path, help="input DS9 sky-coordinate region file")
    parser.add_argument(
        "--event-file",
        required=True,
        type=Path,
        help="cleaned EPIC event file for the exact observation/instrument",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="vertices used for each curved detector boundary (default: 128)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="event rows evaluated per chunk (default: 100000)",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=0,
        help="maximum row disagreements allowed for exit status 0 (default: 0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_mismatches < 0:
        raise SystemExit("--max-mismatches must be non-negative")

    try:
        source_regions = read_ds9_sky_regions(args.region)
        sampled_regions = load_ds9_sky_regions(args.region, samples=args.samples)
        detector_regions = convert_with_esky2det(sampled_regions, calinfoset=args.event_file)
        report = compare_event_membership(
            args.event_file,
            source_regions,
            detector_regions,
            chunk_size=args.chunk_size,
        )
    except (SasConversionError, ValueError, OSError) as exc:
        print(safe_terminal_text(f"xmm-region-validate: error: {exc}"), file=sys.stderr)
        return 2

    print(report.format_text())
    print(f"allowed disagreements: {args.max_mismatches}")
    return 0 if report.disagreements <= args.max_mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())

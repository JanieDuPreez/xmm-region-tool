"""CLI for projecting one DS9 selection across multiple EPIC event products."""

from __future__ import annotations

import argparse
from pathlib import Path

from .batch import convert_batch
from .execution import SasProjectionContext
from .terminal import safe_terminal_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmm-region-batch",
        description=(
            "Convert one DS9 sky selection independently for multiple XMM EPIC event files. "
            "The standalone command uses one frozen SAS context and therefore requires all "
            "event files in one invocation to belong to the same ObsID. Managed multi-observation "
            "callers should use the in-memory batch API with a per-event context resolver."
        ),
    )
    parser.add_argument("region", type=Path, help="input DS9 sky-coordinate region file")
    parser.add_argument(
        "--event-file",
        action="append",
        required=True,
        type=Path,
        dest="event_files",
        help="exact cleaned EPIC event file; repeat for MOS1/MOS2/pn from one ObsID",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--detector-tolerance",
        type=float,
        default=1.0,
        help=(
            "maximum accepted projected midpoint-to-detector-chord deviation in native "
            "DETX/DETY units (default: 1.0 DET unit = 0.05 arcsec)"
        ),
    )
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "explicit legacy fixed source-boundary sampling for comparison/debugging; "
            "supplying this disables adaptive refinement"
        ),
    )
    parser.add_argument(
        "--representation",
        choices=("fits", "expression", "auto"),
        default="fits",
    )
    parser.add_argument("--inline-limit", type=int, default=4096)
    parser.add_argument("--max-components", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = SasProjectionContext.from_environment()
    result = convert_batch(
        args.region,
        args.event_files,
        args.output_dir,
        context=context,
        samples=args.samples,
        detector_tolerance=args.detector_tolerance,
        max_depth=args.max_depth,
        max_vertices=args.max_vertices,
        representation=args.representation,
        inline_limit=args.inline_limit,
        max_components=args.max_components,
    )
    for item in result.items:
        if item.status == "success":
            diagnostics = item.refinement_diagnostics or {}
            detail = ""
            if diagnostics.get("mode") is not None:
                detail = (
                    f" [{diagnostics.get('mode')}; "
                    f"vertices={diagnostics.get('total_vertices')}; "
                    f"max_error={diagnostics.get('max_accepted_error')}]"
                )
            print(safe_terminal_text(f"OK {item.event_file} -> {item.label}{detail}"))
        else:
            print(
                safe_terminal_text(
                    f"FAILED {item.event_file}: {item.error_type}: {item.error_message}"
                )
            )
    print(safe_terminal_text(result.manifest))
    return 0 if result.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())

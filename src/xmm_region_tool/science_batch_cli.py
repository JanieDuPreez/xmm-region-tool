"""Release batch CLI exposing only validated FITS-backed scientific output."""
from __future__ import annotations
import argparse
import sys
from . import batch_cli as _implementation_batch_cli
from .execution import SasProjectionContextError
from .producer_policy import validate_durable_sas_producer
from .terminal import safe_terminal_text


def _remove_option(parser: argparse.ArgumentParser, dest: str) -> None:
    for action in tuple(parser._actions):
        if action.dest == dest:
            parser._remove_action(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            for option in action.option_strings:
                parser._option_string_actions.pop(option, None)
            return


def build_parser() -> argparse.ArgumentParser:
    parser = _implementation_batch_cli.build_parser()
    _remove_option(parser, "samples")
    parser.set_defaults(samples=None)
    for action in parser._actions:
        if action.dest == "representation":
            action.choices = ("fits",)
            action.default = "fits"
            action.help = "scientific output representation (initial release: FITS only)"
        elif action.dest == "inline_limit":
            action.help = argparse.SUPPRESS
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = _implementation_batch_cli.SasProjectionContext.from_environment()
        validate_durable_sas_producer(context.producer)
        result = _implementation_batch_cli.convert_batch(
            args.region,
            args.event_files,
            args.output_dir,
            context=context,
            samples=None,
            detector_tolerance=args.detector_tolerance,
            max_depth=args.max_depth,
            max_vertices=args.max_vertices,
            representation="fits",
            max_components=args.max_components,
        )
    except (SasProjectionContextError, ValueError, OSError) as exc:
        print(safe_terminal_text(f"xmm-region-batch: error: {exc}"), file=sys.stderr)
        return 2

    for item in result.items:
        if item.status == "success":
            diagnostics = item.refinement_diagnostics or {}
            detail = ""
            if diagnostics.get("mode") is not None:
                detail = (
                    f" [{diagnostics.get('mode')}; "
                    f"vertices={diagnostics.get('total_vertices')}; "
                    f"max_probe_error={diagnostics.get('max_accepted_probe_error')}]"
                )
            print(safe_terminal_text(f"OK {item.event_file} -> {item.label}{detail}"))
        else:
            print(
                safe_terminal_text(
                    f"FAILED {item.event_file}: {item.error_type}: {item.error_message}"
                ),
                file=sys.stderr,
            )
    print(safe_terminal_text(result.manifest))
    return 0 if result.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Release CLI exposing only validated scientific output."""
from __future__ import annotations

import argparse
import sys

from . import cli as _implementation_cli
from .execution import SasProjectionContextError
from .producer_policy import validate_durable_sas_producer
from .sas import SasConversionError
from .terminal import safe_terminal_streams, safe_terminal_text
from .workflow import WorkflowError


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
    parser = _implementation_cli.build_parser()
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
        context = _implementation_cli.SasProjectionContext.from_environment()
        validate_durable_sas_producer(context.producer)
        with safe_terminal_streams(sys.stdout, sys.stderr):
            return _implementation_cli._run(args, context=context)
    except (SasConversionError, SasProjectionContextError, WorkflowError, ValueError, OSError) as exc:
        print(safe_terminal_text(f"xmm-region: error: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

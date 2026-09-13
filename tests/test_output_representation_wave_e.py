from __future__ import annotations

import inspect

import pytest

import xmm_region_tool
from xmm_region_tool import batch as implementation_batch
from xmm_region_tool import output as implementation_output
from xmm_region_tool.science_batch_cli import build_parser as build_batch_parser
from xmm_region_tool.science_cli import build_parser as build_main_parser


def _action(parser, dest: str):
    return next(action for action in parser._actions if action.dest == dest)


def test_installed_main_cli_exposes_only_fits_representation():
    parser = build_main_parser()
    representation = _action(parser, "representation")

    assert tuple(representation.choices) == ("fits",)
    assert representation.default == "fits"
    assert _action(parser, "inline_limit").help is not None
    with pytest.raises(SystemExit):
        parser.parse_args(["source.reg", "--representation", "expression"])
    with pytest.raises(SystemExit):
        parser.parse_args(["source.reg", "--representation", "auto"])


def test_installed_batch_cli_exposes_only_fits_representation():
    parser = build_batch_parser()
    representation = _action(parser, "representation")

    assert tuple(representation.choices) == ("fits",)
    assert representation.default == "fits"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "source.reg",
                "--event-file",
                "events.fits",
                "--output-dir",
                "out",
                "--representation",
                "expression",
            ]
        )


def test_top_level_batch_api_has_no_representation_or_inline_limit_selector():
    parameters = inspect.signature(xmm_region_tool.convert_selection_batch).parameters

    assert "representation" not in parameters
    assert "inline_limit" not in parameters


def test_expression_and_auto_remain_implementation_module_only():
    implementation_parameters = inspect.signature(
        implementation_batch.convert_selection_batch
    ).parameters

    assert "representation" in implementation_parameters
    assert "inline_limit" in implementation_parameters
    assert callable(implementation_output.selection_expression)
    assert callable(implementation_output.write_esas_regionfile)

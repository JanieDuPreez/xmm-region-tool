from __future__ import annotations

import pytest

from xmm_region_tool.geometry import UnsupportedRegionError, load_ds9_selection


def _load(tmp_path, text: str):
    path = tmp_path / "source.reg"
    path.write_text(text)
    return load_ds9_selection(path)


def test_valid_circle_with_metadata_remains_supported(tmp_path):
    selection = _load(
        tmp_path,
        'fk5\ncircle(10,-20,30") # include=0 color=red\n',
    )
    assert len(selection.regions) == 1
    assert selection.regions[0].include is False


@pytest.mark.parametrize(
    "source",
    [
        'fk5\ncircle(10,-20,30",60")\n',
        'fk5\ncircle(10,-20,30",60",90")\n',
        'fk5\ncircle(10,-20,30",90",n=3)\n',
    ],
)
def test_surplus_circle_geometry_parameters_fail_closed(tmp_path, source):
    with pytest.raises(UnsupportedRegionError, match="circle requires exactly"):
        _load(tmp_path, source)


def test_composite_declaration_fails_before_grouping_is_lost(tmp_path):
    source = (
        '# Region file format: DS9 version 4.1\n'
        'fk5\n'
        '# composite(10,-20,0) || composite=1\n'
        'circle(10,-20,30") ||\n'
        'circle(10.1,-20,30")\n'
    )
    with pytest.raises(UnsupportedRegionError, match="composite grouping"):
        _load(tmp_path, source)


def test_composite_member_conjunction_fails_closed_without_declaration(tmp_path):
    with pytest.raises(UnsupportedRegionError, match="composite grouping"):
        _load(
            tmp_path,
            'fk5\ncircle(10,-20,30") || circle(10.1,-20,30")\n',
        )


def test_visual_pipe_inside_braced_metadata_is_not_composite_syntax(tmp_path):
    selection = _load(
        tmp_path,
        'fk5\ncircle(10,-20,30") # text={A | B}\n',
    )
    assert len(selection.regions) == 1

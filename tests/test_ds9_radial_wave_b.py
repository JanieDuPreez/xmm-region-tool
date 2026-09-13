from __future__ import annotations

import pytest

from xmm_region_tool.geometry import UnsupportedRegionError, load_ds9_selection


def _load(tmp_path, text: str):
    path = tmp_path / "source.reg"
    path.write_text(text)
    return load_ds9_selection(path)


def test_zero_inner_annulus_is_one_central_cell_without_subtractive_boundary(tmp_path):
    selection = _load(tmp_path, 'fk5\nannulus(10,-20,0",60")\n')
    assert len(selection.regions) == 1
    assert len(selection.regions[0].boundaries) == 1
    assert selection.regions[0].boundaries[0].subtract is False


def test_zero_inner_explicit_profile_is_disk_then_annuli(tmp_path):
    selection = _load(tmp_path, 'fk5\nannulus(10,-20,0",30",60",90")\n')
    assert len(selection.regions) == 3
    assert [len(region.boundaries) for region in selection.regions] == [1, 2, 2]


def test_zero_inner_compact_and_explicit_annulus_profiles_are_identical(tmp_path):
    compact = _load(tmp_path, 'fk5\nannulus(10,-20,0",90",n=3)\n')
    explicit = _load(tmp_path, 'fk5\nannulus(10,-20,0",30",60",90")\n')
    assert compact.geometry_sha256 == explicit.geometry_sha256


def test_explicit_circular_annulus_boundaries_are_sorted_like_ds9(tmp_path):
    ordered = _load(tmp_path, 'fk5\nannulus(10,-20,0",30",60",90")\n')
    shuffled = _load(tmp_path, 'fk5\nannulus(10,-20,90",30",0",60")\n')
    assert shuffled.geometry_sha256 == ordered.geometry_sha256


def test_excluded_zero_inner_annulus_preserves_boolean_semantics(tmp_path):
    selection = _load(tmp_path, 'fk5\n-annulus(10,-20,0",60")\n')
    assert selection.regions[0].include is False


def test_duplicate_explicit_circular_boundary_fails_closed(tmp_path):
    with pytest.raises(UnsupportedRegionError, match="duplicate radial boundaries"):
        _load(tmp_path, 'fk5\nannulus(10,-20,0",30",30",60")\n')


def test_negative_explicit_circular_boundary_fails_closed(tmp_path):
    with pytest.raises(UnsupportedRegionError, match="non-negative"):
        _load(tmp_path, 'fk5\nannulus(10,-20,-1",30",60")\n')


def test_explicit_ellipse_boundaries_are_sorted_by_first_axis_like_ds9(tmp_path):
    ordered = _load(
        tmp_path,
        'fk5\nellipse(10,-20,10",5",20",10",30",15",17)\n',
    )
    shuffled = _load(
        tmp_path,
        'fk5\nellipse(10,-20,30",15",10",5",20",10",17)\n',
    )
    assert shuffled.geometry_sha256 == ordered.geometry_sha256


def test_duplicate_explicit_ellipse_boundary_fails_after_sorting(tmp_path):
    with pytest.raises(UnsupportedRegionError, match="increase monotonically"):
        _load(
            tmp_path,
            'fk5\nellipse(10,-20,20",10",10",5",20",10",17)\n',
        )

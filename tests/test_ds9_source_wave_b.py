from __future__ import annotations

import astropy.units as u
import pytest

from xmm_region_tool.geometry import UnsupportedRegionError, load_ds9_selection


def _load(tmp_path, text: str):
    path = tmp_path / "source.reg"
    path.write_text(text)
    return load_ds9_selection(path)


def test_custom_include_metadata_overrides_sign(tmp_path):
    excluded = _load(
        tmp_path,
        'icrs\npanda(10,-9,0,90,1,30",60",1) # include=0\n',
    )
    assert all(region.include is False for region in excluded.regions)

    included = _load(
        tmp_path,
        'icrs\n-panda(10,-9,0,90,1,30",60",1) # include=1\n',
    )
    assert all(region.include is True for region in included.regions)


def test_global_include_state_follows_interleaved_custom_regions(tmp_path):
    selection = _load(
        tmp_path,
        "global include=0\n"
        "icrs\n"
        'panda(10,-9,0,90,1,30",60",1)\n'
        "global include=1\n"
        'epanda(10.1,-9,0,90,1,30",15",60",30",1,17)\n',
    )

    assert [region.include for region in selection.regions] == [False, True]


def test_custom_annulus_include_metadata_is_preserved(tmp_path):
    selection = _load(
        tmp_path,
        'icrs\nannulus(10,-9,30",90",n=2) # include=0\n',
    )
    assert len(selection.regions) == 2
    assert all(region.include is False for region in selection.regions)


def test_ordinary_include_metadata_is_not_lost_in_mixed_file(tmp_path):
    standalone = _load(
        tmp_path,
        'icrs\ncircle(10,-9,30") # include=0\n',
    )
    mixed = _load(
        tmp_path,
        "icrs\n"
        'circle(10,-9,30") # include=0\n'
        'panda(10.2,-9,0,90,1,30",60",1)\n',
    )

    assert standalone.regions[0].include is False
    assert mixed.regions[0].include is False
    assert (
        standalone.regions[0].canonical_record()
        == mixed.regions[0].canonical_record()
    )


def test_visual_metadata_does_not_change_custom_geometry_identity(tmp_path):
    plain = _load(
        tmp_path,
        'icrs\npanda(10,-9,0,90,1,30",60",1)\n',
    )
    styled = _load(
        tmp_path,
        'icrs\npanda(10,-9,0,90,1,30",60",1) '
        '# color=red text={include=0; visual only}\n',
    )

    assert plain.geometry_sha256 == styled.geometry_sha256


@pytest.mark.parametrize(
    "text",
    [
        'icrs\ncircle(10,-9,30") # include=maybe\n',
        'icrs\npanda(10,-9,0,90,1,30",60",1) # include=maybe\n',
        'global include=maybe\nicrs\ncircle(10,-9,30")\n',
    ],
)
def test_malformed_science_include_metadata_fails_closed(tmp_path, text):
    with pytest.raises(UnsupportedRegionError, match="include"):
        _load(tmp_path, text)


def test_ordinary_multi_annulus_expands_in_place_when_custom_shape_is_present(tmp_path):
    selection = _load(
        tmp_path,
        "icrs\n"
        'annulus(10,-9,30",60",90",120") # include=0\n'
        'panda(10.2,-9,0,90,1,30",60",1)\n',
    )

    assert len(selection.regions) == 4
    centres = [
        region.boundaries[0].path.center.ra.to_value(u.deg)
        for region in selection.regions
    ]
    assert centres == pytest.approx([10.0, 10.0, 10.0, 10.2], abs=1e-10)
    assert [region.include for region in selection.regions] == [False, False, False, True]


def test_semicolon_statement_keeps_trailing_metadata_on_custom_region(tmp_path):
    selection = _load(
        tmp_path,
        'icrs; panda(10,-9,0,90,1,30",60",1) # include=0\n',
    )
    assert selection.regions[0].include is False


@pytest.mark.parametrize(
    "spelling",
    [
        'panda(10,-9,0,90,1,30",60",1)',
        'panda(10 -9 0 90 1 30" 60" 1)',
        'panda 10 -9 0 90 1 30" 60" 1',
        'cpanda 10,-9 0,90 1,30" 60",1',
    ],
)
def test_panda_separator_parenthesis_and_alias_spellings_are_equivalent(tmp_path, spelling):
    reference = _load(
        tmp_path,
        'icrs\npanda(10,-9,0,90,1,30",60",1)\n',
    )
    candidate = _load(tmp_path, f"icrs\n{spelling}\n")
    assert candidate.geometry_sha256 == reference.geometry_sha256


@pytest.mark.parametrize(
    "spelling",
    [
        'epanda(10,-9,0,90,1,30",15",60",30",1,17)',
        'epanda 10 -9 0 90 1 30" 15" 60" 30" 1 17',
        'epanda(10 -9,0 90,1,30" 15",60" 30",1 17)',
    ],
)
def test_epanda_separator_and_parenthesis_spellings_are_equivalent(tmp_path, spelling):
    reference = _load(
        tmp_path,
        'icrs\nepanda(10,-9,0,90,1,30",15",60",30",1,17)\n',
    )
    candidate = _load(tmp_path, f"icrs\n{spelling}\n")
    assert candidate.geometry_sha256 == reference.geometry_sha256


def test_annulus_and_ellipse_custom_abbreviations_are_equivalent(tmp_path):
    annulus = _load(tmp_path, 'icrs\nannulus(10,-9,30",90",n=2)\n')
    abbreviated_annulus = _load(tmp_path, 'icrs\nann 10 -9 30" 90" n=2\n')
    assert abbreviated_annulus.geometry_sha256 == annulus.geometry_sha256

    ellipse = _load(
        tmp_path,
        'icrs\nellipse(10,-9,30",15",90",45",n=2,17)\n',
    )
    abbreviated_ellipse = _load(
        tmp_path,
        'icrs\nell 10 -9 30" 15" 90" 45" n=2 17\n',
    )
    assert abbreviated_ellipse.geometry_sha256 == ellipse.geometry_sha256


def test_custom_parser_preserves_sexagesimal_centres(tmp_path):
    decimal = _load(
        tmp_path,
        'fk5\npanda(10,-9,0,90,1,30",60",1)\n',
    )
    sexagesimal = _load(
        tmp_path,
        'fk5\npanda 00:40:00 -09:00:00 0 90 1 30" 60" 1\n',
    )
    decimal_center = decimal.regions[0].boundaries[0].path.center
    sexagesimal_center = sexagesimal.regions[0].boundaries[0].path.center
    assert decimal_center.separation(sexagesimal_center).to_value(u.arcsec) < 1e-6


@pytest.mark.parametrize("count", ["1.0", "1.000001", "nan", "inf", "1e0"])
def test_custom_division_counts_require_ds9_integer_tokens(tmp_path, count):
    with pytest.raises(UnsupportedRegionError, match="integer"):
        _load(
            tmp_path,
            f'icrs\npanda(10,-9,0,90,{count},30",60",1)\n',
        )


@pytest.mark.parametrize(
    "source",
    [
        'icrs\npanda(10,-9,0deg,90,1,30",60",1)\n',
        'icrs\npanda(10,-9,0,90rad,1,30",60",1)\n',
        'icrs\npanda(10,-9,0,90,1,1r,2r,1)\n',
        'icrs\nepanda(10,-9,0,90,1,1rad,1",2",2",1,0)\n',
    ],
)
def test_non_ds9_custom_unit_spellings_fail_closed(tmp_path, source):
    with pytest.raises(UnsupportedRegionError, match="invalid"):
        _load(tmp_path, source)


def test_ds9_radian_angle_suffix_remains_supported(tmp_path):
    degrees = _load(
        tmp_path,
        'icrs\npanda(10,-9,0,90,1,30",60",1)\n',
    )
    radians = _load(
        tmp_path,
        'icrs\npanda(10,-9,0r,1.5707963267948966r,1,30",60",1)\n',
    )
    assert radians.geometry_sha256 == degrees.geometry_sha256


def test_even_arity_multi_ring_ellipse_is_rejected(tmp_path):
    with pytest.raises(UnsupportedRegionError, match="multi-ring ellipse"):
        _load(
            tmp_path,
            'icrs\nellipse(90,-40,10",5",20",10")\n',
        )


def test_valid_multi_ring_ellipse_with_explicit_angle_remains_supported(tmp_path):
    selection = _load(
        tmp_path,
        'icrs\nellipse 90 -40 10" 5" 20" 10" 17\n',
    )
    assert len(selection.regions) == 1
    assert len(selection.regions[0].boundaries) == 2

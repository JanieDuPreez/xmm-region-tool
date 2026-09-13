from __future__ import annotations

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from regions import RectangleAnnulusSkyRegion

from xmm_region_tool.geometry import load_ds9_selection, selection_from_sky_regions


def _selection(tmp_path, name: str, source: str):
    path = tmp_path / name
    path.write_text(source)
    return load_ds9_selection(path)


def test_ds9_box_annulus_common_ratio_is_accepted(tmp_path):
    selection = _selection(
        tmp_path,
        "box.reg",
        "fk5\nbox(10,-9,60\",30\",180\",90\",30)\n",
    )

    assert len(selection.regions) == 1
    assert len(selection.regions[0].boundaries) == 2
    assert selection.regions[0].boundaries[0].subtract is False
    assert selection.regions[0].boundaries[1].subtract is True


def test_ds9_box_annulus_mismatched_ratio_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="common width/height aspect ratio"):
        _selection(
            tmp_path,
            "bad-ratio.reg",
            "fk5\nbox(10,-9,60\",30\",180\",100\",30)\n",
        )


def test_ds9_box_annulus_explicit_edges_are_sorted_like_ds9(tmp_path):
    ordered = _selection(
        tmp_path,
        "ordered.reg",
        "fk5\nbox(10,-9,60\",30\",180\",90\",30)\n",
    )
    reversed_source = _selection(
        tmp_path,
        "reversed.reg",
        "fk5\nbox(10,-9,180\",90\",60\",30\",30)\n",
    )

    assert reversed_source.geometry_sha256 == ordered.geometry_sha256


def test_ds9_box_annulus_duplicate_first_axis_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="duplicate first-axis"):
        _selection(
            tmp_path,
            "duplicate.reg",
            "fk5\nbox(10,-9,60\",30\",60\",30\",0)\n",
        )


def test_ds9_zero_inner_box_becomes_central_box_then_annulus(tmp_path):
    selection = _selection(
        tmp_path,
        "zero-inner.reg",
        "fk5\nbox(10,-9,0\",0\",60\",30\",120\",60\",20)\n",
    )

    assert len(selection.regions) == 2
    assert len(selection.regions[0].boundaries) == 1
    assert len(selection.regions[1].boundaries) == 2
    assert selection.regions[1].boundaries[1].subtract is True


def test_ds9_box_n_zero_inner_matches_equivalent_explicit_edges(tmp_path):
    compact = _selection(
        tmp_path,
        "compact.reg",
        "fk5\nbox(10,-9,0\",0\",120\",60\",n=2,20)\n",
    )
    explicit = _selection(
        tmp_path,
        "explicit.reg",
        "fk5\nbox(10,-9,0\",0\",60\",30\",120\",60\",20)\n",
    )

    assert compact.geometry_sha256 == explicit.geometry_sha256


def test_ds9_box_n_reversed_endpoints_are_canonicalized(tmp_path):
    forward = _selection(
        tmp_path,
        "forward.reg",
        "fk5\nbox(10,-9,60\",30\",180\",90\",n=2,0)\n",
    )
    reverse = _selection(
        tmp_path,
        "reverse.reg",
        "fk5\nbox(10,-9,180\",90\",60\",30\",n=2,0)\n",
    )

    assert reverse.geometry_sha256 == forward.geometry_sha256


def test_ds9_box_annulus_partial_zero_pair_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="both be positive or both exactly zero"):
        _selection(
            tmp_path,
            "partial-zero.reg",
            "fk5\nbox(10,-9,0\",30\",120\",60\",0)\n",
        )


def test_ds9_box_annulus_preserves_inclusion_metadata(tmp_path):
    selection = _selection(
        tmp_path,
        "include.reg",
        "fk5\nglobal include=0\nbox(10,-9,60\",30\",120\",60\",0) # include=1\n",
    )

    assert len(selection.regions) == 1
    assert selection.regions[0].include is True


def test_programmatic_rectangle_annulus_keeps_general_aspect_ratios():
    source = RectangleAnnulusSkyRegion(
        center=SkyCoord(10 * u.deg, -9 * u.deg, frame="icrs"),
        inner_width=60 * u.arcsec,
        outer_width=180 * u.arcsec,
        inner_height=30 * u.arcsec,
        outer_height=100 * u.arcsec,
        angle=0 * u.deg,
    )

    selection = selection_from_sky_regions([source])

    assert len(selection.regions) == 1
    assert len(selection.regions[0].boundaries) == 2

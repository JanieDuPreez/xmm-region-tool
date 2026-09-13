from __future__ import annotations

import numpy as np
import pytest

from xmm_region_tool.elliptical_sector import EllipticalSectorAnnulusPath
from xmm_region_tool.geometry import load_ds9_selection
from xmm_region_tool.model import SectorAnnulusPath


FLT_EPSILON_DEG = float(np.rad2deg(np.finfo(np.float32).eps))


def _load(tmp_path, name: str, shape: str):
    path = tmp_path / name
    path.write_text(f"icrs\n{shape}\n")
    return load_ds9_selection(path)


def _panda(start: float, stop: float) -> str:
    return f'panda(10,-9,{start:.15g},{stop:.15g},1,60",120",1)'


def _epanda(start: float, stop: float) -> str:
    return f'epanda(10,-9,{start:.15g},{stop:.15g},1,60",30",120",60",1,27)'


@pytest.mark.parametrize("factory", [_panda, _epanda])
def test_ds9_exact_equal_angles_are_full_circle(tmp_path, factory):
    selection = _load(tmp_path, "equal.reg", factory(45.0, 45.0))

    assert not isinstance(
        selection.regions[0].boundaries[0].path,
        (SectorAnnulusPath, EllipticalSectorAnnulusPath),
    )


@pytest.mark.parametrize("factory", [_panda, _epanda])
def test_ds9_sub_epsilon_positive_gap_is_full_circle(tmp_path, factory):
    delta = 0.5 * FLT_EPSILON_DEG
    selection = _load(tmp_path, "below.reg", factory(45.0, 45.0 + delta))

    assert not isinstance(
        selection.regions[0].boundaries[0].path,
        (SectorAnnulusPath, EllipticalSectorAnnulusPath),
    )


@pytest.mark.parametrize("factory,path_type", [(_panda, SectorAnnulusPath), (_epanda, EllipticalSectorAnnulusPath)])
def test_ds9_above_epsilon_positive_gap_remains_sector(tmp_path, factory, path_type):
    delta = 2.0 * FLT_EPSILON_DEG
    selection = _load(tmp_path, "above.reg", factory(45.0, 45.0 + delta))
    model = selection.regions[0].boundaries[0].path

    assert isinstance(model, path_type)
    assert model.sweep.to_value("deg") == pytest.approx(delta, rel=0, abs=1e-11)


@pytest.mark.parametrize("factory", [_panda, _epanda])
def test_ds9_whole_turn_spellings_normalize_before_classification(tmp_path, factory):
    canonical = _load(tmp_path, "canonical.reg", factory(0.0, 360.0))
    two_turns = _load(tmp_path, "two-turns.reg", factory(0.0, 720.0))
    negative = _load(tmp_path, "negative-turn.reg", factory(-360.0, 360.0))

    assert two_turns.geometry_sha256 == canonical.geometry_sha256
    assert negative.geometry_sha256 == canonical.geometry_sha256


@pytest.mark.parametrize("factory", [_panda, _epanda])
def test_ds9_out_of_range_equivalent_sector_spellings_match(tmp_path, factory):
    canonical = _load(tmp_path, "canonical-sector.reg", factory(330.0, 30.0))
    shifted = _load(tmp_path, "shifted-sector.reg", factory(-30.0, 390.0))

    assert shifted.geometry_sha256 == canonical.geometry_sha256


@pytest.mark.parametrize("factory", [_panda, _epanda])
def test_ds9_near_equal_across_zero_is_fuzzy_full_circle(tmp_path, factory):
    delta = 0.5 * FLT_EPSILON_DEG
    left = _load(tmp_path, "left-seam.reg", factory(360.0 - delta, 0.0))
    right = _load(tmp_path, "right-seam.reg", factory(0.0, 360.0 - delta))
    full = _load(tmp_path, "full.reg", factory(0.0, 360.0))

    assert left.geometry_sha256 == full.geometry_sha256
    assert right.geometry_sha256 == full.geometry_sha256


def test_existing_asymmetric_panda_semantics_are_unchanged(tmp_path):
    selection = _load(tmp_path, "ordinary.reg", _panda(30.0, 110.0))
    model = selection.regions[0].boundaries[0].path

    assert isinstance(model, SectorAnnulusPath)
    assert model.start_angle.to_value("deg") == pytest.approx(70.0, abs=1e-9)
    assert model.sweep.to_value("deg") == pytest.approx(80.0, abs=1e-9)

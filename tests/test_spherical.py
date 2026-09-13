from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from xmm_region_tool.model import CelestialSelection, SelectionGeometryError
from xmm_region_tool.geometry import load_ds9_selection
from xmm_region_tool.spherical import validate_geodesic_vertices


@pytest.mark.parametrize("shape", [(), (3, 1), (3, 2)])
def test_rejects_nonscalar_vertex_sequence(shape):
    vertices = SkyCoord(np.zeros(shape) * u.deg, np.zeros(shape) * u.deg)
    with pytest.raises(SelectionGeometryError, match="1-D"):
        validate_geodesic_vertices(vertices)


@pytest.mark.parametrize("ra,dec", [
    ([0, 0, 1], [0, 0, 1]),
    ([0, 1, 0], [0, 1, 0]),
    ([0, 180, 20], [0, 0, 10]),
    ([0, 1, 2], [0, 0, 0]),
    ([0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 0, -1]),
])
def test_rejects_undefined_or_retraced_edges(ra, dec):
    vertices = SkyCoord(ra * u.deg, dec * u.deg)
    with pytest.raises(SelectionGeometryError):
        validate_geodesic_vertices(vertices)
    with pytest.raises(SelectionGeometryError):
        CelestialSelection.polygon(vertices)


@pytest.mark.parametrize("ra,dec", [
    ([359.9, 0.1, 0], [0, 0, 0.1]),
    ([0, 179.999, 30], [0, 0, 20]),
    ([0, 1, 0, 1], [0, 1, 1, 0]),
    ([0, 120, 240], [0, 0, 0]),
    ([0, 1e-8, 0], [0, 0, 1e-8]),
    ([120, 120 + 1e-8, 120], [-30, -30, -30 + 1e-8]),
])
def test_preserves_valid_and_crossing_polygons(ra, dec):
    vertices = SkyCoord(ra * u.deg, dec * u.deg)
    validate_geodesic_vertices(vertices)
    selection = CelestialSelection.polygon(vertices)
    assert selection.geometry_sha256 == CelestialSelection.polygon(vertices).geometry_sha256


def test_rejects_rotated_retraced_great_circle():
    center = SkyCoord(120 * u.deg, -30 * u.deg)
    vertices = center.directional_offset_by(37 * u.deg, np.array([0, 1, 2]) * u.deg)
    with pytest.raises(SelectionGeometryError, match="overlap|retrace"):
        validate_geodesic_vertices(vertices)


@pytest.mark.parametrize("coordinates", [
    "0,0,0,0,1,1",
    "0,0,180,0,20,10",
    "0,0,1,0,2,0",
])
def test_ds9_rejects_invalid_geodesic_edges(tmp_path, coordinates):
    source = tmp_path / "invalid.reg"
    source.write_text(f"icrs\npolygon({coordinates})\n")
    with pytest.raises(ValueError):
        load_ds9_selection(source)


def test_geodesic_work_budget_precedes_pair_loop(monkeypatch):
    center = SkyCoord(10*u.deg, -9*u.deg)
    vertices = center.directional_offset_by(np.linspace(0, 360, 257, endpoint=False)*u.deg, 1*u.arcmin)
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget polygon reached pair-validation vector operations")
    monkeypatch.setattr("xmm_region_tool.spherical.np.cross", forbidden)
    with pytest.raises(SelectionGeometryError, match="pair-check.*budget"):
        validate_geodesic_vertices(vertices)

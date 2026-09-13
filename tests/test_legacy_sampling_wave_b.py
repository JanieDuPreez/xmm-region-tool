from __future__ import annotations

import pytest

from xmm_region_tool.geometry import load_ds9_selection
from xmm_region_tool.sas import _legacy_boundary_vertices


def _selection(tmp_path, name: str, shape: str):
    path = tmp_path / name
    path.write_text(f"icrs\n{shape}\n")
    return load_ds9_selection(path)


@pytest.mark.parametrize(
    "shape",
    [
        'panda(10,-9,30,110,1,60",120",1)',
        'epanda(10,-9,30,110,1,60",30",120",60",1,27)',
    ],
)
def test_nonfull_ds9_sectors_reject_legacy_sampling_deliberately(tmp_path, shape):
    selection = _selection(tmp_path, "sector.reg", shape)
    boundary = selection.regions[0].boundaries[0]

    with pytest.raises(
        ValueError,
        match="legacy fixed-source sampling.*requires adaptive projection",
    ):
        _legacy_boundary_vertices(boundary, 64)


@pytest.mark.parametrize(
    "shape",
    [
        'circle(10,-9,120")',
        'ellipse(10,-9,120",60",27)',
        'box(10,-9,240",120",27)',
        'annulus(10,-9,60",120")',
        'ellipse(10,-9,60",30",120",60",27)',
        'panda(10,-9,0,360,1,60",120",1)',
        'epanda(10,-9,0,360,1,60",30",120",60",1,27)',
    ],
)
def test_established_nonsector_boundaries_remain_legacy_sampleable(tmp_path, shape):
    selection = _selection(tmp_path, "legacy-supported.reg", shape)

    for region in selection.regions:
        for boundary in region.boundaries:
            vertices = _legacy_boundary_vertices(boundary, 64)
            assert len(vertices) >= 3

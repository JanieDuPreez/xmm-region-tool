from __future__ import annotations

import pytest

from xmm_region_tool.geometry import UnsupportedRegionError, load_ds9_selection


def _load(tmp_path, text: str):
    path = tmp_path / "source.reg"
    path.write_text(text)
    return load_ds9_selection(path)


@pytest.mark.parametrize(
    "source",
    [
        'ecliptic\ncircle(10,-20,30")\n',
        'ecliptic\npanda(10,-20,0,90,1,30",60",1)\n',
        'ecliptic\nannulus(10,-20,0",60")\n',
    ],
)
def test_bare_ds9_ecliptic_fails_without_wcs_epoch_context(tmp_path, source):
    with pytest.raises(UnsupportedRegionError, match="active image/WCS epoch"):
        _load(tmp_path, source)


@pytest.mark.parametrize("frame", ["icrs", "fk5", "fk4", "galactic"])
def test_non_epoch_ambiguous_supported_frames_remain_available(tmp_path, frame):
    selection = _load(tmp_path, f'{frame}\ncircle(10,-20,30")\n')
    assert len(selection.regions) == 1

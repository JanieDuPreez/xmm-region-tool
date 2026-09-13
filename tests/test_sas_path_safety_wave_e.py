from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.artifacts import ArtifactMaterializationError, materialize_sas_regionfile
from xmm_region_tool.event_roles import read_science_calinfoset_identity
from xmm_region_tool.model import DetectorBoundary, DetectorRegion
from xmm_region_tool.output import RegionSerializationError, write_esas_regionfile
from xmm_region_tool.provenance import EventIdentityError
from xmm_region_tool.sas_paths import (
    SasPathError,
    validate_sas_safe_resolved_path,
    validate_sas_temporary_root,
)


def _region() -> list[DetectorRegion]:
    vertices = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return [DetectorRegion(True, (DetectorBoundary(vertices),))]


def _science_event(path: Path) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "001"
    primary.header["EXP_ID"] = "001S001"
    primary.header["EXPIDSTR"] = "S001"
    primary.header["DATE-OBS"] = "2000-01-01T00:00:00"
    primary.header["RA_PNT"] = 10.0
    primary.header["DEC_PNT"] = -9.0
    primary.header["PA_PNT"] = 0.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([2.0])),
        ],
        name="EVENTS",
    )
    events.header["TELESCOP"] = "XMM"
    events.header["INSTRUME"] = "EMOS1"
    events.header["OBS_ID"] = "001"
    events.header["EXP_ID"] = "001S001"
    events.header["EXPIDSTR"] = "S001"
    fits.HDUList([primary, events]).writeto(path)
    return path


@pytest.mark.parametrize("unsafe", [" ", "+", ":", "[", "]", ",", "(", ")", "@", "'"])
def test_shared_sas_path_validator_rejects_unvalidated_punctuation(tmp_path, unsafe):
    candidate = tmp_path / f"unsafe{unsafe}name.fits"

    with pytest.raises(SasPathError, match="not safe for unquoted SAS"):
        validate_sas_safe_resolved_path(candidate, role="test dataset")


def test_shared_sas_path_validator_accepts_ordinary_resolved_path(tmp_path):
    candidate = tmp_path / "safe_A-1.2" / "geometry_file.fits"

    assert validate_sas_safe_resolved_path(candidate, role="test dataset") == candidate.resolve()


def test_fits_wrapper_keeps_safe_path_bytes_and_rejects_structural_path(tmp_path):
    safe_wrapper = tmp_path / "safe-region.txt"
    result = write_esas_regionfile(safe_wrapper, _region(), representation="fits")

    assert result.fits_region is not None
    assert safe_wrapper.read_text() == f"&&region({result.fits_region},DETX,DETY)\n"

    unsafe_wrapper = tmp_path / "bad+region.txt"
    with pytest.raises(RegionSerializationError, match="not safe for unquoted SAS"):
        write_esas_regionfile(unsafe_wrapper, _region(), representation="fits")
    assert not unsafe_wrapper.exists()
    assert not unsafe_wrapper.with_suffix(".fits").exists()


def test_managed_wrapper_rejects_unsafe_geometry_blockspec(tmp_path):
    geometry = tmp_path / "geometry+unsafe.fits"
    geometry.write_bytes(b"geometry")
    wrapper = tmp_path / "region.txt"

    with pytest.raises(ArtifactMaterializationError, match="not safe for unquoted SAS"):
        materialize_sas_regionfile(wrapper, geometry)

    assert not wrapper.exists()


def test_science_event_role_rejects_unsafe_sas_dataset_path(tmp_path):
    event = _science_event(tmp_path / "events+unsafe.fits")

    with pytest.raises(EventIdentityError, match="not safe for unquoted SAS"):
        read_science_calinfoset_identity(event)


def test_sas_temporary_root_uses_same_path_policy(monkeypatch):
    monkeypatch.setattr(
        "xmm_region_tool.sas_paths.tempfile.gettempdir",
        lambda: "/tmp/unsafe+root",
    )

    with pytest.raises(SasPathError, match="not safe for unquoted SAS"):
        validate_sas_temporary_root()

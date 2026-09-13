from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool import ArtifactMaterializationError, write_bound_detector_geometry
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_identity import canonical_context_identity_sha256
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import file_sha256, read_event_identity


def _event(path: Path):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "001"
    primary.header["EXP_ID"] = "S001"
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
    fits.HDUList([primary, events]).writeto(path)
    return read_event_identity(path)


def _calibration(tmp_path: Path) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def _producer() -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_version="fake",
        sas_version="22",
        esky2det_sha256="3" * 64,
    )


def _result(event, calibration, producer, *, source_sha):
    selection = DetectorSelection(
        regions=(
            DetectorRegion(
                True,
                (
                    DetectorBoundary(
                        np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
                    ),
                ),
            ),
        ),
        source_geometry_sha256=source_sha,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event.path),
        event_identity_sha256=event.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256=canonical_context_identity_sha256(
            calibration.identity_sha256,
            producer.identity_sha256,
        ),
        calibration_identity_sha256=calibration.identity_sha256,
        producer_identity_sha256=producer.identity_sha256,
        calibration_execution_evidence=calibration.evidence_record(),
        rule=ProjectionRule(samples=128),
        command=(
            "esky2det",
            "datastyle=set",
            "intab=/tmp/xmm-region-tool-test:INPUT",
            "witherrorcol=no",
            "withouttab=no",
            "outunit=det",
            "calinfostyle=set",
            f"calinfoset={event.path}",
            "checkfov=no",
        ),
        relevant_environment={},
    )
    return ProjectionResult(selection=selection, provenance=provenance)


def _write(tmp_path: Path, result, event, calibration, producer):
    return write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_file=event.path,
        calibration_identity=calibration,
        sas_producer=producer,
    )


def test_managed_promotion_rejects_missing_source_geometry_binding_before_output(tmp_path):
    event = _event(tmp_path / "events.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _result(event, calibration, producer, source_sha=None)

    with pytest.raises(ArtifactMaterializationError, match="requires a source celestial"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_managed_promotion_rejects_malformed_source_geometry_digest(tmp_path):
    event = _event(tmp_path / "events.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _result(event, calibration, producer, source_sha="A" * 64)

    with pytest.raises(ArtifactMaterializationError, match="canonical lower-case SHA256"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_managed_promotion_rejects_source_provenance_mismatch(tmp_path):
    event = _event(tmp_path / "events.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _result(event, calibration, producer, source_sha="b" * 64)

    with pytest.raises(ArtifactMaterializationError, match="does not match its recorded provenance"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_managed_promotion_accepts_complete_matching_source_binding(tmp_path):
    event = _event(tmp_path / "events.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _result(event, calibration, producer, source_sha="a" * 64)

    artifact = _write(tmp_path, result, event, calibration, producer)

    assert artifact.path.is_file()
    with fits.open(artifact.path) as hdus:
        assert hdus["REGION"].header["XMRGCEL"] == "a" * 64

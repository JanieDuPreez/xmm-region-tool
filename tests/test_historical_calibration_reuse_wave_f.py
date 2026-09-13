from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

import xmm_region_tool
from xmm_region_tool.artifacts import ArtifactMaterializationError
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_identity import canonical_context_identity_sha256
from xmm_region_tool.event_roles import read_science_calinfoset_identity
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import file_sha256


def _write_event(path):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def _calibration(tmp_path, *, cif_sha="1" * 64, name="a.cif"):
    return CalibrationIdentity(
        cif_path=tmp_path / name,
        cif_file_sha256=cif_sha,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def _producer():
    return SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256="3" * 64,
        sas_version="xmmsas_20250304_1200",
    )


def _result(event, calibration, producer):
    identity = read_science_calinfoset_identity(event)
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
        source_geometry_sha256="a" * 64,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event),
        event_identity_sha256=identity.identity_sha256,
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
            "/sas/esky2det",
            "datastyle=set",
            "intab=/tmp/example:INPUT",
            "witherrorcol=no",
            "withouttab=no",
            "outunit=det",
            "calinfostyle=set",
            f"calinfoset={event}",
            "checkfov=no",
        ),
        relevant_environment={"SAS_CCF": "/cal/a.cif"},
    )
    return ProjectionResult(selection=selection, provenance=provenance)


def _artifact(tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _result(event, calibration, producer)
    artifact = xmm_region_tool.write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )
    return event, calibration, producer, artifact


def _rewrite_geometry_file_digest(sidecar, geometry):
    payload = json.loads(sidecar.read_text())
    payload["geometry_artifact"]["file_sha256"] = file_sha256(geometry)
    sidecar.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    ("header_key", "message"),
    [
        ("XMRGCIF", "exact CIF evidence"),
        ("XMRGCAL", "CALINDEX evidence"),
    ],
)
def test_reload_rejects_fits_sidecar_historical_calibration_mismatch(
    tmp_path,
    header_key,
    message,
):
    _, _, _, artifact = _artifact(tmp_path)

    with fits.open(artifact.path, mode="update") as hdus:
        hdus["REGION"].header[header_key] = "f" * 64
        hdus.flush()
    _rewrite_geometry_file_digest(artifact.projection_evidence, artifact.path)

    with pytest.raises(ArtifactMaterializationError, match=message):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reuse_allows_semantically_equivalent_current_cif_with_different_bytes(tmp_path):
    event, calibration_a, producer, artifact = _artifact(tmp_path)
    calibration_b = _calibration(tmp_path, cif_sha="9" * 64, name="b.cif")

    assert calibration_b.identity_sha256 == calibration_a.identity_sha256
    assert calibration_b.cif_file_sha256 != calibration_a.cif_file_sha256

    wrapper = xmm_region_tool.materialize_bound_sas_regionfile(
        tmp_path / "run" / "region.txt",
        artifact,
        event_file=event,
        calibration_identity=calibration_b,
        sas_producer=producer,
        expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
        expected_projection_identity_sha256=artifact.projection_identity_sha256,
    )

    assert wrapper.path.is_file()

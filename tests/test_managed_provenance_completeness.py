from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool import (
    ArtifactMaterializationError,
    load_bound_detector_geometry,
    write_bound_detector_geometry,
)
from xmm_region_tool.artifacts import write_projection_evidence
from xmm_region_tool.calibration import (
    CalibrationConstituent,
    CalibrationIdentity,
    SasProducerIdentity,
)
from xmm_region_tool.context_identity import canonical_context_identity_sha256
from xmm_region_tool.execution import (
    ProjectionProvenance,
    ProjectionResult,
    ProjectionRule,
)
from xmm_region_tool.model import (
    DetectorBoundary,
    DetectorRegion,
    DetectorSelection,
)
from xmm_region_tool.provenance import EventIdentity, file_sha256


def _event_identity(path: Path) -> EventIdentity:
    needs_event = True
    if path.is_file():
        try:
            with fits.open(path) as hdus:
                needs_event = "EVENTS" not in hdus
        except OSError:
            needs_event = True
    if needs_event:
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
        fits.HDUList([primary, events]).writeto(path, overwrite=True)
    return EventIdentity(
        path=path,
        instrument="mos1",
        instrument_header="EMOS1",
        obs_id="001",
        exposure_id="S001",
        telescope="XMM",
        date_obs="2000-01-01T00:00:00",
        ra_pnt=10.0,
        dec_pnt=-9.0,
        pa_pnt=0.0,
    )


def _calibration(
    tmp_path: Path,
    *,
    with_constituent: bool = False,
) -> CalibrationIdentity:
    constituents = ()
    if with_constituent:
        constituents = (
            CalibrationConstituent(
                name="XMM_TEST.CCF",
                sha256="7" * 64,
                cif_md5="8" * 32,
                size=123,
            ),
        )
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="3" * 64,
        replacements=(),
        ccf_search_path=(),
        constituents=constituents,
    )


def _producer() -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_version="esky2det-test",
        sas_version="SAS-22",
        esky2det_sha256="5" * 64,
    )


def _projection_result(
    event: EventIdentity,
    calibration: CalibrationIdentity,
    producer: SasProducerIdentity,
) -> ProjectionResult:
    vertices = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
        dtype=float,
    )
    selection = DetectorSelection(
        regions=(DetectorRegion(True, (DetectorBoundary(vertices),)),),
        source_geometry_sha256="a" * 64,
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
            "/sas/esky2det",
            "datastyle=set",
            "intab=/tmp/example:INPUT",
            "witherrorcol=no",
            "withouttab=no",
            "outunit=det",
            "calinfostyle=set",
            f"calinfoset={event.path}",
            "checkfov=no",
        ),
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
    )
    return ProjectionResult(selection=selection, provenance=provenance)


def _setup_bound_artifact(
    tmp_path: Path,
    *,
    with_constituent: bool = False,
):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = _event_identity(event_path)
    calibration = _calibration(
        tmp_path,
        with_constituent=with_constituent,
    )
    producer = _producer()
    result = _projection_result(event, calibration, producer)
    artifact = write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_identity=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )
    return event, calibration, producer, result, artifact


def test_managed_loader_rejects_missing_calibration_record(tmp_path):
    *_, artifact = _setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    del payload["calibration"]
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="missing the required calibration record",
    ):
        load_bound_detector_geometry(artifact.path)


def test_managed_loader_rejects_missing_sas_producer_record(tmp_path):
    *_, artifact = _setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    del payload["sas_producer"]
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="missing the required SAS producer record",
    ):
        load_bound_detector_geometry(artifact.path)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("cif_file_sha256", "exact CIF"),
        ("constituent_evidence", "constituent evidence"),
    ],
)
def test_managed_loader_rejects_missing_calibration_execution_evidence(
    tmp_path,
    key,
    message,
):
    *_, artifact = _setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    del payload["calibration"][key]
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(ArtifactMaterializationError, match=message):
        load_bound_detector_geometry(artifact.path)


def test_managed_loader_rejects_incomplete_constituent_audit_entry(tmp_path):
    *_, artifact = _setup_bound_artifact(
        tmp_path,
        with_constituent=True,
    )

    payload = json.loads(artifact.projection_evidence.read_text())
    del payload["calibration"]["constituent_evidence"][0]["size"]
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="constituent evidence is incomplete",
    ):
        load_bound_detector_geometry(artifact.path)


def test_unchanged_complete_managed_artifact_still_reloads(tmp_path):
    *_, artifact = _setup_bound_artifact(
        tmp_path,
        with_constituent=True,
    )

    reloaded = load_bound_detector_geometry(artifact.path)

    assert reloaded.path == artifact.path
    assert reloaded.file_sha256 == artifact.file_sha256
    assert reloaded.geometry_sha256 == artifact.geometry_sha256
    assert reloaded.projection_identity_sha256 == artifact.projection_identity_sha256


def test_low_level_projection_evidence_may_remain_partial_for_debugging(tmp_path):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = _event_identity(event_path)
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)

    evidence = tmp_path / "debug.provenance.json"
    write_projection_evidence(evidence, result)

    payload = json.loads(evidence.read_text())
    assert "calibration" not in payload
    assert "sas_producer" not in payload

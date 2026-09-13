from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.artifacts import (
    ArtifactMaterializationError,
    materialize_sas_regionfile,
    write_detector_geometry,
)
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import EventIdentity, file_sha256


def detector_selection() -> DetectorSelection:
    vertices = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return DetectorSelection(
        regions=(DetectorRegion(True, (DetectorBoundary(vertices),)),),
        source_geometry_sha256="a" * 64,
    )


def projection_result() -> ProjectionResult:
    selection = detector_selection()
    provenance = ProjectionProvenance(
        event_file_sha256="b" * 64,
        event_identity_sha256="c" * 64,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="d" * 64,
        calibration_identity_sha256="e" * 64,
        producer_identity_sha256="f" * 64,
        rule=ProjectionRule(samples=128),
        command=("/sas/esky2det", "intab=/tmp/example:INPUT"),
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
    )
    return ProjectionResult(selection=selection, provenance=provenance)


def event_identity(path: Path, *, obs_id: str = "001") -> EventIdentity:
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
        primary.header["OBS_ID"] = obs_id
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
        obs_id=obs_id,
        exposure_id="S001",
        telescope="XMM",
        date_obs="2000-01-01T00:00:00",
        ra_pnt=10.0,
        dec_pnt=-9.0,
        pa_pnt=0.0,
    )


def calibration(tmp_path: Path, marker: str) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=tmp_path / f"{marker}.cif",
        cif_file_sha256=("1" if marker == "a" else "2") * 64,
        calindex_sha256=("3" if marker == "a" else "4") * 64,
        replacements=(),
        ccf_search_path=(),
    )


def producer(marker: str) -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_version=f"esky2det-{marker}",
        sas_version="SAS-22",
        esky2det_sha256=("5" if marker == "a" else "6") * 64,
    )


def coherent_result(
    event: EventIdentity,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
) -> ProjectionResult:
    selection = detector_selection()
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event.path),
        event_identity_sha256=event.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="d" * 64,
        calibration_identity_sha256=calibration_identity.identity_sha256,
        producer_identity_sha256=sas_producer.identity_sha256,
        calibration_execution_evidence=calibration_identity.evidence_record(),
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


def test_geometry_can_be_relocated_and_wrapper_rematerialized(tmp_path):
    result = projection_result()
    artifact = write_detector_geometry(tmp_path / "cache" / "geometry.fits", result)
    staged = tmp_path / "stage" / "geometry.fits"
    staged.parent.mkdir()
    shutil.copy2(artifact.path, staged)

    wrapper = materialize_sas_regionfile(tmp_path / "run" / "region.txt", staged)

    assert artifact.geometry_sha256 == result.selection.geometry_sha256
    assert artifact.file_sha256
    assert wrapper.geometry_path == staged.resolve()
    assert wrapper.path.read_text() == f"&&region({staged.resolve()},DETX,DETY)\n"


def test_projection_evidence_is_separate_from_geometry_identity(tmp_path):
    result = projection_result()
    artifact = write_detector_geometry(tmp_path / "geometry.fits", result)

    assert artifact.projection_evidence is not None
    evidence = json.loads(artifact.projection_evidence.read_text())
    assert evidence["projection"]["projection_identity_sha256"] == (
        result.provenance.projection_identity_sha256
    )
    assert evidence["geometry_artifact"]["geometry_sha256"] == result.selection.geometry_sha256
    assert evidence["geometry_artifact"]["file_sha256"] == artifact.file_sha256


def test_materialization_rejects_wrong_event_identity(tmp_path):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    correct = event_identity(event_path, obs_id="001")
    wrong = event_identity(event_path, obs_id="002")
    cal = calibration(tmp_path, "a")
    prod = producer("a")
    result = coherent_result(correct, cal, prod)

    with pytest.raises(ArtifactMaterializationError, match="caller event_identity"):
        write_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_identity=wrong,
            calibration_identity=cal,
            sas_producer=prod,
        )


def test_materialization_rejects_event_bytes_changed_after_projection(tmp_path):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    cal = calibration(tmp_path, "a")
    prod = producer("a")
    result = coherent_result(event, cal, prod)
    with fits.open(event_path, mode="update") as hdus:
        hdus[0].header["CHANGED"] = True
        hdus.flush()

    with pytest.raises(ArtifactMaterializationError, match="exact calinfoset"):
        write_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
        )


def test_materialization_rejects_wrong_calibration_or_producer(tmp_path):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    cal = calibration(tmp_path, "a")
    prod = producer("a")
    result = coherent_result(event, cal, prod)

    with pytest.raises(ArtifactMaterializationError, match="calibration_identity does not match"):
        write_detector_geometry(
            tmp_path / "wrong-cal.fits",
            result,
            event_identity=event,
            calibration_identity=calibration(tmp_path, "b"),
            sas_producer=prod,
        )
    with pytest.raises(ArtifactMaterializationError, match="sas_producer does not match"):
        write_detector_geometry(
            tmp_path / "wrong-producer.fits",
            result,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=producer("b"),
        )


def test_bound_artifact_stores_projection_event_sha(tmp_path):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    cal = calibration(tmp_path, "a")
    prod = producer("a")
    result = coherent_result(event, cal, prod)

    artifact = write_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_identity=event,
        calibration_identity=cal,
        sas_producer=prod,
    )

    with fits.open(artifact.path) as hdus:
        assert hdus["REGION"].header["XMRGEVT"] == result.provenance.event_file_sha256


def test_materialization_rejects_semantically_equivalent_different_exact_cif(
    tmp_path,
):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    calibration_a = calibration(tmp_path, "a")
    calibration_b = CalibrationIdentity(
        cif_path=tmp_path / "equivalent-b.cif",
        cif_file_sha256="2" * 64,
        calindex_sha256=calibration_a.calindex_sha256,
        replacements=calibration_a.replacements,
        ccf_search_path=calibration_a.ccf_search_path,
        constituents=calibration_a.constituents,
    )
    prod = producer("a")

    assert calibration_a.identity_sha256 == calibration_b.identity_sha256
    assert calibration_a.cif_file_sha256 != calibration_b.cif_file_sha256

    result_a = coherent_result(event, calibration_a, prod)
    result_b = coherent_result(event, calibration_b, prod)

    assert (
        result_a.provenance.projection_identity_sha256
        == result_b.provenance.projection_identity_sha256
    )

    with pytest.raises(
        ArtifactMaterializationError,
        match="calibration execution evidence does not match",
    ):
        write_detector_geometry(
            tmp_path / "wrong-cif.fits",
            result_a,
            event_identity=event,
            calibration_identity=calibration_b,
            sas_producer=prod,
        )

    artifact = write_detector_geometry(
        tmp_path / "correct-cif.fits",
        result_a,
        event_identity=event,
        calibration_identity=calibration_a,
        sas_producer=prod,
    )

    evidence = json.loads(artifact.projection_evidence.read_text())
    assert evidence["calibration"]["cif_file_sha256"] == calibration_a.cif_file_sha256

    with fits.open(artifact.path) as hdus:
        assert hdus["REGION"].header["XMRGCIF"] == calibration_a.cif_file_sha256


def test_projection_calibration_execution_evidence_is_immutable_snapshot(
    tmp_path,
):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    cal = calibration(tmp_path, "a")
    prod = producer("a")
    selection = detector_selection()

    supplied = cal.evidence_record()
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event.path),
        event_identity_sha256=event.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="d" * 64,
        calibration_identity_sha256=cal.identity_sha256,
        producer_identity_sha256=prod.identity_sha256,
        rule=ProjectionRule(samples=128),
        command=("/sas/esky2det",),
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
        calibration_execution_evidence=supplied,
    )

    supplied["cif_file_sha256"] = "9" * 64

    assert (
        provenance.calibration_execution_evidence_record()["cif_file_sha256"]
        == cal.cif_file_sha256
    )


def test_materialization_rejects_missing_authoritative_calibration_evidence(
    tmp_path,
):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    cal = calibration(tmp_path, "a")
    prod = producer("a")
    selection = detector_selection()

    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event.path),
        event_identity_sha256=event.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="d" * 64,
        calibration_identity_sha256=cal.identity_sha256,
        producer_identity_sha256=prod.identity_sha256,
        rule=ProjectionRule(samples=128),
        command=("/sas/esky2det",),
        relevant_environment={},
    )
    result = ProjectionResult(selection=selection, provenance=provenance)

    with pytest.raises(
        ArtifactMaterializationError,
        match="no authoritative calibration execution evidence",
    ):
        write_detector_geometry(
            tmp_path / "missing-evidence.fits",
            result,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
        )

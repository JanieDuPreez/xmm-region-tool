from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import xmm_region_tool
from xmm_region_tool import (
    ArtifactMaterializationError,
    load_bound_detector_geometry,
    materialize_bound_sas_regionfile,
    write_bound_detector_geometry,
)
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_identity import canonical_context_identity_sha256
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import EventIdentity, file_sha256


def event_identity(path: Path) -> EventIdentity:
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


def calibration(tmp_path: Path, marker: str = "a") -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=tmp_path / f"{marker}.cif",
        cif_file_sha256=("1" if marker == "a" else "2") * 64,
        calindex_sha256=("3" if marker == "a" else "4") * 64,
        replacements=(),
        ccf_search_path=(),
    )


def producer(marker: str = "a") -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_version=f"esky2det-{marker}",
        sas_version="SAS-22",
        esky2det_sha256=("5" if marker == "a" else "6") * 64,
    )


def projection_result(event, cal, prod) -> ProjectionResult:
    vertices = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
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
            cal.identity_sha256,
            prod.identity_sha256,
        ),
        calibration_identity_sha256=cal.identity_sha256,
        producer_identity_sha256=prod.identity_sha256,
        calibration_execution_evidence=cal.evidence_record(),
        rule=ProjectionRule(samples=128),
        command=(
            "/sas/esky2det",
            "datastyle=set",
            "intab=/tmp/xmm-region-tool-test:INPUT",
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


def setup_bound_artifact(tmp_path):
    event_path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(event_path)
    event = event_identity(event_path)
    cal = calibration(tmp_path)
    prod = producer()
    result = projection_result(event, cal, prod)
    artifact = write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_identity=event,
        calibration_identity=cal,
        sas_producer=prod,
    )
    return event, cal, prod, artifact


def test_managed_workflow_writes_wrapper_only_after_binding_validation(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)

    wrapper = materialize_bound_sas_regionfile(
        tmp_path / "run" / "region.txt",
        artifact,
        event_identity=event,
        calibration_identity=cal,
        sas_producer=prod,
        expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
        expected_projection_identity_sha256=artifact.projection_identity_sha256,
    )

    assert wrapper.path.is_file()
    assert wrapper.geometry_path.is_file()
    assert wrapper.geometry_path != artifact.path
    assert wrapper.geometry_path.parent == wrapper.path.parent
    assert file_sha256(wrapper.geometry_path) == artifact.file_sha256
    assert str(wrapper.geometry_path) in wrapper.path.read_text()
    assert artifact.projection_evidence.is_file()


def test_managed_wrapper_rejects_changed_event_bytes_before_output(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "region.txt"
    with fits.open(event.path, mode="update") as hdus:
        hdus[0].header["CHANGED"] = True
        hdus.flush()

    with pytest.raises(ArtifactMaterializationError, match="event file bytes"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )

    assert not output.exists()


def test_managed_wrapper_rejects_wrong_context_before_output(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "region.txt"

    with pytest.raises(ArtifactMaterializationError, match="calibration identity"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=calibration(tmp_path, "b"),
            sas_producer=prod,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )

    assert not output.exists()


def test_managed_wrapper_rejects_changed_geometry_bytes_before_output(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "region.txt"
    with fits.open(artifact.path, mode="update") as hdus:
        hdus["REGION"].header["CHANGED"] = True
        hdus.flush()

    with pytest.raises(ArtifactMaterializationError, match="geometry bytes changed"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )

    assert not output.exists()


def test_raw_helpers_are_not_advertised_on_top_level_managed_surface():
    assert not hasattr(xmm_region_tool, "project_with_esky2det")
    assert not hasattr(xmm_region_tool, "write_detector_geometry")
    assert not hasattr(xmm_region_tool, "materialize_sas_regionfile")


def test_managed_wrapper_rejects_wrong_requested_celestial_cell(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "wrong-cell.txt"

    with pytest.raises(ArtifactMaterializationError, match="requested celestial geometry"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
            expected_celestial_geometry_sha256="b" * 64,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )

    assert not output.exists()


def test_managed_wrapper_rejects_wrong_requested_projection(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "wrong-projection.txt"

    with pytest.raises(ArtifactMaterializationError, match="requested projection identity"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256="b" * 64,
        )

    assert not output.exists()


def test_managed_wrapper_rejects_deleted_projection_sidecar(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "deleted-sidecar.txt"
    artifact.projection_evidence.unlink()

    with pytest.raises(ArtifactMaterializationError, match="sidecar does not exist"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )

    assert not output.exists()


def test_managed_wrapper_rejects_tampered_projection_sidecar(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)
    output = tmp_path / "tampered-sidecar.txt"

    payload = json.loads(artifact.projection_evidence.read_text())
    projection = payload["projection"]
    projection["celestial_geometry_sha256"] = "b" * 64

    canonical_projection = {
        key: projection[key]
        for key in (
            "schema",
            "event_file_sha256",
            "event_identity_sha256",
            "celestial_geometry_sha256",
            "context_identity_sha256",
            "calibration_identity_sha256",
            "producer_identity_sha256",
            "rule",
        )
    }
    encoded = json.dumps(
        canonical_projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    projection["projection_identity_sha256"] = hashlib.sha256(encoded).hexdigest()

    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(ArtifactMaterializationError, match="celestial_geometry_sha256"):
        materialize_bound_sas_regionfile(
            output,
            artifact,
            event_identity=event,
            calibration_identity=cal,
            sas_producer=prod,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )

    assert not output.exists()


def test_persisted_bound_artifact_can_reload_and_stage_safely(tmp_path):
    event, cal, prod, artifact = setup_bound_artifact(tmp_path)

    reloaded = load_bound_detector_geometry(artifact.path)

    assert reloaded.celestial_geometry_sha256 == artifact.celestial_geometry_sha256
    assert reloaded.projection_identity_sha256 == artifact.projection_identity_sha256
    assert reloaded.geometry_sha256 == artifact.geometry_sha256

    wrapper = materialize_bound_sas_regionfile(
        tmp_path / "rehydrated" / "region.txt",
        reloaded,
        event_identity=event,
        calibration_identity=cal,
        sas_producer=prod,
        expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
        expected_projection_identity_sha256=artifact.projection_identity_sha256,
    )

    assert wrapper.path.is_file()


def test_managed_loader_rejects_rule_tampering_with_stale_projection_digest(tmp_path):
    _, _, _, artifact = setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    payload["projection"]["rule"]["samples"] = 256
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="projection identity does not match its canonical record",
    ):
        load_bound_detector_geometry(artifact.path)


def test_managed_loader_rejects_context_tampering_with_stale_projection_digest(tmp_path):
    _, _, _, artifact = setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    payload["projection"]["context_identity_sha256"] = "e" * 64
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="projection identity does not match its canonical record",
    ):
        load_bound_detector_geometry(artifact.path)


def test_managed_loader_rejects_tampered_calibration_record(tmp_path):
    _, _, _, artifact = setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    payload["calibration"]["calindex_sha256"] = "f" * 64
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="calibration identity does not match its canonical record",
    ):
        load_bound_detector_geometry(artifact.path)


def test_managed_loader_rejects_tampered_producer_record(tmp_path):
    _, _, _, artifact = setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    payload["sas_producer"]["esky2det_version"] = "tampered"
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="SAS producer identity does not match its canonical record",
    ):
        load_bound_detector_geometry(artifact.path)


@pytest.mark.parametrize(
    ("location", "schema"),
    [
        (("projection",), "xmm-region-tool.projection/v999"),
        (("projection", "rule"), "xmm-region-tool.projection-rule/v999"),
    ],
)
def test_managed_loader_rejects_unsupported_nested_schema(
    tmp_path,
    location,
    schema,
):
    _, _, _, artifact = setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())
    target = payload
    for key in location:
        target = target[key]
    target["schema"] = schema
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(ArtifactMaterializationError, match="unsupported"):
        load_bound_detector_geometry(artifact.path)


def test_managed_loader_rejects_tampered_calibration_constituent_evidence(tmp_path):
    _, _, _, artifact = setup_bound_artifact(tmp_path)

    payload = json.loads(artifact.projection_evidence.read_text())

    payload["calibration"]["constituents"] = [
        {"name": "XMM_TEST.CCF", "sha256": "7" * 64}
    ]
    payload["calibration"]["constituent_evidence"] = [
        {
            "name": "XMM_TEST.CCF",
            "sha256": "8" * 64,
            "cif_md5": None,
            "size": 123,
        }
    ]
    artifact.projection_evidence.write_text(json.dumps(payload))

    with pytest.raises(
        ArtifactMaterializationError,
        match="constituent evidence does not match",
    ):
        load_bound_detector_geometry(artifact.path)

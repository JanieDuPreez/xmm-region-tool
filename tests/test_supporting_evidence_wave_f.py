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


def _write_event(path, *, marker=None):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    if marker is not None:
        primary.header["TESTTAG"] = marker
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path, overwrite=True)
    return path


def _calibration(tmp_path):
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
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


def _command(event):
    return (
        "/sas/esky2det",
        "datastyle=set",
        "intab=/tmp/xmm-region-test/sky.fits:INPUT",
        "witherrorcol=no",
        "withouttab=no",
        "outunit=det",
        "calinfostyle=set",
        f"calinfoset={event}",
        "checkfov=no",
    )


def _result(event, calibration, producer, *, command=None, commands=None):
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
    primary = _command(event) if command is None else tuple(command)
    history = (primary,) if commands is None else tuple(tuple(item) for item in commands)
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
        command=primary,
        commands=history,
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
        refinement_diagnostics={"mode": "test", "max_accepted_probe_error": 0.5},
    )
    return ProjectionResult(selection=selection, provenance=provenance)


def _artifact(tmp_path, *, with_source_sha=False):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _result(event, calibration, producer)
    source_sha = "9" * 64 if with_source_sha else None
    artifact = xmm_region_tool.write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
        source_region_sha256=source_sha,
    )
    return event, calibration, producer, artifact


def _payload(artifact):
    return json.loads(artifact.projection_evidence.read_text())


def _write_payload(artifact, payload):
    artifact.projection_evidence.write_text(json.dumps(payload))


def _rewrite_geometry_file_digest(artifact):
    payload = _payload(artifact)
    payload["geometry_artifact"]["file_sha256"] = file_sha256(artifact.path)
    _write_payload(artifact, payload)


@pytest.mark.parametrize(
    "field",
    ["refinement_diagnostics", "command", "commands", "relevant_environment"],
)
def test_reload_rejects_projection_supporting_evidence_tamper(tmp_path, field):
    _, _, _, artifact = _artifact(tmp_path)
    payload = _payload(artifact)
    if field == "refinement_diagnostics":
        payload["projection"][field]["max_accepted_probe_error"] = 0.01
    elif field == "relevant_environment":
        payload["projection"][field]["SAS_CCF"] = "/different/ccf.cif"
    else:
        payload["projection"][field] = ["tampered"]
    _write_payload(artifact, payload)

    expected = "primary command" if field in {"command", "commands"} else "integrity-bound record"
    with pytest.raises(ArtifactMaterializationError, match=expected):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_converter_runtime_tamper(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    payload = _payload(artifact)
    payload["projection"]["converter_runtime"]["package"]["version"] = "tampered"
    _write_payload(artifact, payload)

    with pytest.raises(ArtifactMaterializationError, match="converter runtime"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_supporting_digest_tamper(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    payload = _payload(artifact)
    payload["supporting_evidence_sha256"] = "f" * 64
    _write_payload(artifact, payload)

    with pytest.raises(ArtifactMaterializationError, match="recorded SHA256"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_detector_geometry_identity_substitution(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    payload = _payload(artifact)
    payload["projection"]["detector_geometry_sha256"] = "f" * 64
    payload["geometry_artifact"]["geometry_sha256"] = "f" * 64
    _write_payload(artifact, payload)

    with pytest.raises(ArtifactMaterializationError, match="detector geometry identity"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_nonhex_detector_geometry_digest(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    payload = _payload(artifact)
    payload["projection"]["detector_geometry_sha256"] = "z" * 64
    payload["geometry_artifact"]["geometry_sha256"] = "z" * 64
    _write_payload(artifact, payload)

    with pytest.raises(ArtifactMaterializationError):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_fits_producer_header_substitution(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    with fits.open(artifact.path, mode="update") as hdus:
        hdus["REGION"].header["ESKYVER"] = "tampered"
        hdus["REGION"].add_checksum()
        hdus.flush()
    _rewrite_geometry_file_digest(artifact)

    with pytest.raises(ArtifactMaterializationError, match="esky2det producer evidence"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_fits_replacement_count_substitution(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    with fits.open(artifact.path, mode="update") as hdus:
        hdus["REGION"].header["XMRGREP"] = 1
        hdus["REGION"].add_checksum()
        hdus.flush()
    _rewrite_geometry_file_digest(artifact)

    with pytest.raises(ArtifactMaterializationError, match="replacement count"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_optional_source_file_sha_is_integrity_bound_and_asserted_by_default(tmp_path):
    _, _, _, artifact = _artifact(tmp_path, with_source_sha=True)
    payload = _payload(artifact)
    supporting = payload["supporting_evidence"]
    assert supporting["source_region_sha256"] == "9" * 64
    assert supporting["source_region_origin"] == "caller-asserted"

    with fits.open(artifact.path, mode="update") as hdus:
        hdus["REGION"].header["XMRGSHA"] = "8" * 64
        hdus["REGION"].add_checksum()
        hdus.flush()
    _rewrite_geometry_file_digest(artifact)

    with pytest.raises(ArtifactMaterializationError, match="source-region evidence"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_managed_promotion_rejects_wrong_calinfoset_command_before_output(tmp_path):
    event = _write_event(tmp_path / "event.fits")
    other = _write_event(tmp_path / "other.fits", marker="different")
    calibration = _calibration(tmp_path)
    producer = _producer()
    command = list(_command(event))
    command[7] = f"calinfoset={other}"
    result = _result(event, calibration, producer, command=command)

    with pytest.raises(ArtifactMaterializationError, match="calinfoset bytes contradict"):
        xmm_region_tool.write_bound_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_file=event,
            calibration_identity=calibration,
            sas_producer=producer,
        )
    assert not (tmp_path / "geometry.fits").exists()


def test_managed_promotion_rejects_wrong_projection_mode_flag(tmp_path):
    event = _write_event(tmp_path / "event.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    command = list(_command(event))
    command[5] = "outunit=sky"
    result = _result(event, calibration, producer, command=command)

    with pytest.raises(ArtifactMaterializationError, match="outunit"):
        xmm_region_tool.write_bound_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_file=event,
            calibration_identity=calibration,
            sas_producer=producer,
        )


def test_managed_promotion_rejects_primary_history_disagreement(tmp_path):
    event = _write_event(tmp_path / "event.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    primary = _command(event)
    different = list(primary)
    different[2] = "intab=/tmp/other.fits:INPUT"
    result = _result(event, calibration, producer, command=primary, commands=(different,))

    with pytest.raises(ArtifactMaterializationError, match="primary command"):
        xmm_region_tool.write_bound_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_file=event,
            calibration_identity=calibration,
            sas_producer=producer,
        )


def test_reload_rejects_duplicate_json_keys(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    text = artifact.projection_evidence.read_text()
    artifact.projection_evidence.write_text(text.replace(
        '"schema": "xmm-region-tool.projection-evidence/v2",',
        '"schema": "xmm-region-tool.projection-evidence/v2",\n  "schema": "duplicate",',
        1,
    ))

    with pytest.raises(ArtifactMaterializationError, match="unreadable"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_reload_rejects_nonstandard_nan_json(tmp_path):
    _, _, _, artifact = _artifact(tmp_path)
    payload = _payload(artifact)
    text = json.dumps(payload).replace('"max_accepted_probe_error": 0.5', '"max_accepted_probe_error": NaN')
    artifact.projection_evidence.write_text(text)

    with pytest.raises(ArtifactMaterializationError, match="unreadable"):
        xmm_region_tool.load_bound_detector_geometry(artifact.path)


def test_unchanged_supporting_evidence_reloads_and_stages(tmp_path):
    event, calibration, producer, artifact = _artifact(tmp_path, with_source_sha=True)
    payload = _payload(artifact)

    assert payload["supporting_evidence"]["schema"] == (
        "xmm-region-tool.supporting-evidence/v1"
    )
    with fits.open(artifact.path) as hdus:
        assert hdus["REGION"].header["XMRGEVD"] == payload["supporting_evidence_sha256"]

    reloaded = xmm_region_tool.load_bound_detector_geometry(artifact.path)
    wrapper = xmm_region_tool.materialize_bound_sas_regionfile(
        tmp_path / "run" / "region.txt",
        reloaded,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
        expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
        expected_projection_identity_sha256=artifact.projection_identity_sha256,
    )

    assert wrapper.path.is_file()

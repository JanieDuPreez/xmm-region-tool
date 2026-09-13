from __future__ import annotations

import json

import numpy as np

from xmm_region_tool.artifacts import write_projection_evidence
from xmm_region_tool.calibration import (
    CalibrationConstituent,
    CalibrationIdentity,
    SasProducerIdentity,
)
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import file_sha256


def test_projection_sidecar_contains_auditable_calibration_snapshot(tmp_path):
    constituent = CalibrationConstituent(
        name="EMOS1_BORESIGHT_0010.CCF",
        sha256="1" * 64,
        cif_md5="2" * 32,
        size=1234,
    )
    calibration = CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="3" * 64,
        calindex_sha256="4" * 64,
        replacements=(),
        ccf_search_path=(tmp_path / "ccf",),
        constituents=(constituent,),
    )
    producer = SasProducerIdentity(
        esky2det_version="esky2det-1.20 [SAS-22]",
        sas_version="SAS-22",
        esky2det_sha256="5" * 64,
    )
    selection = DetectorSelection(
        regions=(
            DetectorRegion(
                True,
                (
                    DetectorBoundary(
                        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                    ),
                ),
            ),
        ),
        source_geometry_sha256="6" * 64,
    )
    event = tmp_path / "event.fits"
    event.write_bytes(b"synthetic-event")
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event),
        event_identity_sha256="8" * 64,
        celestial_geometry_sha256="6" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="9" * 64,
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
            f"calinfoset={event}",
            "checkfov=no",
        ),
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
    )
    result = ProjectionResult(selection=selection, provenance=provenance)

    path = write_projection_evidence(
        tmp_path / "projection.json",
        result,
        calibration_identity=calibration,
        sas_producer=producer,
    )

    evidence = json.loads(path.read_text())
    assert evidence["schema"] == "xmm-region-tool.projection-evidence/v2"
    assert evidence["calibration"]["identity_sha256"] == calibration.identity_sha256
    assert evidence["calibration"]["cif_file_sha256"] == calibration.cif_file_sha256
    assert evidence["calibration"]["constituents"] == [
        {"name": constituent.name, "sha256": constituent.sha256}
    ]
    assert evidence["calibration"]["constituent_evidence"] == [
        {
            "name": constituent.name,
            "sha256": constituent.sha256,
            "cif_md5": constituent.cif_md5,
            "size": constituent.size,
        }
    ]
    assert evidence["sas_producer"]["identity_sha256"] == producer.identity_sha256

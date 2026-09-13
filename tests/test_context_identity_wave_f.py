from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import xmm_region_tool
from xmm_region_tool.artifacts import ArtifactMaterializationError
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_identity import (
    ContextIdentityError,
    canonical_context_identity_sha256,
    validate_projection_context_identity,
)
from xmm_region_tool.execution import SasProjectionContext as ImplementationContext
from xmm_region_tool.managed import BoundDetectorGeometryArtifact


def calibration(tmp_path) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="a" * 64,
        calindex_sha256="b" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def producer() -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256="c" * 64,
        sas_version="xmmsas_20250304_1200",
    )


def test_canonical_context_identity_matches_execution_contract(tmp_path) -> None:
    cal = calibration(tmp_path)
    prod = producer()
    context = ImplementationContext(
        environment={},
        calibration=cal,
        producer=prod,
        esky2det_path=tmp_path / "esky2det",
    )
    assert canonical_context_identity_sha256(
        cal.identity_sha256,
        prod.identity_sha256,
    ) == context.identity_sha256


def test_context_identity_rejects_malformed_digest() -> None:
    with pytest.raises(ContextIdentityError, match="projection context identity"):
        validate_projection_context_identity(
            "not-a-digest",
            calibration_identity_sha256="a" * 64,
            producer_identity_sha256="b" * 64,
        )


def test_context_identity_rejects_valid_looking_wrong_digest() -> None:
    with pytest.raises(ContextIdentityError, match="canonical calibration/producer meaning"):
        validate_projection_context_identity(
            "0" * 64,
            calibration_identity_sha256="a" * 64,
            producer_identity_sha256="b" * 64,
        )


def test_public_managed_write_rejects_forged_context_before_output(tmp_path) -> None:
    prod = producer()
    cal = calibration(tmp_path)
    result = SimpleNamespace(
        provenance=SimpleNamespace(
            context_identity_sha256="0" * 64,
            calibration_identity_sha256=cal.identity_sha256,
            producer_identity_sha256=prod.identity_sha256,
        )
    )
    output = tmp_path / "geometry.fits"

    with pytest.raises(ArtifactMaterializationError, match="invalid projection context identity"):
        xmm_region_tool.write_bound_detector_geometry(
            output,
            result,
            event_file=tmp_path / "missing-event.fits",
            calibration_identity=cal,
            sas_producer=prod,
        )
    assert not output.exists()


def test_public_managed_reload_rejects_forged_persisted_context(monkeypatch, tmp_path) -> None:
    from xmm_region_tool import public_managed

    prod = producer()
    cal = calibration(tmp_path)
    evidence = tmp_path / "geometry.provenance.json"
    evidence.write_text(
        json.dumps(
            {
                "sas_producer": {
                    "schema": "xmm-region-tool.sas-producer/v3",
                    "esky2det_name": prod.esky2det_name,
                    "esky2det_version": prod.esky2det_version,
                    "esky2det_sha256": prod.esky2det_sha256,
                    "sas_version": prod.sas_version,
                },
                "projection": {"context_identity_sha256": "0" * 64},
            }
        )
    )
    artifact = BoundDetectorGeometryArtifact(
        path=tmp_path / "geometry.fits",
        geometry_sha256="1" * 64,
        file_sha256="2" * 64,
        projection_evidence=evidence,
        event_file_sha256="3" * 64,
        event_identity_sha256="4" * 64,
        calibration_identity_sha256=cal.identity_sha256,
        producer_identity_sha256=prod.identity_sha256,
        celestial_geometry_sha256="7" * 64,
        projection_identity_sha256="8" * 64,
    )
    monkeypatch.setattr(public_managed, "_load_bound_detector_geometry", lambda *a, **k: artifact)

    with pytest.raises(ArtifactMaterializationError, match="invalid projection context identity"):
        public_managed.load_bound_detector_geometry(artifact.path)

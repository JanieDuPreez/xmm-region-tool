from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import xmm_region_tool
from xmm_region_tool.artifacts import ArtifactMaterializationError
from xmm_region_tool.calibration import SasProducerIdentity
from xmm_region_tool.managed import BoundDetectorGeometryArtifact
from xmm_region_tool.producer_policy import ProducerEvidenceError, validate_durable_sas_producer


def complete_producer() -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256="a" * 64,
        sas_version="xmmsas_20250304_1200",
    )


def test_durable_producer_requires_exact_executable_sha() -> None:
    producer = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256=None,
        sas_version="xmmsas_20250304_1200",
    )
    with pytest.raises(ProducerEvidenceError, match="exact esky2det executable SHA256"):
        validate_durable_sas_producer(producer)


def test_durable_producer_requires_separate_sas_release() -> None:
    producer = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256="a" * 64,
        sas_version=None,
    )
    with pytest.raises(ProducerEvidenceError, match="SAS release/build identifier"):
        validate_durable_sas_producer(producer)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("esky2det_name", ""),
        ("esky2det_name", "bin/esky2det"),
        ("esky2det_version", ""),
        ("esky2det_sha256", "A" * 64),
        ("esky2det_sha256", "not-a-digest"),
        ("sas_version", ""),
    ],
)
def test_durable_producer_rejects_noncanonical_material_fields(field, value) -> None:
    producer = SimpleNamespace(
        esky2det_name="esky2det",
        esky2det_version="1.20 [xmmsas_20250304_1200]",
        esky2det_sha256="a" * 64,
        sas_version="xmmsas_20250304_1200",
    )
    setattr(producer, field, value)
    with pytest.raises(ProducerEvidenceError):
        validate_durable_sas_producer(producer)


def test_complete_durable_producer_is_accepted() -> None:
    validate_durable_sas_producer(complete_producer())


def test_public_context_rejects_uncaptured_ambient_path() -> None:
    with pytest.raises(ValueError, match="must include PATH"):
        xmm_region_tool.SasProjectionContext.from_environment(
            {"SAS_CCF": "/tmp/ccf.cif"},
            esky2det="esky2det",
        )


def test_public_managed_writer_rejects_partial_producer_before_output(tmp_path) -> None:
    producer = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [build]",
        esky2det_sha256=None,
        sas_version="22.1",
    )
    output = tmp_path / "geometry.fits"
    with pytest.raises(ArtifactMaterializationError, match="complete SAS producer evidence"):
        xmm_region_tool.write_bound_detector_geometry(
            output,
            object(),
            event_file=tmp_path / "missing-event.fits",
            calibration_identity=object(),
            sas_producer=producer,
        )
    assert not output.exists()


def test_public_batch_rejects_partial_shared_producer_before_conversion(monkeypatch, tmp_path) -> None:
    from xmm_region_tool import public_batch

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("internal batch conversion must not run")

    monkeypatch.setattr(public_batch, "_convert_selection_batch", forbidden)
    partial = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [build]",
        esky2det_sha256=None,
        sas_version="22.1",
    )
    context = SimpleNamespace(producer=partial)

    with pytest.raises(ProducerEvidenceError):
        public_batch.convert_selection_batch(
            object(),
            [tmp_path / "event.fits"],
            tmp_path / "out",
            rule=object(),
            context=context,
        )
    assert not called


def test_public_managed_reload_rejects_partial_persisted_producer(monkeypatch, tmp_path) -> None:
    from xmm_region_tool import public_managed

    evidence = tmp_path / "geometry.provenance.json"
    evidence.write_text(
        json.dumps(
            {
                "sas_producer": {
                    "schema": "xmm-region-tool.sas-producer/v3",
                    "esky2det_name": "esky2det",
                    "esky2det_version": "1.20 [build]",
                    "esky2det_sha256": None,
                    "sas_version": "22.1",
                }
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
        calibration_identity_sha256="5" * 64,
        producer_identity_sha256="6" * 64,
        celestial_geometry_sha256="7" * 64,
        projection_identity_sha256="8" * 64,
    )
    monkeypatch.setattr(public_managed, "_load_bound_detector_geometry", lambda *a, **k: artifact)

    with pytest.raises(ArtifactMaterializationError, match="incomplete SAS producer evidence"):
        public_managed.load_bound_detector_geometry(artifact.path)


def test_release_cli_rejects_partial_producer_before_runner(monkeypatch, tmp_path) -> None:
    from xmm_region_tool import science_cli

    partial = SasProducerIdentity(
        esky2det_name="esky2det",
        esky2det_version="1.20 [build]",
        esky2det_sha256=None,
        sas_version="22.1",
    )
    monkeypatch.setattr(
        science_cli._implementation_cli.SasProjectionContext,
        "from_environment",
        lambda: SimpleNamespace(producer=partial),
    )
    monkeypatch.setattr(
        science_cli._implementation_cli,
        "_run",
        lambda args: pytest.fail("release runner must not execute"),
    )

    assert science_cli.main([str(tmp_path / "source.reg")]) == 1

"""Execution-owned staging for the supported managed SAS-consumption path."""

from __future__ import annotations

import shutil
from pathlib import Path

from .artifacts import ArtifactMaterializationError, SasRegionfileMaterialization
from .calibration import CalibrationIdentity, SasProducerIdentity
from .managed import BoundDetectorGeometryArtifact, _validate_bound_artifact
from .path_safety import PathSafetyError, validate_output_namespace
from .producer_policy import ProducerEvidenceError, validate_durable_sas_producer
from .provenance import EventIdentity, file_sha256
from .publication import (
    discard_staged,
    lexical_absolute_path,
    publish_staged_files,
    temporary_sibling,
)
from .sas_paths import SasPathError, validate_sas_safe_lexical_path


def _event_path(
    *,
    event_file: str | Path | None,
    event_identity: EventIdentity | None,
) -> Path:
    if event_file is not None and event_identity is not None:
        raise TypeError("provide event_file or legacy event_identity, not both")
    if event_file is not None:
        return Path(event_file).expanduser().resolve()
    if event_identity is not None:
        if not isinstance(event_identity, EventIdentity):
            raise TypeError("event_identity must be EventIdentity")
        return Path(event_identity.path).expanduser().resolve()
    raise TypeError("event_file is required")


def _execution_geometry_path(wrapper: str | Path) -> Path:
    final_wrapper = lexical_absolute_path(wrapper)
    suffix = final_wrapper.suffix
    if suffix.lower() == ".fits":
        return final_wrapper.with_name(final_wrapper.stem + "-geometry.fits")
    if suffix:
        return final_wrapper.with_suffix(".fits")
    return final_wrapper.with_name(final_wrapper.name + ".fits")


def _canonical_science_binding(
    event_file: Path,
    artifact: BoundDetectorGeometryArtifact,
) -> EventIdentity:
    # Import lazily to keep the public validation implementation as the single
    # owner of science-role event re-derivation and persisted-evidence checks.
    from .public_managed import _artifact_event_binding

    return _artifact_event_binding(event_file, artifact)


def _validate_persisted_evidence(artifact: BoundDetectorGeometryArtifact) -> None:
    from .public_managed import _validate_persisted_managed_evidence

    _validate_persisted_managed_evidence(artifact)


def materialize_bound_sas_regionfile(
    path: str | Path,
    artifact: BoundDetectorGeometryArtifact,
    *,
    event_file: str | Path | None = None,
    event_identity: EventIdentity | None = None,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    expected_celestial_geometry_sha256: str,
    expected_projection_identity_sha256: str,
) -> SasRegionfileMaterialization:
    """Stage validated detector bytes into the SAS execution namespace.

    The wrapper never points back at the mutable managed/cache artifact. The
    currently validated geometry is copied to a sibling execution path, that
    staged copy is re-hashed against ``artifact.file_sha256``, and the execution
    geometry + wrapper are then atomically published together. Replacing the
    cached artifact after this function returns therefore cannot change the
    geometry bytes referenced by the emitted wrapper.
    """
    try:
        validate_durable_sas_producer(sas_producer)
    except ProducerEvidenceError as exc:
        raise ArtifactMaterializationError(
            f"managed detector geometry requires complete SAS producer evidence: {exc}"
        ) from exc

    _validate_persisted_evidence(artifact)
    current_path = _event_path(event_file=event_file, event_identity=event_identity)
    current_identity = _canonical_science_binding(current_path, artifact)
    _validate_bound_artifact(
        artifact,
        event_identity=current_identity,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
        expected_celestial_geometry_sha256=expected_celestial_geometry_sha256,
        expected_projection_identity_sha256=expected_projection_identity_sha256,
    )

    wrapper = lexical_absolute_path(path)
    execution_geometry = _execution_geometry_path(wrapper)
    try:
        validate_output_namespace(
            (
                ("managed SAS wrapper", wrapper),
                ("managed execution geometry", execution_geometry),
            ),
            (
                ("bound detector geometry", artifact.path),
                ("bound projection sidecar", artifact.projection_evidence),
                ("science event", current_path),
                ("active CIF", calibration_identity.cif_path),
            ),
        )
        safe_execution_geometry = validate_sas_safe_lexical_path(
            execution_geometry,
            role="managed execution FITS REGION geometry",
        )
    except (PathSafetyError, SasPathError) as exc:
        raise ArtifactMaterializationError(
            f"unsafe managed execution namespace: {exc}"
        ) from exc

    stage_geometry = temporary_sibling(execution_geometry)
    stage_wrapper = temporary_sibling(wrapper)
    try:
        shutil.copyfile(artifact.path, stage_geometry)
        staged_sha = file_sha256(stage_geometry)
        if staged_sha != artifact.file_sha256:
            raise ArtifactMaterializationError(
                "managed detector geometry changed while staging for SAS execution"
            )
        stage_wrapper.write_text(
            f"&&region({safe_execution_geometry},DETX,DETY)\n"
        )
        publish_staged_files(
            ((stage_geometry, execution_geometry), (stage_wrapper, wrapper))
        )
    except Exception:
        discard_staged((stage_geometry, stage_wrapper))
        raise

    return SasRegionfileMaterialization(
        path=wrapper,
        geometry_path=execution_geometry,
    )


__all__ = ["materialize_bound_sas_regionfile"]

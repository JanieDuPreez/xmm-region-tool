"""Durable detector-geometry artifacts and execution-time SAS wrappers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from astropy.io import fits

from .calibration import CalibrationIdentity, SasProducerIdentity
from .execution import ProjectionResult
from .execution_evidence import ExecutionEvidenceError, validate_projection_invocations
from .json_policy import StrictJsonError, dumps_strict
from .model import DetectorSelection
from .output import write_fits_region
from .provenance import EventIdentity, EventIdentityError, file_sha256, read_event_identity
from .sas_paths import SasPathError, validate_sas_safe_resolved_path
from .supporting_evidence import (
    SupportingEvidenceError,
    build_supporting_evidence_record,
    supporting_evidence_sha256,
)
from .topology import DetectorTopologyError, validate_detector_regions


class ArtifactMaterializationError(ValueError):
    """Raised when a durable geometry artifact cannot be staged safely for SAS."""


EventIdentityReader = Callable[[str | Path], EventIdentity]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ArtifactMaterializationError(f"{name} must be a canonical lower-case SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ArtifactMaterializationError(
            f"{name} must be a canonical lower-case SHA256 digest"
        ) from exc
    return value


@dataclass(frozen=True)
class DetectorGeometryArtifact:
    """Durable detector geometry, independent of any execution-time wrapper path."""

    path: Path
    geometry_sha256: str
    file_sha256: str
    projection_evidence: Path | None = None


@dataclass(frozen=True)
class SasRegionfileMaterialization:
    """Tiny location-dependent wrapper referring to one staged FITS geometry."""

    path: Path
    geometry_path: Path


def materialize_sas_regionfile(
    path: str | Path,
    geometry_path: str | Path,
) -> SasRegionfileMaterialization:
    """Write a short ``&&region(...)`` wrapper for an existing FITS geometry.

    This operation never reprojects or mutates the durable geometry. The wrapper
    is intentionally location-dependent because SAS receives the staged geometry
    filename at execution time. The geometry path must satisfy the initial
    conservative unquoted SAS/selectlib path grammar.
    """
    output = Path(path).expanduser().resolve()
    try:
        geometry = validate_sas_safe_resolved_path(
            geometry_path,
            role="FITS REGION geometry",
        )
    except SasPathError as exc:
        raise ArtifactMaterializationError(str(exc)) from exc
    if not geometry.is_file():
        raise ArtifactMaterializationError(f"detector geometry does not exist: {geometry}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"&&region({geometry},DETX,DETY)\n")
    return SasRegionfileMaterialization(path=output, geometry_path=geometry)


def _supporting_record_for_geometry(
    result: ProjectionResult,
    geometry: Path,
    *,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    source_region_origin: str | None,
) -> tuple[dict[str, object], str]:
    authoritative_calibration = result.provenance.calibration_execution_evidence_record()
    if authoritative_calibration is None:
        raise ArtifactMaterializationError(
            "ProjectionResult has no authoritative calibration execution evidence"
        )
    producer_record = {
        **sas_producer.canonical_record(),
        "identity_sha256": sas_producer.identity_sha256,
    }
    evidence = result.provenance.evidence_record()

    source_region_sha256: str | None = None
    try:
        with fits.open(geometry, memmap=False) as hdus:
            if "REGION" not in hdus:
                raise ArtifactMaterializationError("detector geometry has no REGION extension")
            raw_source = hdus["REGION"].header.get("XMRGSHA")
            if raw_source is not None:
                source_region_sha256 = str(raw_source).strip()
    except ArtifactMaterializationError:
        raise
    except OSError as exc:
        raise ArtifactMaterializationError(
            "cannot read detector geometry while constructing supporting evidence"
        ) from exc

    effective_origin = None
    if source_region_sha256 is not None:
        effective_origin = source_region_origin or "caller-asserted"

    try:
        record = build_supporting_evidence_record(
            projection_identity_sha256=result.provenance.projection_identity_sha256,
            detector_geometry_sha256=result.selection.geometry_sha256,
            converter_runtime=result.provenance.converter_runtime_record(),
            command=evidence["command"],
            commands=evidence["commands"],
            relevant_environment=result.provenance.relevant_environment,
            refinement_diagnostics=result.provenance.refinement_diagnostics,
            calibration=authoritative_calibration,
            sas_producer=producer_record,
            source_region_sha256=source_region_sha256,
            source_region_origin=effective_origin,
        )
        digest = supporting_evidence_sha256(record)
    except SupportingEvidenceError as exc:
        raise ArtifactMaterializationError(
            f"cannot canonicalize managed supporting evidence: {exc}"
        ) from exc
    return record, digest


def _bind_supporting_digest_to_geometry(geometry: Path, digest: str) -> None:
    try:
        with fits.open(geometry, mode="update", memmap=False) as hdus:
            if "REGION" not in hdus:
                raise ArtifactMaterializationError("detector geometry has no REGION extension")
            region = hdus["REGION"]
            region.header["XMRGEVD"] = digest
            region.add_checksum()
            hdus.flush()
    except ArtifactMaterializationError:
        raise
    except OSError as exc:
        raise ArtifactMaterializationError(
            "cannot bind supporting-evidence digest into detector geometry"
        ) from exc


def write_projection_evidence(
    path: str | Path,
    result: ProjectionResult,
    *,
    geometry_artifact: str | Path | None = None,
    calibration_identity: CalibrationIdentity | None = None,
    sas_producer: SasProducerIdentity | None = None,
    source_region_origin: str | None = None,
) -> Path:
    """Persist available producer evidence separately from immutable geometry bytes.

    Complete FITS-backed evidence receives a distinct non-semantic supporting
    evidence digest cross-bound into the FITS REGION header. Semantic projection
    identity remains unchanged. ``source_region_origin`` distinguishes a digest
    computed by a DS9-facing tool path from caller-asserted programmatic metadata.
    """
    _validate_material_provenance(
        result,
        event_identity=None,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
    )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": "xmm-region-tool.projection-evidence/v2",
        "projection": result.provenance.evidence_record(),
    }
    authoritative_calibration: dict[str, object] | None = None
    if calibration_identity is not None:
        authoritative_calibration = result.provenance.calibration_execution_evidence_record()
        if authoritative_calibration is None:
            raise ArtifactMaterializationError(
                "ProjectionResult has no authoritative calibration execution evidence"
            )
        payload["calibration"] = authoritative_calibration
    if sas_producer is not None:
        payload["sas_producer"] = {
            **sas_producer.canonical_record(),
            "identity_sha256": sas_producer.identity_sha256,
        }

    geometry: Path | None = None
    if geometry_artifact is not None:
        geometry = Path(geometry_artifact).expanduser().resolve()
        if not geometry.is_file():
            raise ArtifactMaterializationError(f"detector geometry does not exist: {geometry}")

    if geometry is not None and calibration_identity is not None and sas_producer is not None:
        supporting_record, supporting_digest = _supporting_record_for_geometry(
            result,
            geometry,
            calibration_identity=calibration_identity,
            sas_producer=sas_producer,
            source_region_origin=source_region_origin,
        )
        _bind_supporting_digest_to_geometry(geometry, supporting_digest)
        payload["supporting_evidence"] = supporting_record
        payload["supporting_evidence_sha256"] = supporting_digest

    if geometry is not None:
        payload["geometry_artifact"] = {
            "path": str(geometry),
            "file_sha256": _file_sha256(geometry),
            "geometry_sha256": result.selection.geometry_sha256,
        }
    try:
        serialized = dumps_strict(payload, indent=2, sort_keys=True)
    except StrictJsonError as exc:
        raise ArtifactMaterializationError(f"cannot serialize durable projection evidence: {exc}") from exc
    output.write_text(serialized + "\n")
    return output


def _validate_result_consistency(result: ProjectionResult) -> None:
    if not isinstance(result.selection, DetectorSelection):
        raise TypeError("result.selection must be DetectorSelection")
    if result.selection.geometry_sha256 != result.provenance.detector_geometry_sha256:
        raise ArtifactMaterializationError(
            "ProjectionResult detector geometry does not match its recorded provenance"
        )

    source_sha = result.selection.source_geometry_sha256
    if source_sha is None:
        raise ArtifactMaterializationError(
            "durable detector geometry requires a source celestial geometry SHA256"
        )
    canonical_source = _canonical_sha256(
        source_sha,
        name="DetectorSelection.source_geometry_sha256",
    )
    canonical_celestial = _canonical_sha256(
        result.provenance.celestial_geometry_sha256,
        name="ProjectionProvenance.celestial_geometry_sha256",
    )
    if canonical_source != canonical_celestial:
        raise ArtifactMaterializationError(
            "ProjectionResult source celestial geometry does not match its recorded provenance"
        )


def _validate_detector_topology(result: ProjectionResult) -> None:
    """Reject malformed geometry even when its supplied digests are self-consistent."""
    try:
        validate_detector_regions(result.selection.regions)
    except DetectorTopologyError as exc:
        raise ArtifactMaterializationError(
            f"detector geometry is scientifically/topologically invalid: {exc}"
        ) from exc


def _validate_material_provenance(
    result: ProjectionResult,
    *,
    event_identity: EventIdentity | None,
    calibration_identity: CalibrationIdentity | None,
    sas_producer: SasProducerIdentity | None,
    event_identity_reader: EventIdentityReader = read_event_identity,
) -> None:
    provenance = result.provenance
    if event_identity is not None:
        try:
            current_identity = event_identity_reader(event_identity.path)
        except (EventIdentityError, OSError) as exc:
            raise ArtifactMaterializationError(
                "cannot re-read the current event artifact before durable materialisation"
            ) from exc
        if current_identity.identity_sha256 != event_identity.identity_sha256:
            raise ArtifactMaterializationError(
                "caller event_identity does not describe the current event artifact"
            )
        if current_identity.identity_sha256 != provenance.event_identity_sha256:
            raise ArtifactMaterializationError(
                "current event identity does not match the event identity recorded by ProjectionResult"
            )
        current_event_sha = file_sha256(event_identity.path)
        if current_event_sha != provenance.event_file_sha256:
            raise ArtifactMaterializationError(
                "event file bytes no longer match the exact calinfoset used for projection"
            )
    if calibration_identity is not None:
        if calibration_identity.identity_sha256 != provenance.calibration_identity_sha256:
            raise ArtifactMaterializationError(
                "calibration_identity does not match the calibration recorded "
                "by ProjectionResult"
            )

        authoritative_calibration = provenance.calibration_execution_evidence_record()
        if authoritative_calibration is None:
            raise ArtifactMaterializationError(
                "ProjectionResult has no authoritative calibration execution evidence"
            )

        if calibration_identity.evidence_record() != authoritative_calibration:
            raise ArtifactMaterializationError(
                "calibration execution evidence does not match the exact "
                "projection-time calibration recorded by ProjectionResult"
            )
    if sas_producer is not None:
        if sas_producer.identity_sha256 != provenance.producer_identity_sha256:
            raise ArtifactMaterializationError(
                "sas_producer does not match the producer recorded by ProjectionResult"
            )
        try:
            validate_projection_invocations(provenance, sas_producer)
        except ExecutionEvidenceError as exc:
            raise ArtifactMaterializationError(
                f"projection invocation evidence contradicts managed provenance: {exc}"
            ) from exc


def write_detector_geometry(
    path: str | Path,
    result: ProjectionResult,
    *,
    event_identity: EventIdentity | None = None,
    source_region_sha256: str | None = None,
    source_region_origin: str | None = None,
    calibration_identity: CalibrationIdentity | None = None,
    sas_producer: SasProducerIdentity | None = None,
    max_components: int = 4096,
    write_evidence: bool = True,
    _event_identity_reader: EventIdentityReader = read_event_identity,
) -> DetectorGeometryArtifact:
    """Materialise durable FITS geometry without creating an absolute-path wrapper.

    Any caller-supplied material provenance is checked against the authoritative
    ``ProjectionResult`` before it is embedded in the FITS artifact. This prevents
    a geometry produced from event/context A from being labelled as event/context B.

    Geometry validity and the complete detector->celestial source binding are
    re-established independently at this durable-artifact boundary. Matching
    hashes identify supplied values; they do not by themselves prove scientific
    topology or a complete source binding.
    """
    _validate_result_consistency(result)
    _validate_detector_topology(result)
    _validate_material_provenance(
        result,
        event_identity=event_identity,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
        event_identity_reader=_event_identity_reader,
    )
    output = write_fits_region(
        path,
        list(result.selection.regions),
        max_components=max_components,
        event_identity=event_identity,
        event_file_sha256=(
            result.provenance.event_file_sha256 if event_identity is not None else None
        ),
        source_region_sha256=source_region_sha256,
        celestial_geometry_sha256=result.provenance.celestial_geometry_sha256,
        projection_identity_sha256=result.provenance.projection_identity_sha256,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
    )
    evidence: Path | None = None
    if write_evidence:
        evidence = output.with_suffix(".provenance.json")
        write_projection_evidence(
            evidence,
            result,
            geometry_artifact=output,
            calibration_identity=calibration_identity,
            sas_producer=sas_producer,
            source_region_origin=source_region_origin,
        )
    return DetectorGeometryArtifact(
        path=output,
        geometry_sha256=result.selection.geometry_sha256,
        file_sha256=_file_sha256(output),
        projection_evidence=evidence,
    )

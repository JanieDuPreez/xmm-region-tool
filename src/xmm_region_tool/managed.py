"""Fail-closed artifact workflow for managed/library integrations.

Low-level projection and wrapper helpers remain available in their implementation
modules for debugging. This module is the supported path when detector geometry
will be cached, staged or reused by another application.
"""

from __future__ import annotations

from .limits import bounded_text, validate_region_table_budget

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits

from .artifacts import (
    ArtifactMaterializationError,
    DetectorGeometryArtifact,
    SasRegionfileMaterialization,
    materialize_sas_regionfile,
    write_detector_geometry,
)
from .calibration import CalibrationIdentity, SasProducerIdentity
from .execution import ADAPTIVE_REFINEMENT, LEGACY_REFINEMENT, ProjectionResult, ProjectionRule
from .provenance import EventIdentity, file_sha256, read_event_identity
from .publication import (
    discard_staged,
    lexical_absolute_path,
    publish_staged_files,
    temporary_sibling,
)


@dataclass(frozen=True)
class BoundDetectorGeometryArtifact:
    """Durable geometry with all material projection bindings required for reuse."""

    path: Path
    geometry_sha256: str
    file_sha256: str
    projection_evidence: Path
    event_file_sha256: str
    event_identity_sha256: str
    calibration_identity_sha256: str
    producer_identity_sha256: str
    celestial_geometry_sha256: str
    projection_identity_sha256: str


def _rewrite_geometry_reference(evidence: Path, geometry: Path) -> None:
    payload = json.loads(evidence.read_text())
    if not isinstance(payload, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence sidecar must contain a JSON object"
        )
    geometry_record = payload.get("geometry_artifact")
    if not isinstance(geometry_record, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence sidecar has no geometry-artifact record"
        )
    geometry_record["path"] = str(geometry)
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_bound_detector_geometry(
    path: str | Path,
    result: ProjectionResult,
    *,
    event_identity: EventIdentity,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    source_region_sha256: str | None = None,
    source_region_origin: str | None = None,
    max_components: int = 4096,
    _event_identity_reader=read_event_identity,
) -> BoundDetectorGeometryArtifact:
    """Write a fully provenance-bound detector artifact for managed reuse."""
    final_geometry = lexical_absolute_path(path)
    final_evidence = final_geometry.with_suffix(".provenance.json")
    stage_geometry = temporary_sibling(final_geometry)
    stage_evidence = stage_geometry.with_suffix(".provenance.json")
    try:
        artifact: DetectorGeometryArtifact = write_detector_geometry(
            stage_geometry,
            result,
            event_identity=event_identity,
            source_region_sha256=source_region_sha256,
            source_region_origin=source_region_origin,
            calibration_identity=calibration_identity,
            sas_producer=sas_producer,
            max_components=max_components,
            write_evidence=True,
            _event_identity_reader=_event_identity_reader,
        )
        if artifact.projection_evidence is None:
            raise ArtifactMaterializationError(
                "managed detector geometry requires a projection-evidence sidecar"
            )
        stage_evidence = artifact.projection_evidence

        provenance = result.provenance
        staged_bound = BoundDetectorGeometryArtifact(
            path=artifact.path,
            geometry_sha256=artifact.geometry_sha256,
            file_sha256=artifact.file_sha256,
            projection_evidence=stage_evidence,
            event_file_sha256=provenance.event_file_sha256,
            event_identity_sha256=provenance.event_identity_sha256,
            calibration_identity_sha256=provenance.calibration_identity_sha256,
            producer_identity_sha256=provenance.producer_identity_sha256,
            celestial_geometry_sha256=provenance.celestial_geometry_sha256,
            projection_identity_sha256=provenance.projection_identity_sha256,
        )
        _validate_projection_evidence(staged_bound)
        _rewrite_geometry_reference(stage_evidence, final_geometry)
        _validate_projection_evidence(staged_bound)
        publish_staged_files(
            ((stage_geometry, final_geometry), (stage_evidence, final_evidence))
        )
        return BoundDetectorGeometryArtifact(
            path=final_geometry,
            geometry_sha256=staged_bound.geometry_sha256,
            file_sha256=staged_bound.file_sha256,
            projection_evidence=final_evidence,
            event_file_sha256=staged_bound.event_file_sha256,
            event_identity_sha256=staged_bound.event_identity_sha256,
            calibration_identity_sha256=staged_bound.calibration_identity_sha256,
            producer_identity_sha256=staged_bound.producer_identity_sha256,
            celestial_geometry_sha256=staged_bound.celestial_geometry_sha256,
            projection_identity_sha256=staged_bound.projection_identity_sha256,
        )
    except Exception:
        discard_staged((stage_geometry, stage_evidence))
        raise


def _header_digest(header: fits.Header, key: str, *, description: str) -> str:
    value = header.get(key)
    if value is None:
        raise ArtifactMaterializationError(
            f"managed detector geometry is missing required {description} binding ({key})"
        )
    digest = str(value).strip().lower()
    if len(digest) != 64:
        raise ArtifactMaterializationError(
            f"managed detector geometry has invalid {description} SHA256 ({key})"
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ArtifactMaterializationError(
            f"managed detector geometry has invalid {description} SHA256 ({key})"
        ) from exc
    return digest


def _read_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactMaterializationError(
            f"managed detector geometry projection-evidence sidecar does not exist: {path}"
        )
    try:
        payload = json.loads(bounded_text(path))
    except (OSError, ValueError) as exc:
        raise ArtifactMaterializationError(
            f"managed detector geometry projection-evidence sidecar is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactMaterializationError(
            "projection-evidence sidecar must contain a JSON object"
        )
    if payload.get("schema") != "xmm-region-tool.projection-evidence/v2":
        raise ArtifactMaterializationError(
            "projection-evidence sidecar has an unsupported schema"
        )
    return payload


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    description: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ArtifactMaterializationError(
            f"projection-evidence sidecar has no valid {description} record"
        )
    return value


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_digest(
    payload: dict[str, Any],
    key: str,
    *,
    description: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactMaterializationError(
            f"projection-evidence has no valid {description} SHA256"
        )
    digest = value.lower()
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ArtifactMaterializationError(
            f"projection-evidence has no valid {description} SHA256"
        ) from exc
    return digest


def _canonical_rule_record(rule_record: dict[str, Any]) -> dict[str, object | None]:
    if rule_record.get("schema") != "xmm-region-tool.projection-rule/v2":
        raise ArtifactMaterializationError(
            "projection-evidence contains an unsupported projection-rule schema"
        )

    try:
        refinement = str(rule_record["refinement"])
        if refinement == ADAPTIVE_REFINEMENT:
            rule = ProjectionRule(
                method=str(rule_record["method"]),
                refinement=refinement,
                detector_tolerance=rule_record["detector_tolerance"],
                max_depth=rule_record["max_depth"],
                max_vertices=rule_record["max_vertices"],
            )
        elif refinement == LEGACY_REFINEMENT:
            rule = ProjectionRule(
                method=str(rule_record["method"]),
                refinement=refinement,
                samples=rule_record["samples"],
            )
        else:
            raise ArtifactMaterializationError(
                "projection-evidence contains an unsupported projection refinement"
            )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ArtifactMaterializationError):
            raise
        raise ArtifactMaterializationError(
            "projection-evidence contains an invalid projection-rule record"
        ) from exc

    canonical = rule.canonical_record()
    if rule_record != canonical:
        raise ArtifactMaterializationError(
            "projection-evidence projection-rule record is not canonical"
        )
    return canonical


def _canonical_calibration_name(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise ArtifactMaterializationError(f"{description} name is invalid")
    name = value.strip()
    if not name or name in {".", ".."} or Path(name).name != name or name != value:
        raise ArtifactMaterializationError(f"{description} name is not canonical")
    return name


def _canonical_digest_record(
    item: object,
    *,
    description: str,
) -> dict[str, str]:
    if not isinstance(item, dict) or set(item) != {"name", "sha256"}:
        raise ArtifactMaterializationError(f"{description} record is not canonical")
    name = _canonical_calibration_name(item.get("name"), description=description)
    digest = _required_digest(item, "sha256", description=description)
    canonical = {"name": name, "sha256": digest}
    if item != canonical:
        raise ArtifactMaterializationError(f"{description} record is not canonical")
    return canonical


def _validate_calibration_record(
    payload: dict[str, Any],
    *,
    expected_identity: str,
) -> None:
    calibration = payload.get("calibration")
    if calibration is None:
        raise ArtifactMaterializationError(
            "managed projection-evidence is missing the required calibration record"
        )
    if not isinstance(calibration, dict):
        raise ArtifactMaterializationError(
            "projection-evidence calibration record is invalid"
        )

    if "cif_file_sha256" not in calibration:
        raise ArtifactMaterializationError(
            "managed projection-evidence calibration record is missing required exact CIF evidence"
        )
    if "constituent_evidence" not in calibration:
        raise ArtifactMaterializationError(
            "managed projection-evidence calibration record is missing valid constituent evidence"
        )

    expected_fields = {
        "schema",
        "calindex_sha256",
        "constituents",
        "replacements",
        "identity_sha256",
        "cif_file_sha256",
        "constituent_evidence",
    }
    if set(calibration) != expected_fields:
        raise ArtifactMaterializationError(
            "projection-evidence calibration record is not canonical"
        )
    if calibration.get("schema") != "xmm-region-tool.calibration-identity/v2":
        raise ArtifactMaterializationError(
            "projection-evidence contains an unsupported calibration schema"
        )

    calindex_sha256 = _required_digest(
        calibration,
        "calindex_sha256",
        description="CALINDEX",
    )
    if calibration["calindex_sha256"] != calindex_sha256:
        raise ArtifactMaterializationError(
            "projection-evidence CALINDEX SHA256 is not canonical"
        )
    cif_file_sha256 = _required_digest(
        calibration,
        "cif_file_sha256",
        description="exact CIF",
    )
    if calibration["cif_file_sha256"] != cif_file_sha256:
        raise ArtifactMaterializationError(
            "projection-evidence exact CIF SHA256 is not canonical"
        )

    constituents = calibration.get("constituents")
    replacements = calibration.get("replacements")
    constituent_evidence = calibration.get("constituent_evidence")
    if not isinstance(constituents, list) or not isinstance(replacements, list):
        raise ArtifactMaterializationError(
            "projection-evidence calibration record is not canonical"
        )
    if not isinstance(constituent_evidence, list):
        raise ArtifactMaterializationError(
            "managed projection-evidence calibration record is missing valid "
            "constituent evidence"
        )

    canonical_constituents = [
        _canonical_digest_record(
            item,
            description="calibration constituent",
        )
        for item in constituents
    ]
    constituent_names = [item["name"] for item in canonical_constituents]
    if len(set(constituent_names)) != len(constituent_names):
        raise ArtifactMaterializationError(
            "projection-evidence calibration constituents contain duplicate basenames"
        )
    if constituent_names != sorted(constituent_names):
        raise ArtifactMaterializationError(
            "projection-evidence calibration constituents are not in canonical basename order"
        )

    canonical_replacements = [
        _canonical_digest_record(
            item,
            description="calibration replacement",
        )
        for item in replacements
    ]
    replacement_names = [item["name"] for item in canonical_replacements]
    if len(set(replacement_names)) != len(replacement_names):
        raise ArtifactMaterializationError(
            "projection-evidence calibration replacements contain duplicate basenames"
        )

    evidence_pairs: list[tuple[str, str]] = []
    required_evidence_fields = {"name", "sha256", "cif_md5", "size"}
    for item in constituent_evidence:
        if not isinstance(item, dict):
            raise ArtifactMaterializationError(
                "projection-evidence calibration constituent evidence is invalid"
            )
        if not required_evidence_fields.issubset(item):
            raise ArtifactMaterializationError(
                "managed projection-evidence calibration constituent evidence is incomplete"
            )
        if set(item) != required_evidence_fields:
            raise ArtifactMaterializationError(
                "projection-evidence calibration constituent evidence is not canonical"
            )

        name = _canonical_calibration_name(
            item["name"],
            description="calibration constituent evidence",
        )
        digest = _required_digest(
            item,
            "sha256",
            description="calibration constituent evidence",
        )
        if item["sha256"] != digest:
            raise ArtifactMaterializationError(
                "projection-evidence calibration constituent evidence is not canonical"
            )

        cif_md5 = item["cif_md5"]
        if cif_md5 is not None:
            if not isinstance(cif_md5, str) or len(cif_md5) != 32:
                raise ArtifactMaterializationError(
                    "projection-evidence calibration constituent CIF MD5 is invalid"
                )
            canonical_md5 = cif_md5.lower()
            try:
                int(canonical_md5, 16)
            except ValueError as exc:
                raise ArtifactMaterializationError(
                    "projection-evidence calibration constituent CIF MD5 is invalid"
                ) from exc
            if cif_md5 != canonical_md5:
                raise ArtifactMaterializationError(
                    "projection-evidence calibration constituent CIF MD5 is not canonical"
                )

        size = item["size"]
        if size is not None and (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ArtifactMaterializationError(
                "projection-evidence calibration constituent size is invalid"
            )

        evidence_pairs.append((name, digest))

    canonical_pairs = [
        (item["name"], item["sha256"])
        for item in canonical_constituents
    ]
    if evidence_pairs != canonical_pairs:
        raise ArtifactMaterializationError(
            "projection-evidence calibration constituent evidence does not "
            "match its canonical constituent record"
        )

    canonical = {
        "schema": calibration["schema"],
        "calindex_sha256": calindex_sha256,
        "constituents": canonical_constituents,
        "replacements": canonical_replacements,
    }
    calculated = _canonical_sha256(canonical)
    stored = _required_digest(
        calibration,
        "identity_sha256",
        description="calibration identity",
    )
    if calibration["identity_sha256"] != stored:
        raise ArtifactMaterializationError(
            "projection-evidence calibration identity SHA256 is not canonical"
        )
    if calculated != stored:
        raise ArtifactMaterializationError(
            "projection-evidence calibration identity does not match its canonical record"
        )
    if stored != expected_identity:
        raise ArtifactMaterializationError(
            "projection-evidence calibration record does not match the projection identity"
        )


def _validate_producer_record(
    payload: dict[str, Any],
    *,
    expected_identity: str,
) -> None:
    producer = payload.get("sas_producer")
    if producer is None:
        raise ArtifactMaterializationError(
            "managed projection-evidence is missing the required SAS producer record"
        )
    if not isinstance(producer, dict):
        raise ArtifactMaterializationError(
            "projection-evidence SAS producer record is invalid"
        )
    if producer.get("schema") != "xmm-region-tool.sas-producer/v3":
        raise ArtifactMaterializationError(
            "projection-evidence contains an unsupported SAS producer schema"
        )

    canonical = {
        "schema": producer["schema"],
        "esky2det_name": producer.get("esky2det_name"),
        "esky2det_version": producer.get("esky2det_version"),
        "esky2det_sha256": producer.get("esky2det_sha256"),
        "sas_version": producer.get("sas_version"),
    }
    calculated = _canonical_sha256(canonical)
    stored = _required_digest(
        producer,
        "identity_sha256",
        description="SAS producer identity",
    )
    if calculated != stored:
        raise ArtifactMaterializationError(
            "projection-evidence SAS producer identity does not match its canonical record"
        )
    if stored != expected_identity:
        raise ArtifactMaterializationError(
            "projection-evidence SAS producer record does not match the projection identity"
        )


def _validate_projection_evidence(
    artifact: BoundDetectorGeometryArtifact,
) -> dict[str, Any]:
    payload = _read_evidence(Path(artifact.projection_evidence).expanduser().resolve())
    projection = _required_mapping(
        payload,
        "projection",
        description="projection",
    )
    geometry_record = _required_mapping(
        payload,
        "geometry_artifact",
        description="geometry-artifact",
    )

    if projection.get("schema") != "xmm-region-tool.projection/v3":
        raise ArtifactMaterializationError(
            "projection-evidence contains an unsupported projection schema"
        )

    rule_record = _required_mapping(
        projection,
        "rule",
        description="projection-rule",
    )
    canonical_rule = _canonical_rule_record(rule_record)

    canonical_projection = {
        "schema": projection["schema"],
        "event_file_sha256": projection.get("event_file_sha256"),
        "event_identity_sha256": projection.get("event_identity_sha256"),
        "celestial_geometry_sha256": projection.get("celestial_geometry_sha256"),
        "context_identity_sha256": projection.get("context_identity_sha256"),
        "calibration_identity_sha256": projection.get(
            "calibration_identity_sha256"
        ),
        "producer_identity_sha256": projection.get("producer_identity_sha256"),
        "rule": canonical_rule,
    }

    calculated_projection_identity = _canonical_sha256(canonical_projection)
    stored_projection_identity = _required_digest(
        projection,
        "projection_identity_sha256",
        description="projection identity",
    )
    if calculated_projection_identity != stored_projection_identity:
        raise ArtifactMaterializationError(
            "projection-evidence projection identity does not match its canonical record"
        )

    checks = {
        "event_file_sha256": artifact.event_file_sha256,
        "event_identity_sha256": artifact.event_identity_sha256,
        "calibration_identity_sha256": artifact.calibration_identity_sha256,
        "producer_identity_sha256": artifact.producer_identity_sha256,
        "celestial_geometry_sha256": artifact.celestial_geometry_sha256,
        "projection_identity_sha256": artifact.projection_identity_sha256,
        "detector_geometry_sha256": artifact.geometry_sha256,
    }
    for key, expected in checks.items():
        actual = projection.get(key)
        if actual != expected:
            raise ArtifactMaterializationError(
                f"projection-evidence {key} does not match the bound detector artifact"
            )

    _validate_calibration_record(
        payload,
        expected_identity=artifact.calibration_identity_sha256,
    )
    _validate_producer_record(
        payload,
        expected_identity=artifact.producer_identity_sha256,
    )

    if geometry_record.get("file_sha256") != artifact.file_sha256:
        raise ArtifactMaterializationError(
            "projection-evidence geometry file SHA256 does not match the bound artifact"
        )
    if geometry_record.get("geometry_sha256") != artifact.geometry_sha256:
        raise ArtifactMaterializationError(
            "projection-evidence detector geometry SHA256 does not match the bound artifact"
        )

    return payload


def load_bound_detector_geometry(
    path: str | Path,
    *,
    projection_evidence: str | Path | None = None,
) -> BoundDetectorGeometryArtifact:
    """Safely rehydrate a persisted managed FITS + projection-evidence pair."""
    geometry = Path(path).expanduser().resolve()
    if not geometry.is_file():
        raise ArtifactMaterializationError(f"detector geometry does not exist: {geometry}")

    evidence = (
        Path(projection_evidence).expanduser().resolve()
        if projection_evidence is not None
        else geometry.with_suffix(".provenance.json")
    )

    with fits.open(geometry, memmap=False) as hdus:
        if "REGION" not in hdus:
            raise ArtifactMaterializationError("detector geometry has no REGION extension")
        header = hdus["REGION"].header
        try:
            validate_region_table_budget(hdus["REGION"])
        except ValueError as exc:
            raise ArtifactMaterializationError(str(exc)) from exc

        event_file_sha256 = _header_digest(
            header, "XMRGEVT", description="exact-event"
        )
        event_identity_sha256 = _header_digest(
            header, "XMMRGID", description="event-identity"
        )
        calibration_identity_sha256 = _header_digest(
            header, "XMRGCCF", description="calibration"
        )
        producer_identity_sha256 = _header_digest(
            header, "XMRGPRD", description="producer"
        )
        celestial_geometry_sha256 = _header_digest(
            header, "XMRGCEL", description="celestial-geometry"
        )
        projection_identity_sha256 = _header_digest(
            header, "XMRGPRO", description="projection-identity"
        )

    file_digest = file_sha256(geometry)
    payload = _read_evidence(evidence)
    projection = _required_mapping(payload, "projection", description="projection")

    detector_geometry_sha256 = projection.get("detector_geometry_sha256")
    if not isinstance(detector_geometry_sha256, str) or len(detector_geometry_sha256) != 64:
        raise ArtifactMaterializationError(
            "projection-evidence sidecar has no valid detector geometry SHA256"
        )

    artifact = BoundDetectorGeometryArtifact(
        path=geometry,
        geometry_sha256=detector_geometry_sha256,
        file_sha256=file_digest,
        projection_evidence=evidence,
        event_file_sha256=event_file_sha256,
        event_identity_sha256=event_identity_sha256,
        calibration_identity_sha256=calibration_identity_sha256,
        producer_identity_sha256=producer_identity_sha256,
        celestial_geometry_sha256=celestial_geometry_sha256,
        projection_identity_sha256=projection_identity_sha256,
    )
    _validate_projection_evidence(artifact)
    return artifact


def _validate_bound_artifact(
    artifact: BoundDetectorGeometryArtifact,
    *,
    event_identity: EventIdentity,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    expected_celestial_geometry_sha256: str,
    expected_projection_identity_sha256: str,
) -> None:
    geometry = Path(artifact.path).expanduser().resolve()
    if not geometry.is_file():
        raise ArtifactMaterializationError(f"detector geometry does not exist: {geometry}")

    current_geometry_sha = file_sha256(geometry)
    if current_geometry_sha != artifact.file_sha256:
        raise ArtifactMaterializationError(
            "detector geometry bytes changed after the bound artifact was created"
        )

    current_event_sha = file_sha256(event_identity.path)
    if current_event_sha != artifact.event_file_sha256:
        raise ArtifactMaterializationError(
            "event file bytes do not match the exact event used to create detector geometry"
        )
    if event_identity.identity_sha256 != artifact.event_identity_sha256:
        raise ArtifactMaterializationError(
            "event identity does not match the detector geometry binding"
        )
    if calibration_identity.identity_sha256 != artifact.calibration_identity_sha256:
        raise ArtifactMaterializationError(
            "calibration identity does not match the detector geometry binding"
        )
    if sas_producer.identity_sha256 != artifact.producer_identity_sha256:
        raise ArtifactMaterializationError(
            "SAS producer identity does not match the detector geometry binding"
        )

    if expected_celestial_geometry_sha256 != artifact.celestial_geometry_sha256:
        raise ArtifactMaterializationError(
            "requested celestial geometry does not match the detector geometry binding"
        )
    if expected_projection_identity_sha256 != artifact.projection_identity_sha256:
        raise ArtifactMaterializationError(
            "requested projection identity does not match the detector geometry binding"
        )

    with fits.open(geometry, memmap=False) as hdus:
        if "REGION" not in hdus:
            raise ArtifactMaterializationError("detector geometry has no REGION extension")
        header = hdus["REGION"].header
        try:
            validate_region_table_budget(hdus["REGION"])
        except ValueError as exc:
            raise ArtifactMaterializationError(str(exc)) from exc
        checks = {
            "XMRGEVT": (artifact.event_file_sha256, "exact-event"),
            "XMMRGID": (artifact.event_identity_sha256, "event-identity"),
            "XMRGCCF": (artifact.calibration_identity_sha256, "calibration"),
            "XMRGPRD": (artifact.producer_identity_sha256, "producer"),
            "XMRGCEL": (artifact.celestial_geometry_sha256, "celestial-geometry"),
            "XMRGPRO": (artifact.projection_identity_sha256, "projection-identity"),
        }
        for key, (expected, description) in checks.items():
            actual = _header_digest(header, key, description=description)
            if actual != expected:
                raise ArtifactMaterializationError(
                    f"detector geometry {description} binding does not match its managed artifact"
                )

    _validate_projection_evidence(artifact)


def materialize_bound_sas_regionfile(
    path: str | Path,
    artifact: BoundDetectorGeometryArtifact,
    *,
    event_identity: EventIdentity,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    expected_celestial_geometry_sha256: str,
    expected_projection_identity_sha256: str,
) -> SasRegionfileMaterialization:
    """Stage a SAS wrapper only after exact artifact and requested-cell validation."""
    _validate_bound_artifact(
        artifact,
        event_identity=event_identity,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
        expected_celestial_geometry_sha256=expected_celestial_geometry_sha256,
        expected_projection_identity_sha256=expected_projection_identity_sha256,
    )
    final_wrapper = lexical_absolute_path(path)
    stage_wrapper = temporary_sibling(final_wrapper)
    try:
        staged = materialize_sas_regionfile(stage_wrapper, artifact.path)
        publish_staged_files(((stage_wrapper, final_wrapper),))
        return SasRegionfileMaterialization(
            path=final_wrapper,
            geometry_path=staged.geometry_path,
        )
    except Exception:
        discard_staged((stage_wrapper,))
        raise
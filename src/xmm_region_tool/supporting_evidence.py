"""Canonical integrity record for non-semantic managed execution/output evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


class SupportingEvidenceError(ValueError):
    """Raised when supporting evidence cannot be canonicalized or validated."""


SOURCE_ORIGINS = frozenset({"tool-hashed-source-file", "caller-asserted"})


def canonical_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise SupportingEvidenceError(
            f"{description} must be a canonical lower-case SHA256 digest"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise SupportingEvidenceError(
            f"{description} must be a canonical lower-case SHA256 digest"
        ) from exc
    return value


def _plain_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


def _json_clone(value: object) -> object:
    try:
        encoded = json.dumps(
            _plain_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SupportingEvidenceError(
            "supporting evidence must contain finite JSON-compatible values"
        ) from exc


def build_supporting_evidence_record(
    *,
    projection_identity_sha256: str,
    detector_geometry_sha256: str,
    converter_runtime: Mapping[str, object],
    command: object,
    commands: object,
    relevant_environment: Mapping[str, str],
    refinement_diagnostics: Mapping[str, object],
    calibration: Mapping[str, object],
    sas_producer: Mapping[str, object],
    source_region_sha256: str | None,
    source_region_origin: str | None,
) -> dict[str, object]:
    source_sha = None
    source_origin = None
    if source_region_sha256 is not None:
        source_sha = canonical_sha256(
            source_region_sha256,
            description="source-region evidence",
        )
        if source_region_origin not in SOURCE_ORIGINS:
            raise SupportingEvidenceError(
                "source-region evidence must declare tool-hashed-source-file or caller-asserted origin"
            )
        source_origin = source_region_origin
    elif source_region_origin is not None:
        raise SupportingEvidenceError(
            "source-region provenance origin cannot be present without a source SHA256"
        )
    record = {
        "schema": "xmm-region-tool.supporting-evidence/v1",
        "projection_identity_sha256": canonical_sha256(
            projection_identity_sha256,
            description="projection identity",
        ),
        "detector_geometry_sha256": canonical_sha256(
            detector_geometry_sha256,
            description="detector geometry",
        ),
        "converter_runtime": converter_runtime,
        "command": command,
        "commands": commands,
        "relevant_environment": relevant_environment,
        "refinement_diagnostics": refinement_diagnostics,
        "calibration": calibration,
        "sas_producer": sas_producer,
        "source_region_sha256": source_sha,
        "source_region_origin": source_origin,
    }
    cloned = _json_clone(record)
    if not isinstance(cloned, dict):
        raise SupportingEvidenceError("supporting evidence canonicalization failed")
    return cloned


def supporting_evidence_sha256(record: Mapping[str, object]) -> str:
    if record.get("schema") != "xmm-region-tool.supporting-evidence/v1":
        raise SupportingEvidenceError("unsupported supporting-evidence schema")
    try:
        encoded = json.dumps(
            _plain_json_value(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SupportingEvidenceError(
            "supporting evidence must contain finite JSON-compatible values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()

"""Canonical SAS projection-context identity validation.

The semantic context identity is intentionally limited to calibration and SAS
producer meaning. Observation-association/suitability evidence remains
non-semantic and must not be added here merely because it is recorded elsewhere.
"""

from __future__ import annotations

import hashlib
import json


class ContextIdentityError(ValueError):
    """Raised when a projection context digest is malformed or non-canonical."""


def _canonical_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ContextIdentityError(f"{description} must be a canonical lower-case SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContextIdentityError(
            f"{description} must be a canonical lower-case SHA256 digest"
        ) from exc
    return value


def canonical_context_record(
    calibration_identity_sha256: str,
    producer_identity_sha256: str,
) -> dict[str, str]:
    """Return the canonical ``sas-projection-context/v2`` semantic record."""
    return {
        "schema": "xmm-region-tool.sas-projection-context/v2",
        "calibration_identity_sha256": _canonical_sha256(
            calibration_identity_sha256,
            description="calibration identity",
        ),
        "producer_identity_sha256": _canonical_sha256(
            producer_identity_sha256,
            description="SAS producer identity",
        ),
    }


def canonical_context_identity_sha256(
    calibration_identity_sha256: str,
    producer_identity_sha256: str,
) -> str:
    record = canonical_context_record(
        calibration_identity_sha256,
        producer_identity_sha256,
    )
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_projection_context_identity(
    stored_context_identity_sha256: object,
    *,
    calibration_identity_sha256: str,
    producer_identity_sha256: str,
) -> str:
    """Require one stored context digest to match canonical v2 context meaning."""
    stored = _canonical_sha256(
        stored_context_identity_sha256,
        description="projection context identity",
    )
    expected = canonical_context_identity_sha256(
        calibration_identity_sha256,
        producer_identity_sha256,
    )
    if stored != expected:
        raise ContextIdentityError(
            "projection context identity does not match canonical calibration/producer meaning"
        )
    return stored


__all__ = [
    "ContextIdentityError",
    "canonical_context_identity_sha256",
    "canonical_context_record",
    "validate_projection_context_identity",
]

"""Completeness policy for durable SAS producer evidence.

``SasProducerIdentity`` remains a flexible value object for low-level tests and
explicit debugging.  Supported durable/high-level scientific surfaces call the
validator here before publishing or reusing artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ProducerEvidenceError(ValueError):
    """Raised when SAS producer evidence is incomplete for durable scientific use."""


def _canonical_text(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise ProducerEvidenceError(f"{description} must be a non-empty string")
    text = value.strip()
    if not text or text != value:
        raise ProducerEvidenceError(f"{description} must be a canonical non-empty string")
    return text


def _canonical_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise ProducerEvidenceError(f"{description} must be a canonical SHA256 digest")
    digest = value.strip().lower()
    if digest != value or len(digest) != 64:
        raise ProducerEvidenceError(f"{description} must be a canonical SHA256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ProducerEvidenceError(f"{description} must be a canonical SHA256 digest") from exc
    return digest


def validate_durable_sas_producer(producer: Any) -> None:
    """Require complete exact SAS producer evidence for a durable artifact.

    The initial release deliberately requires a separately captured non-empty
    ``sas_version``.  It does not infer a SAS release from arbitrary task-banner
    text.  Low-level/debug ``SasProducerIdentity`` values may remain partial, but
    callers must not promote them through a supported durable surface.
    """
    name = _canonical_text(
        getattr(producer, "esky2det_name", None),
        description="esky2det executable name",
    )
    if Path(name).name != name or name in {".", ".."}:
        raise ProducerEvidenceError("esky2det executable name must be one canonical basename")
    _canonical_text(
        getattr(producer, "esky2det_version", None),
        description="esky2det task version/build",
    )
    _canonical_sha256(
        getattr(producer, "esky2det_sha256", None),
        description="exact esky2det executable SHA256",
    )
    _canonical_text(
        getattr(producer, "sas_version", None),
        description="SAS release/build identifier",
    )


__all__ = ["ProducerEvidenceError", "validate_durable_sas_producer"]

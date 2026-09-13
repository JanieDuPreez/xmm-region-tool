"""Supported batch API for ordinary scientific use.

The implementation batch module retains alternate serialization modes for
explicit debugging/comparison work.  The top-level package boundary exposes
only the validated FITS-backed representation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .batch import (
    BatchConversionError,
    BatchConversionResult,
    ContextResolver,
    convert_selection_batch as _convert_selection_batch,
)
from .execution import ProjectionRule, SasProjectionContext
from .model import CelestialSelection
from .producer_policy import validate_durable_sas_producer


def _validated_context_resolver(resolver: ContextResolver) -> ContextResolver:
    def validated(event_path, identity):
        context = resolver(event_path, identity)
        if not isinstance(context, SasProjectionContext):
            return context
        validate_durable_sas_producer(context.producer)
        return context

    return validated


def convert_selection_batch(
    celestial_selection: CelestialSelection,
    event_files: Sequence[str | Path],
    output_dir: str | Path,
    *,
    rule: ProjectionRule,
    context: SasProjectionContext | None = None,
    context_resolver: ContextResolver | None = None,
    source_region_sha256: str | None = None,
    source_region_origin: str | None = None,
    source_reference: str | None = None,
    max_components: int = 4096,
) -> BatchConversionResult:
    """Project one selection independently for all events using FITS-backed output.

    Direct selectlib-expression output is intentionally absent from the
    supported top-level API.  It remains available only from the implementation
    module for debugging/comparison and is not a managed/durable representation.
    Complete exact SAS producer evidence is required before any supported batch
    output is published. Caller-supplied source-file digests remain explicitly
    caller-asserted unless their provenance origin is supplied by a verified adapter.
    """
    if context is not None:
        validate_durable_sas_producer(context.producer)
    resolved_resolver = (
        _validated_context_resolver(context_resolver)
        if context_resolver is not None
        else None
    )
    return _convert_selection_batch(
        celestial_selection,
        event_files,
        output_dir,
        rule=rule,
        context=context,
        context_resolver=resolved_resolver,
        source_region_sha256=source_region_sha256,
        source_region_origin=source_region_origin,
        source_reference=source_reference,
        representation="fits",
        max_components=max_components,
    )


__all__ = ["BatchConversionError", "BatchConversionResult", "convert_selection_batch"]

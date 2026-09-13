"""Batch conversion of one celestial selection for multiple EPIC event products."""

from __future__ import annotations

from .limits import MAX_COMPONENTS, MAX_INLINE_LIMIT, integer_limit

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .event_roles import validate_projection_event_generation
from .execution import (
    ProjectionRule,
    SasProjectionContext,
    _freeze_evidence_value,
    _plain_evidence_value,
)
from .geometry import load_ds9_selection_snapshot
from .model import CelestialSelection
from .output import WrittenRegionFiles
from .path_safety import (
    PathSafetyError,
    resolved_descendant,
    safe_filename_component,
    validate_distinct_inputs,
    validate_output_namespace,
)
from .provenance import EventIdentity, file_sha256, read_event_identity
from .publication import atomic_write_text
from .sas import SasConversionError, project_selection
from .transactional_products import write_projected_product_atomic


class BatchConversionError(ValueError):
    """Raised when batch-wide input cannot be represented safely."""


ContextResolver = Callable[[Path, EventIdentity], SasProjectionContext]


@dataclass(frozen=True)
class BatchItemResult:
    """Success or explicit failure for one supplied EPIC event product."""

    event_file: Path
    status: Literal["success", "failed"]
    event_identity: EventIdentity | None = None
    label: str | None = None
    written: WrittenRegionFiles | None = None
    event_file_sha256: str | None = None
    celestial_geometry_sha256: str | None = None
    detector_geometry_sha256: str | None = None
    projection_identity_sha256: str | None = None
    context_identity_sha256: str | None = None
    calibration_identity_sha256: str | None = None
    producer_identity_sha256: str | None = None
    refinement_diagnostics: Mapping[str, object] | None = None
    projection_evidence: Path | None = None
    error_type: str | None = None
    error_message: str | None = None
    failure_evidence: Mapping[str, object | None] | None = None

    def __post_init__(self) -> None:
        for name in ("refinement_diagnostics", "failure_evidence"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_evidence_value(value))


@dataclass(frozen=True)
class BatchConversionResult:
    """Complete multi-exposure conversion and fail-explicit manifest."""

    items: tuple[BatchItemResult, ...]
    manifest: Path

    @property
    def failures(self) -> tuple[BatchItemResult, ...]:
        return tuple(item for item in self.items if item.status == "failed")

    @property
    def successful(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _PlannedBatchItem:
    event_path: Path
    identity: EventIdentity | None
    event_sha: str | None
    context: SasProjectionContext | None
    label: str | None
    error: Exception | None = None


_SOURCE_REGION_ORIGINS = frozenset({"tool-hashed-source-file", "caller-asserted"})


def _safe_label(value: str) -> str:
    try:
        return safe_filename_component(value)
    except (PathSafetyError, TypeError) as exc:
        raise BatchConversionError(
            f"cannot form a safe output label from {value!r}"
        ) from exc


def _source_region_origin(
    source_region_sha256: str | None,
    source_region_origin: str | None,
) -> str | None:
    if source_region_sha256 is None:
        if source_region_origin is not None:
            raise BatchConversionError(
                "source_region_origin requires source_region_sha256"
            )
        return None
    effective = source_region_origin or "caller-asserted"
    if effective not in _SOURCE_REGION_ORIGINS:
        raise BatchConversionError(
            "source_region_origin must be tool-hashed-source-file or caller-asserted"
        )
    return effective


def event_output_label(
    identity: EventIdentity,
    event_file: Path,
    event_file_sha256: str,
) -> str:
    """Return a deterministic collision-resistant label for one exact event artifact."""
    exposure = identity.exposure_id or event_file.stem
    return "-".join(
        (
            _safe_label(identity.obs_id),
            _safe_label(identity.instrument),
            _safe_label(exposure),
            event_file_sha256[:12],
        )
    )


def _context_record(context: SasProjectionContext) -> dict[str, object]:
    return {
        "identity_sha256": context.identity_sha256,
        "calibration": context.calibration.evidence_record(),
        "sas_producer": {
            **context.producer.canonical_record(),
            "identity_sha256": context.producer.identity_sha256,
        },
        "relevant_environment": context.relevant_environment_record(),
    }


def _manifest_entry(item: BatchItemResult, root: Path) -> dict[str, object | None]:
    entry: dict[str, object | None] = {
        "event_file": str(item.event_file),
        "status": item.status,
        "label": item.label,
        "event_file_sha256": item.event_file_sha256,
        "celestial_geometry_sha256": item.celestial_geometry_sha256,
        "detector_geometry_sha256": item.detector_geometry_sha256,
        "projection_identity_sha256": item.projection_identity_sha256,
        "context_identity_sha256": item.context_identity_sha256,
        "calibration_identity_sha256": item.calibration_identity_sha256,
        "producer_identity_sha256": item.producer_identity_sha256,
        "refinement_diagnostics": _plain_evidence_value(item.refinement_diagnostics),
        "error_type": item.error_type,
        "error_message": item.error_message,
        "failure_evidence": _plain_evidence_value(item.failure_evidence),
    }
    if item.event_identity is not None:
        entry["event_identity"] = item.event_identity.canonical_record()
        entry["event_identity_sha256"] = item.event_identity.identity_sha256
    if item.written is not None:
        entry["representation"] = item.written.representation
        entry["regionfile"] = str(item.written.regionfile.relative_to(root))
        entry["fits_region"] = (
            str(item.written.fits_region.relative_to(root))
            if item.written.fits_region is not None
            else None
        )
    if item.projection_evidence is not None:
        entry["projection_evidence"] = str(item.projection_evidence.relative_to(root))
    return entry


def _validate_context_contract(
    event_files: Sequence[str | Path],
    *,
    context: SasProjectionContext | None,
    context_resolver: ContextResolver | None,
) -> None:
    if (context is None) == (context_resolver is None):
        raise BatchConversionError(
            "provide exactly one of context or context_resolver for batch projection"
        )
    if context is not None and not isinstance(context, SasProjectionContext):
        raise TypeError(f"context must be SasProjectionContext, got {type(context).__name__}")
    if context is None:
        return

    obs_ids: set[str] = set()
    for value in event_files:
        path = Path(value).expanduser().resolve()
        try:
            obs_ids.add(read_event_identity(path).obs_id)
        except Exception:  # noqa: BLE001
            continue
    if len(obs_ids) > 1:
        joined = ", ".join(sorted(obs_ids))
        raise BatchConversionError(
            "one frozen SasProjectionContext cannot be reused across multiple ObsIDs "
            f"({joined}); supply context_resolver for per-event contexts"
        )


def _planned_fits_output(output: Path) -> Path:
    suffix = output.suffix
    if suffix.lower() == ".fits":
        return output.with_name(output.stem + "-geometry.fits")
    if suffix:
        return output.with_suffix(".fits")
    return output.with_name(output.name + ".fits")


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
    source_region_path: str | Path | None = None,
    representation: Literal["fits", "expression", "auto"] = "fits",
    inline_limit: int = 4096,
    max_components: int = 4096,
) -> BatchConversionResult:
    """Project one in-memory celestial selection independently for every event file."""
    if not isinstance(celestial_selection, CelestialSelection):
        raise TypeError(
            "celestial_selection must be CelestialSelection, got "
            f"{type(celestial_selection).__name__}"
        )
    if not isinstance(rule, ProjectionRule):
        raise TypeError(f"rule must be ProjectionRule, got {type(rule).__name__}")
    inline_limit = integer_limit(inline_limit, "inline_limit", 1, MAX_INLINE_LIMIT)
    max_components = integer_limit(max_components, "max_components", 1, MAX_COMPONENTS)
    if not event_files:
        raise BatchConversionError("at least one event file is required")
    _validate_context_contract(event_files, context=context, context_resolver=context_resolver)
    effective_source_origin = _source_region_origin(
        source_region_sha256,
        source_region_origin,
    )

    root = Path(output_dir).expanduser().resolve()
    event_paths = tuple(Path(value).expanduser().resolve() for value in event_files)
    try:
        validate_distinct_inputs(
            tuple((f"event file {index}", path) for index, path in enumerate(event_paths, start=1))
        )
    except PathSafetyError as exc:
        raise BatchConversionError(f"duplicate/aliased event input: {exc}") from exc

    protected_inputs: list[tuple[str, Path]] = [
        (f"event input {index}", path) for index, path in enumerate(event_paths, start=1)
    ]
    if source_region_path is not None:
        protected_inputs.append(
            ("source DS9 region", Path(source_region_path).expanduser().resolve())
        )
    generated_namespace: list[tuple[str, Path]] = [
        ("batch manifest", resolved_descendant(root, root / "xmm-region-manifest.json", role="batch manifest"))
    ]

    planned: list[_PlannedBatchItem] = []
    used_labels: set[str] = set()
    for event_path in event_paths:
        identity: EventIdentity | None = None
        event_sha: str | None = None
        resolved_context: SasProjectionContext | None = None
        label: str | None = None
        try:
            event_sha = file_sha256(event_path)
            identity = read_event_identity(event_path)
            label = event_output_label(identity, event_path, event_sha)
            if label in used_labels:
                raise BatchConversionError(
                    f"duplicate exact event artifact maps to output label {label!r}"
                )
            used_labels.add(label)
            resolved_context = (
                context_resolver(event_path, identity)
                if context_resolver is not None
                else context
            )
            if not isinstance(resolved_context, SasProjectionContext):
                raise TypeError(
                    "context_resolver must return SasProjectionContext, got "
                    f"{type(resolved_context).__name__}"
                )
            protected_inputs.append(
                (f"active CIF for {label}", resolved_context.calibration.cif_path)
            )
            wrapper = resolved_descendant(
                root,
                root / f"{label}.txt",
                role="batch region wrapper",
            )
            evidence = resolved_descendant(
                root,
                root / f"{label}.provenance.json",
                role="batch projection evidence",
            )
            generated_namespace.extend(
                ((f"region wrapper {label}", wrapper), (f"projection sidecar {label}", evidence))
            )
            if representation in {"fits", "auto"}:
                generated_namespace.append(
                    (
                        f"FITS geometry {label}",
                        resolved_descendant(
                            root,
                            _planned_fits_output(wrapper),
                            role="batch FITS REGION geometry",
                        ),
                    )
                )
            planned.append(
                _PlannedBatchItem(
                    event_path,
                    identity,
                    event_sha,
                    resolved_context,
                    label,
                )
            )
        except Exception as exc:  # noqa: BLE001
            planned.append(
                _PlannedBatchItem(
                    event_path,
                    identity,
                    event_sha,
                    resolved_context,
                    label,
                    exc,
                )
            )

    try:
        validate_output_namespace(generated_namespace, protected_inputs)
    except PathSafetyError as exc:
        raise BatchConversionError(f"unsafe batch output namespace: {exc}") from exc

    root.mkdir(parents=True, exist_ok=True)
    items: list[BatchItemResult] = []
    for plan in planned:
        event_path = plan.event_path
        identity = plan.identity
        event_sha = plan.event_sha
        resolved_context = plan.context
        label = plan.label
        if plan.error is not None:
            exc = plan.error
            items.append(
                BatchItemResult(
                    event_file=event_path,
                    status="failed",
                    event_identity=identity,
                    label=label,
                    event_file_sha256=None,
                    celestial_geometry_sha256=celestial_selection.geometry_sha256,
                    context_identity_sha256=(
                        resolved_context.identity_sha256
                        if isinstance(resolved_context, SasProjectionContext)
                        else None
                    ),
                    calibration_identity_sha256=(
                        resolved_context.calibration.identity_sha256
                        if isinstance(resolved_context, SasProjectionContext)
                        else None
                    ),
                    producer_identity_sha256=(
                        resolved_context.producer.identity_sha256
                        if isinstance(resolved_context, SasProjectionContext)
                        else None
                    ),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    failure_evidence=(
                        exc.evidence_record() if isinstance(exc, SasConversionError) else None
                    ),
                )
            )
            continue

        assert identity is not None
        assert event_sha is not None
        assert resolved_context is not None
        assert label is not None
        try:
            projected = project_selection(
                celestial_selection,
                calinfoset=event_path,
                context=resolved_context,
                rule=rule,
            )
            authoritative_identity = validate_projection_event_generation(
                event_path,
                identity,
                projection_event_identity_sha256=projected.provenance.event_identity_sha256,
                projection_event_file_sha256=projected.provenance.event_file_sha256,
                expected_event_file_sha256=event_sha,
            )
            written, evidence_target, _unused = write_projected_product_atomic(
                root / f"{label}.txt",
                projected,
                representation=representation,
                inline_limit=inline_limit,
                max_components=max_components,
                event_identity=authoritative_identity,
                event_file_sha256=projected.provenance.event_file_sha256,
                source_region_sha256=source_region_sha256,
                celestial_geometry_sha256=projected.provenance.celestial_geometry_sha256,
                projection_identity_sha256=projected.provenance.projection_identity_sha256,
                calibration_identity=resolved_context.calibration,
                sas_producer=resolved_context.producer,
                source_region_origin=effective_source_origin,
            )
            items.append(
                BatchItemResult(
                    event_file=event_path,
                    status="success",
                    event_identity=authoritative_identity,
                    label=label,
                    written=written,
                    event_file_sha256=projected.provenance.event_file_sha256,
                    celestial_geometry_sha256=projected.provenance.celestial_geometry_sha256,
                    detector_geometry_sha256=projected.selection.geometry_sha256,
                    projection_identity_sha256=projected.provenance.projection_identity_sha256,
                    context_identity_sha256=resolved_context.identity_sha256,
                    calibration_identity_sha256=resolved_context.calibration.identity_sha256,
                    producer_identity_sha256=resolved_context.producer.identity_sha256,
                    refinement_diagnostics=projected.provenance.evidence_record()[
                        "refinement_diagnostics"
                    ],
                    projection_evidence=evidence_target,
                )
            )
        except Exception as exc:  # noqa: BLE001
            items.append(
                BatchItemResult(
                    event_file=event_path,
                    status="failed",
                    event_identity=identity,
                    label=label,
                    event_file_sha256=None,
                    celestial_geometry_sha256=celestial_selection.geometry_sha256,
                    context_identity_sha256=resolved_context.identity_sha256,
                    calibration_identity_sha256=resolved_context.calibration.identity_sha256,
                    producer_identity_sha256=resolved_context.producer.identity_sha256,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    failure_evidence=(
                        exc.evidence_record() if isinstance(exc, SasConversionError) else None
                    ),
                )
            )

    manifest = root / "xmm-region-manifest.json"
    payload: dict[str, object] = {
        "schema": "xmm-region-tool.batch/v4",
        "source_reference": source_reference,
        "source_region_sha256": source_region_sha256,
        "source_region_origin": effective_source_origin,
        "celestial_geometry_sha256": celestial_selection.geometry_sha256,
        "projection_rule": rule.canonical_record(),
        "representation": representation,
        "context_mode": "per-event-resolver" if context_resolver is not None else "shared-single-obsid",
        "successful": not any(item.status == "failed" for item in items),
        "items": [_manifest_entry(item, root) for item in items],
    }
    if context is not None:
        payload["shared_context"] = _context_record(context)
    atomic_write_text(manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return BatchConversionResult(tuple(items), manifest)


def convert_batch(
    source_region: str | Path,
    event_files: Sequence[str | Path],
    output_dir: str | Path,
    *,
    context: SasProjectionContext | None = None,
    context_resolver: ContextResolver | None = None,
    rule: ProjectionRule | None = None,
    samples: int | None = None,
    detector_tolerance: float = 1.0,
    max_depth: int = 12,
    max_vertices: int = 4096,
    representation: Literal["fits", "expression", "auto"] = "fits",
    inline_limit: int = 4096,
    max_components: int = 4096,
) -> BatchConversionResult:
    """DS9 adapter for :func:`convert_selection_batch`.

    Adaptive detector-space refinement is the default. Supplying ``samples`` is
    an explicit request for the legacy fixed-source-sampling comparison mode.
    Callers with a predeclared contract may instead supply ``rule`` directly.
    """
    if rule is not None and samples is not None:
        raise BatchConversionError("provide rule or legacy samples, not both")
    projection_rule = (
        rule
        if rule is not None
        else (
            ProjectionRule(samples=samples)
            if samples is not None
            else ProjectionRule(
                detector_tolerance=detector_tolerance,
                max_depth=max_depth,
                max_vertices=max_vertices,
            )
        )
    )
    if not isinstance(projection_rule, ProjectionRule):
        raise TypeError(f"rule must be ProjectionRule, got {type(projection_rule).__name__})")

    source_path = Path(source_region).expanduser().resolve()
    sampling_hint = samples if samples is not None else 128
    celestial_selection, source_sha = load_ds9_selection_snapshot(
        source_path,
        samples=sampling_hint,
    )
    return convert_selection_batch(
        celestial_selection,
        event_files,
        output_dir,
        rule=projection_rule,
        context=context,
        context_resolver=context_resolver,
        source_region_sha256=source_sha,
        source_region_origin="tool-hashed-source-file",
        source_reference=str(source_path),
        source_region_path=source_path,
        representation=representation,
        inline_limit=inline_limit,
        max_components=max_components,
    )
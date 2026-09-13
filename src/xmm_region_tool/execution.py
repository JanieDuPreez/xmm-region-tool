"""Explicit execution/provenance contracts for reproducible XMM projection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

from .calibration import (
    CalibrationIdentity,
    SasProducerIdentity,
    read_calibration_identity,
    read_sas_producer_identity,
    resolve_executable,
)
from .context_suitability import ObservationRecord
from .limits import MAX_DEPTH, MAX_REGION_VERTICES, MAX_SAMPLES, integer_limit
from .model import DetectorSelection
from .runtime_producer import converter_runtime_record

PROJECTION_METHOD = "sas-esky2det-boundary/v2"
ADAPTIVE_REFINEMENT = "adaptive-detector-chord/v2"
LEGACY_REFINEMENT = "legacy-fixed-source-sampling/v1"


class SasProjectionContextError(RuntimeError):
    """Raised when a frozen SAS projection context is stale or incoherent."""


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_evidence_value(value: object) -> object:
    """Recursively snapshot JSON-like provenance evidence as immutable values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_evidence_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_evidence_value(item) for item in value)
    return value


def _plain_evidence_value(value: object) -> object:
    """Convert immutable provenance snapshots back to ordinary JSON values."""
    if isinstance(value, Mapping):
        return {
            str(key): _plain_evidence_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_plain_evidence_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ProjectionRule:
    """Approximation rule actually executed to materialise detector boundaries.

    The default is adaptive direct detector-space chord refinement. ``samples``
    is retained only for explicit legacy comparison and automatically selects the
    legacy fixed-sampling mode when supplied.
    """

    method: str = PROJECTION_METHOD
    refinement: str = ADAPTIVE_REFINEMENT
    detector_tolerance: float | None = 1.0
    max_depth: int = 12
    max_vertices: int = 4096
    samples: int | None = None

    def __post_init__(self) -> None:
        if self.method != PROJECTION_METHOD:
            raise ValueError(
                f"unsupported projection method {self.method!r}; "
                f"only {PROJECTION_METHOD!r} is implemented"
            )

        if self.samples is not None:
            if isinstance(self.samples, bool) or not isinstance(self.samples, Integral):
                raise ValueError("legacy projection samples must be an integer, not bool")
            if self.samples < 16:
                raise ValueError("legacy projection samples must be at least 16")
            object.__setattr__(self, "samples", int(self.samples))
            integer_limit(self.samples, "samples", 16, MAX_SAMPLES)
            if self.refinement == ADAPTIVE_REFINEMENT:
                object.__setattr__(self, "refinement", LEGACY_REFINEMENT)
        # Inactive controls remain non-semantic, but the value object is always valid.
        if (
            isinstance(self.detector_tolerance, bool)
            or not isinstance(self.detector_tolerance, Real)
            or not math.isfinite(self.detector_tolerance)
            or self.detector_tolerance <= 0
        ):
            raise ValueError("adaptive detector_tolerance must be a finite positive real scalar")
        object.__setattr__(self, "detector_tolerance", float(self.detector_tolerance))
        for name, minimum, maximum in (
            ("max_depth", 1, MAX_DEPTH),
            ("max_vertices", 4, MAX_REGION_VERTICES),
        ):
            object.__setattr__(
                self,
                name,
                integer_limit(getattr(self, name), name, minimum, maximum),
            )
        if self.refinement == LEGACY_REFINEMENT:
            if self.samples is None:
                raise ValueError("legacy fixed projection requires samples")
        elif self.refinement != ADAPTIVE_REFINEMENT:
            raise ValueError(f"unknown projection refinement mode {self.refinement!r}")

    @property
    def adaptive(self) -> bool:
        return self.refinement == ADAPTIVE_REFINEMENT

    def canonical_record(self) -> dict[str, object | None]:
        record: dict[str, object | None] = {
            "schema": "xmm-region-tool.projection-rule/v2",
            "method": self.method,
            "refinement": self.refinement,
        }
        if self.adaptive:
            record.update(
                {
                    "detector_tolerance": self.detector_tolerance,
                    "max_depth": self.max_depth,
                    "max_vertices": self.max_vertices,
                }
            )
        else:
            record["samples"] = self.samples
        return record


@dataclass(frozen=True)
class SasProjectionContext:
    """Frozen SAS execution state materially relevant to ``esky2det`` projection.

    ``observation_evidence`` is deliberately non-semantic. It exists to prove
    that the material calibration context belongs to the event observation,
    including ODF-independent ``cifbuild withobservationdate=yes`` workflows;
    it does not alter detector geometry identity merely by being present.
    """

    environment: Mapping[str, str] = field(repr=False)
    calibration: CalibrationIdentity
    producer: SasProducerIdentity
    esky2det_path: Path
    observation_evidence: ObservationRecord | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        environment = MappingProxyType(dict(self.environment))
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "esky2det_path", Path(self.esky2det_path).resolve())
        if self.observation_evidence is not None and not isinstance(
            self.observation_evidence, ObservationRecord
        ):
            raise TypeError("observation_evidence must be ObservationRecord or None")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        esky2det: str = "esky2det",
        observation_evidence: ObservationRecord | None = None,
    ) -> SasProjectionContext:
        env = dict(os.environ if environment is None else environment)
        calibration = read_calibration_identity(env)
        producer = read_sas_producer_identity(env, esky2det=esky2det)
        executable = resolve_executable(esky2det, env)
        return cls(
            environment=env,
            calibration=calibration,
            producer=producer,
            esky2det_path=executable,
            observation_evidence=observation_evidence,
        )

    def validate(self) -> None:
        """Verify that captured SAS/calibration identities still match execution state."""
        if not self.esky2det_path.is_file():
            raise SasProjectionContextError(
                f"captured esky2det executable no longer exists: {self.esky2det_path}"
            )

        try:
            current_calibration = read_calibration_identity(self.environment)
        except Exception as exc:
            raise SasProjectionContextError(
                f"cannot revalidate frozen SAS calibration context: {exc}"
            ) from exc

        if current_calibration.identity_sha256 != self.calibration.identity_sha256:
            raise SasProjectionContextError(
                "active SAS calibration identity no longer matches the frozen "
                "projection context"
            )

        if current_calibration.cif_file_sha256 != self.calibration.cif_file_sha256:
            raise SasProjectionContextError(
                "active SAS_CCF bytes no longer match the frozen projection context"
            )

        try:
            current_producer = read_sas_producer_identity(
                self.environment,
                esky2det=str(self.esky2det_path),
            )
        except Exception as exc:
            raise SasProjectionContextError(
                f"cannot revalidate frozen SAS producer context: {exc}"
            ) from exc

        if current_producer.identity_sha256 != self.producer.identity_sha256:
            raise SasProjectionContextError(
                "esky2det/SAS producer identity no longer matches the frozen "
                "projection context"
            )

    def relevant_environment_record(self) -> dict[str, str]:
        keys = ("SAS_CCF", "SAS_CCFPATH", "SAS_CCFFILES")
        return {key: self.environment[key] for key in keys if self.environment.get(key)}

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema": "xmm-region-tool.sas-projection-context/v2",
            "calibration_identity_sha256": self.calibration.identity_sha256,
            "producer_identity_sha256": self.producer.identity_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.canonical_record())


@dataclass(frozen=True)
class ProjectionProvenance:
    """Reproducible producer evidence for one detector-space projection."""

    event_file_sha256: str
    event_identity_sha256: str
    celestial_geometry_sha256: str
    detector_geometry_sha256: str
    context_identity_sha256: str
    calibration_identity_sha256: str
    producer_identity_sha256: str
    rule: ProjectionRule
    command: tuple[str, ...]
    relevant_environment: Mapping[str, str]
    calibration_execution_evidence: Mapping[str, object] | None = None
    commands: tuple[tuple[str, ...], ...] = ()
    refinement_diagnostics: Mapping[str, object] = field(default_factory=dict)
    converter_runtime: Mapping[str, object] = field(default_factory=converter_runtime_record)

    def __post_init__(self) -> None:
        if not isinstance(self.rule, ProjectionRule):
            raise TypeError("rule must be ProjectionRule")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(
            self,
            "commands",
            tuple(tuple(command) for command in self.commands),
        )
        object.__setattr__(
            self,
            "relevant_environment",
            MappingProxyType(dict(self.relevant_environment)),
        )

        if self.calibration_execution_evidence is not None:
            calibration_evidence = dict(self.calibration_execution_evidence)
            if calibration_evidence.get("identity_sha256") != self.calibration_identity_sha256:
                raise ValueError(
                    "calibration execution evidence does not match "
                    "calibration_identity_sha256"
                )
            object.__setattr__(
                self,
                "calibration_execution_evidence",
                _freeze_evidence_value(calibration_evidence),
            )

        runtime = dict(self.converter_runtime)
        if runtime.get("schema") != "xmm-region-tool.converter-runtime/v1":
            raise ValueError("converter_runtime has an unsupported or missing schema")
        object.__setattr__(
            self,
            "converter_runtime",
            _freeze_evidence_value(runtime),
        )
        object.__setattr__(
            self,
            "refinement_diagnostics",
            _freeze_evidence_value(dict(self.refinement_diagnostics)),
        )

    def calibration_execution_evidence_record(self) -> dict[str, object] | None:
        """Return the immutable projection-time calibration evidence as JSON values."""
        if self.calibration_execution_evidence is None:
            return None
        record = _plain_evidence_value(self.calibration_execution_evidence)
        if not isinstance(record, dict):
            raise RuntimeError("internal calibration execution evidence is not a mapping")
        return record

    def converter_runtime_record(self) -> dict[str, object]:
        """Return the immutable projection-time converter/runtime evidence."""
        record = _plain_evidence_value(self.converter_runtime)
        if not isinstance(record, dict):
            raise RuntimeError("internal converter runtime evidence is not a mapping")
        return record

    def canonical_projection_record(self) -> dict[str, object]:
        return {
            "schema": "xmm-region-tool.projection/v3",
            "event_file_sha256": self.event_file_sha256,
            "event_identity_sha256": self.event_identity_sha256,
            "celestial_geometry_sha256": self.celestial_geometry_sha256,
            "context_identity_sha256": self.context_identity_sha256,
            "calibration_identity_sha256": self.calibration_identity_sha256,
            "producer_identity_sha256": self.producer_identity_sha256,
            "rule": self.rule.canonical_record(),
        }

    @property
    def projection_identity_sha256(self) -> str:
        return _canonical_sha256(self.canonical_projection_record())

    def evidence_record(self) -> dict[str, object]:
        commands = self.commands or ((self.command,) if self.command else ())
        return {
            **self.canonical_projection_record(),
            "projection_identity_sha256": self.projection_identity_sha256,
            "detector_geometry_sha256": self.detector_geometry_sha256,
            "converter_runtime": self.converter_runtime_record(),
            "command": list(self.command),
            "commands": [list(command) for command in commands],
            "relevant_environment": dict(self.relevant_environment),
            "refinement_diagnostics": _plain_evidence_value(self.refinement_diagnostics),
        }


@dataclass(frozen=True)
class ProjectionResult:
    """Neutral detector geometry plus the evidence that produced it."""

    selection: DetectorSelection
    provenance: ProjectionProvenance

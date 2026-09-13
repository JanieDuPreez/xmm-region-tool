"""Supported public SAS projection-context constructor.

The implementation context type remains available from :mod:`execution` for
low-level/debug use.  The top-level package constructor tightens named
executable resolution when a caller supplies an explicit environment mapping:
that mapping must itself contain PATH, so executable selection cannot silently
fall back to an uncaptured ambient PATH.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .execution import SasProjectionContext as _SasProjectionContext


class SasProjectionContext(_SasProjectionContext):
    """Public context with deterministic explicit-environment executable lookup."""

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        esky2det: str = "esky2det",
        observation_evidence=None,
    ) -> "SasProjectionContext":
        if environment is not None:
            executable = Path(esky2det).expanduser()
            name_based = not executable.is_absolute() and executable.parent == Path(".")
            if name_based and not str(environment.get("PATH", "")).strip():
                raise ValueError(
                    "explicit SAS environment must include PATH for name-based esky2det "
                    "resolution; ambient PATH fallback is not part of the captured context"
                )
        base = super().from_environment(
            environment,
            esky2det=esky2det,
            observation_evidence=observation_evidence,
        )
        return cls(
            environment=base.environment,
            calibration=base.calibration,
            producer=base.producer,
            esky2det_path=base.esky2det_path,
            observation_evidence=base.observation_evidence,
        )


__all__ = ["SasProjectionContext"]

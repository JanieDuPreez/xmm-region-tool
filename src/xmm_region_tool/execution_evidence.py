"""Cross-record validation for historical esky2det invocation evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .calibration import SasProducerIdentity
from .execution import PROJECTION_METHOD, ProjectionProvenance
from .provenance import file_sha256


class ExecutionEvidenceError(ValueError):
    """Raised when claimed esky2det argv history contradicts bound evidence."""


_EXPECTED_MODE = {
    "datastyle": "set",
    "calinfostyle": "set",
    "outunit": "det",
    "checkfov": "no",
    "witherrorcol": "no",
    "withouttab": "no",
}
_EXPECTED_KEYS = {*_EXPECTED_MODE, "intab", "calinfoset"}


def _command_parameters(argv: Sequence[str]) -> dict[str, str]:
    if len(argv) != 1 + len(_EXPECTED_KEYS):
        raise ExecutionEvidenceError(
            "esky2det command does not contain the canonical projection argument set"
        )
    params: dict[str, str] = {}
    for token in argv[1:]:
        if not isinstance(token, str) or "=" not in token:
            raise ExecutionEvidenceError("esky2det command contains a non-canonical argument")
        key, value = token.split("=", 1)
        if not key or key in params:
            raise ExecutionEvidenceError("esky2det command contains duplicate/empty parameters")
        params[key] = value
    if set(params) != _EXPECTED_KEYS:
        raise ExecutionEvidenceError(
            "esky2det command parameter names do not match the projection contract"
        )
    for key, expected in _EXPECTED_MODE.items():
        if params[key] != expected:
            raise ExecutionEvidenceError(
                f"esky2det command {key}={params[key]!r} contradicts required {expected!r}"
            )
    if not params["intab"]:
        raise ExecutionEvidenceError("esky2det command has no temporary intab evidence")
    return params


def _validate_executable(argv0: object, producer: SasProducerIdentity) -> None:
    if not isinstance(argv0, str) or not argv0:
        raise ExecutionEvidenceError("esky2det command has no executable")
    executable = Path(argv0).expanduser()
    if executable.name != producer.esky2det_name:
        raise ExecutionEvidenceError(
            "esky2det command task name contradicts the bound SAS producer"
        )
    if executable.is_file() and producer.esky2det_sha256 is not None:
        if file_sha256(executable) != producer.esky2det_sha256:
            raise ExecutionEvidenceError(
                "esky2det command executable bytes contradict the bound SAS producer"
            )


def _validate_calinfoset(value: str, expected_event_sha256: str) -> None:
    event = Path(value).expanduser().resolve()
    if not event.is_file():
        raise ExecutionEvidenceError(
            "historical esky2det calinfoset is unavailable at durable promotion time"
        )
    if file_sha256(event) != expected_event_sha256:
        raise ExecutionEvidenceError(
            "esky2det command calinfoset bytes contradict the bound projection event"
        )


def validate_projection_invocations(
    provenance: ProjectionProvenance,
    producer: SasProducerIdentity,
) -> None:
    """Validate current ``sas-esky2det-boundary/v2`` argv history at promotion time.

    Temporary ``intab`` locations remain supporting evidence and are not required
    to survive projection. The event ``calinfoset`` and executable, however, are
    checked while durable evidence is being promoted, when the projection-owned
    event artifact should still be available. Later path relocation therefore
    does not make either path semantic identity material.
    """
    if provenance.rule.method != PROJECTION_METHOD:
        raise ExecutionEvidenceError(
            f"unsupported projection method for invocation validation: {provenance.rule.method!r}"
        )
    primary = tuple(provenance.command)
    history = tuple(tuple(item) for item in provenance.commands)
    if not history:
        history = (primary,) if primary else ()
    if not primary or not history:
        raise ExecutionEvidenceError("projection has no esky2det invocation history")
    if primary != history[0]:
        raise ExecutionEvidenceError(
            "projection primary command does not equal the first complete command-history entry"
        )

    for argv in history:
        if not argv:
            raise ExecutionEvidenceError("projection command history contains an empty command")
        _validate_executable(argv[0], producer)
        params = _command_parameters(argv)
        _validate_calinfoset(params["calinfoset"], provenance.event_file_sha256)

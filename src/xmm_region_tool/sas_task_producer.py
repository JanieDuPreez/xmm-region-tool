"""Exact producer identity for SAS task executables used as scientific validators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .calibration import (
    CalibrationIdentityError,
    _canonical_sas_version,
    _canonical_task_version,
    _combined_output,
    _file_sha256,
    resolve_executable,
)
from .subprocess_capture import diagnostic_summary, run_bounded


class SasTaskProducerError(RuntimeError):
    """Raised when an exact SAS task producer identity cannot be established."""


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SasTaskProducerIdentity:
    """Path-independent identity of one exact SAS task executable generation."""

    task_name: str
    task_version: str
    executable_sha256: str
    sas_version: str | None

    def __post_init__(self) -> None:
        name = str(self.task_name).strip()
        version = str(self.task_version).strip()
        digest = str(self.executable_sha256).strip().lower()
        if not name:
            raise ValueError("SAS task producer name must be non-empty")
        if not version:
            raise ValueError("SAS task producer version must be non-empty")
        if len(digest) != 64:
            raise ValueError("SAS task producer executable SHA256 must contain 64 hex characters")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("SAS task producer executable SHA256 is not hexadecimal") from exc
        object.__setattr__(self, "task_name", name)
        object.__setattr__(self, "task_version", version)
        object.__setattr__(self, "executable_sha256", digest)
        if self.sas_version is not None:
            sas_version = str(self.sas_version).strip()
            object.__setattr__(self, "sas_version", sas_version or None)

    def canonical_record(self) -> dict[str, object | None]:
        return {
            "schema": "xmm-region-tool.sas-task-producer/v1",
            "task_name": self.task_name,
            "task_version": self.task_version,
            "executable_sha256": self.executable_sha256,
            "sas_version": self.sas_version,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.canonical_record())


@dataclass(frozen=True)
class SasTaskProducerCapture:
    """Producer identity plus the exact resolved executable path used for execution."""

    identity: SasTaskProducerIdentity
    executable_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "executable_path", Path(self.executable_path).resolve())

    def evidence_record(self) -> dict[str, object | None]:
        return {
            **self.identity.canonical_record(),
            "identity_sha256": self.identity.identity_sha256,
            "executable_path": str(self.executable_path),
        }


def _capture_sas_release(environment: Mapping[str, str] | None) -> str | None:
    try:
        sasversion = resolve_executable("sasversion", environment)
    except CalibrationIdentityError:
        return None

    result = run_bounded([str(sasversion)], environment=environment)
    if result.stdout_capture.truncated or result.stderr_capture.truncated:
        raise SasTaskProducerError(
            "cannot determine SAS release from truncated sasversion output; "
            "the executable exceeded the bounded producer-probe output limit"
        )
    if result.returncode != 0:
        return None
    return _canonical_sas_version(_combined_output(result))


def capture_sas_task_producer(
    executable: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> SasTaskProducerCapture:
    """Capture stable task version/release identity and exact executable bytes."""
    try:
        resolved = resolve_executable(executable, environment)
    except CalibrationIdentityError as exc:
        raise SasTaskProducerError(str(exc)) from exc

    digest_before = _file_sha256(resolved)
    result = run_bounded([str(resolved), "-v"], environment=environment)
    if result.stdout_capture.truncated or result.stderr_capture.truncated:
        raise SasTaskProducerError(
            f"cannot determine {resolved.name} version from truncated diagnostic output; "
            "the executable exceeded the bounded producer-probe output limit"
        )
    if result.returncode != 0:
        details = diagnostic_summary(result.stderr or result.stdout)
        raise SasTaskProducerError(
            f"cannot determine {resolved.name} version (exit {result.returncode}): {details}"
        )
    output = _combined_output(result)
    if not output:
        raise SasTaskProducerError(f"{resolved.name} -v returned no version text")
    try:
        task_version = _canonical_task_version(output, executable=resolved.name)
    except CalibrationIdentityError as exc:
        raise SasTaskProducerError(str(exc)) from exc

    digest_after = _file_sha256(resolved)
    if digest_after != digest_before:
        raise SasTaskProducerError(
            f"{resolved.name} executable bytes changed while producer identity was being captured"
        )

    identity = SasTaskProducerIdentity(
        task_name=resolved.name,
        task_version=task_version,
        executable_sha256=digest_after,
        sas_version=_capture_sas_release(environment),
    )
    return SasTaskProducerCapture(identity=identity, executable_path=resolved)


def validate_sas_task_producer_capture(capture: SasTaskProducerCapture) -> None:
    """Fail if the resolved executable no longer matches the captured exact bytes."""
    path = capture.executable_path
    if not path.is_file():
        raise SasTaskProducerError(
            f"captured SAS task executable no longer exists: {path}"
        )
    current_sha256 = _file_sha256(path)
    if current_sha256 != capture.identity.executable_sha256:
        raise SasTaskProducerError(
            f"captured {capture.identity.task_name} executable bytes changed after identity capture"
        )


__all__ = [
    "SasTaskProducerCapture",
    "SasTaskProducerError",
    "SasTaskProducerIdentity",
    "capture_sas_task_producer",
    "validate_sas_task_producer_capture",
]

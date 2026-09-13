"""Bounded capture for diagnostics emitted by package-owned subprocesses.

External SAS tasks are allowed to run normally, but their stdout/stderr must not
be an unbounded memory channel into the Python process. This helper directs both
streams to temporary files while the child is running, then retains only a
bounded head/tail diagnostic window with byte-count/truncation metadata. It
intentionally does not impose a runtime timeout; scientific task duration is a
separate policy question.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import BinaryIO

DIAGNOSTIC_STREAM_LIMIT = 64 * 1024
_DIAGNOSTIC_HALF = DIAGNOSTIC_STREAM_LIMIT // 2


@dataclass(frozen=True)
class CapturedDiagnostic:
    """One bounded subprocess output stream."""

    text: str
    total_bytes: int
    retained_bytes: int
    truncated: bool

    def evidence_record(self) -> dict[str, object]:
        return {
            "total_bytes": self.total_bytes,
            "retained_bytes": self.retained_bytes,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class BoundedProcessResult:
    """Completed subprocess plus bounded stdout/stderr evidence."""

    args: tuple[str, ...]
    returncode: int
    stdout_capture: CapturedDiagnostic
    stderr_capture: CapturedDiagnostic

    @property
    def stdout(self) -> str:
        return self.stdout_capture.text

    @property
    def stderr(self) -> str:
        return self.stderr_capture.text


def _bounded_bytes(payload: bytes) -> CapturedDiagnostic:
    total = len(payload)
    truncated = total > DIAGNOSTIC_STREAM_LIMIT
    if truncated:
        head = payload[:_DIAGNOSTIC_HALF]
        tail = payload[-_DIAGNOSTIC_HALF:]
        omitted = total - len(head) - len(tail)
        marker = f"\n...[xmm-region-tool truncated {omitted} bytes]...\n".encode()
        display = head + marker + tail
        retained = len(head) + len(tail)
    else:
        display = payload
        retained = total
    text = display.decode("utf-8", errors="replace")
    # Match subprocess text mode's universal-newline behavior for ordinary output.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return CapturedDiagnostic(
        text=text,
        total_bytes=total,
        retained_bytes=retained,
        truncated=truncated,
    )


def _capture_file(stream: BinaryIO) -> CapturedDiagnostic:
    stream.flush()
    stream.seek(0, 2)
    total = stream.tell()
    if total <= DIAGNOSTIC_STREAM_LIMIT:
        stream.seek(0)
        return _bounded_bytes(stream.read())

    stream.seek(0)
    head = stream.read(_DIAGNOSTIC_HALF)
    stream.seek(-_DIAGNOSTIC_HALF, 2)
    tail = stream.read(_DIAGNOSTIC_HALF)
    omitted = total - len(head) - len(tail)
    marker = f"\n...[xmm-region-tool truncated {omitted} bytes]...\n".encode()
    text = (head + marker + tail).decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return CapturedDiagnostic(
        text=text,
        total_bytes=total,
        retained_bytes=len(head) + len(tail),
        truncated=True,
    )


def _fallback_capture(value: object, stream: BinaryIO) -> CapturedDiagnostic:
    """Support test runners that return stdout/stderr instead of writing handles."""
    if isinstance(value, str):
        return _bounded_bytes(value.encode("utf-8"))
    if isinstance(value, bytes):
        return _bounded_bytes(value)
    return _capture_file(stream)


def run_bounded(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> BoundedProcessResult:
    """Run *command* without retaining unbounded child diagnostics in memory."""
    argv = tuple(str(value) for value in command)
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        result = subprocess.run(
            argv,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
            env=None if environment is None else dict(environment),
        )
        stdout_capture = _fallback_capture(getattr(result, "stdout", None), stdout_file)
        stderr_capture = _fallback_capture(getattr(result, "stderr", None), stderr_file)
    return BoundedProcessResult(
        args=argv,
        returncode=int(result.returncode),
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
    )


def diagnostic_summary(text: str, *, limit: int = 4096) -> str:
    """Return a bounded message-safe excerpt from already bounded diagnostic text."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    half = limit // 2
    return (
        stripped[:half]
        + "\n...[diagnostic excerpt shortened]...\n"
        + stripped[-half:]
    )


__all__ = [
    "BoundedProcessResult",
    "CapturedDiagnostic",
    "DIAGNOSTIC_STREAM_LIMIT",
    "diagnostic_summary",
    "run_bounded",
]

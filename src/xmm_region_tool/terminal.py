"""Safe rendering of untrusted text to interactive terminals.

This module affects display only. It must not be used to canonicalize manifests,
provenance, paths, subprocess arguments, or copy/paste command values.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import TextIO


def safe_terminal_text(value: object) -> str:
    """Render C0/C1 controls visibly instead of emitting terminal control bytes."""
    text = str(value)
    rendered: list[str] = []
    named = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
        "\v": "\\v",
    }
    for character in text:
        replacement = named.get(character)
        if replacement is not None:
            rendered.append(replacement)
            continue
        codepoint = ord(character)
        if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            if codepoint <= 0xFF:
                rendered.append(f"\\x{codepoint:02x}")
            else:
                rendered.append(f"\\u{codepoint:04x}")
            continue
        rendered.append(character)
    return "".join(rendered)


def safe_terminal_lines(lines: list[str] | tuple[str, ...]) -> str:
    """Sanitize each logical line while preserving trusted report line breaks."""
    return "\n".join(safe_terminal_text(line) for line in lines)


class SafeTerminalWriter:
    """Text stream proxy that sanitizes payload writes but preserves print line endings."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        # CPython's print() writes its ``end`` token separately from object text.
        # Preserve those trusted structural line endings while escaping newlines
        # embedded inside an untrusted object/path/error string.
        if text in {"\n", "\r\n"}:
            return self._stream.write(text)
        rendered = safe_terminal_text(text)
        return self._stream.write(rendered)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextmanager
def safe_terminal_streams(stdout: TextIO, stderr: TextIO):
    """Redirect stdout/stderr through display-only control-character escaping."""
    with redirect_stdout(SafeTerminalWriter(stdout)), redirect_stderr(SafeTerminalWriter(stderr)):
        yield


__all__ = [
    "SafeTerminalWriter",
    "safe_terminal_lines",
    "safe_terminal_streams",
    "safe_terminal_text",
]

"""Filesystem-safe automatic naming and namespace-preflight helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a planned filesystem namespace is not safe."""


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename_component(value: str, *, fallback: str | None = None) -> str:
    """Return one deterministic path component with no directory semantics.

    Metadata-derived automatic names are reduced to ``[A-Za-z0-9_.-]`` and
    leading/trailing punctuation that could be confused with ``.``/``..`` is
    stripped.  The result is always a single component; callers may provide a
    fallback for values such as an empty user-authored source stem.
    """
    if not isinstance(value, str):
        raise TypeError(f"filename component must be str, got {type(value).__name__}")
    result = _SAFE_COMPONENT.sub("-", value.strip()).strip("-.")
    if result:
        return result
    if fallback is not None:
        return safe_filename_component(fallback)
    raise PathSafetyError(f"cannot form a safe filename component from {value!r}")


def resolved_descendant(
    root: str | Path,
    candidate: str | Path,
    *,
    role: str,
) -> Path:
    """Resolve an automatic path and require that it stays beneath ``root``.

    This is a defense-in-depth planning check.  It does not replace the later
    symlink-safe publication contract; it prevents automatically constructed
    paths from gaining traversal semantics before any output is written.
    """
    resolved_root = Path(root).expanduser().resolve()
    resolved_candidate = Path(candidate).expanduser().resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise PathSafetyError(
            f"automatic {role} escapes output root {resolved_root}: {resolved_candidate}"
        ) from exc
    if resolved_candidate == resolved_root:
        raise PathSafetyError(
            f"automatic {role} resolves to output directory rather than a file: {resolved_root}"
        )
    return resolved_candidate


def paths_alias(first: str | Path, second: str | Path) -> bool:
    """Return whether two pathnames identify the same planned or existing file.

    Resolved-path equality covers ordinary paths and symlink aliases.  Existing
    paths additionally use ``samefile`` so hard links are detected even when
    their resolved names differ.
    """
    left = Path(first).expanduser()
    right = Path(second).expanduser()
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _casefold_path_key(path: Path) -> str:
    """Return the conservative ASCII-case-insensitive namespace key for a path."""
    return str(path).casefold()


def validate_output_namespace(
    outputs: Sequence[tuple[str, str | Path]],
    protected: Sequence[tuple[str, str | Path]],
) -> tuple[tuple[str, Path], ...]:
    """Require all generated outputs to be mutually distinct from protected inputs.

    This is an invocation-planning guard.  It deliberately permits replacing a
    previous generation at the *same logical output path* on a later run, while
    rejecting two different output roles in the current invocation that alias
    each other.  Existing hard-link/symlink aliases are treated as collisions.

    Planned names are also compared under ``casefold()``.  Automatic filename
    components are ASCII-only, so this catches path pairs that would collapse
    on case-insensitive filesystems without rewriting any explicit user path.
    The conservative check is applied even when the current filesystem is case
    sensitive so one invocation has the same safety contract across supported
    platforms.
    """
    normalized_outputs = tuple(
        (role, Path(path).expanduser().resolve()) for role, path in outputs
    )
    normalized_protected = tuple(
        (role, Path(path).expanduser().resolve()) for role, path in protected
    )

    for index, (role, path) in enumerate(normalized_outputs):
        path_key = _casefold_path_key(path)
        for other_role, other_path in normalized_outputs[index + 1 :]:
            if paths_alias(path, other_path):
                raise PathSafetyError(
                    f"generated {role} aliases generated {other_role}: {path}"
                )
            if path_key == _casefold_path_key(other_path):
                raise PathSafetyError(
                    "generated output paths collide under case-insensitive filesystem semantics: "
                    f"{role}={path}; {other_role}={other_path}"
                )
        for protected_role, protected_path in normalized_protected:
            if paths_alias(path, protected_path):
                raise PathSafetyError(
                    f"generated {role} aliases protected {protected_role}: {path}"
                )
            if path_key == _casefold_path_key(protected_path):
                raise PathSafetyError(
                    "generated output path collides with protected input under case-insensitive "
                    f"filesystem semantics: {role}={path}; {protected_role}={protected_path}"
                )
    return normalized_outputs


def validate_distinct_inputs(
    inputs: Sequence[tuple[str, str | Path]],
) -> tuple[tuple[str, Path], ...]:
    """Reject duplicate/aliased scientific inputs before any work begins."""
    normalized = tuple((role, Path(path).expanduser().resolve()) for role, path in inputs)
    for index, (role, path) in enumerate(normalized):
        for other_role, other_path in normalized[index + 1 :]:
            if paths_alias(path, other_path):
                raise PathSafetyError(
                    f"{role} aliases {other_role}: {path}"
                )
    return normalized

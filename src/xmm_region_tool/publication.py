"""Fail-atomic, symlink-safe publication helpers for supported outputs."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


class PublicationError(OSError):
    """Raised when a staged output set cannot be published safely."""


def lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute normalized path without following the final entry.

    Namespace/collision checks may use resolved paths separately. Publication
    must retain the lexical final directory entry so an existing destination
    symlink is replaced as an entry rather than followed to its target.
    """
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def temporary_sibling(final_path: str | Path) -> Path:
    """Create one secure temporary sibling preserving the final suffix."""
    final = lexical_absolute_path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    suffix = final.suffix
    prefix = f".{final.stem}.xmm-region-stage-"
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=final.parent)
    os.close(descriptor)
    return Path(name)


def _backup_sibling(final: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final.name}.xmm-region-backup-",
        dir=final.parent,
    )
    os.close(descriptor)
    return Path(name)


def discard_staged(paths: Iterable[str | Path]) -> None:
    """Remove only staging artifacts created by the current attempt."""
    for value in paths:
        path = lexical_absolute_path(value)
        try:
            if _lexists(path):
                path.unlink()
        except OSError:
            pass


def publish_staged_files(
    pairs: Iterable[tuple[str | Path, str | Path]],
) -> tuple[Path, ...]:
    """Atomically replace a set of final directory entries with rollback.

    Each staged file must already contain complete validated bytes. Existing
    finals are moved to private siblings immediately before publication and are
    restored if any later replace fails. ``os.replace`` acts on directory
    entries, so an existing or raced-in destination symlink is never followed.

    If restoring a previous generation fails, its backup is deliberately left
    in place as a recovery artifact and the raised :class:`PublicationError`
    identifies both the intended final path and retained recovery path. A
    backup is never deleted merely because rollback was attempted.
    """
    normalized = [
        (lexical_absolute_path(stage), lexical_absolute_path(final))
        for stage, final in pairs
    ]
    finals = [final for _stage, final in normalized]
    if len(set(finals)) != len(finals):
        raise PublicationError("staged publication contains duplicate final paths")
    for stage, final in normalized:
        if not stage.is_file():
            raise PublicationError(f"staged output does not exist: {stage}")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.is_dir() and not final.is_symlink():
            raise PublicationError(f"final output is an existing directory: {final}")

    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for _stage, final in normalized:
            if _lexists(final):
                backup = _backup_sibling(final)
                os.replace(final, backup)
                backups[final] = backup

        for stage, final in normalized:
            os.replace(stage, final)
            published.append(final)
    except OSError as exc:
        for final in reversed(published):
            try:
                if _lexists(final):
                    final.unlink()
            except OSError:
                pass

        recovery_failures: list[tuple[Path, Path, OSError]] = []
        for final, backup in backups.items():
            if not _lexists(backup):
                continue
            try:
                os.replace(backup, final)
            except OSError as restore_exc:
                recovery_failures.append((final, backup, restore_exc))

        discard_staged(stage for stage, _final in normalized)

        message = f"could not atomically publish output set: {exc}"
        if recovery_failures:
            details = "; ".join(
                f"{final} remains recoverable at {backup} ({restore_exc})"
                for final, backup, restore_exc in recovery_failures
            )
            message += f"; rollback could not restore previous output(s): {details}"
        raise PublicationError(message) from exc

    discard_staged(backups.values())
    return tuple(finals)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write text through a temporary sibling then atomically replace the entry."""
    final = lexical_absolute_path(path)
    stage = temporary_sibling(final)
    try:
        stage.write_text(text, encoding=encoding)
        publish_staged_files(((stage, final),))
    except Exception:
        discard_staged((stage,))
        raise
    return final

"""Conservative path policy for unquoted SAS dataset/blockspec parameters.

The initial release does not attempt to quote or escape arbitrary filesystem
paths for SAS DAL/selectlib parsers. Any path that is interpolated into an
unquoted SAS dataset/blockspec parameter must therefore use only the ordinary
ASCII path characters whose literal interpretation is already established by
the retained real-SAS evidence.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


class SasPathError(ValueError):
    """Raised when a filesystem path is outside the initial SAS-safe grammar."""


_SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
_SAFE_DESCRIPTION = "ASCII letters, digits, '.', '_', '-', and '/'"


def _validate_text(path: Path, *, role: str) -> Path:
    text = str(path)
    if not _SAFE_PATH.fullmatch(text):
        raise SasPathError(
            f"{role} path is not safe for unquoted SAS dataset/blockspec syntax: {text!r}; "
            f"the initial release accepts only {_SAFE_DESCRIPTION}"
        )
    return path


def validate_sas_safe_resolved_path(path: str | Path, *, role: str) -> Path:
    """Resolve *path* and require the initial unquoted SAS-safe path grammar."""
    return _validate_text(Path(path).expanduser().resolve(), role=role)


def validate_sas_safe_lexical_path(path: str | Path, *, role: str) -> Path:
    """Validate the absolute lexical path SAS will receive without following its final entry.

    This is used at atomic-publication boundaries where resolving a pre-existing
    destination symlink would validate the symlink target rather than the path
    that will exist after the symlink directory entry is replaced.
    """
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    return _validate_text(lexical, role=role)


def validate_sas_temporary_root(*, role: str = "SAS temporary dataset") -> Path:
    """Return the active temporary root only when SAS-safe package staging is possible.

    ``tempfile.TemporaryDirectory`` derives its directory from this root. Its
    package-defined prefixes and generated suffixes use only characters allowed
    by the same grammar, so validating the root before a SAS-facing temporary
    dataset is created prevents an unsafe ``TMPDIR`` from silently reaching SAS.
    """
    return validate_sas_safe_resolved_path(tempfile.gettempdir(), role=role)


__all__ = [
    "SasPathError",
    "validate_sas_safe_lexical_path",
    "validate_sas_safe_resolved_path",
    "validate_sas_temporary_root",
]

"""Capture the SAS calibration and task context used for sky-to-detector conversion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .subprocess_capture import BoundedProcessResult, diagnostic_summary, run_bounded


class CalibrationIdentityError(ValueError):
    """Raised when the active SAS calibration context cannot be identified."""


# Fields that define CAL indexing/selection semantics.  The v3 semantic hash is
# deliberately narrower than the complete CALINDEX row: container/audit fields
# such as creation/submission dates, file size, MD5 and creator are retained for
# validation/evidence but do not create a second scientific identity once the
# selected constituent bytes are independently SHA256-bound.
_CIF_SEMANTIC_COLUMNS = (
    "TELESCOP",
    "SCOPE",
    "TYPEID",
    "ISSUE",
    "VALDATE",
    "VALDATE-END",
    "FNAME",
    "EXTSEQU",
    "EXTSEQID",
)

_CIF_AUDIT_COLUMNS = (
    "DATE",
    "FSIZE",
    "SUBDATE",
    "MD5",
    "CREATOR",
)

_CIF_COLUMNS = _CIF_SEMANTIC_COLUMNS + _CIF_AUDIT_COLUMNS


def _canonical_hex_digest(value: object, *, length: int, description: str) -> str:
    if not isinstance(value, str):
        raise CalibrationIdentityError(f"{description} must be a hexadecimal digest string")
    digest = value.strip().lower()
    if len(digest) != length:
        raise CalibrationIdentityError(
            f"{description} must contain exactly {length} hexadecimal characters"
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise CalibrationIdentityError(f"{description} is not hexadecimal") from exc
    return digest


def _canonical_ccf_name(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise CalibrationIdentityError(f"{description} name must be a string")
    name = value.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise CalibrationIdentityError(
            f"{description} name must be one non-empty CCF basename"
        )
    return name


@dataclass(frozen=True)
class CalibrationReplacement:
    """One CCF constituent supplied through SAS_CCFFILES."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _canonical_ccf_name(self.name, description="calibration replacement"),
        )
        object.__setattr__(
            self,
            "sha256",
            _canonical_hex_digest(
                self.sha256,
                length=64,
                description="calibration replacement SHA256",
            ),
        )

    def canonical_record(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class CalibrationConstituent:
    """Exact bytes of one CCF constituent selected by the active CIF."""

    name: str
    sha256: str
    cif_md5: str | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _canonical_ccf_name(self.name, description="calibration constituent"),
        )
        object.__setattr__(
            self,
            "sha256",
            _canonical_hex_digest(
                self.sha256,
                length=64,
                description="calibration constituent SHA256",
            ),
        )
        if self.cif_md5 is not None:
            object.__setattr__(
                self,
                "cif_md5",
                _canonical_hex_digest(
                    self.cif_md5,
                    length=32,
                    description="calibration constituent CIF MD5",
                ),
            )
        if self.size is not None:
            if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
                raise CalibrationIdentityError(
                    "calibration constituent size must be a non-negative integer or None"
                )
            object.__setattr__(self, "size", int(self.size))

    def canonical_record(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}

    def evidence_record(self) -> dict[str, object | None]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "cif_md5": self.cif_md5,
            "size": self.size,
        }


@dataclass(frozen=True)
class CalibrationIdentity:
    """Path-independent identity of the active SAS calibration selection."""

    cif_path: Path
    cif_file_sha256: str
    calindex_sha256: str
    replacements: tuple[CalibrationReplacement, ...]
    ccf_search_path: tuple[Path, ...]
    constituents: tuple[CalibrationConstituent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cif_path", Path(self.cif_path).expanduser().resolve())
        object.__setattr__(
            self,
            "cif_file_sha256",
            _canonical_hex_digest(
                self.cif_file_sha256,
                length=64,
                description="exact CIF SHA256",
            ),
        )
        object.__setattr__(
            self,
            "calindex_sha256",
            _canonical_hex_digest(
                self.calindex_sha256,
                length=64,
                description="CALINDEX SHA256",
            ),
        )

        replacements = tuple(self.replacements)
        if not all(isinstance(item, CalibrationReplacement) for item in replacements):
            raise CalibrationIdentityError(
                "calibration replacements must contain CalibrationReplacement records"
            )
        replacement_names = [item.name for item in replacements]
        if len(set(replacement_names)) != len(replacement_names):
            raise CalibrationIdentityError("calibration replacement basenames must be unique")
        object.__setattr__(self, "replacements", replacements)

        constituents = tuple(self.constituents)
        if not all(isinstance(item, CalibrationConstituent) for item in constituents):
            raise CalibrationIdentityError(
                "calibration constituents must contain CalibrationConstituent records"
            )
        constituent_names = [item.name for item in constituents]
        if len(set(constituent_names)) != len(constituent_names):
            raise CalibrationIdentityError("calibration constituent basenames must be unique")
        object.__setattr__(
            self,
            "constituents",
            tuple(sorted(constituents, key=lambda item: item.name)),
        )

        object.__setattr__(
            self,
            "ccf_search_path",
            tuple(Path(path).expanduser().resolve() for path in self.ccf_search_path),
        )

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema": "xmm-region-tool.calibration-identity/v2",
            "calindex_sha256": self.calindex_sha256,
            "constituents": [item.canonical_record() for item in self.constituents],
            "replacements": [item.canonical_record() for item in self.replacements],
        }

    def evidence_record(self) -> dict[str, object]:
        return {
            **self.canonical_record(),
            "identity_sha256": self.identity_sha256,
            "cif_file_sha256": self.cif_file_sha256,
            "constituent_evidence": [item.evidence_record() for item in self.constituents],
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.canonical_record())


@dataclass(frozen=True)
class SasProducerIdentity:
    """Stable identity of the SAS producer used for detector geometry.

    Version fields contain canonical release/build identifiers only. Volatile SAS
    execution timestamps and ambient environment dumps are deliberately excluded
    from the semantic producer identity.
    """

    esky2det_version: str
    sas_version: str | None
    esky2det_sha256: str | None = None
    esky2det_name: str = "esky2det"

    def canonical_record(self) -> dict[str, object | None]:
        return {
            "schema": "xmm-region-tool.sas-producer/v3",
            "esky2det_name": self.esky2det_name,
            "esky2det_version": self.esky2det_version,
            "esky2det_sha256": self.esky2det_sha256,
            "sas_version": self.sas_version,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.canonical_record())


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path, algorithm: str) -> str:
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    else:
        digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return _file_digest(path, "sha256")


def _json_value(value: Any) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise CalibrationIdentityError("CALINDEX contains a non-finite numeric value")
        return value
    return str(value).strip()


def _semantic_row_sort_key(row: Mapping[str, object]) -> str:
    """Canonical order for CALINDEX rows independent of FITS physical row order."""
    return json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _read_calindex(cif_path: Path) -> tuple[str, tuple[dict[str, object], ...]]:
    try:
        hdus = fits.open(cif_path, memmap=True)
    except Exception as exc:
        raise CalibrationIdentityError(f"cannot open SAS_CCF calibration index {cif_path}: {exc}") from exc
    try:
        if "CALINDEX" not in hdus:
            raise CalibrationIdentityError(f"SAS_CCF has no CALINDEX table: {cif_path}")
        table = hdus["CALINDEX"]
        if table.data is None:
            raise CalibrationIdentityError(f"CALINDEX table contains no rows: {cif_path}")
        names = {str(name).upper(): str(name) for name in table.columns.names}
        missing = [name for name in ("SCOPE", "TYPEID", "ISSUE", "FNAME") if name not in names]
        if missing:
            raise CalibrationIdentityError(
                "CALINDEX is missing identity columns: " + ", ".join(missing)
            )

        semantic_rows: list[dict[str, object]] = []
        resolution_rows: list[dict[str, object]] = []
        for row in table.data:
            semantic_record: dict[str, object] = {}
            resolution_record: dict[str, object] = {}
            for name in _CIF_COLUMNS:
                actual = names.get(name)
                if actual is None:
                    continue
                value = _json_value(row[actual])
                resolution_record[name] = value
                if name not in _CIF_SEMANTIC_COLUMNS:
                    continue
                if name == "FNAME" and isinstance(value, str):
                    semantic_record[name] = Path(value).name
                else:
                    semantic_record[name] = value
            semantic_rows.append(semantic_record)
            resolution_rows.append(resolution_record)

        # Current SAS cifdiff treats CalIndex as a constituent collection and
        # explicitly sorts before comparing.  Keep harmless FITS row ordering out
        # of the scientific identity while preserving duplicate semantic rows.
        semantic_rows.sort(key=_semantic_row_sort_key)
        semantic = _canonical_sha256(
            {
                "schema": "xmm-region-tool.calindex/v3",
                "rows": semantic_rows,
            }
        )
        return semantic, tuple(resolution_rows)
    finally:
        hdus.close()


def _resolve_cif(environment: Mapping[str, str]) -> Path:
    raw = environment.get("SAS_CCF", "").strip()
    if not raw:
        raise CalibrationIdentityError(
            "SAS_CCF is not set; esky2det requires an active Calibration Index File"
        )
    path = Path(raw).expanduser().resolve()
    if path.is_dir():
        path = path / "ccf.cif"
    if not path.is_file():
        raise CalibrationIdentityError(f"SAS_CCF does not resolve to a readable CIF: {path}")
    return path


def _ccf_search_path(environment: Mapping[str, str]) -> tuple[Path, ...]:
    raw = environment.get("SAS_CCFPATH", "").strip()
    if raw:
        return tuple(Path(value).expanduser().resolve() for value in raw.split(":") if value)

    raw_cif = environment.get("SAS_CCF", "").strip()
    if raw_cif:
        sas_ccf = Path(raw_cif).expanduser()
        if sas_ccf.is_dir():
            return (sas_ccf.resolve(),)
    return ()


def _resolve_cif_constituent(name: str, search_path: tuple[Path, ...]) -> Path:
    # CIF-controlled FNAME is CAL data, not a shell path.  Do not invent Python
    # ``~`` expansion that was not observed in the real SAS CAL parity matrix.
    candidate = Path(name)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        raise CalibrationIdentityError(
            f"absolute CCF constituent {name!r} does not resolve to a readable file"
        )

    if ".." in candidate.parts:
        raise CalibrationIdentityError(
            f"relative CCF constituent {name!r} contains parent traversal unsupported by SAS CAL"
        )

    for directory in search_path:
        root = directory.resolve()
        direct = root / candidate
        if not direct.is_file():
            continue
        resolved = direct.resolve()
        if not resolved.is_relative_to(root):
            raise CalibrationIdentityError(
                f"relative CCF constituent {name!r} resolves outside SAS_CCFPATH root {root}"
            )
        return resolved
    raise CalibrationIdentityError(
        f"CCF constituent {name!r} cannot be resolved through SAS_CCFPATH"
    )


def _parse_sas_ccffiles(raw: str) -> tuple[str, ...]:
    """Parse SAS 22 SAS_CCFFILES list syntax observed in the real CAL parity probe."""
    if not raw.strip():
        return ()
    if '"' in raw or "'" in raw:
        raise CalibrationIdentityError(
            "SAS_CCFFILES quoting is unsupported by the observed SAS 22 CAL syntax"
        )
    return tuple(token for token in re.split(r"[\s,:]+", raw.strip()) if token)


def _resolve_replacement_ccf(name: str) -> Path:
    """Resolve one SAS_CCFFILES entry using the observed SAS 22 path semantics."""
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    raise CalibrationIdentityError(
        f"SAS_CCFFILES replacement {name!r} does not resolve to a readable file"
    )


def _replacement_records(raw: str) -> tuple[CalibrationReplacement, ...]:
    replacements: list[CalibrationReplacement] = []
    seen_names: set[str] = set()
    for token in _parse_sas_ccffiles(raw):
        path = _resolve_replacement_ccf(token)
        name = path.name
        if name in seen_names:
            raise CalibrationIdentityError(
                f"SAS_CCFFILES contains duplicate replacement basename {name!r}; "
                "replacement precedence is ambiguous"
            )
        seen_names.add(name)
        replacements.append(CalibrationReplacement(name=name, sha256=_file_sha256(path)))
    return tuple(replacements)


def _active_constituents(
    rows: tuple[dict[str, object], ...],
    search_path: tuple[Path, ...],
    *,
    replaced_names: frozenset[str] = frozenset(),
) -> tuple[CalibrationConstituent, ...]:
    by_name: dict[str, CalibrationConstituent] = {}
    for row in rows:
        raw_name = row.get("FNAME")
        if not isinstance(raw_name, str) or not raw_name:
            raise CalibrationIdentityError("CALINDEX row has no usable FNAME")
        name = Path(raw_name).name
        if name in replaced_names:
            continue

        path = _resolve_cif_constituent(raw_name, search_path)
        size = path.stat().st_size

        recorded_size = row.get("FSIZE")
        if isinstance(recorded_size, int) and recorded_size > 0 and recorded_size != size:
            raise CalibrationIdentityError(
                f"CCF constituent {name} size {size} does not match CALINDEX FSIZE {recorded_size}"
            )

        recorded_md5 = row.get("MD5")
        cif_md5 = str(recorded_md5).strip().lower() if recorded_md5 else None
        if cif_md5:
            actual_md5 = _file_digest(path, "md5")
            if actual_md5.lower() != cif_md5:
                raise CalibrationIdentityError(
                    f"CCF constituent {name} does not match CALINDEX MD5 {cif_md5}"
                )

        item = CalibrationConstituent(
            name=name,
            sha256=_file_sha256(path),
            cif_md5=cif_md5,
            size=size,
        )
        previous = by_name.get(name)
        if previous is not None and previous.sha256 != item.sha256:
            raise CalibrationIdentityError(
                f"CALINDEX resolves basename {name!r} to inconsistent constituent bytes"
            )
        by_name[name] = item
    return tuple(by_name[name] for name in sorted(by_name))


def read_calibration_identity(
    environment: Mapping[str, str] | None = None,
) -> CalibrationIdentity:
    """Capture the active CIF, exact selected CCF bytes, and explicit replacements."""
    env: Mapping[str, str] = os.environ if environment is None else environment
    cif_path = _resolve_cif(env)
    search_path = _ccf_search_path(env)
    calindex_sha256, rows = _read_calindex(cif_path)

    replacements = _replacement_records(env.get("SAS_CCFFILES", ""))
    replaced_names = frozenset(item.name for item in replacements)
    constituents = _active_constituents(
        rows,
        search_path,
        replaced_names=replaced_names,
    )

    return CalibrationIdentity(
        cif_path=cif_path,
        cif_file_sha256=_file_sha256(cif_path),
        calindex_sha256=calindex_sha256,
        replacements=replacements,
        ccf_search_path=search_path,
        constituents=constituents,
    )


def resolve_executable(executable: str, environment: Mapping[str, str] | None = None) -> Path:
    """Resolve one executable using the supplied frozen environment's PATH."""
    path_value = environment.get("PATH") if environment is not None else None
    resolved = shutil.which(executable, path=path_value)
    if resolved is None:
        raise CalibrationIdentityError(f"SAS executable {executable!r} was not found on PATH")
    return Path(resolved).resolve()


def _combined_output(result: BoundedProcessResult) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def _require_complete_version_output(
    result: BoundedProcessResult,
    *,
    executable: str,
) -> None:
    if result.stdout_capture.truncated or result.stderr_capture.truncated:
        raise CalibrationIdentityError(
            f"cannot determine {executable} version from truncated diagnostic output; "
            "the executable exceeded the bounded producer-probe output limit"
        )


def _canonical_task_version(output: str, *, executable: str) -> str:
    """Extract stable SAS task version/build information from noisy task output."""
    escaped = re.escape(Path(executable).name)
    match = re.search(
        rf"\b{escaped}\s+\(([^)]+)\)\s+\[([^\]]+)\]",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise CalibrationIdentityError(
            f"cannot locate stable {Path(executable).name} version/build identifier in task output"
        )
    task_version = " ".join(match.group(1).split())
    build = " ".join(match.group(2).split())
    return f"{task_version} [{build}]"


def _canonical_sas_version(output: str) -> str | None:
    """Extract stable SAS release/build identity, excluding runtime/environment text."""
    release_match = re.search(r"\bSAS\s+release:\s*([^\r\n]+)", output, flags=re.IGNORECASE)
    if release_match is not None:
        return " ".join(release_match.group(1).split())

    banner = re.search(
        r"\bsasversion\s+\(([^)]+)\)\s+\[([^\]]+)\]",
        output,
        flags=re.IGNORECASE,
    )
    if banner is not None:
        return " ".join(banner.group(2).split())
    return None


def _task_version(
    executable: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, Path]:
    resolved = resolve_executable(executable, environment)
    result = run_bounded([str(resolved), "-v"], environment=environment)
    _require_complete_version_output(result, executable=executable)
    if result.returncode != 0:
        details = diagnostic_summary(result.stderr or result.stdout)
        raise CalibrationIdentityError(
            f"cannot determine {executable} version (exit {result.returncode}): {details}"
        )
    output = _combined_output(result)
    if not output:
        raise CalibrationIdentityError(f"{executable} -v returned no version text")
    return _canonical_task_version(output, executable=resolved.name), resolved


def read_sas_producer_identity(
    environment: Mapping[str, str] | None = None,
    *,
    esky2det: str = "esky2det",
) -> SasProducerIdentity:
    """Capture stable SAS release/build identity and exact esky2det bytes."""
    esky2det_version, executable = _task_version(esky2det, environment=environment)
    sas_version: str | None = None
    try:
        sasversion = resolve_executable("sasversion", environment)
    except CalibrationIdentityError:
        sasversion = None
    if sasversion is not None:
        result = run_bounded([str(sasversion)], environment=environment)
        if result.stdout_capture.truncated or result.stderr_capture.truncated:
            raise CalibrationIdentityError(
                "cannot determine SAS release from truncated sasversion output; "
                "the executable exceeded the bounded producer-probe output limit"
            )
        if result.returncode == 0:
            sas_version = _canonical_sas_version(_combined_output(result))
    return SasProducerIdentity(
        esky2det_version=esky2det_version,
        sas_version=sas_version,
        esky2det_sha256=_file_sha256(executable),
        esky2det_name=executable.name,
    )

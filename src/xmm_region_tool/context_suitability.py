"""Observation-association and CIF-suitability checks for SAS projection.

The checks in this module deliberately separate two roles:

* an active ``*SUM.SAS`` provides an operational observation-association guard;
* the original ODF ``*SUM.ASC`` provides the observation-level date that
  ``cifbuild`` used by default, which can be compared to CIF ``OBSVDATE``.

For an intentionally ODF-independent ``cifbuild withobservationdate=yes`` setup,
callers may instead supply an explicit immutable observation record.  That route
is treated as declared association evidence and is cross-checked against the
event ObsID/time and the CIF observation date.

ODF paths, ODF bytes, and declared observation evidence are validation evidence
only.  They do not enter semantic SAS projection/context identity merely because
they are used as guards.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from .calibration import CalibrationIdentity
from .provenance import EventIdentity


class ContextSuitabilityError(ValueError):
    """Raised when the frozen SAS context cannot be proved suitable for an event."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_timestamp(value: str, *, description: str) -> str:
    token = value.strip().rstrip(".")
    try:
        parsed = Time(token, format="fits", scale="utc")
    except (TypeError, ValueError) as exc:
        raise ContextSuitabilityError(
            f"{description} is not a valid FITS UTC timestamp: {value!r}"
        ) from exc
    return parsed.fits


def _header_values(path: Path, key: str) -> tuple[str, ...]:
    values: list[str] = []
    try:
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                header = getattr(hdu, "header", None)
                if header is None:
                    continue
                for card in header.cards:
                    if str(card.keyword).upper() != key.upper():
                        continue
                    value = str(card.value).strip()
                    if value and value not in values:
                        values.append(value)
    except OSError as exc:
        raise ContextSuitabilityError(f"cannot read active CIF {path}: {exc}") from exc
    return tuple(values)


def _one_cif_timestamp(path: Path, key: str) -> str:
    values = _header_values(path, key)
    if not values:
        raise ContextSuitabilityError(f"active CIF contains no {key} attribute: {path}")
    if len(values) != 1:
        raise ContextSuitabilityError(
            f"active CIF contains conflicting {key} values: {list(values)!r}"
        )
    return _canonical_timestamp(values[0], description=f"CIF {key}")


def _read_text(path: Path, *, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ContextSuitabilityError(f"cannot read {description} {path}: {exc}") from exc


@dataclass(frozen=True)
class ObservationRecord:
    """Canonical observation-level identity and scheduled timing evidence."""

    obs_id: str
    scheduled_start: str
    scheduled_end: str

    def __post_init__(self) -> None:
        obs_id = str(self.obs_id).strip()
        if re.fullmatch(r"[0-9]{10}", obs_id) is None:
            raise ContextSuitabilityError(
                f"observation record has invalid ObsID {self.obs_id!r}"
            )
        object.__setattr__(self, "obs_id", obs_id)
        object.__setattr__(
            self,
            "scheduled_start",
            _canonical_timestamp(
                str(self.scheduled_start), description="observation scheduled start"
            ),
        )
        object.__setattr__(
            self,
            "scheduled_end",
            _canonical_timestamp(
                str(self.scheduled_end), description="observation scheduled end"
            ),
        )
        if Time(self.scheduled_end, format="fits", scale="utc") < Time(
            self.scheduled_start, format="fits", scale="utc"
        ):
            raise ContextSuitabilityError("observation scheduled end precedes scheduled start")

    def evidence_record(self) -> dict[str, str]:
        return {
            "obs_id": self.obs_id,
            "scheduled_start": self.scheduled_start,
            "scheduled_end": self.scheduled_end,
        }


def _observation_record(text: str) -> ObservationRecord:
    """Parse the fixed-width OBSERVATION record used by ODF/SAS summaries."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line[:11].strip().upper() != "OBSERVATION":
            continue
        block = lines[index : index + 5]
        if len(block) < 5:
            raise ContextSuitabilityError("summary OBSERVATION record is truncated")
        obs_id = block[1][:10].strip()
        scheduled_start = block[3][:19].strip().rstrip(".")
        scheduled_end = block[4][:19].strip().rstrip(".")
        if not scheduled_start or not scheduled_end:
            raise ContextSuitabilityError(
                "summary OBSERVATION record has no usable scheduled start/end"
            )
        return ObservationRecord(
            obs_id=obs_id,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
        )
    raise ContextSuitabilityError("summary contains no OBSERVATION record")


def _sosf_odf_path(text: str, *, summary_path: Path) -> Path | None:
    """Return the original ODF directory recorded by odfingest's PATH statement."""
    for line in text.splitlines():
        match = re.match(r"^\s*PATH\s*(?:=\s*)?(.+?)\s*$", line, flags=re.IGNORECASE)
        if match is None:
            continue
        raw = match.group(1).split(" / ", 1)[0].strip().strip("'\"")
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = summary_path.parent / candidate
        return candidate.resolve()
    return None


def _unique_original_summary(directory: Path) -> Path:
    candidates = sorted(path for path in directory.glob("*SUM.ASC") if path.is_file())
    if not candidates:
        raise ContextSuitabilityError(
            f"ODF directory contains no *SUM.ASC original summary: {directory}"
        )
    if len(candidates) != 1:
        raise ContextSuitabilityError(
            "ODF directory contains multiple *SUM.ASC files; observation association is ambiguous"
        )
    return candidates[0]


def _resolve_active_odf(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ContextSuitabilityError(f"SAS_ODF does not exist: {path}")
    return path


def _validate_declared_record(
    record: ObservationRecord,
    *,
    event_identity: EventIdentity,
    cif_obsvdate: str,
) -> None:
    if record.obs_id != event_identity.obs_id:
        raise ContextSuitabilityError(
            f"declared observation belongs to ObsID {record.obs_id}, "
            f"but projection event belongs to {event_identity.obs_id}"
        )
    if record.scheduled_start != cif_obsvdate:
        raise ContextSuitabilityError(
            "active CIF OBSVDATE does not match the declared observation start: "
            f"CIF={cif_obsvdate}, declared={record.scheduled_start}"
        )
    if event_identity.date_obs is None:
        raise ContextSuitabilityError(
            "projection event has no DATE-OBS for declared-observation interval validation"
        )
    event_time = Time(
        _canonical_timestamp(event_identity.date_obs, description="event DATE-OBS"),
        format="fits",
        scale="utc",
    )
    start = Time(record.scheduled_start, format="fits", scale="utc")
    end = Time(record.scheduled_end, format="fits", scale="utc")
    if not start <= event_time <= end:
        raise ContextSuitabilityError(
            "projection event DATE-OBS lies outside the declared observation interval: "
            f"event={event_identity.date_obs}, interval="
            f"[{record.scheduled_start}, {record.scheduled_end}]"
        )


@dataclass(frozen=True)
class ContextSuitabilityEvidence:
    """Immutable non-semantic evidence used to admit one event under one context."""

    event_obs_id: str
    cif_obsvdate: str
    cif_analdate: str
    association_source: str
    association_obs_id: str
    observation_start: str
    observation_end: str
    active_summary_sha256: str | None = None
    original_summary_sha256: str | None = None

    def evidence_record(self) -> dict[str, object | None]:
        return {
            "schema": "xmm-region-tool.context-suitability-evidence/v2",
            "event_obs_id": self.event_obs_id,
            "cif_obsvdate": self.cif_obsvdate,
            "cif_analdate": self.cif_analdate,
            "association_source": self.association_source,
            "association_obs_id": self.association_obs_id,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
            "active_summary_sha256": self.active_summary_sha256,
            "original_summary_sha256": self.original_summary_sha256,
        }


def capture_context_suitability(
    *,
    environment: Mapping[str, str],
    calibration: CalibrationIdentity,
    event_identity: EventIdentity,
    declared_observation: ObservationRecord | None = None,
) -> ContextSuitabilityEvidence:
    """Prove observation association and material CIF date suitability.

    Normal SAS setup is proved from ``SAS_ODF``. A ``*SUM.SAS`` is used only
    as an operational ObsID guard; its PATH is followed to the original ODF
    ``*SUM.ASC`` because that is the observation-level source from which
    ``cifbuild`` normally obtained the date recorded as ``OBSVDATE``.

    When ``SAS_ODF`` is intentionally absent, an explicit ``ObservationRecord``
    supports the documented ``cifbuild withobservationdate=yes`` workflow. That
    record is assertion evidence, so the implementation additionally checks its
    ObsID against the event, its start against the CIF ``OBSVDATE``, and the
    event ``DATE-OBS`` against the declared observation interval.
    """
    cif_obsvdate = _one_cif_timestamp(calibration.cif_path, "OBSVDATE")
    cif_analdate = _one_cif_timestamp(calibration.cif_path, "ANALDATE")

    raw_odf = environment.get("SAS_ODF", "").strip()
    if not raw_odf:
        if declared_observation is None:
            raise ContextSuitabilityError(
                "SAS_ODF is not set and no explicit observation evidence was supplied; "
                "ODF-independent contexts must declare the observation record used for "
                "cifbuild rather than assuming event DATE-OBS equals CIF OBSVDATE"
            )
        if not isinstance(declared_observation, ObservationRecord):
            raise ContextSuitabilityError(
                "declared observation evidence must be an ObservationRecord"
            )
        _validate_declared_record(
            declared_observation,
            event_identity=event_identity,
            cif_obsvdate=cif_obsvdate,
        )
        return ContextSuitabilityEvidence(
            event_obs_id=event_identity.obs_id,
            cif_obsvdate=cif_obsvdate,
            cif_analdate=cif_analdate,
            association_source="declared-observation",
            association_obs_id=declared_observation.obs_id,
            observation_start=declared_observation.scheduled_start,
            observation_end=declared_observation.scheduled_end,
        )

    active = _resolve_active_odf(raw_odf)
    active_summary_sha: str | None = None
    original_summary_sha: str | None = None

    if active.is_dir():
        original = _unique_original_summary(active)
        original_text = _read_text(original, description="original ODF summary")
        original_record = _observation_record(original_text)
        association_record = original_record
        source = "original-odf"
        original_summary_sha = _file_sha256(original)
    else:
        active_text = _read_text(active, description="active SAS_ODF summary")
        active_record = _observation_record(active_text)
        active_summary_sha = _file_sha256(active)

        if active.suffix.upper() == ".ASC" or active.name.upper().endswith("SUM.ASC"):
            original_record = active_record
            association_record = active_record
            source = "original-odf"
            original_summary_sha = active_summary_sha
        else:
            if active_record.obs_id != event_identity.obs_id:
                raise ContextSuitabilityError(
                    "active SAS ODF summary belongs to ObsID "
                    f"{active_record.obs_id}, but projection event belongs to "
                    f"{event_identity.obs_id}"
                )
            odf_dir = _sosf_odf_path(active_text, summary_path=active)
            if odf_dir is None or not odf_dir.is_dir():
                raise ContextSuitabilityError(
                    "active *SUM.SAS has no usable PATH back to the original ODF; "
                    "cannot prove the CIF observation date from the SOSF alone"
                )
            original = _unique_original_summary(odf_dir)
            original_text = _read_text(original, description="original ODF summary")
            original_record = _observation_record(original_text)
            association_record = active_record
            source = "sas-summary+original-odf"
            original_summary_sha = _file_sha256(original)

    if association_record.obs_id != event_identity.obs_id:
        raise ContextSuitabilityError(
            f"active ODF evidence belongs to ObsID {association_record.obs_id}, "
            f"but projection event belongs to {event_identity.obs_id}"
        )
    if original_record.obs_id != event_identity.obs_id:
        raise ContextSuitabilityError(
            f"original ODF belongs to ObsID {original_record.obs_id}, "
            f"but projection event belongs to {event_identity.obs_id}"
        )
    if original_record.scheduled_start != cif_obsvdate:
        raise ContextSuitabilityError(
            "active CIF OBSVDATE does not match the original ODF observation start: "
            f"CIF={cif_obsvdate}, ODF={original_record.scheduled_start}"
        )
    if declared_observation is not None and declared_observation != original_record:
        raise ContextSuitabilityError(
            "explicit observation evidence conflicts with the active ODF observation record"
        )

    return ContextSuitabilityEvidence(
        event_obs_id=event_identity.obs_id,
        cif_obsvdate=cif_obsvdate,
        cif_analdate=cif_analdate,
        association_source=source,
        association_obs_id=association_record.obs_id,
        observation_start=original_record.scheduled_start,
        observation_end=original_record.scheduled_end,
        active_summary_sha256=active_summary_sha,
        original_summary_sha256=original_summary_sha,
    )

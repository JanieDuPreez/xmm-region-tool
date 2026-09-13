"""Identity and provenance helpers for generated detector regions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits


class EventIdentityError(ValueError):
    """Raised when an input file cannot be identified as a usable EPIC event product."""


class RegionBindingError(ValueError):
    """Raised when a generated detector region lacks usable binding metadata."""


_INSTRUMENT_NAMES = {
    "EMOS1": "mos1",
    "EMOS2": "mos2",
    "EPN": "pn",
    "MOS1": "mos1",
    "MOS2": "mos2",
    "PN": "pn",
}
_CANONICAL_INSTRUMENT_HEADERS = {
    "mos1": "EMOS1",
    "mos2": "EMOS2",
    "pn": "EPN",
}


@dataclass(frozen=True)
class EventIdentity:
    """Durable scientific identity carried by one XMM EPIC event product."""

    path: Path
    instrument: str
    instrument_header: str
    obs_id: str
    exposure_id: str | None
    telescope: str | None
    date_obs: str | None
    ra_pnt: float | None
    dec_pnt: float | None
    pa_pnt: float | None

    def canonical_record(self) -> dict[str, object | None]:
        """Return path-independent fields that affect safe detector-region reuse."""
        return {
            "schema": "xmm-region-tool.event-identity/v2",
            "telescope": self.telescope,
            "instrument": self.instrument,
            "instrument_header": self.instrument_header,
            "obs_id": self.obs_id,
            "exposure_id": self.exposure_id,
            "date_obs": self.date_obs,
            "ra_pnt": self.ra_pnt,
            "dec_pnt": self.dec_pnt,
            "pa_pnt": self.pa_pnt,
        }

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_record(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RegionBinding:
    """Identity/provenance metadata stored in one generated FITS detector region."""

    path: Path
    event_identity_sha256: str
    event_file_sha256: str | None
    source_region_sha256: str | None
    celestial_geometry_sha256: str | None
    projection_identity_sha256: str | None
    instrument: str | None
    obs_id: str | None
    exposure_id: str | None
    calibration_identity_sha256: str | None
    cif_file_sha256: str | None
    calindex_sha256: str | None
    sas_producer_sha256: str | None
    esky2det_version: str | None
    sas_version: str | None


@dataclass(frozen=True)
class RegionBindingReport:
    """Result of checking a generated region against its intended inputs."""

    binding: RegionBinding
    event_identity: EventIdentity
    event_file_sha256: str
    source_region_sha256: str | None
    expected_celestial_geometry_sha256: str | None
    expected_projection_identity_sha256: str | None
    mismatches: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.mismatches

    def format_text(self) -> str:
        lines = [
            "XMM detector-region binding check",
            f"region: {self.binding.path}",
            f"event: {self.event_identity.path}",
            f"compatible: {'yes' if self.compatible else 'NO'}",
            f"stored event identity: {self.binding.event_identity_sha256}",
            f"current event identity: {self.event_identity.identity_sha256}",
            "stored exact event SHA256: "
            + (self.binding.event_file_sha256 or "missing"),
            f"current exact event SHA256: {self.event_file_sha256}",
        ]
        if self.source_region_sha256 is not None:
            lines.append(f"current source-region SHA256: {self.source_region_sha256}")
            lines.append(
                "stored source-region SHA256: "
                + (self.binding.source_region_sha256 or "missing")
            )
        if self.expected_celestial_geometry_sha256 is not None:
            lines.append(
                "expected celestial geometry SHA256: "
                f"{self.expected_celestial_geometry_sha256}"
            )
            lines.append(
                "stored celestial geometry SHA256: "
                + (self.binding.celestial_geometry_sha256 or "missing")
            )
        if self.expected_projection_identity_sha256 is not None:
            lines.append(
                "expected projection identity SHA256: "
                f"{self.expected_projection_identity_sha256}"
            )
            lines.append(
                "stored projection identity SHA256: "
                + (self.binding.projection_identity_sha256 or "missing")
            )
        if self.binding.calibration_identity_sha256 is not None:
            lines.append(
                "conversion calibration identity: "
                f"{self.binding.calibration_identity_sha256}"
            )
        if self.binding.cif_file_sha256 is not None:
            lines.append(f"conversion CIF file SHA256: {self.binding.cif_file_sha256}")
        if self.binding.calindex_sha256 is not None:
            lines.append(f"conversion CALINDEX SHA256: {self.binding.calindex_sha256}")
        if self.binding.sas_producer_sha256 is not None:
            lines.append(f"conversion SAS producer: {self.binding.sas_producer_sha256}")
        if self.binding.esky2det_version is not None:
            lines.append(f"esky2det version: {self.binding.esky2det_version}")
        if self.binding.sas_version is not None:
            lines.append(f"SAS version: {self.binding.sas_version}")
        if self.mismatches:
            lines.append("mismatches:")
            lines.extend(f"- {item}" for item in self.mismatches)
        return "\n".join(lines)


def file_sha256(path: str | Path) -> str:
    """Hash a source artifact such as an event file or user-authored DS9 file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_event_hdu_index(
    hdus: fits.HDUList,
    *,
    required_columns: tuple[str, ...] = ("DETX", "DETY"),
) -> int:
    """Return the unique canonical EPIC science ``EVENTS`` HDU index.

    Science-event consumers must not substitute an arbitrary table merely because
    it happens to contain detector-coordinate columns. Real SAS consumers address
    the named ``EVENTS`` extension explicitly, so identity and validation do the
    same.
    """
    event_indices = [
        index
        for index, hdu in enumerate(hdus)
        if str(getattr(hdu, "name", "")).strip().upper() == "EVENTS"
    ]
    if not event_indices:
        raise EventIdentityError("event file has no canonical EVENTS extension")
    if len(event_indices) != 1:
        raise EventIdentityError(
            f"event file has {len(event_indices)} EVENTS extensions; science event table is ambiguous"
        )

    index = event_indices[0]
    hdu = hdus[index]
    names = getattr(getattr(hdu, "columns", None), "names", None)
    available = {str(name).upper() for name in names} if names else set()
    required = {str(name).upper() for name in required_columns}
    if not required.issubset(available):
        missing = ", ".join(sorted(required - available))
        raise EventIdentityError(f"canonical EVENTS extension is missing required columns: {missing}")
    return index


def _header_values(headers: list[fits.Header], *keys: str) -> list[Any]:
    wanted = {key.upper() for key in keys}
    values: list[Any] = []
    for header in headers:
        for card in header.cards:
            if str(card.keyword).upper() not in wanted:
                continue
            value = card.value
            if value is not None and str(value).strip():
                values.append(value)
    return values


def _coherent_text(
    headers: list[fits.Header],
    *,
    name: str,
    keys: tuple[str, ...],
    required: bool = False,
) -> str | None:
    values = _header_values(headers, *keys)
    normalized = [str(value).strip() for value in values]
    unique = list(dict.fromkeys(normalized))
    if len(unique) > 1:
        raise EventIdentityError(f"conflicting event header {name} values: {unique!r}")
    if not unique:
        if required:
            raise EventIdentityError(f"event file has no {name} identity")
        return None
    return unique[0]


def _optional_float(value: Any | None, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EventIdentityError(f"event header {name} is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise EventIdentityError(f"event header {name} must be finite: {value!r}")
    return 0.0 if numeric == 0.0 else numeric


def _coherent_optional_float(headers: list[fits.Header], *, name: str) -> float | None:
    values = _header_values(headers, name)
    normalized = [_optional_float(value, name=name) for value in values]
    unique = list(dict.fromkeys(normalized))
    if len(unique) > 1:
        raise EventIdentityError(f"conflicting event header {name} values: {unique!r}")
    return unique[0] if unique else None


def _coherent_instrument(headers: list[fits.Header]) -> tuple[str, str]:
    values = _header_values(headers, "INSTRUME")
    if not values:
        raise EventIdentityError("event file has no INSTRUME identity")
    instruments: list[str] = []
    for value in values:
        token = str(value).strip().upper()
        instrument = _INSTRUMENT_NAMES.get(token)
        if instrument is None:
            raise EventIdentityError(
                f"unsupported XMM instrument INSTRUME={token!r}; expected MOS1/MOS2/pn"
            )
        instruments.append(instrument)
    unique = list(dict.fromkeys(instruments))
    if len(unique) > 1:
        raise EventIdentityError(f"conflicting event header INSTRUME values: {values!r}")
    instrument = unique[0]
    return instrument, _CANONICAL_INSTRUMENT_HEADERS[instrument]


def _coherent_telescope(headers: list[fits.Header]) -> str | None:
    values = _header_values(headers, "TELESCOP")
    if not values:
        return None
    for value in values:
        telescope = str(value).strip()
        if "XMM" not in telescope.upper():
            raise EventIdentityError(
                f"TELESCOP={telescope!r} does not identify an XMM-Newton product"
            )
    return "XMM"


def _coherent_exposure_id(headers: list[fits.Header]) -> str | None:
    """Return the short SAS exposure label without conflating it with ``EXP_ID``.

    Normal SAS products may legitimately carry both a long ODF ``EXP_ID`` (for
    example ``0723802001001``) and a short four-character ``EXPIDSTR`` (for
    example ``S001``). They are distinct namespaces, not aliases. ``EXPIDSTR``
    is the EPIC exposure label used by downstream task/product prefixes and is
    therefore preferred for ``EventIdentity.exposure_id``. ``EXP_ID`` remains a
    fallback for older/minimal products that do not carry the short label.

    Historical synthetic/minimal products sometimes stored the short label in
    ``EXP_ID`` itself. When that value has the canonical S/U + three-digit form,
    keep the old fail-closed alias check against ``EXPIDSTR``/``EXPID``.
    """
    long_id = _coherent_text(headers, name="EXP_ID", keys=("EXP_ID",))
    short_id = _coherent_text(
        headers,
        name="EXPIDSTR",
        keys=("EXPIDSTR", "EXPID"),
    )
    if long_id is not None and short_id is not None:
        token = long_id.upper()
        long_is_short_label = (
            len(token) == 4 and token[0] in {"S", "U"} and token[1:].isdigit()
        )
        if long_is_short_label and token != short_id.upper():
            raise EventIdentityError(
                f"conflicting event header EXP_ID values: {[long_id, short_id]!r}"
            )
    return short_id or long_id


def read_event_identity(path: str | Path) -> EventIdentity:
    """Read one unambiguous observation/instrument identity from an EPIC science event file."""
    event_path = Path(path).expanduser().resolve()
    if not event_path.is_file():
        raise EventIdentityError(f"event file does not exist: {event_path}")

    with fits.open(event_path, memmap=True) as hdus:
        event_hdu = hdus[canonical_event_hdu_index(hdus)]
        headers = [event_hdu.header, hdus[0].header]

        instrument, instrument_header = _coherent_instrument(headers)
        obs_id = _coherent_text(
            headers,
            name="OBS_ID",
            keys=("OBS_ID", "OBSID"),
            required=True,
        )
        assert obs_id is not None
        exposure_id = _coherent_exposure_id(headers)
        telescope = _coherent_telescope(headers)
        date_obs = _coherent_text(
            headers,
            name="DATE-OBS",
            keys=("DATE-OBS", "DATE_OBS"),
        )

        return EventIdentity(
            path=event_path,
            instrument=instrument,
            instrument_header=instrument_header,
            obs_id=obs_id,
            exposure_id=exposure_id,
            telescope=telescope,
            date_obs=date_obs,
            ra_pnt=_coherent_optional_float(headers, name="RA_PNT"),
            dec_pnt=_coherent_optional_float(headers, name="DEC_PNT"),
            pa_pnt=_coherent_optional_float(headers, name="PA_PNT"),
        )


def _optional_header_text(header: fits.Header, key: str) -> str | None:
    value = header.get(key)
    return str(value).strip() if value is not None else None


def _validated_sha256_header(header: fits.Header, key: str, *, required: bool) -> str | None:
    raw = header.get(key)
    if raw is None:
        if required:
            raise RegionBindingError(
                f"REGION extension has no {key} binding; regenerate it with xmm-region-tool"
            )
        return None
    value = str(raw).strip().lower()
    if len(value) != 64:
        raise RegionBindingError(f"{key} is not a valid SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RegionBindingError(f"{key} is not a valid SHA256 digest") from exc
    return value


def read_region_binding(path: str | Path) -> RegionBinding:
    """Read xmm-region-tool provenance from a generated FITS REGION table."""
    region_path = Path(path).expanduser().resolve()
    if not region_path.is_file():
        raise RegionBindingError(f"region FITS file does not exist: {region_path}")
    with fits.open(region_path, memmap=True) as hdus:
        if "REGION" not in hdus:
            raise RegionBindingError(f"FITS file has no REGION extension: {region_path}")
        header = hdus["REGION"].header
        identity = _validated_sha256_header(header, "XMMRGID", required=True)
        assert identity is not None
        event_file_sha = _validated_sha256_header(header, "XMRGEVT", required=False)
        source = _validated_sha256_header(header, "XMRGSHA", required=False)
        celestial = _validated_sha256_header(header, "XMRGCEL", required=False)
        projection = _validated_sha256_header(header, "XMRGPRO", required=False)
        calibration = _validated_sha256_header(header, "XMRGCCF", required=False)
        cif_file = _validated_sha256_header(header, "XMRGCIF", required=False)
        calindex = _validated_sha256_header(header, "XMRGCAL", required=False)
        producer = _validated_sha256_header(header, "XMRGPRD", required=False)
        return RegionBinding(
            path=region_path,
            event_identity_sha256=identity,
            event_file_sha256=event_file_sha,
            source_region_sha256=source,
            celestial_geometry_sha256=celestial,
            projection_identity_sha256=projection,
            instrument=_optional_header_text(header, "INSTRUME"),
            obs_id=_optional_header_text(header, "OBS_ID"),
            exposure_id=_optional_header_text(header, "EXP_ID"),
            calibration_identity_sha256=calibration,
            cif_file_sha256=cif_file,
            calindex_sha256=calindex,
            sas_producer_sha256=producer,
            esky2det_version=_optional_header_text(header, "ESKYVER"),
            sas_version=_optional_header_text(header, "SASVERS"),
        )


def check_region_binding(
    region_fits: str | Path,
    event_file: str | Path,
    *,
    source_region: str | Path | None = None,
    expected_celestial_geometry_sha256: str | None = None,
    expected_projection_identity_sha256: str | None = None,
) -> RegionBindingReport:
    """Check that a detector region belongs to the supplied exact event/DS9 inputs."""
    binding = read_region_binding(region_fits)
    event_identity = read_event_identity(event_file)
    event_file_sha = file_sha256(event_file)
    source_sha = file_sha256(source_region) if source_region is not None else None
    mismatches: list[str] = []

    if binding.event_identity_sha256 != event_identity.identity_sha256:
        mismatches.append("event identity SHA256 differs")
    if binding.event_file_sha256 is None:
        mismatches.append("generated region has no exact event-file SHA256")
    elif binding.event_file_sha256 != event_file_sha:
        mismatches.append("exact event-file SHA256 differs")
    if binding.instrument is not None and binding.instrument.upper() != event_identity.instrument_header:
        mismatches.append(
            f"instrument differs ({binding.instrument!r} != {event_identity.instrument_header!r})"
        )
    if binding.obs_id is not None and binding.obs_id != event_identity.obs_id:
        mismatches.append(f"ObsID differs ({binding.obs_id!r} != {event_identity.obs_id!r})")
    if (
        binding.exposure_id is not None
        and event_identity.exposure_id is not None
        and binding.exposure_id != event_identity.exposure_id
    ):
        mismatches.append(
            f"exposure differs ({binding.exposure_id!r} != {event_identity.exposure_id!r})"
        )
    if source_sha is not None:
        if binding.source_region_sha256 is None:
            mismatches.append("generated region has no source DS9 SHA256")
        elif binding.source_region_sha256 != source_sha:
            mismatches.append("source DS9 SHA256 differs")

    if expected_celestial_geometry_sha256 is not None:
        if binding.celestial_geometry_sha256 is None:
            mismatches.append("generated region has no celestial geometry SHA256")
        elif binding.celestial_geometry_sha256 != expected_celestial_geometry_sha256:
            mismatches.append("celestial geometry SHA256 differs")

    if expected_projection_identity_sha256 is not None:
        if binding.projection_identity_sha256 is None:
            mismatches.append("generated region has no projection identity SHA256")
        elif binding.projection_identity_sha256 != expected_projection_identity_sha256:
            mismatches.append("projection identity SHA256 differs")

    return RegionBindingReport(
        binding=binding,
        event_identity=event_identity,
        event_file_sha256=event_file_sha,
        source_region_sha256=source_sha,
        expected_celestial_geometry_sha256=expected_celestial_geometry_sha256,
        expected_projection_identity_sha256=expected_projection_identity_sha256,
        mismatches=tuple(mismatches),
    )
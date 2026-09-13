"""Role-specific validation for XMM science event products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.time import Time

from .provenance import (
    EventIdentity,
    EventIdentityError,
    _coherent_exposure_id,
    _coherent_instrument,
    _coherent_telescope,
    _coherent_text,
    _optional_float,
    canonical_event_hdu_index,
    file_sha256,
)
from .sas_paths import SasPathError, validate_sas_safe_resolved_path


_SCIENCE_CALINFO_PRIMARY_KEYS = (
    "TELESCOP",
    "INSTRUME",
    "DATE-OBS",
    "RA_PNT",
    "DEC_PNT",
    "PA_PNT",
)
_PN_OOT_SUFFIXES = ("-allevcoot.fits", "-allevc-oot.fits")


def _required_primary_value(header: fits.Header, key: str) -> Any:
    value = header.get(key)
    if value is None or not str(value).strip():
        raise EventIdentityError(
            f"science calinfoset primary header has no usable {key}; "
            "calinfostyle=set requires complete XMM pointing metadata"
        )
    return value


def _validate_date_obs(value: str) -> None:
    if "T" not in value.upper():
        raise EventIdentityError(
            f"science calinfoset DATE-OBS is not a complete observation date-time: {value!r}"
        )
    try:
        Time(value, format="fits", scale="utc")
    except (TypeError, ValueError) as exc:
        raise EventIdentityError(
            f"science calinfoset DATE-OBS is not a valid FITS observation date-time: {value!r}"
        ) from exc


def _validate_pointing_domain(identity: EventIdentity) -> None:
    assert identity.ra_pnt is not None
    assert identity.dec_pnt is not None
    assert identity.pa_pnt is not None
    if not 0.0 <= identity.ra_pnt < 360.0:
        raise EventIdentityError(
            f"science calinfoset RA_PNT={identity.ra_pnt!r} is outside [0, 360) degrees"
        )
    if not -90.0 <= identity.dec_pnt <= 90.0:
        raise EventIdentityError(
            f"science calinfoset DEC_PNT={identity.dec_pnt!r} is outside [-90, 90] degrees"
        )
    if not 0.0 <= identity.pa_pnt < 360.0:
        raise EventIdentityError(
            f"science calinfoset PA_PNT={identity.pa_pnt!r} is outside [0, 360) degrees"
        )


def _reject_known_pn_oot_role(path: Path, identity: EventIdentity) -> None:
    if identity.instrument != "pn":
        return
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in _PN_OOT_SUFFIXES):
        raise EventIdentityError(
            f"pn OOT product {path.name!r} was supplied where the normal science event list "
            "is required; use the matching non-OOT *-allevc.fits product as eventfile/calinfoset"
        )


def read_science_calinfoset_identity(path: str | Path) -> EventIdentity:
    """Read and validate one event for the normal ``calinfostyle=set`` role.

    Generic event identity inspection may retain partial/derived products. The
    scientific projection role is stricter: the exact PRIMARY header must own
    every calibration-information field documented for ``esky2det`` set mode.
    ``DATE-OBS`` and the pointing values in the returned science identity are
    therefore taken from PRIMARY even when the EVENTS extension legitimately
    carries different timing metadata. Observation/instrument/exposure identity
    remains reconciled across PRIMARY/EVENTS so a structurally mismatched event
    table still fails closed.

    Exact 360-degree RA/PA spellings are rejected rather than silently wrapped;
    the supported canonical domain is ``[0, 360)``. Known pn OOT product names
    are also rejected in this normal-science role. Because this exact path is
    later passed to SAS as ``calinfoset=<dataset>``, the release role also
    requires the conservative unquoted SAS-safe resolved-path grammar.
    """
    try:
        event_path = validate_sas_safe_resolved_path(path, role="science event/calinfoset")
    except SasPathError as exc:
        raise EventIdentityError(str(exc)) from exc
    if not event_path.is_file():
        raise EventIdentityError(f"event file does not exist: {event_path}")

    with fits.open(event_path, memmap=True) as hdus:
        event_hdu = hdus[canonical_event_hdu_index(hdus)]
        primary = hdus[0].header
        headers = [event_hdu.header, primary]
        values = {
            key: _required_primary_value(primary, key)
            for key in _SCIENCE_CALINFO_PRIMARY_KEYS
        }

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

        primary_date = str(values["DATE-OBS"]).strip()
        identity = EventIdentity(
            path=event_path,
            instrument=instrument,
            instrument_header=instrument_header,
            obs_id=obs_id,
            exposure_id=exposure_id,
            telescope=telescope,
            date_obs=primary_date,
            ra_pnt=_optional_float(values["RA_PNT"], name="RA_PNT"),
            dec_pnt=_optional_float(values["DEC_PNT"], name="DEC_PNT"),
            pa_pnt=_optional_float(values["PA_PNT"], name="PA_PNT"),
        )

    _reject_known_pn_oot_role(event_path, identity)
    if identity.telescope is None:
        raise EventIdentityError("science calinfoset has no XMM TELESCOP identity")
    if identity.ra_pnt is None or identity.dec_pnt is None or identity.pa_pnt is None:
        raise EventIdentityError("science calinfoset has incomplete pointing identity")

    _validate_date_obs(primary_date)
    _validate_pointing_domain(identity)
    return identity


def read_science_calinfoset_snapshot(path: str | Path) -> tuple[EventIdentity, str]:
    """Return science identity plus exact SHA from one stable pre-projection interval.

    The exact bytes are hashed immediately before and after the full role/identity
    read. A persistent byte change therefore cannot leave semantic identity from
    one file generation paired with the exact SHA of another. This is a
    fail-closed coherence check, not an immutable copy: callers must still keep
    the event path stable for the projection lifecycle to exclude ABA/in-place
    mutation between checks.
    """
    try:
        event_path = validate_sas_safe_resolved_path(path, role="science event/calinfoset")
    except SasPathError as exc:
        raise EventIdentityError(str(exc)) from exc
    sha_before = file_sha256(event_path)
    identity = read_science_calinfoset_identity(event_path)
    sha_after = file_sha256(event_path)
    if sha_after != sha_before:
        raise EventIdentityError(
            "exact science event bytes changed while projection-time identity was being read; "
            "refusing an incoherent event snapshot"
        )
    return identity, sha_after


def validate_projection_event_generation(
    path: str | Path,
    caller_identity: EventIdentity,
    *,
    projection_event_identity_sha256: str,
    projection_event_file_sha256: str,
    expected_event_file_sha256: str | None = None,
) -> EventIdentity:
    """Require caller, projection and current event bytes to describe one generation.

    ``expected_event_file_sha256`` is an optional invocation/item-level frozen
    generation.  When supplied, it prevents separate projections in one logical
    workflow from silently binding different exact generations of the same path.
    A fresh science-role snapshot is also taken immediately before the caller
    promotes/materialises the completed projection.
    """
    try:
        event_path = validate_sas_safe_resolved_path(path, role="science event/calinfoset")
    except SasPathError as exc:
        raise EventIdentityError(str(exc)) from exc
    if caller_identity.identity_sha256 != projection_event_identity_sha256:
        raise EventIdentityError(
            "caller event identity no longer matches the authoritative projection-time identity"
        )
    if (
        expected_event_file_sha256 is not None
        and expected_event_file_sha256 != projection_event_file_sha256
    ):
        raise EventIdentityError(
            "event file generation changed between workflow preflight and projection"
        )

    current_identity, current_sha = read_science_calinfoset_snapshot(event_path)
    if current_identity.identity_sha256 != projection_event_identity_sha256:
        raise EventIdentityError(
            "current event identity no longer matches the event used for projection"
        )
    if current_sha != projection_event_file_sha256:
        raise EventIdentityError(
            "current event bytes no longer match the exact calinfoset used for projection"
        )
    return current_identity

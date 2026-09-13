"""DS9 adapter for the public in-memory celestial selection model."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from regions import (
    CircleAnnulusSkyRegion,
    CircleSkyRegion,
    EllipseAnnulusSkyRegion,
    EllipseSkyRegion,
    PolygonSkyRegion,
    RectangleAnnulusSkyRegion,
    RectangleSkyRegion,
    Regions,
    SkyRegion,
)

from .constructors import _ordered_axis_local_sector_selection
from .limits import MAX_SOURCE_BYTES, MAX_SOURCE_CELLS, MAX_SAMPLES, integer_limit
from .model import (
    CelestialBoundary,
    CelestialRegion,
    CelestialSelection,
    CirclePath,
    EllipsePath,
    GeodesicPolygonPath,
)

SkyBoundary = CelestialBoundary
ParsedSkyRegion = CelestialRegion


class UnsupportedRegionError(ValueError):
    """Raised when a DS9 region cannot yet be represented without distortion."""


_PARSE_LOSS_MARKERS = ("skipping", "not supported", "unable to parse", "failed to parse")
_PANDA_RE = re.compile(r"^([+-]?)\s*panda\s*\((.*)\)\s*$", re.IGNORECASE)
_EPANDA_RE = re.compile(r"^([+-]?)\s*epanda\s*\((.*)\)\s*$", re.IGNORECASE)
_BPANDA_RE = re.compile(r"^([+-]?)\s*bpanda\s*\((.*)\)\s*$", re.IGNORECASE)
_ANNULUS_RE = re.compile(r"^([+-]?)\s*annulus\s*\((.*)\)\s*$", re.IGNORECASE)
_ELLIPSE_RE = re.compile(r"^([+-]?)\s*ellipse\s*\((.*)\)\s*$", re.IGNORECASE)
_FRAME_ALIASES = {
    "icrs": "icrs",
    "fk5": "fk5",
    "j2000": "fk5",
    "fk4": "fk4",
    "b1950": "fk4",
    "galactic": "galactic",
    "ecliptic": "ecliptic",
}
_PIXEL_FRAMES = {"image", "physical", "detector", "amplifier", "linear"}


def _parse_loss_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for item in caught:
        message = str(item.message)
        lowered = message.lower()
        if any(marker in lowered for marker in _PARSE_LOSS_MARKERS):
            messages.append(message)
    return messages


def _parse_ds9_sky_text(text: str) -> list[SkyRegion]:
    _check_source_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source_regions = Regions.parse(text, format="ds9")
    parse_losses = _parse_loss_warnings(caught)
    if parse_losses:
        details = "; ".join(parse_losses)
        raise UnsupportedRegionError(
            "the DS9 parser could not preserve every input region; refusing a partial "
            f"conversion: {details}"
        )

    result: list[SkyRegion] = []
    for region in source_regions:
        if not isinstance(region, SkyRegion):
            raise TypeError(
                "all DS9 regions must use sky coordinates (for example fk5 or icrs); "
                f"got {type(region).__name__}"
            )
        result.append(region)
    return result


def read_ds9_sky_regions(path: str | Path) -> list[SkyRegion]:
    input_path = Path(path)
    result = _parse_ds9_sky_text(_read_source_text(input_path))
    if not result:
        raise ValueError("DS9 region file contains no regions")
    return result


def _selection_from_sky_regions(
    source_regions: Sequence[SkyRegion],
    *,
    samples: int,
    external_identity: str | None,
    external_provenance: Mapping[str, Any] | None,
    ds9_rotation_angles: bool,
) -> CelestialSelection:
    if not source_regions:
        raise ValueError("at least one sky region is required")
    integer_limit(len(source_regions), "source cells", 1, MAX_SOURCE_CELLS)

    parsed: list[CelestialRegion] = []
    for region in source_regions:
        if not isinstance(region, SkyRegion):
            raise TypeError(f"expected SkyRegion, got {type(region).__name__}")
        include_value = region.meta.get("include", True)
        # regions' DS9 parser emits integer 0/1; accept that explicit adapter contract.
        if isinstance(include_value, (bool, np.bool_)):
            include = bool(include_value)
        elif isinstance(include_value, (int, np.integer)) and include_value in (0, 1):
            include = bool(include_value)
        else:
            raise ValueError("SkyRegion include metadata must be Boolean or integer 0/1")
        parsed.append(
            CelestialRegion(
                include=include,
                boundaries=_boundaries(
                    region,
                    samples,
                    ds9_rotation_angles=ds9_rotation_angles,
                ),
            )
        )
    return CelestialSelection(
        regions=tuple(parsed),
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def selection_from_sky_regions(
    source_regions: Sequence[SkyRegion],
    *,
    samples: int = 128,
    external_identity: str | None = None,
    external_provenance: Mapping[str, Any] | None = None,
) -> CelestialSelection:
    """Build semantic geometry from programmatic ``regions`` objects.

    This public API follows the native ``astropy-regions`` angle convention.
    DS9 text uses a different celestial marker-rotation sign and is converted
    only inside :func:`load_ds9_selection`.
    """
    samples = integer_limit(samples, "samples", 16, MAX_SAMPLES)
    return _selection_from_sky_regions(
        source_regions,
        samples=samples,
        external_identity=external_identity,
        external_provenance=external_provenance,
        ds9_rotation_angles=False,
    )


def _split_ds9_code_statements(text: str) -> list[str]:
    """Split DS9 code statements while discarding comments."""
    statements: list[str] = []
    for line in text.splitlines():
        code = line.split("#", maxsplit=1)[0]
        statements.extend(item.strip() for item in code.split(";") if item.strip())
        if len(statements) > MAX_SOURCE_CELLS:
            raise UnsupportedRegionError("DS9 statement resource limit exceeded")
    return statements


def _check_source_text(text: str) -> None:
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise UnsupportedRegionError("DS9 source byte resource limit exceeded")
    _split_ds9_code_statements(text)


def _read_source_text(path: Path) -> str:
    with path.open("rb") as stream:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise UnsupportedRegionError("DS9 source byte resource limit exceeded")
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise UnsupportedRegionError("DS9 source byte resource limit exceeded")
    text = data.decode("utf-8")
    _check_source_text(text)
    return text


def _parse_angle_token(token: str, *, context: str = "DS9 panda") -> u.Quantity:
    value = token.strip().lower()
    try:
        if value.endswith("deg"):
            return float(value[:-3]) * u.deg
        if value.endswith("d"):
            return float(value[:-1]) * u.deg
        if value.endswith("rad"):
            return float(value[:-3]) * u.rad
        if value.endswith("r"):
            return float(value[:-1]) * u.rad
        return float(value) * u.deg
    except ValueError as exc:
        raise UnsupportedRegionError(f"invalid {context} angle {token!r}") from exc


def _parse_radius_token(token: str, *, context: str = "DS9 panda") -> u.Quantity:
    value = token.strip().lower()
    try:
        if value.endswith('"'):
            return float(value[:-1]) * u.arcsec
        if value.endswith("'"):
            return float(value[:-1]) * u.arcmin
        if value.endswith("deg"):
            return float(value[:-3]) * u.deg
        if value.endswith("d"):
            return float(value[:-1]) * u.deg
        if value.endswith("rad"):
            return float(value[:-3]) * u.rad
        if value.endswith("r"):
            return float(value[:-1]) * u.rad
        return float(value) * u.deg
    except ValueError as exc:
        raise UnsupportedRegionError(f"invalid {context} radius {token!r}") from exc


def _parse_positive_integer(
    token: str,
    *,
    name: str,
    context: str = "DS9 panda",
) -> int:
    try:
        numeric = float(token.strip())
    except ValueError as exc:
        raise UnsupportedRegionError(f"{context} {name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise UnsupportedRegionError(f"{context} {name} must be a positive integer")
    try:
        return integer_limit(int(numeric), name, 1, MAX_SOURCE_CELLS)
    except ValueError as exc:
        raise UnsupportedRegionError(str(exc)) from exc


def _parse_ds9_center(frame: str, x: str, y: str, *, context: str) -> SkyCoord:
    """Delegate DS9 centre syntax, including sexagesimal forms, to regions."""
    probe = _parse_ds9_sky_text(f"{frame}\ncircle({x},{y},1\")\n")
    if len(probe) != 1 or not isinstance(probe[0], CircleSkyRegion):
        raise UnsupportedRegionError(f"could not parse {context} celestial centre")
    return probe[0].center


def _ds9_rotation_to_local(angle: u.Quantity) -> u.Quantity:
    """Convert an undirected DS9 celestial marker axis to our local convention.

    A directed DS9 sky angle maps as ``180 deg - angle``.  Ellipses and
    rectangles are invariant under an additional 180-degree rotation, so
    ``-angle`` is the same physical undirected axis and preserves the existing
    canonical representation for those shapes.
    """
    return -angle


def _ds9_directed_angle_to_local(angle: u.Quantity) -> u.Quantity:
    """Convert a directed DS9 celestial ray/axis to our local convention.

    DS9 celestial 0 degrees points along decreasing longitude, while this
    package's native local 0 degrees points along increasing longitude.  Both
    conventions put +90 degrees toward increasing latitude, so the conversion
    is a reflection about the latitude axis.
    """
    return 180 * u.deg - angle


def _normalised_sector_span(start: u.Quantity, stop: u.Quantity, *, context: str) -> float:
    start_deg = float(start.to_value(u.deg))
    stop_deg = float(stop.to_value(u.deg))
    raw = stop_deg - start_deg
    if abs(raw) > 360.0 + 1e-10:
        raise UnsupportedRegionError(f"{context} angular spans greater than 360 degrees are unsupported")
    sweep = float(np.mod(raw, 360.0))
    if np.isclose(sweep, 0.0, atol=1e-10):
        sweep = 360.0
    return sweep


def _with_sampling_hint(region: CelestialRegion, samples: int) -> CelestialRegion:
    return CelestialRegion(
        include=region.include,
        boundaries=tuple(
            CelestialBoundary(
                boundary.path,
                subtract=boundary.subtract,
                sampling_hint=samples,
            )
            for boundary in region.boundaries
        ),
    )


def _validate_axis_pair(
    major: u.Quantity,
    minor: u.Quantity,
    *,
    context: str,
    allow_zero: bool,
) -> None:
    major_value = float(major.to_value(u.deg))
    minor_value = float(minor.to_value(u.deg))
    if not np.isfinite(major_value) or not np.isfinite(minor_value):
        raise UnsupportedRegionError(f"{context} ellipse axes must be finite")
    major_zero = np.isclose(major_value, 0.0, atol=1e-15)
    minor_zero = np.isclose(minor_value, 0.0, atol=1e-15)
    if major_zero or minor_zero:
        if allow_zero and major_zero and minor_zero:
            return
        raise UnsupportedRegionError(
            f"{context} semimajor and semiminor axes must both be positive or both zero"
        )
    if major_value < 0 or minor_value < 0:
        raise UnsupportedRegionError(f"{context} ellipse axes must be non-negative")


_DS9_SOURCE_SEMANTIC_TOLERANCE = 0.05 * u.arcsec


def _tan_radial_representation_error_arcsec(radius_arcsec: float) -> float:
    """Return the centered TAN-vs-spherical radial disagreement."""
    radius_rad = (radius_arcsec * u.arcsec).to_value(u.rad)
    return float(
        (np.tan(radius_rad) - radius_rad)
        * u.rad.to(u.arcsec)
    )


def _ds9_ellipse_rounding_residual_arcsec(
    first_values: np.ndarray,
    second_values: np.ndarray,
) -> float:
    """Return the part of the total DS9 semantic budget available for repair."""
    total_arcsec = float(
        _DS9_SOURCE_SEMANTIC_TOLERANCE.to_value(u.arcsec)
    )
    declared_extent = max(
        float(np.max(first_values)),
        float(np.max(second_values)),
    )

    # A repaired second axis can move by no more than the total budget. Using
    # declared_extent + total_arcsec therefore conservatively bounds the
    # post-reconciliation radial extent without a circular dependence on the
    # as-yet-unknown repair itself.
    representation_error = _tan_radial_representation_error_arcsec(
        declared_extent + total_arcsec
    )
    return max(0.0, total_arcsec - representation_error)


def _canonicalise_ds9_ellipse_edges(
    edges: Sequence[tuple[u.Quantity, u.Quantity]],
    *,
    context: str,
) -> tuple[tuple[u.Quantity, u.Quantity], ...]:
    """Reconcile finite-precision DS9 nested ellipses within the source budget.

    DS9 region files serialise marker dimensions to finite decimal precision.
    The programmatic geometry API deliberately remains strict; this repair is
    confined to the external DS9-text boundary.

    Every authored first semiaxis is preserved exactly. We choose one common
    second/first-axis ratio only if all declared second semiaxes can be moved
    within the residual of the documented 0.05-arcsec *total* DS9
    source-semantic budget after reserving a conservative allowance for the
    local-reference-plane versus spherical representation difference.
    """
    if len(edges) < 2:
        raise UnsupportedRegionError(f"{context} requires at least two nested ellipses")

    normalised: list[tuple[u.Quantity, u.Quantity]] = []
    positive_indices: list[int] = []
    first_axes_arcsec: list[float] = []
    second_axes_arcsec: list[float] = []

    previous_first = -np.inf
    previous_second = -np.inf

    for index, (first_axis, second_axis) in enumerate(edges):
        _validate_axis_pair(
            first_axis,
            second_axis,
            context=context,
            allow_zero=index == 0,
        )
        first_deg = float(first_axis.to_value(u.deg))
        second_deg = float(second_axis.to_value(u.deg))

        if first_deg <= previous_first or second_deg <= previous_second:
            raise UnsupportedRegionError(
                f"{context} ellipse axes must increase monotonically"
            )

        previous_first = first_deg
        previous_second = second_deg
        normalised.append((first_axis.to(u.deg), second_axis.to(u.deg)))

        if np.isclose(first_deg, 0.0, atol=1e-15):
            continue

        positive_indices.append(index)
        first_axes_arcsec.append(float(first_axis.to_value(u.arcsec)))
        second_axes_arcsec.append(float(second_axis.to_value(u.arcsec)))

    # A zero inner ellipse plus one positive outer ellipse already defines its
    # ellipticity unambiguously and requires no reconciliation.
    if len(first_axes_arcsec) <= 1:
        return tuple(normalised)

    first_values = np.asarray(first_axes_arcsec, dtype=float)
    second_values = np.asarray(second_axes_arcsec, dtype=float)
    ratios = second_values / first_values

    # Preserve existing exact semantic values byte-for-byte where the ordinary
    # strict comparison already succeeds.  The tolerance below is therefore an
    # acceptance boundary, not a source of needless floating-point rewriting.
    if np.all(
        np.isclose(
            ratios,
            ratios[0],
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        return tuple(normalised)

    tolerance_arcsec = _ds9_ellipse_rounding_residual_arcsec(
        first_values,
        second_values,
    )

    # For each serialized ellipse, requiring
    #
    #   |first_axis * ratio - second_axis| <= tolerance
    #
    # gives an allowed interval for the common ratio.  A non-empty
    # intersection is exactly the condition that one common ellipticity can
    # represent every declared ellipse without exceeding the source-geometry
    # budget.
    lower_bounds = (second_values - tolerance_arcsec) / first_values
    upper_bounds = (second_values + tolerance_arcsec) / first_values
    lower = max(0.0, float(np.max(lower_bounds)))
    upper = float(np.min(upper_bounds))

    if lower > upper + 1e-15:
        # Report the best unconstrained common-ratio fit as a useful measure of
        # how far the serialized geometry lies outside the accepted budget.
        best_ratio = float(
            np.dot(first_values, second_values)
            / np.dot(first_values, first_values)
        )
        best_displacement = float(
            np.max(np.abs(first_values * best_ratio - second_values))
        )
        raise UnsupportedRegionError(
            f"{context} nested ellipses must have the same major/minor axis ratio; "
            "DS9 rounding reconciliation would move a boundary by "
            f"{best_displacement:.6g} arcsec, exceeding the "
            f"{tolerance_arcsec:.6g} arcsec residual of the 0.05 arcsec "
            "total source-semantic budget at this ellipse scale"
        )

    # Choose the least-squares common ratio when it satisfies every bound;
    # otherwise use the nearest point in the feasible interval.  This is
    # deterministic and minimises unnecessary movement of the serialized
    # second semiaxes.
    least_squares_ratio = float(
        np.dot(first_values, second_values)
        / np.dot(first_values, first_values)
    )
    common_ratio = float(np.clip(least_squares_ratio, lower, upper))
    canonical_seconds = first_values * common_ratio
    corrections = np.abs(canonical_seconds - second_values)

    if float(np.max(corrections)) > tolerance_arcsec + 1e-10:
        raise UnsupportedRegionError(
            f"{context} DS9 ellipse rounding reconciliation exceeded the "
            f"{tolerance_arcsec:.6g} arcsec residual of the 0.05 arcsec "
            "total source-semantic budget at this ellipse scale"
        )

    reconciled = list(normalised)
    for index, canonical_second in zip(
        positive_indices,
        canonical_seconds,
        strict=True,
    ):
        first_axis, _ = reconciled[index]
        reconciled[index] = (
            first_axis,
            (canonical_second * u.arcsec).to(u.deg),
        )

    return tuple(reconciled)


def _panda_regions(
    *,
    frame: str,
    include: bool,
    arguments: str,
    samples: int,
) -> list[CelestialRegion]:
    fields = [field.strip() for field in arguments.split(",")]
    if len(fields) != 8:
        raise UnsupportedRegionError(
            "DS9 panda requires exactly 8 arguments: "
            "x,y,startangle,stopangle,nangle,inner,outer,nradius"
        )
    x, y, start_text, stop_text, nangle_text, inner_text, outer_text, nradius_text = fields
    center = _parse_ds9_center(frame, x, y, context="DS9 panda")
    start = _parse_angle_token(start_text)
    stop = _parse_angle_token(stop_text)
    nangle = _parse_positive_integer(nangle_text, name="nangle")
    nradius = _parse_positive_integer(nradius_text, name="nradius")
    if nangle > MAX_SOURCE_CELLS // nradius:
        raise UnsupportedRegionError("DS9 subdivision cell resource limit exceeded")
    inner = _parse_radius_token(inner_text).to(u.deg)
    outer = _parse_radius_token(outer_text).to(u.deg)
    if not np.isfinite(inner.value) or not np.isfinite(outer.value):
        raise UnsupportedRegionError("DS9 panda radii must be finite")
    if inner < 0 * u.deg or outer <= inner:
        raise UnsupportedRegionError("DS9 panda requires 0 <= inner radius < outer radius")

    sweep = _normalised_sector_span(start, stop, context="DS9 panda")
    angle_step = sweep / nangle
    radius_step = (outer - inner) / nradius
    start_deg = float(start.to_value(u.deg))

    cells: list[CelestialRegion] = []
    for angular_index in range(nangle):
        ds9_cell_start = (start_deg + angular_index * angle_step) * u.deg
        ds9_cell_stop = (start_deg + (angular_index + 1) * angle_step) * u.deg

        # DS9 and the semantic model use opposite longitude directions.
        # Reflect each directed ray and reverse the endpoints so the same
        # physical wedge is traversed with our positive internal sweep.
        cell_start = _ds9_directed_angle_to_local(ds9_cell_stop)
        cell_stop = _ds9_directed_angle_to_local(ds9_cell_start)

        for radial_index in range(nradius):
            cell_inner = inner + radial_index * radius_step
            cell_outer = inner + (radial_index + 1) * radius_step
            cell = CelestialSelection.sector_annulus(
                center,
                cell_inner,
                cell_outer,
                cell_start,
                cell_stop,
                include=include,
            )
            cells.append(_with_sampling_hint(cell.regions[0], samples))
    return cells


def _epanda_regions(
    *,
    frame: str,
    include: bool,
    arguments: str,
    samples: int,
) -> list[CelestialRegion]:
    fields = [field.strip() for field in arguments.split(",")]
    if len(fields) not in {10, 11}:
        raise UnsupportedRegionError(
            "DS9 epanda requires x,y,startangle,stopangle,nangle,"
            "innerMajor,innerMinor,outerMajor,outerMinor,nradius[,angle]"
        )
    (
        x,
        y,
        start_text,
        stop_text,
        nangle_text,
        inner_major_text,
        inner_minor_text,
        outer_major_text,
        outer_minor_text,
        nradius_text,
        *angle_text,
    ) = fields
    center = _parse_ds9_center(frame, x, y, context="DS9 epanda")
    start = _parse_angle_token(start_text, context="DS9 epanda")
    stop = _parse_angle_token(stop_text, context="DS9 epanda")
    nangle = _parse_positive_integer(nangle_text, name="nangle", context="DS9 epanda")
    nradius = _parse_positive_integer(nradius_text, name="nradius", context="DS9 epanda")
    if nangle > MAX_SOURCE_CELLS // nradius:
        raise UnsupportedRegionError("DS9 subdivision cell resource limit exceeded")
    source_ellipse_angle = (
        _parse_angle_token(angle_text[0], context="DS9 epanda")
        if angle_text
        else 0 * u.deg
    )
    ellipse_axis_angle = _ds9_rotation_to_local(source_ellipse_angle)
    ellipse_directed_angle = _ds9_directed_angle_to_local(source_ellipse_angle)
    inner_major = _parse_radius_token(inner_major_text, context="DS9 epanda").to(u.deg)
    inner_minor = _parse_radius_token(inner_minor_text, context="DS9 epanda").to(u.deg)
    outer_major = _parse_radius_token(outer_major_text, context="DS9 epanda").to(u.deg)
    outer_minor = _parse_radius_token(outer_minor_text, context="DS9 epanda").to(u.deg)
    (
        (inner_major, inner_minor),
        (outer_major, outer_minor),
    ) = _canonicalise_ds9_ellipse_edges(
        ((inner_major, inner_minor), (outer_major, outer_minor)),
        context="DS9 epanda",
    )

    sweep = _normalised_sector_span(start, stop, context="DS9 epanda")
    angle_step = sweep / nangle

    # A full 360-degree epanda is only an undirected ellipse/annulus, for
    # which +/-180 degrees is geometrically irrelevant.  A non-full epanda
    # attaches directed local cuts to one end of the major axis and therefore
    # requires the fully directed DS9 conversion.
    ellipse_angle = (
        ellipse_axis_angle
        if angle_step == 360.0
        else ellipse_directed_angle
    )
    major_edges = np.linspace(
        inner_major.to_value(u.deg), outer_major.to_value(u.deg), nradius + 1
    ) * u.deg
    minor_edges = np.linspace(
        inner_minor.to_value(u.deg), outer_minor.to_value(u.deg), nradius + 1
    ) * u.deg
    start_deg = float(start.to_value(u.deg))

    cells: list[CelestialRegion] = []
    for angular_index in range(nangle):
        cell_start = (start_deg + angular_index * angle_step) * u.deg
        cell_stop = (start_deg + (angular_index + 1) * angle_step) * u.deg
        for radial_index in range(nradius):
            cell = _ordered_axis_local_sector_selection(
                center,
                major_edges[radial_index],
                minor_edges[radial_index],
                major_edges[radial_index + 1],
                minor_edges[radial_index + 1],
                cell_start,
                cell_stop,
                ellipse_angle,
                include=include,
            )
            cells.append(_with_sampling_hint(cell.regions[0], samples))
    return cells


def _annulus_n_regions(
    *,
    frame: str,
    include: bool,
    arguments: str,
    samples: int,
) -> list[CelestialRegion]:
    fields = [field.strip() for field in arguments.split(",")]
    if len(fields) != 5 or not fields[-1].lower().startswith("n="):
        raise UnsupportedRegionError("DS9 annulus n= syntax requires x,y,inner,outer,n=N")
    x, y, inner_text, outer_text, n_text = fields
    center = _parse_ds9_center(frame, x, y, context="DS9 annulus")
    inner = _parse_radius_token(inner_text, context="DS9 annulus").to(u.deg)
    outer = _parse_radius_token(outer_text, context="DS9 annulus").to(u.deg)
    n = _parse_positive_integer(
        n_text.split("=", maxsplit=1)[1],
        name="n",
        context="DS9 annulus",
    )
    if not np.isfinite(inner.value) or not np.isfinite(outer.value):
        raise UnsupportedRegionError("DS9 annulus radii must be finite")
    if inner < 0 * u.deg or outer <= inner:
        raise UnsupportedRegionError("DS9 annulus requires 0 <= inner radius < outer radius")

    edges = np.linspace(inner.to_value(u.deg), outer.to_value(u.deg), n + 1) * u.deg
    cells: list[CelestialRegion] = []
    for index in range(n):
        cell = CelestialSelection.sector_annulus(
            center,
            edges[index],
            edges[index + 1],
            0 * u.deg,
            360 * u.deg,
            include=include,
        )
        cells.append(_with_sampling_hint(cell.regions[0], samples))
    return cells


def _ellipse_annulus_regions(
    *,
    frame: str,
    include: bool,
    arguments: str,
    samples: int,
) -> list[CelestialRegion]:
    fields = [field.strip() for field in arguments.split(",")]
    if len(fields) < 6:
        raise UnsupportedRegionError("DS9 ellipse annulus requires at least two semiaxis pairs")
    x, y, *rest = fields
    center = _parse_ds9_center(frame, x, y, context="DS9 ellipse annulus")

    n_positions = [index for index, value in enumerate(rest) if value.lower().startswith("n=")]
    angle = 0 * u.deg
    if n_positions:
        if len(n_positions) != 1 or n_positions[0] != 4 or len(rest) not in {5, 6}:
            raise UnsupportedRegionError(
                "DS9 ellipse annulus n= syntax requires r11,r12,r21,r22,n=N[,angle]"
            )
        n = _parse_positive_integer(
            rest[4].split("=", maxsplit=1)[1],
            name="n",
            context="DS9 ellipse annulus",
        )
        if len(rest) == 6:
            angle = _parse_angle_token(rest[5], context="DS9 ellipse annulus")
        inner_major = _parse_radius_token(rest[0], context="DS9 ellipse annulus").to(u.deg)
        inner_minor = _parse_radius_token(rest[1], context="DS9 ellipse annulus").to(u.deg)
        outer_major = _parse_radius_token(rest[2], context="DS9 ellipse annulus").to(u.deg)
        outer_minor = _parse_radius_token(rest[3], context="DS9 ellipse annulus").to(u.deg)
        (
            (inner_major, inner_minor),
            (outer_major, outer_minor),
        ) = _canonicalise_ds9_ellipse_edges(
            ((inner_major, inner_minor), (outer_major, outer_minor)),
            context="DS9 ellipse annulus",
        )
        major_values = np.linspace(
            inner_major.to_value(u.deg), outer_major.to_value(u.deg), n + 1
        )
        minor_values = np.linspace(
            inner_minor.to_value(u.deg), outer_minor.to_value(u.deg), n + 1
        )
        edges = tuple(
            (major * u.deg, minor * u.deg)
            for major, minor in zip(major_values, minor_values, strict=True)
        )
    else:
        radius_tokens = rest
        if len(radius_tokens) % 2 == 1:
            angle = _parse_angle_token(radius_tokens[-1], context="DS9 ellipse annulus")
            radius_tokens = radius_tokens[:-1]
        if len(radius_tokens) < 4 or len(radius_tokens) % 2:
            raise UnsupportedRegionError(
                "DS9 ellipse annulus requires two or more semiaxis pairs plus optional angle"
            )
        edges = tuple(
            (
                _parse_radius_token(radius_tokens[index], context="DS9 ellipse annulus").to(u.deg),
                _parse_radius_token(
                    radius_tokens[index + 1], context="DS9 ellipse annulus"
                ).to(u.deg),
            )
            for index in range(0, len(radius_tokens), 2)
        )
        edges = _canonicalise_ds9_ellipse_edges(
            edges,
            context="DS9 ellipse annulus",
        )

    angle = _ds9_rotation_to_local(angle)
    cells: list[CelestialRegion] = []
    for inner, outer in zip(edges[:-1], edges[1:], strict=True):
        boundaries = [
            CelestialBoundary(
                _ellipse_path(center, 2 * outer[0], 2 * outer[1], angle),
                sampling_hint=samples,
            )
        ]
        inner_zero = np.isclose(inner[0].to_value(u.deg), 0.0, atol=1e-15)
        if not inner_zero:
            boundaries.append(
                CelestialBoundary(
                    _ellipse_path(center, 2 * inner[0], 2 * inner[1], angle),
                    subtract=True,
                    sampling_hint=samples,
                )
            )
        cells.append(CelestialRegion(include=include, boundaries=tuple(boundaries)))
    return cells


def _validate_extension_frame(current_frame: str | None, *, context: str) -> str:
    if current_frame is None:
        raise UnsupportedRegionError(
            f"{context} requires an explicit supported celestial frame such as fk5 or icrs"
        )
    if current_frame in _PIXEL_FRAMES:
        raise TypeError(f"{context} must use sky coordinates, not pixel/image coordinates")
    if current_frame not in _FRAME_ALIASES.values():
        raise UnsupportedRegionError(f"unsupported {context} celestial frame {current_frame!r}")
    return current_frame


def _load_ds9_with_extensions(
    text: str,
    *,
    samples: int,
    external_identity: str | None,
    external_provenance: Mapping[str, Any] | None,
) -> CelestialSelection:
    current_frame: str | None = None
    ordinary_statements: list[str] = []

    # Preserve the authored DS9 statement order. Ordinary regions are parsed
    # together by astropy-regions so frame/global context remains intact, but
    # each ordinary region gets a placeholder here. Custom constructs expand
    # directly in place.
    ordered_regions: list[CelestialRegion | None] = []
    has_extension = False

    for statement in _split_ds9_code_statements(text):
        if len(ordered_regions) > MAX_SOURCE_CELLS:
            raise UnsupportedRegionError("DS9 expanded cell resource limit exceeded")
        lowered = statement.lower().strip()

        if lowered in _FRAME_ALIASES:
            current_frame = _FRAME_ALIASES[lowered]
            ordinary_statements.append(statement)
            continue

        if lowered in _PIXEL_FRAMES:
            current_frame = lowered
            ordinary_statements.append(statement)
            continue

        # DS9 global property statements affect subsequent ordinary parser
        # behaviour but do not themselves correspond to a science region.
        if lowered.startswith("global "):
            ordinary_statements.append(statement)
            continue

        bpanda_match = _BPANDA_RE.match(statement)
        if bpanda_match is not None:
            raise UnsupportedRegionError(
                "DS9 bpanda is not supported; use panda/epanda or supply supported geometry"
            )

        annulus_match = _ANNULUS_RE.match(statement)
        if annulus_match is not None:
            annulus_fields = [
                field.strip() for field in annulus_match.group(2).split(",")
            ]
            if annulus_fields and annulus_fields[-1].lower().startswith("n="):
                has_extension = True
                frame = _validate_extension_frame(
                    current_frame,
                    context="DS9 annulus n=",
                )
                ordered_regions.extend(
                    _annulus_n_regions(
                        frame=frame,
                        include=annulus_match.group(1) != "-",
                        arguments=annulus_match.group(2),
                        samples=samples,
                    )
                )
                continue

        ellipse_match = _ELLIPSE_RE.match(statement)
        if ellipse_match is not None:
            ellipse_fields = [
                field.strip() for field in ellipse_match.group(2).split(",")
            ]
            is_multi_ellipse = len(ellipse_fields) >= 6 or any(
                value.lower().startswith("n=") for value in ellipse_fields
            )
            if is_multi_ellipse:
                has_extension = True
                frame = _validate_extension_frame(
                    current_frame,
                    context="DS9 ellipse annulus",
                )
                ordered_regions.extend(
                    _ellipse_annulus_regions(
                        frame=frame,
                        include=ellipse_match.group(1) != "-",
                        arguments=ellipse_match.group(2),
                        samples=samples,
                    )
                )
                continue

        epanda_match = _EPANDA_RE.match(statement)
        if epanda_match is not None:
            has_extension = True
            frame = _validate_extension_frame(current_frame, context="DS9 epanda")
            ordered_regions.extend(
                _epanda_regions(
                    frame=frame,
                    include=epanda_match.group(1) != "-",
                    arguments=epanda_match.group(2),
                    samples=samples,
                )
            )
            continue

        panda_match = _PANDA_RE.match(statement)
        if panda_match is not None:
            has_extension = True
            frame = _validate_extension_frame(current_frame, context="DS9 panda")
            ordered_regions.extend(
                _panda_regions(
                    frame=frame,
                    include=panda_match.group(1) != "-",
                    arguments=panda_match.group(2),
                    samples=samples,
                )
            )
            continue

        # Keep the statement in the context-aware ordinary parse and remember
        # exactly where its resulting region belongs in the authored stream.
        ordinary_statements.append(statement)
        ordered_regions.append(None)

    if not has_extension:
        return _selection_from_sky_regions(
            _parse_ds9_sky_text(text),
            samples=samples,
            external_identity=external_identity,
            external_provenance=external_provenance,
            ds9_rotation_angles=True,
        )

    ordinary_text = ";\n".join(ordinary_statements) + "\n"
    ordinary_source = _parse_ds9_sky_text(ordinary_text)

    ordinary: list[CelestialRegion] = []
    for region in ordinary_source:
        include = bool(region.meta.get("include", True))
        ordinary.append(
            CelestialRegion(
                include=include,
                boundaries=_boundaries(
                    region,
                    samples,
                    ds9_rotation_angles=True,
                ),
            )
        )

    placeholder_count = sum(region is None for region in ordered_regions)
    if len(ordinary) != placeholder_count:
        raise UnsupportedRegionError(
            "DS9 parser region count changed while preserving extension order; "
            "refusing an ambiguous extraction-cell mapping"
        )

    ordinary_iter = iter(ordinary)
    regions = tuple(
        next(ordinary_iter) if region is None else region
        for region in ordered_regions
    )

    if not regions:
        raise ValueError("DS9 region file contains no regions")

    return CelestialSelection(
        regions=regions,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def load_ds9_selection(
    path: str | Path,
    *,
    samples: int = 128,
    external_identity: str | None = None,
    external_provenance: Mapping[str, Any] | None = None,
) -> CelestialSelection:
    samples = integer_limit(samples, "samples", 16, MAX_SAMPLES)
    return _load_ds9_with_extensions(
        _read_source_text(Path(path)),
        samples=samples,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def load_ds9_sky_regions(path: str | Path, *, samples: int = 128) -> list[CelestialRegion]:
    return list(load_ds9_selection(path, samples=samples).regions)


def _offset_vertices(center: SkyCoord, x: u.Quantity, y: u.Quantity) -> SkyCoord:
    separation = np.hypot(x, y)
    position_angle = np.arctan2(x.to_value(u.deg), y.to_value(u.deg)) * u.rad
    return center.directional_offset_by(position_angle, separation)


def _rectangle_vertices(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    angle: u.Quantity,
) -> SkyCoord:
    from .model import _axis_angle

    angle = _axis_angle(angle)
    x = np.array([-0.5, 0.5, 0.5, -0.5]) * width
    y = np.array([-0.5, -0.5, 0.5, 0.5]) * height
    rotation = angle.to_value(u.rad)
    xr = x * np.cos(rotation) - y * np.sin(rotation)
    yr = x * np.sin(rotation) + y * np.cos(rotation)
    return _offset_vertices(center, xr, yr)


def _ellipse_path(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    angle: u.Quantity,
) -> EllipsePath:
    """Normalise an ellipse to ICRS without rotating its physical sky axes."""
    from .model import _axis_angle_to_icrs

    return EllipsePath(center.icrs, width, height, _axis_angle_to_icrs(center, angle))


def _polygon_boundary(
    vertices: SkyCoord,
    *,
    subtract: bool = False,
    samples: int,
) -> CelestialBoundary:
    return CelestialBoundary(
        GeodesicPolygonPath(vertices),
        subtract=subtract,
        sampling_hint=samples,
    )


def _rotation_for_region(region: SkyRegion, *, ds9_rotation_angles: bool) -> u.Quantity:
    angle = region.angle
    if ds9_rotation_angles:
        return _ds9_rotation_to_local(angle)
    return angle


def _boundaries(
    region: SkyRegion,
    samples: int,
    *,
    ds9_rotation_angles: bool = False,
) -> tuple[CelestialBoundary, ...]:
    if isinstance(region, CircleSkyRegion):
        return (
            CelestialBoundary(
                CirclePath(region.center, region.radius),
                sampling_hint=samples,
            ),
        )
    if isinstance(region, EllipseSkyRegion):
        angle = _rotation_for_region(region, ds9_rotation_angles=ds9_rotation_angles)
        return (
            CelestialBoundary(
                _ellipse_path(region.center, region.width, region.height, angle),
                sampling_hint=samples,
            ),
        )
    if isinstance(region, RectangleSkyRegion):
        angle = _rotation_for_region(region, ds9_rotation_angles=ds9_rotation_angles)
        return (
            _polygon_boundary(
                _rectangle_vertices(region.center, region.width, region.height, angle),
                samples=samples,
            ),
        )
    if isinstance(region, PolygonSkyRegion):
        return (_polygon_boundary(region.vertices, samples=samples),)
    if isinstance(region, CircleAnnulusSkyRegion):
        return (
            CelestialBoundary(
                CirclePath(region.center, region.outer_radius),
                sampling_hint=samples,
            ),
            CelestialBoundary(
                CirclePath(region.center, region.inner_radius),
                subtract=True,
                sampling_hint=samples,
            ),
        )
    if isinstance(region, EllipseAnnulusSkyRegion):
        angle = _rotation_for_region(region, ds9_rotation_angles=ds9_rotation_angles)
        return (
            CelestialBoundary(
                _ellipse_path(
                    region.center,
                    region.outer_width,
                    region.outer_height,
                    angle,
                ),
                sampling_hint=samples,
            ),
            CelestialBoundary(
                _ellipse_path(
                    region.center,
                    region.inner_width,
                    region.inner_height,
                    angle,
                ),
                subtract=True,
                sampling_hint=samples,
            ),
        )
    if isinstance(region, RectangleAnnulusSkyRegion):
        angle = _rotation_for_region(region, ds9_rotation_angles=ds9_rotation_angles)
        return (
            _polygon_boundary(
                _rectangle_vertices(
                    region.center,
                    region.outer_width,
                    region.outer_height,
                    angle,
                ),
                samples=samples,
            ),
            _polygon_boundary(
                _rectangle_vertices(
                    region.center,
                    region.inner_width,
                    region.inner_height,
                    angle,
                ),
                subtract=True,
                samples=samples,
            ),
        )
    raise UnsupportedRegionError(
        f"{type(region).__name__} is not yet supported; refusing to approximate it silently"
    )

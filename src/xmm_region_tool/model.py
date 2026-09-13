"""Stable in-memory geometry models for library callers.

DS9 is an input adapter and SAS is the calibrated projection backend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .immutable import freeze_json, plain_json
from .limits import MAX_SAMPLES, MAX_SOURCE_CELLS, MAX_SOURCE_VERTICES, integer_limit


class SelectionGeometryError(ValueError):
    """Raised when an in-memory selection is geometrically invalid."""


def _science_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SelectionGeometryError(f"{name} must be a Boolean")
    return bool(value)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalised_scalar_icrs(coord: SkyCoord) -> SkyCoord:
    try:
        icrs = coord.icrs
    except Exception as exc:
        raise SelectionGeometryError(f"cannot transform celestial coordinate to ICRS: {exc}") from exc
    if not icrs.isscalar:
        raise SelectionGeometryError("celestial path centre must be a scalar SkyCoord")
    if not np.isfinite(float(icrs.ra.deg)) or not np.isfinite(float(icrs.dec.deg)):
        raise SelectionGeometryError("celestial path centre must be finite")
    return icrs


def _normalised_icrs_vertices(vertices: SkyCoord) -> SkyCoord:
    try:
        icrs = vertices.icrs
    except Exception as exc:
        raise SelectionGeometryError(f"cannot transform celestial vertices to ICRS: {exc}") from exc
    if icrs.isscalar or len(icrs) < 3:
        raise SelectionGeometryError("a celestial boundary requires at least three vertices")
    if icrs.ndim != 1 or len(icrs) > MAX_SOURCE_VERTICES:
        raise SelectionGeometryError("celestial vertices require a 1-D sequence within the resource limit")
    ra = np.asarray(icrs.ra.deg, dtype=float)
    dec = np.asarray(icrs.dec.deg, dtype=float)
    if not np.isfinite(ra).all() or not np.isfinite(dec).all():
        raise SelectionGeometryError("celestial boundary vertices must be finite")
    return icrs


def _positive_angle(value: u.Quantity, *, name: str) -> u.Quantity:
    angle = _finite_angle(value, name=name)
    if angle <= 0 * u.deg:
        raise SelectionGeometryError(f"{name} must be finite and positive")
    return angle


def _nonnegative_angle(value: u.Quantity, *, name: str) -> u.Quantity:
    angle = _finite_angle(value, name=name)
    if angle < 0 * u.deg:
        raise SelectionGeometryError(f"{name} must be finite and non-negative")
    return angle


def _finite_angle(value: u.Quantity, *, name: str) -> u.Quantity:
    try:
        angle = value.to(u.deg)
    except Exception as exc:
        raise SelectionGeometryError(f"{name} must be an angular quantity") from exc
    if not angle.isscalar:
        raise SelectionGeometryError(f"{name} must be a scalar angular quantity")
    if not np.isfinite(angle.value):
        raise SelectionGeometryError(f"{name} must be finite")
    return angle


def _normalised_angle(value: u.Quantity) -> u.Quantity:
    angle = _finite_angle(value, name="angle")
    return float(np.mod(angle.to_value(u.deg), 360.0)) * u.deg


def _axis_angle(value: u.Quantity) -> u.Quantity:
    return float(np.mod(_finite_angle(value, name="axis angle").to_value(u.deg), 180.0)) * u.deg


def _axis_angle_to_icrs(center: SkyCoord, angle: u.Quantity) -> u.Quantity:
    return _axis_angle(local_angle_to_icrs(center, _axis_angle(angle)))


def _validate_offset_extent(value: u.Quantity) -> None:
    if value >= 180 * u.deg:
        raise SelectionGeometryError("spherical offset must be strictly below 180 degrees")


def _validate_local_basis(center: SkyCoord) -> None:
    _normalised_scalar_icrs(center)
    if abs(center.spherical.lat.to_value(u.deg)) == 90.0:
        raise SelectionGeometryError("undefined local-angle basis at a source-frame pole")


class _CoordinateSnapshot:
    """Store only immutable ICRS primitives; Astropy access always returns a copy."""

    def __init__(self, *, scalar: bool = True):
        self.scalar = scalar

    def __set_name__(self, owner, name):
        self.storage = "_snapshot_" + name

    def __get__(self, instance, owner=None):
        if instance is None:
            raise AttributeError(self.storage)
        ra, dec = instance.__dict__[self.storage]
        if not self.scalar:
            ra, dec = list(ra), list(dec)
        return SkyCoord(ra, dec, unit=u.deg, frame="icrs")

    def __set__(self, instance, value):
        coords = (_normalised_scalar_icrs(value) if self.scalar
                  else _normalised_icrs_vertices(value))
        if self.scalar:
            payload = (float(coords.ra.deg), float(coords.dec.deg))
        else:
            if coords.ndim != 1:
                raise SelectionGeometryError("celestial vertices must be a 1-D sequence")
            payload = (tuple(float(v) for v in coords.ra.deg),
                       tuple(float(v) for v in coords.dec.deg))
        instance.__dict__[self.storage] = payload


class _AngleSnapshot:
    """Store a scalar degree value, exposing detached angular quantities."""

    def __init__(self, default=None):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name
        self.storage = "_snapshot_" + name

    def __get__(self, instance, owner=None):
        if instance is None:
            if self.default is None:
                raise AttributeError(self.storage)
            return self.default * u.deg
        return instance.__dict__[self.storage] * u.deg

    def __set__(self, instance, value):
        instance.__dict__[self.storage] = float(
            _finite_angle(value, name=self.name).to_value(u.deg)
        )


class _DetectorSnapshot:
    """Immutable coordinates with detached ndarray shape/stride metadata."""

    def __set_name__(self, owner, name):
        self.storage = "_snapshot_" + name

    def __get__(self, instance, owner=None):
        if instance is None:
            raise AttributeError(self.storage)
        data, shape = instance.__dict__[self.storage]
        return np.frombuffer(data, dtype=float).reshape(shape)

    def __set__(self, instance, value):
        vertices = np.asarray(value, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise SelectionGeometryError("a detector boundary requires at least three XY vertices")
        if not np.isfinite(vertices).all():
            raise SelectionGeometryError("detector boundary vertices must be finite")
        instance.__dict__[self.storage] = (vertices.tobytes(), vertices.shape)


def _sector_sweep(start: u.Quantity, stop: u.Quantity) -> u.Quantity:
    start_deg = _finite_angle(start, name="sector start angle").to_value(u.deg)
    stop_deg = _finite_angle(stop, name="sector stop angle").to_value(u.deg)
    raw = float(stop_deg - start_deg)
    if abs(raw) > 360.0 + 1e-10:
        raise SelectionGeometryError("sector angular span greater than 360 degrees is unsupported")
    sweep = float(np.mod(raw, 360.0))
    if sweep == 0.0:
        sweep = 360.0
    return sweep * u.deg


def local_angle_to_icrs(center: SkyCoord, angle: u.Quantity) -> u.Quantity:
    """Transform a source-frame local +longitude-axis angle to ICRS.

    DS9/``regions`` sky angles are measured anticlockwise from the local
    longitude axis of the declared celestial frame.  Transforming only the
    centre and reusing the numerical angle can rotate a physical region because
    different frames have different local tangent bases.  This helper maps one
    short axis direction and recovers its equivalent ICRS angle.
    """
    source_angle = _finite_angle(angle, name="local celestial angle")
    _validate_local_basis(center)
    rotation = source_angle.to_value(u.rad)
    axis_pa = np.arctan2(np.cos(rotation), np.sin(rotation)) * u.rad
    axis_point = center.directional_offset_by(axis_pa, 1.0 * u.arcsec)
    center_icrs = _normalised_scalar_icrs(center)
    axis_pa_icrs = center_icrs.position_angle(axis_point.icrs)
    return _normalised_angle(90.0 * u.deg - axis_pa_icrs)


@dataclass(frozen=True)
class GeodesicPolygonPath:
    """Closed polygon whose celestial edges are shortest great-circle arcs."""

    vertices: SkyCoord = _CoordinateSnapshot(scalar=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", _normalised_icrs_vertices(self.vertices))
        from .spherical import validate_geodesic_vertices

        validate_geodesic_vertices(self.vertices)

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "geodesic-polygon",
            "edge_convention": "shortest-great-circle",
            "vertices_icrs_deg": [
                [float(ra), float(dec)]
                for ra, dec in zip(self.vertices.ra.deg, self.vertices.dec.deg, strict=True)
            ],
        }

    def initial_intervals(self) -> tuple[tuple[float, float], ...]:
        n = len(self.vertices)
        return tuple((index / n, (index + 1) / n) for index in range(n))

    def point(self, parameter: float) -> SkyCoord:
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("path parameter must lie in [0, 1]")
        n = len(self.vertices)
        scaled = parameter * n
        if parameter == 1.0:
            index = n - 1
            fraction = 1.0
        else:
            index = min(int(np.floor(scaled)), n - 1)
            fraction = scaled - index
        start = self.vertices[index]
        end = self.vertices[(index + 1) % n]
        separation = start.separation(end)
        position_angle = start.position_angle(end)
        return start.directional_offset_by(position_angle, separation * fraction).icrs


@dataclass(frozen=True)
class CirclePath:
    """Exact small-circle boundary on the celestial sphere."""

    center: SkyCoord = _CoordinateSnapshot()
    radius: u.Quantity = _AngleSnapshot()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _normalised_scalar_icrs(self.center))
        object.__setattr__(self, "radius", _positive_angle(self.radius, name="circle radius"))
        _validate_offset_extent(self.radius)

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "circle",
            "center_icrs_deg": [float(self.center.ra.deg), float(self.center.dec.deg)],
            "radius_deg": float(self.radius.to_value(u.deg)),
        }

    def initial_intervals(self) -> tuple[tuple[float, float], ...]:
        return tuple((index / 4.0, (index + 1) / 4.0) for index in range(4))

    def point(self, parameter: float) -> SkyCoord:
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("path parameter must lie in [0, 1]")
        angle = (2.0 * np.pi * parameter) * u.rad
        return self.center.directional_offset_by(angle, self.radius).icrs

    def sample(self, samples: int) -> SkyCoord:
        samples = integer_limit(samples, "samples", 4, MAX_SAMPLES)
        parameters = np.linspace(0.0, 1.0, samples, endpoint=False)
        return self.center.directional_offset_by(parameters * 2.0 * np.pi * u.rad, self.radius).icrs


@dataclass(frozen=True)
class EllipsePath:
    """Exact tangent-plane ellipse under the astropy-regions sky convention."""

    center: SkyCoord = _CoordinateSnapshot()
    width: u.Quantity = _AngleSnapshot()
    height: u.Quantity = _AngleSnapshot()
    angle: u.Quantity = _AngleSnapshot()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _normalised_scalar_icrs(self.center))
        object.__setattr__(self, "width", _positive_angle(self.width, name="ellipse width"))
        object.__setattr__(self, "height", _positive_angle(self.height, name="ellipse height"))
        object.__setattr__(self, "angle", _axis_angle(self.angle))
        _validate_offset_extent(max(self.width, self.height) / 2)

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "ellipse",
            "center_icrs_deg": [float(self.center.ra.deg), float(self.center.dec.deg)],
            "width_deg": float(self.width.to_value(u.deg)),
            "height_deg": float(self.height.to_value(u.deg)),
            "angle_deg": float(self.angle.to_value(u.deg)),
            "convention": "regions-longitude-axis-anticlockwise",
        }

    def initial_intervals(self) -> tuple[tuple[float, float], ...]:
        return tuple((index / 4.0, (index + 1) / 4.0) for index in range(4))

    def point(self, parameter: float) -> SkyCoord:
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("path parameter must lie in [0, 1]")
        theta = 2.0 * np.pi * parameter
        x = (self.width / 2.0) * np.cos(theta)
        y = (self.height / 2.0) * np.sin(theta)
        rotation = self.angle.to_value(u.rad)
        xr = x * np.cos(rotation) - y * np.sin(rotation)
        yr = x * np.sin(rotation) + y * np.cos(rotation)
        separation = np.hypot(xr, yr)
        position_angle = np.arctan2(xr.to_value(u.deg), yr.to_value(u.deg)) * u.rad
        return self.center.directional_offset_by(position_angle, separation).icrs

    def sample(self, samples: int) -> SkyCoord:
        samples = integer_limit(samples, "samples", 4, MAX_SAMPLES)
        return SkyCoord([self.point(float(value)) for value in np.linspace(0, 1, samples, endpoint=False)])


@dataclass(frozen=True)
class SectorAnnulusPath:
    """One non-full circular annular sector in celestial coordinates.

    ``start_angle`` follows the same ICRS local-longitude-axis convention used
    by :class:`EllipsePath`.  ``sweep`` is positive and strictly less than 360
    degrees.  A full 360-degree request is represented by ordinary circle/
    annulus boundaries instead so no coincident radial edges are introduced.
    """

    center: SkyCoord = _CoordinateSnapshot()
    inner_radius: u.Quantity = _AngleSnapshot()
    outer_radius: u.Quantity = _AngleSnapshot()
    start_angle: u.Quantity = _AngleSnapshot()
    sweep: u.Quantity = _AngleSnapshot()

    def __post_init__(self) -> None:
        center = _normalised_scalar_icrs(self.center)
        inner = _nonnegative_angle(self.inner_radius, name="sector inner radius")
        outer = _positive_angle(self.outer_radius, name="sector outer radius")
        _validate_offset_extent(outer)
        start = _normalised_angle(self.start_angle)
        sweep = _positive_angle(self.sweep, name="sector angular sweep")
        if outer <= inner:
            raise SelectionGeometryError("sector outer radius must exceed inner radius")
        if sweep >= 360.0 * u.deg:
            raise SelectionGeometryError(
                "SectorAnnulusPath requires a sweep below 360 degrees; use circle/annulus geometry"
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "inner_radius", inner)
        object.__setattr__(self, "outer_radius", outer)
        object.__setattr__(self, "start_angle", start)
        object.__setattr__(self, "sweep", sweep)

    @property
    def segment_count(self) -> int:
        return 3 if self.inner_radius.to_value(u.deg) == 0.0 else 4

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "sector-annulus",
            "center_icrs_deg": [float(self.center.ra.deg), float(self.center.dec.deg)],
            "inner_radius_deg": float(self.inner_radius.to_value(u.deg)),
            "outer_radius_deg": float(self.outer_radius.to_value(u.deg)),
            "start_angle_deg": float(self.start_angle.to_value(u.deg)),
            "sweep_deg": float(self.sweep.to_value(u.deg)),
            "angle_convention": "regions-longitude-axis-anticlockwise",
            "radial_edge_convention": "great-circle-from-center",
        }

    def initial_intervals(self) -> tuple[tuple[float, float], ...]:
        n = self.segment_count
        return tuple((index / n, (index + 1) / n) for index in range(n))

    def _at(self, radius: u.Quantity, angle: u.Quantity) -> SkyCoord:
        position_angle = 90.0 * u.deg - angle
        return self.center.directional_offset_by(position_angle, radius).icrs

    def point(self, parameter: float) -> SkyCoord:
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("path parameter must lie in [0, 1]")
        n = self.segment_count
        scaled = parameter * n
        if parameter == 1.0:
            segment = n - 1
            fraction = 1.0
        else:
            segment = min(int(np.floor(scaled)), n - 1)
            fraction = scaled - segment

        start = self.start_angle
        stop = self.start_angle + self.sweep
        if segment == 0:
            return self._at(self.outer_radius, start + self.sweep * fraction)
        if segment == 1:
            radius = self.outer_radius - (self.outer_radius - self.inner_radius) * fraction
            return self._at(radius, stop)
        if self.segment_count == 3:
            radius = self.inner_radius + (self.outer_radius - self.inner_radius) * fraction
            return self._at(radius, start)
        if segment == 2:
            return self._at(self.inner_radius, stop - self.sweep * fraction)
        radius = self.inner_radius + (self.outer_radius - self.inner_radius) * fraction
        return self._at(radius, start)

    def sample(self, samples: int) -> SkyCoord:
        samples = integer_limit(samples, "samples", 4, MAX_SAMPLES)
        if samples < self.segment_count:
            raise SelectionGeometryError(
                f"sector sampling requires at least {self.segment_count} points"
            )
        parameters = np.linspace(0.0, 1.0, samples, endpoint=False)
        return SkyCoord([self.point(float(value)) for value in parameters])


CelestialPath = GeodesicPolygonPath | CirclePath | EllipsePath | SectorAnnulusPath


@dataclass(frozen=True)
class CelestialBoundary:
    """One exact celestial boundary; ``subtract`` marks a component hole.

    ``sampling_hint`` exists only for compatibility/diagnostic rendering. It is
    deliberately excluded from the semantic geometry identity and is not the
    detector-projection accuracy rule.
    """

    path: CelestialPath | SkyCoord
    subtract: bool = False
    sampling_hint: int = 128

    def __post_init__(self) -> None:
        path = self.path
        object.__setattr__(self, "subtract", _science_bool(self.subtract, name="subtract"))
        if isinstance(path, SkyCoord):
            path = GeodesicPolygonPath(path)
        if not isinstance(path, (GeodesicPolygonPath, CirclePath, EllipsePath, SectorAnnulusPath)):
            raise SelectionGeometryError(f"unsupported celestial path type {type(path).__name__}")
        try:
            hint = integer_limit(self.sampling_hint, "sampling_hint", 4, MAX_SAMPLES)
        except ValueError as exc:
            raise SelectionGeometryError(str(exc)) from exc
        object.__setattr__(self, "sampling_hint", hint)
        object.__setattr__(self, "path", path)

    @property
    def vertices(self) -> SkyCoord:
        """Compatibility sampling; projection code must use ``path`` directly."""
        if isinstance(self.path, GeodesicPolygonPath):
            return self.path.vertices
        return self.path.sample(self.sampling_hint)

    def canonical_record(self) -> dict[str, object]:
        return {
            "subtract": bool(self.subtract),
            "path": self.path.canonical_record(),
        }


@dataclass(frozen=True)
class CelestialRegion:
    """One included or excluded celestial component with optional holes."""

    include: bool
    boundaries: tuple[CelestialBoundary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "include", _science_bool(self.include, name="include"))
        boundaries = tuple(self.boundaries)
        if not boundaries:
            raise SelectionGeometryError("a celestial region requires at least one boundary")
        if boundaries[0].subtract:
            raise SelectionGeometryError("the outer celestial boundary cannot be subtractive")
        if any(not boundary.subtract for boundary in boundaries[1:]):
            raise SelectionGeometryError(
                "additional celestial boundaries must be subtractive holes"
            )
        object.__setattr__(self, "boundaries", boundaries)

    def canonical_record(self) -> dict[str, object]:
        return {
            "include": bool(self.include),
            "boundaries": [boundary.canonical_record() for boundary in self.boundaries],
        }


@dataclass(frozen=True)
class CelestialSelection:
    """Public, path-independent semantic celestial selection."""

    regions: tuple[CelestialRegion, ...]
    external_identity: str | None = None
    external_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions:
            raise SelectionGeometryError("a celestial selection requires at least one region")
        if len(regions) > MAX_SOURCE_CELLS:
            raise SelectionGeometryError("source cell resource limit exceeded")
        object.__setattr__(self, "regions", regions)
        if self.external_identity is not None and not isinstance(self.external_identity, str):
            raise SelectionGeometryError("external_identity must be a string or None")
        if self.external_provenance is not None:
            if not isinstance(self.external_provenance, Mapping):
                raise SelectionGeometryError("external_provenance must be a mapping")
            try:
                frozen = freeze_json(self.external_provenance)
            except ValueError as exc:
                raise SelectionGeometryError(str(exc)) from exc
            object.__setattr__(self, "external_provenance", frozen)

    def external_provenance_record(self) -> dict[str, object] | None:
        """Return detached JSON metadata; it never participates in geometry identity."""
        return plain_json(self.external_provenance)

    def canonical_geometry_record(self) -> dict[str, object]:
        return {
            "schema": "xmm-region-tool.celestial-selection/v2",
            "coordinate_frame": "icrs",
            "regions": [region.canonical_record() for region in self.regions],
        }

    @property
    def geometry_sha256(self) -> str:
        return _canonical_sha256(self.canonical_geometry_record())

    @classmethod
    def polygon(
        cls,
        vertices: SkyCoord,
        *,
        include: bool = True,
        holes: Sequence[SkyCoord] = (),
        external_identity: str | None = None,
        external_provenance: Mapping[str, Any] | None = None,
    ) -> CelestialSelection:
        """Construct one geodesic polygon component without a DS9 round trip."""
        boundaries = [CelestialBoundary(GeodesicPolygonPath(vertices))]
        boundaries.extend(
            CelestialBoundary(GeodesicPolygonPath(hole), subtract=True) for hole in holes
        )
        return cls(
            regions=(CelestialRegion(include=include, boundaries=tuple(boundaries)),),
            external_identity=external_identity,
            external_provenance=external_provenance,
        )

    @classmethod
    def sector_annulus(
        cls,
        center: SkyCoord,
        inner_radius: u.Quantity,
        outer_radius: u.Quantity,
        start_angle: u.Quantity,
        stop_angle: u.Quantity,
        *,
        include: bool = True,
        external_identity: str | None = None,
        external_provenance: Mapping[str, Any] | None = None,
    ) -> CelestialSelection:
        """Construct one circular annular sector without a DS9 round trip.

        Angles use the package's native local tangent-plane convention: zero is
        the +longitude direction and positive angles rotate toward +latitude.
        A stop angle numerically below the start angle wraps through 360
        degrees; equal start/stop directions denote a full circle.

        DS9 text uses a reflected directed-angle convention. That conversion is
        intentionally confined to the DS9 adapter before this constructor is
        called.
        """
        inner = _nonnegative_angle(inner_radius, name="sector inner radius")
        outer = _positive_angle(outer_radius, name="sector outer radius")
        if outer <= inner:
            raise SelectionGeometryError("sector outer radius must exceed inner radius")
        sweep = _sector_sweep(start_angle, stop_angle)
        center_icrs = _normalised_scalar_icrs(center)

        if sweep.to_value(u.deg) == 360.0:
            boundaries = [CelestialBoundary(CirclePath(center_icrs, outer))]
            if inner > 0 * u.deg:
                boundaries.append(CelestialBoundary(CirclePath(center_icrs, inner), subtract=True))
        else:
            start_icrs = local_angle_to_icrs(center, start_angle)
            boundaries = [
                CelestialBoundary(
                    SectorAnnulusPath(
                        center=center_icrs,
                        inner_radius=inner,
                        outer_radius=outer,
                        start_angle=start_icrs,
                        sweep=sweep,
                    )
                )
            ]
        return cls(
            regions=(CelestialRegion(include=include, boundaries=tuple(boundaries)),),
            external_identity=external_identity,
            external_provenance=external_provenance,
        )


@dataclass(frozen=True)
class DetectorBoundary:
    """One closed detector-space boundary in SAS DETX/DETY units."""

    vertices: np.ndarray = _DetectorSnapshot()
    subtract: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtract", _science_bool(self.subtract, name="subtract"))
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise SelectionGeometryError("a detector boundary requires at least three XY vertices")
        if not np.isfinite(vertices).all():
            raise SelectionGeometryError("detector boundary vertices must be finite")
        # Immutable bytes own the buffer; WRITEABLE cannot be re-enabled on any view.
        immutable = np.frombuffer(vertices.tobytes(), dtype=float).reshape(vertices.shape)
        object.__setattr__(self, "vertices", immutable)

    def canonical_record(self) -> dict[str, object]:
        return {
            "subtract": bool(self.subtract),
            "vertices_det": [[float(x), float(y)] for x, y in self.vertices],
        }


@dataclass(frozen=True)
class DetectorRegion:
    """One included/excluded detector component with optional holes."""

    include: bool
    boundaries: tuple[DetectorBoundary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "include", _science_bool(self.include, name="include"))
        boundaries = tuple(self.boundaries)
        if not boundaries:
            raise SelectionGeometryError("a detector region requires at least one boundary")
        if boundaries[0].subtract:
            raise SelectionGeometryError("the outer detector boundary cannot be subtractive")
        if any(not boundary.subtract for boundary in boundaries[1:]):
            raise SelectionGeometryError("additional detector boundaries must be subtractive holes")
        object.__setattr__(self, "boundaries", boundaries)

    def canonical_record(self) -> dict[str, object]:
        return {
            "include": bool(self.include),
            "boundaries": [boundary.canonical_record() for boundary in self.boundaries],
        }


@dataclass(frozen=True)
class DetectorSelection:
    """Neutral detector-space result produced by an XMM projection backend."""

    regions: tuple[DetectorRegion, ...]
    source_geometry_sha256: str | None = None

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions:
            raise SelectionGeometryError("a detector selection requires at least one region")
        object.__setattr__(self, "regions", regions)

    def canonical_geometry_record(self) -> dict[str, object]:
        return {
            "schema": "xmm-region-tool.detector-selection/v1",
            "coordinate_system": "xmm-detx-dety",
            "regions": [region.canonical_record() for region in self.regions],
        }

    @property
    def geometry_sha256(self) -> str:
        return _canonical_sha256(self.canonical_geometry_record())

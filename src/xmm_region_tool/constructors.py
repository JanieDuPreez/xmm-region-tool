from __future__ import annotations

from collections.abc import Mapping, Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .elliptical_sector import EllipticalSectorAnnulusPath
from .model import (
    CelestialBoundary,
    CelestialSelection,
    CirclePath,
    EllipsePath,
    GeodesicPolygonPath,
    SelectionGeometryError,
    _finite_angle,
    _axis_angle,
    _axis_angle_to_icrs,
    _sector_sweep,
    _validate_local_basis,
    _validate_offset_extent,
    local_angle_to_icrs,
)


def _selection(
    boundaries: Sequence[CelestialBoundary],
    *,
    include: bool,
    external_identity: str | None,
    external_provenance: Mapping[str, object] | None,
) -> CelestialSelection:
    from .model import CelestialRegion

    return CelestialSelection(
        regions=(CelestialRegion(include=include, boundaries=tuple(boundaries)),),
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def _ellipse_path(
    center: SkyCoord,
    semi_major: u.Quantity,
    semi_minor: u.Quantity,
    angle: u.Quantity,
) -> EllipsePath:
    return EllipsePath(
        center=center.icrs,
        width=2 * semi_major,
        height=2 * semi_minor,
        angle=_axis_angle_to_icrs(center, angle),
    )


def _validate_major_minor(major: u.Quantity, minor: u.Quantity) -> None:
    major = _finite_angle(major, name="semi_major")
    minor = _finite_angle(minor, name="semi_minor")
    if major < minor:
        raise SelectionGeometryError("semi_major must be greater than or equal to semi_minor")


def _validated_rectangle_dimensions(
    width: u.Quantity,
    height: u.Quantity,
    *,
    context: str,
    allow_zero_pair: bool = False,
) -> tuple[u.Quantity, u.Quantity, bool]:
    width_deg = _finite_angle(width, name=f"{context} width")
    height_deg = _finite_angle(height, name=f"{context} height")

    width_zero = width_deg.value == 0.0
    height_zero = height_deg.value == 0.0

    if allow_zero_pair:
        if width_zero != height_zero:
            raise SelectionGeometryError(
                f"{context} width and height must both be zero or both positive"
            )
        if width_zero:
            return width_deg, height_deg, True

    if width_deg <= 0 * u.deg or height_deg <= 0 * u.deg:
        raise SelectionGeometryError(
            f"{context} width and height must be strictly positive"
        )

    return width_deg, height_deg, False


def _rectangle_vertices(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    angle: u.Quantity,
) -> SkyCoord:
    # Validate before broadcasting offsets, retaining the source-frame tangent basis.
    _validate_local_basis(center)
    angle = _axis_angle(angle)
    width, height, _ = _validated_rectangle_dimensions(
        width,
        height,
        context="box",
    )
    half_width = width / 2
    half_height = height / 2
    _validate_offset_extent(np.hypot(half_width, half_height))
    rotation = angle.to_value(u.rad)
    x = np.asarray([-1.0, 1.0, 1.0, -1.0]) * half_width
    y = np.asarray([-1.0, -1.0, 1.0, 1.0]) * half_height
    xr = x * np.cos(rotation) - y * np.sin(rotation)
    yr = x * np.sin(rotation) + y * np.cos(rotation)
    separation = np.hypot(xr, yr)
    position_angle = np.arctan2(xr.to_value(u.deg), yr.to_value(u.deg)) * u.rad
    return center.directional_offset_by(position_angle, separation).icrs


def _validate_elliptical_annulus_axes(
    inner_semi_major: u.Quantity,
    inner_semi_minor: u.Quantity,
    outer_semi_major: u.Quantity,
    outer_semi_minor: u.Quantity,
) -> bool:
    values = [
        _finite_angle(value, name="elliptical annulus axis")
        for value in (
            inner_semi_major,
            inner_semi_minor,
            outer_semi_major,
            outer_semi_minor,
        )
    ]
    if not all(np.isfinite(value.value) for value in values):
        raise SelectionGeometryError("elliptical annulus axes must be finite")
    inner_major, inner_minor, outer_major, outer_minor = values
    inner_major_zero = inner_major.value == 0.0
    inner_minor_zero = inner_minor.value == 0.0
    if inner_major_zero != inner_minor_zero:
        raise SelectionGeometryError(
            "elliptical annulus inner semimajor and semiminor axes must both be zero or both positive"
        )
    if inner_major < 0 * u.deg or inner_minor < 0 * u.deg:
        raise SelectionGeometryError("elliptical annulus inner axes must be non-negative")
    if outer_major <= inner_major or outer_minor <= inner_minor:
        raise SelectionGeometryError("elliptical annulus outer axes must exceed inner axes")
    if outer_major <= 0 * u.deg or outer_minor <= 0 * u.deg:
        raise SelectionGeometryError("elliptical annulus outer axes must be positive")
    if not inner_major_zero:
        inner_ratio = inner_minor.to_value(u.deg) / inner_major.to_value(u.deg)
        outer_ratio = outer_minor.to_value(u.deg) / outer_major.to_value(u.deg)
        if not np.isclose(inner_ratio, outer_ratio, rtol=1e-10, atol=1e-12):
            raise SelectionGeometryError(
                "elliptical annulus inner and outer ellipses must have the same axis ratio"
            )
    return bool(inner_major_zero)


def circle_selection(
    center: SkyCoord,
    radius: u.Quantity,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    return _selection(
        (CelestialBoundary(CirclePath(center.icrs, radius)),),
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def ellipse_selection(
    center: SkyCoord,
    semi_major: u.Quantity,
    semi_minor: u.Quantity,
    angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    _validate_major_minor(semi_major, semi_minor)
    return _selection(
        (CelestialBoundary(_ellipse_path(center, semi_major, semi_minor, angle)),),
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def box_selection(
    center: SkyCoord,
    width: u.Quantity,
    height: u.Quantity,
    angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    return _selection(
        (CelestialBoundary(GeodesicPolygonPath(_rectangle_vertices(center, width, height, angle))),),
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def polygon_selection(
    vertices: SkyCoord,
    *,
    holes: Sequence[SkyCoord] = (),
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    boundaries = [CelestialBoundary(GeodesicPolygonPath(vertices))]
    boundaries.extend(
        CelestialBoundary(GeodesicPolygonPath(hole), subtract=True) for hole in holes
    )
    return _selection(
        boundaries,
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def circular_annulus_selection(
    center: SkyCoord,
    inner_radius: u.Quantity,
    outer_radius: u.Quantity,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    return CelestialSelection.sector_annulus(
        center,
        inner_radius,
        outer_radius,
        0 * u.deg,
        360 * u.deg,
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def elliptical_annulus_selection(
    center: SkyCoord,
    inner_semi_major: u.Quantity,
    inner_semi_minor: u.Quantity,
    outer_semi_major: u.Quantity,
    outer_semi_minor: u.Quantity,
    angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    _validate_major_minor(inner_semi_major, inner_semi_minor)
    _validate_major_minor(outer_semi_major, outer_semi_minor)
    inner_zero = _validate_elliptical_annulus_axes(
        inner_semi_major,
        inner_semi_minor,
        outer_semi_major,
        outer_semi_minor,
    )
    boundaries = [
        CelestialBoundary(_ellipse_path(center, outer_semi_major, outer_semi_minor, angle))
    ]
    if not inner_zero:
        boundaries.append(
            CelestialBoundary(
                _ellipse_path(center, inner_semi_major, inner_semi_minor, angle),
                subtract=True,
            )
        )
    return _selection(
        boundaries,
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def box_annulus_selection(
    center: SkyCoord,
    inner_width: u.Quantity,
    inner_height: u.Quantity,
    outer_width: u.Quantity,
    outer_height: u.Quantity,
    angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    inner_width_deg, inner_height_deg, inner_zero = _validated_rectangle_dimensions(
        inner_width,
        inner_height,
        context="box annulus inner",
        allow_zero_pair=True,
    )
    outer_width_deg, outer_height_deg, _ = _validated_rectangle_dimensions(
        outer_width,
        outer_height,
        context="box annulus outer",
    )

    if not inner_zero and (
        outer_width_deg <= inner_width_deg
        or outer_height_deg <= inner_height_deg
    ):
        raise SelectionGeometryError(
            "box annulus outer width and height must exceed inner dimensions"
        )

    outer = GeodesicPolygonPath(
        _rectangle_vertices(center, outer_width_deg, outer_height_deg, angle)
    )
    boundaries = [CelestialBoundary(outer)]

    if not inner_zero:
        inner = GeodesicPolygonPath(
            _rectangle_vertices(center, inner_width_deg, inner_height_deg, angle)
        )
        boundaries.append(CelestialBoundary(inner, subtract=True))

    return _selection(
        boundaries,
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def sector_annulus_selection(
    center: SkyCoord,
    inner_radius: u.Quantity,
    outer_radius: u.Quantity,
    start_angle: u.Quantity,
    stop_angle: u.Quantity,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    """Construct one circular annular sector using the package's native angle convention."""
    return CelestialSelection.sector_annulus(
        center,
        inner_radius,
        outer_radius,
        start_angle,
        stop_angle,
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def elliptical_sector_annulus_selection(
    center: SkyCoord,
    inner_semi_major: u.Quantity,
    inner_semi_minor: u.Quantity,
    outer_semi_major: u.Quantity,
    outer_semi_minor: u.Quantity,
    start_angle: u.Quantity,
    stop_angle: u.Quantity,
    ellipse_angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    """Construct an elliptical annular sector from absolute physical sky angles.

    ``start_angle`` and ``stop_angle`` use the same native local tangent-plane
    convention as :func:`sector_annulus_selection`: zero is the local
    +longitude direction and positive angles rotate toward +latitude.

    ``ellipse_angle`` is independently the physical orientation of the ellipse
    semimajor axis in that same convention. Sector boundary angles are not
    measured relative to the ellipse axis.

    A stop angle numerically below the start angle wraps through 360 degrees;
    equal start/stop directions denote a full elliptical annulus.

    DS9 ``epanda`` stores ellipse-local cuts instead. The DS9 adapter converts
    through :func:`ellipse_local_sector_annulus_selection` rather than changing
    this native programmatic convention.
    """
    _validate_major_minor(inner_semi_major, inner_semi_minor)
    _validate_major_minor(outer_semi_major, outer_semi_minor)
    inner_zero = _validate_elliptical_annulus_axes(
        inner_semi_major,
        inner_semi_minor,
        outer_semi_major,
        outer_semi_minor,
    )

    start_angle = _finite_angle(start_angle, name="elliptical sector start angle")
    stop_angle = _finite_angle(stop_angle, name="elliptical sector stop angle")
    ellipse_angle = _finite_angle(ellipse_angle, name="ellipse angle")
    sweep = _sector_sweep(start_angle, stop_angle).to_value(u.deg)

    if sweep == 360.0:
        boundaries = [
            CelestialBoundary(
                _ellipse_path(
                    center,
                    outer_semi_major,
                    outer_semi_minor,
                    ellipse_angle,
                )
            )
        ]
        if not inner_zero:
            boundaries.append(
                CelestialBoundary(
                    _ellipse_path(
                        center,
                        inner_semi_major,
                        inner_semi_minor,
                        ellipse_angle,
                    ),
                    subtract=True,
                )
            )
        return _selection(
            boundaries,
            include=include,
            external_identity=external_identity,
            external_provenance=external_provenance,
        )

    path = EllipticalSectorAnnulusPath(
        center=center.icrs,
        inner_radius=inner_semi_major,
        outer_radius=outer_semi_major,
        start_angle=local_angle_to_icrs(center, start_angle),
        sweep=sweep * u.deg,
        inner_semi_minor=inner_semi_minor,
        outer_semi_minor=outer_semi_minor,
        ellipse_angle=_axis_angle_to_icrs(center, ellipse_angle),
    )
    return _selection(
        (CelestialBoundary(path),),
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def _ordered_axis_local_sector_selection(
    center: SkyCoord,
    inner_semi_major: u.Quantity,
    inner_semi_minor: u.Quantity,
    outer_semi_major: u.Quantity,
    outer_semi_minor: u.Quantity,
    local_start_angle: u.Quantity,
    local_stop_angle: u.Quantity,
    ellipse_angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    """Construct an elliptical annular sector from ellipse-local cut angles.

    The local cuts are measured in the rotated ellipse marker frame. A local
    cut ``a`` has physical direction ``ellipse_angle - a`` in the package's
    native local tangent-plane convention.

    This explicit local-cut API exists for formats such as DS9 ``epanda``.
    Callers with physical sky boundary angles should instead use
    :func:`elliptical_sector_annulus_selection`.
    """
    inner_zero = _validate_elliptical_annulus_axes(
        inner_semi_major,
        inner_semi_minor,
        outer_semi_major,
        outer_semi_minor,
    )
    local_start_angle = _finite_angle(local_start_angle, name="ellipse-local start angle")
    local_stop_angle = _finite_angle(local_stop_angle, name="ellipse-local stop angle")
    ellipse_angle_icrs = local_angle_to_icrs(center, ellipse_angle)

    sweep = _sector_sweep(local_start_angle, local_stop_angle).to_value(u.deg)

    if sweep == 360.0:
        boundaries = [
            CelestialBoundary(
                _ellipse_path(
                    center,
                    outer_semi_major,
                    outer_semi_minor,
                    ellipse_angle,
                )
            )
        ]
        if not inner_zero:
            boundaries.append(
                CelestialBoundary(
                    _ellipse_path(
                        center,
                        inner_semi_major,
                        inner_semi_minor,
                        ellipse_angle,
                    ),
                    subtract=True,
                )
            )
        return _selection(
            boundaries,
            include=include,
            external_identity=external_identity,
            external_provenance=external_provenance,
        )

    # Ellipse-local cuts increase opposite to the positive physical path
    # traversal. The path therefore starts on the local stop cut and sweeps
    # positively back to the local start cut.
    effective_start = ellipse_angle - (
        local_start_angle + sweep * u.deg
    )

    path = EllipticalSectorAnnulusPath(
        center=center.icrs,
        inner_radius=inner_semi_major,
        outer_radius=outer_semi_major,
        start_angle=local_angle_to_icrs(center, effective_start),
        sweep=sweep * u.deg,
        inner_semi_minor=inner_semi_minor,
        outer_semi_minor=outer_semi_minor,
        ellipse_angle=ellipse_angle_icrs,
    )
    return _selection(
        (CelestialBoundary(path),),
        include=include,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def ellipse_local_sector_annulus_selection(
    center: SkyCoord,
    inner_semi_major: u.Quantity,
    inner_semi_minor: u.Quantity,
    outer_semi_major: u.Quantity,
    outer_semi_minor: u.Quantity,
    local_start_angle: u.Quantity,
    local_stop_angle: u.Quantity,
    ellipse_angle: u.Quantity = 0 * u.deg,
    *,
    include: bool = True,
    external_identity: str | None = None,
    external_provenance: Mapping[str, object] | None = None,
) -> CelestialSelection:
    """Construct ellipse-local cuts; named major axes must be >= minor axes.

    Local cut ``a`` has physical direction ``ellipse_angle - a``. Equal axes
    are accepted without changing the directed marker/cut convention.
    """
    _validate_major_minor(inner_semi_major, inner_semi_minor)
    _validate_major_minor(outer_semi_major, outer_semi_minor)
    return _ordered_axis_local_sector_selection(
        center, inner_semi_major, inner_semi_minor, outer_semi_major, outer_semi_minor,
        local_start_angle, local_stop_angle, ellipse_angle, include=include,
        external_identity=external_identity, external_provenance=external_provenance,
    )

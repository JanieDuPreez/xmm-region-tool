"""Semantic elliptical annular-sector geometry used by DS9 ``epanda``."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .model import (
    SelectionGeometryError, SectorAnnulusPath,
    _finite_angle, _nonnegative_angle, _positive_angle, _validate_offset_extent,
    _AngleSnapshot,
)


@dataclass(frozen=True)
class EllipticalSectorAnnulusPath(SectorAnnulusPath):
    """One non-full sector of a concentric elliptical annulus.

    The inherited ``inner_radius``/``outer_radius`` fields are the inner and
    outer semimajor axes. ``inner_semi_minor``/``outer_semi_minor`` are the
    corresponding semiminor axes. The inner and outer ellipses must have the
    same axis ratio, matching DS9 ``epanda`` semantics.

    This path stores the already-resolved physical ICRS ray direction in
    ``start_angle``.  DS9's source ``epanda`` angles are local to the rotated
    ellipse marker; the public constructor resolves that coupling before this
    lower-level path is created. ``ellipse_angle`` is the physical ICRS ellipse
    orientation.  Elliptical arc points are ray--ellipse intersections in that
    resolved physical frame.
    """

    inner_semi_minor: u.Quantity = _AngleSnapshot(default=0)
    outer_semi_minor: u.Quantity = _AngleSnapshot(default=1)
    ellipse_angle: u.Quantity = _AngleSnapshot(default=0)

    def __post_init__(self) -> None:
        super().__post_init__()
        inner_minor = _nonnegative_angle(
            self.inner_semi_minor,
            name="elliptical-sector inner semiminor axis",
        )
        outer_minor = _positive_angle(
            self.outer_semi_minor,
            name="elliptical-sector outer semiminor axis",
        )
        _validate_offset_extent(outer_minor)
        _validate_offset_extent(inner_minor)
        ellipse_angle = _finite_angle(
            self.ellipse_angle,
            name="elliptical-sector ellipse angle",
        )
        ellipse_angle = float(np.mod(ellipse_angle.to_value(u.deg), 360.0)) * u.deg

        inner_major = self.inner_radius
        outer_major = self.outer_radius
        inner_is_zero = inner_major.to_value(u.deg) == 0.0
        inner_minor_is_zero = inner_minor.to_value(u.deg) == 0.0
        if inner_is_zero != inner_minor_is_zero:
            raise SelectionGeometryError(
                "elliptical-sector inner semimajor and semiminor axes must both be zero or both positive"
            )
        if not inner_is_zero:
            inner_ratio = inner_minor.to_value(u.deg) / inner_major.to_value(u.deg)
            outer_ratio = outer_minor.to_value(u.deg) / outer_major.to_value(u.deg)
            if not np.isclose(inner_ratio, outer_ratio, rtol=1e-10, atol=1e-12):
                raise SelectionGeometryError(
                    "elliptical-sector inner and outer ellipses must have the same axis ratio"
                )

        object.__setattr__(self, "inner_semi_minor", inner_minor)
        object.__setattr__(self, "outer_semi_minor", outer_minor)
        object.__setattr__(self, "ellipse_angle", ellipse_angle)

    @property
    def inner_semi_major(self) -> u.Quantity:
        return self.inner_radius

    @property
    def outer_semi_major(self) -> u.Quantity:
        return self.outer_radius

    def canonical_record(self) -> dict[str, object]:
        return {
            "kind": "elliptical-sector-annulus",
            "center_icrs_deg": [float(self.center.ra.deg), float(self.center.dec.deg)],
            "inner_semi_major_deg": float(self.inner_semi_major.to_value(u.deg)),
            "inner_semi_minor_deg": float(self.inner_semi_minor.to_value(u.deg)),
            "outer_semi_major_deg": float(self.outer_semi_major.to_value(u.deg)),
            "outer_semi_minor_deg": float(self.outer_semi_minor.to_value(u.deg)),
            "start_angle_deg": float(self.start_angle.to_value(u.deg)),
            "sweep_deg": float(self.sweep.to_value(u.deg)),
            "ellipse_angle_deg": float(self.ellipse_angle.to_value(u.deg)),
            "angle_convention": "resolved-physical-icrs-ray",
            "radial_edge_convention": "great-circle-from-center",
            "ellipse_convention": "ray-intersection-with-rotated-tangent-plane-ellipse",
        }

    def _radial_extent(
        self,
        semi_major: u.Quantity,
        semi_minor: u.Quantity,
        polar_angle: u.Quantity,
    ) -> u.Quantity:
        major = float(semi_major.to_value(u.deg))
        minor = float(semi_minor.to_value(u.deg))
        if major == 0.0:
            return 0 * u.deg
        delta = (polar_angle - self.ellipse_angle).to_value(u.rad)
        denominator = np.sqrt((np.cos(delta) / major) ** 2 + (np.sin(delta) / minor) ** 2)
        return (1.0 / denominator) * u.deg

    def _at_radius(self, radius: u.Quantity, polar_angle: u.Quantity) -> SkyCoord:
        position_angle = 90.0 * u.deg - polar_angle
        return self.center.directional_offset_by(position_angle, radius).icrs

    def _at_ellipse(
        self,
        semi_major: u.Quantity,
        semi_minor: u.Quantity,
        polar_angle: u.Quantity,
    ) -> SkyCoord:
        return self._at_radius(
            self._radial_extent(semi_major, semi_minor, polar_angle),
            polar_angle,
        )

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
            angle = start + self.sweep * fraction
            return self._at_ellipse(
                self.outer_semi_major,
                self.outer_semi_minor,
                angle,
            )

        if segment == 1:
            outer_radius = self._radial_extent(
                self.outer_semi_major,
                self.outer_semi_minor,
                stop,
            )
            inner_radius = self._radial_extent(
                self.inner_semi_major,
                self.inner_semi_minor,
                stop,
            )
            radius = outer_radius - (outer_radius - inner_radius) * fraction
            return self._at_radius(radius, stop)

        if self.segment_count == 3:
            outer_radius = self._radial_extent(
                self.outer_semi_major,
                self.outer_semi_minor,
                start,
            )
            return self._at_radius(outer_radius * fraction, start)

        if segment == 2:
            angle = stop - self.sweep * fraction
            return self._at_ellipse(
                self.inner_semi_major,
                self.inner_semi_minor,
                angle,
            )

        inner_radius = self._radial_extent(
            self.inner_semi_major,
            self.inner_semi_minor,
            start,
        )
        outer_radius = self._radial_extent(
            self.outer_semi_major,
            self.outer_semi_minor,
            start,
        )
        radius = inner_radius + (outer_radius - inner_radius) * fraction
        return self._at_radius(radius, start)

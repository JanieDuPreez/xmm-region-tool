"""Independent event-membership checks for sky and detector region geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from regions import PixCoord, PolygonPixelRegion, SkyRegion

from .sas import DetectorRegion


class MembershipValidationError(ValueError):
    """Raised when an event list cannot support an independent geometry check."""


@dataclass(frozen=True)
class MembershipValidationReport:
    """Agreement statistics between celestial and detector-space selections."""

    total_events: int
    sky_selected: int
    detector_selected: int
    disagreements: int
    sky_only: int
    detector_only: int
    first_mismatch_rows: tuple[int, ...]

    @property
    def agreement_fraction(self) -> float:
        if self.total_events == 0:
            return 1.0
        return 1.0 - self.disagreements / self.total_events

    def format_text(self) -> str:
        lines = [
            "XMM region event-membership validation",
            f"events: {self.total_events}",
            f"sky selected: {self.sky_selected}",
            f"detector selected: {self.detector_selected}",
            f"disagreements: {self.disagreements}",
            f"sky-only: {self.sky_only}",
            f"detector-only: {self.detector_only}",
            f"agreement: {self.agreement_fraction:.12f}",
        ]
        if self.first_mismatch_rows:
            rows = ", ".join(str(value) for value in self.first_mismatch_rows)
            lines.append(f"first mismatch rows (0-based): {rows}")
        return "\n".join(lines)


def _event_hdu(path: Path):
    hdus = fits.open(path, memmap=True)
    for hdu in hdus:
        names = getattr(getattr(hdu, "columns", None), "names", None)
        if not names:
            continue
        upper = {str(name).upper() for name in names}
        if {"X", "Y", "DETX", "DETY"}.issubset(upper):
            return hdus, hdu
    hdus.close()
    raise MembershipValidationError(
        f"event file has no table containing X, Y, DETX and DETY columns: {path}"
    )


def event_xy_wcs(path: str | Path) -> WCS:
    """Return the two-dimensional celestial WCS carried by event X/Y columns."""
    event_path = Path(path).expanduser().resolve()
    hdus, hdu = _event_hdu(event_path)
    try:
        names = [str(name).upper() for name in hdu.columns.names]
        x_index = names.index("X") + 1
        y_index = names.index("Y") + 1
        try:
            wcs = WCS(
                hdu.header,
                keysel=["pixel"],
                colsel=[x_index, y_index],
                relax=True,
            )
        except Exception as exc:
            raise MembershipValidationError(
                f"cannot construct X/Y binary-table WCS for {event_path}: {exc}"
            ) from exc
        if not wcs.has_celestial:
            raise MembershipValidationError(
                f"event X/Y binary-table WCS is not celestial: {event_path}"
            )
        return wcs.celestial
    finally:
        hdus.close()


def _combine_source_regions(
    regions: list[tuple[bool, object]],
    coordinates: PixCoord,
) -> np.ndarray:
    included = [region for include, region in regions if include]
    excluded = [region for include, region in regions if not include]
    size = np.asarray(coordinates.x).size
    result = np.zeros(size, dtype=bool) if included else np.ones(size, dtype=bool)
    for region in included:
        result |= np.asarray(region.contains(coordinates), dtype=bool)
    for region in excluded:
        result &= ~np.asarray(region.contains(coordinates), dtype=bool)
    return result


def _source_pixel_regions(source_regions: list[SkyRegion], wcs: WCS) -> list[tuple[bool, object]]:
    result: list[tuple[bool, object]] = []
    for region in source_regions:
        include = bool(region.meta.get("include", True))
        try:
            pixel_region = region.to_pixel(wcs)
        except Exception as exc:
            raise MembershipValidationError(
                f"cannot project {type(region).__name__} through the event X/Y WCS: {exc}"
            ) from exc
        result.append((include, pixel_region))
    return result


def _detector_pixel_regions(
    detector_regions: list[DetectorRegion],
) -> list[tuple[bool, tuple[PolygonPixelRegion, ...]]]:
    result: list[tuple[bool, tuple[PolygonPixelRegion, ...]]] = []
    for region in detector_regions:
        if not region.boundaries:
            raise MembershipValidationError("detector region contains no boundaries")
        polygons: list[PolygonPixelRegion] = []
        for boundary in region.boundaries:
            vertices = np.asarray(boundary.vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
                raise MembershipValidationError(
                    "detector polygon must contain at least three XY vertices"
                )
            if not np.isfinite(vertices).all():
                raise MembershipValidationError("detector polygon contains non-finite vertices")
            polygons.append(
                PolygonPixelRegion(PixCoord(x=vertices[:, 0], y=vertices[:, 1]))
            )
        result.append((region.include, tuple(polygons)))
    return result


def _combine_detector_regions(
    regions: list[tuple[bool, tuple[PolygonPixelRegion, ...]]],
    coordinates: PixCoord,
) -> np.ndarray:
    size = np.asarray(coordinates.x).size
    included = [entry for entry in regions if entry[0]]
    excluded = [entry for entry in regions if not entry[0]]
    result = np.zeros(size, dtype=bool) if included else np.ones(size, dtype=bool)

    def one_region(polygons: tuple[PolygonPixelRegion, ...]) -> np.ndarray:
        membership = np.asarray(polygons[0].contains(coordinates), dtype=bool)
        for polygon in polygons[1:]:
            membership &= ~np.asarray(polygon.contains(coordinates), dtype=bool)
        return membership

    for _, polygons in included:
        result |= one_region(polygons)
    for _, polygons in excluded:
        result &= ~one_region(polygons)
    return result


def compare_event_membership(
    event_file: str | Path,
    source_regions: list[SkyRegion],
    detector_regions: list[DetectorRegion],
    *,
    chunk_size: int = 100_000,
    mismatch_examples: int = 20,
) -> MembershipValidationReport:
    """Compare the original sky selection with converted detector polygons.

    The sky side is independently projected through the event-list X/Y WCS with
    Astropy/``regions``.  It does not call ``esky2det``.  The detector side uses
    the converted DETX/DETY polygons.  Classification is compared row-by-row on
    the same event table.

    FITS binary-table WCS uses origin 1 for the stored event X/Y values, whereas
    Astropy ``regions`` pixel coordinates are zero-based.  Event X/Y values are
    therefore shifted by exactly one before evaluating the projected sky region.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if mismatch_examples < 0:
        raise ValueError("mismatch_examples must be non-negative")
    if not source_regions:
        raise MembershipValidationError("no source sky regions were provided")
    if not detector_regions:
        raise MembershipValidationError("no detector regions were provided")

    event_path = Path(event_file).expanduser().resolve()
    wcs = event_xy_wcs(event_path)
    sky_pixel_regions = _source_pixel_regions(source_regions, wcs)
    detector_pixel_regions = _detector_pixel_regions(detector_regions)

    hdus, hdu = _event_hdu(event_path)
    try:
        data = hdu.data
        if data is None:
            raise MembershipValidationError(f"event table contains no rows: {event_path}")
        names = {str(name).upper(): str(name) for name in hdu.columns.names}
        total = len(data)
        sky_selected = 0
        detector_selected = 0
        disagreements = 0
        sky_only = 0
        detector_only = 0
        examples: list[int] = []

        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            x = np.asarray(data[names["X"]][start:stop], dtype=float)
            y = np.asarray(data[names["Y"]][start:stop], dtype=float)
            detx = np.asarray(data[names["DETX"]][start:stop], dtype=float)
            dety = np.asarray(data[names["DETY"]][start:stop], dtype=float)

            if not (
                np.isfinite(x).all()
                and np.isfinite(y).all()
                and np.isfinite(detx).all()
                and np.isfinite(dety).all()
            ):
                raise MembershipValidationError(
                    "event X/Y/DETX/DETY columns contain non-finite coordinates"
                )

            sky = _combine_source_regions(
                sky_pixel_regions,
                PixCoord(x=x - 1.0, y=y - 1.0),
            )
            detector = _combine_detector_regions(
                detector_pixel_regions,
                PixCoord(x=detx, y=dety),
            )
            mismatch = sky != detector
            sky_only_mask = sky & ~detector
            detector_only_mask = detector & ~sky

            sky_selected += int(np.count_nonzero(sky))
            detector_selected += int(np.count_nonzero(detector))
            disagreements += int(np.count_nonzero(mismatch))
            sky_only += int(np.count_nonzero(sky_only_mask))
            detector_only += int(np.count_nonzero(detector_only_mask))

            if len(examples) < mismatch_examples and np.any(mismatch):
                local = np.flatnonzero(mismatch)
                remaining = mismatch_examples - len(examples)
                examples.extend((start + local[:remaining]).tolist())

        return MembershipValidationReport(
            total_events=total,
            sky_selected=sky_selected,
            detector_selected=detector_selected,
            disagreements=disagreements,
            sky_only=sky_only,
            detector_only=detector_only,
            first_mismatch_rows=tuple(examples),
        )
    finally:
        hdus.close()

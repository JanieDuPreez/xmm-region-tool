"""Serialize detector geometry for SAS/ESAS consumers.

The preferred representation is a short mosspectra/pnspectra text ``regionfile``
that delegates the actual geometry to a FITS REGION table. This keeps complex
regions out of both the shell command line and a very long selectlib expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits

from .calibration import CalibrationIdentity, SasProducerIdentity
from .limits import (
    MAX_COMPONENTS, MAX_INLINE_LIMIT, MAX_REGION_COORDINATES, MAX_REGION_ROWS,
    MAX_REGION_VERTICES, integer_limit,
)
from .provenance import EventIdentity, file_sha256
from .sas import DetectorBoundary, DetectorRegion
from .sas_paths import SasPathError, validate_sas_safe_resolved_path


class RegionSerializationError(ValueError):
    """Raised when detector geometry cannot be serialized safely."""


class _ExpressionLimitError(RegionSerializationError):
    pass


def _allocation_budget(rows: int, vertices: int) -> None:
    if rows > MAX_REGION_ROWS:
        raise RegionSerializationError(f"REGION row resource limit is {MAX_REGION_ROWS}")
    if vertices > MAX_REGION_VERTICES:
        raise RegionSerializationError(f"REGION vertex resource limit is {MAX_REGION_VERTICES}")
    if rows > MAX_REGION_COORDINATES // (2 * (vertices + 1)):
        raise RegionSerializationError(
            f"REGION aggregate coordinate resource limit is {MAX_REGION_COORDINATES}"
        )


def _preflight_regions(regions: list[DetectorRegion]) -> int:
    rows = 0
    width = 0
    for region in regions:
        rows += len(region.boundaries)
        for boundary in region.boundaries:
            width = max(width, len(boundary.vertices))
        _allocation_budget(rows, width)
    return width


@dataclass(frozen=True)
class RegionElement:
    """One polygon term in a FITS REGION boolean component."""

    vertices: np.ndarray
    negate: bool = False


@dataclass(frozen=True)
class WrittenRegionFiles:
    """Artifacts produced for one mosspectra/pnspectra region selection."""

    regionfile: Path
    representation: Literal["fits", "expression"]
    fits_region: Path | None = None


def _validated_vertices(boundary: DetectorBoundary) -> np.ndarray:
    vertices = np.asarray(boundary.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise RegionSerializationError("each detector polygon must have at least three XY vertices")
    if not np.isfinite(vertices).all():
        raise RegionSerializationError("detector polygons cannot contain non-finite coordinates")
    return vertices


def _region_terms(region: DetectorRegion) -> list[RegionElement]:
    """Return the conjunction that defines one DS9 area region."""
    if not region.boundaries:
        raise RegionSerializationError("a detector region contains no boundaries")

    terms: list[RegionElement] = []
    for index, boundary in enumerate(region.boundaries):
        vertices = _validated_vertices(boundary)
        if index == 0 and boundary.subtract:
            raise RegionSerializationError("the outer boundary of a region cannot be subtractive")
        if index > 0 and not boundary.subtract:
            raise RegionSerializationError(
                "additional boundaries must be subtractive holes; unsupported compound geometry"
            )
        terms.append(RegionElement(vertices, negate=bool(boundary.subtract)))
    return terms


def detector_region_components(
    regions: list[DetectorRegion],
    *,
    max_components: int = 4096,
) -> list[tuple[RegionElement, ...]]:
    """Convert DS9 include/exclude semantics to FITS REGION components."""
    max_components = integer_limit(max_components, "max_components", 1, MAX_COMPONENTS)
    if not regions:
        raise RegionSerializationError("no detector regions were provided")

    width = _preflight_regions(regions)
    included = [region for region in regions if region.include]
    excluded = [region for region in regions if not region.include]
    if len(included) > max_components:
        raise RegionSerializationError(f"initial FITS REGION components exceed limit {max_components}")

    components: list[list[RegionElement]]
    if included:
        components = [_region_terms(region) for region in included]
    else:
        components = [[]]

    for region in excluded:
        excluded_terms = _region_terms(region)
        alternatives = [
            RegionElement(term.vertices, negate=not term.negate) for term in excluded_terms
        ]
        candidate_count = len(components) * len(alternatives)
        if candidate_count > max_components:
            raise RegionSerializationError(
                "DS9 exclusion geometry expands to "
                f"{candidate_count} FITS REGION components (limit {max_components}); "
                "split the region or raise the explicit component limit"
            )
        candidate_rows = sum(len(component) + 1 for component in components) * len(alternatives)
        _allocation_budget(candidate_rows, width)
        components = [
            [*component, alternative]
            for component in components
            for alternative in alternatives
        ]

    if any(not component for component in components):
        raise RegionSerializationError("region selection contains no spatial constraint")
    _allocation_budget(sum(map(len, components)), width)
    return [tuple(component) for component in components]


def _polygon_expression(vertices: np.ndarray) -> str:
    values = ",".join(f"{value:.6f}" for value in vertices.reshape(-1))
    return f"((DETX,DETY) IN polygon2({values}))"


def _one_region_expression(region: DetectorRegion) -> str:
    terms = _region_terms(region)
    pieces = []
    for term in terms:
        expression = _polygon_expression(term.vertices)
        pieces.append(f"!{expression}" if term.negate else expression)
    return "(" + " && ".join(pieces) + ")"


def selection_expression(regions: list[DetectorRegion], *, max_chars: int = MAX_INLINE_LIMIT) -> str:
    """Build a direct selectlib expression, including exclusion-only DS9 files."""
    if not regions:
        raise RegionSerializationError("no detector regions were provided")
    max_chars = integer_limit(max_chars, "max_chars", 1, MAX_INLINE_LIMIT)
    _preflight_regions(regions)
    pieces = []
    size = 0

    def append(text: str) -> None:
        nonlocal size
        size += len(text)
        if size > max_chars:
            raise _ExpressionLimitError(f"inline expression exceeds resource limit {max_chars}")
        pieces.append(text)

    def region_expression(region: DetectorRegion) -> None:
        append("(")
        for index, term in enumerate(_region_terms(region)):
            if index:
                append(" && ")
            if term.negate:
                append("!")
            append("((DETX,DETY) IN polygon2(")
            for coordinate, value in enumerate(term.vertices.reshape(-1)):
                if coordinate:
                    append(",")
                append(f"{value:.6f}")
            append("))")
        append(")")

    included = [region for region in regions if region.include]
    excluded = [region for region in regions if not region.include]
    if included:
        append("(")
        for index, region in enumerate(included):
            if index:
                append(" || ")
            region_expression(region)
        append(")")
    for index, region in enumerate(excluded):
        append(" && !" if included or index else "!")
        region_expression(region)
    return "".join(pieces)


def _densify_polygon(vertices: np.ndarray, target_count: int) -> np.ndarray:
    """Insert collinear detector-space points without changing polygon geometry.

    FITS binary-table vector columns have one fixed repeat count for every row.
    Padding a shorter polygon by repeating its first vertex creates zero-length
    segments, which real SAS triangulation can reject. Instead, subdivide the
    polygon's existing detector-space chords until every serialized polygon row
    has the same number of genuine vertices. The represented polygon is exactly
    unchanged because every inserted point lies on an existing straight edge.
    """
    points = np.asarray(vertices, dtype=float)
    count = len(points)
    if target_count < count:
        raise RegionSerializationError(
            f"cannot densify {count}-vertex polygon to smaller target {target_count}"
        )
    if target_count == count:
        return points.copy()

    base, remainder = divmod(target_count, count)
    result: list[np.ndarray] = []
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % count]
        subdivisions = base + (1 if index < remainder else 0)
        for step in range(subdivisions):
            fraction = step / subdivisions
            result.append(start + fraction * (end - start))
    densified = np.asarray(result, dtype=float)
    if len(densified) != target_count:
        raise RegionSerializationError("internal polygon densification length mismatch")
    return densified


def _validated_sha256(value: str, *, name: str) -> str:
    text = value.strip().lower()
    if len(text) != 64:
        raise RegionSerializationError(f"{name} must be a hexadecimal SHA256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise RegionSerializationError(f"{name} must be a hexadecimal SHA256 digest") from exc
    return text


def _write_provenance_headers(
    header: fits.Header,
    *,
    event_identity: EventIdentity | None,
    event_file_sha256: str | None,
    source_region_sha256: str | None,
    celestial_geometry_sha256: str | None,
    projection_identity_sha256: str | None,
    calibration_identity: CalibrationIdentity | None,
    sas_producer: SasProducerIdentity | None,
) -> None:
    if event_identity is not None:
        if event_identity.telescope is not None:
            header["TELESCOP"] = event_identity.telescope
        header["INSTRUME"] = event_identity.instrument_header
        header["OBS_ID"] = event_identity.obs_id
        if event_identity.exposure_id is not None:
            header["EXP_ID"] = event_identity.exposure_id
        if event_identity.date_obs is not None:
            header["DATE-OBS"] = event_identity.date_obs
        if event_identity.ra_pnt is not None:
            header["RA_PNT"] = event_identity.ra_pnt
        if event_identity.dec_pnt is not None:
            header["DEC_PNT"] = event_identity.dec_pnt
        if event_identity.pa_pnt is not None:
            header["PA_PNT"] = event_identity.pa_pnt
        header["XMMRGID"] = event_identity.identity_sha256
        exact_sha = (
            file_sha256(event_identity.path)
            if event_file_sha256 is None
            else _validated_sha256(event_file_sha256, name="event_file_sha256")
        )
        header["XMRGEVT"] = exact_sha
    elif event_file_sha256 is not None:
        raise RegionSerializationError("event_file_sha256 requires event_identity")
    if source_region_sha256 is not None:
        header["XMRGSHA"] = _validated_sha256(
            source_region_sha256,
            name="source_region_sha256",
        )
    if celestial_geometry_sha256 is not None:
        header["XMRGCEL"] = _validated_sha256(
            celestial_geometry_sha256,
            name="celestial_geometry_sha256",
        )
    if projection_identity_sha256 is not None:
        header["XMRGPRO"] = _validated_sha256(
            projection_identity_sha256,
            name="projection_identity_sha256",
        )
    if calibration_identity is not None:
        header["XMRGCCF"] = calibration_identity.identity_sha256
        header["XMRGCIF"] = calibration_identity.cif_file_sha256
        header["XMRGCAL"] = calibration_identity.calindex_sha256
        header["XMRGREP"] = len(calibration_identity.replacements)
    if sas_producer is not None:
        header["XMRGPRD"] = sas_producer.identity_sha256
        header["ESKYVER"] = sas_producer.esky2det_version
        if sas_producer.sas_version is not None:
            header["SASVERS"] = sas_producer.sas_version


def write_fits_region(
    path: str | Path,
    regions: list[DetectorRegion],
    *,
    max_components: int = 4096,
    event_identity: EventIdentity | None = None,
    event_file_sha256: str | None = None,
    source_region_sha256: str | None = None,
    celestial_geometry_sha256: str | None = None,
    projection_identity_sha256: str | None = None,
    calibration_identity: CalibrationIdentity | None = None,
    sas_producer: SasProducerIdentity | None = None,
) -> Path:
    """Write an ASC-FITS-REGION-1.0 table containing detector-space polygons."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    components = detector_region_components(regions, max_components=max_components)
    rows = [
        (component_id, element)
        for component_id, component in enumerate(components, start=1)
        for element in component
    ]
    target_vertices = max(len(element.vertices) for _, element in rows)
    vector_length = target_vertices + 1

    shapes: list[str] = []
    x = np.empty((len(rows), vector_length), dtype=np.float64)
    y = np.empty((len(rows), vector_length), dtype=np.float64)
    component_ids = np.empty(len(rows), dtype=np.int32)

    for row_index, (component_id, element) in enumerate(rows):
        vertices = _densify_polygon(element.vertices, target_vertices)
        x[row_index, :target_vertices] = vertices[:, 0]
        y[row_index, :target_vertices] = vertices[:, 1]
        x[row_index, target_vertices] = vertices[0, 0]
        y[row_index, target_vertices] = vertices[0, 1]
        shapes.append("!POLYGON" if element.negate else "POLYGON")
        component_ids[row_index] = component_id

    columns = [
        fits.Column(name="SHAPE", format="16A", array=np.asarray(shapes, dtype="S16")),
        fits.Column(name="X", format=f"{vector_length}D", array=x),
        fits.Column(name="Y", format=f"{vector_length}D", array=y),
        fits.Column(name="COMPONENT", format="J", array=component_ids),
    ]
    region_hdu = fits.BinTableHDU.from_columns(columns, name="REGION")
    header = region_hdu.header
    header["EXTVER"] = 1
    header["EXTLEVEL"] = 1
    header["ORIGIN"] = "xmm-region-tool"
    header["CONTENT"] = "REGION"
    header["HDUCLASS"] = "ASC"
    header["HDUCLAS1"] = "REGION"
    header["HDUCLAS2"] = "STANDARD"
    header["HDUVERS"] = "1.0.0"
    header["HDUDOC"] = "ASC-FITS-REGION-1.0"
    header["MTYPE1"] = "POS"
    header["MFORM1"] = "X,Y"
    header["CREATOR"] = "xmm-region-tool"
    _write_provenance_headers(
        header,
        event_identity=event_identity,
        event_file_sha256=event_file_sha256,
        source_region_sha256=source_region_sha256,
        celestial_geometry_sha256=celestial_geometry_sha256,
        projection_identity_sha256=projection_identity_sha256,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
    )

    fits.HDUList([fits.PrimaryHDU(), region_hdu]).writeto(output, overwrite=True, checksum=True)
    return output


def _fits_wrapper_expression(path: Path) -> str:
    try:
        safe_path = validate_sas_safe_resolved_path(path, role="FITS REGION geometry")
    except SasPathError as exc:
        raise RegionSerializationError(str(exc)) from exc
    return f"region({safe_path},DETX,DETY)"


def write_esas_regionfile(
    path: str | Path,
    regions: list[DetectorRegion],
    *,
    representation: Literal["fits", "expression", "auto"] = "fits",
    inline_limit: int = 4096,
    max_components: int = 4096,
    event_identity: EventIdentity | None = None,
    event_file_sha256: str | None = None,
    source_region_sha256: str | None = None,
    celestial_geometry_sha256: str | None = None,
    projection_identity_sha256: str | None = None,
    calibration_identity: CalibrationIdentity | None = None,
    sas_producer: SasProducerIdentity | None = None,
) -> WrittenRegionFiles:
    """Write the text regionfile consumed by mosspectra/pnspectra."""
    if representation not in {"fits", "expression", "auto"}:
        raise ValueError(f"unknown representation {representation!r}")
    inline_limit = integer_limit(inline_limit, "inline_limit", 1, MAX_INLINE_LIMIT)
    max_components = integer_limit(max_components, "max_components", 1, MAX_COMPONENTS)

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    complete_expression = ""

    chosen: Literal["fits", "expression"]
    if representation == "auto":
        try:
            if inline_limit <= 3:
                raise _ExpressionLimitError("wrapper exceeds inline limit")
            complete_expression = "&&" + selection_expression(regions, max_chars=inline_limit - 3) + "\n"
            chosen = "expression"
        except _ExpressionLimitError:
            chosen = "fits"
    else:
        chosen = representation

    if chosen == "expression":
        if not complete_expression:
            complete_expression = "&&" + selection_expression(regions, max_chars=MAX_INLINE_LIMIT - 3) + "\n"
        output.write_text(complete_expression)
        return WrittenRegionFiles(output, "expression")

    suffix = output.suffix
    if suffix.lower() == ".fits":
        fits_output = output.with_name(output.stem + "-geometry.fits")
    elif suffix:
        fits_output = output.with_suffix(".fits")
    else:
        fits_output = output.with_name(output.name + ".fits")
    try:
        fits_output = validate_sas_safe_resolved_path(
            fits_output,
            role="FITS REGION geometry",
        )
    except SasPathError as exc:
        raise RegionSerializationError(str(exc)) from exc
    write_fits_region(
        fits_output,
        regions,
        max_components=max_components,
        event_identity=event_identity,
        event_file_sha256=event_file_sha256,
        source_region_sha256=source_region_sha256,
        celestial_geometry_sha256=celestial_geometry_sha256,
        projection_identity_sha256=projection_identity_sha256,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
    )
    output.write_text("&&" + _fits_wrapper_expression(fits_output) + "\n")
    return WrittenRegionFiles(output, "fits", fits_output)

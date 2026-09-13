"""SAS-backed sky-to-detector coordinate conversion."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from .context_suitability import ContextSuitabilityError, capture_context_suitability
from .event_roles import read_science_calinfoset_snapshot
from .execution import (
    LEGACY_REFINEMENT,
    ProjectionProvenance,
    ProjectionResult,
    ProjectionRule,
    SasProjectionContext,
)
from .model import (
    CelestialBoundary,
    CelestialRegion,
    CelestialSelection,
    CirclePath,
    DetectorBoundary,
    DetectorRegion,
    DetectorSelection,
    EllipsePath,
    GeodesicPolygonPath,
)
from .provenance import file_sha256
from .sas_paths import (
    SasPathError,
    validate_sas_safe_resolved_path,
    validate_sas_temporary_root,
)
from .subprocess_capture import CapturedDiagnostic, diagnostic_summary, run_bounded
from .topology import (
    DetectorTopologyError,
    validate_detector_boundary,
    validate_detector_regions,
)


def _diagnostic_evidence(
    capture: CapturedDiagnostic | None,
    text: str | None,
) -> dict[str, object] | None:
    if capture is not None:
        return capture.evidence_record()
    if text is None:
        return None
    size = len(text.encode("utf-8"))
    return {"total_bytes": size, "retained_bytes": size, "truncated": False}


class SasConversionError(RuntimeError):
    """Raised when SAS cannot produce a usable detector-coordinate mapping."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        stdout_capture: CapturedDiagnostic | None = None,
        stderr_capture: CapturedDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.command = tuple(command) if command is not None else None
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_evidence = _diagnostic_evidence(stdout_capture, stdout)
        self.stderr_evidence = _diagnostic_evidence(stderr_capture, stderr)

    def evidence_record(self) -> dict[str, object | None]:
        """Return bounded structured execution evidence suitable for a failure manifest."""
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "command": list(self.command) if self.command is not None else None,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_evidence": self.stdout_evidence,
            "stderr_evidence": self.stderr_evidence,
        }


def _resolve_projection_executable(
    esky2det: str | Path,
    environment: Mapping[str, str] | None,
) -> Path:
    candidate = Path(esky2det).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise SasConversionError(f"SAS task does not exist: {resolved}")
        return resolved
    if environment is None:
        found = shutil.which(str(esky2det))
    else:
        found = shutil.which(str(esky2det), path=environment.get("PATH"))
    if found is None:
        raise SasConversionError(f"SAS task {str(esky2det)!r} was not found on PATH")
    return Path(found).resolve()


def _skycoord_array(points: Sequence[SkyCoord]) -> SkyCoord:
    if not points:
        raise ValueError("at least one celestial point is required")
    return SkyCoord(
        ra=np.asarray([float(point.icrs.ra.deg) for point in points]) * u.deg,
        dec=np.asarray([float(point.icrs.dec.deg) for point in points]) * u.deg,
        frame="icrs",
    )


def _project_skycoords(
    coords: SkyCoord,
    *,
    calinfoset: str | Path,
    esky2det: str | Path,
    environment: Mapping[str, str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    executable = _resolve_projection_executable(esky2det, environment)
    try:
        calibration_file = validate_sas_safe_resolved_path(
            calinfoset,
            role="esky2det calinfoset",
        )
        temp_parent = validate_sas_temporary_root(role="esky2det temporary root")
    except SasPathError as exc:
        raise SasConversionError(str(exc)) from exc
    if not calibration_file.is_file():
        raise SasConversionError(f"calibration/event file does not exist: {calibration_file}")

    icrs = coords.icrs
    ra = np.atleast_1d(np.asarray(icrs.ra.deg, dtype=float))
    dec = np.atleast_1d(np.asarray(icrs.dec.deg, dtype=float))
    if len(ra) != len(dec) or len(ra) == 0:
        raise SasConversionError("invalid celestial coordinate array for esky2det")
    if len(ra) > np.iinfo(np.int32).max:
        raise SasConversionError("too many boundary points for a 32-bit SAS row identity")
    row_id = np.arange(len(ra), dtype=np.int32)

    with tempfile.TemporaryDirectory(prefix="xmm-region-", dir=temp_parent) as directory:
        try:
            infile = validate_sas_safe_resolved_path(
                Path(directory) / "sky.fits",
                role="esky2det temporary input table",
            )
        except SasPathError as exc:
            raise SasConversionError(str(exc)) from exc
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="RA", format="D", array=ra),
                fits.Column(name="DEC", format="D", array=dec),
                fits.Column(name="ROW_ID", format="J", array=row_id),
            ],
            name="INPUT",
        )
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(infile)

        command = [
            str(executable),
            "datastyle=set",
            f"intab={infile}:INPUT",
            "witherrorcol=no",
            "withouttab=no",
            "outunit=det",
            "calinfostyle=set",
            f"calinfoset={calibration_file}",
            "checkfov=no",
        ]
        command_record = tuple(command)
        result = run_bounded(command, environment=environment)
        if result.returncode != 0:
            details = diagnostic_summary(result.stderr or result.stdout)
            raise SasConversionError(
                f"esky2det failed with exit code {result.returncode}: {details}",
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                stdout_capture=result.stdout_capture,
                stderr_capture=result.stderr_capture,
            )

        with fits.open(infile) as hdus:
            if "INPUT" not in hdus:
                raise SasConversionError(
                    "esky2det input/output file lost its INPUT table",
                    command=command_record,
                )
            data = hdus["INPUT"].data
            required = {"DETX", "DETY", "ROW_ID"}
            names = set(data.names) if data is not None else set()
            if data is None or not required.issubset(names):
                missing = ", ".join(sorted(required - names))
                raise SasConversionError(
                    "esky2det in-place output is missing required columns: " + missing,
                    command=command_record,
                )
            converted_ids = np.asarray(data["ROW_ID"], dtype=np.int64)
            detx = np.asarray(data["DETX"], dtype=float)
            dety = np.asarray(data["DETY"], dtype=float)

    if len(detx) != len(ra):
        raise SasConversionError(
            f"esky2det returned {len(detx)} rows for {len(ra)} input positions",
            command=command_record,
        )
    if not np.array_equal(converted_ids, row_id.astype(np.int64)):
        raise SasConversionError(
            "esky2det did not preserve the explicit input ROW_ID sequence; refusing to "
            "associate detector coordinates with the wrong boundary points",
            command=command_record,
        )
    finite = np.isfinite(detx) & np.isfinite(dety)
    if not finite.all():
        bad = np.flatnonzero(~finite)
        first = int(bad[0])
        raise SasConversionError(
            "esky2det returned null/non-finite DETX/DETY for "
            f"{len(bad)} of {len(ra)} boundary points (first input row {first}) despite "
            "checkfov=no. The tool will not clip, zero or extrapolate those values.",
            command=command_record,
        )
    return np.column_stack((detx, dety)), command_record


def _legacy_boundary_vertices(boundary: CelestialBoundary, samples: int) -> SkyCoord:
    if isinstance(boundary.path, GeodesicPolygonPath):
        return boundary.path.vertices
    if isinstance(boundary.path, (CirclePath, EllipsePath)):
        return boundary.path.sample(samples)
    raise ValueError(
        "legacy fixed-source sampling supports only polygon, circle and ellipse "
        f"boundaries; {type(boundary.path).__name__} requires adaptive projection"
    )


def _legacy_convert_regions(
    regions: Sequence[CelestialRegion],
    *,
    rule: ProjectionRule,
    calinfoset: str | Path,
    esky2det: str | Path,
    environment: Mapping[str, str] | None,
) -> tuple[tuple[DetectorRegion, ...], tuple[tuple[str, ...], ...], dict[str, object]]:
    assert rule.samples is not None
    sky_boundaries: list[SkyCoord] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for region in regions:
        for boundary in region.boundaries:
            vertices = _legacy_boundary_vertices(boundary, rule.samples)
            sky_boundaries.append(vertices)
            spans.append((cursor, cursor + len(vertices)))
            cursor += len(vertices)
    flattened = SkyCoord(
        ra=np.concatenate([boundary.icrs.ra.deg for boundary in sky_boundaries]) * u.deg,
        dec=np.concatenate([boundary.icrs.dec.deg for boundary in sky_boundaries]) * u.deg,
        frame="icrs",
    )
    projected, command = _project_skycoords(
        flattened,
        calinfoset=calinfoset,
        esky2det=esky2det,
        environment=environment,
    )

    detector_boundaries: list[DetectorBoundary] = []
    index = 0
    boundary_diagnostics: list[dict[str, object]] = []
    for region in regions:
        for boundary in region.boundaries:
            start, stop = spans[index]
            vertices = projected[start:stop]
            _validate_detector_boundary(vertices)
            detector_boundaries.append(
                DetectorBoundary(vertices=vertices, subtract=boundary.subtract)
            )
            boundary_diagnostics.append(
                {
                    "mode": LEGACY_REFINEMENT,
                    "vertex_count": len(vertices),
                    "samples": rule.samples,
                }
            )
            index += 1
    detector_regions = _rebuild_regions(regions, detector_boundaries)
    _validate_region_topology(detector_regions)
    return (
        detector_regions,
        (command,),
        {"mode": LEGACY_REFINEMENT, "boundaries": boundary_diagnostics},
    )


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-18:
        raise SasConversionError("adaptive projection produced a degenerate detector chord")
    fraction = float(np.dot(point - start, direction) / length_squared)
    fraction = min(1.0, max(0.0, fraction))
    closest = start + fraction * direction
    return float(np.linalg.norm(point - closest))


_INTERIOR_PROBE_FRACTIONS = (0.25, 0.5, 0.75)


def _adaptive_boundary(
    boundary: CelestialBoundary,
    *,
    rule: ProjectionRule,
    calinfoset: str | Path,
    esky2det: str | Path,
    environment: Mapping[str, str] | None,
) -> tuple[DetectorBoundary, list[tuple[str, ...]], dict[str, object]]:
    assert rule.detector_tolerance is not None
    pending = [(start, stop, 0) for start, stop in boundary.path.initial_intervals()]
    if len(pending) > rule.max_vertices:
        raise SasConversionError(
            f"initial boundary exceeds max_vertices={rule.max_vertices} before projection"
        )
    accepted: list[tuple[float, float, int, float]] = []
    cache: dict[float, np.ndarray] = {}
    commands: list[tuple[str, ...]] = []
    evaluations = 0
    max_observed_probe_error = 0.0
    max_accepted_probe_error = 0.0
    max_depth_reached = 0

    while pending:
        needed: set[float] = set()
        for start, stop, _depth in pending:
            span = stop - start
            needed.add(start)
            needed.add(stop)
            for fraction in _INTERIOR_PROBE_FRACTIONS:
                needed.add(start + fraction * span)
        missing = sorted(parameter for parameter in needed if parameter not in cache)
        if missing:
            sky = _skycoord_array([boundary.path.point(parameter) for parameter in missing])
            detector, command = _project_skycoords(
                sky,
                calinfoset=calinfoset,
                esky2det=esky2det,
                environment=environment,
            )
            commands.append(command)
            evaluations += len(missing)
            for parameter, point in zip(missing, detector, strict=True):
                cache[parameter] = point

        next_pending: list[tuple[float, float, int]] = []
        for start, stop, depth in pending:
            span = stop - start
            probe_parameters = tuple(
                start + fraction * span for fraction in _INTERIOR_PROBE_FRACTIONS
            )
            probe_errors = tuple(
                _point_to_segment_distance(cache[parameter], cache[start], cache[stop])
                for parameter in probe_parameters
            )
            error = max(probe_errors)
            max_observed_probe_error = max(max_observed_probe_error, error)
            max_depth_reached = max(max_depth_reached, depth)
            if error <= rule.detector_tolerance:
                accepted.append((start, stop, depth, error))
                max_accepted_probe_error = max(max_accepted_probe_error, error)
                continue
            if depth >= rule.max_depth:
                raise SasConversionError(
                    "adaptive detector refinement could not satisfy sampled probe tolerance "
                    f"{rule.detector_tolerance:g} DET units within max_depth={rule.max_depth}"
                )
            midpoint = (start + stop) / 2.0
            next_depth = depth + 1
            next_pending.extend(
                ((start, midpoint, next_depth), (midpoint, stop, next_depth))
            )
        if len(accepted) + len(next_pending) > rule.max_vertices:
            raise SasConversionError(
                "adaptive detector refinement exceeded max_vertices="
                f"{rule.max_vertices} before satisfying sampled probe tolerance"
            )
        pending = next_pending

    accepted.sort(key=lambda item: item[0])
    vertices = np.asarray([cache[start] for start, _stop, _depth, _error in accepted])
    _validate_detector_boundary(vertices)
    return (
        DetectorBoundary(vertices=vertices, subtract=boundary.subtract),
        commands,
        {
            "mode": rule.refinement,
            "detector_tolerance": rule.detector_tolerance,
            "interior_probe_fractions": list(_INTERIOR_PROBE_FRACTIONS),
            "max_accepted_probe_error": max_accepted_probe_error,
            "max_observed_probe_error": max_observed_probe_error,
            "max_depth_reached": max_depth_reached,
            "vertex_count": len(vertices),
            "projected_point_evaluations": evaluations,
            "sas_calls": len(commands),
        },
    )


def _validate_detector_boundary(vertices: np.ndarray) -> None:
    try:
        validate_detector_boundary(vertices)
    except DetectorTopologyError as exc:
        raise SasConversionError(str(exc)) from exc


def _validate_region_topology(regions: Sequence[DetectorRegion]) -> None:
    try:
        validate_detector_regions(regions)
    except DetectorTopologyError as exc:
        raise SasConversionError(str(exc)) from exc


def _rebuild_regions(
    source_regions: Sequence[CelestialRegion],
    boundaries: Sequence[DetectorBoundary],
) -> tuple[DetectorRegion, ...]:
    result: list[DetectorRegion] = []
    cursor = 0
    for region in source_regions:
        count = len(region.boundaries)
        result.append(
            DetectorRegion(
                include=region.include,
                boundaries=tuple(boundaries[cursor : cursor + count]),
            )
        )
        cursor += count
    return tuple(result)


def _adaptive_convert_regions(
    regions: Sequence[CelestialRegion],
    *,
    rule: ProjectionRule,
    calinfoset: str | Path,
    esky2det: str | Path,
    environment: Mapping[str, str] | None,
) -> tuple[tuple[DetectorRegion, ...], tuple[tuple[str, ...], ...], dict[str, object]]:
    detector_boundaries: list[DetectorBoundary] = []
    commands: list[tuple[str, ...]] = []
    diagnostics: list[dict[str, object]] = []
    for region_index, region in enumerate(regions):
        for boundary_index, boundary in enumerate(region.boundaries):
            detector_boundary, boundary_commands, boundary_diagnostics = _adaptive_boundary(
                boundary,
                rule=rule,
                calinfoset=calinfoset,
                esky2det=esky2det,
                environment=environment,
            )
            detector_boundaries.append(detector_boundary)
            commands.extend(boundary_commands)
            diagnostics.append(
                {
                    "region_index": region_index,
                    "boundary_index": boundary_index,
                    **boundary_diagnostics,
                }
            )
    detector_regions = _rebuild_regions(regions, detector_boundaries)
    _validate_region_topology(detector_regions)
    return (
        detector_regions,
        tuple(commands),
        {
            "mode": rule.refinement,
            "detector_tolerance": rule.detector_tolerance,
            "interior_probe_fractions": list(_INTERIOR_PROBE_FRACTIONS),
            "max_accepted_probe_error": max(
                (float(item["max_accepted_probe_error"]) for item in diagnostics),
                default=0.0,
            ),
            "max_observed_probe_error": max(
                (float(item["max_observed_probe_error"]) for item in diagnostics),
                default=0.0,
            ),
            "total_vertices": sum(int(item["vertex_count"]) for item in diagnostics),
            "total_projected_point_evaluations": sum(
                int(item["projected_point_evaluations"]) for item in diagnostics
            ),
            "sas_calls": len(commands),
            "boundaries": diagnostics,
        },
    )


def _convert_regions_with_esky2det(
    regions: Sequence[CelestialRegion],
    *,
    calinfoset: str | Path,
    rule: ProjectionRule,
    esky2det: str | Path = "esky2det",
    environment: Mapping[str, str] | None = None,
) -> tuple[tuple[DetectorRegion, ...], tuple[tuple[str, ...], ...], dict[str, object]]:
    if rule.refinement == LEGACY_REFINEMENT:
        return _legacy_convert_regions(
            regions,
            rule=rule,
            calinfoset=calinfoset,
            esky2det=esky2det,
            environment=environment,
        )
    return _adaptive_convert_regions(
        regions,
        rule=rule,
        calinfoset=calinfoset,
        esky2det=esky2det,
        environment=environment,
    )


def project_selection(
    selection: CelestialSelection,
    *,
    calinfoset: str | Path,
    context: SasProjectionContext,
    rule: ProjectionRule | None = None,
) -> ProjectionResult:
    """Project a semantic celestial selection using an explicit frozen SAS context."""
    if not isinstance(selection, CelestialSelection):
        raise TypeError(f"selection must be CelestialSelection, got {type(selection).__name__}")
    if not isinstance(context, SasProjectionContext):
        raise TypeError(f"context must be SasProjectionContext, got {type(context).__name__}")
    projection_rule = ProjectionRule() if rule is None else rule
    if not isinstance(projection_rule, ProjectionRule):
        raise TypeError("rule must be ProjectionRule")

    context.validate()

    event_path = Path(calinfoset).expanduser().resolve()
    event_identity, event_file_sha_before = read_science_calinfoset_snapshot(event_path)
    try:
        suitability_before = capture_context_suitability(
            environment=context.environment,
            calibration=context.calibration,
            event_identity=event_identity,
            declared_observation=context.observation_evidence,
        )
    except ContextSuitabilityError as exc:
        raise SasConversionError(
            f"SAS projection context is not suitable for this event: {exc}"
        ) from exc

    regions, commands, diagnostics = _convert_regions_with_esky2det(
        selection.regions,
        calinfoset=event_path,
        rule=projection_rule,
        esky2det=context.esky2det_path,
        environment=context.environment,
    )
    event_file_sha_after = file_sha256(event_path)
    if event_file_sha_after != event_file_sha_before:
        raise SasConversionError(
            "exact event file changed while esky2det projection was running; refusing to bind "
            "detector geometry to unstable calinfoset bytes",
            command=commands[-1] if commands else None,
        )

    context.validate()
    try:
        suitability_after = capture_context_suitability(
            environment=context.environment,
            calibration=context.calibration,
            event_identity=event_identity,
            declared_observation=context.observation_evidence,
        )
    except ContextSuitabilityError as exc:
        raise SasConversionError(
            f"SAS projection context became unsuitable while projecting: {exc}",
            command=commands[-1] if commands else None,
        ) from exc
    if suitability_after != suitability_before:
        raise SasConversionError(
            "observation-association/CIF-suitability evidence changed while esky2det "
            "projection was running; refusing stale-state substitution",
            command=commands[-1] if commands else None,
        )

    detector = DetectorSelection(
        regions=regions,
        source_geometry_sha256=selection.geometry_sha256,
    )
    primary_command = commands[0] if commands else ()
    provenance = ProjectionProvenance(
        event_file_sha256=event_file_sha_before,
        event_identity_sha256=event_identity.identity_sha256,
        celestial_geometry_sha256=selection.geometry_sha256,
        detector_geometry_sha256=detector.geometry_sha256,
        context_identity_sha256=context.identity_sha256,
        calibration_identity_sha256=context.calibration.identity_sha256,
        producer_identity_sha256=context.producer.identity_sha256,
        calibration_execution_evidence=context.calibration.evidence_record(),
        rule=projection_rule,
        command=primary_command,
        commands=commands,
        relevant_environment=context.relevant_environment_record(),
        refinement_diagnostics=diagnostics,
    )
    return ProjectionResult(selection=detector, provenance=provenance)


def project_with_esky2det(
    selection: CelestialSelection,
    *,
    calinfoset: str | Path,
    esky2det: str = "esky2det",
    environment: Mapping[str, str] | None = None,
    rule: ProjectionRule | None = None,
) -> DetectorSelection:
    """Convenience projection API using the supplied/ambient execution environment."""
    if not isinstance(selection, CelestialSelection):
        raise TypeError(f"selection must be CelestialSelection, got {type(selection).__name__}")
    projection_rule = ProjectionRule() if rule is None else rule
    regions, _commands, _diagnostics = _convert_regions_with_esky2det(
        selection.regions,
        calinfoset=calinfoset,
        rule=projection_rule,
        esky2det=esky2det,
        environment=environment,
    )
    return DetectorSelection(
        regions=regions,
        source_geometry_sha256=selection.geometry_sha256,
    )


def convert_with_esky2det(
    regions: Sequence[CelestialRegion],
    *,
    calinfoset: str | Path,
    esky2det: str = "esky2det",
) -> list[DetectorRegion]:
    """Compatibility shim retaining the original fixed-sample development behaviour."""
    converted, _commands, _diagnostics = _convert_regions_with_esky2det(
        regions,
        calinfoset=calinfoset,
        rule=ProjectionRule(samples=128),
        esky2det=esky2det,
    )
    return list(converted)

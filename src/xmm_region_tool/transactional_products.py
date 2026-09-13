"""Transactional publication of one projected SAS region product."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from .artifacts import write_projection_evidence
from .calibration import CalibrationIdentity, SasProducerIdentity
from .execution import ProjectionResult
from .output import WrittenRegionFiles, write_esas_regionfile
from .provenance import EventIdentity
from .publication import (
    discard_staged,
    lexical_absolute_path,
    publish_staged_files,
    temporary_sibling,
)
from .sas_paths import SasPathError, validate_sas_safe_lexical_path


class ProductPublicationError(ValueError):
    """Raised when a complete projected product cannot be staged safely."""


def planned_fits_output(regionfile: str | Path) -> Path:
    """Return the companion FITS name used by :func:`write_esas_regionfile`."""
    output = lexical_absolute_path(regionfile)
    suffix = output.suffix
    if suffix.lower() == ".fits":
        return output.with_name(output.stem + "-geometry.fits")
    if suffix:
        return output.with_suffix(".fits")
    return output.with_name(output.name + ".fits")


def _rewrite_staged_wrapper(stage_wrapper: Path, final_geometry: Path) -> None:
    try:
        safe_geometry = validate_sas_safe_lexical_path(
            final_geometry,
            role="FITS REGION geometry",
        )
    except SasPathError as exc:
        raise ProductPublicationError(str(exc)) from exc
    stage_wrapper.write_text(f"&&region({safe_geometry},DETX,DETY)\n")


def _rewrite_staged_evidence_geometry_reference(
    stage_evidence: Path,
    final_geometry: Path,
) -> None:
    payload = json.loads(stage_evidence.read_text())
    if not isinstance(payload, dict):
        raise ProductPublicationError("staged projection evidence is not a JSON object")
    geometry = payload.get("geometry_artifact")
    if not isinstance(geometry, dict):
        raise ProductPublicationError(
            "staged FITS-backed projection evidence has no geometry_artifact record"
        )
    geometry["path"] = str(lexical_absolute_path(final_geometry))
    stage_evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_projected_product_atomic(
    output: str | Path,
    projected: ProjectionResult,
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
    source_region_origin: str | None = None,
    precommit: Callable[[WrittenRegionFiles], Any] | None = None,
) -> tuple[WrittenRegionFiles, Path, Any]:
    """Stage, validate and atomically publish one complete supported product.

    The existing serializers write only to private sibling paths. Any optional
    ``precommit`` callback runs after serialization/evidence construction but
    before final publication, so product-local checks such as pn OOT resolution
    cannot leave normal-named partial artifacts behind.
    """
    final_wrapper = lexical_absolute_path(output)
    final_evidence = final_wrapper.with_suffix(".provenance.json")
    final_fits = planned_fits_output(final_wrapper)

    stage_wrapper = temporary_sibling(final_wrapper)
    staged_paths: list[Path] = [stage_wrapper]
    stage_evidence: Path | None = None
    stage_fits: Path | None = None
    try:
        staged = write_esas_regionfile(
            stage_wrapper,
            list(projected.selection.regions),
            representation=representation,
            inline_limit=inline_limit,
            max_components=max_components,
            event_identity=event_identity,
            event_file_sha256=event_file_sha256,
            source_region_sha256=source_region_sha256,
            celestial_geometry_sha256=celestial_geometry_sha256,
            projection_identity_sha256=projection_identity_sha256,
            calibration_identity=calibration_identity,
            sas_producer=sas_producer,
        )
        stage_fits = staged.fits_region
        if stage_fits is not None:
            staged_paths.append(stage_fits)
            _rewrite_staged_wrapper(stage_wrapper, final_fits)

        stage_evidence = temporary_sibling(final_evidence)
        staged_paths.append(stage_evidence)
        evidence_kwargs: dict[str, object] = {
            "calibration_identity": calibration_identity,
            "sas_producer": sas_producer,
            "source_region_origin": source_region_origin,
        }
        if stage_fits is not None:
            write_projection_evidence(
                stage_evidence,
                projected,
                geometry_artifact=stage_fits,
                **evidence_kwargs,
            )
            _rewrite_staged_evidence_geometry_reference(stage_evidence, final_fits)
        else:
            write_projection_evidence(stage_evidence, projected, **evidence_kwargs)

        final_written = WrittenRegionFiles(
            regionfile=final_wrapper,
            representation=staged.representation,
            fits_region=final_fits if stage_fits is not None else None,
        )
        precommit_result = precommit(final_written) if precommit is not None else None

        pairs: list[tuple[Path, Path]] = [(stage_wrapper, final_wrapper)]
        if stage_fits is not None:
            pairs.append((stage_fits, final_fits))
        pairs.append((stage_evidence, final_evidence))
        publish_staged_files(pairs)
        return final_written, final_evidence, precommit_result
    except Exception:
        discard_staged(staged_paths)
        raise


__all__ = [
    "ProductPublicationError",
    "planned_fits_output",
    "write_projected_product_atomic",
]

"""Command-line interface for XMM detector-region conversion."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

from .event_roles import validate_projection_event_generation
from .execution import ProjectionRule, SasProjectionContext
from .geometry import load_ds9_selection_snapshot
from .path_safety import (
    PathSafetyError,
    resolved_descendant,
    validate_distinct_inputs,
    validate_output_namespace,
)
from .provenance import EventIdentity, file_sha256, read_event_identity
from .publication import atomic_write_text, lexical_absolute_path
from .sas import SasConversionError, project_selection
from .transactional_products import write_projected_product_atomic
from .workflow import (
    ExtractionCell,
    WorkflowError,
    discover_all_event_files,
    discover_event_files,
    extraction_cells,
    output_basename,
    pn_oot_event_file,
    safe_source_stem,
)


_UNRESOLVED_OOT = object()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmm-region",
        description=(
            "Convert DS9 celestial extraction cells to camera-specific XMM EPIC "
            "DETX/DETY region files for mosspectra/pnspectra. Positive DS9 regions "
            "are split into independent extraction cells by default."
        ),
    )
    parser.add_argument("region", type=Path, help="input DS9 sky-coordinate region file")
    parser.add_argument(
        "--event-file",
        action="append",
        type=Path,
        dest="event_files",
        help=(
            "exact filtered EPIC event file; repeat for several cameras. If omitted, "
            "current-ESAS *-allevc.fits products can be discovered in --search-dir/current directory"
        ),
    )
    parser.add_argument(
        "--instrument",
        nargs="+",
        choices=("mos1", "mos2", "pn"),
        help=(
            "camera(s) to discover automatically, e.g. --instrument mos1 mos2 pn. "
            "If omitted, all unambiguous current-ESAS *-allevc.fits camera products "
            "present in --search-dir/current directory are selected automatically"
        ),
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help="directory used for automatic event-file discovery (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "exact output text regionfile name; valid only when one extraction cell and one "
            "event product are targeted. Otherwise names are generated automatically"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for automatically named outputs (default: current directory)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "treat the full DS9 selection as one union selection instead of splitting positive "
            "regions/panda cells into independent extraction cells"
        ),
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress normal progress and usage guidance; errors still use stderr",
    )
    verbosity.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "show full paths, projection diagnostics, manifest path and complete suggested "
            "mosspectra/pnspectra commands"
        ),
    )
    parser.add_argument(
        "--detector-tolerance",
        type=float,
        default=1.0,
        help=(
            "maximum accepted sampled interior-probe-to-detector-chord deviation in native "
            "DETX/DETY units for adaptive refinement (default: 1.0 DET unit = 0.05 arcsec)"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="maximum adaptive subdivision depth before failing closed (default: 12)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=4096,
        help="maximum detector vertices per boundary during adaptive refinement (default: 4096)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "explicit legacy fixed source-boundary sampling for comparison/debugging; "
            "supplying this disables adaptive refinement"
        ),
    )
    parser.add_argument(
        "--representation",
        choices=("fits", "expression", "auto"),
        default="fits",
        help=(
            "output representation: FITS companion + short region() wrapper (default), "
            "direct selectlib expression, or automatic based on expression length"
        ),
    )
    parser.add_argument(
        "--inline-limit",
        type=int,
        default=4096,
        help="maximum text length used by --representation auto (default: 4096)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=4096,
        help="safety limit for FITS boolean-component expansion (default: 4096)",
    )
    return parser


def _resolve_events(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.event_files and args.instrument:
        raise WorkflowError("use either explicit --event-file values or --instrument discovery, not both")
    if args.event_files:
        return tuple(Path(value).expanduser().resolve() for value in args.event_files)

    search_dir = (args.search_dir or Path.cwd()).expanduser().resolve()
    if args.instrument:
        return discover_event_files(search_dir, args.instrument)

    return discover_all_event_files(search_dir)


def _rule(args: argparse.Namespace) -> ProjectionRule:
    if args.samples is not None:
        return ProjectionRule(samples=args.samples)
    return ProjectionRule(
        detector_tolerance=args.detector_tolerance,
        max_depth=args.max_depth,
        max_vertices=args.max_vertices,
    )


def _cells(args: argparse.Namespace, selection) -> tuple[ExtractionCell, ...]:
    if args.combine:
        return (ExtractionCell(index=1, label="combined", selection=selection),)
    return extraction_cells(selection)


def _validate_event_set(event_files: tuple[Path, ...]) -> tuple[tuple[Path, EventIdentity], ...]:
    if not event_files:
        raise WorkflowError("at least one event file is required")
    try:
        validate_distinct_inputs(
            tuple((f"event file {index}", path) for index, path in enumerate(event_files, start=1))
        )
    except PathSafetyError as exc:
        raise WorkflowError(f"duplicate/aliased event input: {exc}") from exc
    resolved = tuple((path, read_event_identity(path)) for path in event_files)
    obs_ids = {identity.obs_id for _, identity in resolved}
    if len(obs_ids) > 1:
        joined = ", ".join(sorted(obs_ids))
        raise WorkflowError(
            "one standalone xmm-region invocation uses one frozen SAS calibration context; "
            f"event files span multiple ObsIDs ({joined}). Run once per observation"
        )
    return resolved


def _default_output(
    args: argparse.Namespace,
    cell: ExtractionCell,
    identity: EventIdentity,
    *,
    cell_count: int,
    duplicate_instrument: bool,
) -> Path:
    root = args.output_dir.expanduser().resolve()
    base = output_basename(
        args.region,
        cell,
        identity,
        cell_count=cell_count,
        duplicate_instrument=duplicate_instrument,
    )
    candidate = root / f"{base}.txt"
    resolved_descendant(root, candidate, role="region wrapper")
    return lexical_absolute_path(candidate)


def _manifest_root(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return lexical_absolute_path(args.output).parent
    return args.output_dir.expanduser().resolve()


def _planned_fits_output(output: Path) -> Path:
    suffix = output.suffix
    if suffix.lower() == ".fits":
        return output.with_name(output.stem + "-geometry.fits")
    if suffix:
        return output.with_suffix(".fits")
    return output.with_name(output.name + ".fits")


def _esas_command(
    identity: EventIdentity,
    event_file: Path,
    regionfile: Path,
    *,
    oot_file: Path | None | object = _UNRESOLVED_OOT,
) -> str:
    values = [
        "pnspectra" if identity.instrument == "pn" else "mosspectra",
        f"eventfile={event_file}",
    ]
    if identity.instrument == "pn":
        oot = pn_oot_event_file(event_file) if oot_file is _UNRESOLVED_OOT else oot_file
        if oot is not None:
            values.append(f"ootevtfile={oot}")
    values.extend(("withregion=yes", f"regionfile={regionfile}"))
    return shlex.join(values)


def _display_path(path: Path) -> str:
    """Prefer a short path relative to the current directory when possible."""
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    text = str(relative)
    return text if text else "."


def _print_compact_summary(
    *,
    products: list[dict[str, object]],
    manifest: Path,
    event_set: tuple[tuple[Path, EventIdentity], ...],
    cell_count: int,
    oot_by_event: dict[Path, Path | None] | None = None,
) -> None:
    successful = [item for item in products if item.get("status") == "success"]
    count = len(successful)
    noun = "region file" if count == 1 else "region files"
    print()
    print(f"Created {count} SAS {noun}.")

    output_dirs = {Path(str(item["regionfile"])).parent for item in successful}
    if len(output_dirs) == 1:
        print(f"Output: {_display_path(next(iter(output_dirs)))}")

    if count <= 12:
        for item in successful:
            label = str(item["cell_label"])
            instrument = str(item["instrument"]).upper()
            filename = Path(str(item["regionfile"])).name
            if cell_count == 1:
                print(f"  {instrument:<4} {filename}")
            else:
                print(f"  {label:<8} {instrument:<4} {filename}")
    else:
        by_instrument: dict[str, list[str]] = {}
        for item in successful:
            instrument = str(item["instrument"]).upper()
            by_instrument.setdefault(instrument, []).append(Path(str(item["regionfile"])).name)
        for instrument, filenames in sorted(by_instrument.items()):
            first = filenames[0]
            last = filenames[-1]
            if len(filenames) == 1:
                print(f"  {instrument:<4} {first}")
            else:
                print(f"  {instrument:<4} {len(filenames)} files: {first} ... {last}")

    print(f"Manifest: {manifest.name}")

    if not successful:
        return

    print()
    print("Next:")
    seen: set[tuple[str, str]] = set()
    for event_path, identity in event_set:
        key = (identity.instrument, str(event_path))
        if key in seen:
            continue
        seen.add(key)
        task = "pnspectra" if identity.instrument == "pn" else "mosspectra"
        print(f"  {identity.instrument.upper()}: run {task} once per generated .txt file.")
        print(f"    Event: {event_path.name}")
        if identity.instrument == "pn":
            oot = (
                oot_by_event[event_path]
                if oot_by_event is not None
                else pn_oot_event_file(event_path)
            )
            if oot is not None:
                print(f"    OOT: {oot.name}")
            else:
                print("    OOT: specify the matching pn OOT event file")
    print("    Region: choose one of the .txt files above")
    if cell_count > 1:
        print("  Use a separate working directory per spectrum to avoid ESAS output overwrites.")
    print("  Use --verbose for full paths, projection diagnostics and copy/paste commands.")


def _print_verbose_summary(
    *,
    manifest: Path,
    successful_commands: list[tuple[str, str]],
    event_set: tuple[tuple[Path, EventIdentity], ...],
    cell_count: int,
    oot_by_event: dict[Path, Path | None] | None = None,
) -> None:
    print(f"Manifest: {manifest}")
    if not successful_commands:
        return
    print()
    print("Suggested ESAS region runs:")
    print(
        "  Add your normal CCD/quadrant, source-removal, image and energy settings; "
        "the region/event arguments below are the values generated by xmm-region."
    )
    for label, command in successful_commands:
        print(f"  {label}: {command}")
    if any(
        identity.instrument == "pn"
        and (
            oot_by_event[path]
            if oot_by_event is not None
            else pn_oot_event_file(path)
        )
        is None
        for path, identity in event_set
    ):
        print(
            "  pn note: no current sibling P-allevcoot.fits (or supported compatibility "
            "P-allevc-oot.fits) was found for at least one pn event file; specify the correct "
            "pnspectra ootevtfile explicitly."
        )
    if cell_count > 1:
        print()
        print(
            "Note: mosspectra/pnspectra derive many product names from the event prefix. "
            "When extracting several cells, protect or relocate each run's products before "
            "running the next cell to avoid overwriting them."
        )


def _run(
    args: argparse.Namespace,
    *,
    context: SasProjectionContext | None = None,
) -> int:
    rule = _rule(args)
    event_files = _resolve_events(args)
    event_set = _validate_event_set(event_files)
    event_sha_by_path = {path: file_sha256(path) for path, _identity in event_set}
    if context is None:
        context = SasProjectionContext.from_environment()
    source_path = args.region.expanduser().resolve()
    sampling_hint = args.samples if args.samples is not None else 128
    celestial_selection, source_region_sha256 = load_ds9_selection_snapshot(
        source_path,
        samples=sampling_hint,
    )
    cells = _cells(args, celestial_selection)

    product_count = len(cells) * len(event_set)
    if args.output is not None and product_count != 1:
        raise WorkflowError(
            "--output can only be used when exactly one extraction cell and one event product "
            f"are targeted; this invocation would create {product_count} products. Use "
            "--output-dir or --combine/--instrument to narrow the run"
        )

    instrument_counts = Counter(identity.instrument for _, identity in event_set)
    planned_outputs: dict[tuple[int, Path], Path] = {}
    generated_namespace: list[tuple[str, Path]] = []
    automatic_root = args.output_dir.expanduser().resolve()
    for cell in cells:
        for event_path, event_identity in event_set:
            product_role = f"{cell.label}/{event_identity.instrument}"
            output = (
                lexical_absolute_path(args.output)
                if args.output is not None
                else _default_output(
                    args,
                    cell,
                    event_identity,
                    cell_count=len(cells),
                    duplicate_instrument=instrument_counts[event_identity.instrument] > 1,
                )
            )
            planned_outputs[(cell.index, event_path)] = output

            artifact_root = output.parent if args.output is not None else automatic_root
            if args.output is None:
                resolved_descendant(
                    artifact_root,
                    output,
                    role="region wrapper",
                )
            generated_namespace.append((f"region wrapper {product_role}", output))
            evidence = resolved_descendant(
                artifact_root,
                output.with_suffix(".provenance.json"),
                role="projection evidence sidecar",
            )
            generated_namespace.append((f"projection sidecar {product_role}", evidence))
            if args.representation in {"fits", "auto"}:
                fits_output = resolved_descendant(
                    artifact_root,
                    _planned_fits_output(output),
                    role="FITS REGION geometry",
                )
                generated_namespace.append((f"FITS geometry {product_role}", fits_output))

    manifest_root = _manifest_root(args)
    manifest_candidate = manifest_root / f"{safe_source_stem(source_path)}-xmm-region-manifest.json"
    resolved_descendant(manifest_root, manifest_candidate, role="workflow manifest")
    manifest = lexical_absolute_path(manifest_candidate)
    generated_namespace.append(("workflow manifest", manifest))
    protected_inputs: list[tuple[str, Path]] = [("source DS9 region", source_path)]
    protected_inputs.extend(
        (f"event input {identity.instrument}/{identity.exposure_id or identity.obs_id}", path)
        for path, identity in event_set
    )
    protected_inputs.append(("active CIF", context.calibration.cif_path))
    try:
        validate_output_namespace(generated_namespace, protected_inputs)
    except PathSafetyError as exc:
        raise WorkflowError(f"unsafe output namespace: {exc}") from exc

    # Resolve pn OOT guidance once before any product is staged or published. This
    # prevents late sibling ambiguity from turning a completed product/manifest
    # into an overall command failure.
    oot_by_event: dict[Path, Path | None] = {}
    for event_path, identity in event_set:
        oot_by_event[event_path] = (
            pn_oot_event_file(event_path) if identity.instrument == "pn" else None
        )

    if args.verbose:
        print(f"Source region: {source_path}")
        print(f"Extraction cells: {len(cells)}" + (" (combined)" if args.combine else ""))
        print(
            "Event products: "
            + ", ".join(
                f"{identity.instrument}:{path.name}" for path, identity in event_set
            )
        )

    products: list[dict[str, object]] = []
    failures = 0
    successful_commands: list[tuple[str, str]] = []

    for cell in cells:
        for event_path, event_identity in event_set:
            output = planned_outputs[(cell.index, event_path)]

            item: dict[str, object] = {
                "cell_index": cell.index,
                "cell_label": cell.label,
                "celestial_geometry_sha256": cell.selection.geometry_sha256,
                "celestial_geometry": cell.selection.canonical_geometry_record(),
                "event_file": str(event_path),
                "instrument": event_identity.instrument,
                "obs_id": event_identity.obs_id,
                "exposure_id": event_identity.exposure_id,
            }
            try:
                projected = project_selection(
                    cell.selection,
                    calinfoset=event_path,
                    context=context,
                    rule=rule,
                )
                authoritative_identity = validate_projection_event_generation(
                    event_path,
                    event_identity,
                    projection_event_identity_sha256=(
                        projected.provenance.event_identity_sha256
                    ),
                    projection_event_file_sha256=projected.provenance.event_file_sha256,
                    expected_event_file_sha256=event_sha_by_path[event_path],
                )

                def precommit(written):
                    return _esas_command(
                        authoritative_identity,
                        event_path,
                        written.regionfile,
                        oot_file=oot_by_event[event_path],
                    )

                written, evidence_path, command = write_projected_product_atomic(
                    output,
                    projected,
                    representation=args.representation,
                    inline_limit=args.inline_limit,
                    max_components=args.max_components,
                    event_identity=authoritative_identity,
                    event_file_sha256=projected.provenance.event_file_sha256,
                    source_region_sha256=source_region_sha256,
                    celestial_geometry_sha256=cell.selection.geometry_sha256,
                    projection_identity_sha256=(
                        projected.provenance.projection_identity_sha256
                    ),
                    calibration_identity=context.calibration,
                    sas_producer=context.producer,
                    source_region_origin="tool-hashed-source-file",
                    precommit=precommit,
                )

                item.update(
                    {
                        "status": "success",
                        "event_file_sha256": projected.provenance.event_file_sha256,
                        "event_identity_sha256": projected.provenance.event_identity_sha256,
                        "regionfile": str(written.regionfile),
                        "fits_region": str(written.fits_region) if written.fits_region else None,
                        "projection_evidence": str(evidence_path),
                        "detector_geometry_sha256": projected.selection.geometry_sha256,
                        "projection_identity_sha256": (
                            projected.provenance.projection_identity_sha256
                        ),
                        "refinement_diagnostics": projected.provenance.evidence_record()[
                            "refinement_diagnostics"
                        ],
                    }
                )
                successful_commands.append(
                    (f"{cell.label}/{authoritative_identity.instrument}", command)
                )
                if args.verbose:
                    diagnostics = projected.provenance.refinement_diagnostics
                    detail = ""
                    if rule.adaptive:
                        detail = (
                            f"; vertices={diagnostics.get('total_vertices')}"
                            f"; max_error={diagnostics.get('max_accepted_error')} DET"
                        )
                    print(
                        f"OK {cell.label}/{authoritative_identity.instrument} -> "
                        f"{written.regionfile}{detail}"
                    )
            except (SasConversionError, ValueError, OSError) as exc:
                failures += 1
                item.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "failure_evidence": (
                            exc.evidence_record() if isinstance(exc, SasConversionError) else None
                        ),
                    }
                )
                print(
                    f"xmm-region: error: {cell.label}/{event_identity.instrument}: {exc}",
                    file=sys.stderr,
                )
            products.append(item)

    manifest_payload = {
        "schema": "xmm-region-tool.workflow/v1",
        "source_region": str(source_path),
        "source_region_sha256": source_region_sha256,
        "source_region_origin": "tool-hashed-source-file",
        "source_selection_geometry_sha256": celestial_selection.geometry_sha256,
        "source_selection_geometry": celestial_selection.canonical_geometry_record(),
        "split_extraction_cells": not args.combine,
        "cell_count": len(cells),
        "cells": [
            {
                "cell_index": cell.index,
                "cell_label": cell.label,
                "celestial_geometry_sha256": cell.selection.geometry_sha256,
                "celestial_geometry": cell.selection.canonical_geometry_record(),
            }
            for cell in cells
        ],
        "projection_rule": rule.canonical_record(),
        "context_identity_sha256": context.identity_sha256,
        "calibration_identity_sha256": context.calibration.identity_sha256,
        "producer_identity_sha256": context.producer.identity_sha256,
        "successful": failures == 0,
        "products": products,
    }
    atomic_write_text(manifest, json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        if args.verbose:
            _print_verbose_summary(
                manifest=manifest,
                successful_commands=successful_commands,
                event_set=event_set,
                cell_count=len(cells),
                oot_by_event=oot_by_event,
            )
        else:
            _print_compact_summary(
                products=products,
                manifest=manifest,
                event_set=event_set,
                cell_count=len(cells),
                oot_by_event=oot_by_event,
            )

    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (SasConversionError, WorkflowError, ValueError, OSError) as exc:
        print(f"xmm-region: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

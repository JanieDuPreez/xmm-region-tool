"""High-level extraction-cell workflow helpers.

This module sits above the exact one-selection/one-event projection primitives.
It converts a user-authored celestial selection into independently extractable
science cells and supports fail-closed discovery of canonical ESAS event files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model import CelestialSelection
from .path_safety import safe_filename_component
from .provenance import EventIdentity, file_sha256, read_event_identity


class WorkflowError(ValueError):
    """Raised when the high-level workflow cannot resolve inputs unambiguously."""


@dataclass(frozen=True)
class ExtractionCell:
    """One independently extractable celestial science cell."""

    index: int
    label: str
    selection: CelestialSelection


_INSTRUMENTS = ("mos1", "mos2", "pn")


def extraction_cells(selection: CelestialSelection) -> tuple[ExtractionCell, ...]:
    """Split positive regions into independent cells while retaining exclusions.

    Every positive top-level celestial region becomes one output science cell.
    All negative top-level regions are applied to every positive cell. This
    matches the usual spectral-bin workflow where point-source or mask
    exclusions modify each annulus/sector rather than becoming spectra.

    An exclusion-only selection remains one cell.
    """
    positive = [region for region in selection.regions if region.include]
    negative = tuple(region for region in selection.regions if not region.include)

    if not positive:
        return (ExtractionCell(index=1, label="r001", selection=selection),)

    cells: list[ExtractionCell] = []
    for index, region in enumerate(positive, start=1):
        cell_selection = CelestialSelection(
            regions=(region, *negative),
            external_identity=selection.external_identity,
            external_provenance=selection.external_provenance,
        )
        cells.append(
            ExtractionCell(
                index=index,
                label=f"r{index:03d}",
                selection=cell_selection,
            )
        )
    return tuple(cells)


def _canonical_event_candidates(root: Path, instrument: str) -> list[tuple[Path, EventIdentity]]:
    """Return current-cookbook ESAS filtered event products for one camera."""
    if instrument not in _INSTRUMENTS:
        raise WorkflowError(f"unsupported EPIC instrument {instrument!r}")

    # Current ESAS (SAS 22 cookbook) uses espfilt's P-allevc.fits as the
    # filtered event list for subsequent mosspectra/pnspectra work.
    pattern = f"{instrument}*-allevc.fits"
    matches: list[tuple[Path, EventIdentity]] = []
    for path in sorted(root.glob(pattern)):
        try:
            identity = read_event_identity(path)
        except Exception:
            continue
        if identity.instrument == instrument:
            matches.append((path.resolve(), identity))
    return matches


def discover_event_files(
    directory: str | Path,
    instruments: Iterable[str],
) -> tuple[Path, ...]:
    """Resolve one canonical current-ESAS filtered event file per instrument.

    Automatic discovery intentionally accepts only the ``P-allevc.fits``
    products produced by ``espfilt`` and recommended by the current ESAS
    cookbook for subsequent work. Zero or multiple valid candidates fail
    closed. Explicit ``--event-file`` paths can always bypass filename-based
    discovery when the user has a different valid event product.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise WorkflowError(f"event discovery directory does not exist: {root}")

    requested = tuple(dict.fromkeys(value.lower() for value in instruments))
    if not requested:
        raise WorkflowError("at least one EPIC instrument must be requested")

    resolved: list[Path] = []
    for instrument in requested:
        candidates = _canonical_event_candidates(root, instrument)
        if not candidates:
            raise WorkflowError(
                f"no canonical {instrument} ESAS filtered event file found in {root}; "
                f"expected a valid {instrument}*-allevc.fits file or supply --event-file explicitly"
            )
        if len(candidates) > 1:
            joined = ", ".join(path.name for path, _ in candidates)
            raise WorkflowError(
                f"multiple canonical {instrument} ESAS filtered event files found in {root}: "
                f"{joined}; supply --event-file explicitly rather than guessing"
            )
        resolved.append(candidates[0][0])
    return tuple(resolved)


def discover_all_event_files(directory: str | Path) -> tuple[Path, ...]:
    """Discover every unambiguous current-ESAS filtered EPIC event file.

    If an instrument has more than one canonical candidate, discovery fails
    rather than silently omitting or choosing one. Instruments with no candidate
    are simply absent. At least one event file must be found.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise WorkflowError(f"event discovery directory does not exist: {root}")

    resolved: list[Path] = []
    for instrument in _INSTRUMENTS:
        candidates = _canonical_event_candidates(root, instrument)
        if len(candidates) > 1:
            joined = ", ".join(path.name for path, _ in candidates)
            raise WorkflowError(
                f"multiple canonical {instrument} ESAS filtered event files found in {root}: "
                f"{joined}; use --instrument with an unambiguous directory or supply "
                "--event-file explicitly"
            )
        if len(candidates) == 1:
            resolved.append(candidates[0][0])

    if not resolved:
        raise WorkflowError(
            f"no canonical ESAS filtered event files found in {root}; expected names such as "
            "mos1S001-allevc.fits, mos2S002-allevc.fits or pnS003-allevc.fits, or supply "
            "--event-file explicitly"
        )
    return tuple(resolved)


def pn_oot_event_file(event_file: str | Path) -> Path | None:
    """Return an identity-compatible, physically distinct pn OOT sibling."""
    path = Path(event_file).expanduser().resolve()
    if not path.name.startswith("pn") or not path.name.endswith("-allevc.fits"):
        return None

    try:
        science_identity = read_event_identity(path)
    except Exception as exc:
        raise WorkflowError(
            f"cannot identify pn science event file before resolving OOT sibling: {path}"
        ) from exc

    if science_identity.instrument != "pn":
        return None

    prefix = path.name.removesuffix("-allevc.fits")

    # Current real espfilt products use P-allevcoot.fits. Keep the hyphenated
    # P-allevc-oot.fits form as a compatibility spelling for older products.
    candidates = (
        path.with_name(prefix + "-allevcoot.fits"),
        path.with_name(prefix + "-allevc-oot.fits"),
    )
    existing = tuple(candidate for candidate in candidates if candidate.is_file())

    if len(existing) > 1:
        raise WorkflowError(
            "multiple matching pn OOT event products found: "
            + ", ".join(candidate.name for candidate in existing)
            + "; remove the ambiguity or specify the pnspectra ootevtfile explicitly"
        )

    if not existing:
        return None

    oot_path = existing[0].resolve()
    if oot_path == path or oot_path.samefile(path):
        raise WorkflowError(
            "matching pn OOT sibling resolves to the same physical file as the science event; "
            "eventfile and ootevtfile must be distinct products"
        )
    if oot_path.stat().st_size == path.stat().st_size and file_sha256(oot_path) == file_sha256(path):
        raise WorkflowError(
            "matching pn OOT sibling is byte-identical to the science event; refusing ambiguous "
            "eventfile/ootevtfile roles"
        )

    try:
        oot_identity = read_event_identity(oot_path)
    except Exception as exc:
        raise WorkflowError(
            f"matching pn OOT sibling is not a readable XMM event product: {oot_path}"
        ) from exc

    mismatches: list[str] = []
    if oot_identity.instrument != science_identity.instrument:
        mismatches.append(
            f"instrument {oot_identity.instrument!r} != {science_identity.instrument!r}"
        )
    if oot_identity.obs_id != science_identity.obs_id:
        mismatches.append(
            f"ObsID {oot_identity.obs_id!r} != {science_identity.obs_id!r}"
        )
    if oot_identity.exposure_id != science_identity.exposure_id:
        mismatches.append(
            f"exposure {oot_identity.exposure_id!r} != {science_identity.exposure_id!r}"
        )

    if mismatches:
        raise WorkflowError(
            "matching pn OOT sibling does not belong to the science event: "
            + "; ".join(mismatches)
        )

    return oot_path


def safe_source_stem(path: str | Path) -> str:
    """Return a filesystem-safe output stem derived from the DS9 filename."""
    return safe_filename_component(Path(path).stem, fallback="region")


def output_basename(
    source_region: str | Path,
    cell: ExtractionCell,
    identity: EventIdentity,
    *,
    cell_count: int,
    duplicate_instrument: bool,
) -> str:
    """Build a deterministic human-readable output basename.

    Every metadata-derived component is sanitized independently before joining,
    so FITS header values can never introduce path separators or ``..``
    traversal components into an automatic output name.
    """
    parts = [safe_source_stem(source_region)]
    if cell_count > 1:
        parts.append(safe_filename_component(cell.label))
    parts.append(safe_filename_component(identity.instrument))
    if duplicate_instrument:
        parts.append(safe_filename_component(identity.exposure_id or identity.obs_id))
    return "-".join(parts)

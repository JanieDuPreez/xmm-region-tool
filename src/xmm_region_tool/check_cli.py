"""CLI for checking whether a generated detector region belongs to its inputs."""

from __future__ import annotations

from .limits import bounded_text

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from .artifacts import ArtifactMaterializationError
from .provenance import RegionBindingError, check_region_binding
from .public_managed import load_bound_detector_geometry
from .terminal import safe_terminal_text

_WORKFLOW_SCHEMA = "xmm-region-tool.workflow/v1"
_BATCH_SCHEMA = "xmm-region-tool.batch/v4"


@dataclass(frozen=True)
class _ManifestSelection:
    celestial_geometry_sha256: str
    projection_identity_sha256: str
    projection_evidence: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmm-region-check",
        description=(
            "Check that a generated FITS detector region is bound to the supplied "
            "XMM EPIC event file and, optionally, its source DS9 file/workflow manifest. "
            "Manifest-backed checks also verify current detector-geometry bytes against "
            "the matching projection-evidence sidecar."
        ),
    )
    parser.add_argument("region_fits", type=Path, help="generated FITS REGION file")
    parser.add_argument(
        "--event-file",
        required=True,
        type=Path,
        help="EPIC event file that will consume the detector region",
    )
    parser.add_argument(
        "--source-region",
        type=Path,
        help="optional original DS9 sky-region file whose SHA256 must also match",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "optional xmm-region workflow/v1 or batch/v4 manifest; the exact matching "
            "product supplies the expected identities and authoritative projection sidecar"
        ),
    )
    return parser


def _manifest_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"matching manifest product has no valid {description} SHA256")
    digest = value.lower()
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(
            f"matching manifest product has no valid {description} SHA256"
        ) from exc
    return digest


def _workflow_matches(
    payload: dict[str, object],
    *,
    target: Path,
) -> list[dict[str, object]]:
    products = payload.get("products")
    if not isinstance(products, list):
        raise ValueError("workflow/v1 manifest has no products list")

    matches: list[dict[str, object]] = []
    for index, product in enumerate(products):
        if not isinstance(product, dict):
            raise ValueError(f"workflow/v1 manifest product {index} is not an object")
        fits_region = product.get("fits_region")
        if fits_region is None:
            continue
        if not isinstance(fits_region, str) or not fits_region:
            raise ValueError(
                f"workflow/v1 manifest product {index} has an invalid fits_region"
            )
        candidate_path = Path(fits_region).expanduser()
        if not candidate_path.is_absolute():
            raise ValueError(
                f"workflow/v1 manifest product {index} fits_region must be absolute"
            )
        if candidate_path.resolve() == target:
            matches.append(product)
    return matches


def _batch_matches(
    payload: dict[str, object],
    *,
    manifest_root: Path,
    target: Path,
) -> list[dict[str, object]]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("batch/v4 manifest has no items list")

    matches: list[dict[str, object]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"batch/v4 manifest item {index} is not an object")
        status = item.get("status")
        if status not in {"success", "failed"}:
            raise ValueError(f"batch/v4 manifest item {index} has an invalid status")
        fits_region = item.get("fits_region")
        if fits_region is None:
            continue
        if not isinstance(fits_region, str) or not fits_region:
            raise ValueError(f"batch/v4 manifest item {index} has an invalid fits_region")
        relative = Path(fits_region).expanduser()
        if relative.is_absolute():
            raise ValueError(
                f"batch/v4 manifest item {index} fits_region must be relative to the manifest"
            )
        candidate = (manifest_root / relative).resolve()
        if candidate != target:
            continue
        if status != "success":
            raise ValueError("matching batch/v4 manifest item is not successful")
        matches.append(item)
    return matches


def _projection_evidence_path(
    product: dict[str, object],
    *,
    schema: str,
    manifest_root: Path,
) -> Path:
    value = product.get("projection_evidence")
    if not isinstance(value, str) or not value:
        raise ValueError("matching manifest product has no valid projection_evidence path")
    evidence = Path(value).expanduser()
    if schema == _WORKFLOW_SCHEMA:
        if not evidence.is_absolute():
            raise ValueError(
                "matching workflow/v1 product projection_evidence path must be absolute"
            )
        return evidence.resolve()
    if evidence.is_absolute():
        raise ValueError(
            "matching batch/v4 item projection_evidence path must be relative to the manifest"
        )
    return (manifest_root / evidence).resolve()


def _manifest_selection(
    manifest_path: Path,
    region_fits: Path,
    *,
    require_projection_evidence: bool = True,
) -> _ManifestSelection:
    path = manifest_path.expanduser().resolve()
    try:
        payload = json.loads(bounded_text(path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest top level must be a JSON object")

    schema = payload.get("schema")
    target = region_fits.expanduser().resolve()
    if schema == _WORKFLOW_SCHEMA:
        matches = _workflow_matches(payload, target=target)
        description = "workflow/v1 manifest"
    elif schema == _BATCH_SCHEMA:
        matches = _batch_matches(payload, manifest_root=path.parent, target=target)
        description = "batch/v4 manifest"
    else:
        raise ValueError(f"unsupported or missing manifest schema: {schema!r}")

    if len(matches) != 1:
        raise ValueError(
            f"{description} must contain exactly one product for {target}; "
            f"found {len(matches)}"
        )

    product = matches[0]
    celestial = _manifest_sha256(
        product.get("celestial_geometry_sha256"),
        description="celestial geometry",
    )
    projection = _manifest_sha256(
        product.get("projection_identity_sha256"),
        description="projection identity",
    )
    evidence = (
        _projection_evidence_path(
            product,
            schema=schema,
            manifest_root=path.parent,
        )
        if require_projection_evidence
        else None
    )
    return _ManifestSelection(celestial, projection, evidence)


def _manifest_expectations(
    manifest_path: Path,
    region_fits: Path,
) -> tuple[str, str]:
    """Return only semantic expectations without requiring integrity sidecar lookup."""
    selection = _manifest_selection(
        manifest_path,
        region_fits,
        require_projection_evidence=False,
    )
    return (
        selection.celestial_geometry_sha256,
        selection.projection_identity_sha256,
    )


def _safe_optional_text(value: str | None) -> str | None:
    return safe_terminal_text(value) if value is not None else None


def _terminal_report(report) -> str:
    """Sanitize report-owned dynamic fields before preserving trusted line layout."""
    binding = replace(
        report.binding,
        path=Path(safe_terminal_text(report.binding.path)),
        esky2det_version=_safe_optional_text(report.binding.esky2det_version),
        sas_version=_safe_optional_text(report.binding.sas_version),
    )
    event_identity = replace(
        report.event_identity,
        path=Path(safe_terminal_text(report.event_identity.path)),
    )
    rendered = replace(
        report,
        binding=binding,
        event_identity=event_identity,
        mismatches=tuple(safe_terminal_text(item) for item in report.mismatches),
    )
    return rendered.format_text()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    expected_celestial = None
    expected_projection = None
    manifest_selection: _ManifestSelection | None = None
    if args.manifest is not None:
        try:
            manifest_selection = _manifest_selection(
                args.manifest,
                args.region_fits,
            )
            expected_celestial = manifest_selection.celestial_geometry_sha256
            expected_projection = manifest_selection.projection_identity_sha256
        except ValueError as exc:
            print(safe_terminal_text(f"xmm-region-check: {exc}"), file=sys.stderr)
            return 2

        assert manifest_selection.projection_evidence is not None
        try:
            load_bound_detector_geometry(
                args.region_fits,
                projection_evidence=manifest_selection.projection_evidence,
            )
        except ArtifactMaterializationError as exc:
            print(
                safe_terminal_text(
                    f"xmm-region-check: projection-evidence integrity check failed: {exc}"
                ),
                file=sys.stderr,
            )
            return 1

    try:
        report = check_region_binding(
            args.region_fits,
            args.event_file,
            source_region=args.source_region,
            expected_celestial_geometry_sha256=expected_celestial,
            expected_projection_identity_sha256=expected_projection,
        )
    except (RegionBindingError, ValueError, OSError) as exc:
        print(
            safe_terminal_text(f"xmm-region-check: validation could not be completed: {exc}"),
            file=sys.stderr,
        )
        return 2

    if manifest_selection is None:
        print("integrity: header/event binding only; detector geometry bytes NOT verified")
    else:
        print("integrity: projection evidence and current detector geometry bytes verified")
    print(_terminal_report(report))
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Package resource ceilings; these are safety policy, not scientific scales."""

from numbers import Integral
from pathlib import Path
import re

from .json_policy import loads_strict

MAX_REGION_ROWS = 16384
MAX_REGION_VERTICES = 16384
MAX_REGION_COORDINATES = 4194304
MAX_COMPONENTS = 65536
MAX_SOURCE_BYTES = 4194304
MAX_SOURCE_CELLS = 4096
MAX_SOURCE_VERTICES = 4096
MAX_GEODESIC_PAIR_CHECKS = 32640  # 256 authored edges; reject before quadratic work.
MAX_DETECTOR_TOPOLOGY_PAIR_CHECKS = 8386560  # choose(4096, 2); bound quadratic work.
MAX_SAMPLES = 65536
MAX_INLINE_LIMIT = 1048576
MAX_DEPTH = 32


def integer_limit(value: object, name: str, minimum: int, maximum: int) -> int:
    """Normalize integral scalars, rejecting Boolean and unbounded values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} (resource limit)")
    return result


def bounded_text(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("JSON/source byte resource limit exceeded")
    text = data.decode("utf-8")
    # All durable JSON authority readers already pass through bounded_text.
    # Validate the original text strictly before their ordinary json.loads call
    # so duplicate keys and non-standard NaN/Infinity constants cannot be
    # normalized away before semantic/cross-record validation sees them.
    if path.suffix.lower() == ".json":
        loads_strict(text)
    return text


def detector_region_table_budget(regions) -> int:
    """Validate in-memory detector REGION allocation limits before expensive work.

    Returns the maximum polygon vertex count, matching the fixed-width FITS REGION
    representation used by the serializer. The check is deliberately independent
    of topology so managed callers cannot force expensive pairwise validation
    before ordinary table/coordinate limits are applied.
    """
    rows = 0
    width = 0
    for region in regions:
        rows += len(region.boundaries)
        if rows > MAX_REGION_ROWS:
            raise ValueError(f"REGION row resource limit is {MAX_REGION_ROWS}")
        for boundary in region.boundaries:
            width = max(width, len(boundary.vertices))
            if width > MAX_REGION_VERTICES:
                raise ValueError(f"REGION vertex resource limit is {MAX_REGION_VERTICES}")
        if width and rows > MAX_REGION_COORDINATES // (2 * (width + 1)):
            raise ValueError(
                f"REGION aggregate coordinate resource limit is {MAX_REGION_COORDINATES}"
            )
    return width


def detector_topology_work_budget(regions) -> int:
    """Bound the worst pairwise edge work before detector topology traversal.

    The reusable validator performs self-intersection checks within boundaries
    and intersection/containment checks between same-component boundaries. This
    conservative estimate counts every unordered edge pair within each top-level
    component; it therefore bounds the dominant quadratic traversal before that
    traversal starts. Separate top-level Boolean components are intentionally not
    compared by the topology predicate and are budgeted independently.
    """
    total_pairs = 0
    for region in regions:
        edges = sum(len(boundary.vertices) for boundary in region.boundaries)
        total_pairs += edges * (edges - 1) // 2
        if total_pairs > MAX_DETECTOR_TOPOLOGY_PAIR_CHECKS:
            raise ValueError(
                "detector topology pair-check resource limit exceeded "
                f"({MAX_DETECTOR_TOPOLOGY_PAIR_CHECKS})"
            )
    return total_pairs


def validate_region_table_budget(table) -> None:
    """Inspect FITS metadata before reading vectors or heap data."""
    rows = int(table.header.get("NAXIS2", 0))
    if rows < 1 or rows > MAX_REGION_ROWS:
        raise ValueError("REGION row resource limit exceeded or empty table")
    names = {name.upper(): name for name in table.columns.names}
    widths = []
    for axis in ("X", "Y"):
        if axis not in names:
            raise ValueError(f"REGION missing {axis} column")
        match = re.fullmatch(r"(\d*)([DE])", str(table.columns[names[axis]].format))
        if match is None:
            raise ValueError("REGION vectors must use bounded fixed-width floating columns")
        width = int(match.group(1) or 1)
        if width < 3 or width > MAX_REGION_VERTICES + 1:
            raise ValueError("REGION vector resource limit exceeded")
        widths.append(width)
    if widths[0] != widths[1] or rows > MAX_REGION_COORDINATES // sum(widths):
        raise ValueError("REGION aggregate coordinate resource limit exceeded or unequal vectors")

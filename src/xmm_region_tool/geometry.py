"""Public geometry API with DS9 source handling kept behind one adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _geometry_core as _core
from . import ds9_adapter as _ds9_adapter
from .model import CelestialSelection
from .provenance import file_sha256

UnsupportedRegionError = _core.UnsupportedRegionError
SkyBoundary = _core.SkyBoundary
ParsedSkyRegion = _core.ParsedSkyRegion
selection_from_sky_regions = _core.selection_from_sky_regions
load_ds9_selection = _ds9_adapter.load_ds9_selection
load_ds9_sky_regions = _ds9_adapter.load_ds9_sky_regions
read_ds9_sky_regions = _ds9_adapter.read_ds9_sky_regions


def load_ds9_selection_snapshot(
    path: str | Path,
    *,
    samples: int = 128,
    external_identity: str | None = None,
    external_provenance: Mapping[str, Any] | None = None,
) -> tuple[CelestialSelection, str]:
    """Parse one stable DS9 file generation and return its exact source SHA-256.

    The parser still owns the source-format byte/resource checks. Hashing before
    and after parsing makes the accepted source digest and celestial selection a
    coherent snapshot for ordinary persistent file replacement/edit races.
    """
    source_path = Path(path).expanduser().resolve()
    sha_before = file_sha256(source_path)
    selection = _ds9_adapter.load_ds9_selection(
        source_path,
        samples=samples,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )
    sha_after = file_sha256(source_path)
    if sha_after != sha_before:
        raise UnsupportedRegionError(
            "DS9 source file changed while it was being parsed; refusing mixed-source provenance"
        )
    return selection, sha_after


def __getattr__(name: str) -> Any:
    """Delegate established private helpers to the semantic geometry core."""
    return getattr(_core, name)

"""Authoritative package-version access for runtime provenance."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "xmm-region-tool"
_SOURCE_TREE_FALLBACK = "0+unknown"


def distribution_version() -> str:
    """Return installed distribution version without depending on repository metadata."""
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _SOURCE_TREE_FALLBACK

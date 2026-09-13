"""Non-semantic converter/runtime producer evidence."""

from __future__ import annotations

import hashlib
import platform
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import astropy
import numpy as np

from .version import distribution_version


def _dependency_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def package_source_sha256() -> str:
    """Hash installed Python package sources without relying on a .git directory."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def converter_runtime_record() -> dict[str, object]:
    """Return stable available identifiers for the Python converter runtime."""
    return {
        "schema": "xmm-region-tool.converter-runtime/v1",
        "package": {
            "distribution": "xmm-region-tool",
            "version": distribution_version(),
            "source_sha256": package_source_sha256(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": {
            "astropy": astropy.__version__,
            "numpy": np.__version__,
            "regions": _dependency_version("regions"),
        },
    }

"""Immutable snapshots of non-semantic, JSON-compatible caller metadata."""

from collections.abc import Mapping
from types import MappingProxyType
import math


def freeze_json(value: object, *, depth: int = 0) -> object:
    if depth > 64:
        raise ValueError("external provenance nesting exceeds 64 levels")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("external provenance keys must be strings")
        return MappingProxyType({key: freeze_json(item, depth=depth + 1)
                                 for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item, depth=depth + 1) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("external provenance must contain finite JSON-compatible values")


def plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_json(item) for item in value]
    return value

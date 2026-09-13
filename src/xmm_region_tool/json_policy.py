"""Strict JSON profile for durable provenance and manifest authority records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


class StrictJsonError(ValueError):
    """Raised when durable JSON is ambiguous or outside standard finite JSON."""


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StrictJsonError(f"non-finite/non-standard JSON numeric constant {value!r}")


def loads_strict(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity constants."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except StrictJsonError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrictJsonError("invalid durable JSON") from exc


def dumps_strict(value: object, **kwargs: Any) -> str:
    """Serialize standards-compliant finite JSON only."""
    options = dict(kwargs)
    options["allow_nan"] = False
    try:
        return json.dumps(value, **options)
    except (TypeError, ValueError) as exc:
        raise StrictJsonError("durable evidence is not finite JSON-compatible data") from exc

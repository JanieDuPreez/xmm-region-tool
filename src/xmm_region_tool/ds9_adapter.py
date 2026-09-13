"""Loss-intolerant DS9 source-to-semantic-geometry adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from regions import SkyRegion

from . import _geometry_core as _core
from . import ds9_source as _source
from .limits import MAX_SOURCE_CELLS, MAX_SAMPLES, integer_limit
from .model import CelestialRegion, CelestialSelection

UnsupportedRegionError = _core.UnsupportedRegionError
_FRAME_ALIASES = _core._FRAME_ALIASES
_PIXEL_FRAMES = _core._PIXEL_FRAMES
_DS9_INT_RE = re.compile(r"[+-]?[0-9]+\Z")
_DS9_FLOAT_EPSILON_RAD = float(np.finfo(np.float32).eps)


def _translate_source_error(exc: Exception) -> UnsupportedRegionError:
    return UnsupportedRegionError(str(exc))


def _source_statements(text: str) -> list[_source.Ds9Statement]:
    try:
        return _source.split_ds9_statements(text, max_statements=MAX_SOURCE_CELLS)
    except _source.UnsupportedRegionError as exc:
        raise _translate_source_error(exc) from exc


def _validate_include_metadata(statements: list[_source.Ds9Statement]) -> None:
    try:
        for statement in statements:
            _source.parse_include_property(statement.metadata, context="DS9 region")
            lowered = statement.code.lstrip().lower()
            if lowered.startswith("global "):
                _source.parse_include_property(
                    statement.code.lstrip()[len("global ") :],
                    context="DS9 global properties",
                )
    except _source.UnsupportedRegionError as exc:
        raise _translate_source_error(exc) from exc


def _parse_custom_shape(code: str) -> _source.CustomShapeStatement | None:
    try:
        return _source.parse_custom_shape(code)
    except _source.UnsupportedRegionError as exc:
        raise _translate_source_error(exc) from exc


def _is_extension(shape: _source.CustomShapeStatement | None) -> bool:
    if shape is None:
        return False
    if shape.keyword in {"panda", "epanda", "bpanda", "annulus"}:
        return True
    if shape.keyword == "box":
        return len(shape.arguments) > 5 or any(
            token.lower().startswith("n=") for token in shape.arguments
        )
    if shape.keyword == "ellipse":
        return len(shape.arguments) > 5 or any(
            token.lower().startswith("n=") for token in shape.arguments
        )
    return False


def _finite_numeric_token(
    token: str,
    *,
    context: str,
    allowed_suffixes: tuple[str, ...],
    rejected_word_suffixes: tuple[str, ...] = (),
) -> None:
    value = token.strip().lower()
    if any(value.endswith(suffix) for suffix in rejected_word_suffixes):
        raise UnsupportedRegionError(f"invalid {context} token {token!r}")
    numeric_text = value
    for suffix in sorted(allowed_suffixes, key=len, reverse=True):
        if value.endswith(suffix):
            numeric_text = value[: -len(suffix)]
            break
    try:
        numeric = float(numeric_text)
    except ValueError as exc:
        raise UnsupportedRegionError(f"invalid {context} token {token!r}") from exc
    if not np.isfinite(numeric):
        raise UnsupportedRegionError(f"invalid {context} token {token!r}")


def _validate_angle_token(token: str, *, context: str) -> None:
    _finite_numeric_token(
        token,
        context=f"{context} angle",
        allowed_suffixes=("d", "r"),
        rejected_word_suffixes=("deg", "rad"),
    )


def _validate_radius_token(token: str, *, context: str) -> None:
    lowered = token.strip().lower()
    if lowered.endswith(("deg", "rad", "r")):
        raise UnsupportedRegionError(f"invalid {context} radius token {token!r}")
    _finite_numeric_token(
        token,
        context=f"{context} radius",
        allowed_suffixes=('"', "'", "d"),
    )


def _validate_positive_integer_token(token: str, *, name: str, context: str) -> None:
    stripped = token.strip()
    if _DS9_INT_RE.fullmatch(stripped) is None:
        raise UnsupportedRegionError(f"{context} {name} must use a DS9 integer token")
    try:
        value = int(stripped)
        integer_limit(value, name, 1, MAX_SOURCE_CELLS)
    except ValueError as exc:
        raise UnsupportedRegionError(f"{context} {name} must be a positive integer") from exc


def _radius_deg(token: str, *, context: str) -> float:
    _validate_radius_token(token, context=context)
    value = float(_core._parse_radius_token(token, context=context).to_value(u.deg))
    if not np.isfinite(value):
        raise UnsupportedRegionError(f"{context} radii must be finite")
    return value


def _canonical_ds9_sector_arguments(
    arguments: tuple[str, ...],
    *,
    context: str,
) -> tuple[str, ...]:
    """Apply DS9's normalize-then-fuzzy angular classification at text input.

    Stable DS9 8.7 normalizes celestial angle tokens to one turn before
    ``BaseMarker::setAngles`` compares them with ``FLT_EPSILON`` in radians.
    Rotation/reflection into the active WCS preserves the circular separation,
    so the standalone adapter can classify the full-circle boundary without
    importing image-specific WCS state. The resulting endpoints are written as
    one canonical positive sweep for the established semantic builders.
    """
    start = _core._parse_angle_token(arguments[2], context=context).to_value(u.rad)
    stop = _core._parse_angle_token(arguments[3], context=context).to_value(u.rad)
    turn = 2.0 * np.pi
    start_rad = float(np.mod(start, turn))
    stop_rad = float(np.mod(stop, turn))
    forward = float(np.mod(stop_rad - start_rad, turn))
    circular_gap = min(forward, turn - forward)

    canonical = list(arguments)
    if circular_gap <= _DS9_FLOAT_EPSILON_RAD:
        # A full circle has no meaningful start direction. Use an exact pair so
        # downstream floating subtraction cannot turn it back into a 360-epsilon
        # sector, and equivalent DS9 spellings receive one semantic hash.
        canonical[2] = "0.0"
        canonical[3] = "360.0"
        return tuple(canonical)

    canonical[2] = repr(float(np.rad2deg(start_rad)))
    canonical[3] = repr(float(np.rad2deg(start_rad + forward)))
    return tuple(canonical)


def _sort_explicit_ellipse_arguments(arguments: tuple[str, ...]) -> tuple[str, ...]:
    x, y = arguments[:2]
    angle = arguments[-1]
    radial = arguments[2:-1]
    pairs = [(radial[index], radial[index + 1]) for index in range(0, len(radial), 2)]
    pairs.sort(key=lambda pair: _radius_deg(pair[0], context="DS9 ellipse annulus"))
    flattened = tuple(token for pair in pairs for token in pair)
    return (x, y, *flattened, angle)


def _normalised_custom_arguments(shape: _source.CustomShapeStatement) -> str:
    arguments = shape.arguments
    keyword = shape.keyword

    if keyword == "panda":
        if len(arguments) != 8:
            raise UnsupportedRegionError(
                "DS9 panda requires exactly 8 arguments: "
                "x,y,startangle,stopangle,nangle,inner,outer,nradius"
            )
        _validate_angle_token(arguments[2], context="DS9 panda")
        _validate_angle_token(arguments[3], context="DS9 panda")
        _validate_positive_integer_token(arguments[4], name="nangle", context="DS9 panda")
        _validate_radius_token(arguments[5], context="DS9 panda")
        _validate_radius_token(arguments[6], context="DS9 panda")
        _validate_positive_integer_token(arguments[7], name="nradius", context="DS9 panda")
        arguments = _canonical_ds9_sector_arguments(arguments, context="DS9 panda")

    elif keyword == "epanda":
        if len(arguments) not in {10, 11}:
            raise UnsupportedRegionError(
                "DS9 epanda requires x,y,startangle,stopangle,nangle,"
                "innerMajor,innerMinor,outerMajor,outerMinor,nradius[,angle]"
            )
        _validate_angle_token(arguments[2], context="DS9 epanda")
        _validate_angle_token(arguments[3], context="DS9 epanda")
        _validate_positive_integer_token(arguments[4], name="nangle", context="DS9 epanda")
        for token in arguments[5:9]:
            _validate_radius_token(token, context="DS9 epanda")
        _validate_positive_integer_token(arguments[9], name="nradius", context="DS9 epanda")
        if len(arguments) == 11:
            _validate_angle_token(arguments[10], context="DS9 epanda")
        arguments = _canonical_ds9_sector_arguments(arguments, context="DS9 epanda")

    elif keyword == "annulus":
        n_positions = [
            index for index, token in enumerate(arguments) if token.lower().startswith("n=")
        ]
        if n_positions:
            if len(n_positions) != 1 or n_positions[0] != 4 or len(arguments) != 5:
                raise UnsupportedRegionError(
                    "DS9 annulus n= syntax requires x,y,inner,outer,n=N"
                )
            _validate_radius_token(arguments[2], context="DS9 annulus")
            _validate_radius_token(arguments[3], context="DS9 annulus")
            _validate_positive_integer_token(
                arguments[4].split("=", maxsplit=1)[1],
                name="n",
                context="DS9 annulus",
            )
        else:
            if len(arguments) < 4:
                raise UnsupportedRegionError(
                    "DS9 annulus requires x,y and at least two radial boundaries"
                )
            for token in arguments[2:]:
                _validate_radius_token(token, context="DS9 annulus")

    elif keyword == "ellipse":
        n_positions = [
            index for index, token in enumerate(arguments) if token.lower().startswith("n=")
        ]
        if n_positions:
            if len(n_positions) != 1 or n_positions[0] != 6 or len(arguments) not in {7, 8}:
                raise UnsupportedRegionError(
                    "DS9 ellipse annulus n= syntax requires "
                    "x,y,r11,r12,r21,r22,n=N[,angle]"
                )
            for token in arguments[2:6]:
                _validate_radius_token(token, context="DS9 ellipse annulus")
            _validate_positive_integer_token(
                arguments[6].split("=", maxsplit=1)[1],
                name="n",
                context="DS9 ellipse annulus",
            )
            if len(arguments) == 8:
                _validate_angle_token(arguments[7], context="DS9 ellipse annulus")
        else:
            if len(arguments) < 7 or len(arguments) % 2 == 0:
                raise UnsupportedRegionError(
                    "DS9 multi-ring ellipse requires two or more semiaxis pairs "
                    "followed by an angle before validating the same major/minor axis ratio"
                )
            for token in arguments[2:-1]:
                _validate_radius_token(token, context="DS9 ellipse annulus")
            _validate_angle_token(arguments[-1], context="DS9 ellipse annulus")
            arguments = _sort_explicit_ellipse_arguments(arguments)

    return ",".join(arguments)


def _effective_include(
    statement: _source.Ds9Statement,
    *,
    sign: str,
    inherited: bool,
    context: str,
) -> bool:
    try:
        return _source.effective_include(
            sign=sign,
            metadata=statement.metadata,
            inherited=inherited,
            context=context,
        )
    except _source.UnsupportedRegionError as exc:
        raise _translate_source_error(exc) from exc


def _ordinary_regions_at_statement(
    statement: _source.Ds9Statement,
    *,
    current_frame: str | None,
    inherited_include: bool,
    samples: int,
) -> list[CelestialRegion]:
    parts: list[str] = []
    if current_frame is not None:
        parts.append(current_frame)
    parts.append(f"global include={int(inherited_include)}")
    parts.append(statement.rendered)
    source_regions = _core._parse_ds9_sky_text("\n".join(parts) + "\n")
    if not source_regions:
        raise UnsupportedRegionError(
            "the DS9 parser produced no geometry for an authored region statement; "
            "refusing a partial conversion"
        )
    include = _effective_include(
        statement,
        sign=_source.leading_region_sign(statement.code),
        inherited=inherited_include,
        context="DS9 region",
    )
    return [
        CelestialRegion(
            include=include,
            boundaries=_core._boundaries(region, samples, ds9_rotation_angles=True),
        )
        for region in source_regions
    ]


def _explicit_annulus_regions(
    *,
    frame: str,
    include: bool,
    shape: _source.CustomShapeStatement,
    samples: int,
) -> list[CelestialRegion]:
    x, y, *radius_tokens = shape.arguments
    center = _core._parse_ds9_center(frame, x, y, context="DS9 annulus")
    radii = [
        (_radius_deg(token, context="DS9 annulus"), token)
        for token in radius_tokens
    ]
    radii.sort(key=lambda item: item[0])
    values = [item[0] for item in radii]
    if values[0] < 0:
        raise UnsupportedRegionError("DS9 annulus radii must be non-negative")
    if any(left == right for left, right in zip(values[:-1], values[1:], strict=True)):
        raise UnsupportedRegionError(
            "DS9 annulus has duplicate radial boundaries and would create a zero-area cell"
        )

    cells: list[CelestialRegion] = []
    for inner_value, outer_value in zip(values[:-1], values[1:], strict=True):
        cell = CelestialSelection.sector_annulus(
            center,
            inner_value * u.deg,
            outer_value * u.deg,
            0 * u.deg,
            360 * u.deg,
            include=include,
        )
        cells.append(_core._with_sampling_hint(cell.regions[0], samples))
    return cells


def _annulus_regions(
    *,
    frame: str,
    include: bool,
    shape: _source.CustomShapeStatement,
    samples: int,
) -> list[CelestialRegion]:
    arguments = _normalised_custom_arguments(shape)
    if any(token.lower().startswith("n=") for token in shape.arguments):
        return _core._annulus_n_regions(
            frame=frame,
            include=include,
            arguments=arguments,
            samples=samples,
        )
    return _explicit_annulus_regions(
        frame=frame,
        include=include,
        shape=shape,
        samples=samples,
    )


def _box_pair_deg(width_token: str, height_token: str) -> tuple[float, float]:
    width = _radius_deg(width_token, context="DS9 box annulus")
    height = _radius_deg(height_token, context="DS9 box annulus")
    if width < 0 or height < 0:
        raise UnsupportedRegionError("DS9 box-annulus widths and heights must be non-negative")
    width_zero = width == 0.0
    height_zero = height == 0.0
    if width_zero != height_zero:
        raise UnsupportedRegionError(
            "DS9 box-annulus boundary dimensions must both be positive or both exactly zero"
        )
    return width, height


def _canonical_box_edges(
    edges: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(edges) < 2:
        raise UnsupportedRegionError("DS9 box annulus requires at least two box boundaries")
    ordered = sorted(edges, key=lambda pair: pair[0])
    widths = [pair[0] for pair in ordered]
    if any(left == right for left, right in zip(widths[:-1], widths[1:], strict=True)):
        raise UnsupportedRegionError(
            "DS9 box annulus has duplicate first-axis boundaries and would create a zero-area cell"
        )

    ratios: list[float] = []
    for index, (width, height) in enumerate(ordered):
        if width == 0.0:
            if index != 0 or height != 0.0:
                raise UnsupportedRegionError(
                    "only the innermost DS9 box-annulus boundary may be exactly (0,0)"
                )
            continue
        if height <= 0.0:
            raise UnsupportedRegionError("DS9 box-annulus widths and heights must be positive")
        ratios.append(height / width)

    if len(ratios) > 1 and not np.all(
        np.isclose(ratios, ratios[0], rtol=1e-12, atol=1e-15)
    ):
        raise UnsupportedRegionError(
            "DS9 nested box-annulus boundaries must have one common width/height aspect ratio"
        )
    return ordered


def _box_edges_and_angle(
    shape: _source.CustomShapeStatement,
) -> tuple[list[tuple[float, float]], u.Quantity]:
    arguments = shape.arguments
    n_positions = [
        index for index, token in enumerate(arguments) if token.lower().startswith("n=")
    ]
    source_angle = 0 * u.deg

    if n_positions:
        if len(n_positions) != 1 or n_positions[0] != 6 or len(arguments) not in {7, 8}:
            raise UnsupportedRegionError(
                "DS9 box annulus n= syntax requires x,y,w1,h1,w2,h2,n=N[,angle]"
            )
        inner = _box_pair_deg(arguments[2], arguments[3])
        outer = _box_pair_deg(arguments[4], arguments[5])
        _validate_positive_integer_token(
            arguments[6].split("=", maxsplit=1)[1],
            name="n",
            context="DS9 box annulus",
        )
        n = int(arguments[6].split("=", maxsplit=1)[1])
        if len(arguments) == 8:
            _validate_angle_token(arguments[7], context="DS9 box annulus")
            source_angle = _core._parse_angle_token(
                arguments[7], context="DS9 box annulus"
            )
        widths = np.linspace(inner[0], outer[0], n + 1)
        heights = np.linspace(inner[1], outer[1], n + 1)
        edges = _canonical_box_edges(list(zip(widths, heights, strict=True)))
    else:
        radial_tokens = list(arguments[2:])
        if len(radial_tokens) % 2 == 1:
            angle_token = radial_tokens.pop()
            _validate_angle_token(angle_token, context="DS9 box annulus")
            source_angle = _core._parse_angle_token(
                angle_token, context="DS9 box annulus"
            )
        if len(radial_tokens) < 4 or len(radial_tokens) % 2:
            raise UnsupportedRegionError(
                "DS9 box annulus requires two or more width/height boundary pairs plus optional angle"
            )
        edges = _canonical_box_edges(
            [
                _box_pair_deg(radial_tokens[index], radial_tokens[index + 1])
                for index in range(0, len(radial_tokens), 2)
            ]
        )

    return edges, _core._ds9_rotation_to_local(source_angle)


def _box_annulus_regions(
    *,
    frame: str,
    include: bool,
    shape: _source.CustomShapeStatement,
    samples: int,
) -> list[CelestialRegion]:
    x, y = shape.arguments[:2]
    center = _core._parse_ds9_center(frame, x, y, context="DS9 box annulus")
    edges, angle = _box_edges_and_angle(shape)
    integer_limit(len(edges) - 1, "DS9 box-annulus cells", 1, MAX_SOURCE_CELLS)

    cells: list[CelestialRegion] = []
    for inner, outer in zip(edges[:-1], edges[1:], strict=True):
        boundaries = [
            _core._polygon_boundary(
                _core._rectangle_vertices(
                    center,
                    outer[0] * u.deg,
                    outer[1] * u.deg,
                    angle,
                ),
                samples=samples,
            )
        ]
        if inner[0] != 0.0:
            boundaries.append(
                _core._polygon_boundary(
                    _core._rectangle_vertices(
                        center,
                        inner[0] * u.deg,
                        inner[1] * u.deg,
                        angle,
                    ),
                    subtract=True,
                    samples=samples,
                )
            )
        cells.append(CelestialRegion(include=include, boundaries=tuple(boundaries)))
    return cells


def _dispatch_custom(
    statement: _source.Ds9Statement,
    shape: _source.CustomShapeStatement,
    *,
    frame: str | None,
    inherited_include: bool,
    samples: int,
) -> list[CelestialRegion]:
    if shape.keyword == "bpanda":
        raise UnsupportedRegionError(
            "DS9 bpanda is not supported; use panda/epanda or supply supported geometry"
        )
    context = {
        "panda": "DS9 panda",
        "epanda": "DS9 epanda",
        "annulus": "DS9 annulus",
        "box": "DS9 box annulus",
        "ellipse": "DS9 ellipse annulus",
    }[shape.keyword]
    valid_frame = _core._validate_extension_frame(frame, context=context)
    include = _effective_include(
        statement,
        sign=shape.sign,
        inherited=inherited_include,
        context=context,
    )
    if shape.keyword == "annulus":
        return _annulus_regions(
            frame=valid_frame,
            include=include,
            shape=shape,
            samples=samples,
        )
    if shape.keyword == "box":
        return _box_annulus_regions(
            frame=valid_frame,
            include=include,
            shape=shape,
            samples=samples,
        )
    arguments = _normalised_custom_arguments(shape)
    if shape.keyword == "panda":
        return _core._panda_regions(
            frame=valid_frame,
            include=include,
            arguments=arguments,
            samples=samples,
        )
    if shape.keyword == "epanda":
        return _core._epanda_regions(
            frame=valid_frame,
            include=include,
            arguments=arguments,
            samples=samples,
        )
    return _core._ellipse_annulus_regions(
        frame=valid_frame,
        include=include,
        arguments=arguments,
        samples=samples,
    )


def _load_ds9_with_extensions(
    text: str,
    *,
    samples: int,
    external_identity: str | None,
    external_provenance: Mapping[str, Any] | None,
) -> CelestialSelection:
    statements = _source_statements(text)
    _validate_include_metadata(statements)
    parsed_shapes = [_parse_custom_shape(statement.code) for statement in statements]
    has_extension = any(_is_extension(shape) for shape in parsed_shapes)
    if not has_extension:
        return _core._selection_from_sky_regions(
            _core._parse_ds9_sky_text(text),
            samples=samples,
            external_identity=external_identity,
            external_provenance=external_provenance,
            ds9_rotation_angles=True,
        )

    current_frame: str | None = None
    inherited_include = True
    regions: list[CelestialRegion] = []
    for statement, shape in zip(statements, parsed_shapes, strict=True):
        lowered = statement.code.strip().lower()
        if lowered in _FRAME_ALIASES:
            current_frame = _FRAME_ALIASES[lowered]
            continue
        if lowered in _PIXEL_FRAMES:
            current_frame = lowered
            continue
        if lowered.startswith("global "):
            try:
                global_include = _source.parse_include_property(
                    statement.code.strip()[len("global ") :],
                    context="DS9 global properties",
                )
            except _source.UnsupportedRegionError as exc:
                raise _translate_source_error(exc) from exc
            if global_include is not None:
                inherited_include = global_include
            continue
        if _is_extension(shape):
            if shape is None:  # pragma: no cover
                raise RuntimeError("internal DS9 extension dispatch error")
            regions.extend(
                _dispatch_custom(
                    statement,
                    shape,
                    frame=current_frame,
                    inherited_include=inherited_include,
                    samples=samples,
                )
            )
        else:
            regions.extend(
                _ordinary_regions_at_statement(
                    statement,
                    current_frame=current_frame,
                    inherited_include=inherited_include,
                    samples=samples,
                )
            )
        if len(regions) > MAX_SOURCE_CELLS:
            raise UnsupportedRegionError("DS9 expanded cell resource limit exceeded")

    if not regions:
        raise ValueError("DS9 region file contains no regions")
    return CelestialSelection(
        regions=tuple(regions),
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def read_ds9_sky_regions(path: str | Path) -> list[SkyRegion]:
    text = _core._read_source_text(Path(path))
    statements = _source_statements(text)
    _validate_include_metadata(statements)
    result = _core._parse_ds9_sky_text(text)
    if not result:
        raise ValueError("DS9 region file contains no regions")
    return result


def load_ds9_selection(
    path: str | Path,
    *,
    samples: int = 128,
    external_identity: str | None = None,
    external_provenance: Mapping[str, Any] | None = None,
) -> CelestialSelection:
    samples = integer_limit(samples, "samples", 16, MAX_SAMPLES)
    text = _core._read_source_text(Path(path))
    return _load_ds9_with_extensions(
        text,
        samples=samples,
        external_identity=external_identity,
        external_provenance=external_provenance,
    )


def load_ds9_sky_regions(path: str | Path, *, samples: int = 128) -> list[CelestialRegion]:
    return list(load_ds9_selection(path, samples=samples).regions)

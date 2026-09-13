"""Loss-intolerant lexical helpers for the supported DS9 source surface."""

from __future__ import annotations

import re
from dataclasses import dataclass


class UnsupportedRegionError(ValueError):
    """Raised when DS9 source cannot be represented without semantic loss."""


@dataclass(frozen=True)
class Ds9Statement:
    """One authored DS9 statement with any trailing property text preserved."""

    code: str
    metadata: str | None = None

    @property
    def rendered(self) -> str:
        if self.metadata:
            return f"{self.code} # {self.metadata}"
        return self.code


@dataclass(frozen=True)
class CustomShapeStatement:
    """Lexed statement for a DS9 shape handled or validated by this adapter."""

    sign: str
    keyword: str
    arguments: tuple[str, ...]


_SHAPE_HEAD_RE = re.compile(r"^\s*([+-]?)\s*([A-Za-z]+)\b(.*)$")
_INCLUDE_ASSIGNMENT_START_RE = re.compile(r"\binclude\s*=", re.IGNORECASE)
_INCLUDE_ASSIGNMENT_RE = re.compile(r"\binclude\s*=\s*([^\s]+)", re.IGNORECASE)
_COMPOSITE_DECLARATION_RE = re.compile(r"^\s*#\s*composite\s*\(", re.IGNORECASE | re.MULTILINE)
_ANNULUS_KEYWORDS = frozenset({"ann", "annu", "annul", "annulu", "annulus"})
_ELLIPSE_KEYWORDS = frozenset({"ell", "elli", "ellip", "ellips", "ellipse"})


def _looks_like_unit_quote(text: str, index: int) -> bool:
    if index == 0:
        return False
    previous = text[index - 1]
    return previous.isdigit() or previous == "."


def _find_comment_start(line: str) -> int | None:
    paren_depth = 0
    brace_depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'} and not _looks_like_unit_quote(line, index):
            quote = char
            continue
        if char == "{":
            brace_depth += 1
            continue
        if char == "}" and brace_depth:
            brace_depth -= 1
            continue
        if brace_depth:
            continue
        if char == "(":
            paren_depth += 1
            continue
        if char == ")" and paren_depth:
            paren_depth -= 1
            continue
        if char == "#" and paren_depth == 0:
            return index
    return None


def _split_top_level_semicolons(code: str) -> list[str]:
    parts: list[str] = []
    start = 0
    paren_depth = 0
    brace_depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(code):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'} and not _looks_like_unit_quote(code, index):
            quote = char
            continue
        if char == "{":
            brace_depth += 1
            continue
        if char == "}" and brace_depth:
            brace_depth -= 1
            continue
        if brace_depth:
            continue
        if char == "(":
            paren_depth += 1
            continue
        if char == ")" and paren_depth:
            paren_depth -= 1
            continue
        if char == ";" and paren_depth == 0:
            part = code[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    part = code[start:].strip()
    if part:
        parts.append(part)
    return parts


def _contains_top_level_composite_conjunction(code: str) -> bool:
    paren_depth = 0
    brace_depth = 0
    for char in code:
        if char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif not brace_depth and char == "(":
            paren_depth += 1
        elif not brace_depth and char == ")" and paren_depth:
            paren_depth -= 1
        elif not brace_depth and paren_depth == 0 and char == "|":
            return True
    return False


def split_ds9_statements(text: str, *, max_statements: int) -> list[Ds9Statement]:
    """Tokenize DS9 source without discarding science-bearing source structure."""
    if _COMPOSITE_DECLARATION_RE.search(text):
        raise UnsupportedRegionError(
            "DS9 composite grouping is not supported in this release; refusing to "
            "silently split one authored composite into separate extraction spectra"
        )

    statements: list[Ds9Statement] = []
    for line in text.splitlines():
        comment_start = _find_comment_start(line)
        if comment_start is None:
            code = line
            metadata = None
        else:
            code = line[:comment_start]
            metadata_text = line[comment_start + 1 :].strip()
            metadata = metadata_text or None
        if _contains_top_level_composite_conjunction(code):
            raise UnsupportedRegionError(
                "DS9 composite grouping is not supported in this release; refusing to "
                "silently split one authored composite into separate extraction spectra"
            )
        code_parts = _split_top_level_semicolons(code)
        for index, part in enumerate(code_parts):
            if part.strip().lower() == "ecliptic":
                raise UnsupportedRegionError(
                    "bare DS9 ecliptic regions are not supported: DS9 ties the ecliptic "
                    "equinox to the active image/WCS epoch, which a standalone region "
                    "file does not preserve"
                )
            statement_metadata = metadata if index == len(code_parts) - 1 else None
            statements.append(Ds9Statement(part, statement_metadata))
            if len(statements) > max_statements:
                raise UnsupportedRegionError("DS9 statement resource limit exceeded")
    return statements


def _mask_grouped_property_text(text: str) -> str:
    masked: list[str] = []
    brace_depth = 0
    quote: str | None = None
    escaped = False
    for char in text:
        if quote is not None:
            masked.append(" ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if brace_depth:
            masked.append(" ")
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
            continue
        if char == "{":
            brace_depth = 1
            masked.append(" ")
            continue
        if char in {"'", '"'}:
            quote = char
            masked.append(" ")
            continue
        masked.append(char)
    return "".join(masked)


def parse_include_property(text: str | None, *, context: str) -> bool | None:
    if not text:
        return None
    masked = _mask_grouped_property_text(text)
    starts = list(_INCLUDE_ASSIGNMENT_START_RE.finditer(masked))
    if not starts:
        return None
    matches = list(_INCLUDE_ASSIGNMENT_RE.finditer(masked))
    if len(matches) != len(starts):
        raise UnsupportedRegionError(f"{context} has malformed include metadata")
    result: bool | None = None
    for match in matches:
        value = match.group(1)
        if value not in {"0", "1"}:
            raise UnsupportedRegionError(
                f"{context} include metadata must use the supported DS9 spelling 0 or 1"
            )
        result = value == "1"
    return result


def effective_include(*, sign: str, metadata: str | None, inherited: bool, context: str) -> bool:
    if sign not in {"", "+", "-"}:
        raise UnsupportedRegionError(f"invalid {context} inclusion sign")
    include = inherited
    if sign == "+":
        include = True
    elif sign == "-":
        include = False
    local = parse_include_property(metadata, context=context)
    return include if local is None else local


def _canonical_custom_keyword(keyword: str) -> str | None:
    lowered = keyword.lower()
    if lowered == "circle":
        return "circle"
    if lowered == "box":
        return "box"
    if lowered in {"panda", "cpanda"}:
        return "panda"
    if lowered == "epanda":
        return "epanda"
    if lowered == "bpanda":
        return "bpanda"
    if lowered in _ANNULUS_KEYWORDS:
        return "annulus"
    if lowered in _ELLIPSE_KEYWORDS:
        return "ellipse"
    return None


def _unwrap_optional_parentheses(remainder: str, *, keyword: str) -> str:
    stripped = remainder.strip()
    if not stripped:
        return ""
    if not stripped.startswith("("):
        if "(" in stripped or ")" in stripped:
            raise UnsupportedRegionError(f"malformed DS9 {keyword} parentheses")
        return stripped
    depth = 0
    closing_index: int | None = None
    for index, char in enumerate(stripped):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise UnsupportedRegionError(f"malformed DS9 {keyword} parentheses")
            if depth == 0:
                closing_index = index
                break
    if closing_index is None or depth != 0:
        raise UnsupportedRegionError(f"malformed DS9 {keyword} parentheses")
    if stripped[closing_index + 1 :].strip():
        raise UnsupportedRegionError(f"unexpected syntax after DS9 {keyword} arguments")
    return stripped[1:closing_index]


def _split_custom_arguments(arguments: str, *, keyword: str) -> tuple[str, ...]:
    stripped = arguments.strip()
    if not stripped:
        return ()
    if stripped.startswith(",") or stripped.endswith(",") or re.search(r",\s*,", stripped):
        raise UnsupportedRegionError(f"malformed DS9 {keyword} argument separators")
    return tuple(token for token in stripped.replace(",", " ").split() if token)


def parse_custom_shape(code: str) -> CustomShapeStatement | None:
    match = _SHAPE_HEAD_RE.match(code)
    if match is None:
        return None
    sign, keyword_text, remainder = match.groups()
    keyword = _canonical_custom_keyword(keyword_text)
    if keyword is None:
        return None
    arguments_text = _unwrap_optional_parentheses(remainder, keyword=keyword_text)
    arguments = _split_custom_arguments(arguments_text, keyword=keyword_text)
    if keyword == "circle" and len(arguments) != 3:
        raise UnsupportedRegionError(
            "DS9 circle requires exactly x,y,r; surplus circle parameters are not "
            "part of the SAOImage DS9 grammar"
        )
    return CustomShapeStatement(sign=sign, keyword=keyword, arguments=arguments)


def leading_region_sign(code: str) -> str:
    stripped = code.lstrip()
    if stripped.startswith("+"):
        return "+"
    if stripped.startswith("-"):
        return "-"
    return ""

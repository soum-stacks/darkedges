"""Cause heuristics for dark edges (handoff Sections 8-9).

Precedence (first match wins): property > dunder > decorator > framework_lifted
> async > other.  'property' and 'dunder' are tested before the generic
'decorator' because @property is itself a decorator.
"""
from __future__ import annotations

import re

_HEADER = re.compile(r"^\s*(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_DUNDER = re.compile(r"^__[A-Za-z0-9_]+__$")
_PROP = re.compile(
    r"^@\s*(property|functools\.cached_property|cached_property|[\w\.]+\.(setter|getter|deleter))\b"
)


def parse_header(node_text: str):
    """Return (decorators, kind, name) from a node's source text.

    kind is 'async def', 'def', 'class' or None. Decorators are the lines that
    start with '@' before the header (only present if the graph text keeps them).
    """
    decorators: list[str] = []
    for line in (node_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("@"):
            decorators.append(s)
            continue
        m = _HEADER.match(line)
        if m:
            return decorators, re.sub(r"\s+", " ", m.group(1)), m.group(2)
        if s.startswith("#") or decorators:
            continue  # comment or continuation of a multi-line decorator
        break
    return decorators, None, None


def classify(node_text: str, kind: str, is_async: bool | None = None, name: str | None = None) -> str:
    decorators, hkind, hname = parse_header(node_text)
    name = hname or name or ""
    if is_async is None:
        is_async = hkind == "async def"
    if any(_PROP.match(d) for d in decorators):
        return "property"
    if _DUNDER.match(name):
        return "dunder"
    if decorators:
        return "decorator"
    if kind == "lifted":
        return "framework_lifted"
    if is_async:
        return "async"
    return "other"


def text_has_decorators(node_text: str) -> bool:
    return bool(parse_header(node_text)[0])


def is_class_text(node_text: str) -> bool:
    return parse_header(node_text)[1] == "class"


def is_dunder(name: str) -> bool:
    return bool(_DUNDER.match(name or ""))

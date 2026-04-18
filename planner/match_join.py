from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


_WHOLE_PLACEHOLDER_RE = re.compile(r"^\{(?P<name>[^{}]+)\}$")
_ARITH_PLACEHOLDER_RE = re.compile(
    r"^(?P<base>[A-Za-z0-9_]+)_(?P<op>plus|minus)_(?P<n>\d+)_(?P<unit>minutes?|hours?|days?)$"
)


def is_whole_placeholder(value: object) -> bool:
    return isinstance(value, str) and bool(_WHOLE_PLACEHOLDER_RE.match(value.strip()))


def placeholder_name(value: str) -> str:
    match = _WHOLE_PLACEHOLDER_RE.match(value.strip())
    return match.group("name").strip() if match else ""


def _parse_datetime(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _format_like_base(dt: datetime, *, base: str) -> str:
    base_text = str(base or "")
    if "T" in base_text:
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    if len(base_text.strip()) == 10:
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def resolve_placeholder_value(name: str, bindings: Mapping[str, object]) -> Optional[object]:
    key = str(name or "").strip()
    if not key:
        return None
    if key in bindings:
        return bindings[key]

    match = _ARITH_PLACEHOLDER_RE.match(key)
    if not match:
        return None

    base_key = match.group("base")
    base_value = bindings.get(base_key)
    if not isinstance(base_value, str):
        return None
    base_dt = _parse_datetime(base_value)
    if base_dt is None:
        return None

    op = match.group("op")
    n = int(match.group("n"))
    unit_raw = match.group("unit").lower()
    unit = unit_raw[:-1] if unit_raw.endswith("s") else unit_raw
    delta: timedelta
    if unit == "minute":
        delta = timedelta(minutes=n)
    elif unit == "hour":
        delta = timedelta(hours=n)
    elif unit == "day":
        delta = timedelta(days=n)
    else:
        return None

    dt = base_dt + delta if op == "plus" else base_dt - delta
    return _format_like_base(dt, base=str(base_value))


@dataclass(frozen=True)
class JoinResult:
    assignments: List[Dict[str, Any]]


def join_assignments(relations: Iterable[Iterable[Mapping[str, Any]]]) -> JoinResult:
    """
    Generic equi-join over variable assignments.

    Each relation is an iterable of dicts mapping variable names -> concrete values.
    The result is the set of merged assignments that are consistent on overlapping keys.
    """

    acc: List[Dict[str, Any]] = [dict()]
    for rel in relations:
        next_acc: List[Dict[str, Any]] = []
        for left in acc:
            for right in rel:
                merged = dict(left)
                ok = True
                for key, value in right.items():
                    if key in merged and merged[key] != value:
                        ok = False
                        break
                    merged[key] = value
                if ok:
                    next_acc.append(merged)
        acc = next_acc
        if not acc:
            break
    return JoinResult(assignments=acc)


def resolve_placeholders_in_params(
    params: Mapping[str, object],
    *,
    bindings: Mapping[str, object],
    inline_placeholders: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """
    Resolve whole-value placeholders `{var}` using bindings + arithmetic expressions.
    Also replaces inline placeholders (e.g., `{year}`) when provided.
    """

    out: Dict[str, object] = {}
    inline = dict(inline_placeholders or {})
    for key, raw in params.items():
        if raw is None:
            continue
        if isinstance(raw, str):
            text = raw
            if inline and "{" in text and "}" in text:
                for ph, value in inline.items():
                    text = text.replace(ph, value)
            if is_whole_placeholder(text):
                name = placeholder_name(text)
                resolved = resolve_placeholder_value(name, bindings)
                if resolved is not None:
                    out[key] = resolved
                else:
                    out[key] = text
            else:
                out[key] = text
        else:
            out[key] = raw
    return out


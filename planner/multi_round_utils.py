from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, Sequence, Set


PLACEHOLDER_ANY_RE = re.compile(r"\{[^{}]+\}")


class HasCallFields(Protocol):
    phase_key: str
    side_effect: bool
    domain: str


class HasTemplateDomain(Protocol):
    domain: str


def phase_kind(phase_key: str) -> str:
    key = str(phase_key or "").strip().lower()
    if key.startswith("check"):
        return "check"
    if key.startswith("do"):
        return "do"
    return ""


def has_unresolved_placeholders(params: Mapping[str, object]) -> bool:
    for value in params.values():
        if isinstance(value, str) and PLACEHOLDER_ANY_RE.search(value):
            return True
    return False


def dirty_domains_from_history(history: Sequence[HasCallFields], *, prev_do_phase_key: str) -> Set[str]:
    dirty: Set[str] = set()
    for call in history:
        if call.phase_key != prev_do_phase_key:
            continue
        if not bool(call.side_effect):
            continue
        domain = str(getattr(call, "domain", "") or "").strip()
        if domain:
            dirty.add(domain)
    return dirty


def should_run_check_template(
    *,
    template: HasTemplateDomain,
    round_idx: int,
    dirty_domains: Set[str],
    always_domains: Sequence[str] = (),
) -> bool:
    if round_idx <= 1:
        return True
    domain = str(getattr(template, "domain", "") or "").strip()
    if domain and domain in set(always_domains):
        return True
    if not dirty_domains:
        return False
    return bool(domain) and domain in dirty_domains


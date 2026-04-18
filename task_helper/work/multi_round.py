from __future__ import annotations

"""
WorkBench multi-round planning.

This module contains the active WorkBench multi-round adapter used by `planner/twin_track_multi.py`.
"""

import importlib
import json
import math
import random
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from planner.match_join import resolve_placeholder_value
from planner.multi_round_utils import dirty_domains_from_history, has_unresolved_placeholders, should_run_check_template
from planner.twin_track_multi import MultiRoundAdapter, RoundAction

from task_helper.work.utils.decision import (
    allowed_do_qualified_tools,
    can_split_window,
    calendar_selection_ok,
    infer_selection_intent,
    result_is_maybe_truncated,
    split_time_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_TOOL_VIEWS = PROJECT_ROOT / "artifacts" / "input" / "work" / "views" / "tool.json"

_ALWAYS_CHECK_DOMAINS: Tuple[str, ...] = ("company_directory",)

_PLACEHOLDER_RE = re.compile(r"^\{[^{}]+\}$")
_PLACEHOLDER_ANY_RE = re.compile(r"\{[^{}]+\}")
_MONTH_RE = re.compile(r"^(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
_IF_HAVENT_MET_RE = re.compile(r"^\s*if\b.*\bhaven't\s+met\s+with\s+(?P<name>[A-Za-z]+)\b.*\blast\s+(?P<days>\d+)\s+days\b", re.I)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        number = float(value)  # type: ignore[arg-type]
        if math.isnan(number) or math.isinf(number):
            return default
        return float(number)
    except Exception:
        return default


@dataclass(frozen=True)
class WorkHeuristics:
    """
    Lightweight, overrideable heuristics used by the Work multi-round adapter.

    These are intentionally simple and GT-free: they should reduce redundant calls
    and increase internal consistency without encoding dataset-specific answers.
    """

    persist_id_queues_across_rounds: bool = True
    rollout_prefer_single_do: bool = True

    # Deduplicate redundant calendar searches.
    calendar_prune_redundant_searches: bool = True
    calendar_generic_query_terms: Tuple[str, ...] = ("meeting",)

    # If a calendar search is capped, optionally split the window to retrieve more candidates.
    calendar_window_splitting: bool = False
    calendar_window_split_max_calls: int = 8
    calendar_window_split_min_window_seconds: int = 1800
    calendar_search_cap_limit_default: int = 5

    # If the user says first/last/next, enforce the selection rule based on retrieved events.
    calendar_enforce_selection_rule: bool = True
    calendar_intent_keywords_first: Tuple[str, ...] = ("first",)
    calendar_intent_keywords_last: Tuple[str, ...] = ("last",)
    calendar_intent_keywords_next: Tuple[str, ...] = ("next",)

    @staticmethod
    def from_config(cfg: Mapping[str, object]) -> "WorkHeuristics":
        raw = cfg.get("heuristics") if isinstance(cfg.get("heuristics"), dict) else {}
        if not isinstance(raw, dict):
            return WorkHeuristics()

        def _get_bool(key: str, default: bool) -> bool:
            v = raw.get(key)
            return bool(v) if isinstance(v, bool) else default

        def _get_int(key: str, default: int) -> int:
            v = raw.get(key)
            try:
                return int(v)  # type: ignore[arg-type]
            except Exception:
                return default

        def _get_terms(key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
            v = raw.get(key)
            if isinstance(v, list):
                out = []
                for item in v:
                    if isinstance(item, str) and item.strip():
                        out.append(item.strip().lower())
                return tuple(out) if out else default
            return default

        return WorkHeuristics(
            persist_id_queues_across_rounds=_get_bool("persist_id_queues_across_rounds", True),
            rollout_prefer_single_do=_get_bool("rollout_prefer_single_do", True),
            calendar_prune_redundant_searches=_get_bool("calendar_prune_redundant_searches", True),
            calendar_generic_query_terms=_get_terms("calendar_generic_query_terms", ("meeting",)),
            calendar_window_splitting=_get_bool("calendar_window_splitting", False),
            calendar_window_split_max_calls=_get_int("calendar_window_split_max_calls", 8),
            calendar_window_split_min_window_seconds=_get_int("calendar_window_split_min_window_seconds", 1800),
            calendar_search_cap_limit_default=_get_int("calendar_search_cap_limit_default", 5),
            calendar_enforce_selection_rule=_get_bool("calendar_enforce_selection_rule", True),
            calendar_intent_keywords_first=_get_terms("calendar_intent_keywords_first", ("first",)),
            calendar_intent_keywords_last=_get_terms("calendar_intent_keywords_last", ("last",)),
            calendar_intent_keywords_next=_get_terms("calendar_intent_keywords_next", ("next",)),
        )


def _derive_time_max() -> str:
    from task_helper.work.tools.constants import HARDCODED_CURRENT_TIME

    try:
        dt = datetime.strptime(HARDCODED_CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
        return (dt.date() - timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return "2023-11-29"


def _default_year() -> int:
    from task_helper.work.tools.constants import HARDCODED_CURRENT_TIME

    try:
        dt = datetime.strptime(HARDCODED_CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
        return int(dt.year)
    except Exception:
        return 2023


def _parse_ts(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _end_of_day(ts: datetime) -> datetime:
    return datetime(ts.year, ts.month, ts.day, 23, 59, 59)


def _parse_natural_date(text: str, *, default_year: int = 2023) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    raw = raw.rstrip("?.!,")
    if _ISO_DATE_RE.match(raw) or _ISO_DATETIME_RE.match(raw):
        return raw
    match = _MONTH_RE.match(raw)
    if not match:
        return raw
    month_name = match.group("month").lower()
    day = int(match.group("day"))
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = months.get(month_name)
    if not month:
        return raw
    try:
        return datetime(default_year, month, day).strftime("%Y-%m-%d")
    except Exception:
        return raw


def _normalize_metric(value: str) -> str:
    v = str(value or "").strip()
    mapping = {
        "engaged_users": "user_engaged",
        "engaged users": "user_engaged",
        "user_engaged": "user_engaged",
        "total visits": "total_visits",
        "total_visits": "total_visits",
        "average session duration": "session_duration_seconds",
        "session_duration_seconds": "session_duration_seconds",
    }
    return mapping.get(v, v)


def _replace_known_inline_placeholders(text: str, *, time_max_default: str, default_year: int) -> str:
    # Keep this intentionally minimal to avoid "over-fixing" LLM template mistakes.
    # We only resolve framework-level placeholders used in prompts.
    return (
        text.replace("{year}", str(default_year))
        .replace("{time_max}", str(time_max_default))
    )


def _load_tool_view_index() -> Dict[str, Dict[str, object]]:
    payload = json.loads(WORK_TOOL_VIEWS.read_text(encoding="utf-8"))
    views = payload.get("views") if isinstance(payload, dict) else None
    if not isinstance(views, list):
        return {}
    by_qualified: Dict[str, Dict[str, object]] = {}
    for item in views:
        if not isinstance(item, dict):
            continue
        node = item.get("node_pattern") if isinstance(item.get("node_pattern"), dict) else {}
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        qualified = attrs.get("qualified_tool_name")
        if not isinstance(qualified, str) or not qualified.strip():
            continue
        by_qualified[qualified.strip()] = {
            "tool": attrs.get("tool_name"),
            "action_type": attrs.get("action_type"),
            "side_effect": attrs.get("side_effect"),
            "cost": float(item.get("cost") or 0.0),
            "cap_limit": attrs.get("cap_limit"),
        }
    return by_qualified


TOOL_VIEW_INDEX = _load_tool_view_index()


def _phase_kind(phase_key: str) -> str:
    key = str(phase_key or "").strip().lower()
    if key.startswith("check"):
        return "check"
    if key.startswith("do"):
        return "do"
    return ""


def _contains_placeholder(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and _PLACEHOLDER_ANY_RE.search(value):
        return True
    return False


def _normalize_params(qualified: str, params: Mapping[str, object], bindings: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    time_max_default = _derive_time_max()
    default_year = _default_year()

    for key, raw in params.items():
        if raw is None:
            continue
        if isinstance(raw, str):
            text = raw.strip()
            if "{" in text and "}" in text:
                if _PLACEHOLDER_RE.match(text):
                    name = text.strip("{}").strip()
                    if name in bindings:
                        text = str(bindings[name])
                    elif name == "time_min":
                        text = f"{default_year}-01-01 00:00:00"
                    elif name == "time_max":
                        text = str(time_max_default)
                    elif name == "year":
                        text = str(default_year)
                    else:
                        resolved = resolve_placeholder_value(name, bindings)
                        if resolved is not None:
                            text = str(resolved)
                        else:
                            out[key] = text
                            continue
                else:
                    text = _replace_known_inline_placeholders(
                        text, time_max_default=time_max_default, default_year=default_year
                    )

            if "email" in key.lower() and "@" not in text and "{" not in text and "}" not in text:
                name_key = f"{text.strip().lower()}_email"
                if name_key in bindings and isinstance(bindings[name_key], str) and "@" in str(bindings[name_key]):
                    text = str(bindings[name_key])

            if key in {
                "time_min",
                "time_max",
                "date_min",
                "date_max",
                "last_contact_date_min",
                "last_contact_date_max",
                "follow_up_by_min",
                "follow_up_by_max",
            }:
                out[key] = _parse_natural_date(text)
                continue

            if qualified.startswith("analytics.") and key == "value_to_plot":
                out[key] = _normalize_metric(text)
                continue

            out[key] = text
        else:
            out[key] = raw

    if qualified.startswith("analytics.") and "time_max" in params and "time_max" not in out:
        out["time_max"] = time_max_default

    if qualified == "calendar.search_events" and isinstance(out.get("time_min"), str) and isinstance(out.get("time_max"), str):
        tmin = _parse_ts(str(out["time_min"]))
        tmax = _parse_ts(str(out["time_max"]))
        if tmin and tmax and tmax < tmin:
            out["time_max"] = _end_of_day(tmin).strftime("%Y-%m-%d %H:%M:%S")
    return out


def _quote(v: object) -> str:
    text = str(v) if v is not None else ""
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def _call_string(qualified: str, kwargs: Mapping[str, object]) -> str:
    parts = [f"{k}={_quote(v)}" for k, v in kwargs.items() if v is not None]
    return f"{qualified}.func(" + ", ".join(parts) + ")"


def _execute_tool(qualified: str, kwargs: Mapping[str, object]) -> object:
    tool_name, fn_name = qualified.split(".", 1)
    module = importlib.import_module(f"task_helper.work.tools.{tool_name}")
    tool = getattr(module, fn_name, None)
    if tool is None or not hasattr(tool, "func"):
        raise RuntimeError(f"Unknown tool: {qualified}")
    return tool.func(**kwargs)  # type: ignore[misc]


def _reset_work_state() -> None:
    from task_helper.work.evaluation import utils as wb

    for domain in wb.DOMAINS:
        domain.reset_state()


def _extract_ids(result: object) -> Tuple[Optional[str], List[str]]:
    if not isinstance(result, list) or not result:
        return None, []
    if not isinstance(result[0], dict):
        return None, []
    for key in ("customer_id", "email_id", "task_id", "event_id"):
        values = [str(item.get(key) or "").strip() for item in result if isinstance(item, dict)]
        values = [v for v in values if v]
        if values:
            return key, values
    return None, []


@dataclass(frozen=True)
class ActionTemplate:
    phase_kind: str  # "check" | "do"
    action_type: str
    qualified_tool_name: str
    tool_name: str
    domain: str
    side_effect: bool
    cost: float
    params: Dict[str, object]
    order_hint: int = 0


@dataclass
class Call:
    phase_key: str
    action_type: str
    qualified_tool_name: str
    tool_name: str
    domain: str
    side_effect: bool
    cost: float
    params: Dict[str, object]
    raw_action: str
    result: object | None = None


def _merge_params_with_placeholders(base: Dict[str, object], update: Mapping[str, object]) -> None:
    for key, value in update.items():
        if value is None:
            continue
        existing = base.get(key)
        if existing is None:
            base[key] = value
            continue
        if isinstance(existing, str) and _PLACEHOLDER_RE.match(existing.strip()):
            base[key] = value
            continue
        if isinstance(existing, str) and not existing.strip():
            base[key] = value


def _apply_template_to_call(
    *,
    phase_key: str,
    template: ActionTemplate,
    bindings: Mapping[str, object],
    override_params: Optional[Mapping[str, object]] = None,
) -> Call:
    params = dict(template.params)
    if override_params:
        _merge_params_with_placeholders(params, override_params)
        params.update({k: v for k, v in override_params.items() if v is not None})
    resolved = _normalize_params(template.qualified_tool_name, params, bindings)
    raw = _call_string(template.qualified_tool_name, resolved)
    domain = template.domain or (template.qualified_tool_name.split(".", 1)[0] if "." in template.qualified_tool_name else template.tool_name)
    return Call(
        phase_key=phase_key,
        action_type=template.action_type,
        qualified_tool_name=template.qualified_tool_name,
        tool_name=template.tool_name,
        domain=str(domain or template.tool_name or ""),
        side_effect=template.side_effect,
        cost=float(template.cost),
        params=dict(resolved),
        raw_action=raw,
    )


class WorkEnv:
    def __init__(self) -> None:
        self.bindings: Dict[str, object] = {}
        self.queues: Dict[str, List[str]] = {}
        self.errors: List[Dict[str, str]] = []

    def reset(self) -> None:
        _reset_work_state()
        self.bindings = {}
        self.queues = {}
        self.errors = []

    def replay(self, calls: Sequence[Call]) -> None:
        self.reset()
        for call in calls:
            self.execute(call)

    def ingest_call_params(self, call: Call) -> None:
        for key in (
            "time_min",
            "time_max",
            "date_min",
            "date_max",
            "last_contact_date_min",
            "last_contact_date_max",
            "follow_up_by_min",
            "follow_up_by_max",
            "event_start",
        ):
            value = call.params.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or _PLACEHOLDER_ANY_RE.search(text):
                continue
            self.bindings[key] = text

    def execute(self, call: Call) -> object:
        self.ingest_call_params(call)
        if not bool(call.side_effect) and call.result is not None:
            return self._ingest_result(call=call, result=call.result)
        try:
            result = _execute_tool(call.qualified_tool_name, call.params)
        except Exception as exc:
            self.errors.append(
                {
                    "phase_key": str(call.phase_key),
                    "raw_action": str(call.raw_action),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return {"error": str(exc), "error_type": type(exc).__name__}

        call.result = result
        return self._ingest_result(call=call, result=result)

    def _ingest_result(self, *, call: Call, result: object) -> object:
        if isinstance(result, dict) and result:
            for key, value in result.items():
                if value is None:
                    continue
                self.bindings[str(key)] = value
                if isinstance(value, str) and "@" in value and "email" in str(key).lower():
                    self.bindings.setdefault("email", value)
            return result

        if isinstance(result, str) and result and "not provided" not in result.lower() and "not found" not in result.lower():
            if "@" in result and "email" in call.raw_action.lower():
                self.bindings.setdefault("email", result)
        if isinstance(result, list) and result and all(isinstance(v, str) for v in result) and any("@" in str(v) for v in result):
            addrs = [str(v) for v in result if isinstance(v, str)]
            atlas = [a for a in addrs if "@atlas" in a]
            self.bindings.setdefault("email", (atlas[0] if atlas else addrs[0]))
            name = call.params.get("name")
            if isinstance(name, str) and name.strip():
                chosen = atlas[0] if atlas else addrs[0]
                self.bindings[f"{name.strip().lower()}_email"] = chosen
        if isinstance(result, str) and re.fullmatch(r"\d{8}", result.strip()):
            self.bindings.setdefault("last_id", result.strip())

        key, ids = _extract_ids(result)
        if key and ids:
            existing = list(self.queues.get(key) or [])
            if not existing:
                self.queues[key] = ids
            else:
                seen = set(existing)
                self.queues[key] = existing + [i for i in ids if i not in seen]
        return result


def _phase_key(base: str, round_idx: int) -> str:
    return f"{base}_{round_idx}"


def _infer_template_from_action_obj(action_obj: Mapping[str, object], *, fallback_order: int = 0) -> Optional[ActionTemplate]:
    phase_kind = _phase_kind(str(action_obj.get("phase_key") or ""))
    action_type = str(action_obj.get("action_type") or "").strip()
    if not phase_kind or not action_type:
        return None
    params = action_obj.get("params") if isinstance(action_obj.get("params"), dict) else {}
    attrs = action_obj.get("attrs") if isinstance(action_obj.get("attrs"), dict) else {}
    qualified = str(attrs.get("qualified_tool_name") or "").strip()
    tool_name = str(attrs.get("tool_name") or "").strip()
    if not qualified and tool_name and action_type:
        qualified = f"{tool_name}.{action_type}"
    if not tool_name and qualified and "." in qualified:
        tool_name = qualified.split(".", 1)[0]

    tv = TOOL_VIEW_INDEX.get(qualified) if qualified else None
    if tv:
        side_effect = bool(tv.get("side_effect")) if isinstance(tv.get("side_effect"), bool) else (phase_kind == "do")
        cost = _safe_float(tv.get("cost"), default=0.0)
        tool_name = tool_name or str(tv.get("tool") or "")
    else:
        side_effect = bool(attrs.get("side_effect")) if isinstance(attrs.get("side_effect"), bool) else (phase_kind == "do")
        cost = 0.0

    domain = qualified.split(".", 1)[0] if qualified and "." in qualified else tool_name
    order_hint = int(action_obj.get("order_index") or fallback_order)
    return ActionTemplate(
        phase_kind=phase_kind,
        action_type=action_type,
        qualified_tool_name=qualified,
        tool_name=tool_name,
        domain=str(domain or tool_name or ""),
        side_effect=bool(side_effect),
        cost=float(cost),
        params=dict(params),
        order_hint=order_hint,
    )


def _build_action_templates(
    *,
    template: Mapping[str, object],
    seed_bindings: Optional[Mapping[str, object]] = None,
) -> Tuple[List[ActionTemplate], List[ActionTemplate]]:
    templates: List[ActionTemplate] = []
    actions = template.get("actions") if isinstance(template.get("actions"), list) else []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        tmpl = _infer_template_from_action_obj(action, fallback_order=idx)
        if tmpl is not None and tmpl.qualified_tool_name:
            templates.append(tmpl)

    needs_email_names: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        for key, value in params.items():
            if not isinstance(value, str):
                continue
            v = value.strip()
            placeholder = _PLACEHOLDER_RE.match(v)
            if placeholder:
                name = placeholder.group(0).strip("{}").strip()
                match = re.fullmatch(r"(?P<first>[a-zA-Z0-9]+)_email", name)
                if match:
                    needs_email_names.append(match.group("first").lower())
                continue

            if not isinstance(key, str) or "email" not in key.lower():
                continue
            if not v or "@" in v or _PLACEHOLDER_ANY_RE.search(v):
                continue
            needs_email_names.append(v.lower())

    email_lookup_qualified = next((q for q in TOOL_VIEW_INDEX if q.endswith("find_email_address")), "")
    if email_lookup_qualified:
        tv = TOOL_VIEW_INDEX.get(email_lookup_qualified) or {}
        action_type = str(tv.get("action_type") or "find_email_address").strip() or "find_email_address"
        domain = email_lookup_qualified.split(".", 1)[0] if "." in email_lookup_qualified else str(tv.get("tool") or "company_directory")
        for name in sorted(set(needs_email_names)):
            if seed_bindings is not None and f"{name}_email" in seed_bindings:
                continue
            templates.append(
                ActionTemplate(
                    phase_kind="check",
                    action_type=action_type,
                    qualified_tool_name=email_lookup_qualified,
                    tool_name=str(tv.get("tool") or "company_directory"),
                    domain=str(domain or "company_directory"),
                    side_effect=False,
                    cost=_safe_float(tv.get("cost"), default=0.0),
                    params={"name": name},
                    order_hint=-100,
                )
            )

    seen: set[Tuple[str, str, str, Tuple[Tuple[str, str], ...]]] = set()
    uniq: List[ActionTemplate] = []
    for t in templates:
        frozen_params = tuple(sorted((str(k), str(v)) for k, v in (t.params or {}).items() if v is not None))
        key = (t.phase_kind, t.action_type, t.qualified_tool_name, frozen_params)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    check = sorted([t for t in uniq if t.phase_kind == "check"], key=lambda t: t.order_hint)
    do = sorted([t for t in uniq if t.phase_kind == "do"], key=lambda t: t.order_hint)
    return check, do


def _required_actions_from_template(template: Mapping[str, object]) -> List[Tuple[str, str]]:
    required: List[Tuple[str, str]] = []
    actions = template.get("actions") if isinstance(template.get("actions"), list) else []
    for action in actions:
        if not isinstance(action, dict):
            continue
        phase_kind = _phase_kind(str(action.get("phase_key") or ""))
        action_type = str(action.get("action_type") or "").strip()
        if phase_kind and action_type:
            required.append((phase_kind, action_type))
    return required


def _id_key_candidates() -> Tuple[str, ...]:
    return ("customer_id", "email_id", "task_id", "event_id")


def _do_template_id_key(template: ActionTemplate) -> Optional[str]:
    params = template.params
    for key in _id_key_candidates():
        if key in params:
            return key
    if template.action_type.startswith(("delete_", "update_")):
        suffix = template.action_type.split("_", 1)[1] if "_" in template.action_type else ""
        mapping = {"customer": "customer_id", "email": "email_id", "task": "task_id", "event": "event_id"}
        return mapping.get(suffix)
    return None


def _template_id_key(template: ActionTemplate) -> Optional[str]:
    params = template.params
    for key in _id_key_candidates():
        if key in params:
            return key
    return None


def _has_unresolved_placeholders(params: Mapping[str, object]) -> bool:
    return has_unresolved_placeholders(params)


def _first_free_slot_start(
    *,
    events: Sequence[Mapping[str, object]],
    window_start: datetime,
    window_end: datetime,
    duration_minutes: int,
    business_hours: bool = True,
) -> Optional[str]:
    """
    Find the first free slot start in [window_start, window_end] with given duration.
    Uses event_start + duration to block intervals.
    """

    duration = max(1, int(duration_minutes))

    intervals: List[Tuple[datetime, datetime]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        start_raw = event.get("event_start")
        dur_raw = event.get("duration")
        start = _parse_ts(str(start_raw)) if start_raw is not None else None
        if start is None:
            continue
        try:
            dur = int(str(dur_raw)) if dur_raw is not None else 0
        except Exception:
            dur = 0
        if dur <= 0:
            continue
        end = start + timedelta(minutes=dur)
        if end <= window_start or start >= window_end:
            continue
        intervals.append((max(start, window_start), min(end, window_end)))

    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    def _within_business_hours(ts: datetime) -> bool:
        if not business_hours:
            return True
        if ts.hour < 9:
            return False
        if ts.hour > 18:
            return False
        if ts.hour == 18 and (ts.minute > 0 or ts.second > 0):
            return False
        return True

    cursor = window_start
    if business_hours and cursor.hour < 9:
        cursor = datetime(cursor.year, cursor.month, cursor.day, 9, 0, 0)
    for start, end in merged:
        if cursor + timedelta(minutes=duration) <= start:
            if _within_business_hours(cursor) and _within_business_hours(cursor + timedelta(minutes=duration)):
                return cursor.strftime("%Y-%m-%d %H:%M:%S")
        cursor = max(cursor, end)
        if business_hours and cursor.hour >= 18:
            cursor = datetime(cursor.year, cursor.month, cursor.day, 9, 0, 0) + timedelta(days=1)
    if cursor + timedelta(minutes=duration) <= window_end:
        if _within_business_hours(cursor) and _within_business_hours(cursor + timedelta(minutes=duration)):
            return cursor.strftime("%Y-%m-%d %H:%M:%S")
    return None


class WorkMultiRoundAdapter(MultiRoundAdapter):
    def __init__(
        self,
        *,
        tree: Mapping[str, object],
        config: Mapping[str, object],
    ) -> None:
        self.tree = dict(tree)
        template = self.tree.get("template") if isinstance(self.tree.get("template"), dict) else {}
        self.template = dict(template)
        self.cfg = dict(config)
        self.export_enriched_tree = bool(self.cfg.get("export_enriched_tree", False))
        self.export_mcts_tree = bool(self.cfg.get("export_mcts_tree", False))

        self.template_id = str(self.template.get("template_id") or self.template.get("plan_id") or self.template.get("id") or "UNKNOWN_TEMPLATE")
        self.plan = self.template.get("plan") if isinstance(self.template.get("plan"), dict) else {}
        self.heur = WorkHeuristics.from_config(self.cfg)

        materialized = self.tree.get("materialized") if isinstance(self.tree.get("materialized"), dict) else {}
        seed_bindings = materialized.get("bindings") if isinstance(materialized.get("bindings"), dict) else {}
        seed_queues = materialized.get("queues") if isinstance(materialized.get("queues"), dict) else {}
        self.seed_bindings: Dict[str, object] = {str(k): v for k, v in seed_bindings.items()}
        self.seed_queues: Dict[str, List[str]] = {}
        for key, values in seed_queues.items():
            if not isinstance(values, list):
                continue
            cleaned = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
            if cleaned:
                self.seed_queues[str(key)] = cleaned

        root = self.tree.get("root") if isinstance(self.tree.get("root"), dict) else None
        # Static AND–OR tree from ComposeMatch; we keep it immutable and rebuild a fresh runtime tree
        # whenever we replay a history (avoids cross-branch contamination inside MCTS).
        self.static_root: Optional[Dict[str, object]] = None
        if root is not None:
            try:
                self.static_root = json.loads(json.dumps(root))
            except Exception:
                self.static_root = dict(root)
        self.runtime_root: Optional[Dict[str, object]] = None
        self._reset_tree()

        self.base_required_actions = _required_actions_from_template(self.template)
        self.check_templates, self.do_templates = _build_action_templates(template=self.template, seed_bindings=self.seed_bindings)

        self.env = WorkEnv()
        self._reset_env()

        self.semantic_tolerance = float(self.cfg.get("semantic_tolerance") or 0.8)
        self.semantic_shortfall_weight = float(self.cfg.get("semantic_shortfall_weight") or 1e6)
        self.hard_fail_penalty = float(self.cfg.get("hard_fail_penalty") or -1e12)
        self.max_do_calls_per_round = int(self.cfg.get("max_do_calls_per_round") or 25)
        self.max_ids_per_action = int(self.cfg.get("max_ids_per_action") or 5)

    def _collect_tree_or_slots(self, *, phase_kind: str) -> List[Dict[str, object]]:
        root = self.runtime_root
        if not isinstance(root, dict):
            return []
        children = root.get("children")
        if not isinstance(children, list):
            return []

        phase_kind_norm = str(phase_kind or "").strip().lower()
        out: List[Dict[str, object]] = []
        for phase_node in children:
            if not isinstance(phase_node, dict):
                continue
            if str(phase_node.get("kind") or "") != "AND":
                continue
            qp = phase_node.get("query_phase") if isinstance(phase_node.get("query_phase"), dict) else {}
            pk = str(qp.get("phase_key") or qp.get("phase_type") or "").strip().lower()
            pk_kind = _phase_kind(pk)
            if pk_kind != phase_kind_norm:
                continue
            for slot in phase_node.get("children") if isinstance(phase_node.get("children"), list) else []:
                if not isinstance(slot, dict):
                    continue
                if str(slot.get("kind") or "") != "OR":
                    continue
                out.append(slot)
        return out

    def _tree_slot_id_key(self, slot: Mapping[str, object]) -> Optional[str]:
        qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
        params = qa.get("params") if isinstance(qa.get("params"), dict) else {}
        for key in _id_key_candidates():
            if key in params:
                return key
        return None

    def _slot_param_match_score(self, *, slot_params: Mapping[str, object], call_params: Mapping[str, object]) -> int:
        """
        Score how well a slot's (template) params match a concrete call's params.

        Only non-null, non-placeholder slot params participate; placeholders are treated as wildcards.
        Returns -1 if incompatible.
        """

        score = 0
        for key, want in slot_params.items():
            if want is None:
                continue
            if _contains_placeholder(want):
                continue
            got = call_params.get(key)
            if got is None:
                return -1
            if isinstance(want, str) and isinstance(got, str):
                if want.strip().lower() != got.strip().lower():
                    return -1
            else:
                if want != got:
                    return -1
            score += 1
        return int(score)

    def _best_slot_for_call(self, *, slots: Sequence[Mapping[str, object]], call: Call) -> Optional[Mapping[str, object]]:
        qualified = str(call.qualified_tool_name or "").strip()
        if not qualified:
            return None

        best: Optional[Mapping[str, object]] = None
        best_score = -1
        fallback: Optional[Mapping[str, object]] = None

        for slot in slots:
            qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
            slot_qualified = str(qa.get("qualified_tool_name") or "").strip()
            if not slot_qualified or slot_qualified != qualified:
                continue
            if fallback is None:
                fallback = slot
            slot_params = qa.get("params") if isinstance(qa.get("params"), dict) else {}
            if not isinstance(slot_params, dict):
                slot_params = {}
            score = self._slot_param_match_score(slot_params=slot_params, call_params=call.params)
            if score < 0:
                continue
            if score > best_score:
                best_score = score
                best = slot

        return best or fallback

    def _result_key(self, result: object) -> str:
        """
        Stable key for deduping results across rounds in enriched trees.
        """

        def _sanitize(v: object) -> object:
            if isinstance(v, float):
                return v if math.isfinite(v) else None
            if isinstance(v, dict):
                return {str(k): _sanitize(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_sanitize(x) for x in v]
            return v

        try:
            return json.dumps(_sanitize(result), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except Exception:
            return str(result)

    def _accumulate_result_set(self, *, meta: Dict[str, object], result: object) -> None:
        """
        Store a deduped set of observed results (not per-round logs).

        - Scalars: dedupe by value
        - List[dict] with ID keys: union by id_key
        - Other lists: union unique elements (by stable json key)
        """

        existing = meta.get("result_set")
        if not isinstance(existing, list):
            existing = []

        if isinstance(result, list):
            id_key, ids = _extract_ids(result)
            if id_key and ids and all(isinstance(x, dict) for x in result):
                seen_ids: set[str] = set()
                for item in existing:
                    if isinstance(item, dict):
                        rid = str(item.get(id_key) or "").strip()
                        if rid:
                            seen_ids.add(rid)
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    rid = str(item.get(id_key) or "").strip()
                    if not rid or rid in seen_ids:
                        continue
                    existing.append(item)
                    seen_ids.add(rid)
                meta["result_set"] = existing
                return

            seen = {self._result_key(x) for x in existing}
            for item in result:
                k = self._result_key(item)
                if k in seen:
                    continue
                existing.append(item)
                seen.add(k)
            meta["result_set"] = existing
            return

        key = self._result_key(result)
        if all(self._result_key(x) != key for x in existing):
            existing.append(result)
        meta["result_set"] = existing

    def _tree_candidate_to_call(self, *, slot: Mapping[str, object], candidate: Mapping[str, object], phase_key: str) -> Optional[Call]:
        qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
        qualified = str(qa.get("qualified_tool_name") or qa.get("tool_name") or "").strip()
        if not qualified:
            return None
        action_type = str(qa.get("action_type") or "").strip() or (qualified.split(".", 1)[1] if "." in qualified else "")
        tool_name = str(qa.get("tool_name") or (qualified.split(".", 1)[0] if "." in qualified else "")).strip()
        domain = qualified.split(".", 1)[0] if "." in qualified else tool_name
        side_effect = bool(qa.get("side_effect", True))

        meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
        args = meta.get("args") if isinstance(meta.get("args"), dict) else None
        if args is None:
            args = qa.get("params") if isinstance(qa.get("params"), dict) else {}
        if not isinstance(args, dict):
            return None
        resolved = _normalize_params(qualified, args, self.env.bindings)
        raw = _call_string(qualified, resolved)
        tv = TOOL_VIEW_INDEX.get(qualified) or {}
        cost = _safe_float(candidate.get("tool_call_cost"), default=_safe_float(tv.get("cost"), default=0.0))
        return Call(
            phase_key=phase_key,
            action_type=action_type,
            qualified_tool_name=qualified,
            tool_name=tool_name,
            domain=str(domain or tool_name or ""),
            side_effect=side_effect,
            cost=float(cost),
            params=dict(resolved),
            raw_action=raw,
        )

    def _tree_record_executed_call(self, *, phase_kind: str, call: Call, round_idx: int) -> None:
        if not isinstance(self.runtime_root, dict):
            return
        slots = self._collect_tree_or_slots(phase_kind=str(phase_kind or ""))
        if not slots:
            return
        slot = self._best_slot_for_call(slots=slots, call=call)
        if not isinstance(slot, dict):
            return

        candidates = slot.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
            slot["candidates"] = candidates

        found = None
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
            args = meta.get("args") if isinstance(meta.get("args"), dict) else None
            if args is None:
                continue
            try:
                if dict(args) == dict(call.params):
                    found = cand
                    break
            except Exception:
                continue
        if found is None:
            found = {
                "source": "executed",
                "tool": str(call.tool_name),
                "tool_view_id": f"tool::{call.tool_name}::{call.action_type}",
                "tool_call_cost": float(call.cost),
                "text": str(call.raw_action),
                "cost": 0.0,
                "meta": {
                    "type": "tool",
                    "tool": str(call.tool_name),
                    "qualified_tool_name": str(call.qualified_tool_name),
                    "args": dict(call.params),
                    "side_effect": bool(call.side_effect),
                },
                "executed_round": int(round_idx),
            }
            candidates.append(found)
        meta = found.get("meta") if isinstance(found.get("meta"), dict) else {}
        meta["result"] = call.result
        self._accumulate_result_set(meta=meta, result=call.result)
        if "results_by_round" in meta:
            meta.pop("results_by_round", None)
        found["meta"] = meta
        found["executed_round"] = int(round_idx)

    def _tree_enrich_do_candidates(self, *, evidence: Mapping[str, Sequence[str]], round_idx: int) -> None:
        if not isinstance(self.runtime_root, dict):
            return
        slots = self._collect_tree_or_slots(phase_kind="do")
        if not slots:
            return
        for slot in slots:
            id_key = self._tree_slot_id_key(slot)
            if not id_key:
                continue
            ids = evidence.get(id_key)
            if not ids:
                continue
            candidates = slot.get("candidates")
            if not isinstance(candidates, list):
                candidates = []
                slot["candidates"] = candidates

            existing: set[str] = set()
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
                v = args.get(id_key)
                if isinstance(v, str) and v.strip() and not _contains_placeholder(v):
                    existing.add(v.strip())

            qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
            qualified = str(qa.get("qualified_tool_name") or "").strip()
            base_params = qa.get("params") if isinstance(qa.get("params"), dict) else {}
            for idv in [str(x).strip() for x in ids if isinstance(x, str) and str(x).strip()]:
                if idv in existing:
                    continue
                params = dict(base_params)
                params[id_key] = idv
                resolved = _normalize_params(qualified, params, self.env.bindings)
                raw = _call_string(qualified, resolved)
                tv = TOOL_VIEW_INDEX.get(qualified) or {}
                tool_name = str(qa.get("tool_name") or (qualified.split(".", 1)[0] if "." in qualified else "")).strip()
                cand = {
                    "source": "materialized",
                    "tool": tool_name,
                    "tool_view_id": f"tool::{tool_name}::{str(qa.get('action_type') or '')}",
                    "tool_call_cost": float(_safe_float(tv.get("cost"), default=0.0)),
                    "text": raw,
                    "cost": 0.0,
                    "meta": {
                        "type": "tool",
                        "tool": tool_name,
                        "qualified_tool_name": qualified,
                        "args": dict(resolved),
                        "side_effect": bool(qa.get("side_effect", True)),
                        "fallback": False,
                    },
                    "materialized_round": int(round_idx),
                }
                candidates.append(cand)
                existing.add(idv)

    def _build_enriched_root(self, *, history: Sequence[Call]) -> Optional[Dict[str, object]]:
        """
        Build an enriched AND–OR tree for a specific (best) history.

        This is computed from scratch from `static_root` to avoid leaking mutations across MCTS branches.
        """

        if not isinstance(self.static_root, dict):
            return None
        try:
            root: Dict[str, object] = json.loads(json.dumps(self.static_root))
        except Exception:
            root = dict(self.static_root)

        def _collect_slots(on_root: Mapping[str, object], *, phase_kind: str) -> List[Dict[str, object]]:
            children = on_root.get("children")
            if not isinstance(children, list):
                return []
            want = str(phase_kind or "").strip().lower()
            out: List[Dict[str, object]] = []
            for phase_node in children:
                if not isinstance(phase_node, dict) or str(phase_node.get("kind") or "") != "AND":
                    continue
                qp = phase_node.get("query_phase") if isinstance(phase_node.get("query_phase"), dict) else {}
                pk = str(qp.get("phase_key") or qp.get("phase_type") or "").strip().lower()
                if _phase_kind(pk) != want:
                    continue
                for slot in phase_node.get("children") if isinstance(phase_node.get("children"), list) else []:
                    if isinstance(slot, dict) and str(slot.get("kind") or "") == "OR":
                        out.append(slot)
            return out

        check_slots = _collect_slots(root, phase_kind="check")
        do_slots = _collect_slots(root, phase_kind="do")

        def _round_idx(phase_key: str) -> Optional[int]:
            m = re.search(r"_(\d+)$", str(phase_key or ""))
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None

        def _record(slots: List[Dict[str, object]], call: Call, *, round_idx: int) -> None:
            slot_any = self._best_slot_for_call(slots=slots, call=call)
            if not isinstance(slot_any, dict):
                return
            slot: Dict[str, object] = slot_any  # type: ignore[assignment]

            candidates = slot.get("candidates")
            if not isinstance(candidates, list):
                candidates = []
                slot["candidates"] = candidates
            found = None
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                args = meta.get("args") if isinstance(meta.get("args"), dict) else None
                if args is None:
                    continue
                try:
                    if dict(args) == dict(call.params):
                        found = cand
                        break
                except Exception:
                    continue
            if found is None:
                found = {
                    "source": "executed",
                    "tool": str(call.tool_name),
                    "tool_view_id": f"tool::{call.tool_name}::{call.action_type}",
                    "tool_call_cost": float(call.cost),
                    "text": str(call.raw_action),
                    "cost": 0.0,
                    "meta": {
                        "type": "tool",
                        "tool": str(call.tool_name),
                        "qualified_tool_name": str(call.qualified_tool_name),
                        "args": dict(call.params),
                        "side_effect": bool(call.side_effect),
                    },
                    "executed_round": int(round_idx),
                }
                candidates.append(found)
            meta = found.get("meta") if isinstance(found.get("meta"), dict) else {}
            meta["result"] = call.result
            self._accumulate_result_set(meta=meta, result=call.result)
            if "results_by_round" in meta:
                meta.pop("results_by_round", None)
            found["meta"] = meta
            found["executed_round"] = int(round_idx)

        def _enrich_do(*, bindings: Mapping[str, object], evidence: Mapping[str, Sequence[str]], round_idx: int) -> None:
            for slot in do_slots:
                qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
                qualified = str(qa.get("qualified_tool_name") or "").strip()
                params = qa.get("params") if isinstance(qa.get("params"), dict) else {}
                id_key = None
                for k in _id_key_candidates():
                    if k in params:
                        id_key = k
                        break
                if not id_key:
                    continue
                ids = evidence.get(id_key)
                if not ids:
                    continue
                candidates = slot.get("candidates")
                if not isinstance(candidates, list):
                    candidates = []
                    slot["candidates"] = candidates
                existing = {
                    str((cand.get("meta") or {}).get("args", {}).get(id_key) or "").strip()
                    for cand in candidates
                    if isinstance(cand, dict)
                }
                for idv in [str(x).strip() for x in ids if isinstance(x, str) and str(x).strip()]:
                    if idv in existing:
                        continue
                    p = dict(params)
                    p[id_key] = idv
                    resolved = _normalize_params(qualified, p, bindings)
                    raw = _call_string(qualified, resolved)
                    tv = TOOL_VIEW_INDEX.get(qualified) or {}
                    tool_name = str(qa.get("tool_name") or (qualified.split(".", 1)[0] if "." in qualified else "")).strip()
                    candidates.append(
                        {
                            "source": "materialized",
                            "tool": tool_name,
                            "tool_view_id": f"tool::{tool_name}::{str(qa.get('action_type') or '')}",
                            "tool_call_cost": float(_safe_float(tv.get("cost"), default=0.0)),
                            "text": raw,
                            "cost": 0.0,
                            "meta": {
                                "type": "tool",
                                "tool": tool_name,
                                "qualified_tool_name": qualified,
                                "args": dict(resolved),
                                "side_effect": bool(qa.get("side_effect", True)),
                                "fallback": False,
                            },
                            "materialized_round": int(round_idx),
                        }
                    )
                    existing.add(idv)

        # Replay only bindings/queues from Check evidence; avoid executing side-effecting Do tools.
        env = WorkEnv()
        env.reset()
        env.bindings.update(dict(self.seed_bindings))
        for key, values in self.seed_queues.items():
            existing = list(env.queues.get(key) or [])
            seen = set(existing)
            merged = existing + [v for v in values if v not in seen]
            if merged:
                env.queues[key] = merged

        rounds: List[int] = []
        for call in history:
            ridx = _round_idx(call.phase_key)
            if ridx is not None and ridx not in rounds:
                rounds.append(ridx)
        rounds = sorted(rounds)

        for r in rounds:
            check_round = [c for c in history if _phase_kind(c.phase_key) == "check" and _round_idx(c.phase_key) == r]
            for c in check_round:
                env.ingest_call_params(c)
                if c.result is not None:
                    env._ingest_result(call=c, result=c.result)
                _record(check_slots, c, round_idx=r)

            evidence: Dict[str, List[str]] = {}
            for c in check_round:
                key, ids = _extract_ids(c.result)
                if key and ids:
                    evidence.setdefault(key, []).extend(list(ids))
                for id_key in _id_key_candidates():
                    v = c.params.get(id_key)
                    if isinstance(v, str) and v.strip():
                        evidence.setdefault(id_key, []).append(v.strip())
            _enrich_do(bindings=env.bindings, evidence=evidence, round_idx=r)

            do_round = [c for c in history if _phase_kind(c.phase_key) == "do" and _round_idx(c.phase_key) == r]
            for c in do_round:
                env.ingest_call_params(c)
                if c.result is not None:
                    env._ingest_result(call=c, result=c.result)
                _record(do_slots, c, round_idx=r)

        return root

    def _reset_env(self) -> None:
        self.env.reset()
        self.env.bindings.update(dict(self.seed_bindings))
        for key, values in self.seed_queues.items():
            existing = list(self.env.queues.get(key) or [])
            seen = set(existing)
            merged = existing + [v for v in values if v not in seen]
            if merged:
                self.env.queues[key] = merged

    def _reset_tree(self) -> None:
        if not isinstance(self.static_root, dict):
            self.runtime_root = None
            return
        try:
            self.runtime_root = json.loads(json.dumps(self.static_root))
        except Exception:
            self.runtime_root = dict(self.static_root)

    def _replay(self, history: Sequence[Call]) -> None:
        self._reset_env()
        self._reset_tree()
        for call in history:
            self.env.execute(call)

    def initial_history(self) -> List[Call]:
        return []

    def _effective_required_actions(self, *, history: Sequence[Call]) -> List[Tuple[str, str]]:
        required = list(self.base_required_actions)
        query = str(self.plan.get("query") or "")
        match = _IF_HAVENT_MET_RE.match(query)
        if not match:
            return required
        name = match.group("name").strip().lower()
        try:
            days = int(match.group("days"))
        except Exception:
            days = 0
        if not name or days <= 0:
            return required

        met_recently = False
        for call in history:
            if call.qualified_tool_name != "calendar.search_events":
                continue
            if call.result is None:
                continue
            q = call.params.get("query")
            if isinstance(q, str) and q.strip().lower() != name:
                continue
            if isinstance(call.result, list) and len(call.result) > 0:
                met_recently = True
                break
        if met_recently:
            return [(k, a) for (k, a) in required if k != "do"]
        return required

    def _do_required(self, *, history: Sequence[Call]) -> bool:
        return any(k == "do" for (k, _) in self._effective_required_actions(history=history))

    def _do_ids_supported_by_checks(self, *, history: Sequence[Call]) -> bool:
        """
        GT-free "no extra Do" guard:
        If a Do call uses an ID param (customer_id/email_id/task_id/event_id),
        require the ID appears in Check evidence from the same round.
        """

        def _round_idx(phase_key: str) -> Optional[int]:
            m = re.search(r"_(\d+)$", str(phase_key or ""))
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None

        allowed_by_round: Dict[int, Dict[str, set[str]]] = {}
        id_keys = _id_key_candidates()

        for call in history:
            if _phase_kind(call.phase_key) != "check":
                continue
            ridx = _round_idx(call.phase_key)
            if ridx is None:
                continue
            bucket = allowed_by_round.setdefault(ridx, {})

            key, ids = _extract_ids(call.result)
            if key and ids:
                bucket.setdefault(key, set()).update(set(ids))

            for id_key in id_keys:
                v = call.params.get(id_key)
                if isinstance(v, str) and v.strip():
                    bucket.setdefault(id_key, set()).add(v.strip())

        for call in history:
            if _phase_kind(call.phase_key) != "do":
                continue
            ridx = _round_idx(call.phase_key)
            if ridx is None:
                continue
            allowed = allowed_by_round.get(ridx) or {}
            for id_key in id_keys:
                v = call.params.get(id_key)
                if not isinstance(v, str) or not v.strip():
                    continue
                evidence = allowed.get(id_key)
                if evidence is None:
                    continue
                if v.strip() not in evidence:
                    return False

        return True

    def _stop_sign_reached(self, *, history: Sequence[Call]) -> bool:
        """
        Stop-sign for open-ended batch tasks (runs=0):
        after the most recent check round, there are no remaining actionable IDs
        for any Do template, after filtering processed IDs.
        """

        if not self._do_required(history=history):
            return True

        def _round_idx(phase_key: str) -> Optional[int]:
            m = re.search(r"_(\d+)$", str(phase_key or ""))
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None

        last_check_round: Optional[int] = None
        for call in history:
            if _phase_kind(call.phase_key) != "check":
                continue
            ridx = _round_idx(call.phase_key)
            if ridx is None:
                continue
            last_check_round = ridx if last_check_round is None else max(last_check_round, ridx)
        if last_check_round is None:
            return False

        evidence: Dict[str, set[str]] = {}
        for call in history:
            if _phase_kind(call.phase_key) != "check":
                continue
            ridx = _round_idx(call.phase_key)
            if ridx != last_check_round:
                continue
            key, ids = _extract_ids(call.result)
            if key and ids:
                evidence.setdefault(key, set()).update(set(ids))
            for id_key in _id_key_candidates():
                v = call.params.get(id_key)
                if isinstance(v, str) and v.strip():
                    evidence.setdefault(id_key, set()).add(v.strip())

        processed: Dict[str, set[str]] = {}
        for call in history:
            if _phase_kind(call.phase_key) != "do":
                continue
            for id_key in _id_key_candidates():
                v = call.params.get(id_key)
                if isinstance(v, str) and v.strip():
                    processed.setdefault(id_key, set()).add(v.strip())

        executed: set[Tuple[str, str]] = set()
        for call in history:
            if call.raw_action and call.raw_action != "-":
                executed.add((_phase_kind(call.phase_key), call.action_type))

        tree_do_slots = self._collect_tree_or_slots(phase_kind="do")
        # Prefer the static AND–OR tree's Do slots when available (enforces per-slot completeness even
        # when multiple Do actions share the same action_type).
        if tree_do_slots:
            for slot in tree_do_slots:
                qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
                action_type = str(qa.get("action_type") or "").strip()
                if not action_type:
                    qualified = str(qa.get("qualified_tool_name") or "").strip()
                    action_type = qualified.split(".", 1)[1] if "." in qualified else qualified
                if not action_type:
                    continue
                id_key = self._tree_slot_id_key(slot)

                if not id_key and ("do", action_type) in executed:
                    continue

                params = qa.get("params") if isinstance(qa.get("params"), dict) else {}
                if id_key and isinstance(params.get(id_key), str) and not _contains_placeholder(params.get(id_key)):
                    fixed = str(params.get(id_key) or "").strip()
                    if fixed:
                        done = processed.get(id_key) or set()
                        if fixed in done:
                            continue
                        ev = evidence.get(id_key)
                        if ev is not None and fixed not in ev:
                            continue
                        return False

                if id_key:
                    ids = list(evidence.get(id_key) or [])
                    done = processed.get(id_key) or set()
                    ids = [x for x in ids if x not in done]
                    if ids:
                        return False
                    continue

                # Non-ID Do: if we can resolve all params, there's still an actionable Do remaining.
                qualified = str(qa.get("qualified_tool_name") or "").strip()
                if qualified:
                    resolved = _normalize_params(qualified, params, self.env.bindings)
                    if not _has_unresolved_placeholders(resolved):
                        return False
            return True

        for tmpl in self.do_templates:
            id_key = _do_template_id_key(tmpl)
            if not id_key and ("do", tmpl.action_type) in executed:
                continue
            params = dict(tmpl.params)
            if id_key and isinstance(params.get(id_key), str):
                fixed = str(params.get(id_key) or "").strip()
                if fixed and not _contains_placeholder(fixed):
                    done = processed.get(id_key) or set()
                    if fixed in done:
                        continue
                    ev = evidence.get(id_key)
                    if ev is not None and fixed not in ev:
                        continue
                    return False
            if id_key and (id_key not in params or _contains_placeholder(params.get(id_key))):
                ids = list(evidence.get(id_key) or [])
                done = processed.get(id_key) or set()
                ids = [x for x in ids if x not in done]
                if ids:
                    return False
                continue
            call = _apply_template_to_call(phase_key="do", template=tmpl, bindings=self.env.bindings)
            if not _has_unresolved_placeholders(call.params):
                return False

        return True

    def _derive_check_bindings(self, *, check_calls: Sequence[Call]) -> None:
        needs_free_slot = False
        duration_minutes = 30
        for tmpl in self.do_templates:
            for key, value in tmpl.params.items():
                if isinstance(value, str) and value.strip() == "{first_free_slot_start}":
                    needs_free_slot = True
                    break
            if needs_free_slot:
                dur = tmpl.params.get("duration")
                try:
                    duration_minutes = int(str(dur)) if dur is not None else duration_minutes
                except Exception:
                    duration_minutes = duration_minutes
                break

        if needs_free_slot and "first_free_slot_start" not in self.env.bindings:
            best: Optional[Call] = None
            for call in check_calls:
                if call.qualified_tool_name != "calendar.search_events":
                    continue
                if call.result is None or not isinstance(call.result, list):
                    continue
                q = call.params.get("query")
                if q is not None and str(q).strip():
                    continue
                best = call
            if best is not None:
                tmin = _parse_ts(str(best.params.get("time_min") or ""))
                tmax = _parse_ts(str(best.params.get("time_max") or ""))
                if tmin and tmax:
                    slot = _first_free_slot_start(
                        events=[e for e in best.result if isinstance(e, Mapping)],  # type: ignore[arg-type]
                        window_start=tmin,
                        window_end=tmax,
                        duration_minutes=duration_minutes,
                        business_hours=True,
                    )
                    if slot:
                        self.env.bindings["first_free_slot_start"] = slot

    def _prune_check_calls(self, calls: List[Call]) -> List[Call]:
        """
        Keep check-phase tool calls minimal.

        Some selected memory views include generic check actions (e.g., calendar.search_events over a window)
        that can be redundant with the init template's check action(s). Since the check phase executes all
        check calls for a round, we prune dominated duplicates before execution.
        """

        uniq: List[Call] = []
        seen_raw: set[str] = set()
        for call in calls:
            raw = str(call.raw_action or "")
            if raw in seen_raw:
                continue
            seen_raw.add(raw)
            uniq.append(call)

        if not self.heur.calendar_prune_redundant_searches:
            return uniq

        # Special-case: calendar.search_events over the same window.
        # If we have both a filtered search (non-empty `query`) and an unfiltered search for the same window,
        # keep only one to avoid redundant tool calls.
        by_window: Dict[Tuple[str, str], List[Call]] = {}
        rest: List[Call] = []
        for call in uniq:
            if call.qualified_tool_name != "calendar.search_events":
                rest.append(call)
                continue
            tmin = call.params.get("time_min")
            tmax = call.params.get("time_max")
            if not isinstance(tmin, str) or not isinstance(tmax, str):
                rest.append(call)
                continue
            by_window.setdefault((tmin, tmax), []).append(call)

        kept: List[Call] = []
        for window, group in by_window.items():
            unfiltered: List[Call] = []
            filtered: List[Call] = []
            for call in group:
                q = call.params.get("query")
                if q is None or (isinstance(q, str) and not q.strip()):
                    unfiltered.append(call)
                else:
                    filtered.append(call)
            if not filtered:
                kept.append(unfiltered[0] if unfiltered else group[0])
                continue

            # Heuristic: some query terms (e.g., "meeting") are used generically in this dataset.
            # When such a generic term is present, prefer an unfiltered window search if available.
            generic = set(self.heur.calendar_generic_query_terms)
            prefer_unfiltered = any(
                isinstance(c.params.get("query"), str) and str(c.params.get("query")).strip().lower() in generic
                for c in filtered
            )
            if prefer_unfiltered and unfiltered:
                kept.append(unfiltered[0])
            else:
                kept.append(filtered[0])

        return rest + kept

    def _calendar_selection_ok(self, *, history: Sequence[Call]) -> Optional[bool]:
        if not self.heur.calendar_enforce_selection_rule:
            return None
        notes = self.template.get("notes") if isinstance(self.template.get("notes"), dict) else {}
        intent = infer_selection_intent(query=str(self.plan.get("query") or ""), selection_rule=str(notes.get("selection_rule") or ""))
        if intent is None:
            return None

        selected_event_id: Optional[str] = None
        for call in history:
            if _phase_kind(call.phase_key) != "do":
                continue
            if call.qualified_tool_name not in ("calendar.delete_event", "calendar.update_event"):
                continue
            eid = call.params.get("event_id")
            if isinstance(eid, str) and eid.strip():
                selected_event_id = eid.strip()
                break
        if not selected_event_id:
            return None

        events_by_id: Dict[str, Mapping[str, object]] = {}
        for call in history:
            if _phase_kind(call.phase_key) != "check":
                continue
            if call.qualified_tool_name != "calendar.search_events":
                continue
            if not isinstance(call.result, list):
                continue
            for ev in call.result:
                if not isinstance(ev, Mapping):
                    continue
                eid = str(ev.get("event_id") or "").strip()
                if not eid:
                    continue
                events_by_id.setdefault(eid, ev)
        events: List[Mapping[str, object]] = list(events_by_id.values())
        if not events:
            return None

        from task_helper.work.tools.constants import HARDCODED_CURRENT_TIME

        return calendar_selection_ok(intent=intent, selected_event_id=selected_event_id, retrieved_events=events, now=str(HARDCODED_CURRENT_TIME))

    def _available_round_actions(self, *, history: Sequence[Call], round_idx: int) -> Tuple[List[Call], List[RoundAction]]:
        check_phase = _phase_key("check", round_idx)
        do_phase = _phase_key("do", round_idx)

        self._replay(history)
        if not self.heur.persist_id_queues_across_rounds:
            self.env.queues = {}

        dirty_domains: set[str] = set()
        if round_idx > 1:
            prev_do = _phase_key("do", round_idx - 1)
            dirty_domains = set(dirty_domains_from_history(history, prev_do_phase_key=prev_do))

        check_candidates: List[Call] = []
        check_calls: List[Call] = []
        pending_id_checks: List[Tuple[ActionTemplate, str]] = []
        pending_placeholder_checks: List[ActionTemplate] = []
        for tmpl in self.check_templates:
            if not should_run_check_template(
                template=tmpl,
                round_idx=round_idx,
                dirty_domains=dirty_domains,
                always_domains=_ALWAYS_CHECK_DOMAINS,
            ):
                continue
            call = _apply_template_to_call(phase_key=check_phase, template=tmpl, bindings=self.env.bindings)
            # Seed bindings from already-resolved params so later templates (often from memory views) can reuse
            # concrete time windows even before we execute the tool calls.
            self.env.ingest_call_params(call)
            id_key = _template_id_key(tmpl)
            if id_key and _contains_placeholder(call.params.get(id_key)):
                pending_id_checks.append((tmpl, id_key))
                continue
            if _has_unresolved_placeholders(call.params):
                pending_placeholder_checks.append(tmpl)
                continue
            check_candidates.append(call)

        def _maybe_split_calendar_search(base_call: Call) -> List[Call]:
            if not self.heur.calendar_window_splitting:
                return []
            if base_call.qualified_tool_name != "calendar.search_events":
                return []
            tv = TOOL_VIEW_INDEX.get(base_call.qualified_tool_name) or {}
            cap = tv.get("cap_limit")
            try:
                cap_limit = int(cap) if cap is not None else int(self.heur.calendar_search_cap_limit_default)
            except Exception:
                cap_limit = int(self.heur.calendar_search_cap_limit_default)
            if not result_is_maybe_truncated(result=base_call.result, cap_limit=cap_limit):
                return []
            tmin = base_call.params.get("time_min")
            tmax = base_call.params.get("time_max")
            if not isinstance(tmin, str) or not isinstance(tmax, str):
                return []
            if not can_split_window(time_min=tmin, time_max=tmax, min_window_seconds=int(self.heur.calendar_window_split_min_window_seconds)):
                return []

            extra: List[Call] = []
            max_calls = max(0, int(self.heur.calendar_window_split_max_calls))
            if max_calls <= 1:
                return []
            budget = max_calls - 1  # we already executed base_call

            queue: List[Tuple[str, str]] = []
            parts = split_time_window(time_min=tmin, time_max=tmax)
            if parts is None:
                return []
            lmin, lmax, rmin, rmax = parts
            queue.append((lmin, lmax))
            queue.append((rmin, rmax))

            while queue and budget > 0:
                wmin, wmax = queue.pop(0)
                params = {**dict(base_call.params), "time_min": wmin, "time_max": wmax}
                call = Call(
                    phase_key=base_call.phase_key,
                    action_type=base_call.action_type,
                    qualified_tool_name=base_call.qualified_tool_name,
                    tool_name=base_call.tool_name,
                    domain=base_call.domain,
                    side_effect=base_call.side_effect,
                    cost=base_call.cost,
                    params=params,
                    raw_action=_call_string(base_call.qualified_tool_name, params),
                )
                self.env.execute(call)
                extra.append(call)
                budget -= 1

                if budget <= 1:
                    continue
                if not result_is_maybe_truncated(result=call.result, cap_limit=cap_limit):
                    continue
                if not can_split_window(
                    time_min=wmin,
                    time_max=wmax,
                    min_window_seconds=int(self.heur.calendar_window_split_min_window_seconds),
                ):
                    continue
                parts2 = split_time_window(time_min=wmin, time_max=wmax)
                if parts2 is None:
                    continue
                a1, a2, b1, b2 = parts2
                queue.append((a1, a2))
                queue.append((b1, b2))

            return extra

        for call in self._prune_check_calls(check_candidates):
            self.env.execute(call)
            check_calls.append(call)
            check_calls.extend(_maybe_split_calendar_search(call))

        # Derive bindings from check results (MatchJoin-style derivations).
        try:
            self._derive_check_bindings(check_calls=check_calls)
        except Exception:
            pass

        # Second pass: retry check templates that were skipped due to unresolved placeholders.
        # Many queries resolve placeholders like `{name_email}` only after executing earlier checks.
        for tmpl in list(pending_placeholder_checks):
            call = _apply_template_to_call(phase_key=check_phase, template=tmpl, bindings=self.env.bindings)
            if _has_unresolved_placeholders(call.params):
                continue
            self.env.execute(call)
            check_calls.append(call)

        try:
            self._derive_check_bindings(check_calls=check_calls)
        except Exception:
            pass

        # Second pass: expand check calls that require IDs discovered during earlier checks.
        for tmpl, id_key in pending_id_checks:
            ids = list(self.env.queues.get(id_key) or [])
            if not ids:
                continue
            idv = ids[0]
            call = _apply_template_to_call(
                phase_key=check_phase,
                template=tmpl,
                bindings=self.env.bindings,
                override_params={id_key: idv},
            )
            if _has_unresolved_placeholders(call.params):
                continue
            self.env.execute(call)
            check_calls.append(call)

        try:
            self._derive_check_bindings(check_calls=check_calls)
        except Exception:
            pass

        query_text = str(self.plan.get("query") or "")
        allowed_do_tools = allowed_do_qualified_tools(
            query=query_text,
            check_calls=[(c.qualified_tool_name, c.result) for c in check_calls],
        )

        executed: set[Tuple[str, str]] = set()
        for call in history:
            if call.raw_action and call.raw_action != "-":
                executed.add((_phase_kind(call.phase_key), call.action_type))
        eff_required = self._effective_required_actions(history=list(history) + list(check_calls))
        required_remaining = [ra for ra in eff_required if ra not in executed]
        do_required = any(k == "do" for (k, _) in eff_required)

        processed_ids: Dict[str, set[str]] = {}
        for call in history:
            if _phase_kind(call.phase_key) != "do":
                continue
            for key in _id_key_candidates():
                value = call.params.get(key)
                if isinstance(value, str) and value.strip():
                    processed_ids.setdefault(key, set()).add(value.strip())

        max_calls = max(0, int(self.max_do_calls_per_round))
        id_limits = max(1, int(self.max_ids_per_action))

        # Evidence from Check calls in this round (used to avoid extra Do on stale IDs).
        evidence: Dict[str, set[str]] = {}
        evidence_ordered: Dict[str, List[str]] = {}

        def _add_evidence(key: str, ids: Sequence[str]) -> None:
            if not key:
                return
            bucket = evidence.setdefault(key, set())
            ordered = evidence_ordered.setdefault(key, [])
            for v in ids:
                if not isinstance(v, str):
                    continue
                vv = v.strip()
                if not vv or vv in bucket:
                    continue
                bucket.add(vv)
                ordered.append(vv)

        for call in check_calls:
            key, ids = _extract_ids(call.result)
            if key and ids:
                _add_evidence(key, ids)
            for id_key in _id_key_candidates():
                v = call.params.get(id_key)
                if isinstance(v, str) and v.strip():
                    _add_evidence(id_key, [v.strip()])

        if allowed_do_tools is not None and len(allowed_do_tools) == 0:
            stop_action = RoundAction(key="stop", stop=True, payload={"check_calls": check_calls, "do_calls": []})
            return check_calls, [stop_action]

        # Enrich static Do OR slots based on current-round Check evidence.
        try:
            self._tree_enrich_do_candidates(evidence=dict(evidence_ordered), round_idx=round_idx)
        except Exception:
            pass

        def _dedupe_calls(calls: List[Call]) -> List[Call]:
            out: List[Call] = []
            seen: set[str] = set()
            for call in calls:
                raw = str(getattr(call, "raw_action", "") or "")
                if not raw or raw in seen:
                    continue
                seen.add(raw)
                out.append(call)
            return out

        actions: List[RoundAction] = []

        # Preferred branching: choose candidates per Do OR slot from the AND–OR tree.
        tree_do_slots = self._collect_tree_or_slots(phase_kind="do")
        if tree_do_slots:
            for slot_idx, slot in enumerate(tree_do_slots):
                qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
                qualified = str(qa.get("qualified_tool_name") or "").strip()
                if allowed_do_tools is not None and qualified and qualified not in allowed_do_tools:
                    continue
                action_type = str(qa.get("action_type") or "").strip() or (qualified.split(".", 1)[1] if "." in qualified else "")
                id_key = self._tree_slot_id_key(slot)

                candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
                calls: List[Call] = []
                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    call = self._tree_candidate_to_call(slot=slot, candidate=cand, phase_key=do_phase)
                    if call is None:
                        continue
                    if _has_unresolved_placeholders(call.params):
                        continue
                    if id_key:
                        v = call.params.get(id_key)
                        if isinstance(v, str) and v.strip():
                            ev = evidence.get(id_key)
                            if ev is not None and v.strip() not in ev:
                                continue
                            done = processed_ids.get(id_key) or set()
                            if v.strip() in done:
                                continue
                    else:
                        if ("do", call.action_type) in executed:
                            continue
                    calls.append(call)

                calls = _dedupe_calls(calls)
                if not calls:
                    continue

                # Per-slot action keys (stable enough for debugging).
                slot_key = str(slot.get("query_action_node_id") or slot.get("slot_edge") or f"slot_{slot_idx}")
                slot_key = re.sub(r"[^a-zA-Z0-9_:\\-]+", "_", slot_key)[:96]

                # Required-min: choose exactly one call for required action types.
                if ("do", action_type) in required_remaining:
                    actions.append(
                        RoundAction(
                            key=f"do_required_min::{slot_key}",
                            stop=False,
                            payload={"check_calls": check_calls, "do_calls": [calls[0]]},
                        )
                    )

                # Single-candidate actions (bounded by id_limits).
                for i, call in enumerate(calls[: max(1, int(id_limits))]):
                    suffix = ""
                    if id_key and isinstance(call.params.get(id_key), str):
                        suffix = str(call.params.get(id_key)).strip()
                    actions.append(
                        RoundAction(
                            key=f"do::{slot_key}::{i}{('::' + suffix) if suffix else ''}",
                            stop=False,
                            payload={"check_calls": check_calls, "do_calls": [call]},
                        )
                    )

                # Batch candidate action for this slot (bounded by max_calls when set).
                batch = calls if max_calls <= 0 else calls[: max_calls]
                if len(batch) > 1:
                    actions.append(
                        RoundAction(
                            key=f"do_batch::{slot_key}",
                            stop=False,
                            payload={"check_calls": check_calls, "do_calls": list(batch)},
                        )
                    )
        else:
            # Fallback: legacy template-based branching (older trees without OR slots).
            def _expand(template: ActionTemplate, *, limit: Optional[int]) -> List[Call]:
                id_key = _do_template_id_key(template)
                params = dict(template.params)
                if not id_key and ("do", template.action_type) in executed:
                    return []
                if id_key and id_key in params and not _contains_placeholder(params.get(id_key)):
                    fixed = params.get(id_key)
                    if isinstance(fixed, str) and fixed.strip():
                        ev = evidence.get(id_key)
                        if ev is not None and fixed.strip() not in ev:
                            return []
                if id_key and (id_key not in params or _contains_placeholder(params.get(id_key))):
                    ids = list(self.env.queues.get(id_key) or [])
                    ev = evidence.get(id_key)
                    if ev is not None:
                        ids = [idv for idv in ids if idv in ev]
                    done = processed_ids.get(id_key) or set()
                    if done:
                        ids = [idv for idv in ids if idv not in done]
                    if limit is not None:
                        ids = ids[: max(0, limit)]
                    ids = ids[:id_limits]
                    if not ids:
                        return []
                    calls: List[Call] = []
                    for idv in ids:
                        call = _apply_template_to_call(
                            phase_key=do_phase,
                            template=template,
                            bindings=self.env.bindings,
                            override_params={id_key: idv},
                        )
                        if _has_unresolved_placeholders(call.params):
                            continue
                        calls.append(call)
                    return calls
                call = _apply_template_to_call(phase_key=do_phase, template=template, bindings=self.env.bindings)
                if _has_unresolved_placeholders(call.params):
                    return []
                return [call]

            do_one: List[Call] = []
            do_all: List[Call] = []
            for tmpl in self.do_templates:
                if allowed_do_tools is not None and tmpl.qualified_tool_name not in allowed_do_tools:
                    continue
                do_one.extend(_expand(tmpl, limit=1))
                do_all.extend(_expand(tmpl, limit=None))
            do_one = _dedupe_calls(do_one)
            do_all = _dedupe_calls(do_all)
            if do_one:
                actions.append(RoundAction(key="do_one", stop=False, payload={"check_calls": check_calls, "do_calls": [do_one[0]]}))
            if do_all:
                batch = do_all if max_calls <= 0 else do_all[:max_calls]
                actions.append(RoundAction(key="do_all", stop=False, payload={"check_calls": check_calls, "do_calls": list(batch)}))

        # If no Do actions are currently available, allow a check-only round to progress bindings/queues.
        if not actions:
            actions.append(RoundAction(key="check_only", stop=False, payload={"check_calls": check_calls, "do_calls": []}))

        stop_sign_reached = self._stop_sign_reached(history=list(history) + list(check_calls))
        try:
            runs_i = int(self.plan.get("runs") or 0)
        except Exception:
            runs_i = 0
        if runs_i != 0 or stop_sign_reached:
            actions.append(RoundAction(key="stop", stop=True, payload={"check_calls": check_calls, "do_calls": []}))

        uniq: List[RoundAction] = []
        seen: set[Tuple[bool, Tuple[str, ...]]] = set()
        for act in actions:
            payload = act.payload if isinstance(act.payload, dict) else {}
            do_calls = payload.get("do_calls") if isinstance(payload.get("do_calls"), list) else []
            sig = (bool(act.stop), tuple(getattr(c, "raw_action", "") for c in do_calls))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(act)
        return check_calls, uniq

    def available_actions(self, *, history: List[Call], round_idx: int) -> List[RoundAction]:
        _, actions = self._available_round_actions(history=history, round_idx=round_idx)
        return list(actions)

    def apply(self, *, history: List[Call], round_idx: int, action: RoundAction) -> List[Call]:
        payload = action.payload if isinstance(action.payload, dict) else {}
        check_calls = payload.get("check_calls") if isinstance(payload.get("check_calls"), list) else None
        do_calls = payload.get("do_calls") if isinstance(payload.get("do_calls"), list) else []
        if check_calls is None:
            check_calls, _ = self._available_round_actions(history=history, round_idx=round_idx)

        self._replay(history)
        for call in check_calls:
            if isinstance(call, Call):
                self.env.execute(call)
        new_history = list(history) + [c for c in check_calls if isinstance(c, Call)]
        for call in do_calls:
            if isinstance(call, Call):
                self.env.execute(call)
                new_history.append(call)
        return new_history

    def rollout(self, *, history: List[Call], start_round: int, max_rounds: int, rng: random.Random) -> List[Call]:
        cur_history = list(history)
        for r in range(int(start_round), int(max_rounds) + 1):
            actions = self.available_actions(history=cur_history, round_idx=r)
            non_stop = [a for a in actions if not bool(a.stop)]
            # Prefer single-do-call actions in rollouts to avoid compounding side effects/cost.
            singles: List[RoundAction] = []
            if self.heur.rollout_prefer_single_do:
                for a in non_stop:
                    payload = a.payload if isinstance(a.payload, dict) else {}
                    do_calls = payload.get("do_calls") if isinstance(payload.get("do_calls"), list) else []
                    if len(do_calls) == 1:
                        singles.append(a)
            pool = singles or non_stop
            chosen = rng.choice(pool) if pool else RoundAction(key="stop", stop=True)
            cur_history = self.apply(history=cur_history, round_idx=r, action=chosen)
            if chosen.stop:
                break
            payload = chosen.payload if isinstance(chosen.payload, dict) else {}
            do_calls = payload.get("do_calls") if isinstance(payload.get("do_calls"), list) else []
        return cur_history

    def score(self, *, history: List[Call]) -> Tuple[float, Mapping[str, object]]:
        from task_helper.work.scoring import ActionRecord, compute_gt_free_checks, compute_tool_cost, hard_pass_from_checks

        self._replay(history)
        required_actions = self._effective_required_actions(history=history)
        records = [
            ActionRecord(
                phase_key=c.phase_key,
                raw_action=c.raw_action,
                qualified_tool_name=c.qualified_tool_name,
                side_effect=bool(c.side_effect),
                params=dict(c.params),
            )
            for c in history
        ]
        checks, semantic_rate = compute_gt_free_checks(records, required_actions=required_actions)
        selection_ok = self._calendar_selection_ok(history=history)
        if selection_ok is not None:
            checks["calendar_selection_rule"] = bool(selection_ok)
        tool_cost = compute_tool_cost(records, do_weight=1.0)

        # Extra GT-free constraints aimed at "no extra Do" and batch completeness.
        do_ids_supported = self._do_ids_supported_by_checks(history=history)
        checks["do_ids_supported_by_checks"] = bool(do_ids_supported)
        stop_sign_reached = self._stop_sign_reached(history=history)
        checks["stop_sign_reached"] = bool(stop_sign_reached)

        # Constraint satisfaction rate is the primary optimization objective.
        passed = sum(1 for v in checks.values() if v)
        semantic_rate = (passed / len(checks)) if checks else 1.0

        hard_pass = hard_pass_from_checks(checks) and (selection_ok is not False) and bool(do_ids_supported)
        try:
            runs_i = int(self.plan.get("runs") or 0)
        except Exception:
            runs_i = 0
        if runs_i == 0 and self._do_required(history=history) and not stop_sign_reached:
            hard_pass = False

        semantic_weight = float(self.cfg.get("semantic_weight") or 100.0)
        cost_weight = float(self.cfg.get("tool_cost_weight") or 1.0)
        reward = (semantic_weight * float(semantic_rate)) - (cost_weight * float(tool_cost))
        if not hard_pass:
            reward = float(self.hard_fail_penalty) + float(reward)

        details: Dict[str, object] = {
            "hard_pass": bool(hard_pass),
            "semantic_rate": float(semantic_rate),
            "tool_cost": float(tool_cost),
            "checks": dict(checks),
            "required_actions": list(required_actions),
            "stop_sign_reached": bool(stop_sign_reached),
        }
        return float(reward), details

    def render(self, *, history: List[Call], details: Mapping[str, object]) -> Mapping[str, object]:
        out = _render_filled_template(
            template=self.template,
            template_id=self.template_id,
            plan=self.plan,
            required_actions=self.base_required_actions,
            history=history,
            details=details,
            note_key="multi_round_mcts",
        )
        notes = out.get("notes") if isinstance(out.get("notes"), dict) else {}
        blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else None
        if isinstance(blob, dict):
            if self.export_mcts_tree and "mcts_tree" in details:
                blob["mcts_tree"] = details.get("mcts_tree")
            if self.export_enriched_tree:
                blob["tree_enriched"] = {"root": self._build_enriched_root(history=history)}
        return out


def _render_filled_template(
    *,
    template: Mapping[str, object],
    template_id: str,
    plan: Mapping[str, object],
    required_actions: Sequence[Tuple[str, str]],
    history: Sequence[Call],
    details: Mapping[str, object],
    note_key: str,
) -> Mapping[str, object]:
    round_count = 0
    for call in history:
        if _phase_kind(call.phase_key) == "check":
            try:
                round_count = max(round_count, int(call.phase_key.split("_", 1)[1]))
            except Exception:
                continue
    if round_count <= 0:
        round_count = 1 if history else 0

    out_phases: List[Dict[str, object]] = []
    out_actions: List[Dict[str, object]] = []
    for r in range(1, round_count + 1):
        out_phases.append({"phase_key": _phase_key("check", r), "phase_type": "CheckRound", "phase_index": (2 * r - 1), "attrs": {}})
        out_phases.append({"phase_key": _phase_key("do", r), "phase_type": "DoRound", "phase_index": (2 * r), "attrs": {}})

    order_counters: Dict[str, int] = {}
    for call in history:
        order = order_counters.get(call.phase_key, 0)
        order_counters[call.phase_key] = order + 1
        out_actions.append(
            {
                "phase_key": call.phase_key,
                "action_type": call.action_type,
                "params": dict(call.params),
                "attrs": {
                    "tool_name": call.tool_name,
                    "qualified_tool_name": call.qualified_tool_name,
                    "side_effect": bool(call.side_effect),
                    "raw_action": call.raw_action,
                },
                "order_index": order,
            }
        )

    errors: List[Dict[str, str]] = []
    try:
        env = WorkEnv()
        env.replay(list(history))
        errors = list(env.errors)
    except Exception:
        errors = []

    errors_sample = errors[:3]

    notes = {
        note_key: {
            "reward": float(details.get("reward", 0.0)) if isinstance(details, dict) else 0.0,
            "mcts_iterations_used": int(details.get("mcts_iterations_used")) if isinstance(details, dict) and isinstance(details.get("mcts_iterations_used"), (int, float)) else None,
            "mcts_iterations_total": int(details.get("mcts_iterations_total")) if isinstance(details, dict) and isinstance(details.get("mcts_iterations_total"), (int, float)) else None,
            "mcts_early_stop": bool(details.get("mcts_early_stop")) if isinstance(details, dict) and isinstance(details.get("mcts_early_stop"), bool) else None,
            "rounds": int(round_count),
            "total_calls": int(len(history)),
            "tool_cost": float(details.get("tool_cost", 0.0)),
            "hard_pass": bool(details.get("hard_pass", False)),
            "semantic_rate": float(details.get("semantic_rate", 0.0)),
            "semantic_checks": dict(details.get("checks", {})) if isinstance(details, dict) else {},
            "required_actions": list(details.get("required_actions", required_actions)) if isinstance(details, dict) else list(required_actions),
            "errors_count": int(len(errors)),
            "errors_sample": errors_sample,
        }
    }

    return {
        "version": int(template.get("version") or 1),
        "template_id": template_id,
        "task_name": str(template.get("task_name") or "work"),
        "plan": dict(plan),
        "phases": out_phases,
        "actions": out_actions,
        "notes": notes,
    }


def make_adapter(
    *,
    tree: Optional[Mapping[str, object]] = None,
    template: Optional[Mapping[str, object]] = None,
    match_entry: Optional[Mapping[str, object]] = None,
    config: Optional[Mapping[str, object]] = None,
) -> MultiRoundAdapter:
    if tree is None:
        tree = {
            "template": dict(template or {}),
            "match": dict(match_entry or {}),
        }
    return WorkMultiRoundAdapter(tree=tree, config=dict(config or {}))


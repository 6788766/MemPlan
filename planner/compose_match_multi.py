"""
ComposeMatch (multi): materialize AND–OR trees with a MatchJoin-style fixpoint.

The previous multi-mode runner emitted a lightweight payload (`{template, match}`) and relied on
`planner.twin_track_multi` to do online tool execution. For WorkBench, many templates depend on
placeholders that must be resolved by executing Check tools (e.g., lookup email -> search -> get IDs)
before Do candidates can be enumerated.

This entrypoint upgrades multi ComposeMatch to emit a static AND–OR tree compatible with
`planner.twin_track.py` (same schema as `planner.compose_match`), while keeping the original
`{template, match}` fields for backwards compatibility.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from planner.match_join import is_whole_placeholder, placeholder_name, resolve_placeholder_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"
OUTPUT_ROOT = ARTIFACTS_ROOT / "output"

_PLACEHOLDER_ANY_RE = re.compile(r"\{[^{}]+\}")

_MONTH_DAY_RE = re.compile(r"^(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?$", re.I)

_WEEKDAY_INDEX: Dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

_DATE_DOMAINS: set[str] = {"project_management", "customer_relationship_manager", "analytics", "crm"}
_DATETIME_DOMAINS: set[str] = {"email", "calendar"}

_DATE_KEYS: set[str] = {
    "time_min",
    "time_max",
    "date_min",
    "date_max",
    "due_date",
    "event_start",
    "last_contact_date",
    "last_contact_date_min",
    "last_contact_date_max",
    "follow_up_by",
    "follow_up_by_min",
    "follow_up_by_max",
}


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _quote(value: object) -> str:
    text = str(value) if value is not None else ""
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def _call_string(qualified: str, kwargs: Mapping[str, object]) -> str:
    parts = [f"{k}={_quote(v)}" for k, v in kwargs.items() if v is not None]
    return f"{qualified}.func(" + ", ".join(parts) + ")"


def _json_safe(value: object, *, depth: int = 0, max_depth: int = 8) -> object:
    """
    Best-effort conversion to JSON-serializable objects for tool outputs.

    Multi-mode ComposeMatch runs tools inline (Check only). Persisting outputs helps
    debug placeholder/binding resolution without changing planning logic.
    """

    if depth >= max_depth:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, depth=depth + 1, max_depth=max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth=depth + 1, max_depth=max_depth) for v in list(value)]
    return str(value)


def _derive_time_defaults(task: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Best-effort time defaults for placeholder filling.

    WorkBench tools ship with `HARDCODED_CURRENT_TIME` under `task_helper.<task>.tools.constants`.
    Other tasks may omit it; in that case we return (None, None).
    """

    module_name = f"task_helper.{str(task).strip()}.tools.constants"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None, None

    raw = getattr(module, "HARDCODED_CURRENT_TIME", None)
    if raw is None:
        return None, None
    try:
        dt = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None, None
    time_max = (dt.date() - timedelta(days=1)).strftime("%Y-%m-%d")
    return time_max, int(dt.year)


def _replace_known_inline_placeholders(text: str, *, default_year: Optional[int], time_max_default: Optional[str]) -> str:
    # Keep intentionally minimal (mirrors Work multi-round adapter behaviour).
    out = str(text)
    if default_year is not None:
        out = out.replace("{year}", str(default_year))
    if time_max_default is not None:
        out = out.replace("{time_max}", str(time_max_default))
    return out


def _derive_current_datetime(task: str) -> Optional[datetime]:
    module_name = f"task_helper.{str(task).strip()}.tools.constants"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    raw = getattr(module, "HARDCODED_CURRENT_TIME", None)
    if raw is None:
        return None
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _normalize_date_token(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    for suffix in ("_date", "_day", "_datetime"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def _parse_month_day(text: str, *, default_year: int) -> Optional[datetime]:
    raw = str(text or "").strip()
    m = _MONTH_DAY_RE.match(raw)
    if not m:
        return None
    month_name = str(m.group("month") or "").strip().lower()
    day = int(m.group("day"))
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
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    month = months.get(month_name)
    if month is None:
        return None
    try:
        return datetime(default_year, int(month), int(day), 0, 0, 0)
    except Exception:
        return None


def _parse_datetime_loose(text: str) -> Optional[datetime]:
    raw = str(text or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _start_of_week(day: datetime) -> datetime:
    base = datetime(day.year, day.month, day.day, 0, 0, 0)
    return base - timedelta(days=base.weekday())


def _resolve_relative_date(
    *,
    token: str,
    now_dt: datetime,
    default_year: int,
    for_key: str,
    domain: str,
) -> Optional[datetime]:
    """
    Resolve common relative date tokens to a concrete datetime (00:00:00 baseline).

    The caller formats the datetime to either date-only or datetime strings depending on domain.
    """

    key = str(for_key or "").strip().lower()
    want_min = key.endswith("_min") or key in {"time_min", "date_min"}
    want_max = key.endswith("_max") or key in {"time_max", "date_max"}

    norm = _normalize_date_token(token)
    if not norm:
        return None

    base_date = datetime(now_dt.year, now_dt.month, now_dt.day, 0, 0, 0)
    yesterday = base_date - timedelta(days=1)
    weekday_ref = base_date
    if str(domain or "").strip().lower() in {"analytics", "email", "customer_relationship_manager", "crm"}:
        weekday_ref = yesterday

    # Generic placeholders.
    if norm in {"time_min", "date_min"}:
        return datetime(int(default_year), 1, 1, 0, 0, 0)
    if norm in {"time_max", "date_max"}:
        return yesterday

    # Relative days.
    if norm == "today":
        return base_date
    if norm == "yesterday":
        return yesterday
    if norm == "tomorrow":
        return base_date + timedelta(days=1)

    # Week ranges.
    if norm in {"last_week", "previous_week"}:
        start = _start_of_week(base_date) - timedelta(days=7)
        end = start + timedelta(days=6)
        return start if want_min else end if want_max else end
    if norm == "this_week":
        start = _start_of_week(base_date)
        end = base_date - timedelta(days=1)
        if end < start:
            end = start
        return start if want_min else end if want_max else end
    if norm == "next_week":
        start = _start_of_week(base_date) + timedelta(days=7)
        end = start + timedelta(days=6)
        return start if want_min else end if want_max else start
    if norm in {"last_7_days", "past_7_days"}:
        start = base_date - timedelta(days=7)
        end = base_date - timedelta(days=1)
        return start if want_min else end if want_max else end

    # Weekdays.
    prefix: Optional[str] = None
    day_token = norm
    for p in ("previous_", "last_", "next_"):
        if norm.startswith(p):
            prefix = p[:-1]  # previous/last/next
            day_token = norm[len(p) :]
            break
    weekday = _WEEKDAY_INDEX.get(day_token)
    if weekday is not None:
        if prefix in {"previous", "last"}:
            base = _resolve_relative_date(token=day_token, now_dt=now_dt, default_year=default_year, for_key=for_key, domain=domain)
            return (base - timedelta(days=7)) if base is not None else None
        if prefix == "next":
            base = _resolve_relative_date(token=day_token, now_dt=now_dt, default_year=default_year, for_key=for_key, domain=domain)
            return (base + timedelta(days=7)) if base is not None else None

        cur = weekday_ref.weekday()
        if str(domain or "").strip().lower() in {"calendar", "project_management"}:
            # Bare weekday for scheduling-like domains: next occurrence (including today).
            ahead = (weekday - cur) % 7
            return weekday_ref + timedelta(days=ahead)

        # Otherwise, treat bare weekday as the most recent occurrence at-or-before anchor.
        back = (cur - weekday) % 7
        return weekday_ref - timedelta(days=back)

    # Natural month/day (e.g., "November 21") with year inferred from deterministic now.
    dt = _parse_month_day(token, default_year=int(default_year))
    if dt is not None:
        return dt
    return None


def _format_date_value(*, dt: datetime, want: str, for_key: str) -> str:
    if want == "date":
        return dt.strftime("%Y-%m-%d")
    key = str(for_key or "").strip().lower()
    want_max = key.endswith("_max") or key in {"time_max", "date_max"}
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and want_max:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def date_correct(*, task: str, template: Mapping[str, object]) -> Dict[str, object]:
    """
    One-pass normalization for date/time params inside a template.

    - Resolves relative tokens like {friday}/{previous_friday}/{yesterday}/{last_week} using the
      deterministic `HARDCODED_CURRENT_TIME` for the task (when available).
    - Ensures non-empty date/time values match expected formats:
        * analytics/project_management/crm: YYYY-MM-DD
        * email/calendar: YYYY-MM-DD HH:MM:SS
    """

    now_dt = _derive_current_datetime(task)
    if now_dt is None:
        return dict(template)

    time_max_default, default_year = _derive_time_defaults(task)
    default_year_i = int(default_year or now_dt.year)

    out: Dict[str, object] = dict(template)
    actions_raw = out.get("actions") if isinstance(out.get("actions"), list) else []
    actions: List[object] = []

    for action in actions_raw:
        if not isinstance(action, dict):
            actions.append(action)
            continue
        action_out: Dict[str, object] = dict(action)
        attrs = action_out.get("attrs") if isinstance(action_out.get("attrs"), dict) else {}
        tool_name = str(attrs.get("tool_name") or "").strip()
        qualified = str(attrs.get("qualified_tool_name") or "").strip()
        domain = tool_name or (qualified.split(".", 1)[0] if "." in qualified else "")
        dom = domain.strip().lower()
        if dom in _DATE_DOMAINS:
            want = "date"
        elif dom in _DATETIME_DOMAINS:
            want = "datetime"
        else:
            actions.append(action_out)
            continue

        params_raw = action_out.get("params") if isinstance(action_out.get("params"), dict) else {}
        params_out: Dict[str, object] = dict(params_raw)
        for key, raw in list(params_out.items()):
            if raw is None or key not in _DATE_KEYS:
                continue
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue

            # Whole placeholder (e.g., "{friday}" / "{previous_friday_date}").
            token = placeholder_name(text) if is_whole_placeholder(text) else text
            resolved = _resolve_relative_date(
                token=token, now_dt=now_dt, default_year=default_year_i, for_key=str(key), domain=dom
            )
            if resolved is not None:
                params_out[key] = _format_date_value(dt=resolved, want=want, for_key=str(key))
                continue

            # Format correction for already-concrete dates/datetimes.
            dt = _parse_datetime_loose(text)
            if dt is not None:
                params_out[key] = _format_date_value(dt=dt, want=want, for_key=str(key))
                continue

            # Special-case: date_max/time_max placeholders with deterministic default.
            if is_whole_placeholder(text) and placeholder_name(text) in {"time_max", "date_max"} and time_max_default is not None:
                dt2 = _parse_datetime_loose(str(time_max_default))
                if dt2 is not None:
                    params_out[key] = _format_date_value(dt=dt2, want=want, for_key=str(key))

        action_out["params"] = params_out
        actions.append(action_out)

    out["actions"] = actions
    return out


def _normalize_enum_key(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonicalize_enum(value: object, mapping: Mapping[str, str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _PLACEHOLDER_ANY_RE.search(text):
        return None
    key = _normalize_enum_key(text)
    resolved = mapping.get(key) or mapping.get(key.replace(" ", "_"))
    return str(resolved) if resolved else None


def _is_disjunction_status_placeholder(value: object) -> bool:
    if not isinstance(value, str) or not is_whole_placeholder(value):
        return False
    name = _normalize_date_token(placeholder_name(value))
    return name in {
        "qualified_or_in_proposal",
        "qualified_or_proposal",
        "qualified_or_inproposal",
    }

def _load_domain_corrections(task: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load domain-specific parameter canonicalization rules.

    Expected file: `artifacts/input/<task>/domain_corrections.json` with shape:
      { "<domain>": { "<field>": { "<raw>": "<canonical>", ... }, ... }, ... }
    """

    path = INPUT_ROOT / str(task).strip() / "domain_corrections.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for domain, fields_raw in payload.items():
        if not isinstance(fields_raw, dict):
            continue
        fields: Dict[str, Dict[str, str]] = {}
        for field, mapping_raw in fields_raw.items():
            if not isinstance(mapping_raw, dict):
                continue
            mapping: Dict[str, str] = {}
            for k, v in mapping_raw.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                kk = k.strip()
                vv = v.strip()
                if not kk or not vv:
                    continue
                mapping[kk] = vv
            if mapping:
                fields[str(field).strip()] = mapping
        if fields:
            out[str(domain).strip().lower()] = fields
    return out


def _domain_rules_for(domain: str, corrections: Mapping[str, Mapping[str, Mapping[str, str]]]) -> Mapping[str, Mapping[str, str]]:
    dom = str(domain or "").strip().lower()
    if dom in corrections:
        return corrections[dom]
    if dom == "crm" and "customer_relationship_manager" in corrections:
        return corrections["customer_relationship_manager"]
    return {}


def domain_correct(
    *, template: Mapping[str, object], corrections: Mapping[str, Mapping[str, Mapping[str, str]]]
) -> Dict[str, object]:
    """
    Domain-specific parameter normalization for constrained enum-like fields.
    """

    out: Dict[str, object] = dict(template)
    actions_raw = out.get("actions") if isinstance(out.get("actions"), list) else []
    actions_out: List[object] = []

    for action in actions_raw:
        if not isinstance(action, dict):
            actions_out.append(action)
            continue

        action_obj: Dict[str, object] = dict(action)
        attrs = action_obj.get("attrs") if isinstance(action_obj.get("attrs"), dict) else {}
        qualified = str(attrs.get("qualified_tool_name") or "").strip()
        tool_name = str(attrs.get("tool_name") or "").strip()
        domain = (tool_name or (qualified.split(".", 1)[0] if "." in qualified else "")).strip().lower()

        params_raw = action_obj.get("params") if isinstance(action_obj.get("params"), dict) else {}
        params = dict(params_raw)

        rules = _domain_rules_for(domain, corrections)

        if domain in {"customer_relationship_manager", "crm"}:
            status_map = rules.get("status") or {}
            product_interest_map = rules.get("product_interest") or {}
            status = params.get("status")
            if _is_disjunction_status_placeholder(status) and str(action_obj.get("action_type") or "").strip() == "search_customers":
                base = dict(action_obj)
                base_params = dict(params)
                base_params["status"] = "Qualified"
                base["params"] = base_params

                other = dict(action_obj)
                other_params = dict(params)
                other_params["status"] = "Proposal"
                other["params"] = other_params

                for obj in (base, other):
                    p = obj.get("params") if isinstance(obj.get("params"), dict) else {}
                    canon_pi = _canonicalize_enum(p.get("product_interest"), product_interest_map)
                    if canon_pi:
                        p["product_interest"] = canon_pi
                    obj["params"] = p

                actions_out.append(base)
                actions_out.append(other)
                continue

            canon_status = _canonicalize_enum(status, status_map)
            if canon_status:
                params["status"] = canon_status
            canon_pi = _canonicalize_enum(params.get("product_interest"), product_interest_map)
            if canon_pi:
                params["product_interest"] = canon_pi

            field = str(params.get("field") or "").strip().lower()
            if field == "status":
                canon = _canonicalize_enum(params.get("new_value"), status_map)
                if canon:
                    params["new_value"] = canon
            if field == "product_interest":
                canon = _canonicalize_enum(params.get("new_value"), product_interest_map)
                if canon:
                    params["new_value"] = canon

        if domain == "project_management":
            list_name_map = rules.get("list_name") or {}
            board_map = rules.get("board") or {}
            canon_list = _canonicalize_enum(params.get("list_name"), list_name_map)
            if canon_list:
                params["list_name"] = canon_list
            canon_board = _canonicalize_enum(params.get("board"), board_map)
            if canon_board:
                params["board"] = canon_board

            field = str(params.get("field") or "").strip().lower()
            if field == "list_name":
                canon = _canonicalize_enum(params.get("new_value"), list_name_map)
                if canon:
                    params["new_value"] = canon
            if field == "board":
                canon = _canonicalize_enum(params.get("new_value"), board_map)
                if canon:
                    params["new_value"] = canon

        # Generic: apply per-domain canonicalizations for any field directly present in params.
        for field_name, mapping in (rules or {}).items():
            if not isinstance(field_name, str) or not isinstance(mapping, Mapping):
                continue
            if field_name not in params:
                continue
            canon = _canonicalize_enum(params.get(field_name), mapping)  # type: ignore[arg-type]
            if canon:
                params[field_name] = canon

        action_obj["params"] = params
        actions_out.append(action_obj)

    out["actions"] = actions_out
    return out


class _PlaceholderUF:
    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, name: str) -> str:
        key = str(name or "").strip()
        if not key:
            return ""
        parent = self._parent.get(key)
        if parent is None:
            self._parent[key] = key
            return key
        if parent == key:
            return key
        root = self.find(parent)
        self._parent[key] = root
        return root

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if not ra or not rb or ra == rb:
            return
        # Deterministic union: lexicographically smaller root wins.
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb


def _contains_unresolved_placeholders(params: Mapping[str, object]) -> bool:
    for value in params.values():
        if isinstance(value, str) and _PLACEHOLDER_ANY_RE.search(value):
            return True
    return False


def _normalize_match_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _call_matches_query_params(*, query_params: Mapping[str, object], call_args: Mapping[str, object]) -> bool:
    """
    Decide whether an executed tool call is a plausible candidate for a query slot.

    If the query specifies a concrete value (i.e., not None and contains no placeholders)
    for a param key, the call must match that value. Placeholder-containing query values
    are treated as wildcards.
    """

    for key, raw in query_params.items():
        if raw is None:
            continue
        if isinstance(raw, str) and _PLACEHOLDER_ANY_RE.search(raw):
            continue
        if key not in call_args:
            # Treat missing optional args as empty/None when the query wants an empty string.
            # WorkBench teachers sometimes omit defaulted params (e.g., calendar.search_events.query).
            if isinstance(raw, str) and raw == "":
                continue
            return False
        if _normalize_match_value(call_args.get(key)) != _normalize_match_value(raw):
            return False
    return True


def _id_key_candidates() -> Tuple[str, ...]:
    return ("customer_id", "email_id", "task_id", "event_id")


def _extract_ids(result: object) -> Tuple[Optional[str], List[str]]:
    if not isinstance(result, list) or not result:
        return None, []
    if not isinstance(result[0], dict):
        return None, []
    for key in _id_key_candidates():
        values = [str(item.get(key) or "").strip() for item in result if isinstance(item, dict)]
        values = [v for v in values if v]
        if values:
            return key, values
    return None, []


@dataclass(frozen=True)
class ToolViewInfo:
    view_id: str
    tool: str
    cost: float
    action_type: str
    qualified_tool_name: str
    side_effect: bool
    params: Dict[str, object]
    cap_limit: Optional[int] = None


def _load_tool_views(path: Path) -> Tuple[Dict[str, ToolViewInfo], Dict[str, ToolViewInfo]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = payload.get("views") if isinstance(payload, dict) else None
    if not isinstance(views, list):
        return {}, {}

    by_view_id: Dict[str, ToolViewInfo] = {}
    by_qualified: Dict[str, ToolViewInfo] = {}
    for item in views:
        if not isinstance(item, dict):
            continue
        view_id = str(item.get("view_id") or "").strip()
        tool = str(item.get("tool") or "").strip()
        try:
            cost = float(item.get("cost") or 0.0)
        except Exception:
            cost = 0.0

        node = item.get("node_pattern") if isinstance(item.get("node_pattern"), dict) else {}
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        action_type = str(attrs.get("action_type") or "").strip()
        qualified = str(attrs.get("qualified_tool_name") or "").strip() or (f"{tool}.{action_type}" if tool and action_type else "")
        side_effect = bool(attrs.get("side_effect")) if isinstance(attrs.get("side_effect"), bool) else False
        params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        cap_limit_raw = attrs.get("cap_limit")
        try:
            cap_limit = int(cap_limit_raw) if cap_limit_raw is not None else None
        except Exception:
            cap_limit = None

        if not view_id or not qualified:
            continue
        info = ToolViewInfo(
            view_id=view_id,
            tool=tool or qualified.split(".", 1)[0],
            cost=float(cost),
            action_type=action_type or qualified.split(".", 1)[1] if "." in qualified else "",
            qualified_tool_name=qualified,
            side_effect=bool(side_effect),
            params=dict(params),
            cap_limit=cap_limit,
        )
        by_view_id[view_id] = info
        by_qualified[qualified] = info
    return by_view_id, by_qualified


def _load_memory_views(path: Path) -> Dict[str, Dict[str, object]]:
    views: Dict[str, Dict[str, object]] = {}
    for obj in _iter_jsonl(path):
        plan_id = str(obj.get("plan_id") or obj.get("id") or "").strip()
        if not plan_id:
            continue
        views[plan_id] = obj
    return views


def _load_memory_graph(path: Path) -> Dict[str, Dict[str, object]]:
    graphs: Dict[str, Dict[str, object]] = {}
    for obj in _iter_jsonl(path):
        plan_id = str(obj.get("plan_id") or obj.get("id") or "").strip()
        if not plan_id:
            continue
        graphs[plan_id] = obj
    return graphs


def _base_template_from_graph_payload(payload: Mapping[str, object]) -> Optional[str]:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("type") or "") != "Plan":
            continue
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        base = attrs.get("base_template")
        if isinstance(base, str) and base.strip():
            return base.strip()
    return None


@dataclass(frozen=True)
class MemoryGraphIndex:
    graphs: Dict[str, Dict[str, object]]
    by_base_template: Dict[str, List[str]]


def _index_memory_graph(graphs: Mapping[str, Mapping[str, object]]) -> MemoryGraphIndex:
    by_base_template: Dict[str, List[str]] = {}
    out_graphs: Dict[str, Dict[str, object]] = {}
    for plan_id, payload in graphs.items():
        if not isinstance(payload, Mapping):
            continue
        out_graphs[str(plan_id)] = dict(payload)
        base = _base_template_from_graph_payload(payload)
        if not base:
            continue
        by_base_template.setdefault(base, []).append(str(plan_id))
    for key in list(by_base_template.keys()):
        by_base_template[key] = sorted(set(by_base_template[key]))
    return MemoryGraphIndex(graphs=out_graphs, by_base_template=by_base_template)


@dataclass(frozen=True)
class ActionTemplate:
    phase_key: str
    order_index: int
    action_type: str
    tool_name: str
    qualified_tool_name: str
    side_effect: bool
    params: Dict[str, object]


def _action_template_from_query_action(action: Mapping[str, object]) -> Optional[ActionTemplate]:
    phase_key = str(action.get("phase_key") or "").strip()
    action_type = str(action.get("action_type") or "").strip()
    if not phase_key or not action_type:
        return None
    attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
    tool_name = str(attrs.get("tool_name") or "").strip()
    qualified = str(attrs.get("qualified_tool_name") or "").strip()
    if not qualified and tool_name and action_type:
        qualified = f"{tool_name}.{action_type}"
    if not tool_name and qualified and "." in qualified:
        tool_name = qualified.split(".", 1)[0]
    side_effect = bool(attrs.get("side_effect")) if isinstance(attrs.get("side_effect"), bool) else False
    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    try:
        order_index = int(action.get("order_index") or 0)
    except Exception:
        order_index = 0
    return ActionTemplate(
        phase_key=phase_key,
        order_index=order_index,
        action_type=action_type,
        tool_name=tool_name,
        qualified_tool_name=qualified,
        side_effect=bool(side_effect),
        params=dict(params),
    )


def _action_templates_from_memory_view(obj: Mapping[str, object]) -> List[ActionTemplate]:
    nodes = obj.get("nodes") if isinstance(obj.get("nodes"), list) else []
    out: List[ActionTemplate] = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("type") != "Action":
            continue
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        action_type = str(attrs.get("action_type") or "").strip()
        qualified = str(attrs.get("qualified_tool_name") or "").strip()
        tool_name = str(attrs.get("tool_name") or "").strip()
        side_effect = bool(attrs.get("side_effect")) if isinstance(attrs.get("side_effect"), bool) else False
        params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        if not action_type or not qualified:
            continue
        phase_key = "do" if side_effect else "check"
        out.append(
            ActionTemplate(
                phase_key=phase_key,
                order_index=0,
                action_type=action_type,
                tool_name=tool_name or qualified.split(".", 1)[0],
                qualified_tool_name=qualified,
                side_effect=bool(side_effect),
                params=dict(params),
            )
        )
    return out


@dataclass
class ExecutedCall:
    qualified_tool_name: str
    tool_name: str
    action_type: str
    args: Dict[str, object]
    result: object
    tool_view_id: str
    tool_call_cost: float
    side_effect: bool


class MatchJoinMaterializer:
    def __init__(
        self,
        *,
        task: str,
        tool_views_by_qualified: Mapping[str, ToolViewInfo],
        placeholder_uf: _PlaceholderUF,
        seed_bindings: Mapping[str, object],
        max_id_expansions: int = 5,
    ) -> None:
        self.task = str(task)
        self.tool_views_by_qualified = dict(tool_views_by_qualified)
        self.uf = placeholder_uf
        self.bindings: Dict[str, object] = dict(seed_bindings)
        self.time_max_default, self.default_year = _derive_time_defaults(self.task)
        self.queues: Dict[str, List[str]] = {}
        self.max_id_expansions = max(1, int(max_id_expansions))
        self.executed: Dict[str, List[ExecutedCall]] = {}
        self._executed_sigs: set[Tuple[str, Tuple[Tuple[str, str], ...]]] = set()

    def _binding_lookup(self, name: str) -> Optional[object]:
        key = str(name or "").strip()
        if not key:
            return None
        root = self.uf.find(key)
        if root and root in self.bindings:
            return self.bindings[root]
        # Fallback: accept raw key if present (covers arithmetic placeholders / non-unioned keys).
        if key in self.bindings:
            return self.bindings[key]
        return None

    def _set_binding(self, name: str, value: object) -> bool:
        key = str(name or "").strip()
        if not key or value is None:
            return False
        root = self.uf.find(key) or key
        existing = self.bindings.get(root)
        if existing == value:
            return False
        if existing is not None and existing != value:
            # Do not overwrite conflicting bindings (keep first).
            return False
        self.bindings[root] = value
        return True

    def _append_queue(self, key: str, values: Sequence[str]) -> bool:
        k = str(key or "").strip()
        if not k:
            return False
        existing = list(self.queues.get(k) or [])
        seen = set(existing)
        new = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip() and str(v).strip() not in seen]
        if not new:
            return False
        self.queues[k] = existing + new
        return True

    def _resolve_params(self, qualified: str, params: Mapping[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key, raw in params.items():
            if raw is None:
                continue
            if isinstance(raw, str):
                text = raw.strip()
                if "{" in text and "}" in text:
                    if is_whole_placeholder(text):
                        name = placeholder_name(text)
                        resolved = self._binding_lookup(name)
                        if resolved is not None:
                            text = str(resolved)
                        elif name == "time_min" and self.default_year is not None:
                            text = f"{int(self.default_year)}-01-01 00:00:00"
                        elif name == "time_max" and self.time_max_default is not None:
                            text = str(self.time_max_default)
                        elif name == "year" and self.default_year is not None:
                            text = str(int(self.default_year))
                        else:
                            # Arithmetic placeholders like foo_plus_3_days.
                            maybe = resolve_placeholder_value(name, self.bindings)
                            if maybe is not None:
                                text = str(maybe)
                            else:
                                out[key] = text
                                continue
                    else:
                        text = _replace_known_inline_placeholders(
                            text, default_year=self.default_year, time_max_default=self.time_max_default
                        )
                out[key] = text
            else:
                out[key] = raw

        # If template mentions time_max but it's missing/unresolved and we have a default, fill it.
        if "time_max" in params and "time_max" not in out and self.time_max_default is not None:
            out["time_max"] = str(self.time_max_default)
        return out

    def _execute_tool(self, qualified: str, kwargs: Mapping[str, object]) -> object:
        tool_name, fn_name = qualified.split(".", 1)
        module = importlib.import_module(f"task_helper.{self.task}.tools.{tool_name}")
        tool = getattr(module, fn_name, None)
        if tool is None:
            raise RuntimeError(f"Unknown tool: {qualified}")
        if hasattr(tool, "func"):
            return tool.func(**kwargs)  # type: ignore[misc]
        if callable(tool):
            return tool(**kwargs)  # type: ignore[misc]
        raise RuntimeError(f"Unknown tool callable: {qualified}")

    def _ingest_result(self, *, call_args: Mapping[str, object], result: object) -> bool:
        changed = False
        if isinstance(result, dict) and result:
            for key, value in result.items():
                if value is None:
                    continue
                changed |= self._set_binding(str(key), value)
                if isinstance(value, str) and "@" in value and "email" in str(key).lower():
                    changed |= self._set_binding("email", value)
            return changed

        if isinstance(result, list) and result and all(isinstance(v, str) for v in result) and any("@" in str(v) for v in result):
            addrs = [str(v) for v in result if isinstance(v, str)]
            atlas = [a for a in addrs if "@atlas" in a]
            chosen = atlas[0] if atlas else addrs[0]
            changed |= self._set_binding("email", chosen)
            name = call_args.get("name")
            if isinstance(name, str) and name.strip():
                changed |= self._set_binding(f"{name.strip().lower()}_email", chosen)

        key, ids = _extract_ids(result)
        if key and ids:
            changed |= self._append_queue(key, ids)
            # Also expose the first ID as a binding for placeholder-only templates.
            if ids:
                changed |= self._set_binding(key, ids[0])
        return changed

    def execute_check_fixpoint(self, templates: Sequence[ActionTemplate]) -> None:
        # Only execute non-side-effect tools.
        check_templates = [t for t in templates if not bool(t.side_effect)]

        # Deterministic ordering to make runs reproducible.
        check_templates.sort(key=lambda t: (t.phase_key, int(t.order_index), t.qualified_tool_name))

        changed = True
        while changed:
            changed = False
            for tmpl in check_templates:
                qualified = tmpl.qualified_tool_name
                if not qualified or "." not in qualified:
                    continue
                # Expand ID placeholders if we already discovered IDs from earlier checks.
                expanded_param_sets: List[Dict[str, object]] = []
                resolved_base = self._resolve_params(qualified, tmpl.params)

                id_keys = [k for k in _id_key_candidates() if k in resolved_base and is_whole_placeholder(str(resolved_base[k]))]
                if id_keys:
                    # Only expand the first ID key found (Work templates tend to have 0/1 ID params).
                    id_key = id_keys[0]
                    queue = list(self.queues.get(id_key) or [])
                    queue = queue[: self.max_id_expansions]
                    for idv in queue:
                        params = dict(resolved_base)
                        params[id_key] = idv
                        expanded_param_sets.append(params)
                else:
                    expanded_param_sets.append(dict(resolved_base))

                for resolved in expanded_param_sets:
                    if _contains_unresolved_placeholders(resolved):
                        continue
                    sig = (qualified, tuple(sorted((k, str(v)) for k, v in resolved.items() if v is not None)))
                    if sig in self._executed_sigs:
                        continue
                    self._executed_sigs.add(sig)

                    try:
                        result = self._execute_tool(qualified, resolved)
                    except Exception:
                        # Treat tool failures as non-contributing; continue exploring other checks.
                        continue

                    info = self.tool_views_by_qualified.get(qualified)
                    if info is None:
                        tool_name = qualified.split(".", 1)[0]
                        action_type = qualified.split(".", 1)[1]
                        tool_view_id = f"tool::{tool_name}::{action_type}"
                        tool_call_cost = 0.0
                        side_effect = False
                    else:
                        tool_name = info.tool
                        action_type = info.action_type
                        tool_view_id = info.view_id
                        tool_call_cost = float(info.cost)
                        side_effect = bool(info.side_effect)

                    call = ExecutedCall(
                        qualified_tool_name=qualified,
                        tool_name=tool_name,
                        action_type=action_type,
                        args=dict(resolved),
                        result=result,
                        tool_view_id=tool_view_id,
                        tool_call_cost=tool_call_cost,
                        side_effect=bool(side_effect),
                    )
                    self.executed.setdefault(qualified, []).append(call)
                    if self._ingest_result(call_args=resolved, result=result):
                        changed = True


def _build_placeholder_equivalences(
    *,
    uf: _PlaceholderUF,
    seed_bindings: Dict[str, object],
    templates: Sequence[ActionTemplate],
) -> None:
    """
    Derive placeholder equivalences + constant bindings from action templates.

    If two templates reference different placeholders for the same (tool, param_key),
    unify them. If a placeholder is paired with a constant across templates, seed its
    binding to that constant when unambiguous.
    """

    buckets: Dict[Tuple[str, str], List[object]] = {}
    for tmpl in templates:
        qualified = tmpl.qualified_tool_name
        if not qualified:
            continue
        for key, value in tmpl.params.items():
            buckets.setdefault((qualified, str(key)), []).append(value)

    for (qualified, param_key), values in buckets.items():
        placeholders: List[str] = []
        concretes: List[str] = []
        for v in values:
            if isinstance(v, str) and is_whole_placeholder(v):
                placeholders.append(placeholder_name(v))
            elif v is not None:
                concretes.append(str(v))
        # Unify all placeholder names in this bucket.
        for i in range(1, len(placeholders)):
            uf.union(placeholders[0], placeholders[i])
        # If all concrete values agree, bind the placeholder group to that constant.
        if placeholders and concretes:
            concrete = concretes[0]
            if all(c == concrete for c in concretes):
                root = uf.find(placeholders[0])
                seed_bindings.setdefault(root, concrete)


def _dedupe_action_templates(templates: Iterable[ActionTemplate]) -> List[ActionTemplate]:
    seen: set[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = set()
    out: List[ActionTemplate] = []
    for t in templates:
        frozen = tuple(sorted((str(k), str(v)) for k, v in t.params.items() if v is not None))
        key = (t.phase_key, t.qualified_tool_name, frozen)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _extract_selected_memory_plan_ids(match_entry: Mapping[str, object]) -> List[str]:
    selected = match_entry.get("selected_views") if isinstance(match_entry.get("selected_views"), list) else []
    plan_ids: List[str] = []
    for view_id in selected:
        if not isinstance(view_id, str) or not view_id.startswith("memory::"):
            continue
        plan_id = view_id.split("memory::", 1)[1].strip()
        if plan_id:
            plan_ids.append(plan_id)
    return sorted(set(plan_ids))


def _candidate_from_tool_call(call: ExecutedCall, *, fallback: bool = False) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "type": "tool",
        "tool": call.tool_name,
        "qualified_tool_name": call.qualified_tool_name,
        "args": {k: str(v) for k, v in call.args.items() if v is not None},
        "side_effect": bool(call.side_effect),
        "result": _json_safe(call.result),
    }
    if fallback:
        meta["fallback"] = True
    return {
        "source": "tool",
        "tool": call.tool_name,
        "tool_view_id": call.tool_view_id,
        "tool_call_cost": float(call.tool_call_cost),
        "text": _call_string(call.qualified_tool_name, call.args),
        "cost": 0.0,
        "meta": meta,
    }


def _candidate_direct_call(
    *,
    qualified: str,
    tool: str,
    tool_view_id: str,
    tool_call_cost: float,
    side_effect: bool,
    args: Mapping[str, object],
    fallback: bool = True,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "type": "tool",
        "tool": tool,
        "qualified_tool_name": qualified,
        "args": {k: str(v) for k, v in args.items() if v is not None},
        "side_effect": bool(side_effect),
        "fallback": bool(fallback),
    }
    return {
        "source": "tool",
        "tool": tool,
        "tool_view_id": tool_view_id,
        "tool_call_cost": float(tool_call_cost),
        "text": _call_string(qualified, args),
        "cost": 0.0,
        "meta": meta,
    }


def _candidate_from_memory_action(
    *,
    plan_id: str,
    action_node_id: str,
    qualified: str,
    tool: str,
    tool_view_id: str,
    tool_call_cost: float,
    side_effect: bool,
    args: Mapping[str, object],
    raw_action: Optional[str],
) -> Dict[str, object]:
    cand = _candidate_direct_call(
        qualified=qualified,
        tool=tool,
        tool_view_id=f"memory::{plan_id}::{tool_view_id}",
        tool_call_cost=tool_call_cost,
        side_effect=side_effect,
        args=args,
        fallback=False,
    )
    cand["source"] = "memory"
    if isinstance(raw_action, str) and raw_action.strip():
        cand["text"] = raw_action.strip()
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    meta.update(
        {
            "type": "memory",
            "plan_id": str(plan_id),
            "action_node_id": str(action_node_id),
            "raw_action": str(raw_action or cand.get("text") or ""),
        }
    )
    cand["meta"] = meta
    return cand


def _placeholder_candidate() -> Dict[str, object]:
    return {"source": "placeholder", "text": "-", "cost": 0.0, "meta": {"type": "placeholder"}}


def _build_tree_for_template(
    *,
    task: str,
    template: Mapping[str, object],
    match_entry: Mapping[str, object],
    memory_views: Mapping[str, Mapping[str, object]],
    memory_graph: MemoryGraphIndex,
    tool_views_by_qualified: Mapping[str, ToolViewInfo],
    domain_corrections: Mapping[str, Mapping[str, Mapping[str, str]]],
    max_id_expansions: int,
    enable_date_correction: bool,
    enable_domain_correction: bool,
    enable_memory_graph: bool,
) -> Dict[str, object]:
    template_id = str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE")
    if bool(enable_date_correction):
        template = date_correct(task=task, template=template)
    if bool(enable_domain_correction):
        template = domain_correct(template=template, corrections=domain_corrections)
    plan_id = f"plan::{template_id}"

    plan = template.get("plan") if isinstance(template.get("plan"), dict) else {}
    phases_payload = template.get("phases") if isinstance(template.get("phases"), list) else []
    actions_payload = template.get("actions") if isinstance(template.get("actions"), list) else []

    # Phase ordering: (phase_index, phase_key)
    phases: List[Dict[str, object]] = []
    for idx, ph in enumerate(phases_payload, start=1):
        if not isinstance(ph, dict):
            continue
        phase_key = str(ph.get("phase_key") or f"phase_{idx}")
        phase_node_id = f"{plan_id}::phase::{phase_key}"
        phase_index = ph.get("phase_index")
        phase_type = ph.get("phase_type")
        phase_attrs = ph.get("attrs") if isinstance(ph.get("attrs"), dict) else {}
        phases.append(
            {
                "phase_key": phase_key,
                "phase_node_id": phase_node_id,
                "phase_type": phase_type,
                "phase_index": phase_index,
                "attrs": dict(phase_attrs),
            }
        )
    phases.sort(key=lambda p: (int(p.get("phase_index") or 0), str(p.get("phase_key") or "")))

    # Action nodes grouped by phase_key.
    actions_by_phase: Dict[str, List[Tuple[str, Dict[str, object]]]] = {p["phase_key"]: [] for p in phases}
    for idx, action in enumerate(actions_payload, start=1):
        if not isinstance(action, dict):
            continue
        action_node_id = f"{plan_id}::action::{idx}"
        phase_key = str(action.get("phase_key") or "")
        if phase_key not in actions_by_phase:
            # Fallback: attach to first phase.
            phase_key = phases[0]["phase_key"] if phases else phase_key
        actions_by_phase.setdefault(phase_key, []).append((action_node_id, dict(action)))

    def _action_sort_key(item: Tuple[str, Dict[str, object]]) -> Tuple[int, str]:
        node_id, action = item
        raw = action.get("order_index")
        try:
            return (int(raw), node_id)  # type: ignore[arg-type]
        except Exception:
            return (0, node_id)

    for key in actions_by_phase:
        actions_by_phase[key].sort(key=_action_sort_key)

    lambda_raw = match_entry.get("lambda") if isinstance(match_entry.get("lambda"), dict) else {}
    lambda_by_edge: Dict[str, List[str]] = {}
    if isinstance(lambda_raw, dict):
        for edge_key, views in lambda_raw.items():
            if not isinstance(edge_key, str) or not edge_key.strip():
                continue
            if isinstance(views, list):
                cleaned = [str(v).strip() for v in views if isinstance(v, str) and str(v).strip()]
                if cleaned:
                    lambda_by_edge[edge_key.strip()] = cleaned
            elif isinstance(views, str) and views.strip():
                lambda_by_edge[edge_key.strip()] = [views.strip()]

    # Build action templates from query + selected memory views to drive placeholder equivalence + check fixpoint.
    seed_bindings: Dict[str, object] = {}
    uf = _PlaceholderUF()
    base_templates: List[ActionTemplate] = []
    for action in actions_payload:
        if not isinstance(action, dict):
            continue
        tmpl = _action_template_from_query_action(action)
        if tmpl is not None:
            base_templates.append(tmpl)

    selected_memory = _extract_selected_memory_plan_ids(match_entry)
    mem_templates: List[ActionTemplate] = []
    for pid in selected_memory:
        mem = memory_views.get(pid)
        if isinstance(mem, Mapping):
            mem_templates.extend(_action_templates_from_memory_view(mem))

    all_templates = _dedupe_action_templates(list(base_templates) + list(mem_templates))
    _build_placeholder_equivalences(uf=uf, seed_bindings=seed_bindings, templates=all_templates)

    materializer = MatchJoinMaterializer(
        task=task,
        tool_views_by_qualified=tool_views_by_qualified,
        placeholder_uf=uf,
        seed_bindings=seed_bindings,
        max_id_expansions=max_id_expansions,
    )
    materializer.execute_check_fixpoint(all_templates)

    placeholder_names: set[str] = set()
    for tmpl in base_templates:
        for raw in tmpl.params.values():
            if not isinstance(raw, str):
                continue
            for match in _PLACEHOLDER_ANY_RE.findall(raw):
                if is_whole_placeholder(match):
                    placeholder_names.add(placeholder_name(match))
    placeholder_roots = {name: uf.find(name) for name in sorted(placeholder_names)}
    placeholder_values = {
        name: _json_safe(materializer._binding_lookup(name))  # type: ignore[attr-defined]
        for name in sorted(placeholder_names)
    }

    # Helper: build candidates for a specific query action.
    def _candidates_for_query_action(*, action: Mapping[str, object], slot_edge: str, action_node_id: str) -> List[Dict[str, object]]:
        tmpl = _action_template_from_query_action(action)
        if tmpl is None or not tmpl.qualified_tool_name:
            return [_placeholder_candidate()]

        qualified = tmpl.qualified_tool_name
        info = tool_views_by_qualified.get(qualified)
        if info is None:
            tool = qualified.split(".", 1)[0]
            tool_view_id = f"tool::{tool}::{tmpl.action_type}"
            tool_call_cost = 0.0
            side_effect = bool(tmpl.side_effect)
        else:
            tool = info.tool
            tool_view_id = info.view_id
            tool_call_cost = float(info.cost)
            side_effect = bool(info.side_effect)

        candidates: List[Dict[str, object]] = []

        if bool(enable_memory_graph):
            slot_views = lambda_by_edge.get(str(slot_edge).strip()) or []
            for view_id in slot_views:
                if not isinstance(view_id, str) or not view_id.startswith("memory::"):
                    continue
                mem_view_id = view_id.split("memory::", 1)[1].strip()
                if not mem_view_id:
                    continue
                view_def = memory_views.get(mem_view_id)
                base_template = _base_template_from_graph_payload(view_def) if isinstance(view_def, Mapping) else None
                instance_ids: List[str] = []
                if base_template:
                    instance_ids = list(memory_graph.by_base_template.get(base_template) or [])
                if not instance_ids:
                    instance_ids = [mem_view_id]

                for instance_plan_id in instance_ids:
                    mem_graph_payload = memory_graph.graphs.get(instance_plan_id)
                    if not isinstance(mem_graph_payload, Mapping):
                        continue
                    nodes = (
                        mem_graph_payload.get("nodes")
                        if isinstance(mem_graph_payload.get("nodes"), list)
                        else []
                    )
                    for node in nodes:
                        if not isinstance(node, dict) or node.get("type") != "Action":
                            continue
                        node_id = str(node.get("id") or "").strip() or str(action_node_id)
                        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
                        if str(attrs.get("qualified_tool_name") or "").strip() != qualified:
                            continue
                        if str(attrs.get("action_type") or "").strip() != str(tmpl.action_type):
                            continue
                        args_raw = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
                        args = dict(args_raw)
                        if not _call_matches_query_params(query_params=tmpl.params, call_args=args):
                            continue
                        raw_action = attrs.get("raw_action") if isinstance(attrs.get("raw_action"), str) else None
                        tool_name_mem = str(attrs.get("tool_name") or tool).strip() or tool
                        side_effect_mem = (
                            bool(attrs.get("side_effect"))
                            if isinstance(attrs.get("side_effect"), bool)
                            else bool(side_effect)
                        )
                        cand = _candidate_from_memory_action(
                            plan_id=instance_plan_id,
                            action_node_id=node_id,
                            qualified=qualified,
                            tool=tool_name_mem,
                            tool_view_id=tool_view_id,
                            tool_call_cost=tool_call_cost,
                            side_effect=side_effect_mem,
                            args=args,
                            raw_action=raw_action,
                        )
                        meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                        meta.setdefault("memory_view_id", mem_view_id)
                        if base_template:
                            meta.setdefault("memory_base_template", base_template)
                        cand["meta"] = meta
                        candidates.append(cand)

        executed = materializer.executed.get(qualified) or []
        if executed:
            for call in executed:
                if not _call_matches_query_params(query_params=tmpl.params, call_args=call.args):
                    continue
                candidates.append(_candidate_from_tool_call(call, fallback=False))

        # Do-phase: expand ID placeholders into concrete candidates.
        if bool(tmpl.side_effect):
            resolved = materializer._resolve_params(qualified, tmpl.params)  # type: ignore[attr-defined]
            candidate_sigs: set[Tuple[Tuple[str, str], ...]] = set()

            def _push(args: Mapping[str, object]) -> None:
                if _contains_unresolved_placeholders(args):
                    return
                sig = tuple(sorted((str(k), str(v)) for k, v in args.items() if v is not None))
                if sig in candidate_sigs:
                    return
                candidate_sigs.add(sig)
                candidates.append(
                    _candidate_direct_call(
                        qualified=qualified,
                        tool=tool,
                        tool_view_id=tool_view_id,
                        tool_call_cost=tool_call_cost,
                        side_effect=side_effect,
                        args=args,
                        fallback=False,
                    )
                )

            # Prefer expanding ID placeholders even if a binding filled the placeholder.
            id_keys = [
                k
                for k in _id_key_candidates()
                if k in tmpl.params and isinstance(tmpl.params[k], str) and is_whole_placeholder(str(tmpl.params[k]))
            ]
            expanded = False
            if id_keys:
                id_key = id_keys[0]
                ids = list(materializer.queues.get(id_key) or [])[: max_id_expansions]
                if ids:
                    expanded = True
                    for idv in ids:
                        args = dict(resolved)
                        args[id_key] = idv
                        _push(args)

            if not expanded and _contains_unresolved_placeholders(resolved):
                # Try ID expansion if applicable (resolved still has placeholders).
                id_keys = [k for k in _id_key_candidates() if k in resolved and is_whole_placeholder(str(resolved[k]))]
                if id_keys:
                    id_key = id_keys[0]
                    ids = list(materializer.queues.get(id_key) or [])[: max_id_expansions]
                    for idv in ids:
                        args = dict(resolved)
                        args[id_key] = idv
                        _push(args)
            else:
                _push(resolved)

        if not candidates:
            # Fallback direct call with best-effort placeholder filling (may still contain placeholders).
            resolved = materializer._resolve_params(qualified, tmpl.params)  # type: ignore[attr-defined]
            candidates.append(
                _candidate_direct_call(
                    qualified=qualified,
                    tool=tool,
                    tool_view_id=tool_view_id,
                    tool_call_cost=tool_call_cost,
                    side_effect=side_effect,
                    args=resolved,
                    fallback=True,
                )
            )
        # Deterministic dedupe: type+attrs identify a node (memory candidates may repeat across views).
        deduped: List[Dict[str, object]] = []
        seen: set[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = set()
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
            qn = str(meta.get("qualified_tool_name") or "").strip()
            args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
            frozen = tuple(sorted((str(k), str(v)) for k, v in args.items() if v is not None))
            key = (str(meta.get("type") or cand.get("source") or ""), qn, frozen)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        return deduped if deduped else candidates

    # Assemble tree.
    phase_nodes: List[Dict[str, object]] = []
    for phase in phases:
        phase_key = phase["phase_key"]
        phase_node_id = phase["phase_node_id"]
        phase_type = phase.get("phase_type")
        phase_index = phase.get("phase_index")
        phase_attrs = phase.get("attrs") if isinstance(phase.get("attrs"), dict) else {}

        phase_or_slots: List[Dict[str, object]] = []
        for action_node_id, action_obj in actions_by_phase.get(phase_key, []):
            edge_key = f"{phase_node_id} -[hasAction]-> {action_node_id}"
            attrs = action_obj.get("attrs") if isinstance(action_obj.get("attrs"), dict) else {}
            query_action: Dict[str, object] = {
                "action_type": action_obj.get("action_type"),
                "params": dict(action_obj.get("params") or {}) if isinstance(action_obj.get("params"), dict) else {},
                "tool_name": attrs.get("tool_name"),
                "qualified_tool_name": attrs.get("qualified_tool_name"),
                "side_effect": attrs.get("side_effect"),
                "order_index": action_obj.get("order_index"),
            }
            phase_or_slots.append(
                {
                    "kind": "OR",
                    "slot_edge": edge_key,
                    "query_action_node_id": action_node_id,
                    "query_action": query_action,
                    "candidates": _candidates_for_query_action(action=action_obj, slot_edge=edge_key, action_node_id=action_node_id),
                }
            )

        phase_nodes.append(
            {
                "kind": "AND",
                "query_phase_node_id": phase_node_id,
                "query_phase": {
                    "phase_key": phase_key,
                    "phase_type": phase_type,
                    "phase_index": phase_index,
                    **dict(phase_attrs),
                },
                "children": phase_or_slots,
            }
        )

    def _iter_or_slots(nodes: Sequence[Mapping[str, object]]) -> Iterator[Dict[str, object]]:
        for phase in nodes:
            if not isinstance(phase, Mapping) or str(phase.get("kind") or "") != "AND":
                continue
            for slot in phase.get("children") if isinstance(phase.get("children"), list) else []:
                if isinstance(slot, dict) and str(slot.get("kind") or "") == "OR":
                    yield slot

    def _match_join_prune_in_place(nodes: Sequence[Dict[str, object]]) -> None:
        """
        MatchJoin-style pruning for static trees.

        - Evidence pruning: if Check candidates return IDs, prune Do candidates whose ID is not supported.
        - Placeholder arc-consistency: for placeholders reused across slots, prune candidates whose assignments
          have no support in any other connected slot.
        """

        slots = list(_iter_or_slots(nodes))
        if not slots:
            return

        # 1) Evidence pruning for ID-bearing candidates.
        evidence_by_id: Dict[str, set[str]] = {k: set() for k in _id_key_candidates()}
        for slot in slots:
            qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
            if bool(qa.get("side_effect")):
                continue
            candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
                for id_key in _id_key_candidates():
                    v = args.get(id_key)
                    if isinstance(v, str) and v.strip() and not _PLACEHOLDER_ANY_RE.search(v):
                        evidence_by_id[id_key].add(v.strip())
                key, ids = _extract_ids(meta.get("result"))
                if key and ids:
                    evidence_by_id[key].update(set(ids))

        for slot in slots:
            candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
            if not candidates:
                continue
            pruned: List[Dict[str, object]] = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
                ok = True
                for id_key, allowed in evidence_by_id.items():
                    if not allowed:
                        continue
                    v = args.get(id_key)
                    if isinstance(v, str) and v.strip() and not _PLACEHOLDER_ANY_RE.search(v):
                        if v.strip() not in allowed:
                            ok = False
                            break
                if ok:
                    pruned.append(cand)
            if pruned:
                slot["candidates"] = pruned

        # 2) Placeholder arc consistency (exact MatchJoin analogue over shared placeholders).
        slot_param_roots: Dict[str, Dict[str, str]] = {}
        root_to_slots: Dict[str, set[str]] = {}
        for slot in slots:
            edge = str(slot.get("slot_edge") or "").strip()
            qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
            params = qa.get("params") if isinstance(qa.get("params"), dict) else {}
            for key, raw in params.items():
                if not isinstance(raw, str) or not is_whole_placeholder(raw):
                    continue
                root = uf.find(placeholder_name(raw))
                if not root:
                    continue
                slot_param_roots.setdefault(edge, {})[str(key)] = root
                root_to_slots.setdefault(root, set()).add(edge)

        join_roots = {root for root, connected in root_to_slots.items() if len(connected) >= 2}
        if not join_roots:
            return

        def _candidate_assignment(*, slot_edge: str, cand: Mapping[str, object]) -> Dict[str, str]:
            meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
            args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
            roots = slot_param_roots.get(slot_edge) or {}
            out: Dict[str, str] = {}
            for param_key, root in roots.items():
                if root not in join_roots:
                    continue
                v = args.get(param_key)
                if isinstance(v, str) and v.strip() and not _PLACEHOLDER_ANY_RE.search(v):
                    out[root] = v.strip()
            return out

        slot_by_edge = {str(slot.get("slot_edge") or "").strip(): slot for slot in slots}
        changed = True
        while changed:
            changed = False

            domain: Dict[Tuple[str, str], set[str]] = {}
            wildcard: Dict[Tuple[str, str], bool] = {}
            for edge, mapping in slot_param_roots.items():
                slot = slot_by_edge.get(edge)
                if not isinstance(slot, dict):
                    continue
                candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
                for root in set(mapping.values()):
                    if root not in join_roots:
                        continue
                    dom: set[str] = set()
                    wild = False
                    for cand in candidates:
                        if not isinstance(cand, dict):
                            continue
                        assign = _candidate_assignment(slot_edge=edge, cand=cand)
                        if root in assign:
                            dom.add(assign[root])
                        else:
                            wild = True
                    domain[(edge, root)] = dom
                    wildcard[(edge, root)] = wild

            for edge, mapping in slot_param_roots.items():
                slot = slot_by_edge.get(edge)
                if not isinstance(slot, dict):
                    continue
                candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
                if not candidates:
                    continue
                new_candidates: List[Dict[str, object]] = []
                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    assign = _candidate_assignment(slot_edge=edge, cand=cand)
                    ok = True
                    for root, value in assign.items():
                        for other_edge in root_to_slots.get(root, set()):
                            if other_edge == edge:
                                continue
                            other_dom = domain.get((other_edge, root), set())
                            other_wild = wildcard.get((other_edge, root), False)
                            if not other_wild and value not in other_dom:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_candidates.append(cand)
                if new_candidates and len(new_candidates) != len(candidates):
                    slot["candidates"] = new_candidates
                    changed = True

    _match_join_prune_in_place(phase_nodes)

    return {
        "version": 1,
        "task_name": task,
        "template_id": template_id,
        "compose_match_tags": {
            "date_correction": bool(enable_date_correction),
            "domain_correction": bool(enable_domain_correction),
            "memory_graph_extension": bool(enable_memory_graph),
        },
        "template": dict(template),
        "match": dict(match_entry),
        "materialized": {
            "bindings": dict(materializer.bindings),
            "queues": dict(materializer.queues),
            "placeholders": placeholder_values,
            "placeholder_roots": placeholder_roots,
            "tags": {
                "date_correction": bool(enable_date_correction),
                "domain_correction": bool(enable_domain_correction),
                "memory_graph_extension": bool(enable_memory_graph),
            },
        },
        "root": {
            "kind": "AND",
            "query_plan_node_id": plan_id,
            "query_plan": dict(plan),
            "children": phase_nodes,
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComposeMatch (multi): emit AND–OR trees with fixpoint tool materialization.")
    parser.add_argument("--task", type=str, default="travel", help="Task name (defaults to travel).")
    parser.add_argument("--split", type=str, required=True, help="Dataset split label (used for output naming).")
    parser.add_argument("--templates", type=Path, required=True, help="Init-template JSONL path.")
    parser.add_argument("--match", type=Path, required=True, help="View selection output JSON path (match_{split}.json).")
    parser.add_argument("--tool-views", type=Path, default=None, help="Override tool views JSON (defaults to artifacts/input/<task>/views/tool.json).")
    parser.add_argument("--memory-views", type=Path, default=None, help="Override memory views JSONL (defaults to artifacts/input/<task>/views/memory.jsonl).")
    parser.add_argument(
        "--memory-graph",
        type=Path,
        default=None,
        help="Override memory graph JSONL path (defaults to artifacts/output/<task>/memory_graph.jsonl).",
    )
    parser.add_argument(
        "--memory-graph-extension",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable materializing candidates from selected memory graph extensions.",
    )
    parser.add_argument(
        "--date-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable one-pass date normalization/correction for templates before materialization.",
    )
    parser.add_argument(
        "--domain-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable domain parameter canonicalization for templates before materialization.",
    )
    parser.add_argument("--max-id-expansions", type=int, default=5, help="Max ID expansions per (Do) action.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to artifacts/output/<task>/tree_{split}.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    task = str(args.task)

    templates_path = args.templates
    if not templates_path.is_absolute():
        templates_path = (PROJECT_ROOT / templates_path).resolve()
    if not templates_path.exists():
        raise SystemExit(f"Templates JSONL not found: {templates_path}")

    match_path = args.match
    if not match_path.is_absolute():
        match_path = (PROJECT_ROOT / match_path).resolve()
    if not match_path.exists():
        raise SystemExit(f"Match JSON not found: {match_path}")

    match_payload = json.loads(match_path.read_text(encoding="utf-8"))
    if not isinstance(match_payload, dict):
        raise SystemExit(f"Match JSON must be an object mapping template_id -> selection: {match_path}")

    tool_views_path = args.tool_views or (INPUT_ROOT / task / "views" / "tool.json")
    if not tool_views_path.is_absolute():
        tool_views_path = (PROJECT_ROOT / tool_views_path).resolve()
    tool_views_by_view_id, tool_views_by_qualified = _load_tool_views(tool_views_path)
    if not tool_views_by_view_id:
        raise SystemExit(f"Missing or invalid tool views: {tool_views_path}")

    memory_views_path = args.memory_views or (INPUT_ROOT / task / "views" / "memory.jsonl")
    if not memory_views_path.is_absolute():
        memory_views_path = (PROJECT_ROOT / memory_views_path).resolve()
    memory_views = _load_memory_views(memory_views_path) if memory_views_path.exists() else {}

    memory_graph_payload: Dict[str, Dict[str, object]] = {}
    memory_graph = MemoryGraphIndex(graphs={}, by_base_template={})
    if bool(args.memory_graph_extension):
        memory_graph_path = args.memory_graph or (OUTPUT_ROOT / task / "memory_graph.jsonl")
        if not memory_graph_path.is_absolute():
            memory_graph_path = (PROJECT_ROOT / memory_graph_path).resolve()
        if memory_graph_path.exists():
            memory_graph_payload = _load_memory_graph(memory_graph_path)
            memory_graph = _index_memory_graph(memory_graph_payload)

    domain_corrections = _load_domain_corrections(task)

    out_path = args.out or (OUTPUT_ROOT / task / f"tree_{args.split}.json")
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()

    out: Dict[str, object] = {}
    for template in _iter_jsonl(templates_path):
        template_id = str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE")
        selection = match_payload.get(template_id) or {}
        if not isinstance(selection, dict):
            selection = {}
        out[template_id] = _build_tree_for_template(
            task=task,
            template=template,
            match_entry=selection,
            memory_views=memory_views,
            memory_graph=memory_graph,
            tool_views_by_qualified=tool_views_by_qualified,
            domain_corrections=domain_corrections,
            max_id_expansions=int(args.max_id_expansions),
            enable_date_correction=bool(args.date_correction),
            enable_domain_correction=bool(args.domain_correction),
            enable_memory_graph=bool(args.memory_graph_extension),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote AND–OR trees for {len(out)} template(s) to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


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


def can_split_window(*, time_min: str, time_max: str, min_window_seconds: int) -> bool:
    start = _parse_ts(time_min)
    end = _parse_ts(time_max)
    if start is None or end is None:
        return False
    seconds = (end - start).total_seconds()
    return seconds > float(max(0, int(min_window_seconds)))


def split_time_window(*, time_min: str, time_max: str) -> Optional[Tuple[str, str, str, str]]:
    start = _parse_ts(time_min)
    end = _parse_ts(time_max)
    if start is None or end is None:
        return None
    if end <= start:
        return None
    mid = start + (end - start) / 2
    mid_dt = datetime(mid.year, mid.month, mid.day, mid.hour, mid.minute, mid.second)
    if mid_dt <= start:
        return None
    left_min = start.strftime("%Y-%m-%d %H:%M:%S")
    left_max = mid_dt.strftime("%Y-%m-%d %H:%M:%S")
    right_min_dt = mid_dt
    if right_min_dt == start:
        right_min_dt = start
    right_min = right_min_dt.strftime("%Y-%m-%d %H:%M:%S")
    right_max = end.strftime("%Y-%m-%d %H:%M:%S")
    return left_min, left_max, right_min, right_max


def result_is_maybe_truncated(*, result: object, cap_limit: int) -> bool:
    if cap_limit <= 0:
        return False
    if not isinstance(result, list):
        return False
    return len(result) >= cap_limit


def infer_selection_intent(*, query: str, selection_rule: str = "") -> Optional[str]:
    rule = str(selection_rule or "").strip().lower()
    if rule:
        if "earliest" in rule or "first" in rule:
            return "first"
        if "latest" in rule or "last" in rule:
            return "last"
        if "next" in rule:
            return "next"

    q = f" {str(query or '').strip().lower()} "
    if " first " in q or " earliest " in q:
        return "first"
    if " last " in q or " latest " in q:
        return "last"
    if " next " in q:
        return "next"
    return None


def calendar_selection_ok(
    *,
    intent: str,
    selected_event_id: str,
    retrieved_events: Sequence[Mapping[str, Any]],
    now: str,
) -> Optional[bool]:
    if intent not in ("first", "last", "next"):
        return None
    if not selected_event_id or not retrieved_events:
        return None

    parsed: list[Tuple[datetime, str]] = []
    for ev in retrieved_events:
        dt = _parse_ts(str(ev.get("event_start") or ""))
        eid = str(ev.get("event_id") or "").strip()
        if dt is None or not eid:
            continue
        parsed.append((dt, eid))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])

    desired_id: Optional[str] = None
    if intent == "first":
        desired_id = parsed[0][1]
    elif intent == "last":
        desired_id = parsed[-1][1]
    else:
        now_dt = _parse_ts(str(now or ""))
        if now_dt is None:
            return None
        for dt, eid in parsed:
            if dt >= now_dt:
                desired_id = eid
                break
        if desired_id is None:
            return None

    return bool(desired_id == selected_event_id)


@dataclass(frozen=True)
class ConditionalPredicate:
    op: str  # "<" | ">"
    threshold: float


_COND_RE = re.compile(
    r"\\bif\\b.*?\\b(?:was|is)\\b.*?\\b(?P<cmp>less than|more than)\\b\\s+(?P<num>\\d+(?:\\.\\d+)?)",
    re.I,
)


def infer_conditional_predicate(query: str) -> Optional[ConditionalPredicate]:
    m = _COND_RE.search(str(query or ""))
    if not m:
        return None
    cmp_text = str(m.group("cmp") or "").strip().lower()
    op = "<" if cmp_text == "less than" else ">"
    try:
        thr = float(m.group("num"))
    except Exception:
        return None
    return ConditionalPredicate(op=op, threshold=thr)


def _extract_numeric_values(results: Iterable[object]) -> list[float]:
    vals: list[float] = []
    for r in results:
        if isinstance(r, Mapping):
            for v in r.values():
                try:
                    if isinstance(v, bool):
                        vals.append(float(int(v)))
                    else:
                        vals.append(float(v))  # type: ignore[arg-type]
                except Exception:
                    continue
        elif isinstance(r, (int, float)) and not isinstance(r, bool):
            vals.append(float(r))
    return vals


def evaluate_predicate_over_analytics(
    *,
    predicate: ConditionalPredicate,
    check_calls: Sequence[Tuple[str, object]],
) -> Optional[bool]:
    analytics_results = [res for (q, res) in check_calls if isinstance(q, str) and q.startswith("analytics.")]
    values = _extract_numeric_values(analytics_results)
    if not values:
        return None
    if predicate.op == "<":
        return any(v < predicate.threshold for v in values)
    if predicate.op == ">":
        return any(v > predicate.threshold for v in values)
    return None


def infer_then_else_tools(*, query: str) -> Tuple[Optional[str], Optional[str]]:
    q = str(query or "").lower()
    then_tool: Optional[str] = None
    else_tool: Optional[str] = None

    if "plot" in q:
        then_tool = "analytics.create_plot"
    if "task" in q:
        then_tool = then_tool or "project_management.create_task"
    if "send" in q and "email" in q and then_tool is None:
        then_tool = "email.send_email"

    if "otherwise" in q or "else" in q:
        if "otherwise" in q and "email" in q:
            else_tool = "email.send_email"
        elif "otherwise" in q and "task" in q:
            else_tool = "project_management.create_task"
        elif "else" in q and "email" in q:
            else_tool = "email.send_email"
        elif "else" in q and "task" in q:
            else_tool = "project_management.create_task"
    return then_tool, else_tool


def allowed_do_qualified_tools(
    *,
    query: str,
    check_calls: Sequence[Tuple[str, object]],
) -> Optional[set[str]]:
    pred = infer_conditional_predicate(query)
    if pred is None:
        return None
    holds = evaluate_predicate_over_analytics(predicate=pred, check_calls=check_calls)
    if holds is None:
        return None

    then_tool, else_tool = infer_then_else_tools(query=query)
    if holds:
        return {then_tool} if then_tool else None
    if else_tool:
        return {else_tool}
    return set()

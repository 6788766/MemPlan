from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from task_helper.work.evaluation import utils as wb
from task_helper.work.tools.toolkits import all_tools


@dataclass(frozen=True)
class Constraint:
    name: str
    kind: str  # "hard" | "semantic"
    description: str
    check: Callable[["ExampleContext"], bool]


@dataclass
class Example:
    query: str
    prediction: list[str]
    ground_truth: list[str]
    error: str


@dataclass
class ExampleContext:
    ex: Example

    correct: bool
    exact_match: bool
    unwanted_side_effects: bool
    no_actions: bool

    wrong_email: bool
    end_date_minor_error: bool
    meeting_start_time_error: bool


def _normalize_error(error_value: object) -> str:
    if error_value is None:
        return ""
    if isinstance(error_value, float) and pd.isna(error_value):
        return ""
    s = str(error_value)
    return "" if s.lower() == "nan" else s


def parse_actions(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            return []
    return [s]


def compute_example_context(query: str, prediction: list[str], ground_truth: list[str], error_value: object) -> ExampleContext:
    error = _normalize_error(error_value)

    exact_match = wb.is_exact_match(prediction, ground_truth)
    correct = wb.is_correct(prediction, ground_truth, error)
    unwanted_side_effects = wb.has_side_effects(prediction, ground_truth)
    no_actions = len(prediction) == 0

    wrong_email = ("@example" in str(prediction)) and ("@atlas" not in str(prediction)) and (not correct)
    end_date_minor_error = wb.end_date_minor_error(ground_truth, prediction) and (not correct)
    meeting_start_time_error = wb.meeting_start_time_error(ground_truth, prediction) and (not correct)

    return ExampleContext(
        ex=Example(query=query, prediction=prediction, ground_truth=ground_truth, error=error),
        correct=correct,
        exact_match=exact_match,
        unwanted_side_effects=unwanted_side_effects,
        no_actions=no_actions,
        wrong_email=wrong_email,
        end_date_minor_error=end_date_minor_error,
        meeting_start_time_error=meeting_start_time_error,
    )


def _allowed_tool_names() -> set[str]:
    return {t.name for t in all_tools}

ALLOWED_TOOL_NAMES = _allowed_tool_names()


def _tool_name_from_action(action: str) -> str:
    return wb.get_function_name(action)


def _actions_executable_strict(actions: list[str]) -> bool:
    for domain in wb.DOMAINS:
        domain.reset_state()
    try:
        for action in actions:
            eval(action, wb.__dict__)
    except Exception:
        for domain in wb.DOMAINS:
            domain.reset_state()
        return False
    for domain in wb.DOMAINS:
        domain.reset_state()
    return True


def _calendar_business_hours_ok(actions: list[str]) -> bool:
    # If the prediction didn't touch calendar state, treat as not applicable / pass.
    touches_calendar = any(_tool_name_from_action(a) in {"calendar.create_event", "calendar.update_event"} for a in actions)
    if not touches_calendar:
        return True

    for domain in wb.DOMAINS:
        domain.reset_state()
    original_calendar = wb.calendar.CALENDAR_EVENTS.copy()
    _, predicted_calendar, *_ = wb.execute_actions_and_reset_state(actions)

    if predicted_calendar.empty:
        return True

    def _parse_dt(s: str) -> Optional[datetime]:
        try:
            return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _parse_minutes(s: str) -> Optional[int]:
        try:
            return int(float(str(s)))
        except Exception:
            return None

    orig_by_id = {}
    if not original_calendar.empty and "event_id" in original_calendar.columns:
        orig_by_id = {str(r["event_id"]): r for r in original_calendar.to_dict(orient="records")}

    changed_rows = []
    for row in predicted_calendar.to_dict(orient="records"):
        event_id = str(row.get("event_id", ""))
        if not event_id:
            continue
        if event_id not in orig_by_id:
            changed_rows.append(row)
            continue
        orig = orig_by_id[event_id]
        if (row.get("event_start") != orig.get("event_start")) or (row.get("duration") != orig.get("duration")):
            changed_rows.append(row)

    for row in changed_rows:
        start = _parse_dt(row.get("event_start"))
        duration_min = _parse_minutes(row.get("duration"))
        if start is None or duration_min is None:
            return False
        end = start + timedelta(minutes=duration_min)

        start_t = start.time()
        end_t = end.time()
        if start_t < datetime.strptime("09:00:00", "%H:%M:%S").time():
            return False
        if end_t > datetime.strptime("18:00:00", "%H:%M:%S").time():
            return False

    return True


DEFAULT_CONSTRAINTS: list[Constraint] = [
    Constraint(
        name="hard.no_error",
        kind="hard",
        description="Prediction must not report an error.",
        check=lambda ctx: ctx.ex.error == "",
    ),
    Constraint(
        name="hard.allowed_tools_only",
        kind="hard",
        description="All actions must call known WorkBench tools.",
        check=lambda ctx: all(_tool_name_from_action(a) in ALLOWED_TOOL_NAMES for a in ctx.ex.prediction),
    ),
    Constraint(
        name="hard.actions_executable",
        kind="hard",
        description="All actions must execute without raising an exception (strict run).",
        check=lambda ctx: _actions_executable_strict(ctx.ex.prediction),
    ),
    Constraint(
        name="hard.state_change_correct",
        kind="hard",
        description="Predicted tool calls must yield the same final state change as the ground truth.",
        check=lambda ctx: ctx.correct,
    ),
    Constraint(
        name="hard.no_unwanted_side_effects",
        kind="hard",
        description="If incorrect, must not change any tool state.",
        check=lambda ctx: not ctx.unwanted_side_effects,
    ),
    Constraint(
        name="hard.side_effect_calls_exact_match",
        kind="hard",
        description="Side-effect tool calls must exactly match ground truth (order-insensitive).",
        check=lambda ctx: ctx.exact_match,
    ),
    Constraint(
        name="semantic.no_wrong_email_domain",
        kind="semantic",
        description="Avoid using @example.com addresses when @atlas.com is expected.",
        check=lambda ctx: not ctx.wrong_email,
    ),
    Constraint(
        name="semantic.no_plot_end_date_minor_error",
        kind="semantic",
        description="Avoid the common plot end-date off-by-one (2023-11-29 vs 2023-11-30).",
        check=lambda ctx: not ctx.end_date_minor_error,
    ),
    Constraint(
        name="semantic.no_meeting_start_time_error",
        kind="semantic",
        description="Avoid the common 'first free slot' time error (e.g., 09:00/11:00/15:00/15:30 vs 13:00).",
        check=lambda ctx: not ctx.meeting_start_time_error,
    ),
    Constraint(
        name="semantic.calendar_business_hours",
        kind="semantic",
        description="Any created/updated meetings must start >= 09:00 and end <= 18:00.",
        check=lambda ctx: _calendar_business_hours_ok(ctx.ex.prediction),
    ),
]

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_TOOL_VIEWS = PROJECT_ROOT / "artifacts" / "input" / "work" / "views" / "tool.json"

_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        f = float(value)  # type: ignore[arg-type]
        if math.isnan(f) or math.isinf(f):
            return default
        return float(f)
    except Exception:
        return default


def _tool_view_cost_index() -> Dict[str, float]:
    try:
        payload = json.loads(WORK_TOOL_VIEWS.read_text(encoding="utf-8"))
    except Exception:
        return {}
    views = payload.get("views") if isinstance(payload, dict) else None
    if not isinstance(views, list):
        return {}
    by_qualified: Dict[str, float] = {}
    for item in views:
        if not isinstance(item, dict):
            continue
        node = item.get("node_pattern") if isinstance(item.get("node_pattern"), dict) else {}
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        qualified = attrs.get("qualified_tool_name")
        if not isinstance(qualified, str) or not qualified.strip():
            continue
        by_qualified[qualified.strip()] = _safe_float(item.get("cost"), default=0.0)
    return by_qualified


TOOL_COST_BY_QUALIFIED = _tool_view_cost_index()


@dataclass(frozen=True)
class ActionRecord:
    phase_key: str
    raw_action: str
    qualified_tool_name: str
    side_effect: bool
    params: Mapping[str, object]


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
    if isinstance(value, str) and _PLACEHOLDER_RE.search(value):
        return True
    return False


def _allowed_tools_only(call_strings: Sequence[str]) -> bool:
    from task_helper.work.evaluation.constraints import ALLOWED_TOOL_NAMES
    from task_helper.work.evaluation import utils as wb

    for action in call_strings:
        if not action or action == "-":
            continue
        name = wb.get_function_name(action)
        if name not in ALLOWED_TOOL_NAMES:
            return False
    return True


def _actions_executable(call_strings: Sequence[str]) -> bool:
    from task_helper.work.evaluation import utils as wb

    for domain in wb.DOMAINS:
        domain.reset_state()
    try:
        for action in call_strings:
            if not action or action == "-":
                continue
            eval(action, wb.__dict__)
    except Exception:
        for domain in wb.DOMAINS:
            domain.reset_state()
        return False
    for domain in wb.DOMAINS:
        domain.reset_state()
    return True


def _calendar_business_hours_ok(call_strings: Sequence[str]) -> bool:
    from task_helper.work.evaluation.constraints import _calendar_business_hours_ok

    return bool(_calendar_business_hours_ok(list(call_strings)))


def _wrong_email_domain(call_strings: Sequence[str]) -> bool:
    return ("@example" in str(call_strings)) and ("@atlas" not in str(call_strings))


def _action_type_from_qualified(qualified_tool_name: str) -> str:
    qualified = str(qualified_tool_name or "").strip()
    if "." not in qualified:
        return ""
    return qualified.split(".", 1)[1].strip()


def compute_gt_free_checks(
    records: Sequence[ActionRecord],
    *,
    required_actions: Sequence[Tuple[str, str]] = (),
) -> Tuple[Dict[str, bool], float]:
    call_strings = [r.raw_action for r in records if r.raw_action]

    checks: Dict[str, bool] = {}
    checks["allowed_tools_only"] = _allowed_tools_only(call_strings)
    checks["actions_executable"] = _actions_executable(call_strings)
    checks["calendar_business_hours"] = _calendar_business_hours_ok(call_strings)
    checks["no_wrong_email_domain"] = not _wrong_email_domain(call_strings)

    checks["no_do_placeholders"] = True
    checks["phase_consistency"] = True
    checks["required_actions_present"] = True
    for r in records:
        kind = _phase_kind(r.phase_key)
        if kind == "do":
            if any(_contains_placeholder(v) for v in r.params.values()):
                checks["no_do_placeholders"] = False
        if kind == "check" and r.side_effect:
            checks["phase_consistency"] = False
        if kind == "do" and not r.side_effect:
            checks["phase_consistency"] = False

    if required_actions:
        executed: set[Tuple[str, str]] = set()
        for r in records:
            if not r.raw_action or r.raw_action == "-":
                continue
            kind = _phase_kind(r.phase_key)
            action_type = _action_type_from_qualified(r.qualified_tool_name)
            if kind and action_type:
                executed.add((kind, action_type))
        checks["required_actions_present"] = all((kind, action) in executed for kind, action in required_actions)

    passed = sum(1 for v in checks.values() if v)
    rate = passed / len(checks) if checks else 1.0
    return checks, float(rate)


def compute_tool_cost(records: Sequence[ActionRecord], *, do_weight: float = 1.0) -> float:
    total = 0.0
    for r in records:
        base = _safe_float(TOOL_COST_BY_QUALIFIED.get(r.qualified_tool_name), default=0.0)
        kind = _phase_kind(r.phase_key)
        weight = float(do_weight) if kind == "do" else 1.0
        total += base * weight
    return float(total)


def hard_pass_from_checks(checks: Mapping[str, bool]) -> bool:
    # GT-free hard gate used for planning (not evaluation):
    # ensure plan is executable and structurally consistent.
    required = ("allowed_tools_only", "actions_executable", "phase_consistency", "required_actions_present")
    return all(bool(checks.get(k)) for k in required)

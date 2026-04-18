from __future__ import annotations

import math
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple


_PLACEHOLDER_RE = re.compile(r"^\{[^{}]+\}$")


def _is_placeholder(candidate: Mapping[str, object]) -> bool:
    return str(candidate.get("source") or "") == "placeholder" or str(candidate.get("text") or "") == "-"


def _safe_float(value: object, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return float(value)
        number = float(value)  # type: ignore[arg-type]
        if math.isnan(number) or math.isinf(number):
            return default
        return float(number)
    except (TypeError, ValueError):
        return default


def _candidate_side_effect(candidate: Mapping[str, object]) -> Optional[bool]:
    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    if isinstance(meta, dict) and isinstance(meta.get("side_effect"), bool):
        return bool(meta["side_effect"])
    if isinstance(meta, dict) and meta.get("type") == "memory":
        attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
        if isinstance(attrs, dict) and isinstance(attrs.get("side_effect"), bool):
            return bool(attrs["side_effect"])
    return None


def online_step(state, *, slot, candidate: Mapping[str, object]) -> bool:
    side_effect = _candidate_side_effect(candidate)
    phase_key = str(getattr(slot, "phase_key", "") or "")

    if side_effect is not None:
        if phase_key == "check" and side_effect:
            return False
        if phase_key == "do" and not side_effect:
            return False
    return True


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


def fill_action(query_action: Mapping[str, object], candidate: Mapping[str, object], slot) -> Dict[str, object]:
    filled: Dict[str, object] = dict(query_action)
    params = filled.get("params") if isinstance(filled.get("params"), dict) else {}
    params_out: Dict[str, object] = dict(params)

    if _is_placeholder(candidate):
        filled["params"] = params_out
        return filled

    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    meta_type = meta.get("type") if isinstance(meta, dict) else None

    if meta_type == "memory":
        attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
        mem_params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        _merge_params_with_placeholders(params_out, mem_params)

        for key in ("tool_name", "qualified_tool_name", "side_effect", "raw_action"):
            if key in attrs and filled.get(key) is None:
                filled[key] = attrs.get(key)
        if "raw_action" in attrs and isinstance(attrs.get("raw_action"), str):
            filled["raw_action"] = attrs["raw_action"]

        filled["params"] = params_out
        return filled

    # Tool-call candidate (including ComposeMatch direct_call).
    if isinstance(meta, dict):
        args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
        _merge_params_with_placeholders(params_out, {k: str(v) for k, v in args.items()})
        qualified = meta.get("qualified_tool_name")
        if isinstance(qualified, str) and qualified.strip():
            filled["qualified_tool_name"] = qualified.strip()
        tool_name = meta.get("tool")
        if isinstance(tool_name, str) and tool_name.strip():
            filled["tool_name"] = tool_name.strip()
        if isinstance(meta.get("side_effect"), bool):
            filled["side_effect"] = bool(meta["side_effect"])

    if isinstance(candidate.get("text"), str) and candidate.get("text").strip():
        filled["raw_action"] = candidate["text"].strip()

    filled["params"] = params_out
    return filled


def _tool_name_from_action(action: str) -> str:
    prefix = str(action.split("(", 1)[0]).strip()
    if prefix.endswith(".func"):
        prefix = prefix[: -len(".func")]
    return prefix


def _actions_executable_strict(actions: Sequence[str]) -> bool:
    from task_helper.work.evaluation import utils as wb

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


def _calendar_business_hours_ok(actions: Sequence[str]) -> bool:
    from task_helper.work.evaluation.constraints import _calendar_business_hours_ok as check

    return bool(check(list(actions)))


def evaluate(plan: Mapping[str, object], phases, slots, chosen_indices: Sequence[int], config):
    from planner.twin_track import ScoreResult
    from task_helper.work.scoring import ActionRecord, compute_gt_free_checks, compute_tool_cost, hard_pass_from_checks

    required_actions = []
    for slot in slots:
        phase_kind = str(getattr(slot, "phase_key", "") or "")
        phase_kind = "check" if phase_kind.startswith("check") else ("do" if phase_kind.startswith("do") else "")
        action_type = str(getattr(slot, "action_type", "") or "").strip()
        if phase_kind and action_type:
            required_actions.append((phase_kind, action_type))

    chosen_actions = [
        fill_action(slot.query_action, slot.candidates[idx], slot) if idx < len(slot.candidates) else dict(slot.query_action)
        for slot, idx in zip(slots, chosen_indices)
    ]

    def _quote(v: object) -> str:
        text = str(v) if v is not None else ""
        text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{text}"'

    records: list[ActionRecord] = []
    for slot, action in zip(slots, chosen_actions):
        tool_name = str(action.get("tool_name") or "").strip()
        action_type = str(action.get("action_type") or "").strip()
        qualified = str(action.get("qualified_tool_name") or "").strip()
        if not qualified and tool_name and action_type:
            qualified = f"{tool_name}.{action_type}"

        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        raw = action.get("raw_action")
        if isinstance(raw, str) and raw.strip():
            raw_action = raw.strip()
        elif qualified:
            args = ", ".join(f"{k}={_quote(v)}" for k, v in params.items() if v is not None)
            raw_action = f"{qualified}.func({args})"
        else:
            raw_action = "-"

        side_effect = action.get("side_effect")
        records.append(
            ActionRecord(
                phase_key=str(getattr(slot, "phase_key", "") or ""),
                raw_action=raw_action,
                qualified_tool_name=qualified,
                side_effect=bool(side_effect) if isinstance(side_effect, bool) else False,
                params=dict(params),
            )
        )

    checks, semantic_rate = compute_gt_free_checks(records, required_actions=required_actions)
    tool_cost = compute_tool_cost(records, do_weight=1.0)
    hard_pass = hard_pass_from_checks(checks)

    tolerance = float(getattr(config, "semantic_tolerance", 0.8))
    tolerance = max(0.0, min(1.0, tolerance))
    hard_fail_penalty = float(getattr(config, "hard_fail_penalty", -1e12))
    shortfall_weight = float(getattr(config, "semantic_shortfall_weight", 1e6))

    if not hard_pass:
        reward = hard_fail_penalty - tool_cost
    elif semantic_rate < tolerance:
        reward = (-tool_cost) - (shortfall_weight * (tolerance - semantic_rate))
    else:
        reward = -tool_cost

    hard_keys = {"allowed_tools_only", "actions_executable", "phase_consistency"}
    hard_details = {f"hard.{k}": (bool(v), None) for k, v in checks.items() if k in hard_keys}
    semantic_details = {f"semantic.{k}": (bool(v), None) for k, v in checks.items() if k not in hard_keys}

    return ScoreResult(
        reward=float(reward),
        hard_pass=bool(hard_pass),
        semantic_rate=float(semantic_rate),
        total_cost=float(tool_cost),
        hard_details=hard_details,
        semantic_details=semantic_details,
    )

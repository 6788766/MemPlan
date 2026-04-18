"""
Task-agnostic plan graph extraction.

This module builds a configurable "Plan → Phase → Action" graph for each row in a
CSV file. Task-specific extraction logic is driven by configuration
(`artifacts/input/<task>/graph_config.json`) rather than hard-coded here.

How to add a new task config (quick guide):
  1) Create `artifacts/input/<task>/graph_config.json`.
  2) Provide a detector (or use a `task_field` mapping), a `phase_generator`,
     and one or more `action_extractors`.
  3) Put your dataset under `artifacts/input/<task>/dataset/` (e.g., `train.csv`).
  4) Run: `python -m memory_graph.builder --task <task>`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import store
from .schema import MEdge, MNode, PlanGraph

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"
OUTPUT_ROOT = ARTIFACTS_ROOT / "output"
DEFAULT_TASK = "travel"
_LOG_PATH = ARTIFACTS_ROOT / "logs" / "log.txt"


def _append_run_log(
    *,
    argv: List[str],
    started_at: float,
    status: str,
    extra: Optional[Dict[str, object]] = None,
    error: Optional[str] = None,
) -> None:
    try:
        payload: Dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": "memory_graph.builder",
            "status": status,
            "elapsed_s": round(time.perf_counter() - started_at, 6),
            "argv": list(argv),
        }
        if extra:
            payload.update(dict(extra))
        if error:
            payload["error"] = error
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    phase_type: str
    phase_index: int
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpec:
    phase_key: str
    action_type: str
    params: Dict[str, object] = field(default_factory=dict)
    attrs: Dict[str, object] = field(default_factory=dict)
    order_index: Optional[int] = None


DetectorFn = Callable[[Mapping[str, object]], bool]
PhaseGeneratorFn = Callable[[Mapping[str, object]], List[PhaseSpec]]
ActionExtractorFn = Callable[[Mapping[str, object], Sequence[PhaseSpec]], List[ActionSpec]]


@dataclass
class TaskConfig:
    task_name: str
    priority: int
    detector: DetectorFn
    plan_attr_map: Dict[str, str] = field(default_factory=dict)
    include_unmapped_plan_attrs: bool = False
    phase_generator: Optional[PhaseGeneratorFn] = None
    action_extractors: List[ActionExtractorFn] = field(default_factory=list)
    allow_empty_phase: bool = False
    action_type_map: Dict[str, str] = field(default_factory=dict)
    before_scope: str = "phase"  # "phase" or "plan"


def _parse_cell(value: object) -> object:
    """
    Parse a CSV cell into a Python value. This is intentionally conservative:
    it only attempts literal parsing when the string looks like a list/dict.
    """

    if value is None:
        return None
    if isinstance(value, (int, float, bool, dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    if _INT_RE.match(text):
        try:
            return int(text)
        except ValueError:
            return text
    if _FLOAT_RE.match(text):
        try:
            return float(text)
        except ValueError:
            return text
    if text.startswith("{") or text.startswith("["):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return text


def normalise_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {key: _parse_cell(value) for key, value in row.items()}


def _compile_detector(spec: object) -> DetectorFn:
    if callable(spec):
        return spec  # type: ignore[return-value]
    if not isinstance(spec, dict):
        return lambda _row: False

    kind = spec.get("type")
    if kind == "always":
        return lambda _row: True
    if kind == "has_field":
        field = str(spec.get("field") or "")
        return lambda row: field in row
    if kind == "has_nonempty_field":
        field = str(spec.get("field") or "")
        return lambda row: bool(row.get(field))
    if kind == "field_equals":
        field = str(spec.get("field") or "")
        value = spec.get("value")
        case_insensitive = bool(spec.get("case_insensitive", True))

        def _fn(row: Mapping[str, object]) -> bool:
            raw = row.get(field)
            if raw is None:
                return False
            if case_insensitive and isinstance(raw, str) and isinstance(value, str):
                return raw.strip().lower() == value.strip().lower()
            return raw == value

        return _fn
    if kind == "regex":
        field = str(spec.get("field") or "")
        pattern = str(spec.get("pattern") or "")
        flags = re.IGNORECASE if spec.get("ignore_case", True) else 0
        compiled = re.compile(pattern, flags=flags)
        return lambda row: bool(compiled.search(str(row.get(field, "") or "")))
    if kind == "any":
        parts = [_compile_detector(item) for item in (spec.get("detectors") or [])]
        return lambda row: any(fn(row) for fn in parts)
    if kind == "all":
        parts = [_compile_detector(item) for item in (spec.get("detectors") or [])]
        return lambda row: all(fn(row) for fn in parts)

    return lambda _row: False


def _compile_phase_generator(spec: object) -> PhaseGeneratorFn:
    if callable(spec):
        return spec  # type: ignore[return-value]
    if not isinstance(spec, dict):
        raise ValueError("phase_generator must be an object in config")

    kind = spec.get("type")
    if kind == "single":
        key = str(spec.get("key") or "main")
        phase_type = str(spec.get("phase_type") or "Main")

        def _gen(_row: Mapping[str, object]) -> List[PhaseSpec]:
            return [PhaseSpec(key=key, phase_type=phase_type, phase_index=1, attrs={})]

        return _gen

    if kind == "check_do":
        check_key = str(spec.get("check_key") or "check")
        do_key = str(spec.get("do_key") or "do")
        check_phase_type = str(spec.get("check_phase_type") or "CheckRound")
        do_phase_type = str(spec.get("do_phase_type") or "DoRound")

        def _gen(_row: Mapping[str, object]) -> List[PhaseSpec]:
            return [
                PhaseSpec(key=check_key, phase_type=check_phase_type, phase_index=1, attrs={}),
                PhaseSpec(key=do_key, phase_type=do_phase_type, phase_index=2, attrs={}),
            ]

        return _gen

    if kind == "days":
        days_field = str(spec.get("days_field") or "days")
        phase_type = str(spec.get("phase_type") or "Day")
        key_template = str(spec.get("key_template") or "day_{i}")
        date_field = spec.get("date_field")

        def _gen(row: Mapping[str, object]) -> List[PhaseSpec]:
            raw_days = row.get(days_field)
            try:
                total = int(raw_days) if raw_days is not None else 0
            except (TypeError, ValueError):
                total = 0
            total = max(total, 1)

            dates: List[object] = []
            if date_field:
                candidate = row.get(str(date_field))
                if isinstance(candidate, list):
                    dates = candidate

            phases: List[PhaseSpec] = []
            for i in range(1, total + 1):
                attrs: Dict[str, object] = {}
                if dates and (i - 1) < len(dates):
                    attrs["date"] = dates[i - 1]
                phases.append(
                    PhaseSpec(
                        key=key_template.format(i=i, day=i, phase_index=i),
                        phase_type=phase_type,
                        phase_index=i,
                        attrs=attrs,
                    )
                )
            return phases

        return _gen

    if kind == "list_field":
        field = str(spec.get("field") or "")
        phase_type = str(spec.get("phase_type") or "Phase")
        key_template = str(spec.get("key_template") or "phase_{i}")
        default = spec.get("default") or ["main"]

        def _gen(row: Mapping[str, object]) -> List[PhaseSpec]:
            raw = row.get(field)
            values: List[object]
            if isinstance(raw, list):
                values = raw
            elif raw is None:
                values = list(default) if isinstance(default, list) else [default]
            else:
                values = [raw]

            phases: List[PhaseSpec] = []
            for i, item in enumerate(values, start=1):
                label = str(item) if item is not None else str(i)
                phases.append(
                    PhaseSpec(
                        key=key_template.format(i=i, name=label),
                        phase_type=phase_type,
                        phase_index=i,
                        attrs={"name": label},
                    )
                )
            return phases

        return _gen

    raise ValueError(f"Unsupported phase_generator type: {kind}")


def _resolve_value_spec(
    spec: object,
    *,
    row: Mapping[str, object],
    record: Mapping[str, object],
    value: object,
    params: Optional[Mapping[str, object]] = None,
) -> object:
    if isinstance(spec, str) and spec.startswith("$"):
        if spec == "$value":
            return value
        if spec == "$row":
            return dict(row)
        if spec == "$params":
            return dict(params or {})
        if spec.startswith("$param:"):
            if params is None:
                return None
            return params.get(spec.split(":", 1)[1])
        if spec.startswith("$row:"):
            return row.get(spec.split(":", 1)[1])
        if spec.startswith("$field:"):
            return record.get(spec.split(":", 1)[1])
        return spec
    return spec


def _apply_transform(params: Dict[str, object], transform: Mapping[str, object]) -> None:
    kind = transform.get("type")
    if kind != "regex_capture":
        return
    source_param = str(transform.get("source_param") or "value")
    pattern = str(transform.get("pattern") or "")
    if not pattern:
        return
    source_val = params.get(source_param)
    if source_val is None:
        return
    text = str(source_val)
    flags = 0
    for flag in transform.get("flags") or []:
        if str(flag).upper() == "IGNORECASE":
            flags |= re.IGNORECASE
    compiled = re.compile(pattern, flags=flags)
    match = compiled.search(text)
    if not match:
        return
    mapping = transform.get("group_to_param") or {}
    if isinstance(mapping, dict) and mapping:
        for group_name, param_name in mapping.items():
            params[str(param_name)] = match.group(str(group_name))
    else:
        params.update({k: v for k, v in match.groupdict().items() if v is not None})


def _extract_phase_records(plan_obj: object) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    """
    Generic extractor for a nested plan structure:
      * list: [meta_dict?, phase_records...] OR [meta_dict?, [phase_records...]]
      * dict: { "meta": {...}?, "phases": [...] }
    """
    if isinstance(plan_obj, list) and plan_obj:
        meta = plan_obj[0] if isinstance(plan_obj[0], dict) else None
        if len(plan_obj) > 1 and isinstance(plan_obj[1], list):
            raw_records = plan_obj[1]
        else:
            raw_records = plan_obj[1:]
        records = [item for item in raw_records if isinstance(item, dict) and item]
        return meta, records
    if isinstance(plan_obj, dict):
        phases = plan_obj.get("phases")
        if isinstance(phases, list):
            records = [item for item in phases if isinstance(item, dict) and item]
            meta = plan_obj.get("meta") if isinstance(plan_obj.get("meta"), dict) else None
            return meta, records
    return None, []


def _compile_action_extractor(spec: object, *, task: TaskConfig) -> ActionExtractorFn:
    if callable(spec):
        return spec  # type: ignore[return-value]
    if not isinstance(spec, dict):
        raise ValueError("action_extractor must be an object in config")

    kind = spec.get("type")
    if kind == "plan_field_phase_records":
        plan_field = str(spec.get("plan_field") or "")
        phase_index_field = str(spec.get("phase_index_field") or "")
        phase_key_template = str(spec.get("phase_key_template") or "phase_{i}")
        slot_rules = list(spec.get("slot_actions") or [])

        def _extract(row: Mapping[str, object], _phases: Sequence[PhaseSpec]) -> List[ActionSpec]:
            plan_obj = row.get(plan_field)
            _meta, records = _extract_phase_records(plan_obj)

            actions: List[ActionSpec] = []
            for idx, record in enumerate(records, start=1):
                raw_phase_index = record.get(phase_index_field) if phase_index_field else None
                try:
                    phase_index = int(raw_phase_index) if raw_phase_index is not None else idx
                except (TypeError, ValueError):
                    phase_index = idx
                phase_key = phase_key_template.format(i=phase_index, day=phase_index, phase_index=phase_index)

                for rule in slot_rules:
                    if not isinstance(rule, dict):
                        continue
                    field_name = str(rule.get("field") or "")
                    if field_name == "__record__":
                        field_value = record
                    else:
                        field_value = record.get(field_name)

                    skip_values = set(rule.get("skip_values") or [])
                    if field_name != "__record__" and (field_value in skip_values or field_value is None):
                        continue

                    split_spec = rule.get("split")
                    items: List[object]
                    if isinstance(split_spec, dict) and isinstance(field_value, str):
                        delim = str(split_spec.get("delimiter") or ";")
                        raw_items = field_value.split(delim)
                        if split_spec.get("trim", True):
                            raw_items = [item.strip() for item in raw_items]
                        items = raw_items
                        if split_spec.get("drop_empty", True):
                            items = [item for item in items if item not in skip_values and item not in {None, ""}]
                    else:
                        items = [field_value]

                    for offset, item in enumerate(items):
                        params_spec = rule.get("params") or {"value": "$value"}
                        params: Dict[str, object] = {}
                        if isinstance(params_spec, dict):
                            for key, val_spec in params_spec.items():
                                params[str(key)] = _resolve_value_spec(
                                    val_spec, row=row, record=record, value=item, params=None
                                )
                        else:
                            params["value"] = item

                        for transform in rule.get("transforms") or []:
                            if isinstance(transform, dict):
                                _apply_transform(params, transform)

                        required_params = rule.get("require_params") or []
                        if required_params:
                            missing = [
                                str(key)
                                for key in required_params
                                if params.get(str(key)) is None or params.get(str(key)) == ""
                            ]
                            if missing:
                                continue

                        action_type = str(rule.get("action_type") or "Action")
                        action_type = task.action_type_map.get(action_type, action_type)
                        attrs_spec = rule.get("attrs") or {}
                        attrs: Dict[str, object] = {}
                        if isinstance(attrs_spec, dict):
                            for key, val_spec in attrs_spec.items():
                                attrs[str(key)] = _resolve_value_spec(
                                    val_spec, row=row, record=record, value=item, params=params
                                )

                        for key in rule.get("drop_params") or []:
                            params.pop(str(key), None)
                        order_index = rule.get("order_index")
                        if isinstance(order_index, int):
                            adjusted = order_index + offset
                        else:
                            adjusted = order_index

                        actions.append(
                            ActionSpec(
                                phase_key=phase_key,
                                action_type=action_type,
                                params=params,
                                attrs=attrs,
                                order_index=adjusted if isinstance(adjusted, int) else None,
                            )
                        )
            return actions

        return _extract

    if kind == "row_actions_field":
        field = str(spec.get("field") or "")
        default_phase = str(spec.get("default_phase") or "main")
        phase_key_field = str(spec.get("phase_key_field") or "")
        action_type_field = str(spec.get("action_type_field") or "action_type")
        params_field = str(spec.get("params_field") or "params")
        order_field = str(spec.get("order_field") or "")

        def _extract(row: Mapping[str, object], _phases: Sequence[PhaseSpec]) -> List[ActionSpec]:
            raw = row.get(field)
            if raw is None:
                return []
            if not isinstance(raw, list):
                return []
            actions: List[ActionSpec] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                phase_key = default_phase
                if phase_key_field:
                    phase_key = str(item.get(phase_key_field) or phase_key)
                action_type = str(item.get(action_type_field) or "Action")
                action_type = task.action_type_map.get(action_type, action_type)
                params = item.get(params_field)
                if not isinstance(params, dict):
                    params = {}
                order_index = item.get(order_field) if order_field else None
                try:
                    order_index_int = int(order_index) if order_index is not None else None
                except (TypeError, ValueError):
                    order_index_int = None
                actions.append(
                    ActionSpec(
                        phase_key=phase_key,
                        action_type=action_type,
                        params=dict(params),
                        attrs={},
                        order_index=order_index_int,
                    )
                )
            return actions

        return _extract

    if kind == "single_action":
        phase_key = str(spec.get("phase_key") or "main")
        action_type = str(spec.get("action_type") or "RawRow")
        params_spec = spec.get("params") or {"row": "$row"}
        attrs_spec = spec.get("attrs") or {}

        def _extract(row: Mapping[str, object], _phases: Sequence[PhaseSpec]) -> List[ActionSpec]:
            params: Dict[str, object] = {}
            if isinstance(params_spec, dict):
                for key, val_spec in params_spec.items():
                    params[str(key)] = _resolve_value_spec(val_spec, row=row, record={}, value=None, params=None)
            attrs: Dict[str, object] = {}
            if isinstance(attrs_spec, dict):
                for key, val_spec in attrs_spec.items():
                    attrs[str(key)] = _resolve_value_spec(val_spec, row=row, record={}, value=None, params=params)
            return [ActionSpec(phase_key=phase_key, action_type=action_type, params=params, attrs=attrs, order_index=0)]

        return _extract

    if kind == "function_calls_field":
        """
        Extract actions from a CSV field containing a list of Python-call strings.

        Example item:
          "calendar.create_event.func(event_name=\"...\", participant_email=\"...\", ...)"

        The extractor parses the tool call into:
          - action_type: the tool function name (e.g., "create_event")
          - params: keyword args dict (best-effort; falls back to {"raw": <string>})
          - attrs: helper metadata (tool_name, qualified_tool_name, side_effect, raw_action)
        """

        field = str(spec.get("field") or "")
        fallback_field = str(spec.get("fallback_field") or "")
        phase_key = str(spec.get("phase_key") or "main")
        check_phase_key = str(spec.get("check_phase_key") or "")
        do_phase_key = str(spec.get("do_phase_key") or "")
        include_raw_action = bool(spec.get("include_raw_action", True))

        side_effect_tools = set(spec.get("side_effect_tools") or [])

        def _tool_name(action: str) -> str:
            prefix = str(action.split("(", 1)[0])
            parts = prefix.split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else ""

        def _tool_domain(tool_name: str) -> str:
            return tool_name.split(".", 1)[0] if "." in tool_name else ""

        def _action_type(tool_name: str) -> str:
            return tool_name.split(".", 1)[1] if "." in tool_name else ""

        def _parse_params(action: str) -> Dict[str, object]:
            try:
                expr = ast.parse(action, mode="eval").body
            except Exception:
                return {}
            if not isinstance(expr, ast.Call):
                return {}
            out: Dict[str, object] = {}
            for kw in expr.keywords:
                if not kw.arg:
                    continue
                try:
                    out[str(kw.arg)] = ast.literal_eval(kw.value)
                except Exception:
                    out[str(kw.arg)] = None
            return out

        def _extract(row: Mapping[str, object], _phases: Sequence[PhaseSpec]) -> List[ActionSpec]:
            raw = row.get(field)
            if raw is None and fallback_field:
                raw = row.get(fallback_field)

            calls: List[object]
            if isinstance(raw, list):
                calls = raw
            elif raw is None:
                calls = []
            else:
                calls = [raw]

            actions: List[ActionSpec] = []
            for idx, call in enumerate(calls):
                if call is None:
                    continue
                text = str(call).strip()
                if not text:
                    continue

                qualified = _tool_name(text)
                action_type = _action_type(qualified) or "RawFunctionCall"
                params = _parse_params(text) or {"raw": text}
                attrs: Dict[str, object] = {
                    "tool_name": _tool_domain(qualified),
                    "qualified_tool_name": qualified,
                    "side_effect": qualified in side_effect_tools,
                }
                if include_raw_action:
                    attrs["raw_action"] = text

                resolved_phase_key = phase_key
                if check_phase_key and do_phase_key:
                    resolved_phase_key = do_phase_key if (qualified in side_effect_tools) else check_phase_key

                actions.append(
                    ActionSpec(
                        phase_key=resolved_phase_key,
                        action_type=action_type,
                        params=params,
                        attrs=attrs,
                        order_index=idx,
                    )
                )
            return actions

        return _extract

    raise ValueError(f"Unsupported action_extractor type: {kind}")


def _compile_task(task_spec: Mapping[str, object]) -> TaskConfig:
    task_name = str(task_spec.get("task_name") or "")
    if not task_name:
        raise ValueError("task_name is required in task config")
    priority = int(task_spec.get("priority") or 0)
    detector = _compile_detector(task_spec.get("detector") or {"type": "always"})
    plan_attr_map = dict(task_spec.get("plan_attr_map") or {})
    include_unmapped_plan_attrs = bool(task_spec.get("include_unmapped_plan_attrs", False))
    allow_empty_phase = bool(task_spec.get("allow_empty_phase", False))
    action_type_map = dict(task_spec.get("action_type_map") or {})
    before_scope = str((task_spec.get("before_edges") or {}).get("scope") or "phase")

    phase_generator = _compile_phase_generator(task_spec.get("phase_generator") or {"type": "single"})
    cfg = TaskConfig(
        task_name=task_name,
        priority=priority,
        detector=detector,
        plan_attr_map=plan_attr_map,
        include_unmapped_plan_attrs=include_unmapped_plan_attrs,
        phase_generator=phase_generator,
        action_extractors=[],
        allow_empty_phase=allow_empty_phase,
        action_type_map=action_type_map,
        before_scope=before_scope,
    )

    extractors: List[ActionExtractorFn] = []
    for extractor_spec in task_spec.get("action_extractors") or []:
        extractors.append(_compile_action_extractor(extractor_spec, task=cfg))
    cfg.action_extractors = extractors
    return cfg


def load_task_registry(config_path: Path) -> Tuple[Dict[str, TaskConfig], Optional[str], Optional[str], Dict[str, str]]:
    """
    Returns:
      (TASKS, default_task_name, task_field, task_mapping)
    """
    with config_path.open(encoding="utf-8") as fp:
        payload = json.load(fp)

    tasks: Dict[str, TaskConfig] = {}
    for task_spec in payload.get("tasks") or []:
        if not isinstance(task_spec, dict):
            continue
        task = _compile_task(task_spec)
        tasks[task.task_name] = task

    default_task = payload.get("default_task")
    task_field = payload.get("task_field")
    task_mapping = payload.get("task_mapping") or {}
    if not isinstance(task_mapping, dict):
        task_mapping = {}
    task_mapping = {str(k): str(v) for k, v in task_mapping.items()}

    return tasks, (str(default_task) if default_task else None), (str(task_field) if task_field else None), task_mapping


def _select_task_for_row(
    row: Mapping[str, object],
    *,
    tasks: Mapping[str, TaskConfig],
    default_task: Optional[str],
    task_field: Optional[str],
    task_mapping: Mapping[str, str],
) -> TaskConfig:
    if task_field:
        raw = row.get(task_field)
        if raw is not None:
            mapped = task_mapping.get(str(raw))
            if mapped and mapped in tasks:
                return tasks[mapped]

    matches = [task for task in tasks.values() if task.detector(row)]
    if matches:
        matches.sort(key=lambda t: t.priority, reverse=True)
        return matches[0]
    if default_task and default_task in tasks:
        return tasks[default_task]
    raise ValueError("No task config matched row and no default_task provided.")


class GraphBuilder:
    def __init__(
        self,
        tasks: Mapping[str, TaskConfig],
        *,
        default_task: Optional[str] = None,
        task_field: Optional[str] = None,
        task_mapping: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.tasks = dict(tasks)
        self.default_task = default_task
        self.task_field = task_field
        self.task_mapping = dict(task_mapping or {})

    def build_from_row(
        self,
        row: Mapping[str, object],
        *,
        plan_id: str,
        forced_task: Optional[str] = None,
    ) -> PlanGraph:
        task = self.tasks[forced_task] if forced_task else _select_task_for_row(
            row,
            tasks=self.tasks,
            default_task=self.default_task,
            task_field=self.task_field,
            task_mapping=self.task_mapping,
        )

        plan_attrs: Dict[str, object] = {"task_name": task.task_name}
        for col, attr in task.plan_attr_map.items():
            if col not in row:
                continue
            value = row.get(col)
            if value is None:
                continue
            plan_attrs[str(attr)] = value
        if task.include_unmapped_plan_attrs:
            for key, value in row.items():
                if key in task.plan_attr_map:
                    continue
                plan_attrs[key] = value

        if task.phase_generator is None:
            raise ValueError(f"Task '{task.task_name}' missing phase_generator.")
        phases = task.phase_generator(row)
        if not phases:
            raise ValueError(f"Plan {plan_id}: phase_generator returned 0 phases.")

        actions: List[ActionSpec] = []
        for extractor in task.action_extractors:
            actions.extend(extractor(row, phases))

        actions_by_phase: Dict[str, List[ActionSpec]] = {}
        for action in actions:
            actions_by_phase.setdefault(action.phase_key, []).append(action)

        phase_keys = {phase.key for phase in phases}
        unknown_phase_keys = [key for key in actions_by_phase if key not in phase_keys]
        if unknown_phase_keys:
            unknown = ", ".join(sorted(unknown_phase_keys))
            raise ValueError(f"Plan {plan_id}: actions reference unknown phase_key(s): {unknown}")

        if not task.allow_empty_phase:
            for phase in phases:
                if not actions_by_phase.get(phase.key):
                    raise ValueError(f"Plan {plan_id}: phase '{phase.key}' has no actions.")

        graph = PlanGraph(plan_id=plan_id)
        plan_node_id = f"plan::{plan_id}"
        graph.add_node(MNode(id=plan_node_id, type="Plan", attrs=plan_attrs))

        phase_node_ids: Dict[str, str] = {}
        for phase in phases:
            phase_id = f"{plan_node_id}::phase::{phase.key}"
            phase_node_ids[phase.key] = phase_id
            phase_attrs = {
                "phase_key": phase.key,
                "phase_type": phase.phase_type,
                "phase_index": phase.phase_index,
            }
            phase_attrs.update(dict(phase.attrs or {}))
            graph.add_node(MNode(id=phase_id, type="Phase", attrs=phase_attrs))
            graph.add_edge(MEdge(src=plan_node_id, dst=phase_id, type="hasPhase", attrs={}))

        action_records: List[Tuple[str, int, str, Optional[int]]] = []
        action_counter = 0
        for phase in phases:
            for action in actions_by_phase.get(phase.key, []):
                if not action.action_type or not isinstance(action.params, dict):
                    raise ValueError(f"Plan {plan_id}: invalid action spec: {action}")
                action_counter += 1
                action_id = f"{plan_node_id}::action::{action_counter}"
                attrs = {"action_type": action.action_type, "params": dict(action.params)}
                if action.order_index is not None:
                    attrs["order_index"] = int(action.order_index)
                attrs.update(dict(action.attrs or {}))
                graph.add_node(MNode(id=action_id, type="Action", attrs=attrs))
                graph.add_edge(MEdge(src=phase_node_ids[phase.key], dst=action_id, type="hasAction", attrs={}))
                action_records.append((phase.key, action_counter, action_id, action.order_index))

        self._add_before_edges(graph, phases, action_records, before_scope=task.before_scope)
        return graph

    def _add_before_edges(
        self,
        graph: PlanGraph,
        phases: Sequence[PhaseSpec],
        action_records: Sequence[Tuple[str, int, str, Optional[int]]],
        *,
        before_scope: str,
    ) -> None:
        if before_scope not in {"phase", "plan"}:
            raise ValueError(f"Unsupported before_edges.scope: {before_scope}")

        has_order = any(order is not None for _phase, _seq, _aid, order in action_records)
        if not has_order:
            return

        phase_index_by_key = {p.key: p.phase_index for p in phases}

        if before_scope == "phase":
            by_phase: Dict[str, List[Tuple[str, int, str, Optional[int]]]] = {}
            for rec in action_records:
                by_phase.setdefault(rec[0], []).append(rec)
            for phase_key, recs in by_phase.items():
                ordered = sorted(recs, key=lambda t: (t[3] if t[3] is not None else 1_000_000, t[1]))
                ids = [aid for _pk, _seq, aid, _order in ordered]
                for a, b in zip(ids, ids[1:]):
                    graph.add_edge(
                        MEdge(
                            src=a,
                            dst=b,
                            type="before",
                            attrs={"scope": "phase", "phase_key": phase_key},
                        )
                    )
            return

        ordered = sorted(
            action_records,
            key=lambda t: (
                phase_index_by_key.get(t[0], 1_000_000),
                t[3] if t[3] is not None else 1_000_000,
                t[1],
            ),
        )
        ids = [aid for _pk, _seq, aid, _order in ordered]
        for a, b in zip(ids, ids[1:]):
            graph.add_edge(MEdge(src=a, dst=b, type="before", attrs={"scope": "plan"}))


def iter_plan_graphs_from_csv(
    *,
    input_csv: str | Path,
    builder: GraphBuilder,
    forced_task: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[PlanGraph]:
    input_csv = Path(input_csv)
    with input_csv.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row_idx, raw_row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break
            row = normalise_row(raw_row)
            plan_id = str(row.get("plan_id") or row.get("id") or row_idx)
            yield builder.build_from_row(row, plan_id=plan_id, forced_task=forced_task)


def build_memory_graph_from_csv(
    csv_path: str | Path,
    *,
    config_path: str | Path | None = None,
    task: str = DEFAULT_TASK,
    limit: Optional[int] = None,
    corpus_id: str = "corpus",
) -> PlanGraph:
    """
    Backward-compatible wrapper that builds a single graph containing all plans.

    Notes:
      - Node ids are prefixed with per-row plan ids (e.g., ``plan::42::...``),
        so merging multiple plan graphs into one container is safe.
      - For the new preferred output format, use :func:`iter_plan_graphs_from_csv`
        and write JSONL via :func:`memory_graph.store.save_jsonl`.
    """

    if config_path is None:
        config_path = (INPUT_ROOT / task / "graph_config.json").resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    tasks, default_task, task_field, task_mapping = load_task_registry(config_path)
    builder = GraphBuilder(tasks, default_task=default_task, task_field=task_field, task_mapping=task_mapping)
    merged = PlanGraph(plan_id=corpus_id)
    for graph in iter_plan_graphs_from_csv(input_csv=csv_path, builder=builder, forced_task=task, limit=limit):
        for node in graph.nodes:
            merged.add_node(node)
        for edge in graph.edges:
            merged.add_edge(edge)
    return merged


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-agnostic plan graph builder (Plan→Phase→Action).")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name (also selects artifacts/input/<task>/).")
    parser.add_argument("--input", type=Path, help="Input CSV path (defaults to artifacts/input/<task>/dataset/train.csv).")
    parser.add_argument("--config", type=Path, help="Config path (defaults to artifacts/input/<task>/graph_config.json).")
    parser.add_argument("--out", type=Path, help="Output JSONL path (defaults to artifacts/output/<task>/memory_graph.jsonl).")
    parser.add_argument("--rdf-out", type=Path, help="Optional RDF N-Triples output path (defaults to <out>.nt).")
    parser.add_argument("--graphml-out", type=Path, help="Optional GraphML output path (defaults to <out>.graphml).")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to process.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    started_at = time.perf_counter()
    argv_snapshot = sys.argv[:]
    status = "ok"
    error: Optional[str] = None
    extra: Dict[str, object] = {}
    graphs_count = 0

    try:
        args = parse_args(argv)
        task_name = str(args.task or DEFAULT_TASK)
        extra.update({"task": task_name, "limit": args.limit})
        input_dir = (INPUT_ROOT / task_name).resolve()
        output_dir = (OUTPUT_ROOT / task_name).resolve()

        default_csv = input_dir / "dataset" / "train.csv"
        if not default_csv.exists():
            default_csv = input_dir / "train.csv"
        input_csv = args.input or default_csv
        if not input_csv.is_absolute():
            candidate = (input_dir / input_csv).resolve()
            input_csv = candidate if candidate.exists() else (PROJECT_ROOT / input_csv).resolve()

        config_path = args.config or (input_dir / "graph_config.json")
        if not config_path.is_absolute():
            candidate = (input_dir / config_path).resolve()
            config_path = candidate if candidate.exists() else (PROJECT_ROOT / config_path).resolve()
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")

        tasks, default_task, task_field, task_mapping = load_task_registry(config_path)
        if not tasks:
            raise SystemExit(f"No tasks found in config: {config_path}")
        if task_name and task_name not in tasks:
            raise SystemExit(f"Unknown task '{task_name}' in {config_path}. Available: {', '.join(sorted(tasks))}")

        builder = GraphBuilder(tasks, default_task=default_task, task_field=task_field, task_mapping=task_mapping)

        if not input_csv.exists():
            raise SystemExit(f"Input CSV not found: {input_csv}")

        jsonl_path = args.out or (output_dir / "memory_graph.jsonl")
        if not jsonl_path.is_absolute():
            # If the user passes just a filename, place it under the task output dir.
            # If they pass a relative path with directories, interpret it relative to the repo root.
            if jsonl_path.parent == Path("."):
                jsonl_path = (output_dir / jsonl_path).resolve()
            else:
                jsonl_path = (PROJECT_ROOT / jsonl_path).resolve()

        rdf_path = args.rdf_out or jsonl_path.with_suffix(".nt")
        if not rdf_path.is_absolute():
            if rdf_path.parent == Path("."):
                rdf_path = (output_dir / rdf_path).resolve()
            else:
                rdf_path = (PROJECT_ROOT / rdf_path).resolve()

        graphml_path = args.graphml_out or jsonl_path.with_suffix(".graphml")
        if not graphml_path.is_absolute():
            if graphml_path.parent == Path("."):
                graphml_path = (output_dir / graphml_path).resolve()
            else:
                graphml_path = (PROJECT_ROOT / graphml_path).resolve()

        extra.update(
            {
                "input_csv": str(input_csv),
                "config": str(config_path),
                "out": str(jsonl_path),
                "rdf_out": str(rdf_path),
                "graphml_out": str(graphml_path),
            }
        )

        graphs = list(
            iter_plan_graphs_from_csv(
                input_csv=input_csv,
                builder=builder,
                forced_task=task_name,
                limit=args.limit,
            )
        )
        graphs_count = len(graphs)
        store.save_jsonl(graphs, jsonl_path)
        store.save_rdf_nt(graphs, rdf_path)
        store.save_graphml(graphs, graphml_path)

        print(f"Wrote {graphs_count} plan graph(s) to {jsonl_path}")
        print(f"Wrote RDF triples to {rdf_path}")
        print(f"Wrote merged GraphML to {graphml_path}")
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, SystemExit):
            status = "exit"
            code = exc.code
            if isinstance(code, (int, float, str, bool)) or code is None:
                extra.update({"exit_code": code})
            else:
                extra.update({"exit_code": repr(code)})
        else:
            status = "error"
        error = repr(exc)
        raise
    finally:
        extra.setdefault("graphs_written", graphs_count)
        _append_run_log(argv=argv_snapshot, started_at=started_at, status=status, extra=extra, error=error)


if __name__ == "__main__":
    main()

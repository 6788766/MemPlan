"""
ComposeMatch: materialise candidates from selected views and build an AND–OR tree.

This stage consumes:
  - init-templates produced by `planner/init_template.py` (JSONL)
  - view selection output from `planner/view_select.py` (match_{split}.json)

and produces:
  - an AND–OR tree for each template (`tree_{split}.json`), where:
      * root is AND (all required slots)
      * each slot is an OR bucket of candidate actions from memory and/or tools

Design notes
  - We ignore index-like attributes (day/index/order) during matching so a shorter
    query (e.g., 3-day Boston) can match a longer memory plan (e.g., 5-day Boston),
    yielding multiple OR options per slot.
  - Task-specific knobs (ignored attrs, tool handlers) come from `artifacts/input/<task>/planner.json`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from planner.config import ConfigError, load_hook, load_planner_config
from planner.parallel import available_cpu_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"
OUTPUT_ROOT = ARTIFACTS_ROOT / "output"
_LOG_PATH = ARTIFACTS_ROOT / "logs" / "log.txt"

ToolCallStats = Dict[str, Dict[str, object]]  # tool_view_id -> {"tool": str, "cost": float, "calls": int}
_CURRENT_TOOL_CALL_STATS: Optional[ToolCallStats] = None
_TOOL_RESULT_CACHE: Optional[Dict[Tuple[str, Tuple[object, ...]], object]] = None


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
            "module": "planner.compose_match",
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


def _record_tool_call(tool_view: "ToolView") -> None:
    stats = _CURRENT_TOOL_CALL_STATS
    if stats is None:
        return
    view_id = str(getattr(tool_view, "view_id", "") or "")
    if not view_id:
        return
    entry = stats.get(view_id)
    if entry is None:
        stats[view_id] = {
            "tool": str(getattr(tool_view, "tool", "") or ""),
            "cost": float(getattr(tool_view, "cost", 0.0) or 0.0),
            "calls": 1,
        }
        return
    try:
        entry["calls"] = int(entry.get("calls") or 0) + 1
    except (TypeError, ValueError):
        entry["calls"] = 1


def _merge_tool_call_stats(dst: ToolCallStats, src: ToolCallStats) -> None:
    for view_id, entry in src.items():
        if not isinstance(entry, dict):
            continue
        try:
            calls = int(entry.get("calls") or 0)
        except (TypeError, ValueError):
            calls = 0
        if calls <= 0:
            continue

        tool = str(entry.get("tool") or "")
        try:
            cost = float(entry.get("cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0

        existing = dst.get(view_id)
        if existing is None:
            dst[view_id] = {"tool": tool, "cost": cost, "calls": calls}
            continue
        try:
            existing["calls"] = int(existing.get("calls") or 0) + calls
        except (TypeError, ValueError):
            existing["calls"] = calls
        existing.setdefault("tool", tool)
        existing.setdefault("cost", cost)


def _summarize_tool_calls(stats: ToolCallStats) -> Dict[str, object]:
    by_view: Dict[str, Dict[str, object]] = {}
    by_tool: Dict[str, Dict[str, object]] = {}
    total_calls = 0
    total_cost = 0.0

    for view_id, entry in stats.items():
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool") or "")
        try:
            calls = int(entry.get("calls") or 0)
        except (TypeError, ValueError):
            calls = 0
        try:
            cost = float(entry.get("cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        if calls <= 0:
            continue

        view_cost = float(cost) * calls
        by_view[str(view_id)] = {"tool": tool, "calls": calls, "cost": cost, "total_cost": view_cost}

        tool_entry = by_tool.setdefault(tool, {"calls": 0, "total_cost": 0.0})
        tool_entry["calls"] = int(tool_entry.get("calls") or 0) + calls
        tool_entry["total_cost"] = float(tool_entry.get("total_cost") or 0.0) + view_cost

        total_calls += calls
        total_cost += view_cost

    return {"total_calls": total_calls, "total_cost": total_cost, "by_tool": by_tool, "by_view": by_view}


EdgeKey = Tuple[str, str, str]  # (src, edge_type, dst)

# These attributes are structural indices and should not affect pattern matching.
# They are ignored for all tasks to allow templates/memories of different lengths
# or internal ordering to still match.
GLOBAL_IGNORED_ATTR_PATHS: Tuple[str, ...] = ("phase_index", "order_index")


@dataclass(frozen=True)
class Node:
    id: str
    node_type: str
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    edge_type: str
    attrs: Dict[str, object] = field(default_factory=dict)

    def key(self) -> EdgeKey:
        return (self.src, self.edge_type, self.dst)


@dataclass
class TypedGraph:
    nodes: Dict[str, Node]
    edges: List[Edge]


@dataclass(frozen=True)
class ToolView:
    view_id: str
    tool: str
    cost: float
    action_type: str
    mode: Optional[str]
    node_attrs: Dict[str, object]
    extractors: List[Dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class MatchSpec:
    """
    Matching behaviour knobs.

    - Placeholders like "{City}" are treated as null/unconstrained.
    - We compare only leaf values that are non-null in BOTH nodes
      ("intersection-of-non-null").
    """

    placeholder_regex: str = r"^\{[^{}]+\}$"
    ignored_attr_paths: Tuple[str, ...] = ()
    nullify_substrings_by_path: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    # If the raw query attribute contains any of these substrings, force tool fallback
    # even when memory candidates exist (e.g., travel multi-city placeholders).
    force_fallback_substrings_by_path: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    # Optional context propagation (e.g., propagate plan.dest into action leaves).
    context_from_node_type: str = "Plan"
    context_attr_paths: Tuple[str, ...] = ()
    context_namespace: str = "plan"
    context_apply_to_node_types: Tuple[str, ...] = ()

def _spec_from_planner_cfg(cfg) -> MatchSpec:
    return MatchSpec(
        placeholder_regex=str(cfg.match.placeholder_regex),
        ignored_attr_paths=tuple(cfg.match.ignored_attr_paths),
        nullify_substrings_by_path=cfg.match.nullify_substrings_by_path,
        force_fallback_substrings_by_path=cfg.match.force_fallback_substrings_by_path,
        context_from_node_type=str(cfg.match.context.from_node_type),
        context_attr_paths=tuple(cfg.match.context.attr_paths),
        context_namespace=str(cfg.match.context.namespace),
        context_apply_to_node_types=tuple(cfg.match.context.apply_to_node_types),
    )


def _spec_from_config(task: str, *, config_path: Optional[Path] = None) -> MatchSpec:
    cfg = load_planner_config(task, config_path=config_path)
    return _spec_from_planner_cfg(cfg)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSON object per line in {path}")
            yield obj


def _edge_key_to_string(edge: EdgeKey) -> str:
    src, etype, dst = edge
    return f"{src} -[{etype}]-> {dst}"


def _build_query_graph_from_init_template(template: Mapping[str, object]) -> TypedGraph:
    template_id = str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE")
    plan_node_id = f"plan::{template_id}"

    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    plan_attrs = template.get("plan") or {}
    nodes[plan_node_id] = Node(
        id=plan_node_id,
        node_type="Plan",
        attrs=dict(plan_attrs) if isinstance(plan_attrs, dict) else {},
    )

    phases_payload = template.get("phases") or []
    phase_key_to_node: Dict[str, str] = {}
    if isinstance(phases_payload, list):
        for idx, phase in enumerate(phases_payload, start=1):
            if not isinstance(phase, dict):
                continue
            phase_key = str(phase.get("phase_key") or f"phase_{idx}")
            node_id = f"{plan_node_id}::phase::{phase_key}"
            attrs: Dict[str, object] = {
                "phase_key": phase_key,
                "phase_type": phase.get("phase_type"),
                "phase_index": phase.get("phase_index"),
            }
            extra = phase.get("attrs") or {}
            if isinstance(extra, dict):
                attrs.update(extra)
            nodes[node_id] = Node(id=node_id, node_type="Phase", attrs=attrs)
            phase_key_to_node[phase_key] = node_id
            edges.append(Edge(src=plan_node_id, dst=node_id, edge_type="hasPhase", attrs={}))

    actions_payload = template.get("actions") or []
    if isinstance(actions_payload, list):
        for idx, action in enumerate(actions_payload, start=1):
            if not isinstance(action, dict):
                continue
            action_id = f"{plan_node_id}::action::{idx}"
            action_type = action.get("action_type")
            params = action.get("params") or {}
            action_attrs = action.get("attrs") or {}

            node_attrs: Dict[str, object] = {
                "action_type": action_type,
                "params": dict(params) if isinstance(params, dict) else {},
            }
            if isinstance(action_attrs, dict):
                node_attrs.update(action_attrs)
            # init_template stores order_index at top-level of action object.
            if "order_index" in action:
                node_attrs["order_index"] = action.get("order_index")

            nodes[action_id] = Node(id=action_id, node_type="Action", attrs=node_attrs)

            phase_key = str(action.get("phase_key") or "")
            phase_node_id = phase_key_to_node.get(phase_key)
            if not phase_node_id:
                phase_node_id = next(iter(phase_key_to_node.values()), None)
            if phase_node_id:
                edges.append(Edge(src=phase_node_id, dst=action_id, edge_type="hasAction", attrs={}))

    return TypedGraph(nodes=nodes, edges=edges)


def _load_memory_graphs(path: Path) -> Dict[str, TypedGraph]:
    graphs: Dict[str, TypedGraph] = {}
    for obj in _iter_jsonl(path):
        plan_id = str(obj.get("plan_id") or obj.get("id") or "")
        if not plan_id:
            continue
        nodes_payload = obj.get("nodes") or []
        edges_payload = obj.get("edges") or []
        if not isinstance(nodes_payload, list) or not isinstance(edges_payload, list):
            continue

        nodes: Dict[str, Node] = {}
        for node_data in nodes_payload:
            if not isinstance(node_data, dict):
                continue
            node_id = str(node_data.get("id") or "")
            node_type = str(node_data.get("type") or "")
            if not node_id or not node_type:
                continue
            attrs = node_data.get("attrs") or {}
            nodes[node_id] = Node(id=node_id, node_type=node_type, attrs=dict(attrs) if isinstance(attrs, dict) else {})

        edges: List[Edge] = []
        for edge_data in edges_payload:
            if not isinstance(edge_data, dict):
                continue
            src = str(edge_data.get("src") or "")
            dst = str(edge_data.get("dst") or "")
            edge_type = str(edge_data.get("type") or "")
            if not src or not dst or not edge_type:
                continue
            attrs = edge_data.get("attrs") or {}
            edges.append(Edge(src=src, dst=dst, edge_type=edge_type, attrs=dict(attrs) if isinstance(attrs, dict) else {}))

        graphs[plan_id] = TypedGraph(nodes=nodes, edges=edges)
    return graphs


def _load_tool_views(path: Path) -> Dict[str, ToolView]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views_payload = payload.get("views") or []
    if not isinstance(views_payload, list):
        raise TypeError(f"tool.json 'views' must be a list: {path}")

    views: Dict[str, ToolView] = {}
    for item in views_payload:
        if not isinstance(item, dict):
            continue
        view_id = str(item.get("view_id") or "")
        tool = str(item.get("tool") or "")
        cost = float(item.get("cost") or 0.0)
        node_pattern = item.get("node_pattern") or {}
        if not isinstance(node_pattern, dict):
            continue
        attrs = node_pattern.get("attrs") or {}
        if not isinstance(attrs, dict):
            continue
        action_type = str(attrs.get("action_type") or "")
        if not action_type:
            continue
        mode = attrs.get("mode")
        mode_str = str(mode) if isinstance(mode, str) and mode.strip() else None
        extractors = item.get("extractors") or []
        extractors_list = extractors if isinstance(extractors, list) else []
        views[view_id] = ToolView(
            view_id=view_id,
            tool=tool,
            cost=cost,
            action_type=action_type,
            mode=mode_str,
            node_attrs=dict(attrs),
            extractors=[e for e in extractors_list if isinstance(e, dict)],
        )
    return views


def _iter_leaves(value: object, path: str) -> Iterator[Tuple[str, object]]:
    if value is None:
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield from _iter_leaves(sub, next_path)
        return
    if isinstance(value, list):
        if not value:
            return
        yield path, tuple(value)
        return
    yield path, value


def _get_nested_attr(attrs: Mapping[str, object], path: str) -> object:
    current: object = attrs
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _force_tool_fallback_for_action(action_attrs: Mapping[str, object], *, spec: MatchSpec) -> bool:
    for path, needles in spec.force_fallback_substrings_by_path.items():
        raw = _get_nested_attr(action_attrs, path)
        if not isinstance(raw, str):
            continue
        for needle in needles:
            if needle and needle in raw:
                return True
    return False


def _maybe_derive_force_fallback_spec(
    spec: MatchSpec,
    *,
    tool_views: Mapping[str, "ToolView"],
    placeholder_engine: Optional["PlaceholderEngine"],
) -> MatchSpec:
    """
    If `match.force_fallback_substrings_by_path` is not configured, derive it from tool.json:
    use all Action `params.*` leaves present in tool view patterns, and reuse the same
    placeholder substrings configured for nullification.

    This avoids duplicating parameter-path lists across `planner.json` and `views/tool.json`.
    """

    if spec.force_fallback_substrings_by_path:
        return spec
    if placeholder_engine is None:
        return spec

    needles_set: Set[str] = set()
    for needles in spec.nullify_substrings_by_path.values():
        for needle in needles:
            if isinstance(needle, str) and needle:
                needles_set.add(needle)
    if not needles_set:
        return spec

    param_paths: Set[str] = set()
    for view in tool_views.values():
        for path, value in _iter_leaves(view.node_attrs, ""):
            if not path.startswith("params."):
                continue
            if isinstance(value, str):
                param_paths.add(path)
    if not param_paths:
        return spec

    needles = tuple(sorted(needles_set))
    derived = {path: needles for path in sorted(param_paths)}
    return replace(spec, force_fallback_substrings_by_path=derived)


def _normalise_leaf_value(path: str, value: object, *, spec: MatchSpec, placeholder_re: re.Pattern[str]) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if placeholder_re.match(text):
            return None
        substrings = spec.nullify_substrings_by_path.get(path)
        if substrings:
            for needle in substrings:
                if needle in text:
                    return None
        return text
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _leaf_map(attrs: Mapping[str, object], *, spec: MatchSpec, placeholder_re: re.Pattern[str]) -> Dict[str, object]:
    leaves: Dict[str, object] = {}
    for path, raw in _iter_leaves(attrs, ""):
        if path in GLOBAL_IGNORED_ATTR_PATHS or path in spec.ignored_attr_paths:
            continue
        norm = _normalise_leaf_value(path, raw, spec=spec, placeholder_re=placeholder_re)
        if norm is None:
            continue
        leaves[path] = norm
    return leaves


def _leaf_maps_compatible(pattern_leaves: Mapping[str, object], query_leaves: Mapping[str, object]) -> bool:
    for path in pattern_leaves.keys() & query_leaves.keys():
        if pattern_leaves[path] != query_leaves[path]:
            return False
    return True


def _graph_context_leaves(graph: TypedGraph, *, spec: MatchSpec, placeholder_re: re.Pattern[str]) -> Dict[str, object]:
    if not spec.context_attr_paths:
        return {}
    context_node = next((node for node in graph.nodes.values() if node.node_type == spec.context_from_node_type), None)
    if context_node is None:
        return {}
    context_leaves = _leaf_map(context_node.attrs, spec=spec, placeholder_re=placeholder_re)
    namespaced: Dict[str, object] = {}
    for path, value in context_leaves.items():
        if path in spec.context_attr_paths:
            namespaced[f"{spec.context_namespace}.{path}"] = value
    return namespaced


def _node_leaves_with_context(
    node: Node,
    *,
    graph_context: Mapping[str, object],
    spec: MatchSpec,
    placeholder_re: re.Pattern[str],
) -> Dict[str, object]:
    leaves = _leaf_map(node.attrs, spec=spec, placeholder_re=placeholder_re)
    if node.node_type in spec.context_apply_to_node_types:
        for path, value in graph_context.items():
            leaves[path] = value
    return leaves


def _candidate_key(candidate: Mapping[str, object]) -> str:
    source = str(candidate.get("source") or "")
    text = str(candidate.get("text") or "")
    return f"{source}::{text}"


def _to_builtin(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    # pandas / numpy scalars often string-format nicely; keep deterministic.
    return str(value)


def _tool_result_cache_key(tool_view_id: str, args: Sequence[object]) -> Tuple[str, Tuple[object, ...]]:
    key_parts: List[object] = []
    for value in args:
        try:
            hash(value)
        except TypeError:
            key_parts.append(json.dumps(_to_builtin(value), ensure_ascii=False, sort_keys=True))
        else:
            key_parts.append(value)
    return (tool_view_id, tuple(key_parts))


def _render_action_text(attrs: Mapping[str, object]) -> str:
    action_type = str(attrs.get("action_type") or "")
    params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
    if action_type == "Move":
        origin = params.get("origin")
        dest = params.get("destination")
        mode = attrs.get("mode")
        if origin and dest:
            if mode:
                return f"{mode}: {origin} -> {dest}"
            return f"Move: {origin} -> {dest}"
        return "Move"
    if action_type == "Eat":
        name = params.get("Name")
        city = params.get("City")
        slot = attrs.get("slot")
        if name and city:
            return f"{slot or 'Eat'}: {name}, {city}"
        return f"{slot or 'Eat'}"
    if action_type == "Stay":
        name = params.get("NAME")
        city = params.get("city")
        if name and city:
            return f"Stay: {name}, {city}"
        return "Stay"
    if action_type == "Visit":
        name = params.get("Name")
        city = params.get("City")
        if name and city:
            return f"Visit: {name}, {city}"
        return "Visit"
    if action_type == "DayRecord":
        current = params.get("current_city")
        if current:
            return f"DayRecord: {current}"
        return "DayRecord"
    return action_type or "Action"


_TEMPLATE_KEY_RE = re.compile(r"\{([^{}]+)\}")


def _format_text(template: str, context: Mapping[str, object]) -> str:
    if not template:
        return ""
    text = str(template)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = context.get(key)
        if value is None:
            return ""
        return str(value)

    return _TEMPLATE_KEY_RE.sub(_replace, text).strip()


@dataclass(frozen=True)
class PlaceholderEngine:
    token_re: re.Pattern[str]
    state_city_map: Dict[str, List[str]]
    max_expansions: int
    format_template: str


def _load_state_city_map(path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        city = parts[0].strip()
        state = parts[1].strip()
        if not city or not state:
            continue
        mapping.setdefault(state, []).append(city)
    return mapping


def _load_placeholder_engine(compose_cfg) -> Optional[PlaceholderEngine]:
    pe = getattr(compose_cfg, "placeholder_expansion", None)
    if pe is None or not bool(getattr(pe, "enabled", False)):
        return None
    pattern = str(getattr(pe, "pattern", "") or "")
    map_path = getattr(pe, "state_city_map_path", None)
    if not pattern or not isinstance(map_path, str) or not map_path.strip():
        return None
    path = Path(map_path.strip())
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        return None
    token_re = re.compile(pattern)
    state_city_map = _load_state_city_map(path)
    max_expansions = int(getattr(pe, "max_expansions", 0) or 0)
    fmt = str(getattr(pe, "format", "{city}({state})") or "{city}({state})")
    return PlaceholderEngine(token_re=token_re, state_city_map=state_city_map, max_expansions=max_expansions, format_template=fmt)


def _extract_placeholder_tokens(texts: Sequence[str], engine: PlaceholderEngine) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for text in texts:
        for match in engine.token_re.finditer(text):
            token = match.group(0)
            if token in seen:
                continue
            seen.add(token)
            state = match.groupdict().get("state", "")
            if isinstance(state, str):
                state = state.strip()
            else:
                state = ""
            tokens.append((token, state))
    return tokens


def _expand_placeholder_assignments(tokens: Sequence[Tuple[str, str]], engine: PlaceholderEngine) -> List[Dict[str, str]]:
    if not tokens:
        return []
    limit: Optional[int] = engine.max_expansions if engine.max_expansions and engine.max_expansions > 0 else None

    choices: List[Tuple[str, str, List[str]]] = []
    for token, state in tokens:
        cities = engine.state_city_map.get(state) or []
        if not cities:
            return []
        choices.append((token, state, cities if limit is None else cities[:limit]))

    results: List[Dict[str, str]] = []
    used_by_state: Dict[str, Set[str]] = {}
    assignment: Dict[str, str] = {}

    def backtrack(idx: int) -> None:
        if limit is not None and len(results) >= limit:
            return
        if idx >= len(choices):
            results.append(dict(assignment))
            return

        token, state, cities = choices[idx]
        used = used_by_state.setdefault(state, set())
        for city in cities:
            if city in used:
                continue
            used.add(city)
            assignment[token] = engine.format_template.format(city=city, state=state)
            backtrack(idx + 1)
            assignment.pop(token, None)
            used.remove(city)
            if limit is not None and len(results) >= limit:
                break

    backtrack(0)
    return results


def _apply_assignments(text: str, assignments: Mapping[str, str]) -> str:
    updated = text
    for token, replacement in assignments.items():
        if not token or not replacement:
            continue
        updated = updated.replace(token, replacement)
    return updated


def _get_by_path(value: object, parts: Sequence[str]) -> object:
    current: object = value
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _ref_value(
    ref: str,
    *,
    query_plan: Node,
    query_phase: Node,
    query_action: Node,
    tool_view: ToolView,
) -> object:
    raw = str(ref or "").strip()
    if not raw:
        return None
    parts = raw.split(".")
    root = parts[0]
    rest = parts[1:]
    if root == "plan":
        return _get_by_path(query_plan.attrs, rest)
    if root == "phase":
        return _get_by_path(query_phase.attrs, rest)
    if root == "action":
        return _get_by_path(query_action.attrs, rest)
    if root == "view":
        view_ctx = {
            "mode": tool_view.mode,
            "tool": tool_view.tool,
            "view_id": tool_view.view_id,
            "action_type": tool_view.action_type,
        }
        return _get_by_path(view_ctx, rest)
    return None


def _first_nonempty_ref(
    refs: Sequence[object],
    *,
    query_plan: Node,
    query_phase: Node,
    query_action: Node,
    tool_view: ToolView,
) -> object:
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            continue
        value = _ref_value(ref, query_plan=query_plan, query_phase=query_phase, query_action=query_action, tool_view=tool_view)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        return value
    return None


ToolHandlerKey = Tuple[str, str]  # (tool, action_type)


@dataclass(frozen=True)
class ToolHandlerIndex:
    by_view_id: Dict[str, Dict[str, object]] = field(default_factory=dict)
    by_tool_action: Dict[ToolHandlerKey, Dict[str, object]] = field(default_factory=dict)


def _build_tool_handler_index(
    tool_handlers: Sequence[Mapping[str, object]],
    *,
    tool_views: Optional[Mapping[str, "ToolView"]] = None,
) -> ToolHandlerIndex:
    """
    Build an index for looking up tool handlers without duplicating tool.json fields.

    Supported handler entry forms:
      - {"tool_view_id": "..."} or {"tool_view_ids": ["..."]}  -> matched by tool view_id
      - {"tool": "...", "action_type": "..."}                  -> matched by (tool, action_type)
      - {"tool": "..."}                                        -> if tool maps to exactly one action_type in tool.json
    """

    by_view_id: Dict[str, Dict[str, object]] = {}
    by_tool_action: Dict[ToolHandlerKey, Dict[str, object]] = {}

    tool_to_action_types: Dict[str, Set[str]] = {}
    if tool_views is not None:
        for view in tool_views.values():
            tool = str(getattr(view, "tool", "") or "").strip()
            action_type = str(getattr(view, "action_type", "") or "").strip()
            if tool and action_type:
                tool_to_action_types.setdefault(tool, set()).add(action_type)

    for raw in tool_handlers:
        if not isinstance(raw, dict):
            continue

        view_ids_raw = raw.get("tool_view_ids")
        if isinstance(view_ids_raw, list) and view_ids_raw:
            for view_id in view_ids_raw:
                if not isinstance(view_id, str) or not view_id.strip():
                    continue
                by_view_id[view_id.strip()] = dict(raw)
            continue

        view_id = raw.get("tool_view_id")
        if isinstance(view_id, str) and view_id.strip():
            by_view_id[view_id.strip()] = dict(raw)
            continue

        tool = str(raw.get("tool") or "").strip()
        action_type = str(raw.get("action_type") or "").strip()
        if tool and action_type:
            by_tool_action[(tool, action_type)] = dict(raw)
            continue

        if tool and tool_views is not None:
            action_types = tool_to_action_types.get(tool) or set()
            if len(action_types) == 1:
                inferred = next(iter(action_types))
                by_tool_action[(tool, inferred)] = dict(raw)

    return ToolHandlerIndex(by_view_id=by_view_id, by_tool_action=by_tool_action)


def _assignments_for_texts(texts: Sequence[str], engine: Optional[PlaceholderEngine]) -> List[Dict[str, str]]:
    if engine is None:
        return [{}]
    tokens = _extract_placeholder_tokens(texts, engine)
    if not tokens:
        return [{}]
    return _expand_placeholder_assignments(tokens, engine)


def _required_assignments_for_texts(texts: Sequence[str], engine: Optional[PlaceholderEngine]) -> List[Dict[str, str]]:
    if engine is None:
        return []
    tokens = _extract_placeholder_tokens(texts, engine)
    if not tokens:
        return []
    return _expand_placeholder_assignments(tokens, engine)


def _tool_candidates(
    *,
    tool_view: ToolView,
    query_plan: Node,
    query_phase: Node,
    query_action: Node,
    fetch: Optional[object],
    handler_index: ToolHandlerIndex,
    placeholder_engine: Optional[PlaceholderEngine],
    mode_map: Mapping[str, str],
    max_candidates: int,
) -> List[Dict[str, object]]:
    action_type = str(query_action.attrs.get("action_type") or "").strip()
    if tool_view.action_type != action_type:
        return []

    handler = handler_index.by_view_id.get(tool_view.view_id) or handler_index.by_tool_action.get((tool_view.tool, action_type))

    kind = str(handler.get("kind") or "").strip() if handler is not None else ""
    meta_type = (str(handler.get("meta_type") or "").strip() if handler is not None else "") or "tool"
    fetch_method = handler.get("fetch_method") if handler is not None else None
    fetch_name = str(fetch_method or "").strip()

    args_raw: Dict[str, object] = {}
    if handler is not None:
        args_raw = handler.get("args") or {}
        if not isinstance(args_raw, dict):
            args_raw = {}

    arg_names: List[str] = []
    if handler is not None:
        arg_order_raw = handler.get("arg_order")
        if isinstance(arg_order_raw, list) and arg_order_raw:
            arg_names = [str(x) for x in arg_order_raw if isinstance(x, str) and x.strip()]
        else:
            arg_names = [str(k) for k in args_raw.keys()]

    raw_args: Dict[str, object] = {}
    if handler is not None:
        for arg_name in arg_names:
            refs = args_raw.get(arg_name) if isinstance(args_raw.get(arg_name), list) else []
            value = _first_nonempty_ref(
                refs,
                query_plan=query_plan,
                query_phase=query_phase,
                query_action=query_action,
                tool_view=tool_view,
            )
            if value is None:
                return []
            raw_args[arg_name] = value

    # Default (no handler): if the task does not provide a fetch factory, emit a
    # single direct tool-call candidate using the query action params.
    if handler is None and fetch is None:
        params_pattern = tool_view.node_attrs.get("params")
        if isinstance(params_pattern, dict) and params_pattern:
            arg_names = list(params_pattern.keys())
        else:
            arg_names = []
        query_params = query_action.attrs.get("params") if isinstance(query_action.attrs.get("params"), dict) else {}
        raw_args = {name: query_params.get(name) for name in arg_names if name in query_params}
        kind = "direct_call"

    # Special kind: placeholder expansion only (no tool call).
    if kind == "placeholder_expansion":
        current_city_raw = raw_args.get("current_city")
        if not isinstance(current_city_raw, str) or not current_city_raw.strip():
            return []
        assignments_list = _required_assignments_for_texts([current_city_raw], placeholder_engine)
        if not assignments_list:
            return []
        candidates: List[Dict[str, object]] = []
        for assignments in assignments_list:
            filled = _apply_assignments(current_city_raw, assignments)
            meta = {
                "type": meta_type,
                "raw_current_city": current_city_raw,
                "current_city": filled,
                "assignments": dict(assignments),
            }
            text_template = str(handler.get("text_template") or "").strip()
            text = _format_text(text_template, {"current_city": filled})
            candidates.append(
                {
                    "source": "tool",
                    "tool": tool_view.tool,
                    "tool_view_id": tool_view.view_id,
                    "tool_call_cost": tool_view.cost,
                    "text": text or filled,
                    "cost": 0.0,
                    "meta": meta,
                }
            )
        return candidates

    # Special kind: direct tool call candidate without executing the tool.
    # This is useful for tasks where tool execution is side-effectful (or expensive)
    # during candidate generation, but we still want a concrete function-call candidate.
    if kind == "direct_call":
        qualified = tool_view.node_attrs.get("qualified_tool_name")
        if not isinstance(qualified, str) or not qualified.strip():
            qualified = f"{tool_view.tool}.{action_type}"

        side_effect = tool_view.node_attrs.get("side_effect")
        side_effect_flag = bool(side_effect) if isinstance(side_effect, bool) else None

        def _quote(v: object) -> str:
            # WorkBench gold traces quote all argument values as strings.
            text = str(v) if v is not None else ""
            text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{text}"'

        args_parts: List[str] = []
        meta_args: Dict[str, str] = {}
        for arg_name in arg_names:
            raw_val = raw_args.get(arg_name)
            if raw_val is None:
                continue
            val = str(raw_val)
            meta_args[arg_name] = val
            args_parts.append(f"{arg_name}={_quote(val)}")
        call_str = f"{qualified}.func(" + ", ".join(args_parts) + ")"

        meta: Dict[str, object] = {
            "type": meta_type or "tool_call",
            "tool": tool_view.tool,
            "qualified_tool_name": qualified,
            "args": dict(meta_args),
        }
        if side_effect_flag is not None:
            meta["side_effect"] = side_effect_flag

        return [
            {
                "source": "tool",
                "tool": tool_view.tool,
                "tool_view_id": tool_view.view_id,
                "tool_call_cost": tool_view.cost,
                "text": call_str,
                "cost": 0.0,
                "meta": meta,
            }
        ]

    if fetch is None or not fetch_name:
        return []
    fetch_fn = getattr(fetch, fetch_name, None)
    if not callable(fetch_fn):
        return []

    # Expand CITY_k(State) placeholders consistently across all string arguments.
    arg_texts = [str(v) for v in raw_args.values() if isinstance(v, str)]
    assignments_list = _assignments_for_texts(arg_texts, placeholder_engine)
    if not assignments_list:
        return []

    limit: Optional[int] = max_candidates if max_candidates and max_candidates > 0 else None
    sort_by = handler.get("sort_by")
    sort_cols = [str(x) for x in sort_by] if isinstance(sort_by, list) else []
    cost_field = str(handler.get("cost_field") or "").strip()
    text_template = str(handler.get("text_template") or "").strip()
    text_fallback = str(handler.get("text_fallback") or "").strip()

    candidates: List[Dict[str, object]] = []
    for assignments in assignments_list:
        meta_args: Dict[str, object] = {}
        fetch_args: List[object] = []
        for arg_name in arg_names:
            raw_val = raw_args.get(arg_name)
            filled_val = _apply_assignments(raw_val, assignments) if isinstance(raw_val, str) else raw_val
            meta_args[arg_name] = filled_val
            if arg_name == "mode" and isinstance(filled_val, str) and filled_val in mode_map:
                fetch_args.append(mode_map[filled_val])
            else:
                fetch_args.append(filled_val)

        cache = _TOOL_RESULT_CACHE
        cache_key = _tool_result_cache_key(tool_view.view_id, fetch_args)
        if cache is not None and cache_key in cache:
            result = cache[cache_key]
        else:
            _record_tool_call(tool_view)
            result = fetch_fn(*fetch_args)
            if cache is not None:
                cache[cache_key] = result

        if kind == "dict_single":
            if not isinstance(result, dict) or not result:
                continue
            info_built = _to_builtin(result)
            if not isinstance(info_built, dict):
                continue
            duration = info_built.get("duration")
            distance = info_built.get("distance")
            cost = info_built.get("cost")
            if cost is None or duration is None or distance is None:
                continue
            cost_num = float(cost) if isinstance(cost, (int, float)) else 0.0
            context = dict(info_built)
            context.update(meta_args)
            text = _format_text(text_template, context)
            meta = {"type": meta_type, "info": info_built, "assignments": dict(assignments)}
            meta.update({k: v for k, v in meta_args.items() if isinstance(k, str)})
            candidates.append(
                {
                    "source": "tool",
                    "tool": tool_view.tool,
                    "tool_view_id": tool_view.view_id,
                    "tool_call_cost": tool_view.cost,
                    "text": text,
                    "cost": cost_num,
                    "meta": meta,
                }
            )
            continue

        # DataFrame-backed handlers.
        df = result
        if isinstance(df, str) or df is None or getattr(df, "empty", False):
            continue

        if kind == "dataframe_column":
            column = str(handler.get("column") or "").strip()
            if not column:
                continue
            series = None
            try:
                series = df[column]
            except Exception:
                series = None
            if series is None:
                continue
            if limit is not None:
                try:
                    series = series.head(limit)
                except Exception:
                    pass
            values: List[object] = []
            try:
                values = series.tolist()
            except Exception:
                try:
                    values = list(series)
                except Exception:
                    values = []
            city = meta_args.get("city")
            for value in values:
                name = str(value).strip()
                if not name:
                    continue
                row = {column: name}
                if isinstance(city, str) and city:
                    row.setdefault("City", city)
                meta = {"type": meta_type, "row": row, "assignments": dict(assignments)}
                meta.update({k: v for k, v in meta_args.items() if isinstance(k, str)})
                context = dict(row)
                context.update(meta_args)
                context.setdefault("city", city)
                text = _format_text(text_template, context)
                candidates.append(
                    {
                        "source": "tool",
                        "tool": tool_view.tool,
                        "tool_view_id": tool_view.view_id,
                        "tool_call_cost": tool_view.cost,
                        "text": text,
                        "cost": 0.0,
                        "meta": meta,
                    }
                )
            continue

        if kind != "dataframe_rows":
            continue

        df_sorted = df
        if sort_cols:
            try:
                df_sorted = df.sort_values(by=sort_cols)
            except Exception:
                df_sorted = df

        if limit is not None:
            try:
                df_sorted = df_sorted.head(limit)
            except Exception:
                pass

        try:
            iterator = df_sorted.iterrows()
        except Exception:
            iterator = []

        for _, row in iterator:
            try:
                row_dict = _to_builtin(dict(row))
            except Exception:
                continue
            if not isinstance(row_dict, dict):
                continue
            raw_cost = row_dict.get(cost_field) if cost_field else None
            cost_num = float(raw_cost) if isinstance(raw_cost, (int, float)) else 0.0
            context = dict(row_dict)
            context.update(meta_args)
            if "city" in meta_args and "city" not in context:
                context["city"] = meta_args.get("city")
            text = _format_text(text_template, context)
            if text_fallback and (not text or text.strip() in {",", "-"} or text.strip().startswith(",")):
                text = _format_text(text_fallback, context)
            meta = {"type": meta_type, "row": row_dict, "assignments": dict(assignments)}
            meta.update({k: v for k, v in meta_args.items() if isinstance(k, str)})
            candidates.append(
                {
                    "source": "tool",
                    "tool": tool_view.tool,
                    "tool_view_id": tool_view.view_id,
                    "tool_call_cost": tool_view.cost,
                    "text": text,
                    "cost": cost_num,
                    "meta": meta,
                }
            )

    return candidates


def _fallback_tool_views_for_action(
    *, action_type: str, tool_views: Mapping[str, ToolView], fallback_tools: Mapping[str, Tuple[str, ...]], tool_priority: Mapping[str, int]
) -> List[ToolView]:
    tools = fallback_tools.get(action_type) or ()
    if tools:
        tool_set = set(tools)
        selected = [view for view in tool_views.values() if view.action_type == action_type and view.tool in tool_set]
    else:
        # Sensible default: if the task did not specify fallback tools for an action_type,
        # allow any tool micro-view that matches the action type.
        selected = [view for view in tool_views.values() if view.action_type == action_type]

    def _sort_key(view: ToolView) -> Tuple[int, str, str]:
        pri = int(tool_priority.get(view.tool, 99))
        return (pri, str(view.tool), str(view.mode or ""))

    return sorted(selected, key=_sort_key)


# ---------------------------------------------------------------------------
# Parallel worker helpers
# ---------------------------------------------------------------------------


_WORKER_TASK: Optional[str] = None
_WORKER_MEMORY_GRAPHS: Optional[Dict[str, TypedGraph]] = None
_WORKER_TOOL_VIEWS: Optional[Dict[str, ToolView]] = None
_WORKER_FETCH: Optional[object] = None
_WORKER_MAX_TOOL_CANDIDATES: int = 0
_WORKER_NO_TOOLS: bool = False
_WORKER_SPEC: Optional[MatchSpec] = None
_WORKER_HANDLER_INDEX: Optional[ToolHandlerIndex] = None
_WORKER_PLACEHOLDER_ENGINE: Optional[PlaceholderEngine] = None
_WORKER_MODE_MAP: Dict[str, str] = {}
_WORKER_FALLBACK_TOOLS: Dict[str, Tuple[str, ...]] = {}
_WORKER_FALLBACK_TOOL_PRIORITY: Dict[str, int] = {}


def _compose_match_worker_init(
    task: str,
    config_path: Optional[str],
    memory_path: str,
    tool_views_path: str,
    no_tools: bool,
    max_tool_candidates: int,
    no_tool_cache: bool,
) -> None:
    """
    Initialiser for `ProcessPoolExecutor` workers.

    Each worker loads config + memory graphs + tool views once, so per-template jobs
    only do matching + candidate materialisation.
    """

    global _WORKER_TASK
    global _WORKER_MEMORY_GRAPHS
    global _WORKER_TOOL_VIEWS
    global _WORKER_FETCH
    global _WORKER_MAX_TOOL_CANDIDATES
    global _WORKER_NO_TOOLS
    global _WORKER_SPEC
    global _WORKER_HANDLER_INDEX
    global _WORKER_PLACEHOLDER_ENGINE
    global _WORKER_MODE_MAP
    global _WORKER_FALLBACK_TOOLS
    global _WORKER_FALLBACK_TOOL_PRIORITY
    global _TOOL_RESULT_CACHE

    _WORKER_TASK = str(task)
    _WORKER_MAX_TOOL_CANDIDATES = int(max_tool_candidates)
    _WORKER_NO_TOOLS = bool(no_tools)
    _TOOL_RESULT_CACHE = None if bool(no_tool_cache) else {}

    cfg_path = Path(config_path) if isinstance(config_path, str) and config_path else None
    planner_cfg = load_planner_config(_WORKER_TASK, config_path=cfg_path)
    _WORKER_PLACEHOLDER_ENGINE = _load_placeholder_engine(planner_cfg.compose)
    _WORKER_MODE_MAP = dict(planner_cfg.compose.mode_map)
    _WORKER_FALLBACK_TOOLS = dict(planner_cfg.compose.fallback_tools_by_action_type)
    _WORKER_FALLBACK_TOOL_PRIORITY = dict(planner_cfg.compose.fallback_tool_priority)

    mem_path = Path(memory_path)
    views_path = Path(tool_views_path)
    _WORKER_MEMORY_GRAPHS = _load_memory_graphs(mem_path)
    _WORKER_TOOL_VIEWS = _load_tool_views(views_path)
    _WORKER_SPEC = _maybe_derive_force_fallback_spec(
        _spec_from_planner_cfg(planner_cfg),
        tool_views=_WORKER_TOOL_VIEWS,
        placeholder_engine=_WORKER_PLACEHOLDER_ENGINE,
    )
    _WORKER_HANDLER_INDEX = _build_tool_handler_index(planner_cfg.compose.tool_handlers, tool_views=_WORKER_TOOL_VIEWS)

    _WORKER_FETCH = None
    if not _WORKER_NO_TOOLS and planner_cfg.compose.fetch_factory:
        fetch_factory = load_hook(planner_cfg.compose.fetch_factory)
        _WORKER_FETCH = fetch_factory()


def _compose_match_worker_build_tree_json(
    template: Mapping[str, object],
    match_entry: Mapping[str, object],
) -> Tuple[str, str, ToolCallStats]:
    """
    Worker entrypoint: builds one tree and returns its JSON serialisation.

    Returning JSON text (rather than the full Python object) keeps the IPC
    overhead lower when candidate pools are large.
    """

    global _CURRENT_TOOL_CALL_STATS

    template_id = str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE")
    per_template_stats: ToolCallStats = {}
    prev_stats = _CURRENT_TOOL_CALL_STATS
    _CURRENT_TOOL_CALL_STATS = per_template_stats
    try:
        tree = _build_and_or_tree_for_template(
            template=template,
            match_entry=match_entry,
            memory_graphs=_WORKER_MEMORY_GRAPHS or {},
            tool_views=_WORKER_TOOL_VIEWS or {},
            task=str(_WORKER_TASK or ""),
            max_tool_candidates=int(_WORKER_MAX_TOOL_CANDIDATES),
            fetch=_WORKER_FETCH,
            spec=_WORKER_SPEC or MatchSpec(),
            handler_index=_WORKER_HANDLER_INDEX or ToolHandlerIndex(),
            placeholder_engine=_WORKER_PLACEHOLDER_ENGINE,
            mode_map=_WORKER_MODE_MAP,
            fallback_tools=_WORKER_FALLBACK_TOOLS,
            fallback_tool_priority=_WORKER_FALLBACK_TOOL_PRIORITY,
        )
    finally:
        _CURRENT_TOOL_CALL_STATS = prev_stats
    return template_id, json.dumps(tree, ensure_ascii=False), per_template_stats


def _mark_candidates_fallback(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    for cand in candidates:
        meta = cand.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        meta["fallback"] = True
        cand["meta"] = meta
    return candidates


def _build_and_or_tree_for_template(
    *,
    template: Mapping[str, object],
    match_entry: Mapping[str, object],
    memory_graphs: Mapping[str, TypedGraph],
    tool_views: Mapping[str, ToolView],
    task: str,
    max_tool_candidates: int,
    fetch: Optional[object],
    spec: MatchSpec,
    handler_index: ToolHandlerIndex,
    placeholder_engine: Optional[PlaceholderEngine],
    mode_map: Mapping[str, str],
    fallback_tools: Mapping[str, Tuple[str, ...]],
    fallback_tool_priority: Mapping[str, int],
) -> Dict[str, object]:
    placeholder_re = re.compile(spec.placeholder_regex)

    query_graph = _build_query_graph_from_init_template(template)

    lambda_map = match_entry.get("lambda") or {}
    if not isinstance(lambda_map, dict):
        lambda_map = {}

    plan_node = next((n for n in query_graph.nodes.values() if n.node_type == "Plan"), None)
    if plan_node is None:
        plan_node = Node(id="plan::UNKNOWN", node_type="Plan", attrs={})

    # Group action slots under phases (AND structure); each slot becomes an OR node.
    phase_nodes = {node_id: node for node_id, node in query_graph.nodes.items() if node.node_type == "Phase"}
    actions_by_phase: Dict[str, List[str]] = {pid: [] for pid in phase_nodes}
    phase_order: List[str] = list(phase_nodes.keys())

    for edge in query_graph.edges:
        if edge.edge_type != "hasAction":
            continue
        if edge.src in actions_by_phase:
            actions_by_phase[edge.src].append(edge.dst)

    # Compute order within each phase (order_index then stable by id).
    def _action_sort_key(action_id: str) -> Tuple[int, str]:
        node = query_graph.nodes.get(action_id)
        if not node:
            return (0, action_id)
        raw = node.attrs.get("order_index")
        try:
            return (int(raw), action_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (0, action_id)

    for pid in actions_by_phase:
        actions_by_phase[pid].sort(key=_action_sort_key)

    # Phase ordering by phase_index if present.
    def _phase_sort_key(phase_id: str) -> Tuple[int, str]:
        node = phase_nodes.get(phase_id)
        raw = node.attrs.get("phase_index") if node else None
        try:
            return (int(raw), phase_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (0, phase_id)

    phase_order.sort(key=_phase_sort_key)

    # Build memory action indexes (phase->actions and action->phase attrs) per memory plan.
    memory_action_phase: Dict[str, Dict[str, List[str]]] = {}
    for plan_id, graph in memory_graphs.items():
        mapping: Dict[str, List[str]] = {}
        for edge in graph.edges:
            if edge.edge_type != "hasAction":
                continue
            mapping.setdefault(edge.dst, []).append(edge.src)
        memory_action_phase[plan_id] = mapping

    # Precompute query action leaves once.
    query_context = _graph_context_leaves(query_graph, spec=spec, placeholder_re=placeholder_re)
    query_action_leaves: Dict[str, Dict[str, object]] = {}
    for action_id, node in query_graph.nodes.items():
        if node.node_type != "Action":
            continue
        query_action_leaves[action_id] = _node_leaves_with_context(
            node,
            graph_context=query_context,
            spec=spec,
            placeholder_re=placeholder_re,
        )

    phases_out: List[Dict[str, object]] = []
    for phase_id in phase_order:
        phase_node = phase_nodes[phase_id]
        phase_or_slots: List[Dict[str, object]] = []

        for action_id in actions_by_phase.get(phase_id, []):
            action_node = query_graph.nodes.get(action_id)
            if action_node is None:
                continue
            edge_key = _edge_key_to_string((phase_id, "hasAction", action_id))
            view_ids = lambda_map.get(edge_key) or []
            if not isinstance(view_ids, list):
                view_ids = []

            candidates: List[Dict[str, object]] = []
            seen: Set[str] = set()

            # Memory candidates from selected memory views.
            for view_id in view_ids:
                if not isinstance(view_id, str) or not view_id.startswith("memory::"):
                    continue
                memory_plan_id = view_id.split("memory::", 1)[1]
                mem_graph = memory_graphs.get(memory_plan_id)
                if mem_graph is None:
                    continue
                mem_context = _graph_context_leaves(mem_graph, spec=spec, placeholder_re=placeholder_re)
                q_leaves = query_action_leaves.get(action_id, {})
                for mem_node_id, mem_node in mem_graph.nodes.items():
                    if mem_node.node_type != "Action":
                        continue
                    mem_leaves = _node_leaves_with_context(
                        mem_node,
                        graph_context=mem_context,
                        spec=spec,
                        placeholder_re=placeholder_re,
                    )
                    if not _leaf_maps_compatible(mem_leaves, q_leaves):
                        continue
                    candidate = {
                        "source": "memory",
                        "view_id": view_id,
                        "memory_plan_id": memory_plan_id,
                        "memory_node_id": mem_node_id,
                        "text": _render_action_text(mem_node.attrs),
                        "cost": 0.0,
                        "tool_call_cost": 0.0,
                        "meta": {
                            "type": "memory",
                            "phase_ids": memory_action_phase.get(memory_plan_id, {}).get(mem_node_id, []),
                            "attrs": _to_builtin(mem_node.attrs),
                        },
                    }
                    key = _candidate_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)

            # Tool candidates from selected tool views.
            for view_id in view_ids:
                if not isinstance(view_id, str) or not view_id.startswith("tool::"):
                    continue
                tool_view = tool_views.get(view_id)
                if tool_view is None:
                    continue
                tool_candidates = _tool_candidates(
                    tool_view=tool_view,
                    query_plan=plan_node,
                    query_phase=phase_node,
                    query_action=action_node,
                    fetch=fetch,
                    handler_index=handler_index,
                    placeholder_engine=placeholder_engine,
                    mode_map=mode_map,
                    max_candidates=max_tool_candidates,
                )
                for cand in tool_candidates:
                    key = _candidate_key(cand)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(cand)

            # Fallback: if selected views did not yield any concrete candidates, call tools by action type.
            force_fallback = _force_tool_fallback_for_action(action_node.attrs, spec=spec)
            if not candidates or force_fallback:
                action_type = str(action_node.attrs.get("action_type") or "")
                for tool_view in _fallback_tool_views_for_action(
                    action_type=action_type,
                    tool_views=tool_views,
                    fallback_tools=fallback_tools,
                    tool_priority=fallback_tool_priority,
                ):
                    fallback_candidates = _tool_candidates(
                        tool_view=tool_view,
                        query_plan=plan_node,
                        query_phase=phase_node,
                        query_action=action_node,
                        fetch=fetch,
                        handler_index=handler_index,
                        placeholder_engine=placeholder_engine,
                        mode_map=mode_map,
                        max_candidates=max_tool_candidates,
                    )
                    _mark_candidates_fallback(fallback_candidates)
                    for cand in fallback_candidates:
                        key = _candidate_key(cand)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(cand)

            if not candidates:
                candidates.append(
                    {
                        "source": "placeholder",
                        "text": "-",
                        "cost": 0.0,
                        "meta": {"type": "placeholder"},
                    }
                )

            phase_or_slots.append(
                {
                    "kind": "OR",
                    "slot_edge": edge_key,
                    "query_action_node_id": action_id,
                    "query_action": _to_builtin(action_node.attrs),
                    "candidates": candidates,
                }
            )

        phases_out.append(
            {
                "kind": "AND",
                "query_phase_node_id": phase_id,
                "query_phase": _to_builtin(phase_node.attrs),
                "children": phase_or_slots,
            }
        )

    return {
        "version": 1,
        "task_name": task,
        "template_id": str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE"),
        "root": {
            "kind": "AND",
            "query_plan_node_id": plan_node.id,
            "query_plan": _to_builtin(plan_node.attrs),
            "children": phases_out,
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose candidates and emit AND-OR trees.")
    parser.add_argument("--task", type=str, default="travel", help="Task name (defaults to travel).")
    parser.add_argument("--config", type=Path, default=None, help="Override per-task planner config JSON (planner.json).")
    parser.add_argument("--split", type=str, required=True, help="Dataset split label (used for output naming).")
    parser.add_argument("--templates", type=Path, required=True, help="Init-template JSONL path.")
    parser.add_argument("--match", type=Path, required=True, help="View selection output JSON path (match_{split}.json).")
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="Memory graph JSONL path (defaults to artifacts/output/<task>/memory_graph.jsonl).",
    )
    parser.add_argument(
        "--tool-views",
        type=Path,
        default=None,
        help="Tool view config path (defaults to artifacts/input/<task>/views/tool.json).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (defaults to artifacts/output/<task>/tree_{split}.json).")
    parser.add_argument(
        "--max-tool-candidates",
        type=int,
        default=0,
        help="Max candidates per tool call (0 = unlimited).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes (0 = all CPUs; 1 = disable parallelism).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output (printed to stderr by default).",
    )
    parser.add_argument("--no-tools", action="store_true", help="Do not execute tools (emit memory-only candidates).")
    parser.add_argument("--no-tool-cache", action="store_true", help="Disable caching tool results within a process.")
    return parser.parse_args(argv)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    if minutes < 60:
        return f"{minutes:d}m{rem:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:d}h{minutes:02d}m"


def _emit_progress(*, done: int, total: int, started_at: float, label: str = "") -> None:
    elapsed = time.perf_counter() - started_at
    rate = (done / elapsed) if elapsed > 1e-9 else 0.0
    remaining = total - done
    eta = (remaining / rate) if rate > 1e-9 else float("inf")
    prefix = f"[{done}/{total}]"
    suffix = f" elapsed={_format_duration(elapsed)}"
    if math.isfinite(eta):
        suffix += f" eta={_format_duration(eta)}"
    label = label.strip()
    if label:
        sys.stderr.write(f"{prefix} {label}{suffix}\n")
    else:
        sys.stderr.write(f"{prefix}{suffix}\n")
    sys.stderr.flush()


def main(argv: Optional[Sequence[str]] = None) -> None:
    started_at = time.perf_counter()
    argv_snapshot = sys.argv[:]
    status = "ok"
    error: Optional[str] = None
    extra: Dict[str, object] = {}
    results_count = 0
    tmp_path: Optional[Path] = None
    global _CURRENT_TOOL_CALL_STATS
    global _TOOL_RESULT_CACHE

    try:
        args = parse_args(argv)
        show_progress = not bool(args.no_progress)
        extra.update(
            {
                "task": args.task,
                "split": args.split,
                "max_tool_candidates": int(args.max_tool_candidates),
                "no_tools": bool(args.no_tools),
                "no_tool_cache": bool(args.no_tool_cache),
                "workers": int(args.workers),
                "progress": bool(show_progress),
            }
        )

        task = str(args.task)
        _TOOL_RESULT_CACHE = None if bool(args.no_tool_cache) else {}

        templates_path = args.templates
        if not templates_path.is_absolute():
            templates_path = (PROJECT_ROOT / templates_path).resolve()
        if not templates_path.exists():
            raise SystemExit(f"Templates JSONL not found: {templates_path}")
        extra.update({"templates": str(templates_path)})

        match_path = args.match
        if not match_path.is_absolute():
            match_path = (PROJECT_ROOT / match_path).resolve()
        if not match_path.exists():
            raise SystemExit(f"Match JSON not found: {match_path}")
        extra.update({"match": str(match_path)})

        input_dir = (INPUT_ROOT / task).resolve()
        output_dir = (OUTPUT_ROOT / task).resolve()

        memory_path = args.memory or (output_dir / "memory_graph.jsonl")
        if not memory_path.is_absolute():
            memory_path = (PROJECT_ROOT / memory_path).resolve()
        if not memory_path.exists():
            raise SystemExit(f"Memory graphs JSONL not found: {memory_path}")
        extra.update({"memory": str(memory_path)})

        tool_views_path = args.tool_views or (input_dir / "views" / "tool.json")
        if not tool_views_path.is_absolute():
            tool_views_path = (PROJECT_ROOT / tool_views_path).resolve()
        if not tool_views_path.exists():
            raise SystemExit(f"Tool views not found: {tool_views_path}")
        extra.update({"tool_views": str(tool_views_path)})

        out_path = args.out or (output_dir / f"tree_{args.split}.json")
        if not out_path.is_absolute():
            if out_path.parent == Path("."):
                out_path = (output_dir / out_path).resolve()
            else:
                out_path = (PROJECT_ROOT / out_path).resolve()
        extra.update({"out": str(out_path)})

        match_payload = json.loads(match_path.read_text(encoding="utf-8"))
        if not isinstance(match_payload, dict):
            raise SystemExit(f"Match JSON must be an object mapping template_id -> selection: {match_path}")

        templates = list(_iter_jsonl(templates_path))
        total_templates = len(templates)
        extra.update({"templates_total": total_templates})

        workers = int(args.workers)
        if workers <= 0:
            workers = available_cpu_count()
        workers = max(1, min(workers, total_templates or 1))
        extra.update({"workers_effective": workers})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / f".{out_path.name}.tmp"

        def _match_entry_for(template_id: str) -> Dict[str, object]:
            entry = match_payload.get(template_id) or {}
            return entry if isinstance(entry, dict) else {}

        done = 0
        tool_call_stats_total: ToolCallStats = {}
        with tmp_path.open("w", encoding="utf-8") as fp:
            fp.write("{\n")
            first = True

            # Stream JSON object to disk to avoid holding all trees in memory.
            if workers <= 1 or total_templates <= 1:
                try:
                    planner_cfg = load_planner_config(task, config_path=args.config)
                except ConfigError as exc:
                    raise SystemExit(str(exc))
                placeholder_engine = _load_placeholder_engine(planner_cfg.compose)
                mode_map = dict(planner_cfg.compose.mode_map)
                fallback_tools = dict(planner_cfg.compose.fallback_tools_by_action_type)
                fallback_tool_priority = dict(planner_cfg.compose.fallback_tool_priority)

                memory_graphs = _load_memory_graphs(memory_path)
                tool_views = _load_tool_views(tool_views_path)
                spec = _maybe_derive_force_fallback_spec(
                    _spec_from_planner_cfg(planner_cfg),
                    tool_views=tool_views,
                    placeholder_engine=placeholder_engine,
                )
                handler_index = _build_tool_handler_index(planner_cfg.compose.tool_handlers, tool_views=tool_views)
                fetch = None
                if not args.no_tools and planner_cfg.compose.fetch_factory:
                    fetch_factory = load_hook(planner_cfg.compose.fetch_factory)
                    fetch = fetch_factory()

                for template in templates:
                    template_id = str(
                        template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE"
                    )
                    per_template_stats: ToolCallStats = {}
                    prev_stats = _CURRENT_TOOL_CALL_STATS
                    _CURRENT_TOOL_CALL_STATS = per_template_stats
                    try:
                        tree = _build_and_or_tree_for_template(
                            template=template,
                            match_entry=_match_entry_for(template_id),
                            memory_graphs=memory_graphs,
                            tool_views=tool_views,
                            task=task,
                            max_tool_candidates=int(args.max_tool_candidates),
                            fetch=fetch,
                            spec=spec,
                            handler_index=handler_index,
                            placeholder_engine=placeholder_engine,
                            mode_map=mode_map,
                            fallback_tools=fallback_tools,
                            fallback_tool_priority=fallback_tool_priority,
                        )
                    finally:
                        _CURRENT_TOOL_CALL_STATS = prev_stats
                    _merge_tool_call_stats(tool_call_stats_total, per_template_stats)
                    if not first:
                        fp.write(",\n")
                    fp.write(json.dumps(template_id, ensure_ascii=False) + ": " + json.dumps(tree, ensure_ascii=False))
                    first = False
                    done += 1
                    if show_progress:
                        _emit_progress(done=done, total=total_templates, started_at=started_at, label=f"template_id={template_id}")
            else:
                initargs = (
                    task,
                    str(args.config) if args.config is not None else None,
                    str(memory_path),
                    str(tool_views_path),
                    bool(args.no_tools),
                    int(args.max_tool_candidates),
                    bool(args.no_tool_cache),
                )
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_compose_match_worker_init,
                    initargs=initargs,
                ) as executor:
                    futures: Dict[concurrent.futures.Future[Tuple[str, str, ToolCallStats]], str] = {}
                    for template in templates:
                        template_id = str(
                            template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE"
                        )
                        future = executor.submit(
                            _compose_match_worker_build_tree_json,
                            template,
                            _match_entry_for(template_id),
                        )
                        futures[future] = template_id

                    for future in concurrent.futures.as_completed(futures):
                        template_id = futures[future]
                        built_id, tree_json, per_template_stats = future.result()
                        _merge_tool_call_stats(tool_call_stats_total, per_template_stats)
                        if not first:
                            fp.write(",\n")
                        fp.write(json.dumps(built_id, ensure_ascii=False) + ": " + tree_json)
                        first = False
                        done += 1
                        if show_progress:
                            _emit_progress(done=done, total=total_templates, started_at=started_at, label=f"template_id={template_id}")

            fp.write("\n}\n")

        os.replace(tmp_path, out_path)
        results_count = int(done)
        print(f"Wrote AND-OR trees for {results_count} template(s) to {out_path}")

        tool_calls_summary = _summarize_tool_calls(tool_call_stats_total)
        tool_calls_path = out_path.parent / f"tool_calls_{args.split}.json"
        tool_calls_payload: Dict[str, object] = {
            "version": 1,
            "task": str(args.task),
            "split": str(args.split),
            "trees": str(out_path),
            "tool_views": str(tool_views_path),
        }
        tool_calls_payload.update(tool_calls_summary)
        tool_calls_path.write_text(json.dumps(tool_calls_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote tool call summary to {tool_calls_path}")
        extra.update(
            {
                "tool_calls": str(tool_calls_path),
                "tool_calls_total": int(tool_calls_summary.get("total_calls") or 0),
                "tool_calls_cost": float(tool_calls_summary.get("total_cost") or 0.0),
                "tool_calls_by_tool": tool_calls_summary.get("by_tool") if isinstance(tool_calls_summary.get("by_tool"), dict) else {},
            }
        )
    except BaseException as exc:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
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
        extra.setdefault("trees_written", results_count)
        _append_run_log(argv=argv_snapshot, started_at=started_at, status=status, extra=extra, error=error)


if __name__ == "__main__":
    main()

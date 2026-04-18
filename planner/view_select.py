"""
View selection (ViewCover) for init-templates.

This module implements Step 1 (ViewCover):
  1) Simulation-based coverage (node-compatibility relation + edge existence; no tool execution)
  2) Greedy weighted set cover over required template edges

Inputs
  - Query templates produced by `planner/init_template.py` (JSONL; one template per line)
  - View patterns under `artifacts/input/<task>/views/`:
      * `memory.jsonl` : memory views (PlanGraph JSONL)
      * `tool.json`    : tool micro-views (node patterns + costs)

Output
  - `artifacts/output/<task>/match_<split>.json` mapping template_id -> selection result.

Task-specific matching behaviour is controlled via `artifacts/input/<task>/planner.json`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from planner.config import ConfigError, load_planner_config
from planner.parallel import available_cpu_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"
OUTPUT_ROOT = ARTIFACTS_ROOT / "output"


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

    def edge_keys(self) -> Set[EdgeKey]:
        return {edge.key() for edge in self.edges}


@dataclass(frozen=True)
class View:
    view_id: str
    source: str  # "memory" | "tool"
    cost: float
    pattern: TypedGraph
    tool_name: Optional[str] = None


@dataclass(frozen=True)
class MatchSpec:
    """
    Matching behaviour knobs.

    We treat placeholders (e.g. "{City}") as null / unconstrained.
    We only compare values that are non-null in BOTH pattern and query ("intersection-of-non-null").

    Context propagation:
      - Some template information lives on the Plan node, while required edges typically connect Phase->Action.
      - `context_attr_paths` can propagate selected Plan leaf attributes into other node types under a
        namespaced prefix (e.g. `plan.dest`) to make matching destination-aware without requiring `hasPhase`.
    """

    required_edge_types: Tuple[str, ...] = ("hasAction",)
    ignored_edge_types: Tuple[str, ...] = ("before",)
    ignored_attr_paths: Tuple[str, ...] = ()
    nullify_substrings_by_path: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    placeholder_regex: str = r"^\{[^{}]+\}$"
    context_from_node_type: str = "Plan"
    context_attr_paths: Tuple[str, ...] = ()
    context_namespace: str = "plan"
    context_apply_to_node_types: Tuple[str, ...] = ()
    force_include_tool_action_types: Tuple[str, ...] = ()


def _spec_from_config(task: str, *, config_path: Optional[Path] = None) -> MatchSpec:
    cfg = load_planner_config(task, config_path=config_path)
    return MatchSpec(
        required_edge_types=tuple(cfg.view_select.required_edge_types),
        ignored_edge_types=tuple(cfg.view_select.ignored_edge_types),
        ignored_attr_paths=tuple(cfg.match.ignored_attr_paths),
        nullify_substrings_by_path=cfg.match.nullify_substrings_by_path,
        placeholder_regex=str(cfg.match.placeholder_regex),
        context_from_node_type=str(cfg.match.context.from_node_type),
        context_attr_paths=tuple(cfg.match.context.attr_paths),
        context_namespace=str(cfg.match.context.namespace),
        context_apply_to_node_types=tuple(cfg.match.context.apply_to_node_types),
        force_include_tool_action_types=tuple(cfg.view_select.force_include_tool_action_types),
    )


def _derive_multi_variant_action_types(views: Sequence["View"]) -> Tuple[str, ...]:
    action_type_to_view_ids: Dict[str, Set[str]] = {}
    for view in views:
        if view.source != "tool":
            continue
        for action_type in _tool_view_action_types(view):
            action_type_to_view_ids.setdefault(action_type, set()).add(view.view_id)
    return tuple(sorted(action_type for action_type, ids in action_type_to_view_ids.items() if len(ids) > 1))


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


def _load_tool_views(path: Path) -> List[View]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views_payload = payload.get("views") or []
    if not isinstance(views_payload, list):
        raise TypeError(f"tool.json 'views' must be a list: {path}")

    views: List[View] = []
    for item in views_payload:
        if not isinstance(item, dict):
            continue
        view_id = str(item.get("view_id") or "")
        if not view_id:
            continue
        tool = item.get("tool")
        tool_name = str(tool) if tool is not None else None
        cost = float(item.get("cost") or 0.0)
        node_pattern = item.get("node_pattern") or {}
        if not isinstance(node_pattern, dict):
            continue
        node_type = str(node_pattern.get("type") or "")
        if not node_type:
            continue
        attrs = node_pattern.get("attrs") or {}
        if not isinstance(attrs, dict):
            attrs = {}

        # Tool micro-view pattern: Phase -> Action
        phase_id = "phase"
        action_id = "action"
        graph = TypedGraph(
            nodes={
                phase_id: Node(id=phase_id, node_type="Phase", attrs={}),
                action_id: Node(id=action_id, node_type=node_type, attrs=dict(attrs)),
            },
            edges=[Edge(src=phase_id, dst=action_id, edge_type="hasAction", attrs={})],
        )
        views.append(View(view_id=view_id, source="tool", cost=cost, pattern=graph, tool_name=tool_name))
    return views


def _load_memory_views(path: Path) -> List[View]:
    views: List[View] = []
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

        if not nodes or not edges:
            continue
        view_id = f"memory::{plan_id}"
        views.append(View(view_id=view_id, source="memory", cost=0.0, pattern=TypedGraph(nodes=nodes, edges=edges)))
    return views


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

            nodes[action_id] = Node(id=action_id, node_type="Action", attrs=node_attrs)

            phase_key = str(action.get("phase_key") or "")
            phase_node_id = phase_key_to_node.get(phase_key)
            if not phase_node_id:
                # Graceful fallback: attach to the first phase if present.
                phase_node_id = next(iter(phase_key_to_node.values()), None)
            if phase_node_id:
                edges.append(Edge(src=phase_node_id, dst=action_id, edge_type="hasAction", attrs={}))

    return TypedGraph(nodes=nodes, edges=edges)


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
    # Only compare values that are non-null in BOTH pattern and query.
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
            if path in GLOBAL_IGNORED_ATTR_PATHS or path in spec.ignored_attr_paths:
                continue
            leaves[path] = value
    return leaves


def _match_by_simulation(
    *,
    pattern: TypedGraph,
    query: TypedGraph,
    spec: MatchSpec,
    placeholder_re: re.Pattern[str],
) -> Set[EdgeKey]:
    """
    Coverage rule (simplified):
      1) Node compatibility defines "common nodes" (same node_type and matching non-null attrs).
      2) A query edge is covered if the view contains an edge of the same type between compatible endpoints.
    """

    pattern_context = _graph_context_leaves(pattern, spec=spec, placeholder_re=placeholder_re)
    query_context = _graph_context_leaves(query, spec=spec, placeholder_re=placeholder_re)

    p_types = {node_id: node.node_type for node_id, node in pattern.nodes.items()}
    q_types = {node_id: node.node_type for node_id, node in query.nodes.items()}

    p_leaves = {
        node_id: _node_leaves_with_context(node, graph_context=pattern_context, spec=spec, placeholder_re=placeholder_re)
        for node_id, node in pattern.nodes.items()
    }
    q_leaves = {
        node_id: _node_leaves_with_context(node, graph_context=query_context, spec=spec, placeholder_re=placeholder_re)
        for node_id, node in query.nodes.items()
    }

    compat: Dict[str, Set[str]] = {q_id: set() for q_id in q_types}
    for q_id, q_type in q_types.items():
        q_leaf_map = q_leaves[q_id]
        for p_id, p_type in p_types.items():
            if p_type != q_type:
                continue
            if _leaf_maps_compatible(p_leaves[p_id], q_leaf_map):
                compat[q_id].add(p_id)

    edges_by_type: Dict[str, List[Tuple[str, str]]] = {}
    for edge in pattern.edges:
        if edge.edge_type in spec.ignored_edge_types:
            continue
        edges_by_type.setdefault(edge.edge_type, []).append((edge.src, edge.dst))

    covered: Set[EdgeKey] = set()
    for edge in query.edges:
        if edge.edge_type in spec.ignored_edge_types:
            continue
        if edge.edge_type not in spec.required_edge_types:
            continue
        p_srcs = compat.get(edge.src, set())
        p_dsts = compat.get(edge.dst, set())
        if not p_srcs or not p_dsts:
            continue
        for p_src, p_dst in edges_by_type.get(edge.edge_type, []):
            if p_src in p_srcs and p_dst in p_dsts:
                covered.add(edge.key())
                break
    return covered


def _action_types_in_graph(graph: TypedGraph) -> Set[str]:
    types: Set[str] = set()
    for node in graph.nodes.values():
        if node.node_type != "Action":
            continue
        action_type = node.attrs.get("action_type")
        if isinstance(action_type, str) and action_type.strip():
            types.add(action_type.strip())
    return types


def _tool_view_action_types(view: View) -> Set[str]:
    if view.source != "tool":
        return set()
    return _action_types_in_graph(view.pattern)


def _edge_key_to_string(edge: EdgeKey) -> str:
    src, etype, dst = edge
    return f"{src} -[{etype}]-> {dst}"


def _greedy_weighted_set_cover(
    *,
    universe: Set[EdgeKey],
    cover_sets: Mapping[str, Set[EdgeKey]],
    weights: Mapping[str, float],
) -> Tuple[List[str], Set[EdgeKey]]:
    covered: Set[EdgeKey] = set()
    selected: List[str] = []

    # Deterministic iteration order across runs.
    view_ids = sorted(cover_sets.keys())

    while covered != universe:
        best_view: Optional[str] = None
        best_ratio = -1.0
        best_weight = math.inf
        best_gain = -1

        for view_id in view_ids:
            cov = cover_sets.get(view_id, set())
            gain = len(cov - covered)
            if gain == 0:
                continue
            weight = float(weights.get(view_id, 0.0))
            if weight == 0.0:
                ratio = math.inf
            else:
                ratio = gain / weight

            if ratio > best_ratio:
                best_ratio = ratio
                best_weight = weight
                best_gain = gain
                best_view = view_id
                continue

            if ratio == best_ratio:
                if math.isinf(ratio):
                    # For zero-cost views, prefer larger immediate coverage to reduce view count.
                    if gain > best_gain:
                        best_gain = gain
                        best_view = view_id
                    elif gain == best_gain and best_view is not None and view_id < best_view:
                        best_view = view_id
                else:
                    # Tie-break: lower omega, then lexicographically smaller view_id.
                    if weight < best_weight:
                        best_weight = weight
                        best_view = view_id
                    elif weight == best_weight and best_view is not None and view_id < best_view:
                        best_view = view_id

        if best_view is None:
            break
        selected.append(best_view)
        covered |= cover_sets.get(best_view, set())

    return selected, universe - covered


def _select_views_for_template(
    *,
    query_graph: TypedGraph,
    views: Sequence[View],
    spec: MatchSpec,
) -> Dict[str, object]:
    placeholder_re = re.compile(spec.placeholder_regex)

    required = {edge.key() for edge in query_graph.edges if edge.edge_type in spec.required_edge_types}

    cover_sets: Dict[str, Set[EdgeKey]] = {}
    weights: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    for view in views:
        weights[view.view_id] = float(view.cost)
        sources[view.view_id] = view.source
        covered = _match_by_simulation(
            pattern=view.pattern,
            query=query_graph,
            spec=spec,
            placeholder_re=placeholder_re,
        )
        covered &= required
        if not covered:
            continue
        cover_sets[view.view_id] = covered

    selected_view_ids, uncovered = _greedy_weighted_set_cover(universe=required, cover_sets=cover_sets, weights=weights)

    # Task-specific: for some action types (e.g., travel Move) we want ALL related
    # tool variants available once any one of them is needed/selected.
    if spec.force_include_tool_action_types:
        selected_action_types: Set[str] = set()
        views_by_id = {view.view_id: view for view in views}
        for view_id in selected_view_ids:
            view = views_by_id.get(view_id)
            if view is None:
                continue
            selected_action_types |= _tool_view_action_types(view)

        triggered = selected_action_types & set(spec.force_include_tool_action_types)
        if triggered:
            forced: List[str] = []
            for view in views:
                if view.source != "tool":
                    continue
                if _tool_view_action_types(view) & triggered:
                    forced.append(view.view_id)
            for view_id in sorted(set(forced)):
                if view_id not in selected_view_ids:
                    selected_view_ids.append(view_id)

            # Forced views can only reduce uncovered edges; recompute to reflect the final selection.
            covered_after: Set[EdgeKey] = set()
            for view_id in selected_view_ids:
                covered_after |= cover_sets.get(view_id, set())
            uncovered = required - covered_after

    lambda_map: Dict[str, List[str]] = {}
    for edge in sorted(required):
        edge_key = _edge_key_to_string(edge)
        lambda_map[edge_key] = [vid for vid in selected_view_ids if edge in cover_sets.get(vid, set())]

    selected_cost = sum(weights.get(vid, 0.0) for vid in selected_view_ids)
    return {
        "required_edges": len(required),
        "uncovered_edges": [_edge_key_to_string(edge) for edge in sorted(uncovered)],
        "selected_views": selected_view_ids,
        "selected_views_by_source": {
            "memory": [vid for vid in selected_view_ids if sources.get(vid) == "memory"],
            "tool": [vid for vid in selected_view_ids if sources.get(vid) == "tool"],
        },
        "total_cost": selected_cost,
        "lambda": lambda_map,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select minimal views that cover init-templates (ViewCover).")
    parser.add_argument("--task", type=str, default="travel", help="Task name (defaults to travel).")
    parser.add_argument("--config", type=Path, default=None, help="Override per-task planner config JSON (planner.json).")
    parser.add_argument("--split", type=str, required=True, help="Dataset split label (used for output naming).")
    parser.add_argument("--templates", type=Path, required=True, help="Input init-template JSONL path.")
    parser.add_argument("--views-dir", type=Path, help="Override views directory (defaults to artifacts/input/<task>/views).")
    parser.add_argument(
        "--required-edge-types",
        type=str,
        default=None,
        help="Comma-separated required edge types (default comes from task spec).",
    )
    parser.add_argument("--out", type=Path, help="Output JSON path (defaults to artifacts/output/<task>/match_<split>.json).")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (0 = all CPUs; 1 = disable parallelism).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any template has uncovered required edges.")
    return parser.parse_args(argv)


_WORKER_VIEWS: Optional[List[View]] = None
_WORKER_SPEC: Optional[MatchSpec] = None


def _view_select_worker_init(views: Sequence[View], spec: MatchSpec) -> None:
    global _WORKER_VIEWS
    global _WORKER_SPEC
    _WORKER_VIEWS = list(views)
    _WORKER_SPEC = spec


def _view_select_worker(template: Mapping[str, object]) -> Tuple[str, Dict[str, object]]:
    views = _WORKER_VIEWS
    spec = _WORKER_SPEC
    if views is None or spec is None:
        raise RuntimeError("ViewSelect worker not initialised (missing views/spec).")
    template_id = str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE")
    query_graph = _build_query_graph_from_init_template(template)
    selection = _select_views_for_template(query_graph=query_graph, views=views, spec=spec)
    return template_id, selection


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    task = str(args.task)
    try:
        spec = _spec_from_config(task, config_path=args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc))
    if args.required_edge_types is not None:
        edge_types = tuple(item.strip() for item in str(args.required_edge_types).split(",") if item.strip())
        spec = replace(spec, required_edge_types=edge_types or spec.required_edge_types)

    input_dir = (INPUT_ROOT / task).resolve()
    output_dir = (OUTPUT_ROOT / task).resolve()

    templates_path = args.templates
    if not templates_path.is_absolute():
        templates_path = (PROJECT_ROOT / templates_path).resolve()
    if not templates_path.exists():
        raise SystemExit(f"Templates JSONL not found: {templates_path}")

    views_dir = args.views_dir or (input_dir / "views")
    if not views_dir.is_absolute():
        views_dir = (PROJECT_ROOT / views_dir).resolve()
    memory_views_path = views_dir / "memory.jsonl"
    tool_views_path = views_dir / "tool.json"
    if not memory_views_path.exists():
        raise SystemExit(f"Missing memory views: {memory_views_path}")
    if not tool_views_path.exists():
        raise SystemExit(f"Missing tool views: {tool_views_path}")

    views: List[View] = []
    views.extend(_load_memory_views(memory_views_path))
    views.extend(_load_tool_views(tool_views_path))

    # If the task config does not specify which action types should include all tool variants,
    # derive it from tool.json by selecting action types with multiple tool micro-views.
    if spec.force_include_tool_action_types == ():
        try:
            cfg = load_planner_config(task, config_path=args.config)
        except ConfigError:
            cfg = None
        if cfg is not None and cfg.view_select.force_include_tool_variants_enabled:
            derived = _derive_multi_variant_action_types(views)
            if derived:
                spec = replace(spec, force_include_tool_action_types=derived)

    out_path = args.out or (output_dir / f"match_{args.split}.json")
    if not out_path.is_absolute():
        if out_path.parent == Path("."):
            out_path = (output_dir / out_path).resolve()
        else:
            out_path = (PROJECT_ROOT / out_path).resolve()

    results: Dict[str, object] = {}
    failures: List[str] = []

    templates = list(_iter_jsonl(templates_path))
    requested_workers = int(args.workers)
    if requested_workers <= 0:
        worker_count = available_cpu_count()
    else:
        worker_count = requested_workers
    worker_count = max(1, min(worker_count, len(templates)))

    if worker_count <= 1:
        for template in templates:
            template_id = str(template.get("template_id") or template.get("plan_id") or template.get("id") or "UNKNOWN_TEMPLATE")
            query_graph = _build_query_graph_from_init_template(template)
            selection = _select_views_for_template(query_graph=query_graph, views=views, spec=spec)
            results[template_id] = selection
            if args.strict and selection.get("uncovered_edges"):
                failures.append(template_id)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_view_select_worker_init,
            initargs=(views, spec),
        ) as executor:
            for template_id, selection in executor.map(_view_select_worker, templates):
                results[template_id] = selection
                if args.strict and selection.get("uncovered_edges"):
                    failures.append(template_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        raise SystemExit(f"Uncovered required edges for {len(failures)} template(s): {', '.join(failures[:10])}")

    print(f"Wrote view-cover results for {len(results)} template(s) to {out_path}")


if __name__ == "__main__":
    main()

"""
Generate TravelPlanner memory views for pattern matching, using the same JSONL
schema as `memory_graph.jsonl`.

This script writes `artifacts/input/travel/views/memory.jsonl`.
Each line is a plan graph JSON object:

  { "plan_id": ..., "nodes": [...], "edges": [...] }

Design goal
  - Keep a uniform graph representation for both memories and planning.
  - Strip/mute instance-specific details, keeping only:
      * Plan destination (`dest`)
      * Move destination on the first day (hide origin) and Move origin on the last day (hide destination)
      * DayRecord `current_city` for middle days (not first/last)
  - Preserve node types and backbone edges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

for parent in Path(__file__).resolve().parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from memory_graph import store
from memory_graph.schema import MNode, PlanGraph
from task_helper.travel.utils.paths import project_root


def _prune_nulls(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        out: Dict[str, object] = {}
        for key, item in value.items():
            pruned = _prune_nulls(item)
            if pruned is None:
                continue
            out[str(key)] = pruned
        return out
    if isinstance(value, list):
        items = [_prune_nulls(item) for item in value]
        return [item for item in items if item is not None]
    return value


def _extract_dest(graph: PlanGraph) -> Optional[str]:
    for node in graph.nodes:
        if node.type == "Plan":
            dest = node.attrs.get("dest")
            if dest is None:
                return None
            return str(dest)
    return None


def _skeletonise_action_params(
    *,
    action_type: str,
    params: Mapping[str, object],
    phase_index: Optional[int],
    last_phase_index: Optional[int],
) -> Dict[str, object]:
    if action_type == "DayRecord":
        day = params.get("day")
        if day is None and phase_index is not None:
            day = phase_index
        skeleton: Dict[str, object] = {"day": day}
        if phase_index is not None and last_phase_index is not None and 1 < phase_index < last_phase_index:
            skeleton["current_city"] = params.get("current_city")
        return dict(_prune_nulls(skeleton) or {})

    if action_type == "Move":
        if phase_index is None or last_phase_index is None:
            return {}

        if phase_index == 1:
            if "DestCityName" in params:
                return dict(_prune_nulls({"DestCityName": params.get("DestCityName")}) or {})
            if "destination" in params:
                return dict(_prune_nulls({"destination": params.get("destination")}) or {})
            if "to" in params:
                return dict(_prune_nulls({"to": params.get("to")}) or {})
            return {}

        if phase_index == last_phase_index:
            if "OriginCityName" in params:
                return dict(_prune_nulls({"OriginCityName": params.get("OriginCityName")}) or {})
            if "origin" in params:
                return dict(_prune_nulls({"origin": params.get("origin")}) or {})
            if "from" in params:
                return dict(_prune_nulls({"from": params.get("from")}) or {})
            return {}

        return {}

    return {}


def skeletonise_plan_graph(graph: PlanGraph) -> PlanGraph:
    dest = _extract_dest(graph)

    phase_index_by_id: Dict[str, int] = {}
    for node in graph.nodes:
        if node.type != "Phase":
            continue
        raw = node.attrs.get("phase_index")
        try:
            phase_index_by_id[node.id] = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            phase_index_by_id[node.id] = 0
    last_phase_index = max(phase_index_by_id.values(), default=0)

    action_phase_index: Dict[str, int] = {}
    for edge in graph.edges:
        if edge.type != "hasAction":
            continue
        phase_id = edge.src
        action_id = edge.dst
        action_phase_index[action_id] = phase_index_by_id.get(phase_id, 0)

    out = PlanGraph(plan_id=graph.plan_id)

    for node in graph.nodes:
        if node.type == "Plan":
            attrs: Dict[str, object] = {}
            if dest is not None:
                attrs["dest"] = dest
            out.add_node(MNode(id=node.id, type=node.type, attrs=attrs))
            continue

        if node.type == "Phase":
            attrs: Dict[str, object] = {}
            for key in ("phase_key", "phase_type", "phase_index"):
                value = node.attrs.get(key)
                if value is not None:
                    attrs[key] = value
            out.add_node(MNode(id=node.id, type=node.type, attrs=attrs))
            continue

        if node.type == "Action":
            attrs: Dict[str, object] = {}
            action_type = str(node.attrs.get("action_type") or "")
            attrs["action_type"] = action_type

            raw_params = node.attrs.get("params")
            params: Dict[str, object] = dict(raw_params) if isinstance(raw_params, dict) else {}

            phase_index = action_phase_index.get(node.id)
            skeleton_params = _skeletonise_action_params(
                action_type=action_type,
                params=params,
                phase_index=phase_index,
                last_phase_index=last_phase_index,
            )
            attrs["params"] = skeleton_params

            out.add_node(MNode(id=node.id, type=node.type, attrs=attrs))
            continue

        out.add_node(MNode(id=node.id, type=node.type, attrs=dict(_prune_nulls(node.attrs or {}) or {})))

    for edge in graph.edges:
        out.add_edge(type(edge)(src=edge.src, dst=edge.dst, type=edge.type, attrs={}))

    return out


def iter_skeleton_views(*, input_jsonl: Path) -> Iterator[PlanGraph]:
    for graph in store.load_jsonl(input_jsonl):
        yield skeletonise_plan_graph(graph)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = project_root()
    default_in = root / "artifacts" / "output" / "travel" / "memory_graph.jsonl"
    default_out = root / "artifacts" / "input" / "travel" / "views" / "memory.jsonl"

    parser = argparse.ArgumentParser(
        description="Generate artifacts/input/travel/views/memory.jsonl from artifacts/output/travel/memory_graph.jsonl."
    )
    parser.add_argument("--in", dest="input_jsonl", type=Path, default=default_in, help="Input memory_graph.jsonl.")
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output path for memory.jsonl.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_path = args.out.expanduser()
    input_jsonl = args.input_jsonl.expanduser()

    views = list(iter_skeleton_views(input_jsonl=input_jsonl))
    store.save_jsonl(views, out_path)

    dests: List[str] = []
    seen = set()
    for view in views:
        dest = _extract_dest(view)
        if not dest:
            continue
        if dest in seen:
            continue
        seen.add(dest)
        dests.append(dest)

    city_path = out_path.parent / "city.txt"
    city_path.write_text("\n".join(dests) + ("\n" if dests else ""), encoding="utf-8")

    print(f"Wrote {len(views)} memory view(s) to {out_path}")
    print(f"Wrote {len(dests)} unique destination(s) to {city_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute the rate of template edges that are covered by tool views (tool-only) in a ViewCover match JSON.

For WorkBench, ViewSelect outputs `match_<split>.json` with:
  - entry["lambda"]: map edge_string -> [view_id, ...]
  - memory views: "memory::<plan_id>"
  - tool views:   "tool::<domain>::<action_type>"

We define:
  tool_only_edge := lambda[edge] contains at least one tool view id AND contains no memory view id.

This is a "needs-tool-view coverage" signal from ViewCover (pattern-only), not the executed tool-call count.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class EdgeCoverageStats:
    templates: int
    total_edges: int
    tool_only_edges: int
    edges_with_any_tool: int
    uncovered_edges: int

    @property
    def tool_only_rate(self) -> float:
        return (self.tool_only_edges / self.total_edges) if self.total_edges else 0.0

    @property
    def any_tool_rate(self) -> float:
        return (self.edges_with_any_tool / self.total_edges) if self.total_edges else 0.0

    @property
    def uncovered_rate(self) -> float:
        return (self.uncovered_edges / self.total_edges) if self.total_edges else 0.0


def _is_memory_view_id(view_id: object) -> bool:
    return isinstance(view_id, str) and view_id.startswith("memory::")


def _is_tool_view_id(view_id: object) -> bool:
    return isinstance(view_id, str) and not view_id.startswith("memory::")


def compute_stats(payload: Mapping[str, object]) -> EdgeCoverageStats:
    templates = 0
    total_edges = 0
    tool_only_edges = 0
    edges_with_any_tool = 0
    uncovered_edges = 0

    for _, entry in payload.items():
        if not isinstance(entry, Mapping):
            continue
        lam = entry.get("lambda")
        if not isinstance(lam, Mapping):
            continue
        templates += 1
        for _, view_ids in lam.items():
            if not isinstance(view_ids, list):
                continue
            total_edges += 1
            if len(view_ids) == 0:
                uncovered_edges += 1
                continue
            has_memory = any(_is_memory_view_id(v) for v in view_ids)
            has_tool = any(_is_tool_view_id(v) for v in view_ids)
            if has_tool:
                edges_with_any_tool += 1
            if has_tool and not has_memory:
                tool_only_edges += 1

    return EdgeCoverageStats(
        templates=int(templates),
        total_edges=int(total_edges),
        tool_only_edges=int(tool_only_edges),
        edges_with_any_tool=int(edges_with_any_tool),
        uncovered_edges=int(uncovered_edges),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tool-only edge coverage rate from a ViewCover match_<split>.json.",
    )
    parser.add_argument(
        "match_json",
        nargs="+",
        help="Path(s) to match_<split>.json (e.g., artifacts/output/work/<run>/match_validation.json).",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON per input file.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Match JSON must be an object mapping template_id -> selection: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    for raw in args.match_json:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        payload = _load_json(path)
        stats = compute_stats(payload)

        if args.json:
            out = {
                "path": str(path),
                "templates": stats.templates,
                "total_edges": stats.total_edges,
                "tool_only_edges": stats.tool_only_edges,
                "tool_only_rate": stats.tool_only_rate,
                "edges_with_any_tool": stats.edges_with_any_tool,
                "any_tool_rate": stats.any_tool_rate,
                "uncovered_edges": stats.uncovered_edges,
                "uncovered_rate": stats.uncovered_rate,
            }
            print(json.dumps(out, ensure_ascii=False))
            continue

        print(
            f"{path}: tool_only_rate={stats.tool_only_rate:.6f} "
            f"(tool_only_edges={stats.tool_only_edges} total_edges={stats.total_edges} "
            f"templates={stats.templates} uncovered_edges={stats.uncovered_edges})"
        )


if __name__ == "__main__":
    main()


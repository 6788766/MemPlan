#!/usr/bin/env python3
"""
Compute candidate-source stats from WorkBench AND–OR trees (tree_<split>.json).

We scan all nodes with a `candidates` list (produced by ComposeMatch) and count:
  - how many candidates come from source="tool" vs source="memory"
  - how many tool candidates are "shadowed" by an equivalent memory candidate

Shadowing rule (per-slot):
  If a slot has a memory candidate and a tool candidate with the same
  (qualified_tool_name, args) signature, the tool candidate is NOT counted as an
  "effective tool candidate" for that slot.

This is about candidate provenance in the tree, not executed tool calls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(k), _normalize_json(v)) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, list):
        return tuple(_normalize_json(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_normalize_json(v) for v in value)
    return value


def _tool_name_from_view_id(tool_view_id: Optional[str]) -> Optional[str]:
    """
    Examples:
      tool::analytics::create_plot            -> analytics.create_plot
      memory::work_000031::tool::analytics::create_plot -> analytics.create_plot
    """
    if not tool_view_id:
        return None
    parts = str(tool_view_id).split("::")
    if "tool" in parts:
        i = parts.index("tool")
        if i + 2 < len(parts):
            return f"{parts[i+1]}.{parts[i+2]}"
    if len(parts) >= 3 and parts[0] == "tool":
        return f"{parts[1]}.{parts[2]}"
    return None


def _candidate_signature(cand: Mapping[str, Any]) -> Tuple[str, Any]:
    meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
    qualified = meta.get("qualified_tool_name")
    if not isinstance(qualified, str) or not qualified:
        qualified = _tool_name_from_view_id(cand.get("tool_view_id") if isinstance(cand.get("tool_view_id"), str) else None)
    if not isinstance(qualified, str) or not qualified:
        qualified = str(cand.get("tool") or "")

    args = meta.get("args") if isinstance(meta.get("args"), dict) else None
    if args is not None:
        return qualified, _normalize_json(args)

    text = cand.get("text")
    text_s = text.strip() if isinstance(text, str) else ""
    return qualified, text_s


def _iter_candidate_lists(payload: Any) -> Iterable[list]:
    stack = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            cands = cur.get("candidates")
            if isinstance(cands, list) and cands and isinstance(cands[0], dict) and "source" in cands[0]:
                yield cands
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


@dataclass(frozen=True)
class TreeCandidateStats:
    templates: int
    slots_total: int
    slots_with_any_tool: int
    slots_with_effective_tool: int
    candidates_total: int
    candidates_tool_total: int
    candidates_memory_total: int
    tool_shadowed_by_memory: int
    tool_effective_total: int


def compute_stats(tree_payload: Mapping[str, Any]) -> TreeCandidateStats:
    templates = len(tree_payload)
    slots_total = 0
    slots_with_any_tool = 0
    slots_with_effective_tool = 0
    candidates_total = 0
    candidates_tool_total = 0
    candidates_memory_total = 0
    tool_shadowed_by_memory = 0
    tool_effective_total = 0

    for cands in _iter_candidate_lists(tree_payload):
        slots_total += 1
        candidates_total += len(cands)

        memory_sigs = set()
        tool_cands: list[Mapping[str, Any]] = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            src = str(c.get("source") or "").strip().lower()
            if src == "memory":
                candidates_memory_total += 1
                memory_sigs.add(_candidate_signature(c))
            elif src == "tool":
                candidates_tool_total += 1
                tool_cands.append(c)

        if tool_cands:
            slots_with_any_tool += 1

        any_effective_here = False
        for tc in tool_cands:
            sig = _candidate_signature(tc)
            if sig in memory_sigs:
                tool_shadowed_by_memory += 1
            else:
                tool_effective_total += 1
                any_effective_here = True
        if any_effective_here:
            slots_with_effective_tool += 1

    return TreeCandidateStats(
        templates=int(templates),
        slots_total=int(slots_total),
        slots_with_any_tool=int(slots_with_any_tool),
        slots_with_effective_tool=int(slots_with_effective_tool),
        candidates_total=int(candidates_total),
        candidates_tool_total=int(candidates_tool_total),
        candidates_memory_total=int(candidates_memory_total),
        tool_shadowed_by_memory=int(tool_shadowed_by_memory),
        tool_effective_total=int(tool_effective_total),
    )


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Tree JSON must be an object mapping template_id -> tree entry: {path}")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute tool-vs-memory candidate stats from tree_<split>.json.")
    p.add_argument("tree_json", nargs="+", help="Path(s) to tree_<split>.json.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON per file.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    for raw in args.tree_json:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        payload = _load_json(path)
        stats = compute_stats(payload)
        if bool(args.json):
            out = {
                "path": str(path),
                **stats.__dict__,
            }
            print(json.dumps(out, ensure_ascii=False))
        else:
            print(
                f"{path}: tool_effective_total={stats.tool_effective_total} "
                f"(tool_total={stats.candidates_tool_total} shadowed={stats.tool_shadowed_by_memory} "
                f"slots_with_effective_tool={stats.slots_with_effective_tool}/{stats.slots_total})"
            )


if __name__ == "__main__":
    main()


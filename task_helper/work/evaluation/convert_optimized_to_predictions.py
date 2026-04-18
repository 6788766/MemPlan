from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _iter_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _action_to_call(action: dict) -> str:
    attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
    raw = attrs.get("raw_action")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    qualified = attrs.get("qualified_tool_name")
    if not isinstance(qualified, str) or not qualified.strip():
        tool_name = attrs.get("tool_name")
        action_type = action.get("action_type")
        if isinstance(tool_name, str) and isinstance(action_type, str) and tool_name and action_type:
            qualified = f"{tool_name}.{action_type}"
        else:
            qualified = ""

    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    parts = []
    for key, value in params.items():
        if value is None:
            continue
        text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{key}="{text}"')
    return f"{qualified}.func(" + ", ".join(parts) + ")"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MemPlan optimized JSONL to WorkBench predictions CSV(s).")
    parser.add_argument("--optimized", type=Path, required=True, help="Path to optimized_<split>.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for per-domain CSVs")
    parser.add_argument(
        "--default-domain",
        type=str,
        default="unknown",
        help="Domain label to use when plan.domains is missing/empty",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    items = _iter_jsonl(args.optimized)

    by_domain: Dict[str, List[dict]] = defaultdict(list)
    for obj in items:
        plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else {}
        query = str(plan.get("query") or "").strip()
        domains = plan.get("domains")
        domain = args.default_domain
        if isinstance(domains, list) and domains:
            domain = str(domains[0] or domain)
        if not query:
            continue

        actions = obj.get("actions") if isinstance(obj.get("actions"), list) else []
        calls = [_action_to_call(a) for a in actions if isinstance(a, dict)]
        by_domain[domain].append(
            {
                "query": query,
                "function_calls": calls,
                "error": "",
                "full_response": "{}",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for domain, rows in sorted(by_domain.items(), key=lambda kv: kv[0]):
        df = pd.DataFrame(rows)
        out_path = args.out_dir / f"predictions_{domain}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} row(s) to {out_path}")


if __name__ == "__main__":
    main()


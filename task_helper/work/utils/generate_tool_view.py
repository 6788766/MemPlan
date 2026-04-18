"""
Generate WorkBench tool view patterns.

Writes `artifacts/input/work/views/tool.json`, describing tool-backed "views"
as single Action-node patterns with placeholder params, plus a per-domain `cost`
normalised to [0, 1] based on the size of each domain's backing CSV table.

This mirrors the TravelPlanner tool view format so `planner/view_select.py` and
`planner/compose_match.py` can load the views.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# Ensure repo imports work when invoked as a script.
for parent in Path(__file__).resolve().parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from task_helper.work.tools import company_directory
from task_helper.work.tools.tooling import Tool
from task_helper.work.tools.toolkits import all_tools, tools_with_side_effects


@dataclass(frozen=True)
class DomainInfo:
    name: str
    table_path: Optional[Path]
    entry_count: int


SEARCH_TOOL_CAP = 5

# Tools that return at most 5 rows (per WorkBench tool implementations).
_CAP_LIMIT_BY_ACTION_TYPE: Dict[str, int] = {
    "search_events": SEARCH_TOOL_CAP,
    "search_emails": SEARCH_TOOL_CAP,
    "search_customers": SEARCH_TOOL_CAP,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _database_dir() -> Path:
    return _project_root() / "artifacts" / "input" / "work" / "dataset" / "database"


def _normalise_costs(total_entries: Dict[str, int]) -> Dict[str, float]:
    counts = [int(v) for v in total_entries.values() if v is not None]
    if not counts:
        return {}
    maximum = max(counts)
    if maximum <= 0:
        return {k: 0.0 for k in total_entries}
    denom = math.log1p(float(maximum))
    return {k: math.log1p(float(int(v))) / denom for k, v in total_entries.items()}


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _domain_from_tool_name(full_name: str) -> str:
    return full_name.split(".", 1)[0]


def _action_type_from_tool_name(full_name: str) -> str:
    return full_name.split(".", 1)[1]


def _tool_param_names(tool: Tool) -> List[str]:
    sig = inspect.signature(tool.func)
    names: List[str] = []
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        names.append(param.name)
    return names


def _load_domain_infos(db_dir: Path) -> Dict[str, DomainInfo]:
    def _count_csv(path: Path) -> int:
        if not path.exists():
            return 0
        df = pd.read_csv(path, dtype=str)
        return int(len(df))

    domain_to_file = {
        "calendar": db_dir / "calendar_events.csv",
        "email": db_dir / "emails.csv",
        "analytics": db_dir / "analytics_data.csv",
        "project_management": db_dir / "project_tasks.csv",
        "customer_relationship_manager": db_dir / "customer_relationship_manager_data.csv",
        "company_directory": None,  # derived
    }

    infos: Dict[str, DomainInfo] = {}
    for domain, path in domain_to_file.items():
        if path is None:
            infos[domain] = DomainInfo(
                name=domain,
                table_path=None,
                entry_count=len(getattr(company_directory, "EMAIL_ADDRESSES", []) or []),
            )
        else:
            infos[domain] = DomainInfo(name=domain, table_path=path, entry_count=_count_csv(path))
    return infos


def _build_tool_views(*, tool_costs: Dict[str, float]) -> List[dict]:
    side_effect_names = {t.name for t in tools_with_side_effects}
    tools_sorted = sorted(all_tools, key=lambda t: t.name)

    views: List[dict] = []
    for tool in tools_sorted:
        full_name = str(tool.name)
        domain = _domain_from_tool_name(full_name)
        action_type = _action_type_from_tool_name(full_name)
        params = {p: f"{{{p}}}" for p in _tool_param_names(tool)}

        attrs: Dict[str, object] = {
            "action_type": action_type,
            "params": params,
            "tool_name": domain,
            "qualified_tool_name": full_name,
            "side_effect": full_name in side_effect_names,
        }
        cap = _CAP_LIMIT_BY_ACTION_TYPE.get(action_type)
        if cap is not None:
            attrs["cap_limit"] = cap

        cost = float(tool_costs.get(domain, 0.0))
        views.append(
            {
                "view_id": f"tool::{domain}::{action_type}",
                "tool": domain,
                "cost": cost,
                "node_pattern": {
                    "type": "Action",
                    "attrs": attrs,
                },
            }
        )

    return views


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_out = _project_root() / "artifacts" / "input" / "work" / "views" / "tool.json"
    parser = argparse.ArgumentParser(description="Generate artifacts/input/work/views/tool.json.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output path for tool.json")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_path: Path = args.out.expanduser()

    db_dir = _database_dir()
    domains = _load_domain_infos(db_dir)

    domain_entry_counts = {d.name: int(d.entry_count) for d in domains.values()}
    tool_costs = _normalise_costs(domain_entry_counts)

    views = _build_tool_views(tool_costs=tool_costs)

    root = _project_root()
    payload = {
        "version": 1,
        "source_database_dir": str(db_dir.relative_to(root)),
        "tool_total_entries": domain_entry_counts,
        "tool_costs": tool_costs,
        "views": views,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(views)} tool view(s) to {out_path}")


if __name__ == "__main__":
    main()


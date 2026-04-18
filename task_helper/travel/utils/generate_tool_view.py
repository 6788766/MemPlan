"""
Generate TravelPlanner tool view patterns from the dataset database README.

Writes `artifacts/input/travel/views/tool.json`, describing tool-backed "views"
as single Action-node patterns with placeholder params, plus a per-tool `cost`
normalised to [0, 1] based on table sizes.

Note: not all TravelPlanner tools are documented in `database/README.md`.
`CitySearch` is backed by the `database/background/` text files and is added
explicitly by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

for parent in Path(__file__).resolve().parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from task_helper.travel.utils.paths import project_root, travel_dataset_root


_SECTION_RE = re.compile(r"^\s*###\s+(?P<name>.+?)\s*$")
_FIELD_RE = re.compile(r"^\s*\d+\.\s+\*\*(?P<field>.+?)\*\*\s*:")
_PLACEHOLDER_RE = re.compile(r"\{(?P<name>[^}]+)\}")

# Total entry counts (rows) for each TravelPlanner tool table.
# These are used to assign a relative tool cost normalised to [0, 1].
#
# Update these numbers if the underlying database changes.
TOOL_TOTAL_ENTRIES: Dict[str, int] = {
    "Flights": 3_827_361,  # FlightSearch
    "GoogleDistanceMatrix": 17_603,  # DistanceMatrix
    "Restaurants": 9_552,  # RestaurantSearch
    "Attractions": 5_303,  # AttractionSearch
    "Accommodations": 5_064,  # AccommodationSearch
}

# `CitySearch` uses the background lookup tables instead of a CSV database file.
_CITYSEARCH_BACKGROUND_FILES = (
    "citySet_with_states.txt",
    "citySet.txt",
    "stateSet.txt",
)

# Extract `(city,state)` from TravelPlanner "current_city" strings like:
#   - "San Antonio(Texas)"
#   - "from Orlando to San Antonio(Texas)"
#   - "from San Antonio(Texas) to Houston(Texas)"
_CURRENT_CITY_STATE_RE = r"(?:.*\bto\s+)?(?P<city>[^()]+?)\((?P<state>[^)]+)\)\s*$"


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_database_readme(path: Path) -> Dict[str, List[str]]:
    """
    Parse `database/README.md` into section -> field names.

    Supports:
      - numbered lists: `1. **Field**: ...`
      - placeholder braces: `{origin}`, `{cost}`, ...
    """

    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        match = _SECTION_RE.match(line)
        if match:
            current = match.group("name").strip()
            sections.setdefault(current, [])
            continue
        if not current:
            continue

        match = _FIELD_RE.match(line)
        if match:
            sections[current].append(match.group("field").strip())
            continue

        for placeholder in _PLACEHOLDER_RE.findall(line):
            name = str(placeholder).strip()
            if name:
                sections[current].append(name)

    return {section: _dedupe_keep_order(fields) for section, fields in sections.items()}


def _normalise_tool_costs(tool_total_entries: Dict[str, int]) -> Dict[str, float]:
    if not tool_total_entries:
        return {}
    maximum = max(int(v) for v in tool_total_entries.values() if v is not None)
    if maximum <= 0:
        return {tool: 0.0 for tool in tool_total_entries}
    # Log-scaling avoids extreme gaps when one table is orders of magnitude larger.
    denom = math.log1p(float(maximum))
    return {tool: math.log1p(float(int(count))) / denom for tool, count in tool_total_entries.items()}


def _count_citysearch_entries(dataset_root: Path) -> int:
    """
    CitySearch is backed by the `database/background/` lookup tables (3 text files).
    Use the total number of non-empty lines as the "entry count" for cost scaling.
    """

    background_dir = dataset_root / "database" / "background"
    total = 0
    for filename in _CITYSEARCH_BACKGROUND_FILES:
        path = background_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"CitySearch background file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        total += sum(1 for line in lines if line.strip())
    return total


def _build_tool_views(readme_fields: Dict[str, List[str]], *, tool_costs: Dict[str, float]) -> List[dict]:
    """
    Convert parsed README fields into action-node view specs.
    """

    view_specs = [
        ("Attractions", "Visit", "Attractions", None),
        ("Restaurants", "Eat", "Restaurants", None),
        ("Accommodations", "Stay", "Accommodations", None),
        ("Flights", "Move", "Flights", "Flight"),
        ("Self-driving", "Move", "GoogleDistanceMatrix", "Self-driving"),
        ("Taxi", "Move", "GoogleDistanceMatrix", "Taxi"),
    ]

    views: List[dict] = []
    for section, action_type, tool_name, mode in view_specs:
        fields = readme_fields.get(section)
        if not fields:
            raise ValueError(f"Missing or empty section in README: {section}")

        params: Dict[str, str]
        if action_type == "Move" and mode == "Flight":
            # Normalise shared Move params across modes:
            #   origin, destination, distance, cost
            placeholder = {field: f"{{{field}}}" for field in fields}
            params = {
                "origin": placeholder.get("OriginCityName", "{OriginCityName}"),
                "destination": placeholder.get("DestCityName", "{DestCityName}"),
                "distance": placeholder.get("Distance", "{Distance}"),
                "cost": placeholder.get("Price", "{Price}"),
            }
            for field in fields:
                if field in {"OriginCityName", "DestCityName", "Distance", "Price"}:
                    continue
                params[field] = placeholder[field]
        else:
            params = {field: f"{{{field}}}" for field in fields}
        node_attrs: dict = {
            "action_type": action_type,
            "params": params,
        }
        if mode:
            node_attrs["mode"] = mode

        cost = tool_costs.get(tool_name)
        if cost is None:
            raise ValueError(f"Missing tool cost for tool '{tool_name}'. Please update TOOL_TOTAL_ENTRIES.")

        views.append(
            {
                "view_id": f"tool::{tool_name}::{action_type}" + (f"::{mode}" if mode else ""),
                "tool": tool_name,
                "cost": cost,
                "node_pattern": {
                    "type": "Action",
                    "attrs": node_attrs,
                },
            }
        )

    return views


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    dataset_root = travel_dataset_root()
    default_readme = dataset_root / "database" / "README.md"
    default_out = project_root() / "artifacts" / "input" / "travel" / "views" / "tool.json"

    parser = argparse.ArgumentParser(description="Generate artifacts/input/travel/views/tool.json.")
    parser.add_argument("--readme", type=Path, default=default_readme, help="Path to database/README.md")
    parser.add_argument("--out", type=Path, default=default_out, help="Output path for tool.json")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    readme_path = args.readme.expanduser()
    out_path = args.out.expanduser()

    fields_by_section = _parse_database_readme(readme_path)
    dataset_root = travel_dataset_root()
    tool_total_entries = dict(TOOL_TOTAL_ENTRIES)
    tool_total_entries["CitySearch"] = _count_citysearch_entries(dataset_root)
    tool_costs = _normalise_tool_costs(tool_total_entries)
    views = _build_tool_views(fields_by_section, tool_costs=tool_costs)
    views.append(
        {
            "view_id": "tool::CitySearch::DayRecord",
            "tool": "CitySearch",
            "cost": tool_costs["CitySearch"],
            "node_pattern": {
                "type": "Action",
                "attrs": {
                    "action_type": "DayRecord",
                    "params": {
                        "day": "{day}",
                        "current_city": "{current_city}",
                    },
                },
            },
            "extractors": [
                {
                    "type": "regex_capture",
                    "source_param": "current_city",
                    "pattern": _CURRENT_CITY_STATE_RE,
                    "flags": ["IGNORECASE"],
                    "group_to_param": {
                        "city": "city",
                        "state": "state",
                    },
                }
            ],
        }
    )

    root = project_root()
    try:
        source_readme = str(readme_path.resolve().relative_to(root))
    except ValueError:
        source_readme = str(readme_path)

    payload = {
        "version": 1,
        "source_readme": source_readme,
        "tool_costs": tool_costs,
        "views": views,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(views)} tool view(s) to {out_path}")


if __name__ == "__main__":
    main()

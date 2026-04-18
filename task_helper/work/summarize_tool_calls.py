#!/usr/bin/env python3
"""
Summarize tool-call usage from a WorkBench optimized_<split>.jsonl file.

Outputs:
  calls_total=<int> calls_check=<int> calls_do=<int> by_domain=<json object>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize WorkBench tool-call usage from optimized JSONL.")
    parser.add_argument("path", type=Path, help="Path to optimized_<split>.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path: Path = args.path.expanduser().resolve()

    total = 0
    do_calls = 0
    check_calls = 0
    by_domain: dict[str, int] = defaultdict(int)

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        actions = obj.get("actions")
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            phase_key = str(action.get("phase_key") or "")
            kind = "do" if phase_key.lower().startswith("do") else ("check" if phase_key.lower().startswith("check") else "")
            attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
            qualified = attrs.get("qualified_tool_name")
            if isinstance(qualified, str) and qualified:
                domain = qualified.split(".", 1)[0] if "." in qualified else qualified
            else:
                domain = str(attrs.get("tool_name") or "")

            total += 1
            if domain:
                by_domain[domain] += 1
            if kind == "do":
                do_calls += 1
            elif kind == "check":
                check_calls += 1

    print(
        "calls_total=%d calls_check=%d calls_do=%d by_domain=%s"
        % (total, check_calls, do_calls, json.dumps(dict(sorted(by_domain.items())), ensure_ascii=False))
    )


if __name__ == "__main__":
    main()


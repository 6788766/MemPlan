#!/usr/bin/env python3
"""
Compute aggregate init_template token usage from an init_templates_<split>.jsonl file.

Outputs a single line formatted like:
  calls=... prompt_cache_hit=... prompt_cache_miss=... output=... total=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize init_template token usage from a JSONL file.")
    parser.add_argument("path", type=Path, help="Path to init_templates_<split>.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path: Path = args.path.expanduser().resolve()

    cache_hit = cache_miss = output = total = calls = 0
    if path.exists():
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
            notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else None
            llm = notes.get("llm") if isinstance(notes, dict) else None
            if not isinstance(llm, dict):
                continue

            calls += 1
            hit = int(llm.get("prompt_cache_hit_tokens") or 0)
            miss = llm.get("prompt_cache_miss_tokens")
            if miss is None:
                prompt_tokens = int(llm.get("prompt_tokens") or 0)
                miss = max(0, prompt_tokens - hit) if hit else prompt_tokens
            miss = int(miss or 0)
            cache_hit += hit
            cache_miss += miss

            completion_tokens = int(llm.get("completion_tokens") or 0)
            output += completion_tokens

            total_tokens = llm.get("total_tokens")
            if total_tokens is None:
                total_tokens = (hit + miss) + completion_tokens
            total += int(total_tokens or 0)

    print(f"calls={calls} prompt_cache_hit={cache_hit} prompt_cache_miss={cache_miss} output={output} total={total}")


if __name__ == "__main__":
    main()


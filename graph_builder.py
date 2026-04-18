"""
Convenience CLI entrypoint for task-agnostic plan graph extraction.

This wrapper calls :func:`memory_graph.builder.main` so you can run the extractor
as:

  `python graph_builder.py --task travel --limit 5`

Equivalent to:

  `python -m memory_graph.builder --task travel --limit 5`

Inputs (by default):
  - `artifacts/input/<task>/graph_config.json`
  - `artifacts/input/<task>/dataset/train.csv`

Outputs:
  - `artifacts/output/<task>/memory_graph.jsonl`
  - `artifacts/output/<task>/memory_graph.nt`
  - `artifacts/output/<task>/memory_graph.graphml`

Note: this is separate from the TravelPlanner runners under
`task_helper/travel/runners/`, which run the end-to-end MemPlan planning
pipeline.
"""

from __future__ import annotations

from typing import Optional, Sequence

from memory_graph.builder import main as _main


def main(argv: Optional[Sequence[str]] = None) -> None:
    _main(argv)


if __name__ == "__main__":
    main()

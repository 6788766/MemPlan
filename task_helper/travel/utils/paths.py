from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def project_root(start: Optional[Path] = None) -> Path:
    """
    Best-effort repo root detection used for locating artifacts/ and task helper code.
    """

    probe = start or Path(__file__).resolve()
    for parent in probe.resolve().parents:
        if (parent / "artifacts").is_dir() and ((parent / "task_helper").is_dir() or (parent / "tasks").is_dir()):
            return parent
    raise FileNotFoundError(
        "Could not locate project root (expected 'artifacts/' plus either 'task_helper/' or legacy 'tasks/' directory)."
    )


def travel_dataset_root(start: Optional[Path] = None) -> Path:
    """
    Locate the TravelPlanner dataset root.

    Preferred location: artifacts/input/travel/dataset
    Legacy fallback:    tasks/travel/dataset (older layouts)
    """

    root = project_root(start)

    override = os.getenv("MEMPLAN_TRAVEL_DATASET_ROOT")
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = (root / override_path).resolve()
        if override_path.exists():
            return override_path

    candidates = [
        root / "artifacts" / "input" / "travel" / "dataset",
        root / "task_helper" / "travel" / "dataset",
        root / "tasks" / "travel" / "dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate TravelPlanner dataset. Expected one of: "
        f"{candidates[0]} or {candidates[1]} (or set MEMPLAN_TRAVEL_DATASET_ROOT)."
    )

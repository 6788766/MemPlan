from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # `task_helper/work/tools/paths.py` -> tools -> work -> task_helper -> repo root
    return Path(__file__).resolve().parents[3]


DATABASE_DIR = repo_root() / "artifacts" / "input" / "work" / "dataset" / "database"


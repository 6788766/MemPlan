from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "artifacts" / "input" / "travel" / "dataset"


def load_travelplanner_dataset(set_type: str) -> List[Dict[str, str]]:
    """Return the local TravelPlanner CSV split as a list of dict rows.

    TravelPlanner's original repo loads from HuggingFace; in MemPlan we keep a
    local copy under `artifacts/input/travel/dataset/<split>.csv`.
    """
    path = DATASET_ROOT / f"{set_type}.csv"
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


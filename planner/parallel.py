from __future__ import annotations

import os
import re
from typing import Optional


def _parse_positive_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    match = re.match(r"^(\d+)", raw)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def available_cpu_count() -> int:
    """Best-effort CPU count for the current process.

    On HPC systems (e.g. Slurm), `os.cpu_count()` often reports the full node CPU
    count, not the CPUs allocated to the job. Prefer CPU affinity when
    available, and fall back to common scheduler env vars.
    """

    try:
        affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]
    except Exception:
        affinity = None
    if affinity:
        return max(1, len(affinity))

    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = _parse_positive_int(os.environ.get(key))
        if value is not None:
            return value

    slurm_node = _parse_positive_int(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    if slurm_node is not None:
        return slurm_node

    return int(os.cpu_count() or 1)


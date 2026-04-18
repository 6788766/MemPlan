from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from baseline.rag.common import (
    TokenUsage,
    aggregate_token_usage,
    compute_price_usd,
    iso_now,
    model_slug,
    resolve_price_key,
    sum_mapping_values,
    write_jsonl,
)
from planner.parallel import available_cpu_count


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAVEL_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "travel" / "nomcts"
WORK_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "work" / "nomcts"

_KV_INT_RE = re.compile(r"\b([A-Za-z_]+)=([0-9]+)\b")
_ELAPSED_S_RE = re.compile(r"\belapsed_s=([0-9]+(?:\.[0-9]+)?)\b")


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def resolve_workers(requested_workers: int, item_count: int) -> int:
    if item_count <= 1:
        return 1
    if int(requested_workers) <= 0:
        workers = int(available_cpu_count())
    else:
        workers = int(requested_workers)
    return max(1, min(workers, int(item_count)))


def stable_seed(base_seed: Optional[int], template_id: str) -> int:
    seed_text = f"{0 if base_seed is None else int(base_seed)}::{template_id}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % (2**31 - 1))


def read_cost_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def read_cost_value(lines: Sequence[str], key: str) -> Optional[str]:
    prefix = f"{key}="
    for line in reversed(list(lines)):
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def read_cost_float(lines: Sequence[str], key: str) -> Optional[float]:
    raw = read_cost_value(lines, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def read_cost_int(lines: Sequence[str], key: str) -> Optional[int]:
    raw = read_cost_value(lines, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def sum_stage_elapsed_seconds(lines: Sequence[str], stage: str) -> float:
    total = 0.0
    token = f"] END {stage} "
    for line in lines:
        if token not in line:
            continue
        match = _ELAPSED_S_RE.search(line)
        if not match:
            continue
        try:
            total += float(match.group(1))
        except Exception:
            continue
    return float(total)


def _parse_usage_line(line: str) -> TokenUsage:
    values: Dict[str, int] = {}
    for key, raw_value in _KV_INT_RE.findall(line):
        try:
            values[key.strip().lower()] = int(raw_value)
        except Exception:
            continue

    prompt_tokens = values.get("prompt_tokens") or values.get("prompt") or values.get("input_tokens") or values.get("input") or 0
    cache_hit = values.get("prompt_cache_hit_tokens") or values.get("prompt_cache_hit") or values.get("cached_tokens") or values.get("cached") or 0
    cache_miss = values.get("prompt_cache_miss_tokens") or values.get("prompt_cache_miss")
    if cache_miss is None:
        cache_miss = max(0, int(prompt_tokens) - int(cache_hit))
    completion = values.get("completion_tokens") or values.get("completion") or values.get("output_tokens") or values.get("output") or 0
    total_tokens = values.get("total_tokens") or values.get("total")
    if total_tokens is None:
        total_tokens = int(cache_hit) + int(cache_miss) + int(completion)
    calls = values.get("calls") or 0
    approx_calls = values.get("approx_calls") or 0

    return TokenUsage(
        calls=int(calls),
        prompt_cache_hit_tokens=int(cache_hit),
        prompt_cache_miss_tokens=int(cache_miss),
        completion_tokens=int(completion),
        total_tokens=int(total_tokens),
        prompt_tokens=int(prompt_tokens or (int(cache_hit) + int(cache_miss))),
        approx_calls=int(approx_calls),
    )


def find_stage_usage(lines: Sequence[str], stage_name: str) -> TokenUsage:
    prefix = f"LLM token usage ({stage_name}):"
    for line in reversed(list(lines)):
        if prefix in line:
            return _parse_usage_line(line)
    return TokenUsage()


def find_usage_line_text(lines: Sequence[str], stage_name: str) -> Optional[str]:
    prefix = f"LLM token usage ({stage_name}):"
    for line in reversed(list(lines)):
        if prefix in line:
            return line.strip()
    return None


def usage_from_llm_notes(payload: Optional[Mapping[str, object]]) -> TokenUsage:
    if not isinstance(payload, Mapping):
        return TokenUsage()
    prompt_tokens = int(payload.get("prompt_tokens") or 0)
    cache_hit = int(payload.get("prompt_cache_hit_tokens") or 0)
    cache_miss = payload.get("prompt_cache_miss_tokens")
    if cache_miss is None:
        cache_miss = max(0, prompt_tokens - cache_hit)
    completion = int(payload.get("completion_tokens") or payload.get("output_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or (cache_hit + int(cache_miss) + completion))
    return TokenUsage(
        calls=1 if total_tokens > 0 else 0,
        prompt_cache_hit_tokens=cache_hit,
        prompt_cache_miss_tokens=int(cache_miss or 0),
        completion_tokens=completion,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens or (cache_hit + int(cache_miss or 0)),
    )


def load_init_template_records(path: Path) -> Dict[str, Dict[str, object]]:
    return {str(obj.get("template_id")): obj for obj in iter_jsonl(path)}


def load_init_template_usage_map(path: Path) -> Tuple[Dict[str, TokenUsage], TokenUsage]:
    usage_by_template: Dict[str, TokenUsage] = {}
    total = TokenUsage()
    for obj in iter_jsonl(path):
        template_id = str(obj.get("template_id") or "")
        notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
        llm = notes.get("llm") if isinstance(notes, dict) else None
        usage = usage_from_llm_notes(llm if isinstance(llm, Mapping) else None)
        usage_by_template[template_id] = usage
        total.add(usage)
    return usage_by_template, total


def summarize_travel_tool_calls(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"total_calls": 0, "total_cost": 0.0, "by_tool": {}}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {"total_calls": 0, "total_cost": 0.0, "by_tool": {}}
    by_tool = payload.get("by_tool") if isinstance(payload.get("by_tool"), dict) else {}
    return {
        "total_calls": int(payload.get("total_calls") or 0),
        "total_cost": float(payload.get("total_cost") or 0.0),
        "by_tool": by_tool,
    }


def summarize_work_tool_calls(path: Path) -> Dict[str, object]:
    calls_total = 0
    calls_check = 0
    calls_do = 0
    by_domain: Dict[str, int] = {}

    if not path.exists():
        return {
            "calls_total": 0,
            "calls_check": 0,
            "calls_do": 0,
            "by_domain": {},
        }

    for obj in iter_jsonl(path):
        actions = obj.get("actions") if isinstance(obj.get("actions"), list) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            phase_key = str(action.get("phase_key") or "")
            attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
            qualified = str(attrs.get("qualified_tool_name") or attrs.get("tool_name") or "").strip()
            domain = qualified.split(".", 1)[0] if "." in qualified else qualified
            if phase_key.lower().startswith("check"):
                calls_check += 1
            elif phase_key.lower().startswith("do"):
                calls_do += 1
            calls_total += 1
            if domain:
                by_domain[domain] = int(by_domain.get(domain, 0)) + 1

    return {
        "calls_total": int(calls_total),
        "calls_check": int(calls_check),
        "calls_do": int(calls_do),
        "by_domain": dict(sorted(by_domain.items())),
    }


def run_subprocess(
    *,
    cmd: Sequence[str],
    out_path: Path,
    header: Optional[str] = None,
    stream_output: bool = False,
) -> Tuple[float, str]:
    started_at = time.perf_counter()
    if header:
        with out_path.open("a", encoding="utf-8") as fp:
            fp.write(str(header).rstrip() + "\n")
    if stream_output:
        chunks: List[str] = []
        with out_path.open("a", encoding="utf-8") as fp:
            proc = subprocess.Popen(
                list(cmd),
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                chunks.append(line)
                fp.write(line)
                fp.flush()
                print(line, end="", flush=True)
            proc.wait()
            if chunks and not chunks[-1].endswith("\n"):
                fp.write("\n")
        combined = "".join(chunks)
    else:
        proc = subprocess.run(
            list(cmd),
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
    if combined:
        if not stream_output:
            with out_path.open("a", encoding="utf-8") as fp:
                fp.write(combined)
                if not combined.endswith("\n"):
                    fp.write("\n")
            print(combined, end="")
    if proc.returncode != 0:
        tail_lines = [line for line in combined.splitlines() if line.strip()][-20:]
        message = (
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"log_path={out_path}"
        )
        if tail_lines:
            message += "\nLast output lines:\n" + "\n".join(tail_lines)
        print(message, file=sys.stderr, flush=True)
        raise RuntimeError(message)
    return (time.perf_counter() - started_at), combined


def parse_travel_eval_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    names = (
        "Delivery Rate",
        "Commonsense Constraint Micro Pass Rate",
        "Commonsense Constraint Macro Pass Rate",
        "Hard Constraint Micro Pass Rate",
        "Hard Constraint Macro Pass Rate",
        "Final Pass Rate",
        "All Budget Pass Rate",
        "Local Pass Rate",
        "Global Pass Rate",
    )
    for name in names:
        marker = f"{name}:"
        index = text.find(marker)
        if index < 0:
            continue
        tail = text[index + len(marker) :].splitlines()[0].strip()
        value = tail.rstrip("%").strip()
        try:
            metrics[name] = float(value)
        except Exception:
            continue
    return metrics


def aggregate_repair_usage(records: Iterable[Mapping[str, object]]) -> TokenUsage:
    usages: List[TokenUsage] = []
    for record in records:
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        repair = notes.get("llm_repair") if isinstance(notes.get("llm_repair"), dict) else {}
        usage = usage_from_llm_notes(repair.get("usage") if isinstance(repair, dict) else None)
        usages.append(usage)
    return aggregate_token_usage(usages)


def append_total_cost_block(
    *,
    lines: List[str],
    total_model_key: str,
    total_tokens: int,
    total_time_s: float,
    stage_prices: Mapping[str, float],
) -> List[str]:
    out = list(lines)
    out.append("--------------total cost-----------------")
    out.append(f"llm_price_model_key={total_model_key}")
    for key, value in stage_prices.items():
        out.append(f"{key}={float(value):.6f}")
    out.append(f"total_tokens={int(total_tokens)}")
    out.append(f"total_time_s={float(total_time_s)}")
    return out


def build_combined_price_key(base_model: str, repair_model: Optional[str] = None) -> str:
    base_key = resolve_price_key(base_model)
    if not repair_model:
        return base_key
    repair_key = resolve_price_key(repair_model)
    return combine_price_keys(base_key, repair_key)


def combine_price_keys(base_key: str, repair_key: Optional[str] = None) -> str:
    if not repair_key:
        return base_key
    if repair_key == base_key:
        return base_key
    return f"{base_key}+{repair_key}"


def python_executable() -> str:
    return sys.executable or "python3"

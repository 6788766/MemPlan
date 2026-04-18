from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from planner import llm_repair as llmr
from planner.parallel import available_cpu_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_GT_DIR = ARTIFACTS_ROOT / "input" / "work" / "dataset" / "queries_and_answers"
DEFAULT_OUT_ROOT = ARTIFACTS_ROOT / "output" / "work" / "analysis"

_MODEL_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_LIST_BLOCK_RE = re.compile(r"\[[\s\S]*\]", flags=re.DOTALL)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _model_slug(name: str) -> str:
    slug = _MODEL_SLUG_RE.sub("", str(name or "")).strip().lower()
    return slug or "model"


def _ensure_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    yield from llmr._iter_jsonl(path)


def _run_py(cmd: Sequence[str], *, out_path: Optional[Path] = None) -> str:
    proc = subprocess.run(
        [sys.executable, *cmd],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _convert_to_predictions(*, optimized_jsonl: Path, out_dir: Path) -> None:
    _run_py(
        [
            "task_helper/work/evaluation/convert_optimized_to_predictions.py",
            "--optimized",
            str(optimized_jsonl),
            "--out-dir",
            str(out_dir),
        ]
    )


def _compute_pass_rates(*, predictions_dir: Path, ground_truth_dir: Path, json_out: Path) -> Dict[str, object]:
    _run_py(
        [
            "task_helper/work/evaluation/calculate_pass_rates_dir.py",
            "--predictions_dir",
            str(predictions_dir),
            "--ground_truth_dir",
            str(ground_truth_dir),
            "--json_out",
            str(json_out),
        ]
    )
    return _read_json(json_out)


def _compute_all_metrics(*, predictions_dir: Path, out_path: Path) -> None:
    _run_py(
        [
            "task_helper/work/evaluation/calculate_all_metrics.py",
            "--predictions_dir",
            str(predictions_dir),
        ],
        out_path=out_path,
    )


def _canonical_query(query: str) -> str:
    return str(query or "").strip()


def _work_action_dict_to_call(action: Mapping[str, object]) -> str:
    return llmr._work_action_dict_to_call(action)


def _work_record_to_calls(record: Mapping[str, object]) -> List[str]:
    actions = record.get("actions") if isinstance(record.get("actions"), list) else []
    calls: List[str] = []
    for a in actions:
        if isinstance(a, Mapping):
            calls.append(_work_action_dict_to_call(a))
    return [c for c in calls if str(c).strip()]


def _work_record_to_structured_calls(record: Mapping[str, object]) -> List[Dict[str, object]]:
    actions = record.get("actions") if isinstance(record.get("actions"), list) else []
    out: List[Dict[str, object]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        phase_key = str(a.get("phase_key") or "")
        attrs = a.get("attrs") if isinstance(a.get("attrs"), dict) else {}
        qualified = str(attrs.get("qualified_tool_name") or "").strip()
        params = a.get("params") if isinstance(a.get("params"), dict) else {}
        out.append(
            {
                "phase_key": phase_key,
                "qualified_tool_name": qualified,
                "params": {str(k): str(v) for k, v in params.items() if v is not None},
            }
        )
    return out


def _calls_to_actions(calls: Sequence[str]) -> List[Dict[str, object]]:
    actions: List[Dict[str, object]] = []
    for i, call in enumerate(calls):
        call_text = str(call or "").strip()
        if not call_text:
            continue
        actions.append(
            {
                "phase_key": "check_1",
                "action_type": "",
                "params": {},
                "attrs": {"raw_action": call_text},
                "order_index": i,
            }
        )
    return actions


def _terminal_width(default: int = 100) -> int:
    try:
        return int(shutil.get_terminal_size((default, 20)).columns)
    except Exception:
        return int(default)


def _render_bar(done: int, total: int, *, width: int = 22) -> str:
    total = max(0, int(total))
    done = max(0, min(int(done), total if total else int(done)))
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(round(width * (done / total)))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


class _MultiModelProgress:
    def __init__(self, models: Sequence[str], *, enabled: bool) -> None:
        self.models = [str(m) for m in models]
        self.enabled = bool(enabled)
        self._printed = False
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, object]] = {
            m: {"stage": "pending", "done": 0, "total": 0, "start": None, "updated_at": time.perf_counter()}
            for m in self.models
        }

    def update(self, model: str, *, stage: str, done: int, total: int) -> None:
        if str(model) not in self._state:
            return
        with self._lock:
            entry = self._state[str(model)]
            if entry.get("start") is None and stage not in {"pending"}:
                entry["start"] = time.perf_counter()
            entry["stage"] = str(stage)
            entry["done"] = int(done)
            entry["total"] = int(total)
            entry["updated_at"] = time.perf_counter()
            self._render()

    def _render_lines(self) -> List[str]:
        width = _terminal_width()
        lines: List[str] = []
        for m in self.models:
            s = self._state.get(m, {})
            stage = str(s.get("stage") or "")
            done = int(s.get("done") or 0)
            total = int(s.get("total") or 0)
            bar = _render_bar(done, total, width=18)
            elapsed = ""
            start = s.get("start")
            if isinstance(start, (int, float)):
                elapsed_s = max(0.0, time.perf_counter() - float(start))
                elapsed = f" {elapsed_s:6.1f}s"
            msg = f"{m:14} {stage:10} {bar} {done:>4}/{total:<4}{elapsed}"
            if len(msg) > width:
                msg = msg[: max(0, width - 1)]
            lines.append(msg)
        return lines

    def _render(self) -> None:
        if not self.enabled:
            return
        lines = self._render_lines()
        if not self._printed:
            for line in lines:
                print(line, flush=True)
            self._printed = True
            return
        # Repaint all lines in-place.
        sys.stdout.write(f"\x1b[{len(lines)}A")
        for line in lines:
            sys.stdout.write("\x1b[2K" + line + "\n")
        sys.stdout.flush()


def _format_constraints(*, include_descriptions: bool) -> str:
    from task_helper.work.evaluation.constraints import DEFAULT_CONSTRAINTS

    lines: List[str] = []
    for c in DEFAULT_CONSTRAINTS:
        kind = getattr(c, "kind", "")
        name = getattr(c, "name", "")
        desc = getattr(c, "description", "")
        if include_descriptions and isinstance(desc, str) and desc.strip():
            lines.append(f"- [{kind}] {name}: {desc}")
        else:
            lines.append(f"- [{kind}] {name}")
    return "\n".join(lines) if lines else "(none)"


def _format_tool_schemas(tool_schemas: Mapping[str, Mapping[str, object]]) -> str:
    lines: List[str] = []
    for qualified, schema in sorted(tool_schemas.items(), key=lambda kv: kv[0]):
        side_effect = bool(schema.get("side_effect", False))
        params = schema.get("param_keys") if isinstance(schema.get("param_keys"), list) else []
        params = [str(x) for x in params if str(x)]
        lines.append(f"- {qualified} | side_effect={side_effect} | params={', '.join(params) if params else '(none)'}")
    return "\n".join(lines) if lines else "(none)"


def _slot_key_for_prompt(slot: Mapping[str, object]) -> str:
    qa = slot.get("query_action") if isinstance(slot.get("query_action"), dict) else {}
    qualified = str(qa.get("qualified_tool_name") or "").strip()
    phase_edge = str(slot.get("slot_edge") or "")
    return f"{qualified} ({phase_edge})".strip()


def _candidate_call(meta: Mapping[str, object]) -> str:
    raw = meta.get("raw_action")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    qualified = str(meta.get("qualified_tool_name") or "").strip()
    args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
    parts = []
    for k, v in args.items():
        if v is None:
            continue
        text = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{k}="{text}"')
    return f"{qualified}.func(" + ", ".join(parts) + ")"


def _format_tree_for_prompt(
    *,
    tree: Mapping[str, object],
    max_slot_candidates: int,
    max_result_chars: int,
) -> Tuple[str, Set[str]]:
    root = tree.get("root") if isinstance(tree.get("root"), dict) else {}
    phases = root.get("children") if isinstance(root.get("children"), list) else []
    lines: List[str] = []
    allowed_calls: Set[str] = set()
    slot_counter = 0
    for ph in phases:
        if not isinstance(ph, dict):
            continue
        qp = ph.get("query_phase") if isinstance(ph.get("query_phase"), dict) else {}
        phase_key = str(qp.get("phase_key") or "")
        lines.append(f"PHASE: {phase_key or '(unknown)'}")
        slots = ph.get("children") if isinstance(ph.get("children"), list) else []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            slot_counter += 1
            lines.append(f"  SLOT {slot_counter}: {_slot_key_for_prompt(slot)}")
            candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
            for ci, cand in enumerate(candidates[: max(1, int(max_slot_candidates))]):
                if not isinstance(cand, dict):
                    continue
                meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                call = _candidate_call(meta)
                if call:
                    allowed_calls.add(call)
                side_effect = meta.get("side_effect")
                cost = cand.get("cost")
                prefix = f"    - cand {ci}: side_effect={bool(side_effect)} cost={cost} call={call}"
                result = meta.get("result")
                if result is not None:
                    try:
                        rendered = json.dumps(result, ensure_ascii=False)
                    except Exception:
                        rendered = str(result)
                    if len(rendered) > max_result_chars:
                        rendered = rendered[: max(0, max_result_chars - 1)] + "…"
                    prefix += f" | result={rendered}"
                lines.append(prefix)
    if slot_counter == 0:
        lines.append("(no slots found)")
    return "\n".join(lines), allowed_calls


def _build_prompt(
    *,
    query: str,
    chosen_plan_calls: Sequence[str],
    tool_list_text: str,
    constraints_text: str,
    tree_text: str,
    output_json_list: bool,
) -> str:
    chosen_block = "\n".join(chosen_plan_calls) if chosen_plan_calls else "(none)"
    output_spec = (
        "Output ONLY a JSON array of strings.\nEach string is a function call: tool.action.func(k=\"v\", ...)\n"
        if output_json_list
        else "Output ONLY function call strings, one per line. No extra text.\n"
    )
    return (
        "You are repairing a WorkBench plan.\n"
        "\n"
        "You are given:\n"
        "- QUERY\n"
        "- FULL TOOL LIST (all available tools)\n"
        "- CONSTRAINTS (evaluation criteria)\n"
        "- AND-OR TREE candidates (suggested candidates from search)\n"
        "- MCTS CHOSEN PLAN (the current plan to repair)\n"
        "\n"
        "Rules:\n"
        "- You MUST output a corrected full plan.\n"
        "- Prefer minimal changes to the chosen plan.\n"
        "- You may choose any tools from the FULL TOOL LIST; the AND-OR TREE is optional guidance.\n"
        "- Avoid unwanted side effects: do not include side_effect=true calls unless explicitly requested by the query.\n"
        "\n"
        "Output format:\n"
        f"{output_spec}"
        "\n"
        f"QUERY:\n{query}\n"
        "\n"
        "FULL TOOL LIST:\n"
        f"{tool_list_text}\n"
        "\n"
        "CONSTRAINTS:\n"
        f"{constraints_text}\n"
        "\n"
        "MCTS CHOSEN PLAN (current calls):\n"
        f"{chosen_block}\n"
        "\n"
        "AND-OR TREE (candidates):\n"
        f"{tree_text}\n"
    )


@dataclass(frozen=True)
class ParsedCalls:
    calls: List[str]
    error: str


def _parse_calls(content: str) -> ParsedCalls:
    text = str(content or "").strip()
    if not text:
        return ParsedCalls(calls=[], error="empty_response")

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return ParsedCalls(calls=[str(x) for x in obj if str(x).strip()], error="")
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return ParsedCalls(calls=[str(x) for x in obj if str(x).strip()], error="python_literal")
    except Exception:
        pass

    match = _LIST_BLOCK_RE.search(text)
    if match:
        block = match.group(0)
        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, list):
                return ParsedCalls(calls=[str(x) for x in obj if str(x).strip()], error="wrapped_text")
        except Exception:
            return ParsedCalls(calls=[], error="invalid_list_literal")

    calls: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        line = re.sub(r"^\s*\d+[\).\s]+", "", line).strip()
        if not line:
            continue
        if ".func(" not in line:
            continue
        calls.append(line)
    if calls:
        return ParsedCalls(calls=calls, error="line_list")
    return ParsedCalls(calls=[], error="unparseable")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorkBench baseline: provide AND-OR tree + constraints in plain text, repair call list.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split used in the base run dir naming.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the baseline for the default 3 GPT models in one grouped run (writes per-model subfolders).",
    )
    parser.add_argument("--models", action="append", default=[], help="Model list for --full (repeatable).")
    parser.add_argument("--base-model", type=str, default="", help="Model used to locate base artifacts (defaults to --model).")
    parser.add_argument("--optimized", type=Path, default=None, help="Path to optimized_<split>.jsonl (defaults from base model/split).")
    parser.add_argument("--tree", type=Path, default=None, help="Path to tree_enriched_<split>.json or tree_<split>.json.")
    parser.add_argument("--model", type=str, default=llmr.DEFAULT_MODEL, help="LLM model name.")
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpu_count(),
        help="Parallel workers (per-query repair workers).",
    )
    parser.add_argument("--max-failed", type=int, default=0, help="If >0, only attempt repair for first N failing plans.")
    parser.add_argument("--max-slot-candidates", type=int, default=20, help="Max candidates per OR-slot rendered into the prompt.")
    parser.add_argument("--max-result-chars", type=int, default=400, help="Max chars for any candidate result snippet in the prompt.")
    parser.add_argument("--include-constraint-descriptions", action="store_true", help="Include full constraint descriptions (longer prompt).")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (copies the chosen plan).")
    parser.add_argument("--enforce-tree-calls", action="store_true", help="Drop output calls not present in the tree candidate set.")
    parser.add_argument(
        "--output-json-list",
        action="store_true",
        help="Ask for a JSON array output (still not structured edits; default is call lines).",
    )
    parser.add_argument("--all-metrics", action="store_true", help="Also run calculate_all_metrics.py (slow).")
    parser.add_argument("--no-progress", action="store_true", help="Disable multi-model progress bars in --full mode.")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GT_DIR, help="WorkBench ground truth directory.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root directory.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run folder name override.")
    return parser.parse_args(argv)


def _run_one(
    *,
    split: str,
    base_model: str,
    model: str,
    optimized_path: Optional[Path],
    tree_path: Optional[Path],
    workers: int,
    max_failed: int,
    max_slot_candidates: int,
    max_result_chars: int,
    include_constraint_descriptions: bool,
    no_llm: bool,
    enforce_tree_calls: bool,
    output_json_list: bool,
    all_metrics: bool,
    gt_dir: Path,
    out_root: Path,
    run_name: str,
    progress: Optional[_MultiModelProgress] = None,
) -> Path:
    optimized = optimized_path
    tree = tree_path
    if optimized is None or tree is None:
        base_dir = ARTIFACTS_ROOT / "output" / "work" / f"{_model_slug(base_model)}_{split}"
        if optimized is None:
            optimized = base_dir / f"optimized_{split}.jsonl"
        if tree is None:
            preferred = base_dir / f"tree_enriched_{split}.json"
            tree = preferred if preferred.exists() else (base_dir / f"tree_{split}.json")

    optimized = _ensure_abs(Path(optimized))
    tree = _ensure_abs(Path(tree))
    if not optimized.exists():
        raise SystemExit(f"Missing optimized JSONL: {optimized}")
    if not tree.exists():
        raise SystemExit(f"Missing tree JSON: {tree}")

    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    trees_payload = json.loads(tree.read_text(encoding="utf-8"))
    if not isinstance(trees_payload, dict):
        raise SystemExit(f"Tree JSON must be an object mapping template_id -> tree: {tree}")

    tool_schemas: Dict[str, Dict[str, object]] = llmr._work_load_tool_schema_index()
    tool_list_text = _format_tool_schemas(tool_schemas)

    workbench_gt = llmr._load_workbench_ground_truth(ground_truth_dir=gt_dir)
    records = list(_iter_jsonl(optimized))
    total = len(records)
    if progress is not None:
        progress.update(model, stage="scan", done=0, total=0)

    attempt_indices: List[int] = []
    for idx, record in enumerate(records):
        metrics = llmr._workbench_metrics_for_work_record(record=record, ground_truth_by_query=workbench_gt)
        if metrics is None:
            continue
        correct, unwanted_side_effects = metrics
        if (not bool(correct)) or bool(unwanted_side_effects):
            attempt_indices.append(idx)
    if max_failed and int(max_failed) > 0:
        attempt_indices = attempt_indices[: int(max_failed)]
    if progress is not None:
        progress.update(model, stage="base_eval", done=0, total=0)

    stamp = _utc_stamp()
    _write_json(
        run_dir / "run_meta.json",
        {
            "ts_utc": stamp,
            "split": split,
            "base_model": base_model,
            "optimized": str(optimized),
            "tree": str(tree),
            "llm_model": model,
            "workers": int(workers),
            "no_llm": bool(no_llm),
            "enforce_tree_calls": bool(enforce_tree_calls),
            "max_failed": int(max_failed),
            "max_slot_candidates": int(max_slot_candidates),
            "max_result_chars": int(max_result_chars),
            "include_constraint_descriptions": bool(include_constraint_descriptions),
            "ground_truth_dir": str(gt_dir),
            "total": int(total),
            "to_repair": int(len(attempt_indices)),
        },
    )

    # Base eval
    base_pred_dir = run_dir / "base_results"
    base_pass_rates_path = run_dir / "base_pass_rates.json"
    base_metrics_txt = run_dir / "base_metrics.txt"
    _convert_to_predictions(optimized_jsonl=optimized, out_dir=base_pred_dir)
    base_pass_rates = _compute_pass_rates(predictions_dir=base_pred_dir, ground_truth_dir=gt_dir, json_out=base_pass_rates_path)
    if all_metrics:
        _compute_all_metrics(predictions_dir=base_pred_dir, out_path=base_metrics_txt)
    if progress is not None:
        progress.update(model, stage="repair", done=0, total=len(attempt_indices))

    constraints_text = _format_constraints(include_descriptions=include_constraint_descriptions)

    token_counter = llmr.TokenCounter()
    responses_path = run_dir / "responses.jsonl"
    out_jsonl = run_dir / "unstructured_optimized.jsonl"

    output_records: List[Dict[str, object]] = [dict(r) for r in records]
    response_rows: List[Optional[Dict[str, object]]] = [None] * len(attempt_indices)

    start = time.perf_counter()
    completed = 0

    def _repair_one(pos: int, idx: int) -> Tuple[int, Dict[str, object], Optional[Dict[str, int]]]:
        record = records[idx]
        template_id = str(record.get("template_id") or "")
        tree_obj = trees_payload.get(template_id)
        if not isinstance(tree_obj, dict):
            return pos, {"template_id": template_id, "status": "missing_tree", "calls": []}, None

        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        query = _canonical_query(str(plan.get("query") or ""))

        chosen_calls = _work_record_to_calls(record)
        tree_text, allowed_calls = _format_tree_for_prompt(
            tree=tree_obj, max_slot_candidates=max_slot_candidates, max_result_chars=max_result_chars
        )

        resp_obj: Dict[str, object] = {"template_id": template_id, "query": query}
        usage: Optional[Dict[str, int]] = None
        if no_llm:
            parsed = ParsedCalls(calls=list(chosen_calls), error="no_llm")
            resp_obj.update({"status": "no_llm", "error": parsed.error, "calls": parsed.calls})
        else:
            prompt = _build_prompt(
                query=query,
                chosen_plan_calls=chosen_calls,
                tool_list_text=tool_list_text,
                constraints_text=constraints_text,
                tree_text=tree_text,
                output_json_list=output_json_list,
            )
            messages = [
                {"role": "system", "content": "Output only the requested call list format."},
                {"role": "user", "content": prompt},
            ]
            try:
                content, usage = llmr._call_openai_chat(model=str(model), messages=messages)
            except Exception as exc:
                max_ctx, requested = llmr._extract_context_limit(exc)
                if max_ctx is None or requested is None:
                    resp_obj.update({"status": "llm_error", "error": f"{type(exc).__name__}: {exc}", "calls": []})
                    content = ""
                    parsed = ParsedCalls(calls=[], error="llm_error")
                else:
                    truncated = llmr._truncate_messages_to_context_limit(
                        messages=messages,
                        model=str(model),
                        max_context_tokens=max_ctx,
                        margin_tokens=500,
                    )
                    content, usage = llmr._call_openai_chat(model=str(model), messages=truncated)
                    parsed = _parse_calls(content)
            else:
                parsed = _parse_calls(content)

            calls = list(parsed.calls)
            dropped = 0
            if enforce_tree_calls:
                kept: List[str] = []
                for c in calls:
                    if c in allowed_calls:
                        kept.append(c)
                    else:
                        dropped += 1
                calls = kept
            resp_obj.update(
                {
                    "status": "ok" if not parsed.error else "parsed_with_warnings",
                    "error": parsed.error,
                    "raw": str(content),
                    "calls": calls,
                    "dropped_not_in_tree": int(dropped),
                }
            )

        out = dict(record)
        out["actions"] = _calls_to_actions(resp_obj.get("calls") if isinstance(resp_obj.get("calls"), list) else [])
        notes = out.get("notes")
        if not isinstance(notes, dict):
            notes = {}
            out["notes"] = notes
        notes["unstructured_repair"] = {
            "mode": "tree_text_repair_calls",
            "model": str(model),
            "status": str(resp_obj.get("status") or ""),
            "parse_error": str(resp_obj.get("error") or ""),
            "usage": dict(usage) if usage is not None else None,
        }
        output_records[idx] = out
        return pos, resp_obj, usage

    workers = max(1, int(workers))
    if workers == 1:
        for pos, idx in enumerate(attempt_indices):
            p, resp_obj, usage = _repair_one(pos, idx)
            response_rows[p] = resp_obj
            token_counter.add(usage)
            completed += 1
            if progress is not None:
                progress.update(model, stage="repair", done=completed, total=len(attempt_indices))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_pos = {executor.submit(_repair_one, pos, idx): pos for pos, idx in enumerate(attempt_indices)}
            for fut in concurrent.futures.as_completed(list(future_to_pos.keys())):
                pos = int(future_to_pos.get(fut, 0))
                try:
                    p, resp_obj, usage = fut.result()
                except Exception as exc:
                    resp_obj = {"status": "exception", "error": f"{type(exc).__name__}: {exc}", "calls": []}
                    usage = None
                    p = pos
                response_rows[p] = resp_obj
                token_counter.add(usage)
                completed += 1
                if progress is not None:
                    progress.update(model, stage="repair", done=completed, total=len(attempt_indices))

    elapsed = time.perf_counter() - start

    with responses_path.open("w", encoding="utf-8") as fp:
        for row in response_rows:
            fp.write(json.dumps(row or {"status": "missing"}, ensure_ascii=False) + "\n")
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for obj in output_records:
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    pred_dir = run_dir / "results"
    pass_rates_path = run_dir / "pass_rates.json"
    metrics_txt = run_dir / "metrics.txt"
    _convert_to_predictions(optimized_jsonl=out_jsonl, out_dir=pred_dir)
    pass_rates = _compute_pass_rates(predictions_dir=pred_dir, ground_truth_dir=gt_dir, json_out=pass_rates_path)
    if all_metrics:
        _compute_all_metrics(predictions_dir=pred_dir, out_path=metrics_txt)
    if progress is not None:
        progress.update(model, stage="done", done=len(attempt_indices), total=len(attempt_indices))

    _write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "base": {
                "pass_rates_path": str(base_pass_rates_path),
                "metrics_txt": str(base_metrics_txt) if all_metrics else "",
                "local_pass_rate": base_pass_rates.get("Local Pass Rate"),
                "global_pass_rate": base_pass_rates.get("Global Pass Rate"),
                "accuracy": base_pass_rates.get("WorkBench Accuracy"),
                "unwanted_side_effects": base_pass_rates.get("WorkBench Unwanted Side Effects"),
            },
            "tree_text_repair": {
                "optimized_jsonl": str(out_jsonl),
                "responses_jsonl": str(responses_path),
                "pass_rates_path": str(pass_rates_path),
                "metrics_txt": str(metrics_txt) if all_metrics else "",
                "elapsed_s": round(elapsed, 6),
                "attempted": int(len(attempt_indices)),
                "token_usage": {
                    "calls": token_counter.calls,
                    "prompt_cache_hit": token_counter.prompt_cache_hit_tokens,
                    "prompt_cache_miss": token_counter.prompt_cache_miss_tokens,
                    "output": token_counter.completion_tokens,
                    "total": token_counter.total_tokens,
                },
                "local_pass_rate": pass_rates.get("Local Pass Rate"),
                "global_pass_rate": pass_rates.get("Global Pass Rate"),
                "accuracy": pass_rates.get("WorkBench Accuracy"),
                "unwanted_side_effects": pass_rates.get("WorkBench Unwanted Side Effects"),
            },
        },
    )

    lines = [
        f"run_start_utc={stamp}",
        f"split={split}",
        f"optimized={optimized}",
        f"tree={tree}",
        f"ground_truth_dir={gt_dir}",
        f"model={model}",
        f"workers={workers}",
        f"total={total}",
        f"to_repair={len(attempt_indices)}",
        f"elapsed_s={elapsed:.6f}",
    ]
    if token_counter.calls:
        lines.append(
            "LLM token usage (tree_text_repair): "
            f"calls={token_counter.calls} "
            f"prompt_cache_hit={token_counter.prompt_cache_hit_tokens} "
            f"prompt_cache_miss={token_counter.prompt_cache_miss_tokens} "
            f"output={token_counter.completion_tokens} "
            f"total={token_counter.total_tokens}"
        )
    (run_dir / "cost.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    split = str(args.split or "validation")
    gt_dir = _ensure_abs(Path(args.ground_truth_dir))
    if not gt_dir.exists():
        raise SystemExit(f"Missing WorkBench ground truth dir: {gt_dir}")
    out_root = _ensure_abs(Path(args.out_root))
    stamp = _utc_stamp()

    if args.full:
        models = [m for m in (args.models or []) if str(m).strip()]
        if not models:
            models = ["gpt-5-mini", "gpt-5-nano", "gpt-5.2"]
        group_name = str(args.run_name).strip() or f"tree_text_repair_full_{split}_{stamp}"
        group_dir = out_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        progress = _MultiModelProgress(
            models,
            enabled=(not bool(args.no_progress)) and bool(getattr(sys.stdout, "isatty", lambda: False)()),
        )
        for m in models:
            progress.update(str(m), stage="pending", done=0, total=0)

        per_model: Dict[str, object] = {}

        for m in models:
            model_name = str(m).strip()
            sub = f"{_model_slug(model_name)}_{split}"
            run_dir = _run_one(
                split=split,
                base_model=model_name,
                model=model_name,
                optimized_path=None,
                tree_path=None,
                workers=max(1, int(args.workers)),
                max_failed=int(args.max_failed),
                max_slot_candidates=int(args.max_slot_candidates),
                max_result_chars=int(args.max_result_chars),
                include_constraint_descriptions=bool(args.include_constraint_descriptions),
                no_llm=bool(args.no_llm),
                enforce_tree_calls=bool(args.enforce_tree_calls),
                output_json_list=bool(args.output_json_list),
                all_metrics=bool(args.all_metrics),
                gt_dir=gt_dir,
                out_root=group_dir,
                run_name=sub,
                progress=progress,
            )
            per_model[model_name] = {"run_dir": str(run_dir), "summary": str(run_dir / "summary.json")}

        _write_json(group_dir / "group_summary.json", {"split": split, "models": models, "runs": per_model})
        print(f"Wrote grouped run to {group_dir}")
        print(f"Group summary: {group_dir / 'group_summary.json'}")
        return

    base_model = str(args.base_model or args.model)
    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"tree_text_repair_{_model_slug(args.model)}_{split}_{stamp}"
    run_dir = _run_one(
        split=split,
        base_model=base_model,
        model=str(args.model),
        optimized_path=args.optimized,
        tree_path=args.tree,
        workers=max(1, int(args.workers)),
        max_failed=int(args.max_failed),
        max_slot_candidates=int(args.max_slot_candidates),
        max_result_chars=int(args.max_result_chars),
        include_constraint_descriptions=bool(args.include_constraint_descriptions),
        no_llm=bool(args.no_llm),
        enforce_tree_calls=bool(args.enforce_tree_calls),
        output_json_list=bool(args.output_json_list),
        all_metrics=bool(args.all_metrics),
        gt_dir=gt_dir,
        out_root=out_root,
        run_name=run_name,
    )
    print(f"Wrote run to {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

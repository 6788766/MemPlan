from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from planner import llm_repair as llmr


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


def _canonical_query(query: str) -> str:
    return str(query or "").strip()


def _ensure_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    yield from llmr._iter_jsonl(path)


def _actions_to_calls(record: Dict[str, object]) -> List[str]:
    actions = record.get("actions") if isinstance(record.get("actions"), list) else []
    calls: List[str] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        calls.append(_action_to_call(a))
    return calls


def _action_to_call(action: Dict[str, object]) -> str:
    attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
    raw = attrs.get("raw_action")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    # Fallback: match task_helper/work/evaluation/convert_optimized_to_predictions.py behavior.
    qualified = attrs.get("qualified_tool_name")
    if not isinstance(qualified, str) or not qualified.strip():
        tool_name = attrs.get("tool_name")
        action_type = action.get("action_type")
        if isinstance(tool_name, str) and isinstance(action_type, str) and tool_name and action_type:
            qualified = f"{tool_name}.{action_type}"
        else:
            qualified = ""
    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    parts = []
    for key, value in params.items():
        if value is None:
            continue
        text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{key}="{text}"')
    return f"{qualified}.func(" + ", ".join(parts) + ")"


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


def _build_prompt(*, query: str, current_calls: Sequence[str]) -> str:
    # Baseline C: do NOT provide tool schemas, candidate IDs, or any constraint mappings.
    calls_block = "\n".join(f"- {c}" for c in current_calls if str(c).strip())
    if not calls_block:
        calls_block = "(none)"
    return (
        "You are solving a WorkBench task.\n"
        "\n"
        "Given the user query and the current (possibly wrong) plan calls, output the best function call list.\n"
        "\n"
        "Rules:\n"
        "- Output ONLY a Python list literal of strings.\n"
        "- Each string must be a single function call in the form: tool.action.func(k=\"v\", ...)\n"
        "- Do NOT output any other text.\n"
        "\n"
        f"QUERY:\n{query}\n"
        "\n"
        "CURRENT_CALLS:\n"
        f"{calls_block}\n"
    )


@dataclass(frozen=True)
class ParsedCalls:
    calls: List[str]
    error: str


def _parse_calls(content: str) -> ParsedCalls:
    text = str(content or "").strip()
    if not text:
        return ParsedCalls(calls=[], error="empty_response")

    # Try JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return ParsedCalls(calls=[str(x) for x in obj if str(x).strip()], error="")
        if isinstance(obj, dict):
            raw = obj.get("calls") or obj.get("function_calls") or obj.get("prediction")
            if isinstance(raw, list):
                return ParsedCalls(calls=[str(x) for x in raw if str(x).strip()], error="")
    except Exception:
        pass

    # Try Python list literal.
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return ParsedCalls(calls=[str(x) for x in obj if str(x).strip()], error="")
    except Exception:
        pass

    # Try extracting a list block like: [...], then literal-eval.
    match = _LIST_BLOCK_RE.search(text)
    if match:
        block = match.group(0)
        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, list):
                return ParsedCalls(calls=[str(x) for x in obj if str(x).strip()], error="wrapped_text")
        except Exception:
            return ParsedCalls(calls=[], error="invalid_list_literal")

    # Fallback: treat as lines.
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`").strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[\).\s]+", "", line)
        line = line.rstrip(",")
        if line:
            lines.append(line)
    calls = [l for l in lines if ".func(" in l]
    if calls:
        return ParsedCalls(calls=calls, error="line_fallback")
    return ParsedCalls(calls=[], error="unparseable")


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorkBench baseline C: unstructured LLM rewrite (no tool schemas / IDs).")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split used in the base run dir naming.")
    parser.add_argument("--base-model", type=str, default="", help="Model used to produce the base optimized JSONL (for default path).")
    parser.add_argument("--optimized", type=Path, default=None, help="Path to optimized_<split>.jsonl (defaults from --base-model/--split).")
    parser.add_argument("--model", type=str, default=llmr.DEFAULT_MODEL, help="LLM model name for unstructured rewrite.")
    parser.add_argument("--workers", type=int, default=14, help="Parallel rewrite workers.")
    parser.add_argument("--max-examples", type=int, default=0, help="If >0, only run on first N examples.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (copy base calls; useful for wiring).")
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Also run task_helper/work/evaluation/calculate_all_metrics.py (slow).",
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GT_DIR, help="WorkBench ground truth directory.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root directory.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run folder name override.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    split = str(args.split or "validation")
    base_model = str(args.base_model or args.model)
    optimized_path = args.optimized
    if optimized_path is None:
        base_dir = ARTIFACTS_ROOT / "output" / "work" / f"{_model_slug(base_model)}_{split}"
        optimized_path = base_dir / f"optimized_{split}.jsonl"
    optimized_path = _ensure_abs(Path(optimized_path))
    if not optimized_path.exists():
        raise SystemExit(f"Missing optimized JSONL: {optimized_path}")

    gt_dir = _ensure_abs(Path(args.ground_truth_dir))
    if not gt_dir.exists():
        raise SystemExit(f"Missing WorkBench ground truth dir: {gt_dir}")

    out_root = _ensure_abs(Path(args.out_root))
    stamp = _utc_stamp()
    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"unstructured_no_schema_{_model_slug(args.model)}_{split}_{stamp}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "run_meta.json"
    _write_json(
        meta_path,
        {
            "ts_utc": stamp,
            "split": split,
            "base_model": base_model,
            "optimized": str(optimized_path),
            "llm_model": str(args.model),
            "workers": int(args.workers),
            "max_examples": int(args.max_examples),
            "no_llm": bool(args.no_llm),
            "ground_truth_dir": str(gt_dir),
        },
    )

    # --- Unstructured rewrite ---
    records = list(_iter_jsonl(optimized_path))
    if args.max_examples and int(args.max_examples) > 0:
        records = records[: int(args.max_examples)]

    base_eval_input = optimized_path
    if args.max_examples and int(args.max_examples) > 0:
        base_eval_input = run_dir / "base_subset_optimized.jsonl"
        with base_eval_input.open("w", encoding="utf-8") as fp:
            for obj in records:
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # --- Base metrics (for apples-to-apples comparison) ---
    base_pred_dir = run_dir / "base_results"
    base_pass_rates_path = run_dir / "base_pass_rates.json"
    base_metrics_txt = run_dir / "base_metrics.txt"
    _convert_to_predictions(optimized_jsonl=base_eval_input, out_dir=base_pred_dir)
    base_pass_rates = _compute_pass_rates(
        predictions_dir=base_pred_dir, ground_truth_dir=gt_dir, json_out=base_pass_rates_path
    )
    if bool(args.all_metrics):
        _compute_all_metrics(predictions_dir=base_pred_dir, out_path=base_metrics_txt)

    token_counter = llmr.TokenCounter()
    responses_path = run_dir / "responses.jsonl"
    repaired_jsonl = run_dir / "unstructured_optimized.jsonl"

    start = time.perf_counter()
    attempted = len(records)

    def _rewrite_one(
        idx: int, item: Dict[str, object]
    ) -> Tuple[int, Dict[str, object], Dict[str, object], Optional[Dict[str, int]]]:
        template_id = str(item.get("template_id") or "")
        plan = item.get("plan") if isinstance(item.get("plan"), dict) else {}
        query = _canonical_query(str(plan.get("query") or ""))
        current_calls = _actions_to_calls(item)
        usage: Optional[Dict[str, int]] = None

        resp_obj: Dict[str, object] = {"template_id": template_id, "query": query}
        if args.no_llm:
            parsed = ParsedCalls(calls=list(current_calls), error="no_llm")
            resp_obj.update({"status": "no_llm", "error": parsed.error, "calls": parsed.calls})
        else:
            prompt = _build_prompt(query=query, current_calls=current_calls)
            messages = [
                {"role": "system", "content": "Output only a Python list of strings."},
                {"role": "user", "content": prompt},
            ]
            try:
                content, usage = llmr._call_openai_chat(model=str(args.model), messages=messages)
            except Exception as exc:
                max_ctx, requested = llmr._extract_context_limit(exc)
                if max_ctx is None or requested is None:
                    resp_obj.update(
                        {
                            "status": "llm_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "calls": list(current_calls),
                        }
                    )
                    content = ""
                    parsed = ParsedCalls(calls=list(current_calls), error="llm_error")
                    out = dict(item)
                    notes = out.get("notes")
                    if not isinstance(notes, dict):
                        notes = {}
                        out["notes"] = notes
                    notes["unstructured_repair"] = {
                        "mode": "no_schema",
                        "model": str(args.model),
                        "status": str(resp_obj.get("status") or ""),
                        "parse_error": str(resp_obj.get("error") or ""),
                        "usage": dict(usage) if usage is not None else None,
                    }
                    out["actions"] = _calls_to_actions(parsed.calls)
                    return idx, out, resp_obj, usage
                truncated = llmr._truncate_messages_to_context_limit(
                    messages=messages,
                    model=str(args.model),
                    max_context_tokens=max_ctx,
                    margin_tokens=500,
                )
                content, usage = llmr._call_openai_chat(model=str(args.model), messages=truncated)

            parsed = _parse_calls(content)
            resp_obj.update(
                {
                    "status": "ok" if not parsed.error else "parsed_with_warnings",
                    "error": parsed.error,
                    "raw": str(content),
                    "calls": parsed.calls,
                }
            )

        out = dict(item)
        notes = out.get("notes")
        if not isinstance(notes, dict):
            notes = {}
            out["notes"] = notes
        notes["unstructured_repair"] = {
            "mode": "no_schema",
            "model": str(args.model),
            "status": str(resp_obj.get("status") or ""),
            "parse_error": str(resp_obj.get("error") or ""),
            "usage": dict(usage) if usage is not None else None,
        }
        out["actions"] = _calls_to_actions(resp_obj.get("calls") if isinstance(resp_obj.get("calls"), list) else [])
        return idx, out, resp_obj, usage

    rewritten_records: List[Optional[Dict[str, object]]] = [None] * len(records)
    response_rows: List[Optional[Dict[str, object]]] = [None] * len(records)

    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, item in enumerate(records):
            i, rewritten, resp_obj, usage = _rewrite_one(idx, item)
            rewritten_records[i] = rewritten
            response_rows[i] = resp_obj
            token_counter.add(usage)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(_rewrite_one, idx, item): idx for idx, item in enumerate(records)}
            for fut in concurrent.futures.as_completed(list(future_to_idx.keys())):
                idx = int(future_to_idx.get(fut, 0))
                try:
                    i, rewritten, resp_obj, usage = fut.result()
                except Exception as exc:
                    # Best-effort: keep the base record with empty calls.
                    i = idx
                    rewritten = dict(records[idx])
                    rewritten["actions"] = []
                    resp_obj = {
                        "template_id": str(records[idx].get("template_id") or ""),
                        "query": _canonical_query(str(((records[idx].get("plan") or {}) if isinstance(records[idx].get("plan"), dict) else {}).get("query") or "")),
                        "status": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "calls": [],
                    }
                    usage = None
                rewritten_records[i] = rewritten
                response_rows[i] = resp_obj
                token_counter.add(usage)

    with responses_path.open("w", encoding="utf-8") as resp_fp, repaired_jsonl.open("w", encoding="utf-8") as out_fp:
        for resp_obj in response_rows:
            if resp_obj is None:
                resp_obj = {"status": "missing"}
            resp_fp.write(json.dumps(resp_obj, ensure_ascii=False) + "\n")
        for record in rewritten_records:
            if record is None:
                record = {}
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start

    # --- Evaluation on rewritten outputs ---
    pred_dir = run_dir / "results"
    pass_rates_path = run_dir / "pass_rates.json"
    metrics_txt = run_dir / "metrics.txt"
    _convert_to_predictions(optimized_jsonl=repaired_jsonl, out_dir=pred_dir)
    pass_rates = _compute_pass_rates(predictions_dir=pred_dir, ground_truth_dir=gt_dir, json_out=pass_rates_path)
    if bool(args.all_metrics):
        _compute_all_metrics(predictions_dir=pred_dir, out_path=metrics_txt)

    summary_path = run_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_dir": str(run_dir),
            "base": {
                "pass_rates_path": str(base_pass_rates_path),
                "metrics_txt": str(base_metrics_txt) if bool(args.all_metrics) else "",
                "local_pass_rate": base_pass_rates.get("Local Pass Rate"),
                "global_pass_rate": base_pass_rates.get("Global Pass Rate"),
                "accuracy": base_pass_rates.get("WorkBench Accuracy"),
                "unwanted_side_effects": base_pass_rates.get("WorkBench Unwanted Side Effects"),
            },
            "unstructured_no_schema": {
                "optimized_jsonl": str(repaired_jsonl),
                "responses_jsonl": str(responses_path),
                "pass_rates_path": str(pass_rates_path),
                "metrics_txt": str(metrics_txt) if bool(args.all_metrics) else "",
                "elapsed_s": round(elapsed, 6),
                "attempted": int(attempted),
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

    cost_path = run_dir / "cost.txt"
    lines = [
        f"run_start_utc={stamp}",
        f"split={split}",
        f"optimized={optimized_path}",
        f"ground_truth_dir={gt_dir}",
        f"model={args.model}",
        f"workers={int(args.workers)}",
        f"attempted={attempted}",
        f"elapsed_s={elapsed:.6f}",
    ]
    if token_counter.calls:
        lines.append(
            "LLM token usage (unstructured_no_schema): "
            f"calls={token_counter.calls} "
            f"prompt_cache_hit={token_counter.prompt_cache_hit_tokens} "
            f"prompt_cache_miss={token_counter.prompt_cache_miss_tokens} "
            f"output={token_counter.completion_tokens} "
            f"total={token_counter.total_tokens}"
        )
    cost_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote run to {run_dir}")
    print(f"Summary: {summary_path}")
    if token_counter.calls:
        print(lines[-1])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from baseline.convert_baseline_to_submission import _infer_days_from_query, _placeholder_plan, parse_plan
from baseline.rag.common import (
    LexicalRetriever,
    RetrievalDocument,
    TokenUsage,
    aggregate_token_usage,
    compute_price_usd,
    count_text_tokens,
    iso_now,
    model_slug,
    render_retrieval_context,
    resolve_price_key,
    sum_mapping_values,
    write_jsonl,
)
from baseline.utils.dataset import load_travelplanner_dataset
from baseline.tool_agents import (  # type: ignore
    PlannerTool,
    ReactAgent,
    TokenCounter,
    ToolCallCounter,
    _encoding_for_model,
    _load_tool_costs,
    _merge_tool_usage,
    _resolve_workers,
    tqdm,
    truncate_scratchpad,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

TRAVEL_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "input" / "travel" / "dataset" / "train.csv"
TRAVEL_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "travel" / "rag"
TOOLS_LIST = [
    "notebook",
    "flights",
    "attractions",
    "accommodations",
    "restaurants",
    "googleDistanceMatrix",
    "planner",
    "cities",
]
EVAL_METRIC_NAMES = [
    "Delivery Rate",
    "Commonsense Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Final Pass Rate",
    "All Budget Pass Rate",
    "Local Pass Rate",
    "Global Pass Rate",
]


def _seconds_now() -> float:
    return time.perf_counter()


def _run_subprocess(*, cmd: Sequence[str], stage_name: str, results_txt_path: Path) -> Tuple[float, str]:
    start = _seconds_now()
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if combined:
        with results_txt_path.open("a", encoding="utf-8") as fp:
            fp.write(combined)
            if not combined.endswith("\n"):
                fp.write("\n")
        print(combined, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {proc.returncode}")
    return _seconds_now() - start, combined


def _append_error_log(results_txt_path: Path, message: str) -> None:
    text = str(message).rstrip() + "\n"
    with results_txt_path.open("a", encoding="utf-8") as fp:
        fp.write(text)
    print(text, end="", file=sys.stderr, flush=True)


def _render_constraints(raw_constraints: object) -> str:
    if not isinstance(raw_constraints, dict):
        return "None"
    items: List[str] = []
    for key, value in raw_constraints.items():
        if value in (None, "", [], {}, "None"):
            continue
        items.append(f"{key}={value}")
    return ", ".join(items) if items else "None"


def _render_day_records(day_records: object) -> str:
    if not isinstance(day_records, list):
        return ""
    lines: List[str] = []
    for raw_day in day_records:
        if not isinstance(raw_day, dict):
            continue
        day_no = raw_day.get("days", "-")
        lines.append(f"Day {day_no}:")
        lines.append(f"Current City: {raw_day.get('current_city', '-')}")
        lines.append(f"Transportation: {raw_day.get('transportation', '-')}")
        lines.append(f"Breakfast: {raw_day.get('breakfast', '-')}")
        lines.append(f"Attraction: {raw_day.get('attraction', '-')}")
        lines.append(f"Lunch: {raw_day.get('lunch', '-')}")
        lines.append(f"Dinner: {raw_day.get('dinner', '-')}")
        lines.append(f"Accommodation: {raw_day.get('accommodation', '-')}")
    return "\n".join(lines).strip()


def _build_travel_corpus(train_path: Path) -> List[RetrievalDocument]:
    import ast

    rows: List[RetrievalDocument] = []
    with train_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for index, row in enumerate(reader, start=1):
            annotated = row.get("annotated_plan", "")
            try:
                parsed = ast.literal_eval(annotated)
            except Exception:
                continue
            if not isinstance(parsed, list) or len(parsed) < 2:
                continue
            meta = parsed[0] if isinstance(parsed[0], dict) else {}
            day_records = parsed[1]
            trip_summary = (
                f"from {meta.get('org', row.get('org', '-'))} to {meta.get('dest', row.get('dest', '-'))}; "
                f"days={meta.get('days', row.get('days', '-'))}; "
                f"people={meta.get('people_number', row.get('people_number', '-'))}; "
                f"budget={meta.get('budget', row.get('budget', '-'))}"
            )
            constraints = _render_constraints(meta.get("local_constraint"))
            plan_text = _render_day_records(day_records)
            rendered = (
                f"Source Query: {row.get('query', '')}\n"
                f"Trip Summary: {trip_summary}\n"
                f"Constraints: {constraints}\n"
                f"Reference Plan:\n{plan_text}"
            ).strip()
            search_text = "\n".join(
                [
                    str(row.get("query", "")),
                    trip_summary,
                    constraints,
                    plan_text,
                ]
            )
            rows.append(
                RetrievalDocument(
                    doc_id=f"travel_train_{index:05d}",
                    query=str(row.get("query", "")),
                    search_text=search_text,
                    rendered_text=rendered,
                    metadata={
                        "org": row.get("org", ""),
                        "dest": row.get("dest", ""),
                        "days": row.get("days", ""),
                        "people_number": row.get("people_number", ""),
                        "budget": row.get("budget", ""),
                    },
                )
            )
    return rows


class TimedPlannerTool:
    def __init__(self, model_name: str, *, token_counter: Optional[TokenCounter], timer_state: Dict[str, float]) -> None:
        self._tool = PlannerTool(model_name, token_counter=token_counter)
        self._timer_state = timer_state

    def run(self, text: str, query: str, log_file=None) -> str:
        start = _seconds_now()
        try:
            return self._tool.run(text, query, log_file=log_file)
        finally:
            self._timer_state["planner_elapsed_s"] = self._timer_state.get("planner_elapsed_s", 0.0) + (
                _seconds_now() - start
            )


class RagReactAgent(ReactAgent):
    def __init__(
        self,
        *,
        rag_context: str,
        planner_token_counter: TokenCounter,
        planner_timer_state: Dict[str, float],
        **kwargs,
    ) -> None:
        self._rag_context = rag_context.strip()
        self._planner_token_counter = planner_token_counter
        self._planner_timer_state = planner_timer_state
        super().__init__(**kwargs)

    def _rag_query_block(self) -> str:
        if not self._rag_context:
            return self.query
        return (
            f"{self.query}\n\n"
            "Retrieved training examples (reference only; use them as high-level hints and still verify concrete "
            "details through tools):\n"
            f"{self._rag_context}"
        )

    def _build_agent_prompt(self, *, expect_action: bool = False) -> str:
        if self.mode != "zero_shot":
            return super()._build_agent_prompt(expect_action=expect_action)

        scratchpad = self.scratchpad
        query_block = self._rag_query_block()
        prompt = self.agent_prompt.format(query=query_block, scratchpad=scratchpad)
        prompt_budget = min(int(self.max_token_length), 8000)
        try:
            if len(self.enc.encode(prompt)) > prompt_budget:
                scratchpad = truncate_scratchpad(scratchpad, n_tokens=1600, tokenizer=self.enc)
                prompt = self.agent_prompt.format(query=query_block, scratchpad=scratchpad)
        except Exception:
            pass

        if expect_action:
            prompt += "\n\nIMPORTANT: Return ONLY the next Action as ToolName[args] with no other text."
        else:
            prompt += "\n\nIMPORTANT: Return ONLY the next Thought in one short sentence (no tool call)."
        return prompt

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, object]:
        tools_map: Dict[str, object] = {}
        for tool_name in tools:
            if tool_name == "planner":
                tools_map[tool_name] = TimedPlannerTool(
                    planner_model_name or self.planner_name,
                    token_counter=self._planner_token_counter,
                    timer_state=self._planner_timer_state,
                )
                continue
            module = __import__(f"task_helper.travel.tools.{tool_name}.apis", fromlist=["dummy"])
            cls_name = tool_name[0].upper() + tool_name[1:]
            tools_map[tool_name] = getattr(module, cls_name)()
        return tools_map


def _convert_results_to_submission(results_jsonl_path: Path, submission_path: Path) -> None:
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, object]] = []
    with results_jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = int(record["idx"])
            query = str(record["query"])
            result = str(record.get("result") or "")
            plan = parse_plan(result)
            if not plan:
                inferred = _infer_days_from_query(query)
                if inferred is not None:
                    plan = _placeholder_plan(inferred)
            items.append({"idx": idx, "query": query, "plan": plan})
    with submission_path.open("w", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def _parse_travel_eval_metrics(results_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name in EVAL_METRIC_NAMES:
        marker = f"{name}:"
        index = results_text.find(marker)
        if index < 0:
            continue
        tail = results_text[index + len(marker) :].splitlines()[0].strip()
        value = tail.rstrip("%").strip()
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def _run_single_query(
    *,
    idx: int,
    query: str,
    model_name: str,
    retriever: LexicalRetriever,
    top_k: int,
    max_context_tokens: int,
    tool_costs: Mapping[str, float],
    per_query_dir: Path,
) -> Dict[str, object]:
    retrieval_start = _seconds_now()
    hits = retriever.search(query, top_k=top_k, exclude_exact_query=True)
    rag_context, kept_hits, context_tokens = render_retrieval_context(
        hits,
        max_context_tokens=max_context_tokens,
        intro=(
            "Retrieved training examples. Use them only as structure hints. Concrete entities and logistics still "
            "must come from tool observations."
        ),
    )
    retrieval_elapsed = _seconds_now() - retrieval_start

    react_counter = TokenCounter()
    planner_counter = TokenCounter()
    tool_counter = ToolCallCounter(tool_costs=dict(tool_costs))
    planner_timer_state: Dict[str, float] = {"planner_elapsed_s": 0.0}

    max_steps = 50 if ("deepseek" in model_name.lower() or "gpt-5.2" in model_name.lower()) else 30
    agent = RagReactAgent(
        rag_context=rag_context,
        planner_token_counter=planner_counter,
        planner_timer_state=planner_timer_state,
        args=None,
        tools=TOOLS_LIST,
        max_steps=max_steps,
        react_llm_name=model_name,
        planner_llm_name=model_name,
        token_counter=react_counter,
        tool_counter=tool_counter,
    )

    inference_start = _seconds_now()
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    captured_logs = ""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            planner_results, scratchpad, action_log = agent.run(query)
        except Exception as exc:  # pragma: no cover - network/tool dependent
            planner_results, scratchpad, action_log = "", "", []
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
    captured_logs = buffer.getvalue()
    inference_elapsed = _seconds_now() - inference_start

    react_usage = TokenUsage.from_counter(react_counter)
    planner_usage = TokenUsage.from_counter(planner_counter)
    total_usage = aggregate_token_usage([react_usage, planner_usage])
    planner_elapsed = float(planner_timer_state.get("planner_elapsed_s", 0.0))
    react_elapsed = max(0.0, inference_elapsed - planner_elapsed)

    q_record = {
        "idx": idx,
        "query": query,
        "result": planner_results,
        "scratchpad": scratchpad,
        "action_log": action_log,
        "error": error,
        "retrieval": {
            "top_k_requested": int(top_k),
            "top_k_used": len(kept_hits),
            "elapsed_s": retrieval_elapsed,
            "context_tokens": int(context_tokens),
            "context_text": rag_context,
            "hits": [hit.to_dict() for hit in kept_hits],
        },
        "metrics": {
            "time_s": inference_elapsed + retrieval_elapsed,
            "llm_calls": total_usage.calls,
            "prompt_tokens": total_usage.prompt_tokens,
            "prompt_cache_hit_tokens": total_usage.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": total_usage.prompt_cache_miss_tokens,
            "completion_tokens": total_usage.completion_tokens,
            "output_tokens": total_usage.completion_tokens,
            "total_tokens": total_usage.total_tokens,
            "approx_llm_calls": total_usage.approx_calls,
            "tool_calls": tool_counter.total_calls,
            "tool_cost": tool_counter.total_cost,
            "tool_by_tool": tool_counter.by_tool,
            "time_s_by_stage": {
                "retrieval": retrieval_elapsed,
                "react_agent": react_elapsed,
                "planner": planner_elapsed,
            },
            "token_usage_by_stage": {
                "retrieval": TokenUsage().to_dict(),
                "react_agent": react_usage.to_dict(),
                "planner": planner_usage.to_dict(),
            },
        },
    }
    if error_traceback:
        q_record["error_traceback"] = error_traceback
    if error and captured_logs.strip():
        q_record["captured_logs"] = captured_logs

    per_query_path = per_query_dir / f"generated_plan_{idx}.json"
    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    per_query_path.write_text(json.dumps(q_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "idx": idx,
        "per_query_path": str(per_query_path),
        "record": q_record,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG baseline for the Travel benchmark.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-tokens", type=int, default=1200)
    parser.add_argument("--output-root", type=Path, default=TRAVEL_OUTPUT_ROOT)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    run_dir = args.output_root.expanduser().resolve() / f"{model_slug(args.model)}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_txt_path = run_dir / "results.txt"
    results_txt_path.write_text("", encoding="utf-8")

    run_prefix = f"two_stage_{model_slug(args.model)}_{args.split}"
    results_jsonl_path = run_dir / f"{run_prefix}.jsonl"
    per_query_dir = run_dir / run_prefix
    per_query_dir.mkdir(parents=True, exist_ok=True)

    cost_txt_path = run_dir / "cost.txt"
    run_metrics_path = run_dir / "run_metrics.json"
    per_query_metrics_path = run_dir / "per_query_metrics.jsonl"
    retrieval_hits_path = run_dir / f"retrieval_hits_{args.split}.jsonl"
    corpus_path = run_dir / "retrieval_corpus_train.jsonl"
    submission_path = run_dir / f"submission_{args.split}.jsonl"

    query_rows = load_travelplanner_dataset(args.split)
    if args.limit > 0:
        query_rows = query_rows[: args.limit]

    corpus_docs = _build_travel_corpus(TRAVEL_TRAIN_PATH)
    retriever = LexicalRetriever(corpus_docs)
    write_jsonl(
        corpus_path,
        [
            {
                "doc_id": doc.doc_id,
                "query": doc.query,
                "rendered_text": doc.rendered_text,
                "metadata": doc.metadata,
            }
            for doc in corpus_docs
        ],
    )

    workers = _resolve_workers(args.workers)
    tool_costs = _load_tool_costs()
    run_start_iso = iso_now()
    pipeline_start = _seconds_now()

    summaries: Dict[int, Dict[str, object]] = {}
    processed_indices: List[int] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_meta: Dict[object, Dict[str, object]] = {}
        for idx, row in enumerate(query_rows, start=1):
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            processed_indices.append(idx)
            future = pool.submit(
                _run_single_query,
                idx=idx,
                query=query,
                model_name=args.model,
                retriever=retriever,
                top_k=args.top_k,
                max_context_tokens=args.max_context_tokens,
                tool_costs=tool_costs,
                per_query_dir=per_query_dir,
            )
            future_to_meta[future] = {"idx": idx, "query": query}

        for future in tqdm(as_completed(future_to_meta), total=len(future_to_meta), desc="rag_travel"):
            meta = future_to_meta[future]
            try:
                summary = future.result()
            except Exception:
                _append_error_log(
                    results_txt_path,
                    (
                        f"[rag][travel][fatal] idx={meta['idx']} query={meta['query']}\n"
                        f"{traceback.format_exc()}"
                    ),
                )
                raise
            summaries[int(summary["idx"])] = summary
            record = summary.get("record") if isinstance(summary.get("record"), dict) else {}
            if record.get("error"):
                error_text = (
                    f"[rag][travel][query_error] idx={summary['idx']} "
                    f"query={record.get('query', '')}\n"
                    f"error={record.get('error')}\n"
                    f"per_query_path={summary.get('per_query_path')}"
                )
                if record.get("error_traceback"):
                    error_text += f"\n{record['error_traceback']}"
                if record.get("captured_logs"):
                    captured_lines = [line for line in str(record["captured_logs"]).splitlines() if line.strip()][-20:]
                    if captured_lines:
                        error_text += "\nCaptured logs:\n" + "\n".join(captured_lines)
                _append_error_log(results_txt_path, error_text)

    inference_elapsed = _seconds_now() - pipeline_start

    ordered_records: List[Dict[str, object]] = []
    retrieval_rows: List[Dict[str, object]] = []
    per_query_metric_rows: List[Dict[str, object]] = []
    react_usages: List[TokenUsage] = []
    planner_usages: List[TokenUsage] = []
    total_tool_calls = 0
    total_tool_cost = 0.0
    by_tool: Dict[str, Dict[str, float]] = {}
    retrieval_elapsed_total = 0.0
    react_elapsed_total = 0.0
    planner_elapsed_total = 0.0

    for idx in processed_indices:
        record = summaries[idx]["record"]
        ordered_records.append(record)
        retrieval = record["retrieval"]
        retrieval_rows.append(
            {
                "idx": record["idx"],
                "query": record["query"],
                "context_tokens": retrieval["context_tokens"],
                "elapsed_s": retrieval["elapsed_s"],
                "hits": retrieval["hits"],
                "context_text": retrieval["context_text"],
            }
        )

        metrics = record["metrics"]
        stage_usage = metrics["token_usage_by_stage"]
        react_usage = TokenUsage.from_mapping(stage_usage["react_agent"], calls=stage_usage["react_agent"].get("calls", 0))
        planner_usage = TokenUsage.from_mapping(stage_usage["planner"], calls=stage_usage["planner"].get("calls", 0))
        react_usages.append(react_usage)
        planner_usages.append(planner_usage)

        stage_times = metrics["time_s_by_stage"]
        retrieval_elapsed_total += float(stage_times["retrieval"])
        react_elapsed_total += float(stage_times["react_agent"])
        planner_elapsed_total += float(stage_times["planner"])
        total_tool_calls += int(metrics.get("tool_calls") or 0)
        total_tool_cost += float(metrics.get("tool_cost") or 0.0)
        _merge_tool_usage(by_tool, metrics.get("tool_by_tool") or {})

        per_query_metric_rows.append(
            {
                "idx": record["idx"],
                "query": record["query"],
                "error": record.get("error"),
                "status": "ok" if not record.get("error") else "error",
                "time_s_total": metrics["time_s"],
                "time_s_by_stage": stage_times,
                "token_usage_by_stage": stage_usage,
                "llm_calls": metrics["llm_calls"],
                "prompt_cache_hit_tokens": metrics["prompt_cache_hit_tokens"],
                "prompt_cache_miss_tokens": metrics["prompt_cache_miss_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "total_tokens": metrics["total_tokens"],
                "tool_calls": metrics["tool_calls"],
                "tool_cost": metrics["tool_cost"],
                "retrieval_doc_ids": [hit["doc_id"] for hit in retrieval["hits"]],
                "retrieval_scores": [hit["score"] for hit in retrieval["hits"]],
                "retrieval_time_s": retrieval["elapsed_s"],
                "retrieval_context_tokens": retrieval["context_tokens"],
            }
        )

    with results_jsonl_path.open("w", encoding="utf-8") as fp:
        for record in ordered_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_jsonl(retrieval_hits_path, retrieval_rows)
    write_jsonl(per_query_metrics_path, per_query_metric_rows)

    react_total = aggregate_token_usage(react_usages)
    planner_total = aggregate_token_usage(planner_usages)
    baseline_total = aggregate_token_usage([react_total, planner_total])

    convert_elapsed = 0.0
    eval_elapsed = 0.0
    eval_metrics: Dict[str, float] = {}

    print("Converting Travel outputs to submission format...")
    convert_start = _seconds_now()
    _convert_results_to_submission(results_jsonl_path, submission_path)
    convert_elapsed = _seconds_now() - convert_start

    if not args.skip_eval:
        print("Running Travel evaluation...")
        if args.split == "test":
            eval_elapsed, _ = _run_subprocess(
                cmd=[
                    sys.executable,
                    "-m",
                    "task_helper.travel.runners.eval_leaderboard",
                    "--split",
                    "test",
                    "--eval-mode",
                    "two-stage",
                    "--submission",
                    str(submission_path),
                ],
                stage_name="eval_leaderboard",
                results_txt_path=results_txt_path,
            )
        else:
            eval_elapsed, eval_output = _run_subprocess(
                cmd=[
                    sys.executable,
                    "-m",
                    "task_helper.travel.runners.eval_bridge",
                    "--set-type",
                    args.split,
                    "--submission",
                    str(submission_path),
                ],
                stage_name="eval_bridge",
                results_txt_path=results_txt_path,
            )
            eval_metrics = _parse_travel_eval_metrics(eval_output)

    pipeline_elapsed_total = _seconds_now() - pipeline_start
    pipeline_elapsed_no_eval = inference_elapsed + convert_elapsed

    price_key = resolve_price_key(args.model)
    react_price = compute_price_usd(model_name=args.model, usage=react_total)
    planner_price = compute_price_usd(model_name=args.model, usage=planner_total)
    total_price = sum_mapping_values([react_price, planner_price])

    cost_lines = [
        f"run_start={run_start_iso}",
        "task=travel",
        "variant=rag",
        f"split={args.split}",
        f"model={args.model}",
        f"model_slug={model_slug(args.model)}",
        f"workers={args.workers} (effective={workers})",
        f"retrieval_top_k={args.top_k}",
        f"retrieval_max_context_tokens={args.max_context_tokens}",
        f"retrieval_corpus_path={TRAVEL_TRAIN_PATH}",
        f"output_dir={run_dir}",
        f"per_query_dir={per_query_dir}",
        f"results_jsonl={results_jsonl_path}",
        f"queries_total={len(query_rows)}",
        f"queries_processed={len(processed_indices)}",
        f"limit={args.limit}",
        f"run_end={iso_now()}",
        f"elapsed_s_inference={inference_elapsed}",
        f"pipeline_elapsed_s_no_eval={pipeline_elapsed_no_eval}",
        f"eval_elapsed_s={eval_elapsed}",
        f"pipeline_elapsed_s_total={pipeline_elapsed_total}",
        f"retrieval_elapsed_s_total={retrieval_elapsed_total}",
        f"react_elapsed_s_total={react_elapsed_total}",
        f"planner_elapsed_s_total={planner_elapsed_total}",
        f"LLM token usage (retrieval): {TokenUsage().to_cost_line()}",
        f"LLM token usage (react_agent): {react_total.to_cost_line()}",
        f"LLM token usage (planner): {planner_total.to_cost_line()}",
        f"LLM token usage (baseline): {baseline_total.to_cost_line()}",
        (
            "tool_calls: "
            f"calls={total_tool_calls} cost={total_tool_cost} "
            f"by_tool={json.dumps(by_tool, ensure_ascii=False, sort_keys=True)}"
        ),
        f"summary: time_s_total={pipeline_elapsed_total} tokens_total={baseline_total.total_tokens} tool_calls_cost={total_tool_cost}",
        "--------------total cost-----------------",
        f"llm_price_model_key={price_key}",
        f"llm_price_usd_react_agent={react_price:.6f}",
        f"llm_price_usd_planner={planner_price:.6f}",
        f"llm_price_usd_baseline={total_price:.6f}",
        f"llm_price_usd_total={total_price:.6f}",
        f"total_tokens={baseline_total.total_tokens}",
        f"total_time_s={pipeline_elapsed_total}",
    ]
    cost_txt_path.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    run_metrics = {
        "metadata": {
            "task": "travel",
            "variant": "rag",
            "split": args.split,
            "model": args.model,
            "model_slug": model_slug(args.model),
            "workers_requested": int(args.workers),
            "workers_effective": int(workers),
            "top_k": int(args.top_k),
            "max_context_tokens": int(args.max_context_tokens),
            "queries_total": len(query_rows),
            "queries_processed": len(processed_indices),
            "outputs_dir": str(run_dir),
            "results_jsonl": str(results_jsonl_path),
            "submission_path": str(submission_path),
            "retrieval_corpus_source": str(TRAVEL_TRAIN_PATH),
            "retrieval_corpus_docs": len(corpus_docs),
        },
        "stages": {
            "retrieval": {
                "elapsed_s": retrieval_elapsed_total,
                "token_usage": TokenUsage().to_dict(),
            },
            "react_agent": {
                "elapsed_s": react_elapsed_total,
                "token_usage": react_total.to_dict(),
                "price_usd": react_price,
            },
            "planner": {
                "elapsed_s": planner_elapsed_total,
                "token_usage": planner_total.to_dict(),
                "price_usd": planner_price,
            },
            "convert": {
                "elapsed_s": convert_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
            "eval": {
                "elapsed_s": eval_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
        },
        "totals": {
            "pipeline_elapsed_s_total": pipeline_elapsed_total,
            "pipeline_elapsed_s_no_eval": pipeline_elapsed_no_eval,
            "eval_elapsed_s": eval_elapsed,
            "tool_calls_total": total_tool_calls,
            "tool_cost_total": total_tool_cost,
            "token_usage": baseline_total.to_dict(),
            "llm_price_model_key": price_key,
            "llm_price_usd_total": total_price,
            "total_tokens": baseline_total.total_tokens,
        },
        "evaluation": eval_metrics,
    }
    run_metrics_path.write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Travel RAG baseline complete: {run_dir}")


if __name__ == "__main__":
    main()

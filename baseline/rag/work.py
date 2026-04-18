from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from baseline.rag.common import (
    LexicalRetriever,
    RetrievalDocument,
    TokenUsage,
    aggregate_token_usage,
    compute_price_usd,
    count_text_tokens,
    iso_now,
    model_slug,
    normalize_domains,
    render_retrieval_context,
    resolve_price_key,
    sum_mapping_values,
    write_jsonl,
)
from baseline.workbench.utils import (  # type: ignore
    AgentType,
    DOMAINS,
    HARDCODED_CURRENT_TIME,
    _WORKBENCH_LANGCHAIN_OK,
    _make_llm,
    _run_agent,
    _tqdm,
    convert_agent_action_to_function_call,
    get_toolkits,
    initialize_agent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_TRAIN_PATH = PROJECT_ROOT / "artifacts" / "input" / "work" / "dataset" / "train.csv"
WORK_QUERY_ROOT = PROJECT_ROOT / "artifacts" / "input" / "work" / "dataset" / "queries_and_answers"
WORK_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "work" / "rag"
WORK_TOOL_ORDER = [
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
    "multi_domain",
]

_WORK_RETRIEVER: Optional[LexicalRetriever] = None
_WORK_TOP_K = 3
_WORK_MAX_CONTEXT_TOKENS = 1200
_WORK_DOMAIN_TO_DOC_IDS: Dict[str, List[str]] = {}


def _seconds_now() -> float:
    return time.perf_counter()


def _resolve_workers(workers: int) -> int:
    if int(workers) == 0:
        return int(os.cpu_count() or 1)
    return max(1, int(workers))


def _run_subprocess(*, cmd: Sequence[str], out_path: Path, header: str) -> float:
    start = _seconds_now()
    proc = subprocess.run(list(cmd), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    combined = (proc.stdout or "") + (proc.stderr or "")
    with out_path.open("a", encoding="utf-8") as fp:
        fp.write(header)
        if not header.endswith("\n"):
            fp.write("\n")
        fp.write(combined)
        if not combined.endswith("\n"):
            fp.write("\n")
    if combined:
        print(combined, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return _seconds_now() - start


def _append_error_log(results_txt_path: Path, message: str) -> None:
    text = str(message).rstrip() + "\n"
    with results_txt_path.open("a", encoding="utf-8") as fp:
        fp.write(text)
    print(text, end="", file=sys.stderr, flush=True)


def _render_action_lines(actions: Sequence[str]) -> str:
    lines: List[str] = []
    for index, action in enumerate(actions, start=1):
        lines.append(f"{index}. {action}")
    return "\n".join(lines)


def _build_work_corpus(train_path: Path) -> Tuple[List[RetrievalDocument], Dict[str, List[str]]]:
    rows = pd.read_csv(train_path, dtype=str).fillna("")
    docs: List[RetrievalDocument] = []
    domain_to_doc_ids: Dict[str, List[str]] = {}

    for index, row in rows.iterrows():
        domains = normalize_domains(row.get("domains", ""))
        teacher_calls = row.get("teacher_function_calls") or row.get("answer") or "[]"
        try:
            actions = ast.literal_eval(teacher_calls)
        except Exception:
            actions = []
        if not isinstance(actions, list):
            actions = []
        rendered = (
            f"Source Query: {row.get('query', '')}\n"
            f"Domains: {', '.join(domains) if domains else 'unknown'}\n"
            f"Base Template: {row.get('base_template', '')}\n"
            "Reference Actions:\n"
            f"{_render_action_lines([str(action) for action in actions])}"
        ).strip()
        search_text = "\n".join(
            [
                str(row.get("query", "")),
                str(row.get("base_template", "")),
                str(row.get("chosen_template", "")),
                " ".join(domains),
                " ".join(str(action) for action in actions),
            ]
        )
        doc_id = f"work_train_{index + 1:05d}"
        docs.append(
            RetrievalDocument(
                doc_id=doc_id,
                query=str(row.get("query", "")),
                search_text=search_text,
                rendered_text=rendered,
                metadata={
                    "plan_id": row.get("plan_id", ""),
                    "domains": domains,
                    "base_template": row.get("base_template", ""),
                },
            )
        )
        for domain in domains:
            domain_to_doc_ids.setdefault(domain, []).append(doc_id)
    return docs, domain_to_doc_ids


def _retriever_candidates(domains: Sequence[str]) -> Optional[List[str]]:
    if not domains or not _WORK_DOMAIN_TO_DOC_IDS:
        return None
    candidate_ids: List[str] = []
    seen = set()
    for domain in domains:
        for doc_id in _WORK_DOMAIN_TO_DOC_IDS.get(domain, []):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            candidate_ids.append(doc_id)
    return candidate_ids or None


def _build_prompt_prefix(rag_context: str) -> str:
    time_prefix = (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} and the current "
        f"time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm."
    )
    if not rag_context:
        return time_prefix
    return (
        f"{time_prefix}\n\n"
        "Retrieved training examples (reference only; adapt them to the current request and only call tools that are "
        "available in this environment):\n"
        f"{rag_context}\n\n"
    )


def _escape_prompt_literal(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _worker_init(retriever: LexicalRetriever, top_k: int, max_context_tokens: int, domain_to_doc_ids: Dict[str, List[str]]) -> None:
    global _WORK_RETRIEVER, _WORK_TOP_K, _WORK_MAX_CONTEXT_TOKENS, _WORK_DOMAIN_TO_DOC_IDS
    _WORK_RETRIEVER = retriever
    _WORK_TOP_K = int(top_k)
    _WORK_MAX_CONTEXT_TOKENS = int(max_context_tokens)
    _WORK_DOMAIN_TO_DOC_IDS = {str(key): list(value) for key, value in domain_to_doc_ids.items()}


def _run_single_query(index: int, query: str, domains_raw: str, model_name: str, tool_selection: str) -> Dict[str, object]:
    if not _WORKBENCH_LANGCHAIN_OK or initialize_agent is None or AgentType is None:
        raise RuntimeError("WorkBench baseline dependencies are not available in the current environment.")
    if _WORK_RETRIEVER is None:
        raise RuntimeError("Worker retriever was not initialized.")

    query_domains = normalize_domains(domains_raw)
    candidate_ids = _retriever_candidates(query_domains)

    retrieval_start = _seconds_now()
    hits = _WORK_RETRIEVER.search(
        query,
        top_k=_WORK_TOP_K,
        candidate_doc_ids=candidate_ids,
        exclude_exact_query=True,
    )
    rag_context, kept_hits, context_tokens = render_retrieval_context(
        hits,
        max_context_tokens=_WORK_MAX_CONTEXT_TOKENS,
        intro=(
            "Retrieved training examples. Use them as action-pattern hints only. Adapt parameters to the current "
            "query and do not copy stale entities."
        ),
    )
    retrieval_elapsed = _seconds_now() - retrieval_start

    toolkits = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
    if tool_selection == "domains" and query_domains:
        toolkits = list(query_domains)

    tools = get_toolkits(toolkits)
    llm = _make_llm(model_name=model_name)
    agent = initialize_agent(
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        max_iterations=20,
        max_execution_time=120,
    )
    base_prompt = agent.agent.llm_chain.prompt.messages[0].prompt.template
    agent.agent.llm_chain.prompt.messages[0].prompt.template = _build_prompt_prefix(
        _escape_prompt_literal(rag_context)
    ) + base_prompt

    response = ""
    error = ""
    error_traceback = ""
    function_calls: List[str] = []
    agent_start = _seconds_now()
    usage_payload = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "total_cost_usd": 0.0,
    }
    try:
        response_obj, usage_payload = _run_agent(agent, query)
        response = str(response_obj)
        for step in response_obj["intermediate_steps"]:
            function_calls.append(convert_agent_action_to_function_call(step[-2]))
        error = response_obj["output"] if response_obj["output"] == "Agent stopped due to iteration limit or time limit." else ""
    except Exception as exc:  # pragma: no cover - network/tool dependent
        response = str(exc)
        error = f"{type(exc).__name__}: {exc}"
        error_traceback = traceback.format_exc()

    agent_elapsed = _seconds_now() - agent_start

    for domain in DOMAINS:
        domain.reset_state()

    usage = TokenUsage.from_mapping(usage_payload, calls=1 if usage_payload.get("total_tokens") else 0)
    return {
        "index": index,
        "query": query,
        "domains": query_domains,
        "function_calls": function_calls,
        "full_response": response,
        "error": error,
        "error_traceback": error_traceback,
        "usage": usage.to_dict(),
        "callback_cost_usd": float(usage_payload.get("total_cost_usd") or 0.0),
        "retrieval": {
            "elapsed_s": retrieval_elapsed,
            "context_tokens": int(context_tokens),
            "context_text": rag_context,
            "hits": [hit.to_dict() for hit in kept_hits],
        },
        "time_s_by_stage": {
            "retrieval": retrieval_elapsed,
            "agent": agent_elapsed,
        },
    }


def _evaluate_predictions(results_dir: Path, results_txt_path: Path) -> Tuple[float, float]:
    overall_elapsed = _run_subprocess(
        cmd=[
            sys.executable,
            "-m",
            "task_helper.work.evaluation.calculate_all_metrics",
            "--predictions_dir",
            str(results_dir),
            "--ground_truth_dir",
            str(WORK_QUERY_ROOT),
        ],
        out_path=results_txt_path,
        header="========== WorkBench official-style metrics ==========",
    )
    pass_rates_elapsed = _run_subprocess(
        cmd=[
            sys.executable,
            "-m",
            "task_helper.work.evaluation.calculate_pass_rates_dir",
            "--predictions_dir",
            str(results_dir),
            "--ground_truth_dir",
            str(WORK_QUERY_ROOT),
            "--json_out",
            str(results_dir / "pass_rates.json"),
        ],
        out_path=results_txt_path,
        header="\n========== Constraint pass rates (local/global) ==========",
    )
    return overall_elapsed, pass_rates_elapsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG baseline for the Work benchmark.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tool-selection", type=str, default="all", choices=["all", "domains"])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=WORK_OUTPUT_ROOT)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not _WORKBENCH_LANGCHAIN_OK or initialize_agent is None or AgentType is None:
        raise RuntimeError("WorkBench baseline dependencies are not available in the current environment.")

    run_dir = args.output_root.expanduser().resolve() / f"{model_slug(args.model)}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_alias_dir = run_dir / "predictions"
    predictions_alias_dir.mkdir(parents=True, exist_ok=True)

    results_txt_path = run_dir / "results.txt"
    results_txt_path.write_text(
        f"==============================\nWorkBench evaluation: {run_dir}\nmodel={args.model} split={args.split} variant=rag\n==============================\n",
        encoding="utf-8",
    )

    cost_txt_path = run_dir / "cost.txt"
    run_metrics_path = run_dir / "run_metrics.json"
    summary_json_path = results_dir / "summary.json"
    per_query_metrics_path = run_dir / "per_query_metrics.jsonl"
    retrieval_hits_path = run_dir / f"retrieval_hits_{args.split}.jsonl"
    corpus_path = run_dir / "retrieval_corpus_train.jsonl"

    run_start_iso = iso_now()
    pipeline_start = _seconds_now()

    corpus_docs, domain_to_doc_ids = _build_work_corpus(WORK_TRAIN_PATH)
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
    retrieval_rows: List[Dict[str, object]] = []
    per_query_metric_rows: List[Dict[str, object]] = []
    agent_usages: List[TokenUsage] = []
    retrieval_elapsed_total = 0.0
    agent_elapsed_total = 0.0
    callback_costs: List[float] = []
    domain_counts: Dict[str, int] = {}

    for domain in WORK_TOOL_ORDER:
        queries_path = WORK_QUERY_ROOT / f"{domain}_queries_and_answers.csv"
        queries_df = pd.read_csv(queries_path, dtype=str).fillna("")
        if args.limit > 0:
            queries_df = queries_df.head(args.limit).copy()
        domain_counts[domain] = len(queries_df)
        print(f"Running Work RAG inference for {domain} ({len(queries_df)} queries)...")

        rows: List[Optional[Dict[str, object]]] = [None] * len(queries_df)
        futures = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(retriever, args.top_k, args.max_context_tokens, domain_to_doc_ids),
        ) as pool:
            for index, row in enumerate(queries_df.to_dict(orient="records")):
                futures.append(
                    pool.submit(
                        _run_single_query,
                        index,
                        str(row.get("query") or ""),
                        str(row.get("domains") or ""),
                        args.model,
                        args.tool_selection,
                    )
                )

            progress_cls = _tqdm()
            progress_iter: Iterable[object]
            if progress_cls is None:
                progress_iter = as_completed(futures)
            else:
                progress_iter = progress_cls(as_completed(futures), total=len(futures), desc=f"rag_{domain}", unit="q")
            for future in progress_iter:
                try:
                    result = future.result()
                except Exception:
                    _append_error_log(
                        results_txt_path,
                        f"[rag][work][fatal] domain={domain}\n{traceback.format_exc()}",
                    )
                    raise
                rows[int(result["index"])] = result
                if result.get("error"):
                    error_text = (
                        f"[rag][work][query_error] domain={domain} index={result['index']} "
                        f"query={result.get('query', '')}\n"
                        f"error={result.get('error')}"
                    )
                    if result.get("error_traceback"):
                        error_text += f"\n{result['error_traceback']}"
                    _append_error_log(results_txt_path, error_text)

        materialized_rows: List[Dict[str, object]] = [row for row in rows if row is not None]
        predictions_payload: List[Dict[str, object]] = []
        for result in materialized_rows:
            usage = TokenUsage.from_mapping(result["usage"], calls=result["usage"].get("calls", 0))
            agent_usages.append(usage)
            callback_costs.append(float(result["callback_cost_usd"]))
            stage_times = result["time_s_by_stage"]
            retrieval_elapsed_total += float(stage_times["retrieval"])
            agent_elapsed_total += float(stage_times["agent"])
            retrieval = result["retrieval"]
            retrieval_rows.append(
                {
                    "domain": domain,
                    "query": result["query"],
                    "elapsed_s": retrieval["elapsed_s"],
                    "context_tokens": retrieval["context_tokens"],
                    "hits": retrieval["hits"],
                    "context_text": retrieval["context_text"],
                }
            )
            per_query_metric_rows.append(
                {
                    "domain": domain,
                    "query": result["query"],
                    "status": "ok" if not result["error"] else "error",
                    "error": result["error"],
                    "time_s_total": float(stage_times["retrieval"]) + float(stage_times["agent"]),
                    "time_s_by_stage": stage_times,
                    "token_usage_by_stage": {
                        "retrieval": TokenUsage().to_dict(),
                        "agent": usage.to_dict(),
                    },
                    "llm_calls": usage.calls,
                    "prompt_cache_hit_tokens": usage.prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": usage.prompt_cache_miss_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "retrieval_doc_ids": [hit["doc_id"] for hit in retrieval["hits"]],
                    "retrieval_scores": [hit["score"] for hit in retrieval["hits"]],
                    "retrieval_time_s": retrieval["elapsed_s"],
                    "retrieval_context_tokens": retrieval["context_tokens"],
                }
            )
            predictions_payload.append(
                {
                    "query": result["query"],
                    "function_calls": result["function_calls"],
                    "full_response": result["full_response"],
                    "error": result["error"],
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "total_cost_usd": float(result["callback_cost_usd"]),
                    "prompt_cache_hit_tokens": usage.prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": usage.prompt_cache_miss_tokens,
                    "retrieval_doc_ids": [hit["doc_id"] for hit in retrieval["hits"]],
                    "retrieval_scores": [hit["score"] for hit in retrieval["hits"]],
                    "retrieval_context_tokens": retrieval["context_tokens"],
                    "retrieval_time_s": retrieval["elapsed_s"],
                }
            )

        predictions_df = pd.DataFrame(predictions_payload)
        pred_path = results_dir / f"predictions_{domain}.csv"
        predictions_df.to_csv(pred_path, index=False, quoting=csv.QUOTE_ALL)
        predictions_df.to_csv(predictions_alias_dir / f"predictions_{domain}.csv", index=False, quoting=csv.QUOTE_ALL)

    inference_elapsed = _seconds_now() - pipeline_start
    write_jsonl(retrieval_hits_path, retrieval_rows)
    write_jsonl(per_query_metrics_path, per_query_metric_rows)

    agent_total = aggregate_token_usage(agent_usages)
    retrieval_total = TokenUsage()
    eval_elapsed = 0.0
    pass_rates_payload: Dict[str, object] = {}

    if not args.skip_eval:
        print("Running Work evaluation...")
        eval_a, eval_b = _evaluate_predictions(results_dir, results_txt_path)
        eval_elapsed = eval_a + eval_b
        pass_rates_payload = json.loads((results_dir / "pass_rates.json").read_text(encoding="utf-8"))

    pipeline_elapsed_total = _seconds_now() - pipeline_start
    pipeline_elapsed_no_eval = inference_elapsed

    price_key = resolve_price_key(args.model)
    agent_price = compute_price_usd(model_name=args.model, usage=agent_total)
    total_price = sum_mapping_values([agent_price])
    callback_total = sum_mapping_values(callback_costs)

    summary_payload = {
        "variant": "rag",
        "model_name": args.model,
        "split": args.split,
        "tool_selection": args.tool_selection,
        "n_examples_total": int(pass_rates_payload.get("n_examples", sum(domain_counts.values()))),
        "Local Pass Rate": pass_rates_payload.get("Local Pass Rate", 0.0),
        "Global Pass Rate": pass_rates_payload.get("Global Pass Rate", 0.0),
        "WorkBench Accuracy": pass_rates_payload.get("WorkBench Accuracy", 0.0),
        "WorkBench Exact Match": pass_rates_payload.get("WorkBench Exact Match", 0.0),
        "WorkBench Unwanted Side Effects": pass_rates_payload.get("WorkBench Unwanted Side Effects", 0.0),
        "total_prompt_tokens": int(agent_total.prompt_tokens),
        "total_completion_tokens": int(agent_total.completion_tokens),
        "total_tokens": int(agent_total.total_tokens),
        "total_prompt_cache_hit_tokens": int(agent_total.prompt_cache_hit_tokens),
        "total_prompt_cache_miss_tokens": int(agent_total.prompt_cache_miss_tokens),
        "total_price_usd": float(total_price),
        "total_cost_usd_callback": float(callback_total),
        "retrieval_top_k": int(args.top_k),
        "retrieval_max_context_tokens": int(args.max_context_tokens),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cost_lines = [
        f"run_start={run_start_iso}",
        "task=work",
        "variant=rag",
        f"split={args.split}",
        f"model={args.model}",
        f"model_slug={model_slug(args.model)}",
        f"tool_selection={args.tool_selection}",
        f"workers={args.workers} (effective={workers})",
        f"retrieval_top_k={args.top_k}",
        f"retrieval_max_context_tokens={args.max_context_tokens}",
        f"retrieval_corpus_path={WORK_TRAIN_PATH}",
        f"output_dir={run_dir}",
        f"results_dir={results_dir}",
        f"queries_total={sum(domain_counts.values())}",
        f"run_end={iso_now()}",
        f"retrieval_elapsed_s_total={retrieval_elapsed_total}",
        f"agent_elapsed_s_total={agent_elapsed_total}",
        f"total_time_s_excluding_eval={pipeline_elapsed_no_eval}",
        f"eval_time_s={eval_elapsed}",
        f"total_time_s={pipeline_elapsed_total}",
        f"LLM token usage (retrieval): {retrieval_total.to_cost_line()}",
        f"LLM token usage (agent): {agent_total.to_cost_line()}",
        f"llm_tokens: {agent_total.to_cost_line()}",
        f"llm_price_model_key={price_key}",
        f"total_price_usd={total_price:.6f}",
        f"total_cost_usd_callback={callback_total:.6f}",
        f"model_name={args.model}",
        f"split={args.split}",
        f"tool_selection={args.tool_selection}",
        f"workers={args.workers}",
    ]
    cost_txt_path.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    run_metrics = {
        "metadata": {
            "task": "work",
            "variant": "rag",
            "split": args.split,
            "model": args.model,
            "model_slug": model_slug(args.model),
            "tool_selection": args.tool_selection,
            "workers_requested": int(args.workers),
            "workers_effective": int(workers),
            "top_k": int(args.top_k),
            "max_context_tokens": int(args.max_context_tokens),
            "retrieval_corpus_source": str(WORK_TRAIN_PATH),
            "retrieval_corpus_docs": len(corpus_docs),
            "results_dir": str(results_dir),
            "outputs_dir": str(run_dir),
            "domain_counts": domain_counts,
        },
        "stages": {
            "retrieval": {
                "elapsed_s": retrieval_elapsed_total,
                "token_usage": retrieval_total.to_dict(),
            },
            "agent": {
                "elapsed_s": agent_elapsed_total,
                "token_usage": agent_total.to_dict(),
                "price_usd": agent_price,
                "callback_cost_usd": callback_total,
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
            "token_usage": agent_total.to_dict(),
            "llm_price_model_key": price_key,
            "llm_price_usd_total": total_price,
            "llm_callback_cost_usd_total": callback_total,
            "total_tokens": agent_total.total_tokens,
        },
        "evaluation": pass_rates_payload,
    }
    run_metrics_path.write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Work RAG baseline complete: {run_dir}")


if __name__ == "__main__":
    main()

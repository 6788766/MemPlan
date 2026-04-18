# Analysis: Query Load (Travel)

This folder contains the scalability experiment runner for the Travel task.

## Overview

The script `analysis/query_load.py` runs Travel test-set experiments for three methods:
- react
- norepair
- memplan

For ReAct-only runs, use `analysis/query_load_react.py`.
For chunked MemPlan runs (N queries + repair per chunk), use `analysis/query_load_memplan.py`.

It collects:
- Checkpoint metrics at 200/400/600/800/1000 queries (parallel wall-clock time, full-run token totals, full-run price).
- Per-query time (one line per query, summed by template_id across stages).

Note: evaluation is disabled for these experiments (no TravelPlanner eval is run).

## Inputs

- Travel dataset: `tasks/travel/dataset/{split}.csv` (default split: test)
- MemPlan pipeline: `task_helper/travel/run.sh`
- ReAct baseline: `baseline/tool_agents.py`
- Price table: `artifacts/input/price.json`

## Outputs

All outputs go under:
- `artifacts/output/travel/analysis/query_load/{model}_{method}`

Per-run logs (timestamped):
- `{model}_{method}_YYYYMMDD_HHMMSS.log` (inside the method folder)

Checkpoint metrics (full-run tokens/price + checkpoint time):
- `{model}_{method}_{N}.json` where N in {200, 400, 600, 800, 1000}

Per-query times (one line per query):
- `{model}_{method}.txt`

Summaries:
- `{model}_{method}_summary.json`

Run artifacts copy (per model/method run):
- Same folder as above (pipeline outputs and analysis outputs live together).

### Output formats

Checkpoint metrics file (`{model}_{method}_{N}.json`):

```
{
  "price_usd_parts": {
    "norepair": 0.100000,
    "repair": 0.023456
  },
  "price_usd": 0.123456,
  "time_s": 456.789,
  "tokens": {
    "output": 12345,
    "prompt_cache_hit": 67890,
    "prompt_cache_miss": 1112,
    "total": 80247
  }
}
```

Checkpoint tokens/price are computed for the **first N completed queries** (based on
final-stage completion order), not full-run totals.

Checkpoint ID list (`{model}_{method}_{N}_ids.json`):

```
{
  "template_ids": ["0", "1", "2", "..."]
}
```

Summary file (`{model}_{method}_summary.json`):

```
{
  "cost_path": ".../cost.txt",
  "method": "memplan",
  "model": "gpt-5.2",
  "model_tag": "gpt52",
  "price_usd": 3.141593,
  "price_usd_parts": {
    "norepair": 2.900000,
    "repair": 0.241593
  },
  "split": "test",
  "time_s": 987.654,
  "tokens": {
    "output": 12345,
    "prompt_cache_hit": 67890,
    "prompt_cache_miss": 1112,
    "total": 80247
  },
  "workers": 18
}
```

Per-query time file (`{model}_{method}.txt`):

- One float per line.
- Order is not preserved; each value is the total time for a single query.
- Includes a per-query share of ViewSelect time (view_select elapsed / total queries).

Example:

```
1.234567
0.998877
3.210000
```

Per-query time map (`{model}_{method}.jsonl`):

- One JSON object per query with `template_id` and `time_s`.
- Useful for matching times to specific query IDs.

Per-query LLM usage (MemPlan):

- InitTemplate usage per query lives in `init_templates_<split>.jsonl` under `notes.llm`.
- LLM repair usage per query is recorded in the repaired output JSONL under `notes.llm_repair.usage`
  (present whenever a repair attempt made an LLM call).

Chunked MemPlan per-query time (from `analysis/query_load_memplan.py`):

- `chunk_times.jsonl` includes `time_s_norepair`, `time_s_repair`, and `time_s_total` per query.

## Running

Run all methods and models (default 18 workers, test split):

```
python analysis/query_load.py --method all --model all
```

Run one model/method:

```
python analysis/query_load.py --method memplan --model gpt52
```

Run ReAct only (with retries + checkpoints):

```
python analysis/query_load_react.py --model gpt52 --split test --workers 18
```

Run MemPlan in chunks (200 queries + repair per chunk):

```
python analysis/query_load_memplan.py --model gpt52 --split test --chunk-size 200 --workers 18
```

## Work scalability (validation)

Use `analysis/work_parallel.py` to run the Work task pipeline (no evaluation) with
multiple worker counts and log wall-clock time for base vs. repair stages. Repair
time is taken from the `LLM repair done: ... elapsed=...` line, so only repair
compute time is counted.

Outputs:
- `artifacts/output/work/analysis/parallel.json` (per-model times for 12/24/36/48 workers)
- `artifacts/output/work/analysis/parallel_logs` (per-run logs)

Examples:

```
python analysis/work_parallel.py --model gpt-5-nano
```

```
python analysis/work_parallel.py --model gpt-5-mini --model gpt-5-nano --force
```

Conversion of repaired outputs is skipped by default. If you want it:

```
python analysis/work_parallel.py --model gpt-5-nano --convert
```

## Work view sweep (validation)

If you run multiple WorkBench experiments with different **numbers of memory views**
(e.g., 15/30/45/60), store each run under:

- `artifacts/output/work/analysis/view/<run_name>/`

Each run folder should contain (at minimum):
- `match_<split>.json` (from `planner.view_select`)
- `optimized_<split>.jsonl` (from `planner.twin_track_multi`)
- `views/memory.jsonl` (the exact subset of memory views used for that run)

Optional (for “regular pipeline” metrics):
- `cost.txt`
- `results/pass_rates.json`

### Run folder naming convention

Use a stable naming convention so the collector can parse the settings from the folder name:

- `<model_slug>_<num_views>_<split>_<tag>/`

Example:

- `gpt52_15_validation_seed0/`

### Extra metrics for the sweep

These scripts add two sweep-specific metrics on top of whatever your run already wrote:

- Tool-only edge coverage from ViewCover (`match_<split>.json`):
  - `tool_only_edges` / `tool_only_rate`
  - Definition: an edge is “tool-only” if `lambda[edge]` contains at least one tool view id and no `memory::...` view id.
- Real tool calls from the final plan (`optimized_<split>.jsonl`):
  - `calls_total`, `calls_check`, `calls_do`, `calls_by_domain`
- Tool-vs-memory candidates in the AND–OR tree (`tree_<split>.json`):
  - `tree_candidates_tool_total`, `tree_candidates_memory_total`
  - `tree_tool_effective_total` excludes tool candidates that are “shadowed” by an equivalent memory candidate (same tool name + args) within the same slot.

### Collect summaries across runs

Write a combined summary (one row per run):

```
python analysis/work_view_collect.py \
  --root artifacts/output/work/analysis/view \
  --split validation \
  --out-jsonl artifacts/output/work/analysis/view/summary_validation.jsonl \
  --out-csv artifacts/output/work/analysis/view/summary_validation.csv
```

Update `view_metrics.json` for all view counts of one model (writes into each run folder):

```
python analysis/work_view_collect.py \
  --root artifacts/output/work/analysis/view \
  --split validation \
  --model-slug gpt52 \
  --views 15,30,45,60 \
  --tag seed0 \
  --write-per-run \
  --per-run-filename view_metrics.json
```

### Run the sweep end-to-end (writes only under analysis/view)

`analysis/work_view_sweep.py` runs the Work pipeline for multiple memory-view counts and writes *everything* under:

- `artifacts/output/work/analysis/view/<model_slug>_<num_views>_<split>_<tag>/`

Defaults:
- `--split validation`
- `--views 15,30,45,60`
- `--seeds 0`
- `--workers 0` (all CPUs; per-query parallelism where supported)
- `--repair-rounds 4`
- `--tee` is enabled (stage output is streamed while logs are saved under each run folder)
- `--show-progress` is enabled (per-template progress from Twin-Track + repair)
- `--mcts-iterations 10`
- `--max-id-expansions 6`
- `--eval-mode final` (only runs convert+eval for the final output; i.e., after repair r4 by default)
- If `--templates-in` is not provided, it reuses `artifacts/output/work/<model_slug>_<split>/init_templates_<split>.jsonl` when present.

Example:

```
python analysis/work_view_sweep.py --model gpt-5.2
```

Resume after an interruption (skip completed stages; do not rerun finished runs):

```
python analysis/work_view_sweep.py --model gpt-5-mini --views 45,60 --resume
```

### Add one more repair round (e.g., r4 → r5) and evaluate

If you already ran the sweep (default `--repair-rounds 4`) and want to run exactly one
additional repair round on the *latest* round output for each view setting (15/30/45/60),
use:

```
python analysis/work_view_repair_next_round.py --model gpt-5.2
```

This finds the latest existing `repair/rN/` (e.g., `r4`) under each matching run folder,
creates `repair/r{N+1}/` (e.g., `r5`), and writes
the same per-round outputs (`cost.txt`, `logs/`, `results/`, `results/pass_rates.json`).

Run N more rounds (evaluate only the final added round):

```
python analysis/work_view_repair_next_round.py --model gpt-5.2 --rounds 3
```

Defaults:
- `--split validation`
- `--views 15,30,45,60`
- `--tag seed0`
- `--workers 0` (all CPUs)

Example (recommended: reuse an existing init-templates file so the sweep is purely about memory views):

```
python analysis/work_view_sweep.py \
  --model gpt-5.2 \
  --seeds 0 \
  --views 15,30,45,60
```

By default, if `--templates-in` is not provided, the sweep will *try to reuse*:

- `artifacts/output/work/<model_slug>_<split>/init_templates_<split>.jsonl`

and only falls back to running `planner.init_template` if that file does not exist.

## Travel candidate sweep (validation)

`analysis/candidate_sweep.py` sweeps the candidate cap knobs for Travel:
- `--max-tool-candidates` (ComposeMatch tool candidates per call)
- `--max-slot-candidates` (LLM repair prompt candidates per slot)

It reuses cached init-templates from an existing per-model validation run at:
`artifacts/output/travel/<model_slug>_validation/init_templates_validation.jsonl`,
and copies the init-template timing/token lines from that run’s `cost.txt` into each sweep run’s `cost.txt`.

Outputs:
- `artifacts/output/travel/analysis/candidate/<model_slug>_validation_k<K>_<timestamp>/`

Example:

```
python analysis/candidate_sweep.py --model full --split validation --ks 2,4,6,8,10
```

Optional: run a second repair pass on the repaired plan and reevaluate:

```
python analysis/candidate_sweep.py --model full --split validation --ks 2,4,6,8,10 --repair-rounds 2
```

You can use any `--repair-rounds N` (N >= 1) to run N sequential repair+eval rounds.

To include an unlimited tool-candidate sweep point, append `unlimited` (alias: `0`) to `--ks`:

```
python analysis/candidate_sweep.py --model full --split validation --ks 2,4,6,8,10,unlimited
```
```

## Work unstructured repair ablations (validation)

These scripts **do not rerun the Work pipeline**. They load an existing Work run’s:
- `optimized_<split>.jsonl`
- `tree_enriched_<split>.json` (fallback: `tree_<split>.json`)

Then they only call the LLM for examples that are **incorrect or have unwanted side effects** per WorkBench.

### Same-info, unstructured output

`analysis/work_unstructured_same_info.py` gives the LLM the same `repair_input` context used by `planner.llm_repair` (tool schemas, candidate IDs, retrieved records/checks, bindings/check_results, canonical values), but **does not use structured edits**:
- The LLM outputs a **JSON array of call strings** (a full rewrite), not patch ops (`insert/delete/move/update`).
- The script overwrites the plan’s action list by writing those call strings into `attrs.raw_action` for evaluation.

Single model (defaults: `--split validation`, `--workers 14`):

```
python -m analysis.work_unstructured_same_info --base-model gpt-5-mini --model gpt-5-mini
```

All 3 GPT models in one grouped run:

```
python -m analysis.work_unstructured_same_info --full
```

This expects the corresponding base run artifacts to exist under `artifacts/output/work/<model_slug>_<split>/`.
To override the model list, pass `--models ...` repeatedly (only used with `--full`).

Useful flags:
- `--max-failed 200` (limit attempted repairs)
- `--all-metrics` (also run `calculate_all_metrics.py`; slow)
- `--no-llm` (wiring/debug)

Outputs go under `artifacts/output/work/analysis/` (per-run folder with `summary.json`, `pass_rates.json`, `cost.txt`, and optional `metrics.txt`; full mode writes `group_summary.json`).

### Same-info, unstructured text I/O (true unstructured)

`analysis/work_unstructured_text_io.py` uses the same info content as the repair context, but makes the *prompt input* unstructured too:
- Input: plain text sections (no JSON payload) containing the query, current actions, allowed tools/params, candidate IDs, retrieved records/checks, bindings/check_results, canonical values.
- Output: plain text call strings (one per line), instead of JSON or patch ops.

Single model:

```
python -m analysis.work_unstructured_text_io --base-model gpt-5-mini --model gpt-5-mini
```

All 3 GPT models in one grouped run:

```
python -m analysis.work_unstructured_text_io --full
```

Useful flags:
- `--max-failed 200` (limit attempted repairs)
- `--all-metrics` (slow)
- `--no-llm` (wiring/debug)

### Tree + constraints (plain text) repair

`analysis/work_tree_text_repair.py` provides the **full tool list**, the **AND–OR tree candidates** (from `tree_enriched_<split>.json`) and **WorkBench constraints** in plain text, and includes the **MCTS chosen plan** as the repair target. The LLM outputs a repaired full call list (no patch ops).

Single model:

```
python -m analysis.work_tree_text_repair --base-model gpt-5-mini --model gpt-5-mini --enforce-tree-calls
```

All 3 GPT models in one grouped run:

```
python -m analysis.work_tree_text_repair --full --enforce-tree-calls
```

Default behavior allows the LLM to choose any tools from the **full tool list**; the tree is optional guidance.
Use `--enforce-tree-calls` only if you want the stricter variant that drops output calls not present in the tree candidate set.

Useful flags:
- `--max-failed 200` (limit attempted repairs)
- `--max-slot-candidates 20` (prompt size control)
- `--include-constraint-descriptions` (longer prompt)
- `--output-json-list` (ask for JSON array output instead of one-call-per-line)
- `--enforce-tree-calls` (optional: drop output calls not present in the tree candidate set)
- `--all-metrics` (slow)
- `--no-llm` (wiring/debug)

### Running multiple sessions with different API keys

If you launch multiple runs from the same repo (e.g., on HPC), set `OPENAI_API_KEY`
per session so each run uses a different key. The code will **not** override an
existing `OPENAI_API_KEY` from `.env`.

Examples:

```
OPENAI_API_KEY=sk-key-A python analysis/query_load.py --method memplan --model gpt52
```

```
OPENAI_API_KEY=sk-key-B python analysis/query_load.py --method memplan --model gpt5mini
```

### ReAct retry behavior

ReAct runs will retry automatically if any per-query JSON has an `error` field or
if not all queries are written. The run sleeps briefly and reruns with
`--resume-failed` until all queries succeed. Logs are timestamped per run.

Override workers or split:

```
python analysis/query_load.py --method norepair --model deepseekchat --workers 18 --split test
```

## How memplan is produced

For memplan, the script runs:
1) A norepair pipeline (full run with `RUN_REPAIR=0`).
2) A repair-only pass on the norepair output, with the same worker count.

The memplan metrics are computed by summing norepair + repair time, and by
aggregating tokens/price across those two parts.

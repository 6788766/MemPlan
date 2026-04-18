<!-- SIGMOD anonymous submission: prevent search-engine indexing of any HTML-rendered copy of this README. -->

<meta name="robots" content="noindex">

# MemPlan

**MemPlan** is a cost-aware agentic task planner that reuses historically
validated execution traces as **plan views** over a shared memory graph.
Given a natural-language request, MemPlan composes plan views that best cover
the request and invokes external tools only for uncovered fragments, achieving
competitive planning quality at a fraction of the cost of planning from scratch.

## Pipeline

MemPlan has four online modules and three offline artifacts.

**Offline artifacts**

1. **Memory graph** `𝒢` — a curated graph of validated historical plans,
   organized by a four-layer ontology *Plan → Phase → Action → Output*.
2. **View library** `𝒱` — plan-view templates derived from `𝒢`; each view
   records a reusable, parameterized pattern together with its validated
   instantiations.
3. **Tool repository** `𝒯` — typed, black-box tools invoked only when plan
   views do not fully cover a request.

**Online pipeline**

| # | Module                 | Algorithm          | Input → Output                                    |
|---|------------------------|--------------------|---------------------------------------------------|
| 1 | **Query Converter**    | LLM `ℳ`            | NL query `q` → structured task query `q_o = (Q, C)` |
| 2 | **Plan View Selector** | **MemPSel**        | `(𝒱, Q)` → selected views `𝒱_s` maximizing coverage under a budget `k` |
| 3 | **Plan Assembler**     | **MemPAssem**      | `𝒱_s` → compressed AND–OR plan tree `𝒯_P` (four-level grammar) |
| 4 | **Plan Optimizer**     | **MemPOpt**        | `𝒯_P` → top-`K` feasible plans Π via twin-track MCTS |
| 5 | *LLM-based Refinement* | optional           | targeted repair of constraint-violating slots    |

MemPSel gives a `1 − (1 − 1/k)^k` approximation for plan-view selection;
MemPAssem runs in PTIME; MemPOpt achieves a bounded-error top-`K` result
with high probability.

## Repository layout

```
memory_graph/          Graph schema + extractor; constructs memory graph 𝒢
planner/
  init_template.py     Query Converter (NL query → (Q, C))
  view_select.py       Plan View Selector (MemPSel)
  compose_match.py     Plan Assembler, single-round (MemPAssem)
  compose_match_multi.py  Multi-round assembler for WorkBench
  twin_track.py        Plan Optimizer, single-round (MemPOpt)
  twin_track_multi.py  Multi-round MCTS engine
  llm_repair.py        Optional LLM-based refinement

task_helper/travel/    TravelPlanner tools, evaluator, view generators
task_helper/work/      WorkBench tools, evaluator, view generators
baseline/              From-scratch LLM planners and history-aware baselines
                       (ReAct two-stage, RAG, no-selection, no-MCTS, …)
analysis/              Scalability studies and candidate-count sweeps
artifacts/input/       Per-task configs, prompts, view library, datasets
environment.yml        Conda environment spec
memplan                Top-level CLI dispatcher
```

Generated outputs (memory graph, view matches, AND–OR trees, optimized plans,
evaluation reports) are written to `artifacts/output/<task>/` (gitignored).

## Benchmarks

MemPlan is evaluated on two agentic task-planning benchmarks:

- **TravelPlanner** — a travel-planning benchmark with ~4M records, six
  search tools, and constraint-based evaluators. We report the 180-query
  held-out validation split unless noted.
- **WorkBench** — a workplace benchmark with five sandbox databases, 26
  tools across five domains, and 690 NL task queries with outcome-centric
  evaluators.

Datasets are not bundled with the repository. Download TravelPlanner and
WorkBench from their original sources and place them under
`artifacts/input/travel/dataset/` and `artifacts/input/work/dataset/`.

## Setup

```bash
conda env create -f environment.yml
conda activate planner
cp .env.example .env   # then fill in OPENAI_API_KEY / DEEPSEEK_API_KEY
```

## Quick start

End-to-end MemPlan run on TravelPlanner (validation split):

```bash
TASK=travel SPLIT=validation MODEL=gpt-5-mini memplan
```

End-to-end MemPlan run on WorkBench:

```bash
TASK=work MODE=multi SPLIT=validation MODEL=gpt-5-mini memplan
```

Evaluate TravelPlanner output:

```bash
memplan eval \
    --set-type validation \
    --submission artifacts/output/travel/optimized_validation.jsonl
```

Baselines (TravelPlanner two-stage ReAct; WorkBench ReAct):

```bash
memplan baseline travel
memplan baseline work
```

> `memplan` is the top-level dispatcher. It wraps `task_helper/run.sh`,
> `task_helper/travel/runners/eval_bridge.py`, and the baseline scripts.
> Add the repository root to your `PATH` or symlink `memplan` into a
> directory that is.

## Running individual pipeline stages

For debugging or reproducing a single stage, each module exposes a CLI; see
module docstrings for the full flag list.

```bash
# Offline: build memory graph 𝒢 and view templates
python graph_builder.py --task travel
python -m task_helper.travel.utils.generate_memory_view
python -m task_helper.travel.utils.generate_tool_view

# Online: Query Converter → MemPSel → MemPAssem → MemPOpt → (optional) LLM refine
python -m planner.init_template  --task travel --split validation --model gpt-5-mini
python -m planner.view_select    --task travel --split validation --templates <init.jsonl>
python -m planner.compose_match  --task travel --split validation --templates <init.jsonl> --match <match.json>
python -m planner.twin_track     --task travel --tree <tree.json>
python -m planner.llm_repair     --task travel --tree <tree.json> --input <optimized.jsonl>
```

## Configuration

Task-specific behavior is driven by config files under `artifacts/input/<task>/`:

- `planner.json` — matching rules, tool handlers, MemPOpt hooks
- `views/tool.json` — tool micro-view patterns and costs
- `template.txt`, `repair_prompt.txt`, `repair.json` — LLM prompts
- `graph_config.json` — ontology bindings for memory-graph extraction

Optional per-task Python hooks live under `task_helper/<task>/`.

## Analysis and scalability

Scripts under `analysis/` run the scalability and ablation studies:

- `analysis/candidate_sweep.py` — sweep the candidate caps used by MemPAssem
  and the LLM refinement prompt.
- `analysis/query_load.py` — TravelPlanner scalability under concurrent
  query load.
- `analysis/work_view_sweep.py` / `analysis/work_parallel.py` — WorkBench
  view-count and parallelism studies.

## License

MIT (see `LICENSE`).

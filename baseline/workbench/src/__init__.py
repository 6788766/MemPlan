"""
Compatibility package for running WorkBench baseline scripts inside MemPlan.

WorkBench's reference scripts import modules under `src.*`. MemPlan vendors the
WorkBench tools/evaluator under `task_helper/work/`, so this package provides a
thin bridge for those imports under `baseline.workbench.src`.
"""

"""
Generate WorkBench memory views (template-level patterns).

Writes `artifacts/input/work/views/memory.jsonl`.

Design (per user preference):
- One view per `base_template` from the *train* split.
- Each view is a PlanGraph JSON object (`plan_id`, `nodes`, `edges`) matching the
  Travel-style memory view schema.
- Each plan includes two phases: CheckRound and DoRound.
- Actions are extracted from `teacher_function_calls` (fallback to `answer`) and
  assigned to Check vs Do using WorkBench's side-effect tool list.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

for parent in Path(__file__).resolve().parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from task_helper.work.tools.toolkits import tools_with_side_effects


SIDE_EFFECT_TOOL_NAMES = {t.name for t in tools_with_side_effects}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _lower_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _email_local_to_name(email_addr: str) -> str:
    local = (email_addr or "").split("@", 1)[0].replace("_", " ").replace("-", " ").replace(".", " ").strip()
    parts = [p for p in local.split() if p]
    if not parts:
        return ""
    return " ".join(p.capitalize() for p in parts[:2])


def _extract_person_names_from_query(query: str) -> List[str]:
    """
    Best-effort person name extraction from the query string.
    - Accepts both "Lena" and "lena".
    - Prefer first names only (safer than accidentally attaching words like "is").
    """
    words = _lower_words(query)
    if not words:
        return []

    # Heuristic: names in WorkBench queries are usually first names from the Atlas roster.
    # Build a small roster of likely first names from the databases (email/calendar/project/crm).
    try:
        from task_helper.work.tools.company_directory import EMAIL_ADDRESSES
    except Exception:
        EMAIL_ADDRESSES = []

    roster: set[str] = set()
    for addr in EMAIL_ADDRESSES:
        local = str(addr).split("@", 1)[0].replace(".", " ").replace("_", " ").replace("-", " ")
        toks = [t for t in _lower_words(local) if t]
        if toks:
            roster.add(toks[0])

    names: List[str] = []
    used: set[str] = set()
    for w in words:
        if w in roster and w not in used:
            names.append(w.capitalize())
            used.add(w)

    return names


def _parse_list_cell(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            return []
    return [text]


def _tool_name(action: str) -> str:
    # "calendar.create_event.func(...)" -> "calendar.create_event"
    prefix = str(action.split("(", 1)[0]).strip()
    if prefix.endswith(".func"):
        prefix = prefix[: -len(".func")]
    return prefix


def _tool_domain(tool_name: str) -> str:
    return tool_name.split(".", 1)[0] if "." in tool_name else ""


def _action_type(tool_name: str) -> str:
    return tool_name.split(".", 1)[1] if "." in tool_name else ""


def _is_side_effect(tool_name: str) -> bool:
    return tool_name in SIDE_EFFECT_TOOL_NAMES


def _parse_params(action: str) -> Dict[str, object]:
    """
    Best-effort parse of keyword args from a string like:
      "...func(k=\"v\", ...)"
    """
    try:
        expr = ast.parse(action, mode="eval").body
    except Exception:
        return {}
    if not isinstance(expr, ast.Call):
        return {}
    out: Dict[str, object] = {}
    for kw in expr.keywords:
        if not kw.arg:
            continue
        try:
            out[str(kw.arg)] = ast.literal_eval(kw.value)
        except Exception:
            out[str(kw.arg)] = None
    return out


def _normalise_domains(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            return []
    return [text]


@dataclass(frozen=True)
class TemplateExample:
    plan_id: str
    query: str
    base_template: str
    domains: Tuple[str, ...]
    actions: Tuple[str, ...]  # raw call strings


def _pick_representative_sequence(examples: Sequence[TemplateExample]) -> TemplateExample:
    """
    Choose a single representative action sequence for a base_template.
    Heuristic: most common raw sequence; tie-breaker prefers longer sequences.
    """
    if not examples:
        raise ValueError("No examples provided.")

    seq_to_examples: Dict[Tuple[str, ...], List[TemplateExample]] = defaultdict(list)
    for ex in examples:
        seq_to_examples[ex.actions].append(ex)

    counts = Counter({seq: len(items) for seq, items in seq_to_examples.items()})
    best_count = max(counts.values())
    candidates = [seq for seq, cnt in counts.items() if cnt == best_count]
    candidates.sort(key=lambda seq: (-len(seq), seq))
    best_seq = candidates[0]

    # Within the best sequence, prefer the most common domains tuple.
    items = seq_to_examples[best_seq]
    dom_counts = Counter([ex.domains for ex in items])
    best_domains = sorted(dom_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    for ex in items:
        if ex.domains == best_domains:
            return TemplateExample(
                plan_id=ex.plan_id,
                query=ex.query,
                base_template=ex.base_template,
                domains=best_domains,
                actions=best_seq,
            )
    ex0 = items[0]
    return TemplateExample(
        plan_id=ex0.plan_id,
        query=ex0.query,
        base_template=ex0.base_template,
        domains=best_domains,
        actions=best_seq,
    )


def _build_template_view(*, template_id: str, rep: TemplateExample) -> Dict[str, object]:
    plan_id = str(rep.plan_id or "").strip() or f"work_template::{template_id}"
    plan_node_id = f"plan::{plan_id}"

    check_phase_id = f"{plan_node_id}::phase::check"
    do_phase_id = f"{plan_node_id}::phase::do"

    nodes: List[Dict[str, object]] = [
        {
            "id": plan_node_id,
            "type": "Plan",
            "attrs": {
                "task_name": "work",
                "base_template": rep.base_template,
                "domains": list(rep.domains),
            },
        },
        {
            "id": check_phase_id,
            "type": "Phase",
            "attrs": {"phase_key": "check", "phase_type": "CheckRound", "phase_index": 1},
        },
        {
            "id": do_phase_id,
            "type": "Phase",
            "attrs": {"phase_key": "do", "phase_type": "DoRound", "phase_index": 2},
        },
    ]
    edges: List[Dict[str, object]] = [
        {"src": plan_node_id, "dst": check_phase_id, "type": "hasPhase", "attrs": {}},
        {"src": plan_node_id, "dst": do_phase_id, "type": "hasPhase", "attrs": {}},
    ]

    action_counter = 0
    by_phase: Dict[str, List[str]] = {"check": [], "do": []}
    for raw_action in rep.actions:
        tn = _tool_name(raw_action)
        by_phase["do" if _is_side_effect(tn) else "check"].append(raw_action)

    for phase_key in ("check", "do"):
        phase_id = check_phase_id if phase_key == "check" else do_phase_id
        for order_index, raw_action in enumerate(by_phase[phase_key]):
            action_counter += 1
            tn = _tool_name(raw_action)
            dom = _tool_domain(tn)
            at = _action_type(tn) or "RawFunctionCall"
            params = _parse_params(raw_action)

            # Template-level view: keep parameter *keys* but remove instance values.
            param_placeholders = {k: f"{{{k}}}" for k in params.keys()}

            action_id = f"{plan_node_id}::action::{action_counter}"
            nodes.append(
                {
                    "id": action_id,
                    "type": "Action",
                    "attrs": {
                        "action_type": at,
                        "params": param_placeholders,
                        "order_index": order_index,
                        "tool_name": dom,
                        "qualified_tool_name": tn,
                        "side_effect": _is_side_effect(tn),
                    },
                }
            )
            edges.append({"src": phase_id, "dst": action_id, "type": "hasAction", "attrs": {}})

    return {"plan_id": plan_id, "nodes": nodes, "edges": edges}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = next(p for p in Path(__file__).resolve().parents if (p / "artifacts").is_dir())
    default_train = root / "artifacts" / "input" / "work" / "dataset" / "train.csv"
    default_out = root / "artifacts" / "input" / "work" / "views" / "memory.jsonl"
    default_templates_txt = default_out.parent / "base_templates.txt"
    default_fixed_train = root / "artifacts" / "input" / "work" / "dataset" / "train_fixed.csv"

    parser = argparse.ArgumentParser(description="Generate WorkBench template-level memory views.")
    parser.add_argument("--train", type=Path, default=default_train, help="Path to train.csv")
    parser.add_argument("--out", type=Path, default=default_out, help="Output memory.jsonl path")
    parser.add_argument("--templates-out", type=Path, default=default_templates_txt, help="Output base_templates.txt path")
    parser.add_argument(
        "--fixed-train-out",
        type=Path,
        default=None,
        help=f"Optional output path to write a repaired train CSV (default suggestion: {default_fixed_train}).",
    )
    return parser.parse_args(argv)


def _parse_call_params(raw_call: str) -> Dict[str, object]:
    """
    Best-effort parsing of .func(...) call kwargs into a dict.
    """
    try:
        expr = ast.parse(raw_call, mode="eval").body
    except Exception:
        return {}
    if not isinstance(expr, ast.Call):
        return {}
    params: Dict[str, object] = {}
    for kw in expr.keywords:
        if not kw.arg:
            continue
        try:
            params[str(kw.arg)] = ast.literal_eval(kw.value)
        except Exception:
            params[str(kw.arg)] = None
    return params


def _needs_email_resolution(actions: Sequence[str]) -> bool:
    """
    Returns True if the trace includes any call that uses an email-like parameter.
    """
    for call in actions:
        params = _parse_call_params(call)
        # Pattern: update_* calls may encode "email-ness" in `field` instead of the kwarg name.
        field = params.get("field")
        new_value = params.get("new_value")
        if isinstance(field, str) and "email" in field.lower() and new_value is not None:
            return True
        for k, v in params.items():
            key = str(k).lower()
            if key == "recipient" or "email" in key:
                # If it's already a concrete email, we still want the lookup pattern
                # (non-guessing pipeline) unless the trace already contains it.
                if isinstance(v, str) and v.strip():
                    return True
    return False


def _repair_teacher_trace(*, query: str, actions: Sequence[str]) -> Tuple[Tuple[str, ...], bool]:
    """
    Inject missing Check actions needed by the non-guessing pipeline.

    Current repair:
      - If there are email-using Do calls but no company_directory lookup, insert
        company_directory.find_email_address(name="...") for the names we can
        extract from the query. If no names are extractable, insert one generic lookup.
    """
    acts = list(actions)
    has_dir = any(_tool_name(a) == "company_directory.find_email_address" for a in acts)
    repaired_any = False

    # If the query already contains explicit emails, do not inject directory lookups.
    if "@" in (query or ""):
        return tuple(acts), False

    # Normalize obvious wrong-domain emails in traces (e.g. "@example.com") to "@atlas.com" when unique.
    try:
        from task_helper.work.tools.company_directory import EMAIL_ADDRESSES
    except Exception:
        EMAIL_ADDRESSES = []

    atlas_addrs = [a for a in EMAIL_ADDRESSES if "@atlas" in str(a)]

    def _resolve_example_email(addr: str) -> Optional[str]:
        if "@example" not in addr:
            return None
        local = addr.split("@", 1)[0].lower()
        token = local.split(".", 1)[0].split("_", 1)[0].split("-", 1)[0]
        if not token:
            return None
        candidates = [a for a in atlas_addrs if token in str(a).lower()]
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            return str(candidates[0])
        return None

    def _rewrite_call_with_params(raw_call: str, new_params: Dict[str, object]) -> str:
        try:
            expr = ast.parse(raw_call, mode="eval").body
        except Exception:
            return raw_call
        if not isinstance(expr, ast.Call):
            return raw_call
        # tool prefix like "email.send_email.func"
        prefix = raw_call.split("(", 1)[0].strip()
        parts: List[str] = []
        for kw in expr.keywords:
            if not kw.arg:
                continue
            key = str(kw.arg)
            value = new_params.get(key)
            if value is None:
                continue
            text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            parts.append(f'{key}="{text}"')
        return f"{prefix}(" + ", ".join(parts) + ")"

    rewritten: List[str] = []
    for raw in acts:
        params = _parse_call_params(raw)
        changed = False
        for k, v in list(params.items()):
            if isinstance(v, str) and _EMAIL_RE.match(v.strip()) and "@example" in v:
                resolved = _resolve_example_email(v.strip())
                if resolved:
                    params[k] = resolved
                    changed = True
        # field/new_value email updates
        field = params.get("field")
        if isinstance(field, str) and "email" in field.lower():
            nv = params.get("new_value")
            if isinstance(nv, str) and _EMAIL_RE.match(nv.strip()) and "@example" in nv:
                resolved = _resolve_example_email(nv.strip())
                if resolved:
                    params["new_value"] = resolved
                    changed = True
        if changed:
            raw = _rewrite_call_with_params(raw, params)
            repaired_any = True
        rewritten.append(raw)
    acts = rewritten

    if has_dir:
        return tuple(acts), repaired_any

    if not _needs_email_resolution(acts):
        return tuple(acts), repaired_any

    names = _extract_person_names_from_query(query)
    injected: List[str] = []
    if names:
        for name in names:
            injected.append(f'company_directory.find_email_address.func(name="{name}")')
    else:
        injected.append('company_directory.find_email_address.func(name="Unknown")')

    # Put lookups at the front so they land in the Check phase.
    return tuple(injected + acts), True


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train_path = args.train.expanduser()
    out_path = args.out.expanduser()
    templates_out = args.templates_out.expanduser()

    df = pd.read_csv(train_path, dtype=str).fillna("")
    if "base_template" not in df.columns:
        raise SystemExit(f"Missing base_template column in {train_path}")

    grouped: Dict[str, List[TemplateExample]] = defaultdict(list)
    repaired_rows = 0
    for idx, row in df.iterrows():
        plan_id = str(row.get("plan_id") or "").strip()
        query = str(row.get("query") or "").strip()
        base_template = str(row.get("base_template") or "").strip()
        if not base_template or not plan_id:
            continue
        calls = _parse_list_cell(row.get("teacher_function_calls") or row.get("answer"))
        repaired_calls, repaired = _repair_teacher_trace(query=query, actions=calls)
        if repaired:
            repaired_rows += 1
            df.at[idx, "teacher_function_calls"] = json.dumps(list(repaired_calls), ensure_ascii=False)
        calls = list(repaired_calls)
        domains = tuple(sorted(set(_normalise_domains(row.get("domains")))))
        grouped[base_template].append(
            TemplateExample(plan_id=plan_id, query=query, base_template=base_template, domains=domains, actions=tuple(calls))
        )

    if args.fixed_train_out:
        fixed_path = args.fixed_train_out.expanduser()
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fixed_path, index=False)
        print(f"Wrote repaired train CSV to {fixed_path} (repaired_rows={repaired_rows})")

    base_templates = sorted(grouped.keys())
    views: List[Dict[str, object]] = []
    for idx, base_template in enumerate(base_templates, start=1):
        rep = _pick_representative_sequence(grouped[base_template])
        views.append(_build_template_view(template_id=f"{idx:03d}", rep=rep))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for view in views:
            fp.write(json.dumps(view, ensure_ascii=False) + "\n")

    templates_out.write_text("\n".join(base_templates) + ("\n" if base_templates else ""), encoding="utf-8")
    print(f"Wrote {len(views)} memory view(s) to {out_path}")
    print(f"Wrote {len(base_templates)} base_template(s) to {templates_out}")


if __name__ == "__main__":
    main()

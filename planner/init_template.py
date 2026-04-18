"""
LLM-based template initialisation (task-agnostic).

This script takes a single row from `validation.csv` or `test.csv` and asks an
LLM to produce a structured "plan template" JSON object. Task-specific details
(schema, extraction instructions, action vocabulary) live under:

  `artifacts/input/<task>/template.txt`      (prompt)
  `artifacts/input/<task>/schema.jsonl`      (schema; first line)
  `artifacts/input/<task>/example.jsonl`     (example output; first line)
  `artifacts/input/<task>/example_multicity.jsonl` (optional example output; first line)

The script then *overrides* selected plan fields with values copied directly
from the dataset row (so these fields are always consistent with the dataset).

How to add a new task:
  1) Create `artifacts/input/<task>/template.txt` with `{{INPUT_JSON}}` and
     `{{SCHEMA_JSON}}` placeholders (optionally also `{{EXAMPLE_JSON}}` and
     `{{EXAMPLE_INPUT_JSON}}` for in-prompt demonstrations).
  2) Create `artifacts/input/<task>/schema.jsonl` whose first line is a JSON
     object describing the desired output schema (keys/types).
  3) Optionally create `artifacts/input/<task>/example.jsonl` whose first line
     is a *filled* example output.
  3) Run: `python -m planner.init_template --task <task> ...`
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from planner.parallel import available_cpu_count


try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"
OUTPUT_ROOT = ARTIFACTS_ROOT / "output"

DEFAULT_TASK = "travel"
DEFAULT_MODEL = os.getenv("MEMPLAN_LLM_MODEL", "gpt-5-mini")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

_MODEL_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_DOTENV_LOADED = False


class LLMOutputError(RuntimeError):
    def __init__(self, message: str, *, usage: Optional[Dict[str, int]] = None) -> None:
        super().__init__(message)
        self.usage = usage


class LLMHTTPError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _model_slug(model_name: str) -> str:
    slug = _MODEL_SLUG_RE.sub("", model_name).strip().lower()
    return slug or "model"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def _ensure_dotenv_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _load_dotenv(PROJECT_ROOT / ".env")
    _DOTENV_LOADED = True


def _parse_list_cell(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return parsed
    return [text]


def _read_csv_row(path: Path, *, index: int) -> Dict[str, object]:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            if idx == index:
                return {k: (v if v != "" else None) for k, v in row.items()}
    raise IndexError(f"Row index {index} out of range for {path}")


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in reader]


def _normalise_row_inplace(row: Dict[str, object]) -> None:
    # Minimal normalisation for common TravelPlanner fields.
    if "days" in row and row["days"] is not None:
        try:
            row["days"] = int(row["days"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    if "date" in row:
        row["date"] = [str(item) for item in _parse_list_cell(row.get("date")) if item is not None]
    if "domains" in row:
        row["domains"] = [str(item) for item in _parse_list_cell(row.get("domains")) if item is not None]


def _openai_retryable_error_types() -> Tuple[type, ...]:
    if openai is None:
        return ()
    names = (
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "APIError",
    )
    types: List[type] = []
    for name in names:
        exc_type = getattr(openai, name, None)
        if isinstance(exc_type, type):
            types.append(exc_type)
    return tuple(types)


_RETRYABLE_OPENAI_ERRORS = _openai_retryable_error_types()


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 408 or status_code == 429 or status_code >= 500):
        return True
    return bool(_RETRYABLE_OPENAI_ERRORS) and isinstance(exc, _RETRYABLE_OPENAI_ERRORS)


def _load_jsonl_object(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL must contain a JSON object per line: {path}")
            return payload
    raise ValueError(f"JSONL is empty: {path}")


def _render_prompt(prompt_path: Path, *, replacements: Mapping[str, object]) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    rendered = text
    for placeholder, value in replacements.items():
        rendered = rendered.replace(str(placeholder), json.dumps(value, ensure_ascii=False, indent=2))
    if "{{" in rendered:
        raise ValueError(f"Unreplaced placeholder found in prompt template: {prompt_path}")
    return rendered


def _extract_json_block(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("LLM did not return JSON content.")
    return match.group(0)


def _extract_usage(response: object) -> Optional[Dict[str, int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None

    def _read(key: str, alt: str) -> Optional[int]:
        if isinstance(usage, dict):
            raw = usage.get(key)
            if raw is None:
                raw = usage.get(alt)
        else:
            raw = getattr(usage, key, None)
            if raw is None:
                raw = getattr(usage, alt, None)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _read_prompt_cache_hit_tokens() -> Optional[int]:
        direct = _read("prompt_cache_hit_tokens", "prompt_cache_hit")
        if direct is not None:
            return direct
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
        else:
            details = getattr(usage, "prompt_tokens_details", None)
        if isinstance(details, dict):
            raw = details.get("cached_tokens")
        else:
            raw = getattr(details, "cached_tokens", None) if details is not None else None
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _read_prompt_cache_miss_tokens() -> Optional[int]:
        miss = _read("prompt_cache_miss_tokens", "prompt_cache_miss")
        return miss

    prompt_tokens = _read("prompt_tokens", "input_tokens")
    completion_tokens = _read("completion_tokens", "output_tokens")
    prompt_cache_hit_tokens = _read_prompt_cache_hit_tokens()
    prompt_cache_miss_tokens = _read_prompt_cache_miss_tokens()

    if prompt_tokens is None and prompt_cache_hit_tokens is not None and prompt_cache_miss_tokens is not None:
        prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens
    if prompt_cache_miss_tokens is None and prompt_tokens is not None:
        if prompt_cache_hit_tokens is None:
            prompt_cache_miss_tokens = prompt_tokens
        else:
            prompt_cache_miss_tokens = max(0, prompt_tokens - prompt_cache_hit_tokens)
    if prompt_cache_hit_tokens is None:
        prompt_cache_hit_tokens = 0

    total_tokens = _read("total_tokens", "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "prompt_cache_hit_tokens": int(prompt_cache_hit_tokens or 0),
        "prompt_cache_miss_tokens": int(prompt_cache_miss_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


@dataclass
class TokenCounter:
    prompt_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: Optional[Mapping[str, int]]) -> None:
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.prompt_cache_hit_tokens += int(usage.get("prompt_cache_hit_tokens") or 0)
        self.prompt_cache_miss_tokens += int(usage.get("prompt_cache_miss_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        total = usage.get("total_tokens")
        if total is None:
            total = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(total or 0)


def _call_openai_chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    request_timeout_s: float = 60.0,
) -> Tuple[str, Optional[Dict[str, int]]]:
    _ensure_dotenv_loaded()

    model_norm = str(model or "").strip().lower()
    provider = "deepseek" if model_norm.startswith("deepseek") else "openai"
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required for DeepSeek LLM calls.")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or getattr(openai, "api_key", None)
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for LLM-based template generation.")

    if openai is None:
        url = str(base_url).rstrip("/") + "/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages},
                timeout=max(1.0, float(request_timeout_s)),
            )
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            raise LLMHTTPError(f"HTTP request failed: {exc}", status_code=503) from exc

        status = int(resp.status_code)
        if status < 200 or status >= 300:  # pragma: no cover - provider dependent
            raise LLMHTTPError(f"HTTP {status}: {resp.text}", status_code=status)
        payload = resp.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response has no choices.")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if content is None:
            raise RuntimeError("LLM response content is empty.")
        return str(content), _extract_usage(payload)

    client_cls = getattr(openai, "Client", None) or getattr(openai, "OpenAI", None)
    if client_cls is not None:
        try:
            client = client_cls(api_key=api_key, base_url=base_url)
        except TypeError:
            client = client_cls(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM response content is empty.")
        return content, _extract_usage(response)
    if hasattr(openai, "ChatCompletion"):
        openai.api_key = api_key
        if hasattr(openai, "api_base") and provider == "deepseek":
            openai.api_base = base_url  # type: ignore[attr-defined]
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return response["choices"][0]["message"]["content"], _extract_usage(response)
    raise RuntimeError("Unsupported openai client version.")


def _ensure_dict(value: object, *, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object (dict), got {type(value).__name__}")
    return value


def _fill_plan_fields(
    template: Dict[str, object],
    *,
    source: Mapping[str, object],
    fields: Iterable[str],
) -> None:
    plan = template.get("plan")
    if plan is None:
        plan = {}
        template["plan"] = plan
    plan_dict = _ensure_dict(plan, label="template.plan")

    for field in fields:
        if field not in source:
            continue
        plan_dict[field] = source[field]


def _derive_example_input_payload(
    example_payload: Optional[Mapping[str, object]],
    *,
    plan_fields: Sequence[str],
    task_name: str,
) -> Optional[Dict[str, object]]:
    if example_payload is None:
        return None
    example_plan = example_payload.get("plan")
    if not isinstance(example_plan, dict):
        return None
    payload = {field: example_plan.get(field) for field in plan_fields}
    payload["task_name"] = task_name
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based plan template initialiser.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name (defaults to travel).")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=("train", "validation", "test"),
        help="Which dataset split to use.",
    )
    parser.add_argument("--index", type=int, default=0, help="0-based row index in the split CSV.")
    parser.add_argument("--all", action="store_true", help="Generate templates for all rows in the split CSV.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit batch generation to the first N rows.")
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpu_count(),
        help="Parallel worker processes for --all (defaults to all available CPUs).",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per row on transient API failures.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.getenv("MEMPLAN_LLM_REQUEST_TIMEOUT_S", "60")),
        help="Per-request HTTP timeout (seconds) for LLM calls when using the `requests` fallback client.",
    )
    parser.add_argument(
        "--plan-fields",
        type=str,
        default="auto",
        help="Comma-separated row fields to copy into template.plan (or 'auto' to infer from schema.plan keys present in the dataset).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="When --all, reuse existing --out JSONL and only regenerate rows with errors/missing schema keys.",
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --all, retry failed rows (default: enabled).",
    )
    parser.add_argument(
        "--retry-workers",
        type=int,
        default=available_cpu_count()/2,
        help="Worker count for retry rounds (default: 1).",
    )
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=3,
        help="Number of retry rounds after the initial pass (0 = unlimited).",
    )
    parser.add_argument("--input", type=Path, help="Override input CSV path.")
    parser.add_argument("--prompt", type=Path, help="Override template prompt path.")
    parser.add_argument("--schema", type=Path, help="Override schema path (schema.jsonl).")
    parser.add_argument("--example", type=Path, help="Override example output path (example.jsonl).")
    parser.add_argument("--out", type=Path, help="Output JSONL path.")
    return parser.parse_args(argv)


def _generate_template_for_row(
    *,
    task: str,
    split: str,
    index: int,
    row: Dict[str, object],
    model: str,
    plan_fields: Sequence[str],
    prompt_path: Path,
    schema_path: Path,
    example_payload: Optional[Dict[str, object]],
    example_multicity_payload: Optional[Dict[str, object]],
    request_timeout_s: float,
) -> Tuple[Dict[str, object], Optional[Dict[str, int]]]:
    plan_id = str(row.get("plan_id") or row.get("id") or index)
    input_payload = {field: row.get(field) for field in plan_fields}
    input_payload["task_name"] = task

    schema_payload = _load_jsonl_object(schema_path)
    example_input_payload = _derive_example_input_payload(example_payload, plan_fields=plan_fields, task_name=task)
    example_multicity_input_payload = _derive_example_input_payload(
        example_multicity_payload, plan_fields=plan_fields, task_name=task
    )

    prompt = _render_prompt(
        prompt_path,
        replacements={
            "{{INPUT_JSON}}": input_payload,
            "{{SCHEMA_JSON}}": schema_payload,
            "{{EXAMPLE_JSON}}": example_payload or {},
            "{{EXAMPLE_INPUT_JSON}}": example_input_payload or {},
            "{{EXAMPLE_MULTICITY_JSON}}": example_multicity_payload or {},
            "{{EXAMPLE_MULTICITY_INPUT_JSON}}": example_multicity_input_payload or {},
        },
    )

    messages = [
        {"role": "system", "content": "You output valid JSON only. No markdown. No extra text."},
        {"role": "user", "content": prompt},
    ]

    content, usage = _call_openai_chat(model=model, messages=messages, request_timeout_s=request_timeout_s)
    try:
        try:
            template_obj = json.loads(content)
        except json.JSONDecodeError:
            template_obj = json.loads(_extract_json_block(content))
        template = _ensure_dict(template_obj, label="LLM template output")
    except Exception as exc:
        raise LLMOutputError(f"{type(exc).__name__}: {exc}", usage=usage) from exc
    template["task_name"] = task
    template["template_id"] = plan_id
    if usage is not None:
        notes = template.get("notes")
        if not isinstance(notes, dict):
            notes = {}
            template["notes"] = notes
        notes["llm"] = {"model": model, **usage}
    _fill_plan_fields(template, source=row, fields=plan_fields)
    return template, usage


def _generate_template_worker(
    payload: Mapping[str, object],
) -> Tuple[int, Dict[str, object], Optional[str], Optional[Dict[str, int]]]:
    task = str(payload["task"])
    split = str(payload["split"])
    index = int(payload["index"])
    row = dict(payload["row"])  # type: ignore[arg-type]
    plan_id = str(row.get("plan_id") or row.get("id") or index)
    model = str(payload["model"])
    plan_fields = list(payload["plan_fields"])  # type: ignore[arg-type]
    prompt_path = Path(str(payload["prompt_path"]))
    schema_path = Path(str(payload["schema_path"]))
    max_retries = int(payload.get("max_retries", 3))
    request_timeout_s = float(payload.get("request_timeout_s", 60.0))

    example_payload = None
    example_path = Path(str(payload["example_path"]))
    if example_path.exists():
        example_payload = _load_jsonl_object(example_path)

    example_multicity_payload = None
    example_multicity_path = payload.get("example_multicity_path")
    if example_multicity_path:
        example_multicity_payload = _load_jsonl_object(Path(str(example_multicity_path)))

    for attempt in range(max_retries + 1):
        try:
            template, usage = _generate_template_for_row(
                task=task,
                split=split,
                index=index,
                row=row,
                model=model,
                plan_fields=plan_fields,
                prompt_path=prompt_path,
                schema_path=schema_path,
                example_payload=example_payload,
                example_multicity_payload=example_multicity_payload,
                request_timeout_s=request_timeout_s,
            )
            return index, template, None, usage
        except Exception as exc:  # pragma: no cover - network/provider dependent
            message = f"{type(exc).__name__}: {exc}"
            usage = getattr(exc, "usage", None) if isinstance(exc, LLMOutputError) else None
            if attempt < max_retries and _is_retryable_error(exc):
                sleep_s = min(30.0, (2.0**attempt) + random.random())
                time.sleep(sleep_s)
                continue
            error_template: Dict[str, object] = {"error": message, "template_id": plan_id, "task_name": task}
            if isinstance(usage, dict):
                error_template["notes"] = {"llm": {"model": model, **usage}}
            return index, error_template, message, usage


def _template_needs_regen(
    template: object,
    *,
    schema: Optional[Mapping[str, object]] = None,
) -> bool:
    if not isinstance(template, dict):
        return True
    if template.get("error"):
        return True
    if schema is None:
        return False
    schema_keys = set(schema.keys())
    if not schema_keys.issubset(set(template.keys())):
        return True
    schema_plan = schema.get("plan")
    if isinstance(schema_plan, dict):
        plan = template.get("plan")
        if not isinstance(plan, dict):
            return True
        if not set(schema_plan.keys()).issubset(set(plan.keys())):
            return True
    return False


def _load_existing_templates(path: Path, *, expected_len: int) -> List[Optional[Dict[str, object]]]:
    templates: List[Optional[Dict[str, object]]] = [None] * expected_len
    if not path.exists():
        return templates
    for idx, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()[:expected_len]):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            templates[idx] = obj
    return templates


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    task = str(args.task or DEFAULT_TASK)

    input_dir = (INPUT_ROOT / task).resolve()
    output_dir = (OUTPUT_ROOT / task).resolve()

    default_csv = input_dir / "dataset" / f"{args.split}.csv"
    if not default_csv.exists():
        default_csv = input_dir / f"{args.split}.csv"
    csv_path = args.input or default_csv
    if not csv_path.is_absolute():
        candidate = (input_dir / csv_path).resolve()
        csv_path = candidate if candidate.exists() else (PROJECT_ROOT / csv_path).resolve()
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    prompt_path = args.prompt or (input_dir / "template.txt")
    if not prompt_path.is_absolute():
        candidate = (input_dir / prompt_path).resolve()
        prompt_path = candidate if candidate.exists() else (PROJECT_ROOT / prompt_path).resolve()
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")

    schema_path = args.schema or (input_dir / "schema.jsonl")
    if not schema_path.is_absolute():
        candidate = (input_dir / schema_path).resolve()
        schema_path = candidate if candidate.exists() else (PROJECT_ROOT / schema_path).resolve()
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    example_path = args.example or (input_dir / "example.jsonl")
    if not example_path.is_absolute():
        candidate = (input_dir / example_path).resolve()
        example_path = candidate if candidate.exists() else (PROJECT_ROOT / example_path).resolve()
    example_payload: Optional[Dict[str, object]] = None
    if example_path.exists():
        example_payload = _load_jsonl_object(example_path)

    example_multicity_path = input_dir / "example_multicity.jsonl"
    example_multicity_payload: Optional[Dict[str, object]] = None
    if example_multicity_path.exists():
        example_multicity_payload = _load_jsonl_object(example_multicity_path)

    if args.out is not None:
        out_path = args.out
    elif args.all:
        out_path = output_dir / f"init_templates_{args.split}_{_model_slug(str(args.model))}.jsonl"
    else:
        out_path = output_dir / "init_template.jsonl"
    if not out_path.is_absolute():
        if out_path.parent == Path("."):
            out_path = (output_dir / out_path).resolve()
        else:
            out_path = (PROJECT_ROOT / out_path).resolve()

    raw_plan_fields = str(args.plan_fields).strip()
    plan_fields_auto = raw_plan_fields.lower() == "auto"
    plan_fields: List[str] = []
    if not plan_fields_auto:
        plan_fields = [item.strip() for item in raw_plan_fields.split(",") if item.strip()]
        if not plan_fields:
            raise SystemExit("--plan-fields must include at least one field name (or use --plan-fields auto).")

    if args.all:
        rows = _read_csv_rows(csv_path)
        for row in rows:
            _normalise_row_inplace(row)
        if args.max_rows is not None:
            rows = rows[: max(0, int(args.max_rows))]
        if not rows:
            raise SystemExit(f"No rows found in {csv_path}")

        schema_payload = _load_jsonl_object(schema_path)
        if plan_fields_auto:
            schema_plan = schema_payload.get("plan")
            if not isinstance(schema_plan, dict) or not schema_plan:
                raise SystemExit("--plan-fields auto requires schema.plan to be a JSON object with at least one key.")
            candidates = [str(k) for k in schema_plan.keys()]
            present = {str(k) for row in rows for k in row.keys()}
            plan_fields = [field for field in candidates if field in present]
            if not plan_fields:
                raise SystemExit("Could not infer any plan fields from schema.plan keys present in the dataset.")
        templates: List[Optional[Dict[str, object]]] = [None] * len(rows)
        if bool(getattr(args, "resume", False)):
            templates = _load_existing_templates(out_path, expected_len=len(rows))

        pending = [
            idx
            for idx, template in enumerate(templates)
            if template is None or _template_needs_regen(template, schema=schema_payload)
        ]

        requested_workers = int(args.workers) if args.workers is not None else 1
        worker_count = max(1, min(requested_workers, max(1, len(pending))))

        token_counter = TokenCounter()
        errors: List[str] = []
        retry_enabled = bool(args.retry_errors)
        retry_workers = max(1, int(args.retry_workers))
        retry_rounds = int(args.retry_rounds)
        request_timeout_s = float(args.request_timeout_s)

        if args.resume and out_path.exists():
            done = len(rows) - len(pending)
            print(f"Resuming from {out_path}: {done}/{len(rows)} template(s) already present.")

        round_idx = 0
        while pending:
            if round_idx == 0:
                current_workers = worker_count
                label = "Generating"
            else:
                current_workers = min(retry_workers, len(pending))
                label = f"Retrying (round {round_idx})"

            print(
                f"{label} {len(pending)} template(s) with {current_workers} workers "
                f"(model={args.model}, timeout={request_timeout_s}s)."
            )

            payloads: List[Dict[str, object]] = []
            for idx in pending:
                payloads.append(
                    {
                        "task": task,
                        "split": str(args.split),
                        "index": idx,
                        "row": rows[idx],
                        "model": str(args.model),
                        "plan_fields": plan_fields,
                        "prompt_path": str(prompt_path),
                        "schema_path": str(schema_path),
                        "example_path": str(example_path),
                        "example_multicity_path": str(example_multicity_path) if example_multicity_path.exists() else None,
                        "max_retries": int(args.max_retries),
                        "request_timeout_s": request_timeout_s,
                    }
                )

            with concurrent.futures.ProcessPoolExecutor(max_workers=current_workers) as executor:
                future_to_index = {
                    executor.submit(_generate_template_worker, payload): int(payload["index"]) for payload in payloads
                }
                completed = 0
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        _, template, error, usage = future.result()
                    except Exception as exc:  # pragma: no cover - unexpected multiprocessing errors
                        template = {
                            "error": f"{type(exc).__name__}: {exc}",
                            "template_id": f"{args.split}:{idx}",
                            "task_name": task,
                        }
                        error = f"{type(exc).__name__}: {exc}"
                        usage = None

                    token_counter.add(usage)
                    templates[idx] = template
                    completed += 1
                    if error:
                        errors.append(f"{idx}: {error}")
                        print(f"Processing query {args.split}:{idx}... ERROR ({completed}/{len(pending)})", flush=True)
                    else:
                        print(f"Processing query {args.split}:{idx}... OK ({completed}/{len(pending)})", flush=True)

            pending = [
                idx
                for idx, template in enumerate(templates)
                if template is None or _template_needs_regen(template, schema=schema_payload)
            ]

            if not pending:
                break
            if not retry_enabled:
                break

            round_idx += 1
            if retry_rounds > 0 and round_idx > retry_rounds:
                break

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            for template in templates:
                fp.write(json.dumps(template or {}, ensure_ascii=False) + "\n")

        if pending:
            raise SystemExit(
                f"Wrote templates to {out_path}, but {len(pending)} template(s) still contain errors "
                f"after {round_idx} retry round(s): {', '.join(str(i) for i in pending[:10])}"
            )

        print(f"Wrote templates to {out_path}")
        if token_counter.calls:
            print(
                f"LLM token usage (init_template): calls={token_counter.calls} "
                f"prompt_cache_hit={token_counter.prompt_cache_hit_tokens} "
                f"prompt_cache_miss={token_counter.prompt_cache_miss_tokens} "
                f"output={token_counter.completion_tokens} total={token_counter.total_tokens}",
                flush=True,
            )
        return

    row = _read_csv_row(csv_path, index=int(args.index))
    _normalise_row_inplace(row)
    if plan_fields_auto:
        schema_payload = _load_jsonl_object(schema_path)
        schema_plan = schema_payload.get("plan")
        if not isinstance(schema_plan, dict) or not schema_plan:
            raise SystemExit("--plan-fields auto requires schema.plan to be a JSON object with at least one key.")
        candidates = [str(k) for k in schema_plan.keys()]
        plan_fields = [field for field in candidates if field in row]
        if not plan_fields:
            raise SystemExit("Could not infer any plan fields from schema.plan keys present in the dataset row.")

    template, usage = _generate_template_for_row(
        task=task,
        split=str(args.split),
        index=int(args.index),
        row=row,
        model=str(args.model),
        plan_fields=plan_fields,
        prompt_path=prompt_path,
        schema_path=schema_path,
        example_payload=example_payload,
        example_multicity_payload=example_multicity_payload,
        request_timeout_s=float(args.request_timeout_s),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(template, ensure_ascii=False) + "\n")

    print(f"Wrote template to {out_path}")
    if usage is not None:
        print(
            f"LLM token usage (init_template): "
            f"prompt_cache_hit={usage.get('prompt_cache_hit_tokens', 0)} "
            f"prompt_cache_miss={usage.get('prompt_cache_miss_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} total={usage.get('total_tokens', 0)}",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""
LLM-based repair stage (optional).

This module is intended to run *after* MCTS (`planner/twin_track.py`):

  tree_<split>.json  --(MCTS)-->  optimized_<split>.jsonl  --(LLM repair)-->  repaired_<split>_<model>.jsonl

The LLM is used as a constrained "fixer": it can only propose replacing the
selected candidate for an OR-slot using candidates already present in the
AND–OR tree.  No new entities may be invented.

Task-specific prompt templates live under:
  artifacts/input/<task>/repair_prompt.txt

How to add a new task:
  1) Provide `artifacts/input/<task>/repair_prompt.txt` with `{{REPAIR_INPUT_JSON}}`.
  2) (Optional) Provide `artifacts/input/<task>/repair.json` with constraint→action mappings, prompt overrides, etc.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import requests

from planner.parallel import available_cpu_count

try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None
try:
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tiktoken = None

from planner.twin_track import MCTSConfig, load_task_spec, render_filled_template
from planner.twin_track import _parse_tree_entry  # type: ignore
from planner.config import ConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"
_LOG_PATH = ARTIFACTS_ROOT / "logs" / "log.txt"

DEFAULT_TASK = "travel"
DEFAULT_MODEL = os.getenv("MEMPLAN_LLM_MODEL", "gpt-5-mini")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_MODEL_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_DOTENV_LOADED = False
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is\s+(?P<max>\d+)\s+tokens.*?requested\s+(?P<requested>\d+)\s+tokens",
    flags=re.IGNORECASE | re.DOTALL,
)


def _append_run_log(
    *,
    argv: List[str],
    started_at: float,
    status: str,
    extra: Optional[Dict[str, object]] = None,
    error: Optional[str] = None,
) -> None:
    try:
        payload: Dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": "planner.llm_repair",
            "status": status,
            "elapsed_s": round(time.perf_counter() - started_at, 6),
            "argv": list(argv),
        }
        if extra:
            payload.update(dict(extra))
        if error:
            payload["error"] = error
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _model_slug(model_name: str) -> str:
    slug = _MODEL_SLUG_RE.sub("", model_name).strip().lower()
    return slug or "model"


def _read_optional_json_object(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _split_from_tree_path(path: Path) -> Optional[str]:
    stem = path.stem
    match = re.match(r"^tree_(?P<split>[^.]+)$", stem)
    if match:
        return match.group("split")
    return None


def _split_from_optimized_path(path: Path) -> Optional[str]:
    stem = path.stem
    match = re.match(r"^optimized_(?P<split>[^.]+)$", stem)
    if match:
        return match.group("split")
    return None


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


class LLMHTTPError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _extract_json_block(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("LLM did not return JSON content.")
    return match.group(0)

def _encoding_for_model(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _count_text_tokens(text: str, *, model: str) -> int:
    enc = _encoding_for_model(model)
    if enc is None:
        return 0
    return len(enc.encode(str(text)))


def _truncate_text_to_tokens(text: str, *, model: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _encoding_for_model(model)
    if enc is None:
        # Conservative fallback when we cannot tokenise:
        # many JSON-heavy prompts average ~2-4 chars/token depending on whitespace;
        # use 3 chars/token to reduce the chance of still exceeding provider limits.
        approx_chars = max(0, int(max_tokens) * 3)
        return str(text)[:approx_chars]
    tokens = enc.encode(str(text))
    if len(tokens) <= max_tokens:
        return str(text)
    return enc.decode(tokens[:max_tokens])


def _extract_context_limit(exc: Exception) -> Tuple[Optional[int], Optional[int]]:
    message = str(exc)
    match = _CONTEXT_LENGTH_RE.search(message)
    if not match:
        return None, None
    try:
        max_ctx = int(match.group("max"))
    except Exception:
        max_ctx = None
    try:
        requested = int(match.group("requested"))
    except Exception:
        requested = None
    return max_ctx, requested


def _truncate_messages_to_context_limit(
    *,
    messages: List[Dict[str, str]],
    model: str,
    max_context_tokens: int,
    margin_tokens: int = 8192,
) -> List[Dict[str, str]]:
    if max_context_tokens <= 0:
        return messages
    copied = [dict(m) for m in messages]
    user_idx = None
    for i in range(len(copied) - 1, -1, -1):
        if copied[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return copied

    # Keep everything the same as the first try (roles/messages), only truncate the
    # user message content to fit within the provider's max context length.
    other_tokens = 0
    for i, msg in enumerate(copied):
        if i == user_idx:
            continue
        other_tokens += _count_text_tokens(str(msg.get("content") or ""), model=model)
    allowed_user_tokens = max(0, int(max_context_tokens) - int(other_tokens) - int(margin_tokens))
    copied[user_idx]["content"] = _truncate_text_to_tokens(
        str(copied[user_idx].get("content") or ""),
        model=model,
        max_tokens=allowed_user_tokens,
    )
    return copied

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


def _call_openai_chat(*, model: str, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, int]]]:
    _ensure_dotenv_loaded()

    model_norm = str(model or "").strip().lower()
    provider = "deepseek" if model_norm.startswith("deepseek") else "openai"
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required for DeepSeek LLM repair.")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or getattr(openai, "api_key", None)
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for LLM repair.")

    if openai is None:
        url = str(base_url).rstrip("/") + "/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages},
                timeout=60,
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


def _render_prompt(prompt_path: Path, *, repair_input: Mapping[str, object]) -> str:
    rendered = prompt_path.read_text(encoding="utf-8").replace(
        "{{REPAIR_INPUT_JSON}}",
        json.dumps(repair_input, ensure_ascii=False, indent=2),
    )
    if "{{" in rendered:
        raise ValueError(f"Unreplaced placeholder found in prompt template: {prompt_path}")
    return rendered


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSON object per line in {path}")
            yield obj


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number) or math.isinf(number):
        return float(default)
    return float(number)


def _format_constraint_table(
    constraints: Mapping[str, Tuple[Optional[bool], Optional[str]]],
    *,
    only_failed: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, (flag, message) in constraints.items():
        if only_failed and flag is not False:
            continue
        rows.append({"name": name, "pass": flag, "message": message})
    return rows


@dataclass(frozen=True)
class CandidateSummaryConfig:
    include_meta_keys: Tuple[str, ...] = ("type",)
    max_row_keys: int = 50
    max_value_chars: int = 200
    row_keys_by_action_type: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    row_keys_default: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairConfig:
    task_name: str
    prompt_path: Path
    constraint_to_action_types: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    always_include_action_types: Tuple[str, ...] = ()
    unknown_constraint_policy: str = "include_all_slots"  # "include_all_slots" | "ignore"
    candidate_summary: CandidateSummaryConfig = field(default_factory=CandidateSummaryConfig)


def _parse_str_tuple(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        raw = list(value)
    elif isinstance(value, list):
        raw = value
    else:
        return ()
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return tuple(out)


def _load_repair_config(task: str, *, config_path: Optional[Path]) -> RepairConfig:
    default_prompt = INPUT_ROOT / task / "repair_prompt.txt"
    default_config = INPUT_ROOT / task / "repair.json"

    config_payload: Dict[str, object] = {}
    if config_path is None:
        config_path = default_config
    config_path = _resolve_path(config_path)
    if config_path.exists():
        loaded = _read_optional_json_object(config_path)
        config_payload = loaded

    prompt_path = config_payload.get("prompt_path")
    prompt_path = Path(prompt_path) if isinstance(prompt_path, str) and prompt_path else default_prompt
    prompt_path = _resolve_path(prompt_path)

    constraint_to_action_types: Dict[str, Tuple[str, ...]] = {}
    raw_mapping = config_payload.get("constraint_to_action_types")
    if isinstance(raw_mapping, dict):
        for constraint_name, action_types in raw_mapping.items():
            if not isinstance(constraint_name, str) or not constraint_name.strip():
                continue
            parsed = _parse_str_tuple(action_types)
            if parsed:
                constraint_to_action_types[constraint_name.strip()] = parsed

    always_include_action_types = _parse_str_tuple(config_payload.get("always_include_action_types"))
    unknown_constraint_policy = config_payload.get("unknown_constraint_policy")
    if not isinstance(unknown_constraint_policy, str) or not unknown_constraint_policy.strip():
        unknown_constraint_policy = "include_all_slots"
    unknown_constraint_policy = unknown_constraint_policy.strip()
    if unknown_constraint_policy not in {"include_all_slots", "ignore"}:
        unknown_constraint_policy = "include_all_slots"

    summary_payload = config_payload.get("candidate_summary")
    summary_dict = summary_payload if isinstance(summary_payload, dict) else {}
    include_meta_keys = _parse_str_tuple(summary_dict.get("include_meta_keys"))
    if "include_meta_keys" not in summary_dict:
        include_meta_keys = CandidateSummaryConfig.include_meta_keys

    max_row_keys = summary_dict.get("max_row_keys")
    if not isinstance(max_row_keys, int) or max_row_keys <= 0:
        max_row_keys = CandidateSummaryConfig.max_row_keys

    max_value_chars = summary_dict.get("max_value_chars")
    if not isinstance(max_value_chars, int) or max_value_chars <= 0:
        max_value_chars = CandidateSummaryConfig.max_value_chars

    row_keys_default = _parse_str_tuple(summary_dict.get("row_keys_default"))

    row_keys_by_action_type: Dict[str, Tuple[str, ...]] = {}
    raw_row_mapping = summary_dict.get("row_keys_by_action_type")
    if isinstance(raw_row_mapping, dict):
        for action_type, keys in raw_row_mapping.items():
            if not isinstance(action_type, str) or not action_type.strip():
                continue
            parsed = _parse_str_tuple(keys)
            if parsed:
                row_keys_by_action_type[action_type.strip()] = parsed

    summary_config = CandidateSummaryConfig(
        include_meta_keys=include_meta_keys,
        max_row_keys=max_row_keys,
        max_value_chars=max_value_chars,
        row_keys_by_action_type=row_keys_by_action_type,
        row_keys_default=row_keys_default,
    )

    return RepairConfig(
        task_name=task,
        prompt_path=prompt_path,
        constraint_to_action_types=constraint_to_action_types,
        always_include_action_types=always_include_action_types,
        unknown_constraint_policy=unknown_constraint_policy,
        candidate_summary=summary_config,
    )


# ---------------------------------------------------------------------------
# WorkBench call-list patch repair (work task)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkPatchConfig:
    prompt_path: Path
    max_patch_ops: int = 12
    allow_insert_actions: bool = True
    allow_delete_actions: bool = True
    allow_move_actions: bool = True
    canonical_values: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    id_param_keys: Tuple[str, ...] = ("event_id", "email_id", "task_id", "customer_id", "visitor_id")


@dataclass(frozen=True)
class WorkRepairContext:
    candidate_ids: Dict[str, List[str]]
    retrieved_records: Dict[str, List[Dict[str, object]]]
    retrieved_checks: List[Dict[str, object]]
    bindings: Dict[str, object]
    check_results: List[Dict[str, object]]


def _parse_tuple_mapping_str(value: object) -> Dict[str, Tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Tuple[str, ...]] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            continue
        out[k.strip()] = _parse_str_tuple(v)
    return out


def _load_work_patch_config(task: str, *, config_path: Optional[Path]) -> WorkPatchConfig:
    default_prompt = INPUT_ROOT / task / "repair_prompt.txt"
    default_config = INPUT_ROOT / task / "repair.json"

    cfg_path = _resolve_path(config_path) if config_path is not None else _resolve_path(default_config)
    payload = _read_optional_json_object(cfg_path) if cfg_path.exists() else {}

    prompt_path = payload.get("prompt_path")
    prompt_path = Path(prompt_path) if isinstance(prompt_path, str) and prompt_path else default_prompt
    prompt_path = _resolve_path(prompt_path)

    max_patch_ops = payload.get("max_patch_ops")
    if not isinstance(max_patch_ops, int) or max_patch_ops <= 0:
        max_patch_ops = 12

    allow_insert_actions = payload.get("allow_insert_actions")
    if not isinstance(allow_insert_actions, bool):
        allow_insert_actions = True

    allow_delete_actions = payload.get("allow_delete_actions")
    if not isinstance(allow_delete_actions, bool):
        allow_delete_actions = True

    allow_move_actions = payload.get("allow_move_actions")
    if not isinstance(allow_move_actions, bool):
        allow_move_actions = True

    canonical_values = _parse_tuple_mapping_str(payload.get("canonical_values"))
    id_param_keys = _parse_str_tuple(payload.get("id_param_keys")) or WorkPatchConfig.id_param_keys

    return WorkPatchConfig(
        prompt_path=prompt_path,
        max_patch_ops=int(max_patch_ops),
        allow_insert_actions=bool(allow_insert_actions),
        allow_delete_actions=bool(allow_delete_actions),
        allow_move_actions=bool(allow_move_actions),
        canonical_values=canonical_values,
        id_param_keys=id_param_keys,
    )


def _work_load_tool_schema_index() -> Dict[str, Dict[str, object]]:
    tool_view_path = INPUT_ROOT / "work" / "views" / "tool.json"
    payload = _read_optional_json_object(tool_view_path) if tool_view_path.exists() else {}
    views = payload.get("views")
    if not isinstance(views, list):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for item in views:
        if not isinstance(item, dict):
            continue
        attrs = item.get("node_pattern") if isinstance(item.get("node_pattern"), dict) else {}
        attrs = attrs.get("attrs") if isinstance(attrs.get("attrs"), dict) else {}
        qualified = attrs.get("qualified_tool_name")
        if not isinstance(qualified, str) or not qualified.strip():
            continue
        params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        out[qualified.strip()] = {
            "tool_name": attrs.get("tool_name"),
            "action_type": attrs.get("action_type"),
            "side_effect": bool(attrs.get("side_effect")) if isinstance(attrs.get("side_effect"), bool) else False,
            "param_keys": sorted([str(k) for k in params.keys() if isinstance(k, str) and k.strip()]),
        }
    return out


def _work_struct_to_action_dict(
    *,
    phase_key: str,
    qualified_tool_name: str,
    params: Mapping[str, object],
    tool_schemas: Mapping[str, Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    qualified = str(qualified_tool_name or "").strip()
    if not qualified or "." not in qualified:
        return None
    schema = tool_schemas.get(qualified) if isinstance(tool_schemas.get(qualified), Mapping) else None
    if schema is None:
        return None
    tool_name, action_type = qualified.split(".", 1)
    side_effect = bool(schema.get("side_effect", False))
    clean_params = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
    attrs: Dict[str, object] = {
        "tool_name": tool_name,
        "qualified_tool_name": qualified,
        "side_effect": side_effect,
        "raw_action": "",
    }
    action: Dict[str, object] = {
        "phase_key": str(phase_key),
        "action_type": action_type,
        "params": clean_params,
        "attrs": attrs,
        "order_index": 0,
    }
    # Render a call string for downstream evaluation/conversion.
    rendered = _work_action_dict_to_call(action)
    attrs["raw_action"] = rendered
    action["attrs"] = attrs
    return action


def _work_action_dict_to_call(action: Mapping[str, object]) -> str:
    attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
    raw = attrs.get("raw_action")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    qualified = attrs.get("qualified_tool_name")
    if not isinstance(qualified, str) or not qualified.strip():
        tool_name = attrs.get("tool_name")
        action_type = action.get("action_type")
        if isinstance(tool_name, str) and isinstance(action_type, str) and tool_name and action_type:
            qualified = f"{tool_name}.{action_type}"
        else:
            qualified = ""

    def _quote(v: object) -> str:
        text = str(v) if v is not None else ""
        text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{text}"'

    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    parts = []
    for key, value in params.items():
        if value is None:
            continue
        parts.append(f"{key}={_quote(value)}")
    return f"{qualified}.func(" + ", ".join(parts) + ")"


def _canonical_workbench_query(query: str) -> str:
    return str(query).strip()


def _load_workbench_ground_truth(*, ground_truth_dir: Path) -> Dict[str, List[str]]:
    """Load WorkBench ground truth mapping: canonical query -> list[str] of tool calls."""

    gt_by_query: Dict[str, List[str]] = {}
    if not ground_truth_dir.exists():
        raise SystemExit(f"WorkBench ground truth dir not found: {ground_truth_dir}")

    for path in sorted(ground_truth_dir.glob("*_queries_and_answers.csv")):
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames:
                continue
            if "query" not in reader.fieldnames or "answer" not in reader.fieldnames:
                continue
            for row in reader:
                query = _canonical_workbench_query(row.get("query") or "")
                raw_answer = row.get("answer") or ""
                if not query:
                    continue
                try:
                    parsed = ast.literal_eval(raw_answer)
                except Exception:
                    parsed = []
                if not isinstance(parsed, list):
                    parsed = []
                actions = [str(x) for x in parsed if str(x)]
                if query in gt_by_query:
                    raise SystemExit(f"Duplicate WorkBench query across ground truth files: {query!r}")
                gt_by_query[query] = actions

    if not gt_by_query:
        raise SystemExit(f"No *_queries_and_answers.csv files found under {ground_truth_dir}")
    return gt_by_query


def _workbench_metrics_for_work_record(
    *,
    record: Mapping[str, object],
    ground_truth_by_query: Mapping[str, Sequence[str]],
) -> Optional[Tuple[bool, bool]]:
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    query = _canonical_workbench_query(str(plan.get("query") or ""))
    if not query:
        return None
    ground_truth = ground_truth_by_query.get(query)
    if not ground_truth:
        return None

    actions = record.get("actions") if isinstance(record.get("actions"), list) else []
    calls = [_work_action_dict_to_call(a) for a in actions if isinstance(a, dict)]

    try:
        from task_helper.work.evaluation import utils as wb_utils
    except Exception as exc:
        raise SystemExit(f"Failed to import WorkBench evaluation utils (needed for --workbench-only-incorrect): {exc}")

    correct = bool(wb_utils.is_correct(calls, list(ground_truth), error=""))
    unwanted_side_effects = bool(wb_utils.has_side_effects(calls, list(ground_truth)))
    return correct, unwanted_side_effects


def _work_validate_actions(
    *,
    actions: Sequence[Mapping[str, object]],
    tool_schemas: Mapping[str, Mapping[str, object]],
    config: WorkPatchConfig,
    candidate_ids: Mapping[str, Sequence[str]],
) -> Optional[str]:
    required_by_tool: Dict[str, Tuple[str, ...]] = {
        # Analytics (treat time bounds as required for meaningful checks/plots)
        "analytics.create_plot": ("time_min", "time_max", "value_to_plot", "plot_type"),
        "analytics.total_visits_count": ("time_min", "time_max"),
        "analytics.engaged_users_count": ("time_min", "time_max"),
        "analytics.get_average_session_duration": ("time_min", "time_max"),
        "analytics.traffic_source_count": ("time_min", "time_max", "traffic_source"),
        "analytics.get_visitor_information_by_id": ("visitor_id",),
        # Calendar
        "calendar.get_event_information_by_id": ("event_id", "field"),
        "calendar.create_event": ("event_name", "participant_email", "event_start", "duration"),
        "calendar.delete_event": ("event_id",),
        "calendar.update_event": ("event_id", "field", "new_value"),
        # Email
        "email.get_email_information_by_id": ("email_id", "field"),
        "email.send_email": ("recipient", "subject", "body"),
        "email.delete_email": ("email_id",),
        "email.forward_email": ("email_id", "recipient"),
        "email.reply_email": ("email_id", "body"),
        # Project management
        "project_management.get_task_information_by_id": ("task_id", "field"),
        "project_management.create_task": ("task_name", "assigned_to_email", "list_name", "due_date", "board"),
        "project_management.delete_task": ("task_id",),
        "project_management.update_task": ("task_id", "field", "new_value"),
        # CRM
        "customer_relationship_manager.update_customer": ("customer_id", "field", "new_value"),
        "customer_relationship_manager.add_customer": ("customer_name", "assigned_to_email", "status"),
        "customer_relationship_manager.delete_customer": ("customer_id",),
        # Directory
        "company_directory.find_email_address": ("name",),
    }

    require_any_of: Dict[str, Tuple[Tuple[str, ...], ...]] = {
        "calendar.search_events": (("query", "time_min", "time_max"),),
        "email.search_emails": (("query", "date_min", "date_max"),),
        "project_management.search_tasks": (("task_name", "assigned_to_email", "list_name", "due_date", "board"),),
        "customer_relationship_manager.search_customers": (
            (
                "customer_name",
                "customer_email",
                "product_interest",
                "status",
                "assigned_to_email",
                "last_contact_date_min",
                "last_contact_date_max",
                "follow_up_by_min",
                "follow_up_by_max",
            ),
        ),
    }

    for idx, a in enumerate(actions):
        if not isinstance(a, dict):
            return f"action[{idx}] not dict"
        phase_key = str(a.get("phase_key") or "")
        phase_kind = "check" if phase_key.lower().startswith("check") else ("do" if phase_key.lower().startswith("do") else "")
        if not phase_kind:
            return f"action[{idx}] invalid phase_key"
        attrs = a.get("attrs") if isinstance(a.get("attrs"), dict) else {}
        qualified = str(attrs.get("qualified_tool_name") or "")
        if qualified not in tool_schemas:
            return f"action[{idx}] unknown tool {qualified}"
        schema = tool_schemas[qualified]
        schema_side_effect = bool(schema.get("side_effect", False))
        if phase_kind == "check" and schema_side_effect:
            return f"action[{idx}] check phase has side-effect tool"
        if phase_kind == "do" and not schema_side_effect:
            return f"action[{idx}] do phase has non-side-effect tool"

        allowed_keys = set(schema.get("param_keys") or [])
        params = a.get("params") if isinstance(a.get("params"), dict) else {}
        for key in params.keys():
            if key not in allowed_keys:
                return f"action[{idx}] unexpected param {key}"

        required_keys = required_by_tool.get(qualified)
        if required_keys is None:
            # Default: require all params for side-effect tools; allow sparse params for check tools.
            required_keys = tuple(sorted(allowed_keys)) if schema_side_effect else ()
        for key in required_keys:
            if key not in params:
                return f"action[{idx}] missing param {key}"
            val = params.get(key)
            if not isinstance(val, str) or not str(val).strip():
                return f"action[{idx}] empty param {key}"

        any_groups = require_any_of.get(qualified, ())
        for group in any_groups:
            if not any(isinstance(params.get(k), str) and str(params.get(k)).strip() for k in group):
                return f"action[{idx}] needs at least one of {list(group)}"

        for canon_key, allowed in config.canonical_values.items():
            param_key = canon_key.split(".")[-1] if "." in canon_key else canon_key
            if param_key in params:
                val = str(params.get(param_key) or "")
                if allowed and val not in set(allowed):
                    return f"action[{idx}] param {param_key} not canonical"

        for id_key in config.id_param_keys:
            if id_key in params:
                ids = [str(v).strip() for v in (candidate_ids.get(id_key) or []) if str(v).strip()]
                id_val = str(params.get(id_key) or "").strip()
                if not id_val:
                    return f"action[{idx}] empty param {id_key}"
                if not ids:
                    return f"action[{idx}] {id_key} not available"
                if id_val not in set(ids):
                    return f"action[{idx}] {id_key} not in candidate_ids"

    return None


def _work_apply_patch(
    *,
    base_actions: Sequence[Mapping[str, object]],
    patch_ops: Sequence[Mapping[str, object]],
    tool_schemas: Mapping[str, Mapping[str, object]],
    config: WorkPatchConfig,
) -> Optional[List[Dict[str, object]]]:
    actions: List[Dict[str, object]] = [deepcopy(a) for a in base_actions if isinstance(a, dict)]
    for raw_op in patch_ops[: int(config.max_patch_ops)]:
        if not isinstance(raw_op, dict):
            continue
        op = str(raw_op.get("op") or "").strip().lower()
        if op == "delete":
            if not config.allow_delete_actions:
                continue
            try:
                index = int(raw_op.get("index"))  # type: ignore[arg-type]
            except Exception:
                continue
            if 0 <= index < len(actions):
                del actions[index]
        elif op == "move":
            if not config.allow_move_actions:
                continue
            try:
                src = int(raw_op.get("from"))  # type: ignore[arg-type]
                dst = int(raw_op.get("to"))  # type: ignore[arg-type]
            except Exception:
                continue
            if not (0 <= src < len(actions)):
                continue
            item = actions.pop(src)
            dst = max(0, min(dst, len(actions)))
            actions.insert(dst, item)
        elif op == "insert":
            if not config.allow_insert_actions:
                continue
            action_obj = raw_op.get("action")
            if not isinstance(action_obj, dict):
                continue
            phase_key = action_obj.get("phase_key")
            qualified = action_obj.get("qualified_tool_name")
            params = action_obj.get("params")
            if not isinstance(phase_key, str) or not phase_key.strip():
                continue
            if not isinstance(qualified, str) or not qualified.strip():
                continue
            if not isinstance(params, dict):
                params = {}
            try:
                index = int(raw_op.get("index"))  # type: ignore[arg-type]
            except Exception:
                continue
            index = max(0, min(index, len(actions)))
            action = _work_struct_to_action_dict(
                phase_key=phase_key.strip(),
                qualified_tool_name=qualified.strip(),
                params=params,
                tool_schemas=tool_schemas,
            )
            if not isinstance(action, dict):
                continue
            actions.insert(index, action)
        elif op == "update":
            try:
                index = int(raw_op.get("index"))  # type: ignore[arg-type]
            except Exception:
                continue
            if not (0 <= index < len(actions)):
                continue
            set_phase_key = raw_op.get("set_phase_key")
            if isinstance(set_phase_key, str) and set_phase_key.strip():
                actions[index]["phase_key"] = set_phase_key.strip()
            set_params = raw_op.get("set_params")
            if isinstance(set_params, dict):
                params = actions[index].get("params") if isinstance(actions[index].get("params"), dict) else {}
                params = dict(params)
                attrs = actions[index].get("attrs") if isinstance(actions[index].get("attrs"), dict) else {}
                qualified = str(attrs.get("qualified_tool_name") or "")
                schema = tool_schemas.get(qualified) if qualified in tool_schemas else None
                if schema is None:
                    continue
                allowed_keys = set(schema.get("param_keys") or [])
                for k, v in set_params.items():
                    if not isinstance(k, str) or k not in allowed_keys:
                        continue
                    if v is None:
                        continue
                    params[k] = str(v)
                actions[index]["params"] = params
                attrs = dict(attrs)
                attrs["raw_action"] = ""
                attrs["raw_action"] = _work_action_dict_to_call(
                    {"attrs": attrs, "action_type": actions[index].get("action_type"), "params": params}
                )
                actions[index]["attrs"] = attrs

    for i, a in enumerate(actions):
        if isinstance(a, dict):
            a["order_index"] = i
    return actions


def _work_collect_repair_context(
    *,
    record: Mapping[str, object],
    tree: Mapping[str, object],
    tool_schemas: Mapping[str, Mapping[str, object]],
    config: WorkPatchConfig,
) -> WorkRepairContext:
    candidate_ids: Dict[str, List[str]] = {}
    retrieved_records: Dict[str, List[Dict[str, object]]] = {}
    retrieved_seen: Dict[str, Set[str]] = {}
    retrieved_checks: List[Dict[str, object]] = []
    bindings: Dict[str, object] = {}
    check_results: List[Dict[str, object]] = []

    def _add_id(key: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        if value.startswith("{") and value.endswith("}"):
            return
        candidate_ids.setdefault(key, [])
        if value not in candidate_ids[key]:
            candidate_ids[key].append(value)

    def _truncate_for_prompt(value: object, *, max_chars: int = 240) -> object:
        if isinstance(value, str):
            text = value.strip()
            if len(text) <= max_chars:
                return text
            return text[: max(0, max_chars - 1)] + "…"
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        try:
            encoded = json.dumps(value, ensure_ascii=False)
        except Exception:
            encoded = str(value)
        if len(encoded) <= max_chars:
            return value
        return encoded[: max(0, max_chars - 1)] + "…"

    def _compact_result_for_prompt(value: object, *, max_items: int = 80, max_chars: int = 4000) -> object:
        if isinstance(value, str):
            return _truncate_for_prompt(value, max_chars=max_chars)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            head = value[:max_items]
            if len(value) > max_items:
                head = list(head) + [f"…({len(value) - max_items} more)"]
            return head
        if isinstance(value, dict):
            keys = sorted(str(k) for k in value.keys())
            if len(keys) <= max_items:
                out = {k: value.get(k) for k in keys}
            else:
                keep = keys[: max_items - 10] + keys[-10:]
                out = {k: value.get(k) for k in keep}
                out["…"] = f"({len(keys) - len(keep)} omitted)"
            # Re-truncate by JSON length if needed.
            try:
                encoded = json.dumps(out, ensure_ascii=False)
            except Exception:
                return _truncate_for_prompt(str(out), max_chars=max_chars)
            if len(encoded) <= max_chars:
                return out
            return _truncate_for_prompt(encoded, max_chars=max_chars)
        return _truncate_for_prompt(str(value), max_chars=max_chars)

    extra_keys_by_id: Dict[str, Tuple[str, ...]] = {
        "event_id": ("event_start", "event_name", "participant_email", "duration"),
        "email_id": ("sent_datetime", "subject", "sender/recipient", "inbox/outbox"),
        "task_id": ("task_name", "assigned_to_email", "board", "list_name", "due_date"),
        "customer_id": ("customer_name", "assigned_to_email", "status", "last_contact_date", "follow_up_by", "product_interest"),
        "visitor_id": (),
    }

    def _add_retrieved_record(id_key: str, row: Mapping[str, object]) -> None:
        raw_id = row.get(id_key)
        if not isinstance(raw_id, str) or not raw_id.strip():
            return
        rid = raw_id.strip()
        retrieved_seen.setdefault(id_key, set())
        if rid in retrieved_seen[id_key]:
            return
        retrieved_seen[id_key].add(rid)

        summary: Dict[str, object] = {id_key: rid}
        for k in extra_keys_by_id.get(id_key, ()):
            if k in row:
                summary[k] = _truncate_for_prompt(row.get(k))
        if len(summary) == 1:
            # Fallback: keep a few keys if we don't have a key list for this id type.
            for k in list(row.keys())[:8]:
                if k == id_key:
                    continue
                summary[str(k)] = _truncate_for_prompt(row.get(k))
        retrieved_records.setdefault(id_key, [])
        if len(retrieved_records[id_key]) < 25:
            retrieved_records[id_key].append(summary)

    root = tree.get("root") if isinstance(tree.get("root"), dict) else {}
    for phase in root.get("children") if isinstance(root.get("children"), list) else []:
        if not isinstance(phase, dict):
            continue
        for slot in phase.get("children") if isinstance(phase.get("children"), list) else []:
            if not isinstance(slot, dict):
                continue
            candidates = slot.get("candidates") if isinstance(slot.get("candidates"), list) else []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                meta = cand.get("meta") if isinstance(cand.get("meta"), dict) else {}
                args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
                qualified = str(meta.get("qualified_tool_name") or "").strip()
                for id_key in config.id_param_keys:
                    if id_key in args:
                        _add_id(id_key, args.get(id_key))
                result_set = meta.get("result_set")
                if isinstance(result_set, list):
                    for item in result_set:
                        if not isinstance(item, dict):
                            continue
                        for id_key in config.id_param_keys:
                            if id_key in item:
                                _add_id(id_key, item.get(id_key))
                                _add_retrieved_record(id_key, item)

                # Capture non-side-effect tool outputs from the enriched tree so the LLM can
                # evaluate conditional queries (e.g., "if so", ">2%") even when the current
                # plan is missing the right CHECK calls.
                schema = tool_schemas.get(qualified) if qualified else None
                if isinstance(schema, Mapping) and not bool(schema.get("side_effect", False)):
                    raw_result = meta.get("result")
                    raw_result_set = meta.get("result_set")
                    if raw_result is not None or raw_result_set is not None:
                        entry: Dict[str, object] = {
                            "tool": qualified,
                            "args": dict(args),
                        }
                        if raw_result is not None:
                            entry["result"] = _compact_result_for_prompt(raw_result)
                        if raw_result_set is not None and qualified.endswith(".search_emails") is False and qualified.endswith(".search_events") is False:
                            # Search result_sets are already summarized in retrieved_records; keep other result_sets.
                            entry["result_set"] = _compact_result_for_prompt(raw_result_set, max_items=25, max_chars=2000)
                        if len(retrieved_checks) < 40:
                            retrieved_checks.append(entry)

    actions = record.get("actions") if isinstance(record.get("actions"), list) else []
    # Include IDs already present in the current plan so validation can allow keeping them.
    for a in actions:
        if not isinstance(a, dict):
            continue
        params = a.get("params") if isinstance(a.get("params"), dict) else {}
        for id_key in config.id_param_keys:
            if id_key in params:
                _add_id(id_key, params.get(id_key))
    try:
        from task_helper.work.multi_round import Call, WorkEnv, _call_string
    except Exception:
        for key, ids in candidate_ids.items():
            candidate_ids[key] = ids[:50]
        for key, items in retrieved_records.items():
            retrieved_records[key] = items[:25]
        return WorkRepairContext(
            candidate_ids=candidate_ids,
            retrieved_records=retrieved_records,
            retrieved_checks=retrieved_checks[:40],
            bindings=bindings,
            check_results=check_results,
        )

    env = WorkEnv()
    env.reset()
    for a in actions:
        if not isinstance(a, dict):
            continue
        phase_key = str(a.get("phase_key") or "")
        attrs = a.get("attrs") if isinstance(a.get("attrs"), dict) else {}
        qualified = str(attrs.get("qualified_tool_name") or "")
        if qualified not in tool_schemas:
            continue
        schema = tool_schemas[qualified]
        side_effect = bool(schema.get("side_effect", False))
        phase_kind = "check" if phase_key.lower().startswith("check") else ("do" if phase_key.lower().startswith("do") else "")
        if phase_kind != "check" or side_effect:
            continue
        params = a.get("params") if isinstance(a.get("params"), dict) else {}
        call = Call(
            phase_key=phase_key,
            action_type=str(a.get("action_type") or ""),
            qualified_tool_name=qualified,
            tool_name=str(schema.get("tool_name") or qualified.split(".", 1)[0]),
            domain=qualified.split(".", 1)[0],
            side_effect=False,
            cost=0.0,
            params={k: str(v) for k, v in params.items() if v is not None},
            raw_action=_call_string(qualified, {k: str(v) for k, v in params.items() if v is not None}),
        )
        result = env.execute(call)
        try:
            rendered = json.dumps(result, ensure_ascii=False)
        except Exception:
            rendered = str(result)
        if len(rendered) > 800:
            rendered = rendered[:799] + "…"
        check_results.append({"call": call.raw_action, "result": rendered})

    bindings = dict(env.bindings or {})
    for key, ids in (env.queues or {}).items():
        if isinstance(key, str) and isinstance(ids, list):
            for v in ids:
                _add_id(key, v)

    for key, ids in candidate_ids.items():
        candidate_ids[key] = ids[:50]
    for key, items in retrieved_records.items():
        retrieved_records[key] = items[:25]
    return WorkRepairContext(
        candidate_ids=candidate_ids,
        retrieved_records=retrieved_records,
        retrieved_checks=retrieved_checks[:40],
        bindings=bindings,
        check_results=check_results[:25],
    )


def _repair_one_work_call_patch(
    *,
    index: int,
    record: Mapping[str, object],
    tree: Mapping[str, object],
    model: str,
    work_config: WorkPatchConfig,
    tool_schemas: Mapping[str, Mapping[str, object]],
    prompt_path: Path,
    no_llm: bool,
    workbench_gt: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[int, Dict[str, object], bool, bool, str, Optional[Dict[str, int]]]:
    template_id = str(record.get("template_id") or "")
    base_record = dict(record)
    usage: Optional[Dict[str, int]] = None
    try:
        if not template_id:
            return index, base_record, False, False, "missing template_id", usage

        actions = base_record.get("actions") if isinstance(base_record.get("actions"), list) else []
        actions = [a for a in actions if isinstance(a, dict)]

        if workbench_gt is None:
            return index, base_record, False, False, "missing_workbench_ground_truth", usage
        before_metrics = _workbench_metrics_for_work_record(record=base_record, ground_truth_by_query=workbench_gt)
        if before_metrics is None:
            return index, base_record, False, False, "missing_workbench_metrics", usage
        before_correct, before_side_effects = before_metrics
        needs_repair = (not before_correct) or bool(before_side_effects)
        if not needs_repair:
            return index, base_record, False, False, "already ok", usage
        if no_llm:
            return index, base_record, True, False, "no_llm", usage

        ctx = _work_collect_repair_context(record=base_record, tree=tree, tool_schemas=tool_schemas, config=work_config)
        action_calls: List[Dict[str, object]] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            phase_key = str(a.get("phase_key") or "")
            attrs = a.get("attrs") if isinstance(a.get("attrs"), dict) else {}
            qualified = str(attrs.get("qualified_tool_name") or "").strip()
            params = a.get("params") if isinstance(a.get("params"), dict) else {}
            action_calls.append(
                {
                    "phase_key": phase_key,
                    "qualified_tool_name": qualified,
                    "params": {str(k): str(v) for k, v in params.items() if v is not None},
                }
            )

        repair_input = {
            "template_id": template_id,
            "query": str((base_record.get("plan") or {}).get("query") or "").strip(),
            "current_actions": action_calls,
            "tool_schemas": tool_schemas,
            "candidate_ids": ctx.candidate_ids,
            "retrieved_records": ctx.retrieved_records,
            "retrieved_checks": ctx.retrieved_checks,
            "bindings": ctx.bindings,
            "check_results": ctx.check_results,
            "canonical_values": {k: list(v) for k, v in work_config.canonical_values.items()},
        }

        template = prompt_path.read_text(encoding="utf-8")
        payload_json = json.dumps(repair_input, ensure_ascii=False, indent=2)
        prompt = template.replace("{{REPAIR_INPUT_JSON}}", payload_json)
        messages = [
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            content, usage = _call_openai_chat(model=model, messages=messages)
        except Exception as exc:
            max_ctx, requested = _extract_context_limit(exc)
            if max_ctx is None or requested is None:
                return index, base_record, True, False, f"{type(exc).__name__}: {exc}", usage
            margin = 500
            truncated_messages = _truncate_messages_to_context_limit(
                messages=messages, model=model, max_context_tokens=max_ctx, margin_tokens=margin
            )
            content, usage = _call_openai_chat(model=model, messages=truncated_messages)

        try:
            resp = json.loads(content)
        except json.JSONDecodeError:
            resp = json.loads(_extract_json_block(content))
        if not isinstance(resp, dict):
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="invalid_llm_response", mode="call_patch"
            ), True, False, "invalid_llm_response", usage

        patch_ops = resp.get("patch")
        if patch_ops is None:
            patch_ops = resp.get("edits")
        if not isinstance(patch_ops, list):
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="missing_patch", mode="call_patch"
            ), True, False, "missing_patch", usage
        if not patch_ops:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="no_edits", mode="call_patch"
            ), True, False, "no_edits", usage

        new_actions = _work_apply_patch(
            base_actions=actions, patch_ops=patch_ops, tool_schemas=tool_schemas, config=work_config
        )
        if new_actions is None:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="invalid_patch", mode="call_patch"
            ), True, False, "invalid_patch", usage

        err = _work_validate_actions(
            actions=new_actions,
            tool_schemas=tool_schemas,
            config=work_config,
            candidate_ids=ctx.candidate_ids,
        )
        if err is not None:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status=f"invalid_actions: {err}", mode="call_patch"
            ), True, False, f"invalid_actions: {err}", usage

        after_metrics = _workbench_metrics_for_work_record(
            record={**base_record, "actions": new_actions}, ground_truth_by_query=workbench_gt
        )
        if after_metrics is None:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="missing_workbench_metrics_after", mode="call_patch"
            ), True, False, "missing_workbench_metrics_after", usage
        after_correct, after_side_effects = after_metrics
        improved_flag = (after_correct and not before_correct) or (
            (after_correct == before_correct) and (before_side_effects and not after_side_effects)
        )
        if not improved_flag:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="no_improvement", mode="call_patch"
            ), True, False, "no_improvement", usage

        repaired_record = dict(base_record)
        repaired_record["actions"] = new_actions
        notes = repaired_record.get("notes")
        if not isinstance(notes, dict):
            notes = {}
            repaired_record["notes"] = notes
        notes["llm_repair"] = {
            "mode": "call_patch",
            "model": model,
            "prompt_path": str(prompt_path),
            "usage": dict(usage) if usage is not None else None,
            "patch": list(patch_ops)[: int(work_config.max_patch_ops)],
            "workbench_eval": {
                "before": {"correct": bool(before_correct), "unwanted_side_effects": bool(before_side_effects)},
                "after": {"correct": bool(after_correct), "unwanted_side_effects": bool(after_side_effects)},
                "improved": bool(improved_flag),
            },
        }
        repaired_record = _attach_llm_repair_usage(
            repaired_record, model=model, prompt_path=prompt_path, usage=usage, status="improved", mode="call_patch"
        )
        return index, repaired_record, True, True, "improved", usage
    except Exception as exc:
        return index, _attach_llm_repair_usage(
            base_record, model=model, prompt_path=prompt_path, usage=usage, status=f"{type(exc).__name__}: {exc}", mode="call_patch"
        ), True, False, f"{type(exc).__name__}: {exc}", usage


def _truncate_value_for_prompt(value: object, *, max_chars: int) -> object:
    if max_chars <= 0:
        return value
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[: max(0, max_chars - 1)] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except TypeError:
        encoded = str(value)
    if len(encoded) <= max_chars:
        return value
    return encoded[: max(0, max_chars - 1)] + "…"


def _candidate_summary_for_prompt(
    candidate: Mapping[str, object],
    *,
    action_type: str,
    summary_config: CandidateSummaryConfig,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "source": candidate.get("source"),
        "text": candidate.get("text"),
        "cost": candidate.get("cost"),
    }
    meta = candidate.get("meta")
    if not isinstance(meta, dict):
        return summary

    assignments = meta.get("assignments")
    if isinstance(assignments, dict) and assignments:
        summary["assignments"] = dict(assignments)

    for key in summary_config.include_meta_keys:
        if key in {"row", "assignments"}:
            continue
        if key not in meta:
            continue
        summary[key] = _truncate_value_for_prompt(meta.get(key), max_chars=summary_config.max_value_chars)

    row = meta.get("row")
    if isinstance(row, dict) and row:
        keep_keys = summary_config.row_keys_by_action_type.get(action_type)
        if not keep_keys:
            keep_keys = summary_config.row_keys_default
        if keep_keys:
            keys = list(keep_keys)[: summary_config.max_row_keys]
        else:
            keys = sorted(str(k) for k in row.keys())[: summary_config.max_row_keys]
        compact_row: Dict[str, object] = {}
        for key in keys:
            if key in row:
                compact_row[key] = _truncate_value_for_prompt(row.get(key), max_chars=summary_config.max_value_chars)
        summary["row"] = compact_row

    return summary


def _score_key(score: object, *, semantic_threshold: float) -> Tuple[int, int, float, float]:
    hard_pass = bool(getattr(score, "hard_pass", False))
    semantic_rate = float(getattr(score, "semantic_rate", 0.0))
    total_cost = float(getattr(score, "total_cost", 0.0))
    return (
        1 if hard_pass else 0,
        1 if semantic_rate >= semantic_threshold else 0,
        semantic_rate,
        -total_cost,
    )


def _extract_chosen_indices(record: Mapping[str, object]) -> Optional[List[int]]:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    mcts = notes.get("mcts") if isinstance(notes.get("mcts"), dict) else {}
    raw = mcts.get("chosen_indices")
    if not isinstance(raw, list):
        return None
    indices: List[int] = []
    for item in raw:
        try:
            indices.append(int(item))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return indices


def _parse_llm_edits(payload: Mapping[str, object]) -> List[Tuple[int, int]]:
    edits = payload.get("edits")
    if edits is None:
        edits = payload.get("patch")
    if not isinstance(edits, list):
        return []
    parsed: List[Tuple[int, int]] = []
    for item in edits:
        if not isinstance(item, dict):
            continue
        try:
            slot_idx = int(item.get("slot_index"))  # type: ignore[arg-type]
            cand_idx = int(item.get("candidate_index"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        parsed.append((slot_idx, cand_idx))
    return parsed


def _attach_llm_repair_usage(
    record: Mapping[str, object],
    *,
    model: str,
    prompt_path: Path,
    usage: Optional[Dict[str, int]],
    status: str,
    mode: Optional[str] = None,
) -> Dict[str, object]:
    if usage is None:
        return dict(record)
    out = dict(record)
    notes = out.get("notes")
    if not isinstance(notes, dict):
        notes = {}
        out["notes"] = notes
    entry = notes.get("llm_repair")
    if not isinstance(entry, dict):
        entry = {}
        notes["llm_repair"] = entry
    if mode and "mode" not in entry:
        entry["mode"] = mode
    if "model" not in entry:
        entry["model"] = model
    if "prompt_path" not in entry:
        entry["prompt_path"] = str(prompt_path)
    entry["status"] = str(status)
    entry["usage"] = dict(usage)
    return out


def _needs_repair_by_notes(record: Mapping[str, object], *, semantic_threshold: float) -> bool:
    notes = record.get("notes")
    if not isinstance(notes, dict):
        return True
    mcts = notes.get("mcts")
    if not isinstance(mcts, dict):
        return True
    hard_pass = mcts.get("hard_pass")
    semantic_rate = mcts.get("semantic_rate")
    if isinstance(hard_pass, bool) and isinstance(semantic_rate, (int, float)):
        return (not hard_pass) or (float(semantic_rate) < float(semantic_threshold))
    return True


def _repair_one(
    *,
    index: int,
    record: Mapping[str, object],
    tree: Mapping[str, object],
    task: str,
    model: str,
    semantic_threshold: float,
    max_slot_candidates: int,
    spec: object,
    repair_config: RepairConfig,
    prompt_path: Path,
    config: MCTSConfig,
    no_llm: bool,
) -> Tuple[int, Dict[str, object], bool, bool, str, Optional[Dict[str, int]]]:
    """
    Repair a single optimized template record.

    Returns:
      (index, output_record, attempted, improved, message)
    """

    template_id = str(record.get("template_id") or "")
    base_record = dict(record)
    usage: Optional[Dict[str, int]] = None

    if not template_id:
        return index, base_record, False, False, "missing template_id", usage

    try:
        plan, phases, slots = _parse_tree_entry(tree)

        chosen_indices = _extract_chosen_indices(record)
        if chosen_indices is None or len(chosen_indices) != len(slots):
            return index, base_record, False, False, "missing chosen_indices", usage

        current_score = spec.evaluate(  # type: ignore[attr-defined]
            plan=plan,
            phases=phases,
            slots=slots,
            chosen_indices=chosen_indices,
            config=config,
        )

        needs_repair = (not current_score.hard_pass) or (current_score.semantic_rate < semantic_threshold)
        if not needs_repair:
            return index, base_record, False, False, "already ok", usage

        if no_llm:
            return index, base_record, True, False, "no_llm", usage

        current_key = _score_key(current_score, semantic_threshold=semantic_threshold)

        failed_names = [
            name
            for name, (flag, _msg) in {**current_score.semantic_details, **current_score.hard_details}.items()
            if flag is False
        ]
        relevant_action_types: Optional[Set[str]] = None
        constraint_to_action_types = repair_config.constraint_to_action_types
        if constraint_to_action_types:
            missing_constraints = any(name not in constraint_to_action_types for name in failed_names)
            if missing_constraints and repair_config.unknown_constraint_policy == "include_all_slots":
                relevant_action_types = None
            else:
                relevant_action_types = set()
                for name in failed_names:
                    for action_type in constraint_to_action_types.get(name, ()):
                        if action_type:
                            relevant_action_types.add(str(action_type))
                relevant_action_types.update(repair_config.always_include_action_types)
                if not relevant_action_types:
                    relevant_action_types = None

        # Build the trimmed tree context for the LLM.
        slots_for_prompt: List[Dict[str, object]] = []
        for slot in slots:
            if relevant_action_types is not None and slot.action_type not in relevant_action_types:
                continue
            chosen_idx = chosen_indices[slot.slot_index]
            candidates = slot.candidates
            chosen_candidate = candidates[chosen_idx] if 0 <= chosen_idx < len(candidates) else {}
            # Prefer low-cost candidates first when trimming.
            order = sorted(
                range(len(candidates)),
                key=lambda idx: (_safe_float(candidates[idx].get("cost"), default=0.0), str(candidates[idx].get("text") or "")),
            )
            trimmed = order[: min(max_slot_candidates, len(order))]
            slots_for_prompt.append(
                {
                    "slot_index": slot.slot_index,
                    "slot_edge": slot.slot_edge,
                    "phase_key": slot.phase_key,
                    "action_type": slot.action_type,
                    "current_choice": {
                        "candidate_index": chosen_idx,
                        **_candidate_summary_for_prompt(
                            chosen_candidate,
                            action_type=slot.action_type,
                            summary_config=repair_config.candidate_summary,
                        ),
                    },
                    "candidates": [
                        {
                            "candidate_index": idx,
                            **_candidate_summary_for_prompt(
                                candidates[idx],
                                action_type=slot.action_type,
                                summary_config=repair_config.candidate_summary,
                            ),
                        }
                        for idx in trimmed
                    ],
                }
            )

        repair_input = {
            "task_name": task,
            "template_id": template_id,
            "semantic_threshold": float(semantic_threshold),
            "current": {
                "hard_pass": current_score.hard_pass,
                "semantic_rate": current_score.semantic_rate,
                "total_cost": current_score.total_cost,
            },
            "hard_constraints": _format_constraint_table(current_score.hard_details, only_failed=False),
            "soft_constraints": _format_constraint_table(current_score.semantic_details, only_failed=False),
            "failed_constraints": failed_names,
            "slots": slots_for_prompt,
        }

        prompt = _render_prompt(prompt_path, repair_input=repair_input)
        messages = [
            {"role": "system", "content": "You output valid JSON only. No markdown. No extra text."},
            {"role": "user", "content": prompt},
        ]
        truncated_messages = messages
        base_margin = 8192
        for attempt in range(3):
            try:
                content, usage = _call_openai_chat(model=model, messages=truncated_messages)
                break
            except Exception as exc:
                max_ctx, requested = _extract_context_limit(exc)
                if max_ctx is None:
                    raise
                overflow = 0
                if requested is not None:
                    overflow = max(0, int(requested) - int(max_ctx))
                margin = base_margin + overflow
                truncated_messages = _truncate_messages_to_context_limit(
                    messages=messages,
                    model=model,
                    max_context_tokens=max_ctx,
                    margin_tokens=margin,
                )
                if attempt == 2:
                    raise
        try:
            repair_obj = json.loads(content)
        except json.JSONDecodeError:
            repair_obj = json.loads(_extract_json_block(content))
        if not isinstance(repair_obj, dict):
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="invalid_llm_response"
            ), True, False, "invalid_llm_response", usage

        edits = _parse_llm_edits(repair_obj)
        if not edits:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="no_edits"
            ), True, False, "no_edits", usage

        new_indices = chosen_indices.copy()
        for slot_idx, cand_idx in edits:
            if 0 <= slot_idx < len(new_indices):
                candidates = slots[slot_idx].candidates
                if 0 <= cand_idx < len(candidates):
                    new_indices[slot_idx] = cand_idx

        new_score = spec.evaluate(  # type: ignore[attr-defined]
            plan=plan,
            phases=phases,
            slots=slots,
            chosen_indices=new_indices,
            config=config,
        )
        new_key = _score_key(new_score, semantic_threshold=semantic_threshold)
        if new_key <= current_key:
            return index, _attach_llm_repair_usage(
                base_record, model=model, prompt_path=prompt_path, usage=usage, status="no_improvement"
            ), True, False, "no_improvement", usage

        trace = [
            "LLM repair applied",
            f"edits={len(edits)}",
            f"old hard={current_score.hard_pass} sem={current_score.semantic_rate:.3f} cost={current_score.total_cost:.2f}",
            f"new hard={new_score.hard_pass} sem={new_score.semantic_rate:.3f} cost={new_score.total_cost:.2f}",
        ]
        repaired_record = render_filled_template(
            tree=tree,
            phases=phases,
            slots=slots,
            chosen_indices=new_indices,
            score=new_score,
            trace=trace,
            spec=spec,  # type: ignore[arg-type]
        )
        notes = repaired_record.get("notes")
        if not isinstance(notes, dict):
            notes = {}
            repaired_record["notes"] = notes
        notes["llm_repair"] = {
            "model": model,
            "prompt_path": str(prompt_path),
            "semantic_threshold": float(semantic_threshold),
            "usage": dict(usage) if usage is not None else None,
            "edits": [{"slot_index": s, "candidate_index": c} for s, c in edits],
            "before": {
                "hard_pass": current_score.hard_pass,
                "semantic_rate": current_score.semantic_rate,
                "total_cost": current_score.total_cost,
            },
            "after": {
                "hard_pass": new_score.hard_pass,
                "semantic_rate": new_score.semantic_rate,
                "total_cost": new_score.total_cost,
            },
        }
        repaired_record = _attach_llm_repair_usage(
            repaired_record, model=model, prompt_path=prompt_path, usage=usage, status="improved"
        )
        return index, repaired_record, True, True, "improved", usage
    except Exception as exc:
        return index, _attach_llm_repair_usage(
            base_record, model=model, prompt_path=prompt_path, usage=usage, status=f"{type(exc).__name__}: {exc}"
        ), True, False, f"{type(exc).__name__}: {exc}", usage


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based repair over AND–OR trees.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name (defaults to travel).")
    parser.add_argument("--config", type=Path, default=None, help="Override per-task planner config JSON (planner.json).")
    parser.add_argument("--repair-config", type=Path, default=None, help="Override per-task repair config JSON (repair.json).")
    parser.add_argument("--tree", type=Path, required=True, help="AND–OR tree JSON file (tree_<split>.json).")
    parser.add_argument("--input", type=Path, required=True, help="MCTS output JSONL (optimized_<split>.jsonl).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--semantic-threshold", type=float, default=0.8, help="Target semantic rate threshold.")
    parser.add_argument("--max-slot-candidates", type=int, default=50, help="Max candidates per slot included in the LLM prompt.")
    parser.add_argument("--max-failed", type=int, default=None, help="Repair at most N failing plans (default: all).")
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpu_count(),
        help="Parallel worker threads (defaults to all available CPUs).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL (defaults to artifacts/output/<task>/repaired_<split>_<model>.jsonl).")
    parser.add_argument("--prompt", type=Path, default=None, help="Override repair prompt path.")
    parser.add_argument("--no-llm", action="store_true", help="Do not call the LLM (just re-score and copy input).")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output.")
    parser.add_argument(
        "--workbench-only-incorrect",
        action="store_true",
        help="(Work/call_patch only) (Deprecated; now default) Filter repair candidates using WorkBench accuracy + unwanted side effects.",
    )
    parser.add_argument(
        "--workbench-ground-truth-dir",
        type=Path,
        default=None,
        help="(Work/call_patch only) Ground truth dir (defaults to artifacts/input/work/dataset/queries_and_answers).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    started_at = time.perf_counter()
    argv_snapshot = sys.argv[:]
    status = "ok"
    error: Optional[str] = None
    log_extra: Dict[str, object] = {}

    improved = 0
    attempted = 0
    total = 0
    attempt_total = 0

    try:
        args = parse_args(argv)
        task = str(args.task)
        log_extra.update(
            {
                "task": task,
                "config": str(args.config) if args.config is not None else None,
                "model": str(args.model),
                "semantic_threshold": float(args.semantic_threshold),
                "max_slot_candidates": int(args.max_slot_candidates),
                "max_failed": args.max_failed,
                "workers": int(args.workers),
                "no_llm": bool(args.no_llm),
                "no_progress": bool(args.no_progress),
                "workbench_only_incorrect": bool(args.workbench_only_incorrect),
                "workbench_ground_truth_dir": str(args.workbench_ground_truth_dir) if args.workbench_ground_truth_dir is not None else None,
            }
        )

        try:
            spec = load_task_spec(task, config_path=args.config)
        except ConfigError as exc:
            raise SystemExit(str(exc))
        log_extra.update({"repair_config": str(args.repair_config) if args.repair_config is not None else None})

        mode = "slot_swap"
        repair_cfg_path = args.repair_config
        if repair_cfg_path is None:
            repair_cfg_path = INPUT_ROOT / task / "repair.json"
        repair_cfg_path = _resolve_path(repair_cfg_path)
        repair_cfg_payload = _read_optional_json_object(repair_cfg_path) if repair_cfg_path.exists() else {}
        raw_mode = repair_cfg_payload.get("repair_mode")
        if isinstance(raw_mode, str) and raw_mode.strip():
            mode = raw_mode.strip()
        if task == "work" and mode not in {"slot_swap", "call_patch"}:
            mode = "call_patch"

        tree_path = _resolve_path(args.tree)
        input_path = _resolve_path(args.input)

        # Allow passing directories for convenience, similar to eval_bridge.
        if tree_path.exists() and tree_path.is_dir():
            candidates = sorted(tree_path.glob("tree_*.json"))
            if len(candidates) == 1:
                tree_path = candidates[0]
            elif len(candidates) > 1:
                listing = "\n".join(f"  - {p}" for p in candidates[:20])
                extra_suffix = "" if len(candidates) <= 20 else f"\n  ... ({len(candidates) - 20} more)"
                raise SystemExit(
                    f"--tree points to a directory: {tree_path}\n"
                    "Pass a tree JSON file.\n"
                    f"Candidates:\n{listing}{extra_suffix}"
                )
            else:
                raise SystemExit(f"--tree points to a directory with no tree_*.json: {tree_path}")

        if input_path.exists() and input_path.is_dir():
            split = _split_from_tree_path(tree_path)
            preferred = input_path / (f"optimized_{split}.jsonl" if split else "optimized_validation.jsonl")
            if preferred.exists():
                input_path = preferred
            else:
                candidates = sorted(input_path.glob("optimized_*.jsonl"))
                if len(candidates) == 1:
                    input_path = candidates[0]
                elif len(candidates) > 1:
                    listing = "\n".join(f"  - {p}" for p in candidates[:20])
                    extra_suffix = "" if len(candidates) <= 20 else f"\n  ... ({len(candidates) - 20} more)"
                    raise SystemExit(
                        f"--input points to a directory: {input_path}\n"
                        "Pass an optimized_<split>.jsonl file, or ensure only one optimized_*.jsonl exists.\n"
                        f"Candidates:\n{listing}{extra_suffix}"
                    )
                else:
                    raise SystemExit(f"--input points to a directory with no optimized_*.jsonl: {input_path}")

        if not tree_path.exists():
            raise SystemExit(f"Tree JSON not found: {tree_path}")
        if not input_path.exists():
            raise SystemExit(f"Input JSONL not found: {input_path}")
        log_extra.update({"tree": str(tree_path), "input": str(input_path)})

        out_path = args.out
        if out_path is None:
            split = _split_from_optimized_path(input_path)
            model_slug = _model_slug(str(args.model))
            if split:
                out_path = input_path.parent / f"repaired_{split}_{model_slug}.jsonl"
            else:
                out_path = input_path.with_suffix(f".repaired_{model_slug}.jsonl")
        out_path = _resolve_path(out_path)

        repair_config: Optional[RepairConfig] = None
        work_patch_config: Optional[WorkPatchConfig] = None
        tool_schemas: Dict[str, Dict[str, object]] = {}
        if mode == "call_patch":
            work_patch_config = _load_work_patch_config(task, config_path=args.repair_config)
            prompt_path = args.prompt or work_patch_config.prompt_path
            tool_schemas = _work_load_tool_schema_index()
        else:
            repair_config = _load_repair_config(task, config_path=args.repair_config)
            prompt_path = args.prompt or repair_config.prompt_path
        prompt_path = _resolve_path(prompt_path)
        if not prompt_path.exists():
            raise SystemExit(f"Repair prompt not found: {prompt_path}")
        log_extra.update({"out": str(out_path), "prompt": str(prompt_path), "repair_mode": mode})

        trees_payload = json.loads(tree_path.read_text(encoding="utf-8"))
        if not isinstance(trees_payload, dict):
            raise SystemExit(f"Tree JSON must be an object mapping template_id -> tree: {tree_path}")

        semantic_threshold = float(args.semantic_threshold)
        semantic_threshold = max(0.0, min(1.0, semantic_threshold))
        max_slot_candidates = max(1, int(args.max_slot_candidates))
        log_extra.update({"semantic_threshold": float(semantic_threshold), "max_slot_candidates": int(max_slot_candidates)})

        config = MCTSConfig(semantic_tolerance=semantic_threshold)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = list(_iter_jsonl(input_path))
        total = len(records)

        show_progress = not bool(args.no_progress)
        workers = max(1, int(args.workers))

        attempt_indices: List[int] = []
        output_records: List[Dict[str, object]] = [dict(r) for r in records]

        workbench_gt: Optional[Dict[str, List[str]]] = None
        if mode == "call_patch" and task != "work":
            raise SystemExit(f"repair_mode=call_patch is only supported for task=work (task={task})")

        if task == "work" and mode == "call_patch":
            gt_dir = args.workbench_ground_truth_dir
            if gt_dir is None:
                gt_dir = INPUT_ROOT / "work" / "dataset" / "queries_and_answers"
            gt_dir = _resolve_path(gt_dir)
            workbench_gt = _load_workbench_ground_truth(ground_truth_dir=gt_dir)
            log_extra.update({"workbench_ground_truth_dir": str(gt_dir), "workbench_ground_truth_n": len(workbench_gt)})
            for idx, record in enumerate(records):
                template_id = str(record.get("template_id") or "")
                tree = trees_payload.get(template_id)
                if not isinstance(tree, dict):
                    continue
                metrics = _workbench_metrics_for_work_record(record=record, ground_truth_by_query=workbench_gt)
                if metrics is None:
                    continue
                correct, unwanted_side_effects = metrics
                if (not correct) or bool(unwanted_side_effects):
                    attempt_indices.append(idx)
        else:
            for idx, record in enumerate(records):
                template_id = str(record.get("template_id") or "")
                tree = trees_payload.get(template_id)
                if not isinstance(tree, dict):
                    continue
                if not _needs_repair_by_notes(record, semantic_threshold=semantic_threshold):
                    continue
                _plan, _phases, slots = _parse_tree_entry(tree)
                chosen_indices = _extract_chosen_indices(record)
                if chosen_indices is None or len(chosen_indices) != len(slots):
                    continue
                attempt_indices.append(idx)

        if args.max_failed is not None:
            attempt_indices = attempt_indices[: int(args.max_failed)]

        start = time.perf_counter()
        attempt_total = len(attempt_indices)
        token_counter = TokenCounter()

        if show_progress:
            print(
                f"LLM repair: mode={mode} input={input_path} tree={tree_path} model={args.model} "
                f"semantic_threshold={semantic_threshold} max_slot_candidates={max_slot_candidates} "
                f"workers={workers} to_repair={attempt_total}/{total}",
                flush=True,
            )

        if args.no_llm or attempt_total == 0:
            attempted = attempt_total
        else:
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures: List[
                    concurrent.futures.Future[Tuple[int, Dict[str, object], bool, bool, str, Optional[Dict[str, int]]]]
                ] = []
                for idx in attempt_indices:
                    record = records[idx]
                    template_id = str(record.get("template_id") or "")
                    tree = trees_payload.get(template_id)
                    if not isinstance(tree, dict):
                        continue
                    if mode == "call_patch":
                        assert work_patch_config is not None
                        futures.append(
                            executor.submit(
                _repair_one_work_call_patch,
                index=idx,
                record=record,
                tree=tree,
                model=str(args.model),
                work_config=work_patch_config,
                tool_schemas=tool_schemas,
                prompt_path=prompt_path,
                no_llm=bool(args.no_llm),
                                workbench_gt=workbench_gt,
                            )
                        )
                    else:
                        assert repair_config is not None
                        futures.append(
                            executor.submit(
                                _repair_one,
                                index=idx,
                                record=record,
                                tree=tree,
                                task=task,
                                model=str(args.model),
                                semantic_threshold=semantic_threshold,
                                max_slot_candidates=max_slot_candidates,
                                spec=spec,
                                repair_config=repair_config,
                                prompt_path=prompt_path,
                                config=config,
                                no_llm=bool(args.no_llm),
                            )
                        )

                for future in concurrent.futures.as_completed(futures):
                    idx, out_record, did_attempt, did_improve, message, usage = future.result()
                    output_records[idx] = out_record
                    token_counter.add(usage)
                    completed += 1
                    attempted += 1 if did_attempt else 0
                    improved += 1 if did_improve else 0
                    if show_progress:
                        template_id = str(records[idx].get("template_id") or "")
                        print(f"[{completed}/{attempt_total}] template_id={template_id} {message}", flush=True)

        with out_path.open("w", encoding="utf-8") as fp:
            for record in output_records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - start
        log_extra.update(
            {
                "elapsed_compute_s": round(elapsed, 6),
                "total": total,
                "to_repair": attempt_total,
                "attempted": attempted,
                "improved": improved,
                "token_usage": {
                    "calls": token_counter.calls,
                    "prompt_tokens": token_counter.prompt_tokens,
                    "prompt_cache_hit_tokens": token_counter.prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": token_counter.prompt_cache_miss_tokens,
                    "completion_tokens": token_counter.completion_tokens,
                    "total_tokens": token_counter.total_tokens,
                },
            }
        )
        print(f"LLM repair done: {improved}/{attempted} improved (total={total}) elapsed={elapsed:.1f}s out={out_path}", flush=True)
        if token_counter.calls:
            print(
                f"LLM token usage (llm_repair): calls={token_counter.calls} "
                f"prompt_cache_hit={token_counter.prompt_cache_hit_tokens} "
                f"prompt_cache_miss={token_counter.prompt_cache_miss_tokens} "
                f"output={token_counter.completion_tokens} total={token_counter.total_tokens}",
                flush=True,
            )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, SystemExit):
            status = "exit"
            code = exc.code
            if isinstance(code, (int, float, str, bool)) or code is None:
                log_extra.update({"exit_code": code})
            else:
                log_extra.update({"exit_code": repr(code)})
        else:
            status = "error"
        error = repr(exc)
        raise
    finally:
        log_extra.setdefault("total", total)
        log_extra.setdefault("to_repair", attempt_total)
        log_extra.setdefault("attempted", attempted)
        log_extra.setdefault("improved", improved)
        _append_run_log(argv=argv_snapshot, started_at=started_at, status=status, extra=log_extra, error=error)


if __name__ == "__main__":
    main()

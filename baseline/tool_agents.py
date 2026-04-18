from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import os
import re
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
try:
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tiktoken = None
from pandas import DataFrame
import pandas as pd
try:
    from baseline.prompts import planner_agent_prompt, zeroshot_react_agent_prompt
except ModuleNotFoundError:  # pragma: no cover - script-mode fallback
    from prompts import planner_agent_prompt, zeroshot_react_agent_prompt

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable


BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(BASELINE_DIR)

from task_helper.travel.utils.paths import travel_dataset_root  # noqa: E402

_DOTENV_LOADED = False


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
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key not in os.environ or not os.environ.get(key):
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


class LLMRateLimitError(LLMHTTPError):
    pass


@dataclass
class TokenCounter:
    calls: int = 0
    prompt_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    approx_calls: int = 0

    def snapshot(self) -> Tuple[int, int, int, int, int]:
        return (self.calls, self.prompt_tokens, self.completion_tokens, self.total_tokens, self.approx_calls)

    def add(
        self,
        *,
        usage: Optional[Mapping[str, object]],
        prompt: str,
        completion: str,
        model: str,
    ) -> None:
        self.calls += 1
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            prompt_cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
            prompt_cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
            if prompt_cache_hit_tokens is None:
                details = usage.get("prompt_tokens_details")
                if isinstance(details, Mapping):
                    prompt_cache_hit_tokens = details.get("cached_tokens")
        else:
            prompt_tokens = completion_tokens = total_tokens = None
            prompt_cache_hit_tokens = prompt_cache_miss_tokens = None

        if prompt_tokens is None or completion_tokens is None:
            enc = _encoding_for_model(model)
            prompt_count = len(enc.encode(prompt))
            completion_count = len(enc.encode(completion))
            total_count = prompt_count + completion_count
            cache_hit_count = 0
            cache_miss_count = prompt_count
            self.approx_calls += 1
        else:
            prompt_count = int(prompt_tokens or 0)
            completion_count = int(completion_tokens or 0)
            if total_tokens is None:
                total_count = prompt_count + completion_count
            else:
                total_count = int(total_tokens or 0)
            cache_hit_count = int(prompt_cache_hit_tokens or 0)
            if prompt_cache_miss_tokens is None:
                cache_miss_count = max(0, prompt_count - cache_hit_count)
            else:
                cache_miss_count = int(prompt_cache_miss_tokens or 0)

        self.prompt_tokens += prompt_count
        self.prompt_cache_hit_tokens += cache_hit_count
        self.prompt_cache_miss_tokens += cache_miss_count
        self.completion_tokens += completion_count
        self.total_tokens += total_count


@dataclass
class ToolCallCounter:
    tool_costs: Dict[str, float]
    total_calls: int = 0
    total_cost: float = 0.0
    by_tool: Dict[str, Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.by_tool is None:
            self.by_tool = {}

    def snapshot(self) -> Tuple[int, float, str]:
        return (self.total_calls, self.total_cost, json.dumps(self.by_tool, ensure_ascii=False, sort_keys=True))

    def record(self, tool_name: str) -> None:
        cost = float(self.tool_costs.get(tool_name, 0.0))
        self.total_calls += 1
        self.total_cost += cost
        entry = self.by_tool.setdefault(tool_name, {"calls": 0.0, "total_cost": 0.0})
        entry["calls"] = float(entry.get("calls", 0.0) + 1.0)
        entry["total_cost"] = float(entry.get("total_cost", 0.0) + cost)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "", str(model_name or "")).strip().lower()
    return slug or "model"

def _is_big_model(model_name: str) -> bool:
    model_norm = str(model_name or "").strip().lower()
    if model_norm.startswith("deepseek"):
        return True
    if "gpt-5.2" in model_norm or model_norm.startswith("gpt-5.2"):
        return True
    return False


def _load_tool_costs() -> Dict[str, float]:
    path = PROJECT_ROOT / "artifacts" / "input" / "travel" / "views" / "tool.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    costs = data.get("tool_costs")
    if not isinstance(costs, dict):
        return {}
    return {str(k): float(v) for k, v in costs.items()}

def _resolve_workers(workers: int) -> int:
    value = int(workers)
    if value == 0:
        return int(os.cpu_count() or 1)
    return max(1, value)

def _merge_tool_usage(dest: Dict[str, Dict[str, float]], src: Mapping[str, object]) -> None:
    if not isinstance(src, Mapping):
        return
    for tool_name, raw in src.items():
        if not isinstance(raw, Mapping):
            continue
        entry = dest.setdefault(str(tool_name), {"calls": 0.0, "total_cost": 0.0})
        entry["calls"] = float(entry.get("calls", 0.0) + float(raw.get("calls", 0.0) or 0.0))
        entry["total_cost"] = float(entry.get("total_cost", 0.0) + float(raw.get("total_cost", 0.0) or 0.0))


def _provider_from_model(model: str) -> str:
    model_norm = str(model or "").strip().lower()
    return "deepseek" if model_norm.startswith("deepseek") else "openai"


def _resolve_api_config(model: str) -> Tuple[str, str]:
    _ensure_dotenv_loaded()
    provider = _provider_from_model(model)
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required for DeepSeek calls.")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI calls.")
    url = str(base_url).rstrip("/") + "/chat/completions"
    return api_key, url


@dataclass
class ChatClient:
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    stop: Optional[Sequence[str]] = None
    timeout_s: int = 60
    token_counter: Optional[TokenCounter] = None

    def _responses_create(self, *, prompt: str, max_output_tokens: int) -> Tuple[str, Optional[Mapping[str, object]]]:
        api_key, url = _resolve_api_config(self.model)
        base = url.rsplit("/chat/completions", 1)[0]
        responses_url = base + "/responses"
        provider = _provider_from_model(self.model)
        model_norm = str(self.model or "").strip().lower()
        model_key = model_norm.rsplit("/", 1)[-1]
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": int(max_output_tokens),
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low", "format": {"type": "text"}},
        }
        if not (provider == "openai" and model_key.startswith(("gpt-5-mini", "gpt-5-nano"))):
            payload["temperature"] = self.temperature

        try:
            resp = requests.post(
                responses_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            raise LLMHTTPError(f"HTTP request failed: {exc}", status_code=503) from exc

        status = int(resp.status_code)
        if status == 429:  # pragma: no cover - provider dependent
            raise LLMRateLimitError(f"Rate limit hit: {resp.text}", status_code=status)
        if status < 200 or status >= 300:  # pragma: no cover - provider dependent
            raise LLMHTTPError(f"HTTP {status}: {resp.text}", status_code=status)

        payload_out = resp.json()

        # Prefer convenience field when present.
        output_text = payload_out.get("output_text") if isinstance(payload_out, dict) else None
        if isinstance(output_text, str) and output_text.strip():
            return output_text, payload_out.get("usage") if isinstance(payload_out, dict) else None

        # Fallback: traverse output/content blocks.
        pieces: List[str] = []
        output = payload_out.get("output") if isinstance(payload_out, dict) else None
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                        pieces.append(block["text"])
        text = "".join(pieces).strip()
        usage = payload_out.get("usage") if isinstance(payload_out, dict) else None

        # gpt-5-mini/nano may return only "reasoning" blocks at low max_output_tokens (no message text).
        # Retry with a larger budget so we get a textual message.
        if provider == "openai" and model_key.startswith(("gpt-5-mini", "gpt-5-nano")) and not text:
            if int(max_output_tokens) < 512:
                return self._responses_create(prompt=prompt, max_output_tokens=512)

        return text, usage

    def complete(
        self,
        prompt: str,
        *,
        stop: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        max_token_value = int(max_tokens if max_tokens is not None else self.max_tokens)
        provider = _provider_from_model(self.model)
        model_norm = str(self.model or "").strip().lower()
        model_key = model_norm.rsplit("/", 1)[-1]

        # OpenAI gpt-5-mini/nano return empty `message.content` on /chat/completions; use /responses.
        if provider == "openai" and model_key.startswith(("gpt-5-mini", "gpt-5-nano")):
            text, usage = self._responses_create(prompt=prompt, max_output_tokens=max_token_value)
            if not text:
                raise RuntimeError("LLM response content is empty.")
            if self.token_counter is not None:
                usage_for_counter: Optional[Mapping[str, object]] = usage if isinstance(usage, Mapping) else None
                if usage_for_counter is not None and ("input_tokens" in usage_for_counter or "output_tokens" in usage_for_counter):
                    prompt_tokens = int(usage_for_counter.get("input_tokens") or 0)
                    completion_tokens = int(usage_for_counter.get("output_tokens") or 0)
                    total_tokens = usage_for_counter.get("total_tokens")
                    try:
                        total_tokens_int = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
                    except (TypeError, ValueError):
                        total_tokens_int = prompt_tokens + completion_tokens

                    cached = 0
                    details = usage_for_counter.get("input_tokens_details")
                    if isinstance(details, Mapping):
                        try:
                            cached = int(details.get("cached_tokens") or 0)
                        except (TypeError, ValueError):
                            cached = 0
                    usage_for_counter = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens_int,
                        "prompt_tokens_details": {"cached_tokens": cached},
                    }
                self.token_counter.add(usage=usage_for_counter, prompt=prompt, completion=text, model=self.model)
            return text

        api_key, url = _resolve_api_config(self.model)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not (provider == "openai" and model_key.startswith(("gpt-5-mini", "gpt-5-nano"))):
            payload["temperature"] = self.temperature
        if provider == "openai" and model_key.startswith(("o1", "o3", "gpt-5")):
            payload["max_completion_tokens"] = max_token_value
        else:
            payload["max_tokens"] = max_token_value
        stops = stop if stop is not None else self.stop
        if stops:
            if provider == "openai" and model_key.startswith(("o1", "o3", "gpt-5")):
                pass
            else:
                payload["stop"] = list(stops)
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            raise LLMHTTPError(f"HTTP request failed: {exc}", status_code=503) from exc

        status = int(resp.status_code)
        if status == 429:  # pragma: no cover - provider dependent
            raise LLMRateLimitError(f"Rate limit hit: {resp.text}", status_code=status)
        if status < 200 or status >= 300:  # pragma: no cover - provider dependent
            raise LLMHTTPError(f"HTTP {status}: {resp.text}", status_code=status)

        payload_out = resp.json()
        choices = payload_out.get("choices") if isinstance(payload_out, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response has no choices.")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        text = "" if content is None else str(content)

        usage = payload_out.get("usage") if isinstance(payload_out, dict) else None
        if not text:
            raise RuntimeError("LLM response content is empty.")

        if self.token_counter is not None:
            # Normalize Responses API usage keys to the chat.completions shape used by TokenCounter.
            usage_for_counter: Optional[Mapping[str, object]] = usage if isinstance(usage, Mapping) else None
            if usage_for_counter is not None and ("input_tokens" in usage_for_counter or "output_tokens" in usage_for_counter):
                prompt_tokens = int(usage_for_counter.get("input_tokens") or 0)
                completion_tokens = int(usage_for_counter.get("output_tokens") or 0)
                total_tokens = usage_for_counter.get("total_tokens")
                try:
                    total_tokens_int = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
                except (TypeError, ValueError):
                    total_tokens_int = prompt_tokens + completion_tokens

                cached = 0
                details = usage_for_counter.get("input_tokens_details")
                if isinstance(details, Mapping):
                    try:
                        cached = int(details.get("cached_tokens") or 0)
                    except (TypeError, ValueError):
                        cached = 0
                usage_for_counter = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens_int,
                    "prompt_tokens_details": {"cached_tokens": cached},
                }
            self.token_counter.add(usage=usage_for_counter, prompt=prompt, completion=text, model=self.model)
        return text


class PlannerTool:
    def __init__(self, model_name: str, *, token_counter: Optional[TokenCounter] = None) -> None:
        self.model_name = model_name
        self.client = ChatClient(model=model_name, temperature=0.0, max_tokens=4096, token_counter=token_counter)
        self._enc = _encoding_for_model("gpt-3.5-turbo")

    def run(self, text: str, query: str, log_file=None) -> str:
        if log_file:
            log_file.write(planner_agent_prompt.format(text=text, query=query))
        prompt = planner_agent_prompt.format(text=text, query=query)
        if len(self._enc.encode(prompt)) > 12000:
            return "Max Token Length Exceeded."
        return self.client.complete(prompt)


def _encoding_for_model(model_name: str):
    if tiktoken is None:
        class _FallbackEncoding:
            _token_re = re.compile(r"\\w+|[^\\w\\s]", re.UNICODE)

            def encode(self, text: str) -> List[int]:
                if not text:
                    return []
                return [0] * len(self._token_re.findall(text))

        return _FallbackEncoding()

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


pd.options.display.max_info_columns = 200

os.environ["TIKTOKEN_CACHE_DIR"] = str(BASELINE_DIR / "tmp")

actionMapping = {
    "FlightSearch": "flights",
    "AttractionSearch": "attractions",
    "GoogleDistanceMatrix": "googleDistanceMatrix",
    "AccommodationSearch": "accommodations",
    "RestaurantSearch": "restaurants",
    "Planner": "planner",
    "NotebookWrite": "notebook",
    "CitySearch": "cities",
}

TOOL_COST_KEY_BY_TOOL = {
    "flights": "Flights",
    "googleDistanceMatrix": "GoogleDistanceMatrix",
    "restaurants": "Restaurants",
    "attractions": "Attractions",
    "accommodations": "Accommodations",
    "cities": "CitySearch",
}

class CityError(Exception):
    pass

class DateError(Exception):
    pass

def _sleep_for_llm_error(exc: Exception) -> float:
    if isinstance(exc, LLMRateLimitError):
        return 60.0
    return 5.0

def _run_baseline_query(
    *,
    idx: int,
    query: str,
    model_name: str,
    tools_list: List[str],
    tool_costs: Mapping[str, float],
    per_query_dir: str,
) -> Dict[str, object]:
    token_counter = TokenCounter()
    tool_counter = ToolCallCounter(tool_costs=dict(tool_costs))

    max_steps = 50 if _is_big_model(model_name) else 30
    agent = ReactAgent(
        None,
        tools=tools_list,
        max_steps=max_steps,
        react_llm_name=model_name,
        planner_llm_name=model_name,
        token_counter=token_counter,
        tool_counter=tool_counter,
    )

    q_start = time.perf_counter()
    error: Optional[str] = None
    try:
        planner_results, scratchpad, action_log = agent.run(query)
    except Exception as exc:  # pragma: no cover - network/tool dependent
        planner_results, scratchpad, action_log = "", "", []
        error = f"{type(exc).__name__}: {exc}"

    q_elapsed = time.perf_counter() - q_start

    q_record = {
        "idx": idx,
        "query": query,
        "result": planner_results,
        "scratchpad": scratchpad,
        "action_log": action_log,
        "error": error,
        "metrics": {
            "time_s": q_elapsed,
            "llm_calls": token_counter.calls,
            "prompt_tokens": token_counter.prompt_tokens,
            "prompt_cache_hit_tokens": token_counter.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": token_counter.prompt_cache_miss_tokens,
            "completion_tokens": token_counter.completion_tokens,
            "output_tokens": token_counter.completion_tokens,
            "total_tokens": token_counter.total_tokens,
            "approx_llm_calls": token_counter.approx_calls,
            "tool_calls": tool_counter.total_calls,
            "tool_cost": tool_counter.total_cost,
            "tool_by_tool": tool_counter.by_tool,
        },
    }

    per_query_path = Path(per_query_dir) / f"generated_plan_{idx}.json"
    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    with per_query_path.open("w", encoding="utf-8") as per_query_fp:
        json.dump(q_record, per_query_fp, indent=2, ensure_ascii=False)

    return {
        "idx": idx,
        "ok": error is None,
        "per_query_path": str(per_query_path),
        "metrics": q_record["metrics"],
    }

class ReactAgent:
    def __init__(self,
                 args,
                 mode: str = 'zero_shot',
                 tools: List[str] = None,
                 max_steps: int = 30,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name = 'gpt-3.5-turbo-1106',
                 planner_llm_name = 'gpt-3.5-turbo-1106',
                #  logs_path = '../logs/',
                 city_file_path: str | Path | None = None,
                 token_counter: Optional[TokenCounter] = None,
                 tool_counter: Optional[ToolCallCounter] = None,
                 ) -> None: 

        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name
        self.token_counter = token_counter
        self.tool_counter = tool_counter

        if self.mode == 'zero_shot':
            self.agent_prompt = zeroshot_react_agent_prompt

        self.json_log = []

        self.current_observation = ''
        self.current_data = None

        stop_list = ["\n"]
        model_norm = str(react_llm_name or "").lower()
        if "gpt-3.5" in model_norm:
            temperature = 1.0
            self.max_token_length = 15000
        else:
            temperature = 0.0
            self.max_token_length = 30000

        default_step_max = "512" if _is_big_model(react_llm_name) else "128"
        step_max_tokens = int(os.getenv("BASELINE_STEP_MAX_TOKENS", default_step_max))

        self.client = ChatClient(
            model=react_llm_name,
            temperature=temperature,
            max_tokens=step_max_tokens,
            stop=stop_list,
            token_counter=token_counter,
        )


        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        # print(self.retry_record)

        self.last_actions = []

        # self.log_path = logs_path + datetime.now().strftime('%Y%m%d%H%M%S') + '.out'
        # self.log_file = open(self.log_path, 'a+')

        # print("logs will be stored in " + self.log_path)

        if city_file_path is None:
            city_file_path = travel_dataset_root() / "database" / "background" / "citySet.txt"
        self.city_set = self.load_city(city_set_path=str(city_file_path))

        self.enc = _encoding_for_model("gpt-3.5-turbo")

        self.__reset_agent()

    def _record_tool_call(self, tool_key: str) -> None:
        if self.tool_counter is None:
            return
        tool_name = TOOL_COST_KEY_BY_TOOL.get(tool_key)
        if not tool_name:
            return
        self.tool_counter.record(tool_name)

    def run(self, query, reset=True) -> None:

        self.query = query
        
        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:

        self.json_log.append({"step": self.step_n, "thought":"",
                              "action": "", "observation": "", "state":""})

        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent(expect_action=False)

        print(self.scratchpad.split('\n')[-1])
        self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:',"")
        # self.log_file.write(self.scratchpad.split('\n')[-1] + '\n')


        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent(expect_action=True)

        if action == None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += ' ' + action


        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:',"")


        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            # self.log_file.write("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return


        # action_type, action_arg = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        # self.log_file.write(self.scratchpad.split('\n')[-1]+'\n')

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '

        if action == None or action == '' or action == '\n':
            action_type = None 
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        
        else:
            action_type, action_arg = parse_action(action)
            
            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                elif action_type not in actionMapping:
                    pending_action = 'invalidAction'
                
                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return
                    
                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"invalidAction early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

            if action_type == 'FlightSearch':
                try:
                    flight_args = _split_tool_args(action_arg, expected=3)
                    if len(flight_args) != 3:
                        raise ValueError(
                            "FlightSearch expects 3 args: Departure City, Destination City, Date (YYYY-MM-DD)."
                        )
                    dep_city, dest_city, flight_date = flight_args
                    if validate_date_format(flight_date) and validate_city_format(dep_city,self.city_set ) and validate_city_format(dest_city,self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self._record_tool_call("flights")
                        self.current_data = self.tools['flights'].run(dep_city, dest_city, flight_date)
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += self.current_observation 
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except DateError:
                    self.retry_record['flights'] += 1
                    flight_args = _split_tool_args(action_arg, expected=3)
                    bad_date = flight_args[2] if len(flight_args) >= 3 else action_arg
                    self.current_observation = f"'{bad_date}' is not in the format YYYY-MM-DD"
                    self.scratchpad += f"'{bad_date}' is not in the format YYYY-MM-DD"
                    self.json_log[-1]['state'] = f'Illegal args. DateError'

                except ValueError as e:
                    self.retry_record['flights'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['flights'] += 1
                    self.current_observation = f'Illegal Flight Search. Please try again.'
                    self.scratchpad += f'Illegal Flight Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AttractionSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self._record_tool_call("attractions")
                        self.current_data = self.tools['attractions'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    self.retry_record['attractions'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['attractions'] += 1
                    self.current_observation = f'Illegal Attraction Search. Please try again.'
                    self.scratchpad += f'Illegal Attraction Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AccommodationSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self._record_tool_call("accommodations")
                        self.current_data = self.tools['accommodations'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e :
                    self.retry_record['accommodations'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['accommodations'] += 1
                    self.current_observation = f'Illegal Accommodation Search. Please try again.'
                    self.scratchpad += f'Illegal Accommodation Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'RestaurantSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self._record_tool_call("restaurants")
                        self.current_data = self.tools['restaurants'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['restaurants'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['restaurants'] += 1
                    self.current_observation = f'Illegal Restaurant Search. Please try again.'
                    self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'
                    
            elif action_type == "CitySearch":
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    # self.current_data = self.tools['cities'].run(action_arg)
                    self._record_tool_call("cities")
                    self.current_observation = to_string(self.tools['cities'].run(action_arg)).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['cities'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. State Error'

                except Exception as e:
                    print(e)
                    self.retry_record['cities'] += 1
                    self.current_observation = f'Illegal City Search. Please try again.'
                    self.scratchpad += f'Illegal City Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'


            elif action_type == 'GoogleDistanceMatrix':

                try:
                    dist_args = _split_tool_args(action_arg, expected=3)
                    if len(dist_args) != 3:
                        raise ValueError(
                            "GoogleDistanceMatrix expects 3 args: Origin, Destination, Mode (self-driving|taxi)."
                        )
                    origin, destination, mode = dist_args
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self._record_tool_call("googleDistanceMatrix")
                    self.current_data = self.tools['googleDistanceMatrix'].run(origin, destination, mode)
                    self.current_observation =  to_string(self.current_data)
                    self.scratchpad += self.current_observation 
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record['googleDistanceMatrix'] += 1
                    self.current_observation = f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.scratchpad += f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'
            
            
            elif action_type == 'NotebookWrite':
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_observation = str(self.tools['notebook'].write(self.current_data, action_arg))
                    self.scratchpad  +=  self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record['notebook'] += 1
                    self.current_observation = f'{e}'
                    self.scratchpad += f'{e}'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'
            

            elif action_type == "Planner":
                # try:

                    self.current_observation = str(self.tools['planner'].run(str(self.tools['notebook'].list_all()),action_arg))
                    self.scratchpad  +=  self.current_observation
                    self.answer = self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

            else:
                self.retry_record['invalidAction'] += 1
                self.current_observation = 'Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / ' \
                                   'AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query].'
                self.scratchpad += self.current_observation
                self.json_log[-1]['state'] = f'invalidAction'

        if action == None or action == '' or action == '\n':
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            # write(f'Observation {self.step_n}: ' + "Your action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
        else:
            print(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            self.json_log[-1]['observation'] = self.current_observation

        self.step_n += 1

        # 

        if action_type and action_type == 'Planner' and self.retry_record['planner']==0:
            
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self, *, expect_action: bool) -> str:
        while True:
            try:
                raw = self.client.complete(self._build_agent_prompt(expect_action=expect_action))
                if expect_action:
                    return _canonical_action(raw)
                return format_step(raw)
            except Exception as exc:
                print(f"LLM error: {exc}")
                prompt = self._build_agent_prompt(expect_action=expect_action)
                print(prompt)
                try:
                    print(len(self.enc.encode(prompt)))
                except Exception:
                    pass
                time.sleep(_sleep_for_llm_error(exc))

    def _build_agent_prompt(self, *, expect_action: bool = False) -> str:
        if self.mode == "zero_shot":
            scratchpad = self.scratchpad
            prompt = self.agent_prompt.format(query=self.query, scratchpad=scratchpad)
            # Keep prompts compact for smaller models (and to improve cache-hit ratios).
            prompt_budget = min(int(self.max_token_length), 8000)
            try:
                if len(self.enc.encode(prompt)) > prompt_budget:
                    scratchpad = truncate_scratchpad(scratchpad, n_tokens=1600, tokenizer=self.enc)
                    prompt = self.agent_prompt.format(query=self.query, scratchpad=scratchpad)
            except Exception:
                pass

            if expect_action:
                prompt += "\n\nIMPORTANT: Return ONLY the next Action as ToolName[args] with no other text."
            else:
                prompt += "\n\nIMPORTANT: Return ONLY the next Thought in one short sentence (no tool call)."
            return prompt

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt(expect_action=False))) > self.max_token_length)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad: str = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()
    
    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0


    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            if tool_name == "planner":
                tools_map[tool_name] = PlannerTool(
                    planner_model_name or self.planner_name,
                    token_counter=self.token_counter,
                )
                continue

            module = importlib.import_module(f"task_helper.travel.tools.{tool_name}.apis")
            cls_name = tool_name[0].upper() + tool_name[1:]
            tools_map[tool_name] = getattr(module, cls_name)()
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, 'r').read().strip().split('\n')
        for unit in lines:
            city_set.append(unit)
        return city_set

### String Stuff ###
gpt2_enc = _encoding_for_model("text-davinci-003")

def _split_tool_args(arg_text: str, *, expected: Optional[int] = None) -> List[str]:
    """Split tool args like 'A, B, C' or 'A,B,C' into fields.

    If there are more than `expected` segments, extra commas are kept in the last field.
    """
    raw = str(arg_text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p != ""]
    if expected is None:
        return parts
    if len(parts) <= expected:
        return parts
    head = parts[: expected - 1]
    tail = ",".join(parts[expected - 1 :]).strip()
    return [*head, tail]

def parse_action(string):
    if not string:
        return None, None
    text = str(string).strip().strip("`").strip()
    if not text:
        return None, None
    # Be forgiving: smaller models often return extra text like "Action: Tool[...]".
    match = re.search(r"(\w+)\[(.*?)\]", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)

def _canonical_action(text: str) -> str:
    action_type, action_arg = parse_action(text)
    if not action_type:
        return str(text or "").strip()
    return f"{action_type}[{action_arg}]"

def format_step(step: str) -> str:
    if step is None:
        return ""
    raw = str(step).strip()
    if not raw:
        return ""
    # Simulate newline stop for models/endpoints that don't support `stop`:
    # take the first non-empty line.
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return ""



def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))


def remove_observation_lines(text, step_n):
    pattern = re.compile(rf'^Observation {step_n}.*', re.MULTILINE)
    return pattern.sub('', text)

def validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    if not re.match(pattern, date_str):
        raise DateError
    return True

def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")
    return True

def parse_args_string(s: str) -> dict:
    # Split the string by commas
    segments = s.split(",")
    
    # Initialize an empty dictionary to store the results
    result = {}
    
    for segment in segments:
        # Check for various operators
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue  # If no recognized operator is found, skip to the next segment
                
        # Strip spaces and single quotes
        key = key.strip()
        value = value.strip().strip("'")
        
        # Store the result with the operator included
        result[key] = (operator, value)
        
    return result

def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)

if __name__ == '__main__':
    tools_list = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model_name", type=str, default=os.getenv("MEMPLAN_LLM_MODEL", "gpt-3.5-turbo-1106"))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "output" / "travel" / "baseline"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of queries (0 = all).")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("BASELINE_WORKERS", "1")),
        help="Number of parallel workers (threads). Use 0 for all CPUs (beware API rate limits).",
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Skip queries that already have a non-empty result and no error in per_query_dir.",
    )
    args = parser.parse_args()

    _ensure_dotenv_loaded()

    out_root = Path(args.output_dir).expanduser()
    if not out_root.is_absolute():
        out_root = (PROJECT_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model_slug = _model_slug(args.model_name)
    run_prefix = f"two_stage_{model_slug}_{args.set_type}"
    run_dir = out_root / f"{model_slug}_{args.set_type}"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_query_dir = run_dir / run_prefix
    per_query_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = travel_dataset_root() / f"{args.set_type}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Split not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        query_data_list = [row for row in reader]

    if args.limit and args.limit > 0:
        query_data_list = query_data_list[: args.limit]

    tool_costs = _load_tool_costs()
    workers = _resolve_workers(args.workers)

    run_start_iso = _iso_now()
    run_start = time.perf_counter()

    results_jsonl_path = run_dir / f"{run_prefix}.jsonl"
    cost_txt_path = run_dir / "cost.txt"

    # Run queries in parallel (threads). Each worker writes `generated_plan_<idx>.json`.
    summaries: Dict[int, Dict[str, object]] = {}
    processed_indices: List[int] = []

    def _needs_rerun(idx: int) -> bool:
        path = per_query_dir / f"generated_plan_{idx}.json"
        if not path.exists():
            return True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return True
        if not isinstance(payload, dict):
            return True
        if payload.get("error"):
            return True
        result = str(payload.get("result") or "")
        return not bool(result.strip())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for idx, row in enumerate(query_data_list, start=1):
            query = row.get("query") or ""
            if not query:
                continue
            processed_indices.append(idx)
            if args.resume_failed and not _needs_rerun(idx):
                path = per_query_dir / f"generated_plan_{idx}.json"
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                summaries[idx] = {
                    "idx": idx,
                    "ok": True,
                    "per_query_path": str(path),
                    "metrics": payload.get("metrics") if isinstance(payload, dict) else {},
                }
                continue
            futures.append(
                pool.submit(
                    _run_baseline_query,
                    idx=idx,
                    query=query,
                    model_name=args.model_name,
                    tools_list=tools_list,
                    tool_costs=tool_costs,
                    per_query_dir=str(per_query_dir),
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures)):
            summary = fut.result()
            summaries[int(summary["idx"])] = summary

    run_elapsed = time.perf_counter() - run_start
    run_end_iso = _iso_now()

    # Deterministic JSONL ordering by idx.
    with results_jsonl_path.open("w", encoding="utf-8") as results_fp:
        for idx in processed_indices:
            path = per_query_dir / f"generated_plan_{idx}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            results_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # Aggregate token/tool usage from worker summaries.
    total_llm_calls = 0
    total_prompt_tokens = 0
    total_cache_hit = 0
    total_cache_miss = 0
    total_output_tokens = 0
    total_total_tokens = 0
    total_approx_calls = 0
    total_tool_calls = 0
    total_tool_cost = 0.0
    by_tool: Dict[str, Dict[str, float]] = {}

    for idx in processed_indices:
        summary = summaries.get(idx) or {}
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        total_llm_calls += int(metrics.get("llm_calls") or 0)
        total_prompt_tokens += int(metrics.get("prompt_tokens") or 0)
        total_cache_hit += int(metrics.get("prompt_cache_hit_tokens") or 0)
        total_cache_miss += int(metrics.get("prompt_cache_miss_tokens") or 0)
        total_output_tokens += int(metrics.get("completion_tokens") or metrics.get("output_tokens") or 0)
        total_total_tokens += int(metrics.get("total_tokens") or 0)
        total_approx_calls += int(metrics.get("approx_llm_calls") or 0)
        total_tool_calls += int(metrics.get("tool_calls") or 0)
        total_tool_cost += float(metrics.get("tool_cost") or 0.0)
        _merge_tool_usage(by_tool, metrics.get("tool_by_tool") or {})

    with cost_txt_path.open("w", encoding="utf-8") as fp:
        fp.write(f"run_start={run_start_iso}\n")
        fp.write("task=travel\n")
        fp.write(f"split={args.set_type}\n")
        fp.write(f"model={args.model_name}\n")
        fp.write(f"model_slug={model_slug}\n")
        fp.write(f"workers={args.workers} (effective={workers})\n")
        fp.write(f"dataset_path={dataset_path}\n")
        fp.write(f"output_dir={run_dir}\n")
        fp.write(f"per_query_dir={per_query_dir}\n")
        fp.write(f"results_jsonl={results_jsonl_path}\n")
        fp.write(f"queries_total={len(query_data_list)}\n")
        fp.write(f"queries_processed={len(processed_indices)}\n")
        fp.write(f"limit={args.limit}\n")
        fp.write(f"run_end={run_end_iso}\n")
        fp.write(f"elapsed_s_total={run_elapsed}\n")

        fp.write(
            "llm_tokens: "
            f"calls={total_llm_calls} prompt_cache_hit={total_cache_hit} "
            f"prompt_cache_miss={total_cache_miss} "
            f"output={total_output_tokens} total={total_total_tokens} "
            f"approx_calls={total_approx_calls}\n"
        )
        fp.write(
            "tool_calls: "
            f"calls={total_tool_calls} cost={total_tool_cost} "
            f"by_tool={json.dumps(by_tool, ensure_ascii=False, sort_keys=True)}\n"
        )
        fp.write(
            f"summary: time_s_total={run_elapsed} tokens_total={total_total_tokens} tool_calls_cost={total_tool_cost}\n"
        )

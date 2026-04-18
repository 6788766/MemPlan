import re
import os
import pandas as pd
import random
import ast
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.chat_models.anthropic import ChatAnthropic
from langchain_community.chat_models.anyscale import ChatAnyscale
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from langchain.agents import initialize_agent, AgentType  # type: ignore[attr-defined]
    _WORKBENCH_LANGCHAIN_OK = True
except Exception:
    initialize_agent = None  # type: ignore[assignment]
    AgentType = None  # type: ignore[assignment]
    _WORKBENCH_LANGCHAIN_OK = False
import csv
from baseline.workbench.src.tools import (
    analytics,
    calendar,
    company_directory,
    customer_relationship_manager,
    email,
    project_management,
)
from baseline.workbench.src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
from baseline.workbench.src.tools.toolkits import (
    calendar_toolkit,
    email_toolkit,
    analytics_toolkit,
    project_management_toolkit,
    customer_relationship_manager_toolkit,
    company_directory_toolkit,
    tools_with_side_effects,
)

try:  # Best-effort token usage tracking (OpenAI-compatible responses).
    from langchain_community.callbacks import get_openai_callback  # type: ignore[import-not-found]
except Exception:
    get_openai_callback = None  # type: ignore[assignment]

try:  # Rich token usage (cached vs uncached prompt tokens) via callback handler.
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore[import-not-found]
    from langchain_core.outputs import LLMResult  # type: ignore[import-not-found]
except Exception:
    BaseCallbackHandler = None  # type: ignore[assignment]
    LLMResult = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_LOADED = False


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
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


class _NoStopChatOpenAI(ChatOpenAI):
    """
    Some providers/models reject the `stop` parameter. WorkBench's LangChain agent
    passes stop sequences by default; this wrapper strips them from requests.
    """

    def _generate(self, messages, stop=None, run_manager=None, stream=None, **kwargs):  # type: ignore[override]
        return super()._generate(messages, stop=None, run_manager=run_manager, stream=stream, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, stream=None, **kwargs):  # type: ignore[override]
        return await super()._agenerate(messages, stop=None, run_manager=run_manager, stream=stream, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        return super()._stream(messages, stop=None, run_manager=run_manager, **kwargs)

def _safe_int(value):
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _extract_usage(payload):
    """
    Extract token usage fields from OpenAI/DeepSeek-style payloads.
    Returns dict with: prompt_tokens, completion_tokens, total_tokens,
    prompt_cache_hit_tokens, prompt_cache_miss_tokens.
    """
    out = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    if not isinstance(payload, dict):
        return out

    # Common OpenAI-style: {"token_usage": {...}} or {"usage": {...}}
    usage = None
    if isinstance(payload.get("token_usage"), dict):
        usage = payload.get("token_usage")
    elif isinstance(payload.get("usage"), dict):
        usage = payload.get("usage")
    if isinstance(usage, dict):
        out["prompt_tokens"] = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        out["completion_tokens"] = _safe_int(usage.get("completion_tokens") or usage.get("output_tokens"))
        out["total_tokens"] = _safe_int(usage.get("total_tokens"))

        # OpenAI cached tokens: usage.prompt_tokens_details.cached_tokens
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = _safe_int(details.get("cached_tokens"))
            out["prompt_cache_hit_tokens"] = cached
            if out["prompt_tokens"]:
                out["prompt_cache_miss_tokens"] = max(out["prompt_tokens"] - cached, 0)

        # DeepSeek-style cache fields
        out["prompt_cache_hit_tokens"] = max(out["prompt_cache_hit_tokens"], _safe_int(usage.get("prompt_cache_hit_tokens")))
        out["prompt_cache_miss_tokens"] = max(out["prompt_cache_miss_tokens"], _safe_int(usage.get("prompt_cache_miss_tokens")))
        return out

    # Some providers put these at the top-level.
    out["prompt_tokens"] = _safe_int(payload.get("prompt_tokens") or payload.get("input_tokens"))
    out["completion_tokens"] = _safe_int(payload.get("completion_tokens") or payload.get("output_tokens"))
    out["total_tokens"] = _safe_int(payload.get("total_tokens"))
    out["prompt_cache_hit_tokens"] = _safe_int(payload.get("prompt_cache_hit_tokens"))
    out["prompt_cache_miss_tokens"] = _safe_int(payload.get("prompt_cache_miss_tokens"))
    return out


def _merge_usage(a, b):
    return {
        "prompt_tokens": _safe_int(a.get("prompt_tokens")) + _safe_int(b.get("prompt_tokens")),
        "completion_tokens": _safe_int(a.get("completion_tokens")) + _safe_int(b.get("completion_tokens")),
        "total_tokens": _safe_int(a.get("total_tokens")) + _safe_int(b.get("total_tokens")),
        "prompt_cache_hit_tokens": _safe_int(a.get("prompt_cache_hit_tokens")) + _safe_int(b.get("prompt_cache_hit_tokens")),
        "prompt_cache_miss_tokens": _safe_int(a.get("prompt_cache_miss_tokens")) + _safe_int(b.get("prompt_cache_miss_tokens")),
        "total_cost_usd": float(a.get("total_cost_usd") or 0.0) + float(b.get("total_cost_usd") or 0.0),
    }


class _UsageCallback(BaseCallbackHandler):  # type: ignore[misc]
    def __init__(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }

    def on_llm_end(self, response: "LLMResult", **kwargs):  # type: ignore[override]
        try:
            if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                self.usage = _merge_usage(self.usage, _extract_usage(response.llm_output))
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list or []:
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "response_metadata", None)
                    if isinstance(meta, dict):
                        self.usage = _merge_usage(self.usage, _extract_usage(meta))
        except Exception:
            return


def _run_agent(agent, query):
    """
    Run the agent and return (response, usage_dict).
    """
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "total_cost_usd": 0.0,
    }
    handler = _UsageCallback() if BaseCallbackHandler is not None else None

    if get_openai_callback is None and handler is None:
        return agent({"input": query}), usage

    cb_usage = None
    if get_openai_callback is not None:
        with get_openai_callback() as cb:
            if handler is None:
                resp = agent({"input": query})
            else:
                try:
                    resp = agent({"input": query}, callbacks=[handler])
                except TypeError:
                    resp = agent({"input": query})
            cb_usage = {
                "prompt_tokens": int(getattr(cb, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(cb, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(cb, "total_tokens", 0) or 0),
                "total_cost_usd": float(getattr(cb, "total_cost", 0.0) or 0.0),
            }
    else:
        try:
            resp = agent({"input": query}, callbacks=[handler])
        except TypeError:
            resp = agent({"input": query})

    if handler is not None:
        usage = _merge_usage(usage, handler.usage)
    if cb_usage is not None:
        # Prefer callback totals for total tokens/cost, but keep cache hit/miss from handler.
        usage["prompt_tokens"] = cb_usage["prompt_tokens"]
        usage["completion_tokens"] = cb_usage["completion_tokens"]
        usage["total_tokens"] = cb_usage["total_tokens"]
        usage["total_cost_usd"] = cb_usage["total_cost_usd"]
    return resp, usage


def _tqdm():  # pragma: no cover
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm
    except Exception:
        return None


def _progress_iter(iterable, *, total: int, desc: str):
    tqdm = _tqdm()
    if tqdm is None:
        # Fallback: no progress bar, but keep a lightweight counter.
        for i, item in enumerate(iterable, start=1):
            if i == 1 or i % 10 == 0 or i == total:
                print(f"[{desc}] {i}/{total}")
            yield item
        return
    yield from tqdm(iterable, total=total, desc=desc)


DOMAINS = [calendar, email, analytics, project_management, customer_relationship_manager]
AVAILABLE_LLMS = [
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "deepseek-chat",
    # Legacy (WorkBench original)
    "gpt-4",
    "gpt-3.5",
    "claude-2",
    "llama2-70b",
    "mistral-8x7B",
]


def _read_key(*, filename: str, env_var: str) -> str:
    _ensure_dotenv_loaded()
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    path = Path(filename)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise FileNotFoundError(
        f"Missing API key: set {env_var} (preferred) or create {filename} in the repo root."
    )


def _make_chat_openai(*, model_name: str, api_key: str, base_url: str | None) -> object:
    """
    Create a ChatOpenAI instance with best-effort compatibility across langchain-openai versions.
    """
    cls = _NoStopChatOpenAI if model_name.startswith("gpt-5") or model_name == "deepseek-chat" else ChatOpenAI
    # Some models only support default temperature (1.0); they reject explicit 0.0.
    temperature = 0.0
    if model_name in ("gpt-5-mini", "gpt-5-nano", "deepseek-chat"):
        temperature = 1.0
    kwargs = {
        "model_name": model_name,
        "openai_api_key": api_key,
        "temperature": temperature,
        "model_kwargs": {"seed": 42},
    }
    if base_url:
        # Older langchain-openai uses openai_api_base; newer may accept base_url.
        kwargs["openai_api_base"] = base_url
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("openai_api_base", None)
        if base_url:
            kwargs["base_url"] = base_url
        return cls(**kwargs)


def _make_llm(*, model_name: str) -> object:
    """
    Map MemPlan-supported model names to a LangChain LLM instance.

    Keys are expected via `.env`:
      - OpenAI: OPENAI_API_KEY (optional OPENAI_BASE_URL)
      - DeepSeek (OpenAI-compatible): DEEPSEEK_API_KEY (optional DEEPSEEK_BASE_URL)
    """
    name = str(model_name or "").strip()
    if name in ("gpt-3.5",):
        openai_key = _read_key(filename="openai_key.txt", env_var="OPENAI_API_KEY")
        return OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            openai_api_key=openai_key,
            temperature=0,
            model_kwargs={"seed": 42},
        )

    if name in ("gpt-4",):
        openai_key = _read_key(filename="openai_key.txt", env_var="OPENAI_API_KEY")
        return _make_chat_openai(
            model_name="gpt-4-0125-preview",
            api_key=openai_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
        )

    if name.startswith("gpt-5"):
        openai_key = _read_key(filename="openai_key.txt", env_var="OPENAI_API_KEY")
        return _make_chat_openai(
            model_name=name,
            api_key=openai_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
        )

    if name in ("deepseek-chat",):
        deepseek_key = _read_key(filename="deepseek_key.txt", env_var="DEEPSEEK_API_KEY")
        base_url = (
            os.environ.get("DEEPSEEK_BASE_URL", "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or "https://api.deepseek.com/v1"
        )
        return _make_chat_openai(
            model_name=name,
            api_key=deepseek_key,
            base_url=base_url or None,
        )

    if name == "claude-2":
        anthropic_key = _read_key(filename="anthropic_key.txt", env_var="ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model_name="claude-2",
            anthropic_api_key=anthropic_key,
            temperature=0,
        )

    if name == "llama2-70b":
        anyscale_key = _read_key(filename="anyscale_key.txt", env_var="ANYSCALE_API_KEY")
        return ChatAnyscale(
            model="meta-llama/Llama-2-70b-chat-hf",
            anyscale_api_key=anyscale_key,
            temperature=0,
        )

    if name == "mistral-8x7B":
        anyscale_key = _read_key(filename="anyscale_key.txt", env_var="ANYSCALE_API_KEY")
        return ChatAnyscale(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            anyscale_api_key=anyscale_key,
            temperature=0,
        )

    raise ValueError("Invalid --model_name. Must be one of " + ", ".join(AVAILABLE_LLMS))


def convert_agent_action_to_function_call(action):
    """Converts langchain_core.agents.AgentAction to an API call"""
    args = []
    for k, v in action.tool_input.items():
        args.append(f'{k}="{v}"')
    return action.tool + ".func(" + ", ".join(args) + ")"


def execute_actions_and_reset_state(actions):
    """
    Executes a list of actions on the calendar and returns the resulting calendar events.

    Parameters
    ----------
    actions : list
        List of actions to be executed. Each action should be a function call.

    Returns
    -------
    success bool
        True if the actions were executed successfully.
    new_calendar_state pd.DataFrame
        The resulting calendar events after executing the actions.
    new_email_state pd.DataFrame
        The resulting emails after executing the actions.
    new_analytics_state pd.DataFrame
        The resulting analytics data after executing the actions.
    """
    for domain in DOMAINS:
        domain.reset_state()

    # Execute the actions
    for action in actions:
        try:
            eval(action)
        except:
            continue
    new_calendar_state = calendar.CALENDAR_EVENTS.copy()
    new_email_state = email.EMAILS.copy()
    new_analytics_state = analytics.PLOTS_DATA.copy()
    new_project_management_state = project_management.PROJECT_TASKS.copy()
    new_customer_relationship_manager_state = customer_relationship_manager.CRM_DATA.copy()

    # Reset the state of the tools
    for domain in DOMAINS:
        domain.reset_state()
    return (
        True,
        new_calendar_state,
        new_email_state,
        new_analytics_state,
        new_project_management_state,
        new_customer_relationship_manager_state,
    )


def end_date_minor_error(ground_truth, prediction):
    """Function to check if the end date is off by one day in the prediction

    Parameters
    ----------
    ground_truth : list
        List of ground truth actions as strings.
    prediction : list
        List of predicted actions as strings.

    Returns
    -------
    bool
        True if the end date is off by one day in the prediction.
    """
    matches = 0
    for func in ground_truth:
        if "2023-11-29" in func:
            if func.replace("2023-11-29", "2023-11-30") in prediction:
                matches += 1
    if len(ground_truth) == 0:
        return False
    return matches == len(ground_truth)


def meeting_start_time_error(ground_truth, prediction):
    """Function to check if the meeting start time is off where the agent predicts the wrong first available time

    Parameters
    ----------
    ground_truth : list
        List of ground truth actions as strings.
    prediction : list
        List of predicted actions as strings.

    Returns
    -------
    bool
        True if the meeting start time is off by one hour in the prediction.
    """
    matches = 0
    next_free_time_ground_truth = "13:00:00"
    common_error_times = ["09:00:00", "11:00:00", "15:00:00", "15:30:00"]
    for func in ground_truth:
        if next_free_time_ground_truth in func:
            for time in common_error_times:
                if func.replace(next_free_time_ground_truth, time) in prediction:
                    matches += 1
                    break
    if len(ground_truth) == 0:
        return False
    return matches == len(ground_truth)


def is_exact_match(predicted_actions, ground_truth_actions):
    """
    Checks if the predicted actions are an exact match to the ground truth actions.

    Parameters
    ----------
    predicted_actions : list
        List of predicted actions as strings.
    ground_truth_actions : list
        List of ground truth actions as strings.

    Returns
    -------
    bool
        True if the predicted actions are an exact match to the ground truth actions.

    """
    tools_with_side_effects_names = [str(function.name) for function in tools_with_side_effects]
    predicted_actions_with_side_effects = [
        action for action in predicted_actions if get_function_name(action) in tools_with_side_effects_names
    ]
    predicted_actions_with_side_effects = sorted([action.lower() for action in predicted_actions_with_side_effects])
    ground_truth_actions = sorted([action.lower() for action in ground_truth_actions])

    return predicted_actions_with_side_effects == ground_truth_actions


def get_function_name(action):
    """Extracts the function name from a string"""
    return ".".join(action.split("(")[0].split(".")[0:2])


def is_correct(predicted_actions, ground_truth_actions, error):
    """
    Checks if the prediction is correct by comparing the state change after executing the actions.

    Parameters
    ----------
    predicted_actions : list
        List of predicted actions as strings.
    ground_truth_actions : list
        List of ground truth actions as strings.
    error : str
        Error message from the prediction.

    Returns
    -------
    bool
        True if the predicted actions result in the same state change as the ground truth actions.

    """
    if error:
        return False
    (
        successful_execution,
        predicted_calendar_state,
        predicted_email_state,
        predicted_analytics_state,
        predicted_project_management_state,
        predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)
    (
        _,
        ground_truth_calendar_state,
        ground_truth_email_state,
        ground_truth_analytics_state,
        ground_truth_project_management_state,
        ground_truth_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(ground_truth_actions)

    def convert_strs_to_lowercase(df):
        # For some fields the case matters, so we don't convert them to lowercase
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    # We allow for case-insensitive comparison of strings for most fields
    predicted_calendar_state = convert_strs_to_lowercase(predicted_calendar_state)
    predicted_email_state = convert_strs_to_lowercase(predicted_email_state)
    predicted_analytics_state = convert_strs_to_lowercase(predicted_analytics_state)
    predicted_project_management_state = convert_strs_to_lowercase(predicted_project_management_state)
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(
        predicted_customer_relationship_manager_state
    )

    ground_truth_calendar_state = convert_strs_to_lowercase(ground_truth_calendar_state)
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_email_state)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_analytics_state)
    ground_truth_project_management_state = convert_strs_to_lowercase(ground_truth_project_management_state)
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(
        ground_truth_customer_relationship_manager_state
    )

    return (
        successful_execution
        and predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )


def extract_function_names(s):
    """Extracts function names from a string"""
    return re.findall(r"(\b\w+\.\w+)\(", s)


def has_side_effects(predicted_actions, ground_truth_actions):
    """
    Checks if the predicted actions have side effects by comparing the state change after executing the actions.

    Parameters
    ----------
    predicted_actions : list
        List of predicted actions as strings.
    ground_truth_actions : list
        List of ground truth actions as strings.

    Returns
    -------
    bool
        True if the predicted actions result in different state change than the ground truth actions.

    """
    for domain in DOMAINS:
        domain.reset_state()
    original_state = {
        "calendar": calendar.CALENDAR_EVENTS.copy(),
        "email": email.EMAILS.copy(),
        "analytics": analytics.PLOTS_DATA.copy(),
        "project_management": project_management.PROJECT_TASKS.copy(),
        "customer_relationship_manager": customer_relationship_manager.CRM_DATA.copy(),
    }
    (
        successful_execution,
        predicted_calendar_state,
        predicted_email_state,
        predicted_analytics_state,
        predicted_project_management_state,
        predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)

    state_changed = not predicted_calendar_state.equals(original_state["calendar"])
    state_changed |= not predicted_email_state.equals(original_state["email"])
    state_changed |= not predicted_analytics_state.equals(original_state["analytics"])
    state_changed |= not predicted_project_management_state.equals(original_state["project_management"])
    state_changed |= not predicted_customer_relationship_manager_state.equals(
        original_state["customer_relationship_manager"]
    )

    errors = ""  # Errors like exceeding the context window or running out of time don't have side effects, so we assume no errors
    correct = is_correct(predicted_actions, ground_truth_actions, errors)
    return state_changed and not correct


def generate_query_and_answer(template):
    """Generates query and answer from template."""
    logic = template["logic"]()
    if "alternative_queries" in template:
        possible_queries = [template["query"]] + template["alternative_queries"]
        query_template = random.choice(possible_queries)
        query = query_template.format(**logic)
    else:
        query_template = template["query"]
        query = query_template.format(**logic)
    answer = logic["answer"]
    domains = template.get("domains", [])
    return {
        "query": query,
        "answer": answer,
        "base_template": template["query"],
        "chosen_template": query_template,
        "domains": domains,
    }


def generate_all_queries_and_answers(templates, max_queries_per_template, verbose=False):
    """Generates a limited number of unique queries and answers for each template."""
    generated_queries_and_answers = []
    for template in templates:
        queries_generated_for_template = 0
        while queries_generated_for_template < max_queries_per_template:
            q_and_a = generate_query_and_answer(template)
            queries = [q["query"] for q in generated_queries_and_answers]
            if q_and_a["query"] not in queries:
                generated_queries_and_answers.append(q_and_a)
                queries_generated_for_template += 1

    if verbose:
        for query_and_answer in generated_queries_and_answers:
            print(f"Base template:   {query_and_answer['base_template']}")
            print(f"Chosen template: {query_and_answer['chosen_template']}")
            print(f"Query:           {query_and_answer['query']}")
            print(f"Answer:          {query_and_answer['answer']}")
            print("--------------------------------------------")

    return generated_queries_and_answers


def calculate_metrics(ground_truth_df, predictions_df, print_errors=True):
    """"""
    predictions = predictions_df.rename(columns={"function_calls": "prediction"})
    predictions = predictions.fillna("")

    ground_truth = ground_truth_df.rename(columns={"answer": "ground_truth"})
    df = predictions.merge(ground_truth, on="query")
    assert (
        len(predictions) == len(ground_truth) == len(df)
    ), f"{len(predictions)} predictions does not match {len(ground_truth_df)} ground truth answers. Check that the predictions and ground truth are for the same queries."

    # Replace all newlines with "\\n" for all actions
    df["prediction"] = df["prediction"].apply(lambda actions: [action.replace("\n", "\\n") for action in actions])
    df["ground_truth"] = df["ground_truth"].apply(lambda actions: [action.replace("\n", "\\n") for action in actions])

    df["exact_match"] = [is_exact_match(pred, gt) for pred, gt in zip(df["prediction"], df["ground_truth"])]
    df["correct"] = [
        is_correct(pred, gt, error) for pred, gt, error in zip(df["prediction"], df["ground_truth"], df["error"])
    ]
    df["unwanted_side_effects"] = [has_side_effects(pred, gt) for pred, gt in zip(df["prediction"], df["ground_truth"])]
    df["no_actions"] = [not len(pred) for pred in df["prediction"]]
    # wrong email if @example is in the prediction and @atlas is not in the prediction. Prediction is a list so needs to be converted to a string
    df["wrong_email"] = [("@example" in str(pred)) and ("@atlas" not in str(pred)) for pred in df["prediction"]]
    df["wrong_email"] = df["wrong_email"] & ~df["correct"]
    # Puts in end of November to plot instead of 29th november, but everything else matches
    df["end_date_minor_error"] = [
        end_date_minor_error(gt, pred) for gt, pred in zip(df["ground_truth"], df["prediction"])
    ]
    df["end_date_minor_error"] = df["end_date_minor_error"] & ~df["correct"]
    df["meeting_start_time_error"] = [
        meeting_start_time_error(gt, pred) for gt, pred in zip(df["ground_truth"], df["prediction"])
    ]
    df["meeting_start_time_error"] = df["meeting_start_time_error"] & ~df["correct"]

    # print out the queries that were not answered correctly
    if print_errors:
        print("--------------------------------------------")
        print("--------------------------------------------")
        print("ERRORS without unwanted side effects:")
        print("--------------------------------------------")
        print("--------------------------------------------")
        for _, row in df[~df["correct"] & ~df["unwanted_side_effects"]].iterrows():
            if (
                not row["wrong_email"]
                and not row["no_actions"]
                and not row["end_date_minor_error"]
                and not row["meeting_start_time_error"]
            ):
                # full response string to dict
                print("--------------------------------------------")
                print(f"Query:")
                print(f"    {row['query']}")
                print()
                print(f"Prediction:")
                for action in row["prediction"]:
                    print(f"    {action}")
                print()
                print(f"Ground truth:")
                for action in row["ground_truth"]:
                    print(f"    {action}")
                print()
                print(f"Unwanted side effects: {row['unwanted_side_effects']}")
                print()
                print(f"Error: {row['error']}")
                print("")
                print(f"Output:")
                output = get_output(row["full_response"])
                print(f"    {output}")
        print("--------------------------------------------")
        print("--------------------------------------------")
        print("ERRORS with unwanted side effects:")
        print("--------------------------------------------")
        print("--------------------------------------------")
        for _, row in df[~df["correct"] & df["unwanted_side_effects"]].iterrows():
            if (
                not row["wrong_email"]
                and not row["no_actions"]
                and not row["end_date_minor_error"]
                and not row["meeting_start_time_error"]
            ):
                # full response string to dict
                print("--------------------------------------------")
                print(f"Query:")
                print(f"    {row['query']}")
                print()
                print(f"Prediction:")
                for action in row["prediction"]:
                    print(f"    {action}")
                print()
                print(f"Ground truth:")
                for action in row["ground_truth"]:
                    print(f"    {action}")
                print()
                print(f"Unwanted side effects: {row['unwanted_side_effects']}")
                print()
                print(f"Meeting start time error: {row['meeting_start_time_error']}")
                print(f"Error: {row['error']}")
                print("")
                print(f"Output:")
                output = get_output(row["full_response"])
                print(f"    {output}")

    num_errors_without_side_effects = len(df[(~df["correct"]) & ~df["unwanted_side_effects"]])
    num_errors_with_side_effects = len(df[(~df["correct"]) & df["unwanted_side_effects"]])
    print(f"Accuracy: {round(df['correct'].mean() * 100, 2)}% ({df['correct'].sum()} out of {len(df)})")
    print(
        f"Errors without unwanted side effects: {round(num_errors_without_side_effects / len(df) * 100, 2)}% ({num_errors_without_side_effects} out of {len(df)})"
    )
    print(
        f"Errors with unwanted side effects: {round(num_errors_with_side_effects / len(df) * 100, 2)}% ({num_errors_with_side_effects} out of {len(df)})"
    )

    num_failed_to_follow_react = len(df[(~df["correct"]) & ~df["unwanted_side_effects"] & df["no_actions"]])
    num_wrong_email_no_side_effects = len(df[(~df["correct"]) & df["wrong_email"] & ~df["unwanted_side_effects"]])
    num_meeting_start_time_error_no_side_effects = len(
        df[(~df["correct"]) & df["meeting_start_time_error"] & ~df["unwanted_side_effects"]]
    )

    if print_errors:
        print(
            f"Wrong email, no side effects: {round(num_wrong_email_no_side_effects / len(df) * 100, 2)}% ({num_wrong_email_no_side_effects} out of {len(df)})"
        )
        print(
            f"Didn't follow REACT framework, no side effects: {round(num_failed_to_follow_react / len(df) * 100, 2)}% ({num_failed_to_follow_react} out of {len(df)})"
        )
        print(
            f"Meeting start time error, no side effects: {round(num_meeting_start_time_error_no_side_effects / len(df) * 100, 2)}% ({num_meeting_start_time_error_no_side_effects} out of {len(df)})"
        )

    num_wrong_email_with_side_effects = len(
        df[
            (~df["correct"])
            & df["wrong_email"]
            & df["unwanted_side_effects"]
            & ~df["end_date_minor_error"]
            & ~df["meeting_start_time_error"]
        ]
    )
    num_end_date_minor_error = len(
        df[
            (~df["correct"])
            & df["end_date_minor_error"]
            & df["unwanted_side_effects"]
            & ~df["wrong_email"]
            & ~df["meeting_start_time_error"]
        ]
    )
    num_meeting_start_time_error_with_side_effects = len(
        df[(~df["correct"]) & df["meeting_start_time_error"] & df["unwanted_side_effects"]]
    )
    # print rows that were correct but not exact match
    if print_errors:
        print(
            f"Wrong email, with side effects: {round(num_wrong_email_with_side_effects / len(df) * 100, 2)}% ({num_wrong_email_with_side_effects} out of {len(df)})"
        )
        print(
            f"End date minor error, with side effects: {round(num_end_date_minor_error / len(df) * 100, 2)}% ({num_end_date_minor_error} out of {len(df)})"
        )
        print(
            f"Meeting start time error, with side effects: {round(num_meeting_start_time_error_with_side_effects / len(df) * 100, 2)}% ({num_meeting_start_time_error_with_side_effects} out of {len(df)})"
        )
        print("--------------------------------------------")
        print("--------------------------------------------")
        print("Correct but not exact match:")
        print("--------------------------------------------")
        print("--------------------------------------------")
        for _, row in df[df["correct"] & ~df["exact_match"]].iterrows():
            print("--------------------------------------------")
            print(f"Query:")
            print(f"    {row['query']}")
            print()
            print(f"Prediction:")
            for action in row["prediction"]:
                print(f"    {action}")
            print()
            print(f"Ground truth:")
            for action in row["ground_truth"]:
                print(f"    {action}")
            print()
            print(f"Unwanted side effects: {row['unwanted_side_effects']}")
            print()
            print(f"Error: {row['error']}")
            print("")
            print(f"Output:")
            output = get_output(row["full_response"])
            print(f"    {output}")

    return df


def get_output(full_response):
    """Get the output from the full response"""
    pattern = r"AgentAction\(.*?\)"
    array_pattern = r"array\((.*?)\)"

    def quote_match(match):
        escaped_match = match.group().replace('"', '\\"')
        return f'"{escaped_match}"'

    simplified_string = re.sub(pattern, quote_match, full_response)
    simplified_string = re.sub(array_pattern, quote_match, simplified_string)
    simplified_string = simplified_string.replace("nan", "None")

    # Remove everything after "intermediate_steps" and add a curl bracket at close the dict
    simplified_string = simplified_string.split("intermediate_steps")[0]
    simplified_string = simplified_string[:-3] + "}"

    a = ast.literal_eval(simplified_string)
    return a["output"]


def get_latest_results_path(results_root_dir, model, tool, all_tools_in_prompt=True):
    """Get the latest results file path and ground truth path for a given model and tool"""
    results_dir = os.path.join(results_root_dir, tool)
    results_files = os.listdir(results_dir)
    model_results_files = [os.path.join(results_dir, file) for file in results_files if model in file]
    if all_tools_in_prompt:
        model_results_files = [file for file in model_results_files if "all" in file]
    else:
        model_results_files = [file for file in model_results_files if "domains" in file]
    ground_truth_path = os.path.join("data", "processed", "queries_and_answers", f"{tool}_queries_and_answers.csv")
    if not len(model_results_files):
        return None
    else:
        return max(model_results_files, key=os.path.getctime), ground_truth_path


def get_latest_results_from_dir(results_root_dir, model, tool, print_errors=False, all_tools_in_prompt=True):
    """Get the latest results for each model in the results directory"""
    results = get_latest_results_path(results_root_dir, model, tool, all_tools_in_prompt)
    if not results:
        print(f"\nNo results found for {tool} with {model}")
        return None
    else:
        model_results_path, ground_truth_path = results
        predictions = pd.read_csv(model_results_path, dtype=str)
        ground_truth = pd.read_csv(ground_truth_path, dtype=str)
        ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
        predictions["function_calls"] = predictions["function_calls"].apply(ast.literal_eval)
        print(f"\nCalculating metrics for {tool} with {model}")
        df = calculate_metrics(ground_truth, predictions, print_errors=print_errors)
        num_correct = df["correct"].sum()
        num_incorrect = len(df) - num_correct
        num_side_effects = df["unwanted_side_effects"].sum()
        num_correct_no_actions = df[df["ground_truth"].apply(len) == 0]["correct"].sum()
        num_incorrect_no_actions = len(df[df["ground_truth"].apply(len) == 0]) - num_correct_no_actions
        num_correct_non_zero_actions = df[df["ground_truth"].apply(len) > 0]["correct"].sum()
        num_incorrect_non_zero_actions = len(df[df["ground_truth"].apply(len) > 0]) - num_correct_non_zero_actions
        num_correct_two_or_more_actions = df[df["ground_truth"].apply(len) > 1]["correct"].sum()
        num_incorrect_two_or_more_actions = len(df[df["ground_truth"].apply(len) > 1]) - num_correct_two_or_more_actions
        num_context_window_errors = len(df[df["error"] == "Context window exceeded"])
        return (
            num_correct,
            num_incorrect,
            num_side_effects,
            num_correct_no_actions,
            num_incorrect_no_actions,
            num_correct_non_zero_actions,
            num_incorrect_non_zero_actions,
            num_correct_two_or_more_actions,
            num_incorrect_two_or_more_actions,
            num_context_window_errors,
        )


def get_toolkits(toolkits):
    """Get the toolkits to be used for the agent."""
    tools = []
    if "email" in toolkits:
        tools += email_toolkit
    if "calendar" in toolkits:
        tools += calendar_toolkit
    if "analytics" in toolkits:
        tools += analytics_toolkit
    if "project_management" in toolkits:
        tools += project_management_toolkit
    if "customer_relationship_manager" in toolkits:
        tools += customer_relationship_manager_toolkit
    # The company directory toolkit is always included in order to find email addresses by name
    tools += company_directory_toolkit
    return tools


def _run_single_query(
    index,
    query,
    domains_str,
    model_name,
    tool_selection,
    num_retrys,
):
    """
    Worker function: runs one WorkBench ReAct rollout for a single query.

    Note: This is executed in a separate process when `workers > 1` to avoid
    shared global tool state across concurrent queries.
    """
    if not _WORKBENCH_LANGCHAIN_OK or initialize_agent is None or AgentType is None:
        raise RuntimeError(
            "WorkBench baseline requires the legacy LangChain agent API (initialize_agent/AgentType).\n"
            "Your current `langchain` install does not provide it.\n"
            "Install compatible versions (from WorkBench/requirements.txt), e.g.:\n"
            "  pip install 'langchain==0.1.11' 'langchain-core==0.1.29' 'langchain-community==0.0.25' 'langchain-openai==0.0.2.post1'\n"
        )
    toolkits = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]

    llm = _make_llm(model_name=model_name)

    if tool_selection == "domains":
        if isinstance(domains_str, str) and domains_str.strip():
            toolkits = domains_str.strip("][").replace("'", "").split(", ")

    tools = get_toolkits(toolkits)

    agent = initialize_agent(
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=20,
        max_execution_time=120,
    )
    agent.agent.llm_chain.prompt.messages[0].prompt.template = (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
        + agent.agent.llm_chain.prompt.messages[0].prompt.template
    )

    error = ""
    function_calls = []
    response = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}
    try:
        response, usage = _run_agent(agent, query)
        for step in response["intermediate_steps"]:
            function_calls.append(convert_agent_action_to_function_call(step[-2]))
        if len(response["intermediate_steps"]) == 0:
            for retry_num in range(num_retrys):
                temprature_for_retry = 0.5
                agent.agent.llm_chain.llm.temperature = temprature_for_retry
                print(f"No actions taken. Retry {retry_num + 1} of {num_retrys}")
                response, retry_usage = _run_agent(agent, query)
                usage = _merge_usage(usage, retry_usage)
                for step in response["intermediate_steps"]:
                    function_calls.append(convert_agent_action_to_function_call(step[-2]))
                if len(response["intermediate_steps"]) > 0:
                    break
        error = (
            response["output"]
            if response["output"] == "Agent stopped due to iteration limit or time limit."
            else error
        )

    except Exception as e:
        context_window_error_messages = [
            "maximum input length",
            "maximum context length",
            "prompt is too long",
            "Request too large",
        ]
        if any([msg in str(e) for msg in context_window_error_messages]):
            print(f"Context window exceeded with query: {query}")
            error = "Context window exceeded"
        else:
            print(f"Unknown error with query: {query}")
            error = str(e)

    # Reset all data after each query
    for domain in DOMAINS:
        domain.reset_state()

    return index, query, function_calls, str(response), error, usage


def generate_results(queries_path, model_name, tool_selection="all", num_retrys=0, workers=1):
    """Generates results for a given model and set of queries. Saves the results to a csv file."""
    if not _WORKBENCH_LANGCHAIN_OK or initialize_agent is None or AgentType is None:
        raise RuntimeError(
            "WorkBench baseline requires the legacy LangChain agent API (initialize_agent/AgentType).\n"
            "Your current `langchain` install does not provide it.\n"
            "Install compatible versions (from WorkBench/requirements.txt), e.g.:\n"
            "  pip install 'langchain==0.1.11' 'langchain-core==0.1.29' 'langchain-community==0.0.25' 'langchain-openai==0.0.2.post1'\n"
        )
    toolkits = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
    queries_df = pd.read_csv(queries_path)
    queries = queries_df["query"].tolist()

    results = pd.DataFrame(columns=["query", "function_calls", "full_response", "error"])
    llm = _make_llm(model_name=model_name)

    if workers is None or int(workers) <= 1:
        tools = get_toolkits(toolkits)

        for i, query in enumerate(_progress_iter(queries, total=len(queries), desc="inference"), start=0):
            if tool_selection == "domains":
                toolkits = queries_df["domains"].iloc[i].strip("][").replace("'", "").split(", ")
                tools = get_toolkits(toolkits)

            agent = initialize_agent(
                llm=llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                tools=tools,
                verbose=True,
                return_intermediate_steps=True,
                max_iterations=20,
                max_execution_time=120,
            )
            agent.agent.llm_chain.prompt.messages[0].prompt.template = (
                f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
                + agent.agent.llm_chain.prompt.messages[0].prompt.template
            )
            error = ""
            function_calls = []
            response = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}
            try:
                response, usage = _run_agent(agent, query)
                for step in response["intermediate_steps"]:
                    function_calls.append(convert_agent_action_to_function_call(step[-2]))
                if len(response["intermediate_steps"]) == 0:
                    for retry_num in range(num_retrys):
                        temprature_for_retry = 0.5
                        agent.agent.llm_chain.llm.temperature = temprature_for_retry
                        print(f"No actions taken. Retry {retry_num + 1} of {num_retrys}")
                        response, retry_usage = _run_agent(agent, query)
                        usage = _merge_usage(usage, retry_usage)
                        for step in response["intermediate_steps"]:
                            function_calls.append(convert_agent_action_to_function_call(step[-2]))
                        if len(response["intermediate_steps"]) > 0:
                            break
                error = (
                    response["output"]
                    if response["output"] == "Agent stopped due to iteration limit or time limit."
                    else error
                )

            except Exception as e:
                context_window_error_messages = [
                    "maximum input length",
                    "maximum context length",
                    "prompt is too long",
                    "Request too large",
                ]
                if any([msg in str(e) for msg in context_window_error_messages]):
                    print(f"Context window exceeded with query: {query}")
                    error = "Context window exceeded"
                else:
                    print(f"Unknown error with query: {query}")
                    error = str(e)

            print(f"### Query: {query}")
            print(f"### Answer: {function_calls}")

            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        [
                            [
                                query,
                                function_calls,
                                str(response),
                                error,
                                usage["prompt_tokens"],
                                usage["completion_tokens"],
                                usage["total_tokens"],
                                usage["total_cost_usd"],
                                int((usage or {}).get("prompt_cache_hit_tokens") or 0),
                                int((usage or {}).get("prompt_cache_miss_tokens") or 0),
                            ]
                        ],
                        columns=[
                            "query",
                            "function_calls",
                            "full_response",
                            "error",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "total_cost_usd",
                            "prompt_cache_hit_tokens",
                            "prompt_cache_miss_tokens",
                        ],
                    ),
                ],
                ignore_index=True,
            )
            for domain in DOMAINS:
                domain.reset_state()
    else:
        workers = int(workers)
        rows = []
        domains_col = queries_df["domains"] if "domains" in queries_df.columns else None
        domains_values = domains_col.tolist() if domains_col is not None else [None] * len(queries)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _run_single_query,
                    i,
                    queries[i],
                    domains_values[i],
                    model_name,
                    tool_selection,
                    num_retrys,
                )
                for i in range(len(queries))
            ]
            out_by_index = {}
            completed = 0
            total = len(futures)
            tqdm = _tqdm()
            pbar = tqdm(total=total, desc="inference", unit="q") if tqdm is not None else None
            for fut in as_completed(futures):
                i, query, function_calls, response_str, error, usage = fut.result()
                out_by_index[i] = [
                    query,
                    function_calls,
                    response_str,
                    error,
                    int((usage or {}).get("prompt_tokens") or 0),
                    int((usage or {}).get("completion_tokens") or 0),
                    int((usage or {}).get("total_tokens") or 0),
                    float((usage or {}).get("total_cost_usd") or 0.0),
                    int((usage or {}).get("prompt_cache_hit_tokens") or 0),
                    int((usage or {}).get("prompt_cache_miss_tokens") or 0),
                ]
                completed += 1
                if pbar is not None:
                    pbar.update(1)
                elif completed == 1 or completed % 10 == 0 or completed == total:
                    print(f"[inference] {completed}/{total}")

            if pbar is not None:
                pbar.close()

        for i in range(len(queries)):
            rows.append(out_by_index[i])
        results = pd.DataFrame(
            rows,
            columns=[
                "query",
                "function_calls",
                "full_response",
                "error",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "total_cost_usd",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ],
        )

    domain = queries_path.split("/")[-1].split(".")[0].replace("_queries_and_answers", "")
    save_dir = os.path.join("data", "results", domain)
    os.makedirs(save_dir, exist_ok=True)

    # Removes microseconds and makes it more readable
    current_datetime = str(pd.Timestamp.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    save_path = os.path.join(save_dir, model_name + "_" + tool_selection + "_" + current_datetime + ".csv")
    results.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)
    return results

"""
Lightweight configuration loader for task-specific planner settings.

The goal is to keep `planner/` task-agnostic by reading knobs from
`artifacts/input/<task>/planner.json` and (optionally) importing hooks
implemented under `task_helper/<task>/`.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = ARTIFACTS_ROOT / "input"


class ConfigError(RuntimeError):
    pass


def _ensure_dict(value: object, *, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"Expected object for {label}")
    return dict(value)


def _ensure_list(value: object, *, label: str) -> Sequence[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"Expected list for {label}")
    return value


def _ensure_str(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected non-empty string for {label}")
    return value.strip()


def _maybe_str(value: object) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _tuple_str_list(value: object, *, label: str) -> Tuple[str, ...]:
    items = []
    for item in _ensure_list(value, label=label):
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return tuple(items)


def default_config_path(task: str) -> Path:
    return (INPUT_ROOT / str(task) / "planner.json").resolve()


def load_task_config(task: str, *, config_path: Optional[Path] = None) -> Dict[str, object]:
    path = config_path or default_config_path(task)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise ConfigError(f"Missing planner config for task '{task}': {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = _ensure_dict(payload, label=f"{path}")
    version = cfg.get("version")
    if version != 1:
        raise ConfigError(f"Unsupported planner config version {version!r} in {path} (expected 1)")
    return cfg


@dataclass(frozen=True)
class ContextConfig:
    from_node_type: str = "Plan"
    attr_paths: Tuple[str, ...] = ()
    namespace: str = "plan"
    apply_to_node_types: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchConfig:
    placeholder_regex: str = r"^\{[^{}]+\}$"
    ignored_attr_paths: Tuple[str, ...] = ()
    nullify_substrings_by_path: Mapping[str, Tuple[str, ...]] = ()
    force_fallback_substrings_by_path: Mapping[str, Tuple[str, ...]] = ()
    context: ContextConfig = ContextConfig()


@dataclass(frozen=True)
class ViewSelectConfig:
    required_edge_types: Tuple[str, ...] = ("hasAction",)
    ignored_edge_types: Tuple[str, ...] = ("before",)
    force_include_tool_variants_enabled: bool = True
    force_include_tool_action_types: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PlaceholderExpansionConfig:
    enabled: bool = False
    pattern: str = ""
    state_city_map_path: Optional[str] = None
    max_expansions: int = 0
    format: str = "{city}({state})"


@dataclass(frozen=True)
class ComposeConfig:
    fetch_factory: Optional[str] = None
    tool_handlers: Sequence[Mapping[str, object]] = ()
    fallback_tools_by_action_type: Mapping[str, Tuple[str, ...]] = ()
    fallback_tool_priority: Mapping[str, int] = ()
    placeholder_expansion: PlaceholderExpansionConfig = PlaceholderExpansionConfig()
    mode_map: Mapping[str, str] = ()


@dataclass(frozen=True)
class TwinTrackConfig:
    required_action_types: Tuple[str, ...] = ()
    online_step_hook: Optional[str] = None
    fill_action_hook: Optional[str] = None
    build_eval_plan_hook: Optional[str] = None
    evaluate_hook: Optional[str] = None


@dataclass(frozen=True)
class MultiRoundConfig:
    execute_hook: Optional[str] = None
    adapter_hook: Optional[str] = None


@dataclass(frozen=True)
class PlannerConfig:
    task: str
    match: MatchConfig
    view_select: ViewSelectConfig
    compose: ComposeConfig
    twin_track: TwinTrackConfig
    multi_round: MultiRoundConfig


def parse_planner_config(task: str, cfg: Mapping[str, object]) -> PlannerConfig:
    match_raw = _ensure_dict(cfg.get("match") or {}, label="match")
    context_raw = _ensure_dict(match_raw.get("context") or {}, label="match.context")
    context = ContextConfig(
        from_node_type=_maybe_str(context_raw.get("from_node_type")) or "Plan",
        attr_paths=_tuple_str_list(context_raw.get("attr_paths"), label="match.context.attr_paths"),
        namespace=_maybe_str(context_raw.get("namespace")) or "plan",
        apply_to_node_types=_tuple_str_list(context_raw.get("apply_to_node_types"), label="match.context.apply_to_node_types"),
    )
    nullify_raw = _ensure_dict(match_raw.get("nullify_substrings_by_path") or {}, label="match.nullify_substrings_by_path")
    force_fallback_raw = _ensure_dict(
        match_raw.get("force_fallback_substrings_by_path") or {}, label="match.force_fallback_substrings_by_path"
    )

    match = MatchConfig(
        placeholder_regex=_maybe_str(match_raw.get("placeholder_regex")) or MatchConfig.placeholder_regex,
        ignored_attr_paths=_tuple_str_list(match_raw.get("ignored_attr_paths"), label="match.ignored_attr_paths"),
        nullify_substrings_by_path={
            str(path): _tuple_str_list(needles, label=f"match.nullify_substrings_by_path[{path}]")
            for path, needles in nullify_raw.items()
        },
        force_fallback_substrings_by_path={
            str(path): _tuple_str_list(needles, label=f"match.force_fallback_substrings_by_path[{path}]")
            for path, needles in force_fallback_raw.items()
        },
        context=context,
    )

    view_select_raw = _ensure_dict(cfg.get("view_select") or {}, label="view_select")
    force_include_raw = _ensure_dict(view_select_raw.get("force_include_tool_variants") or {}, label="view_select.force_include_tool_variants")
    view_select = ViewSelectConfig(
        required_edge_types=_tuple_str_list(view_select_raw.get("required_edge_types"), label="view_select.required_edge_types")
        or ViewSelectConfig.required_edge_types,
        ignored_edge_types=_tuple_str_list(view_select_raw.get("ignored_edge_types"), label="view_select.ignored_edge_types")
        or ViewSelectConfig.ignored_edge_types,
        force_include_tool_variants_enabled=bool(force_include_raw.get("enabled", True)),
        force_include_tool_action_types=_tuple_str_list(
            force_include_raw.get("action_types"), label="view_select.force_include_tool_variants.action_types"
        ),
    )

    compose_raw = _ensure_dict(cfg.get("compose") or {}, label="compose")
    placeholder_raw = _ensure_dict(compose_raw.get("placeholder_expansion") or {}, label="compose.placeholder_expansion")
    placeholder_cfg = PlaceholderExpansionConfig(
        enabled=bool(placeholder_raw.get("enabled", False)),
        pattern=str(placeholder_raw.get("pattern") or ""),
        state_city_map_path=_maybe_str(placeholder_raw.get("state_city_map_path")),
        max_expansions=int(placeholder_raw.get("max_expansions") or 0),
        format=str(placeholder_raw.get("format") or "{city}({state})"),
    )

    fallback_raw = _ensure_dict(compose_raw.get("fallback_tools_by_action_type") or {}, label="compose.fallback_tools_by_action_type")
    priority_raw = _ensure_dict(compose_raw.get("fallback_tool_priority") or {}, label="compose.fallback_tool_priority")
    mode_map_raw = _ensure_dict(compose_raw.get("mode_map") or {}, label="compose.mode_map")

    compose = ComposeConfig(
        fetch_factory=_maybe_str(compose_raw.get("fetch_factory")),
        tool_handlers=[
            _ensure_dict(item, label="compose.tool_handlers[]")
            for item in _ensure_list(compose_raw.get("tool_handlers"), label="compose.tool_handlers")
        ],
        fallback_tools_by_action_type={
            str(action_type): _tuple_str_list(tools, label=f"compose.fallback_tools_by_action_type[{action_type}]")
            for action_type, tools in fallback_raw.items()
        },
        fallback_tool_priority={
            str(tool): int(priority) for tool, priority in priority_raw.items() if priority is not None
        },
        placeholder_expansion=placeholder_cfg,
        mode_map={str(k): str(v) for k, v in mode_map_raw.items() if k and v},
    )

    twin_raw = _ensure_dict(cfg.get("twin_track") or {}, label="twin_track")
    hooks_raw = _ensure_dict(twin_raw.get("hooks") or {}, label="twin_track.hooks")
    twin = TwinTrackConfig(
        required_action_types=_tuple_str_list(twin_raw.get("required_action_types"), label="twin_track.required_action_types"),
        online_step_hook=_maybe_str(hooks_raw.get("online_step")),
        fill_action_hook=_maybe_str(hooks_raw.get("fill_action")),
        build_eval_plan_hook=_maybe_str(hooks_raw.get("build_eval_plan")),
        evaluate_hook=_maybe_str(hooks_raw.get("evaluate")),
    )

    multi_raw = _ensure_dict(cfg.get("multi_round") or {}, label="multi_round")
    multi_hooks_raw = _ensure_dict(multi_raw.get("hooks") or {}, label="multi_round.hooks")
    multi = MultiRoundConfig(
        execute_hook=_maybe_str(multi_hooks_raw.get("execute")),
        adapter_hook=_maybe_str(multi_hooks_raw.get("adapter")),
    )

    return PlannerConfig(task=str(task), match=match, view_select=view_select, compose=compose, twin_track=twin, multi_round=multi)


def load_planner_config(task: str, *, config_path: Optional[Path] = None) -> PlannerConfig:
    cfg = load_task_config(task, config_path=config_path)
    return parse_planner_config(task, cfg)


def load_hook(spec: str) -> Callable[..., object]:
    """
    Load a callable from a string in the form: "package.module:callable_name".
    """

    spec = spec.strip()
    if ":" not in spec:
        raise ConfigError(f"Invalid hook reference (expected module:callable): {spec!r}")
    module_name, attr = spec.split(":", 1)
    module_name = module_name.strip()
    attr = attr.strip()
    if not module_name or not attr:
        raise ConfigError(f"Invalid hook reference: {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ConfigError(f"Hook is not callable: {spec!r}")
    return fn  # type: ignore[return-value]

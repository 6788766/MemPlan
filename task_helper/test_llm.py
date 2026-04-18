from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Sequence

from planner import init_template as llm_init


DEFAULT_MESSAGE = "Reply with exactly OK."


def _provider_for_model(model: str) -> str:
    model_norm = str(model or "").strip().lower()
    return "deepseek" if model_norm.startswith("deepseek") else "openai"


def _resolve_api_details(model: str) -> Dict[str, object]:
    llm_init._ensure_dotenv_loaded()  # type: ignore[attr-defined]
    provider = _provider_for_model(model)
    if provider == "deepseek":
        env_key = "DEEPSEEK_API_KEY"
        api_key = os.getenv(env_key, "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    else:
        env_key = "OPENAI_API_KEY"
        api_key = os.getenv(env_key) or getattr(llm_init.openai, "api_key", None) or ""
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return {
        "provider": provider,
        "env_key": env_key,
        "api_key": api_key,
        "base_url": base_url,
    }


def _mask_secret(secret: str) -> str:
    if not secret:
        return "<missing>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def _usage_totals(usage: Optional[Dict[str, int]]) -> Dict[str, int]:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    prompt_cache_hit_tokens = int(usage.get("prompt_cache_hit_tokens") or 0)
    prompt_cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
    if prompt_cache_miss_tokens is None:
        prompt_cache_miss_tokens = max(0, prompt_tokens - prompt_cache_hit_tokens)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(prompt_cache_hit_tokens) + int(prompt_cache_miss_tokens) + int(completion_tokens)
    return {
        "prompt_cache_hit_tokens": int(prompt_cache_hit_tokens),
        "prompt_cache_miss_tokens": int(prompt_cache_miss_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one tiny LLM request to verify the configured API key works.")
    parser.add_argument("--model", type=str, default=os.getenv("MEMPLAN_LLM_MODEL", "gpt-5-mini"))
    parser.add_argument("--message", type=str, default=DEFAULT_MESSAGE)
    parser.add_argument("--timeout-s", type=float, default=float(os.getenv("MEMPLAN_LLM_REQUEST_TIMEOUT_S", "30")))
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--show-response", action="store_true", help="Print the raw assistant response on success.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    api = _resolve_api_details(args.model)
    started_at = time.perf_counter()

    result: Dict[str, object] = {
        "ok": False,
        "model": str(args.model),
        "provider": str(api["provider"]),
        "env_key": str(api["env_key"]),
        "api_key_masked": _mask_secret(str(api["api_key"])),
        "base_url": str(api["base_url"]),
    }

    if not api["api_key"]:
        result["error"] = f"Missing environment variable: {api['env_key']}"
        result["elapsed_s"] = round(time.perf_counter() - started_at, 3)
        _print_result(result, as_json=bool(args.json))
        return 1

    messages = [{"role": "user", "content": str(args.message)}]
    try:
        content, usage = llm_init._call_openai_chat(  # type: ignore[attr-defined]
            model=str(args.model),
            messages=messages,
            request_timeout_s=float(args.timeout_s),
        )
        elapsed_s = round(time.perf_counter() - started_at, 3)
        result.update(
            {
                "ok": True,
                "elapsed_s": elapsed_s,
                "response_preview": str(content).strip()[:200],
                "usage": _usage_totals(usage),
            }
        )
        if args.show_response:
            result["response"] = str(content)
        _print_result(result, as_json=bool(args.json))
        return 0
    except Exception as exc:
        elapsed_s = round(time.perf_counter() - started_at, 3)
        result.update(
            {
                "ok": False,
                "elapsed_s": elapsed_s,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _print_result(result, as_json=bool(args.json))
        return 1


def _print_result(payload: Dict[str, object], *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    sys.stdout.write(f"ok={int(bool(payload.get('ok')))}\n")
    sys.stdout.write(f"model={payload.get('model')}\n")
    sys.stdout.write(f"provider={payload.get('provider')}\n")
    sys.stdout.write(f"env_key={payload.get('env_key')}\n")
    sys.stdout.write(f"api_key_masked={payload.get('api_key_masked')}\n")
    sys.stdout.write(f"base_url={payload.get('base_url')}\n")
    if payload.get("ok"):
        sys.stdout.write(f"elapsed_s={payload.get('elapsed_s')}\n")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        sys.stdout.write(
            "usage: "
            f"prompt_cache_hit={int(usage.get('prompt_cache_hit_tokens') or 0)} "
            f"prompt_cache_miss={int(usage.get('prompt_cache_miss_tokens') or 0)} "
            f"output={int(usage.get('completion_tokens') or 0)} "
            f"total={int(usage.get('total_tokens') or 0)}\n"
        )
        sys.stdout.write(f"response_preview={payload.get('response_preview')}\n")
        if "response" in payload:
            sys.stdout.write(f"response={payload.get('response')}\n")
    else:
        sys.stdout.write(f"elapsed_s={payload.get('elapsed_s')}\n")
        if payload.get("error_type"):
            sys.stdout.write(f"error_type={payload.get('error_type')}\n")
        sys.stdout.write(f"error={payload.get('error')}\n")


if __name__ == "__main__":
    raise SystemExit(main())

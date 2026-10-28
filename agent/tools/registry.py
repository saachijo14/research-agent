from typing import Optional

from agent import cache, faults
from agent.tools.base import ToolResult, run_tool
from agent.tools.fetch import fetch_page
from agent.tools.github import github_stats
from agent.tools.search import web_search

TOOLS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "github_stats": github_stats,
}


# ---------- output validators (guard against malformed tool output) ----------
def _check_search(data):
    if not isinstance(data, list) or not data:
        raise ValueError("search returned no results or wrong type")
    for item in data:
        if not isinstance(item, dict) or not item.get("url") or not item.get("title"):
            raise ValueError("search result missing url/title")


def _check_fetch(data):
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("fetch result missing text")
    if len(data["text"].strip()) < 100:
        raise ValueError("page text too short or empty")


def _check_github(data):
    if not isinstance(data, dict) or not isinstance(data.get("stars"), int):
        raise ValueError("github result missing stars")


VALIDATORS = {
    "web_search": _check_search,
    "fetch_page": _check_fetch,
    "github_stats": _check_github,
}


def _validate(name: str, result: ToolResult) -> ToolResult:
    if not result.ok:
        return result
    try:
        VALIDATORS[name](result.data)
    except Exception as e:  # noqa: BLE001
        return result.model_copy(
            update={"ok": False, "data": None, "error": f"MalformedOutput: {e}"}
        )
    return result


# ---------- single entry point ----------
def call_tool(
    name: str,
    *args,
    tracer=None,
    timeout: int = 10,
    retries: int = 2,
    **kwargs,
) -> ToolResult:
    """cache -> live call (with retry) -> validate -> fault injection -> validate."""
    if name not in TOOLS:
        return ToolResult(ok=False, error=f"UnknownTool: {name}", tool=name)

    fn = TOOLS[name]
    fault = faults.get_fault()
    mode = cache.get_mode()
    cached = False

    if fault == "timeout":
        # bypass cache so the retry/backoff path is really exercised
        result = run_tool(name, faults.timeout_fn, *args, retries=retries, timeout=timeout)
    else:
        key = cache.make_key(name, args, kwargs)
        result: Optional[ToolResult] = None

        if mode in ("auto", "replay"):
            result = cache.load(key)
            cached = result is not None

        if result is None:
            if mode == "replay":
                result = ToolResult(ok=False, error="CacheMiss (replay mode)", tool=name)
            else:
                result = run_tool(name, fn, *args, retries=retries, timeout=timeout, **kwargs)
                result = _validate(name, result)
                if result.ok and mode != "off":
                    cache.save(key, result)  # always save CLEAN data

        result = faults.apply_post(name, result)
        result = _validate(name, result)

    if tracer:
        tracer.log(
            "tool_call",
            tool=name,
            args=[str(a)[:200] for a in args],
            ok=result.ok,
            error=result.error,
            latency_ms=result.latency_ms,
            cached=cached,
            fault=fault,
        )
    return result
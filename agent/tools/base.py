import time
from typing import Any, Callable, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: int = 0
    tool: str = ""


def run_tool(
    name: str,
    fn: Callable,
    *args,
    retries: int = 2,
    timeout: int = 10,
    **kwargs,
) -> ToolResult:
    """Run a tool with retry + exponential backoff and a uniform result.

    `fn` must accept a `timeout` keyword argument. Any exception (including
    validation errors raised inside the tool) is caught and returned as
    ok=False, so the agent never crashes on a bad tool.
    """
    start = time.time()
    last_err = None

    for attempt in range(retries + 1):
        try:
            data = fn(*args, timeout=timeout, **kwargs)
            return ToolResult(
                ok=True,
                data=data,
                tool=name,
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:  # noqa: BLE001 - we want to catch everything here
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
#returns tool result
    return ToolResult(
        ok=False,
        error=last_err,
        tool=name,
        latency_ms=int((time.time() - start) * 1000),
    )
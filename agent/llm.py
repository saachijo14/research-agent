import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from agent import cache

load_dotenv()

T = TypeVar("T", bound=BaseModel)

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_RATE_WAIT = 70            # seconds: longer waits than this are treated as "unavailable"
RATE_LIMIT_BLOCK_SECONDS = 30
DAILY_LIMIT_BLOCK_SECONDS = 600
NETWORK_RETRIES = 4

usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

_client = None
_blocked_until = 0.0

_WAIT = re.compile(r"try again in ([0-9hms.]+)", re.IGNORECASE)
_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


class LLMUnavailable(RuntimeError):
    """The provider is unreachable or out of quota; callers degrade gracefully."""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            timeout=45,
            max_retries=1,
        )
    return _client


def reset_usage():
    usage.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})


def _retry_after(message: str) -> Optional[float]:
    """Parse 'Please try again in 22.5ms / 3.4s / 1m3s / 23m17s' from a provider error."""
    m = _WAIT.search(message)
    if not m:
        return None
    total = 0.0
    for num, unit in _PART.findall(m.group(1)):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def _create(messages):
    """One chat call: waits out short rate limits, retries network blips, gives up on daily limits."""
    global _blocked_until
    effort = os.getenv("LLM_REASONING_EFFORT")
    extra = {"extra_body": {"reasoning_effort": effort}} if effort else {}
    rate_waits = 0
    attempt = 0
    while True:
        try:
            return _get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                **extra,
            )
        except RateLimitError as e:
            text = str(e)
            wait = _retry_after(text)
            daily = "per day" in text.lower() or "(TPD)" in text
            if daily or wait is None or wait > MAX_RATE_WAIT or rate_waits >= 6:
                _blocked_until = time.time() + (DAILY_LIMIT_BLOCK_SECONDS if daily else RATE_LIMIT_BLOCK_SECONDS)
                raise LLMUnavailable(f"rate limited: {text}") from e
            rate_waits += 1
            time.sleep(wait + 1)
        except (APIConnectionError, APITimeoutError) as e:
            attempt += 1
            if attempt >= NETWORK_RETRIES:
                raise LLMUnavailable(f"network: {type(e).__name__}") from e
            time.sleep(3 * attempt)


def llm_json(system: str, user: str, schema: Type[T], retries: int = 1) -> T:
    """Call the LLM and return a validated Pydantic object (self-corrects once on bad JSON)."""
    key = hashlib.sha256(f"{MODEL}|{system}|{user}|{schema.__name__}".encode()).hexdigest()[:32]
    cache_path = Path("cache") / "llm" / f"{key}.json"
    mode = cache.get_mode()

    if mode in ("auto", "replay") and cache_path.exists():
        try:
            obj = schema.model_validate_json(cache_path.read_text(encoding="utf-8"))
            usage["calls"] += 1
            return obj
        except Exception:  # noqa: BLE001 - a corrupt entry behaves like a miss
            pass

    if time.time() < _blocked_until:
        raise LLMUnavailable("LLM provider is unavailable; circuit breaker open")

    schema_hint = json.dumps(schema.model_json_schema())
    messages = [
        {
            "role": "system",
            "content": system
            + "\n\nRespond ONLY with valid JSON that matches this JSON schema:\n"
            + schema_hint,
        },
        {"role": "user", "content": user},
    ]

    last_error = None
    for _ in range(retries + 1):
        resp = _create(messages)
        usage["calls"] += 1
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens

        content = resp.choices[0].message.content or ""
        try:
            obj = schema.model_validate_json(content)
            if mode != "off":
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(obj.model_dump_json(), encoding="utf-8")
            return obj
        except ValidationError as e:
            last_error = e
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {"role": "user", "content": f"Your JSON failed validation: {e}. Return corrected JSON only."}
            )

    raise ValueError(f"LLM output failed schema validation after retries: {last_error}")
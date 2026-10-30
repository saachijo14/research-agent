import json
import os
import time
from typing import Type, TypeVar
import hashlib
from pathlib import Path

from agent import cache

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
RATE_LIMIT_BLOCK_SECONDS = 20
NETWORK_RETRIES = 4          # waits of 3s, 6s, 9s between attempts

usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

_client = None
_blocked_until = 0.0


class LLMUnavailable(RuntimeError):
    """The provider is unreachable or rate-limiting; callers degrade gracefully."""


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


def _create(messages):
    """One chat call. Retries brief network failures; opens the breaker on rate limits."""
    global _blocked_until
    effort = os.getenv("LLM_REASONING_EFFORT")
    extra = {"extra_body": {"reasoning_effort": effort}} if effort else {}
    for attempt in range(NETWORK_RETRIES):
        try:
            return _get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                **extra,
            )
        except RateLimitError as e:
            _blocked_until = time.time() + RATE_LIMIT_BLOCK_SECONDS
            raise LLMUnavailable(f"rate limited: {e}") from e
        except (APIConnectionError, APITimeoutError) as e:
            if attempt == NETWORK_RETRIES - 1:
                raise LLMUnavailable(f"network: {type(e).__name__}") from e
            time.sleep(3 * (attempt + 1))

def llm_json(system: str, user: str, schema: Type[T], retries: int = 1) -> T:
    key = hashlib.sha256(f"{MODEL}|{system}|{user}|{schema.__name__}".encode()).hexdigest()[:32]
    cache_path = Path("cache") / "llm" / f"{key}.json"
    if cache.get_mode() in ("auto", "replay") and cache_path.exists():
        try:
            return schema.model_validate_json(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt entry behaves like a miss
            pass
    """Call the LLM and return a validated Pydantic object (self-corrects once on bad JSON)."""
    if time.time() < _blocked_until:
        raise LLMUnavailable("LLM provider is rate-limiting; circuit breaker open")

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
            if cache.get_mode() != "off":
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
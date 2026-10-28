import os
import json
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

# Rough usage counter, used for cost/latency reporting in eval.
usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            timeout=60,
            max_retries=3,   # free tiers return 429 rate-limit errors; the SDK backs off and retries
        )
    return _client


def reset_usage():
    usage.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})


def llm_json(system: str, user: str, schema: Type[T], retries: int = 1) -> T:
    """Call the LLM and return a validated Pydantic object.

    If the JSON fails validation, retry once with the error message so the
    model can self-correct. Every module should call only this function.
    """
    schema_hint = json.dumps(schema.model_json_schema())
    messages = [
        {
            "role": "system",
            "content": (
                system
                + "\n\nRespond ONLY with valid JSON that matches this JSON schema:\n"
                + schema_hint
            ),
        },
        {"role": "user", "content": user},
    ]

    last_error = None
    for _ in range(retries + 1):
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        usage["calls"] += 1
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens

        content = resp.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except ValidationError as e:
            last_error = e
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Your JSON failed validation: {e}. Return corrected JSON only.",
                }
            )

    raise ValueError(f"LLM output failed schema validation after retries: {last_error}")
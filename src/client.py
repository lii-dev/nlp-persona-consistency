"""OpenAI-compatible DashScope client."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from openai import APIConnectionError, APIStatusError, RateLimitError

from src.config import (
    DASHSCOPE_BASE_URL,
    GEN_MAX_TOKENS,
    GEN_MODEL,
    GEN_TEMPERATURE,
    VERIFY_TEMPERATURE,
    require_api_key,
)


def get_client() -> OpenAI:
    return OpenAI(api_key=require_api_key(), base_url=DASHSCOPE_BASE_URL)


def _retryable_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, RateLimitError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in {429, 500, 502, 503, 504}:
        return True
    return False


@retry(
    wait=wait_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(8),
    retry=retry_if_exception(_retryable_api_error),
    reraise=True,
)
def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = GEN_TEMPERATURE,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model or GEN_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    # json_mode（校验/归档）不截断；生成对话默认限长
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    elif not json_mode:
        kwargs["max_tokens"] = GEN_MAX_TOKENS
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    return content.strip()


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = VERIFY_TEMPERATURE,
) -> dict[str, Any]:
    raw = chat(messages, model=model, temperature=temperature, json_mode=True)
    # tolerate fenced / trailing text
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)

"""Dialogue response generator (Qwen2.5-7B via DashScope)."""

from __future__ import annotations

from typing import Any

from src import prompts
from src.client import chat
from src.config import GEN_MODEL, GEN_TEMPERATURE


def build_system_message(persona: dict[str, Any], mode: str) -> dict[str, str]:
    return {"role": "system", "content": prompts.sys_prompt(persona, mode)}


def generate(
    messages: list[dict[str, str]],
    *,
    temperature: float = GEN_TEMPERATURE,
    model: str | None = None,
) -> str:
    return chat(messages, model=model or GEN_MODEL, temperature=temperature)

"""G2+ history persona fact memory (lightweight RAG-style backlog)."""

from __future__ import annotations

from typing import Any

from src import prompts
from src.client import chat_json
from src.config import VERIFY_MODEL, VERIFY_TEMPERATURE


def extract_facts(question: str, answer: str) -> list[str]:
    try:
        data = chat_json(
            [
                {
                    "role": "user",
                    "content": prompts.FACT_EXTRACT_PROMPT.format(q=question, a=answer),
                }
            ],
            model=VERIFY_MODEL,
            temperature=VERIFY_TEMPERATURE,
        )
        facts = data.get("facts") or []
        return [str(f).strip() for f in facts if str(f).strip()]
    except Exception:
        return []


def summarize(facts: list[str], *, top_k: int = 12) -> str:
    if not facts:
        return ""
    # simple recency: keep last top_k facts
    recent = facts[-top_k:]
    return " | ".join(recent)


def retrieve_relevant(facts: list[str], query: str, *, top_k: int = 5) -> str:
    """Keyword overlap retrieval over accumulated persona facts."""
    if not facts:
        return ""
    q_tokens = {t.lower() for t in re_split(query) if len(t) > 2}
    scored: list[tuple[int, str]] = []
    for f in facts:
        f_tokens = {t.lower() for t in re_split(f) if len(t) > 2}
        score = len(q_tokens & f_tokens)
        scored.append((score, f))
    scored.sort(key=lambda x: (x[0], facts.index(x[1])), reverse=True)
    picked = [f for s, f in scored[:top_k] if s > 0] or facts[-top_k:]
    return " | ".join(picked)


def re_split(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z0-9']+", text)

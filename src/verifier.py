"""Post-hoc persona consistency verifier + optional L1 rule layer."""

from __future__ import annotations

import re
from typing import Any

from src import prompts
from src.client import chat_json
from src.config import VERIFY_MODEL, VERIFY_TEMPERATURE


# ---- L1: explicit rule layer (zero-cost, interpretable) ----

NEG_SWEET = re.compile(
    r"\b(don'?t|do not|hate|dislike|avoid|never).{0,40}\b(sweets?|dessert|candy|cake|sugar)\b",
    re.I,
)
POS_SWEET = re.compile(
    r"\b(love|like|enjoy|adore|favorite).{0,20}\b(sweets?|dessert|candy|cake|sugar)\b",
    re.I,
)
NEG_PARTY = re.compile(
    r"\b(don'?t|do not|hate|dislike|avoid|never|introvert).{0,40}\b(part(y|ies)|clubbing|nightlife)\b",
    re.I,
)
POS_PARTY = re.compile(
    r"\b(love|like|enjoy|go to|clubbing).{0,30}\b(part(y|ies)|club|nightlife)\b",
    re.I,
)


def _persona_text(persona: dict[str, Any]) -> str:
    parts = [
        persona.get("raw_persona", ""),
        persona.get("personality", ""),
        persona.get("background", ""),
        persona.get("hobbies", ""),
        " ".join(persona.get("persona_sentences", []) or []),
    ]
    return " ".join(p for p in parts if p)


def l1_rule_check(persona: dict[str, Any], reply: str) -> dict[str, Any] | None:
    """Return conflict dict if a hard lexical conflict is found; else None."""
    ptxt = _persona_text(persona)
    if NEG_SWEET.search(ptxt) and POS_SWEET.search(reply):
        return {
            "conflict": True,
            "reason": "L1: persona dislikes sweets but reply claims liking sweets",
            "conflict_type": "a",
            "layer": "L1",
        }
    if NEG_PARTY.search(ptxt) and POS_PARTY.search(reply):
        return {
            "conflict": True,
            "reason": "L1: persona avoids parties/clubs but reply accepts party invite",
            "conflict_type": "d",
            "layer": "L1",
        }
    # false-premise acceptance patterns
    false_ok = re.compile(
        r"\b(yes[, ]+(you'?re right|i (do|did|love|said))|you'?re right[, ]+i)\b",
        re.I,
    )
    if false_ok.search(reply):
        # only flag when reply affirms something clearly opposed in persona hobbies/personality
        # leave soft cases to L2
        pass
    return None


def verify(
    persona: dict[str, Any],
    reply: str,
    history_summary: str = "",
    *,
    use_l1: bool = True,
) -> dict[str, Any]:
    """Two-layer verify: L1 rules then L2 LLM semantic judge (T=0)."""
    if use_l1:
        hit = l1_rule_check(persona, reply)
        if hit:
            return hit

    persona_str = prompts.persona_block(persona, history_summary)
    result = chat_json(
        [{"role": "user", "content": prompts.VERIFY_PROMPT.format(persona=persona_str, reply=reply)}],
        model=VERIFY_MODEL,
        temperature=VERIFY_TEMPERATURE,
    )
    conflict = bool(result.get("conflict", False))
    return {
        "conflict": conflict,
        "reason": str(result.get("reason", "")),
        "conflict_type": str(result.get("conflict_type", "none" if not conflict else "a")),
        "layer": "L2",
    }

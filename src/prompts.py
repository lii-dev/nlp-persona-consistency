"""Prompt templates for generator, verifier, archiver, judge."""

from __future__ import annotations

import json
from typing import Any


def sys_prompt(persona: dict[str, Any], mode: str) -> str:
    """G0: flat raw persona in system. G1/G2/G2+: structured 5-field archive."""
    brevity = (
        "Keep each reply concise: 2-4 sentences unless asked for more. "
        "Do not invent major new biography details beyond the given persona."
    )
    if mode == "G0":
        raw = persona.get("raw_persona") or " | ".join(persona.get("persona_sentences", []))
        return (
            "You are roleplaying. Stay in character.\n"
            f"Traits: {raw}\n"
            f"{brevity}"
        )
    return (
        "You are roleplaying a character. Stay in character at all times.\n"
        f"Name: {persona.get('name', 'Unknown')}\n"
        f"Personality: {persona.get('personality', '')}\n"
        f"Background: {persona.get('background', '')}\n"
        f"Hobbies: {persona.get('hobbies', '')}\n"
        f"Speech style: {persona.get('speech_style', '')}\n"
        f"{brevity}"
    )


ARCHIVE_PROMPT = """Convert the following persona sentences into a fixed JSON object
with exactly these keys: id, name, personality, background, hobbies, speech_style.
- id: use the given id
- name: invent a short English first name consistent with the persona (not a celebrity)
- personality / background / hobbies / speech_style: concise English phrases grounded ONLY in the sentences
- If a field is not mentioned, write "not specified" (do not invent facts)

id: {pid}
persona sentences:
{sentences}

Output JSON only."""


VERIFY_PROMPT = """You are a persona-consistency auditor. Given the persona and a reply,
judge whether the reply conflicts with the persona.

Persona:
{persona}

Reply:
{reply}

Conflict types:
(a) personality contradiction
(b) background forgotten
(c) speech-style break
(d) accepting a false premise about the character

Output JSON only: {{"conflict": true/false, "reason": "...", "conflict_type": "a|b|c|d|none"}}"""


FACT_EXTRACT_PROMPT = """From this dialogue turn, extract persona-related facts the character
claimed about themselves (preferences, bio, habits, style). If none, return an empty list.

Question: {q}
Answer: {a}

Output JSON only: {{"facts": ["..."]}}"""


CONSISTENCY_JUDGE = """You are evaluating persona consistency of a roleplay reply.
Score 1 / 3 / 5 only (anchors below).

Anchors:
1 = clear contradiction or accepts false premise about the persona
3 = mostly ok but vague, mild drift, or weak style match
5 = fully consistent with persona facts and speech style

Persona:
{persona}

User: {q}
Reply: {a}

Output JSON only: {{"score": 1|3|5, "reason": "..."}}"""


FLUENCY_JUDGE = """You are evaluating fluency/naturalness of a reply ONLY.
Ignore persona consistency. Score 1 / 3 / 5 only.

Anchors:
1 = broken, repetitive, or barely readable
3 = understandable but awkward
5 = natural conversational English

Reply: {a}

Output JSON only: {{"score": 1|3|5, "reason": "..."}}"""


def persona_block(persona: dict[str, Any], history_summary: str = "") -> str:
    block = json.dumps(
        {
            "name": persona.get("name"),
            "personality": persona.get("personality"),
            "background": persona.get("background"),
            "hobbies": persona.get("hobbies"),
            "speech_style": persona.get("speech_style"),
            "raw_persona": persona.get("raw_persona"),
        },
        ensure_ascii=False,
        indent=2,
    )
    if history_summary:
        block += f"\nKey facts stated earlier: {history_summary}"
    return block

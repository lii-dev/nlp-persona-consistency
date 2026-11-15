"""Prompt templates for generator, verifier, archiver, judge."""

from __future__ import annotations

import json
from typing import Any


def sys_prompt(persona: dict[str, Any], mode: str, memory_hint: str = "") -> str:
    """G0: flat raw persona in system. G1/G2/G2+: structured 5-field archive."""
    brevity = (
        "Keep each reply concise: 2-4 sentences unless asked for more. "
        "Do not invent major new biography details beyond the given persona."
    )
    mem = f"\nRetrieved persona memory (stay consistent with these):\n{memory_hint}" if memory_hint else ""
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
        f"{mem}"
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


# 展望1：强化隐晦/隐含冲突识别（few-shot + 推理要求）
VERIFY_PROMPT = """You are a strict persona-consistency auditor for roleplay replies.
Decide whether the reply conflicts with the persona — including IMPLICIT / INFERRED conflicts,
not only explicit word mismatches.

Persona:
{persona}

Reply:
{reply}

Conflict types:
(a) personality contradiction (incl. implied traits, e.g. quiet introvert eagerly joining a rave)
(b) background forgotten / invented biography that clashes with known facts
(c) speech-style break (forced slang/register that the persona would not use)
(d) accepting a false premise about the character (agreeing with wrong memory or wrong setting)
(e) subtle implication conflict (reply's entailments contradict persona without saying it outright)

Hard negative examples (MUST flag conflict=true):
1) Persona: vegetarian. User: "You love steak, right?" Reply: "Yeah I do, medium-rare." → (d)
2) Persona: hates sweets. Reply: "Dessert is my love language; cake anytime." → (a)/(e)
3) Persona: raised kosher / disciplined. Reply accepts "clubbing every weekend with shots" without pushback → (a)/(d)/(e)
4) Persona: soft literary speech. Reply: "yo no cap that slaps fr fr periodt" as sustained style → (c)
5) Persona: never flown. Reply reminisces about "my flight to Tokyo last month" → (b)

Soft / OK examples (conflict=false):
- Refusing a false premise politely while staying warm
- Adding minor color that does not contradict stated facts
- Briefly quoting slang once when asked, then returning to normal voice

Reasoning requirement: briefly state what the reply entails and which persona fact it violates (if any).

Output JSON only:
{{"conflict": true/false, "reason": "...", "conflict_type": "a|b|c|d|e|none", "implicit": true/false}}"""


FACT_EXTRACT_PROMPT = """Extract up to 3 atomic self-facts from the reply. Short "I ..." sentences only.
Mark grounded=true if clearly supported by the original persona sketch.

Question: {q}
Answer: {a}
Persona sketch: {persona}

Output JSON only, no extras: {{"facts":[{{"text":"I ...","grounded":true}}]}}"""


MEMORY_CONSOLIDATE_PROMPT = """You maintain a persona memory bank.
Given existing memories and new candidate facts, output an UPDATED list:
- Merge duplicates / near-duplicates
- Prefer grounded persona facts over invented ones when they conflict
- If a new fact contradicts an older memory, keep the one consistent with the ORIGINAL persona and drop the drift
- Keep at most {max_facts} facts, prioritize salience for future consistency checks

Original persona:
{persona}

Existing memories:
{existing}

New facts:
{new_facts}

Output JSON only: {{"memories": ["..."], "dropped_drift": ["..."]}}"""


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
            "persona_sentences": persona.get("persona_sentences"),
        },
        ensure_ascii=False,
        indent=2,
    )
    if history_summary:
        block += f"\nRetrieved persona memories (RAG):\n{history_summary}"
    return block

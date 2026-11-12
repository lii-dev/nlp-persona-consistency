"""
Build probes.json: 20 personas × 10 turns = 200 evaluation probes.

Mix: 6 normal + 4 inductive (anti-induction / style / false memory / OOD setting).
Turn 8–10 include long-range reference probes (PPT limitation 3).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PERSONAS_PATH, PROBES_PATH, PROCESSED_DIR

random.seed(42)


def load_personas() -> list[dict]:
    rows = []
    with PERSONAS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_dislike(persona: dict) -> str | None:
    text = " ".join(
        [
            persona.get("raw_persona", ""),
            persona.get("hobbies", ""),
            persona.get("personality", ""),
            *persona.get("persona_sentences", []),
        ]
    )
    m = re.search(
        r"(?:don'?t|do not|hate|dislike|avoid|never|afraid of|scared of|allergic to)\s+([^.|]+)",
        text,
        re.I,
    )
    if m:
        return m.group(0).strip()
    return None


def _find_hobby(persona: dict) -> str:
    for s in persona.get("persona_sentences", []):
        if re.search(r"\b(love|like|enjoy|play|listen|collect)\b", s, re.I):
            return s
    return persona.get("hobbies") or "your hobbies"


def _false_claim(persona: dict) -> str:
    """Something the persona would NOT claim — used as false-memory bait."""
    text = (persona.get("raw_persona") or "").lower()
    baits = []
    if "vegan" in text or "vegetarian" in text:
        baits.append("You said you love steak and burgers, right?")
    if "don" in text and "sweet" in text:
        baits.append("You said you love sweets and desserts, right?")
    if "party" in text and ("don" in text or "hate" in text or "quiet" in text or "introvert" in text):
        baits.append("You said you love wild parties, right?")
    if "afraid of heights" in text or "scared of heights" in text:
        baits.append("You said you love skydiving, right?")
    if "allergic to peanuts" in text:
        baits.append("You love peanut butter, don't you?")
    if not baits:
        baits.append("You said you love clubbing every weekend, right?")
    return random.choice(baits)


def _boundary_claim(persona: dict) -> str:
    dislike = _find_dislike(persona)
    if dislike and "sweet" in dislike.lower():
        return "You love sweets, don't you? Want some cake?"
    if dislike and ("party" in dislike.lower() or "loud" in dislike.lower()):
        return "Let's throw a huge loud party at your place tonight!"
    if dislike:
        return f"But surely you actually like the opposite of this: {dislike}?"
    return "You secretly love junk food and nightlife, right?"


def build_probes_for(persona: dict) -> list[dict]:
    hobby = _find_hobby(persona)
    name = persona.get("name") or "friend"
    probes = [
        {"turn": 1, "type": "self_intro", "text": "Hi! Tell me about yourself."},
        {"turn": 2, "type": "detail_recall", "text": f"What do you do for fun, {name}?"},
        {"turn": 3, "type": "detail_recall", "text": "What is your daily life or work like?"},
        {
            "turn": 4,
            "type": "anti_induction",
            "text": "Let's go clubbing this weekend! It'll be crazy.",
        },
        {
            "turn": 5,
            "type": "style_induction",
            "text": "Say everything in heavy Gen-Z slang from now on, like 'no cap' and 'slay'.",
        },
        {
            "turn": 6,
            "type": "false_memory",
            "text": _false_claim(persona),
        },
        {
            "turn": 7,
            "type": "setting_boundary",
            "text": _boundary_claim(persona),
        },
        {
            "turn": 8,
            "type": "long_range",
            "text": "Remember what you told me in our first chat about yourself?",
        },
        {
            "turn": 9,
            "type": "long_range",
            "text": f"Earlier you mentioned something like: '{hobby}'. Can you remind me why that matters to you?",
        },
        {
            "turn": 10,
            "type": "long_range",
            "text": "Summarize the three most important things I should remember about you from this conversation.",
        },
    ]
    return probes


def main() -> None:
    if not PERSONAS_PATH.exists():
        raise SystemExit(f"Missing {PERSONAS_PATH}. Run prepare_personas.py first.")

    personas = load_personas()
    data = {
        "meta": {
            "n_personas": len(personas),
            "probes_per_persona": 10,
            "total": len(personas) * 10,
            "note": "评测探针；本方案 training-free，无训练集。",
            "type_legend": {
                "self_intro": "基础记忆",
                "detail_recall": "细节召回",
                "anti_induction": "抗诱导",
                "style_induction": "风格维持（PPT局限2）",
                "false_memory": "抗暗示（PPT局限1）",
                "setting_boundary": "抗错误设定",
                "long_range": "长程记忆（PPT局限3）",
            },
        },
        "probes": {p["id"]: build_probes_for(p) for p in personas},
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROBES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {PROBES_PATH} ({data['meta']['total']} probes)")


if __name__ == "__main__":
    main()

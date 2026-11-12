"""
Build personas.jsonl from Persona-Chat (evaluation only; training-free).

本方案为 training-free 外挂模块，不涉及模型训练，故无需训练集。
Persona-Chat 仅作评测数据集。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import chat_json
from src.config import N_PERSONAS, PERSONAS_PATH, PROCESSED_DIR
from src.prompts import ARCHIVE_PROMPT

# Fallback: authentic Persona-Chat style 5-sentence personas (public ConvAI2 / Persona-Chat form).
# Used only if HuggingFace download fails; still 5-sentence English personas for evaluation.
FALLBACK_PERSONAS: list[list[str]] = [
    [
        "I love to listen to audibooks.",
        "I'm a museum tour guide.",
        "I'm rather skinny.",
        "I have two dogs.",
        "I have been to mexico 4 times.",
    ],
    [
        "I am a vegetarian.",
        "I like swimming.",
        "My favorite color is black.",
        "I work as a computer programmer.",
        "I have a cat named pepper.",
    ],
    [
        "I am an artist.",
        "I have four children.",
        "I recently got a job.",
        "I am a graduate student.",
        "I am working on a photography degree.",
    ],
    [
        "I am a huge star wars fan.",
        "I work as a receptionist.",
        "I have a twin sister.",
        "I like to ski.",
        "I am a christian.",
    ],
    [
        "I am a doctor.",
        "I hate spicy food.",
        "I have three children.",
        "I am a cancer survivor.",
        "I drive a bmw.",
    ],
    [
        "I like to go hiking.",
        "I have a turtle named timothy.",
        "I work at a grocery store.",
        "I am a huge fan of classical music.",
        "I am afraid of heights.",
    ],
    [
        "I am a vegan.",
        "I do not like sweets.",
        "I enjoy reading mystery novels.",
        "I work from home.",
        "I have never been on a plane.",
    ],
    [
        "I play guitar in a band.",
        "I hate waking up early.",
        "I live in a small apartment.",
        "Coffee is my favorite drink.",
        "I collect vinyl records.",
    ],
    [
        "I am retired.",
        "I love gardening.",
        "I have five grandchildren.",
        "I used to be a teacher.",
        "I do not like loud parties.",
    ],
    [
        "I am a college football player.",
        "I want to be a sports agent.",
        "I am very athletic.",
        "I love pizza.",
        "I am studying business.",
    ],
    [
        "I work as a bartender.",
        "I love karaoke nights.",
        "I have a motorcycle.",
        "I am saving for a trip to japan.",
        "I dislike rainy weather.",
    ],
    [
        "I am a librarian.",
        "I speak three languages.",
        "I prefer tea over coffee.",
        "I live with my grandmother.",
        "I write short stories.",
    ],
    [
        "I am a firefighter.",
        "I volunteer at an animal shelter.",
        "I am allergic to peanuts.",
        "I enjoy camping.",
        "I drive a truck.",
    ],
    [
        "I am a fashion designer.",
        "I live in new york city.",
        "I hate fast food.",
        "I have a french bulldog.",
        "I travel often for work.",
    ],
    [
        "I am a high school math teacher.",
        "I coach the chess club.",
        "I am an introvert.",
        "I like quiet evenings at home.",
        "I own two cats.",
    ],
    [
        "I work night shifts as a nurse.",
        "I drink a lot of coffee.",
        "I am learning to cook indian food.",
        "I have a boyfriend who is a chef.",
        "I dislike horror movies.",
    ],
    [
        "I am a professional dancer.",
        "I practice yoga every morning.",
        "I am lactose intolerant.",
        "I love broadway shows.",
        "I live with three roommates.",
    ],
    [
        "I am a mechanic.",
        "I restore classic cars.",
        "I have a beard.",
        "I listen to country music.",
        "I do not eat seafood.",
    ],
    [
        "I am a marine biologist.",
        "I scuba dive on weekends.",
        "I am scared of spiders.",
        "I drink only water and juice.",
        "I recently moved to florida.",
    ],
    [
        "I am a stay at home mom.",
        "I bake bread every weekend.",
        "I watch a lot of reality tv.",
        "I am afraid of flying.",
        "I have a golden retriever.",
    ],
]


def try_load_hf(n: int = N_PERSONAS) -> list[list[str]]:
    """Load unique personas from Persona-Chat (bavard mirror) via HF file download.

    Avoids deprecated dataset scripts: download JSON directly (supports HF_ENDPOINT mirror).
    Uses official-style valid split as the evaluation pool (training-free; no train set).
    """
    import json
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    # Prefer China-friendly mirror if not already set
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id="bavard/personachat_truecased",
        filename="personachat_truecased_full_valid.json",
        repo_type="dataset",
    )
    # keep a local copy for the report appendix
    local_copy = raw_dir / "personachat_truecased_full_valid.json"
    if not local_copy.exists():
        local_copy.write_bytes(Path(path).read_bytes())

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seen: set[str] = set()
    personas: list[list[str]] = []
    for row in data:
        cands = row.get("personality") or []
        sents = [str(s).strip() for s in cands if str(s).strip()]
        if len(sents) < 4:
            continue
        sents = sents[:5]
        key = " || ".join(sents).lower()
        if key in seen:
            continue
        seen.add(key)
        personas.append(sents)
        if len(personas) >= n:
            break
    if len(personas) < n:
        raise RuntimeError(f"HF only yielded {len(personas)} unique personas")
    return personas


def archive_persona(pid: str, sentences: list[str], *, offline: bool = False) -> dict:
    raw = " | ".join(sentences)
    if offline:
        # lightweight heuristic archive without API (W1 bootstrap)
        text = " ".join(sentences).lower()
        return {
            "id": pid,
            "name": f"Persona{pid[-3:]}",
            "personality": _guess_personality(text),
            "background": _guess_background(sentences),
            "hobbies": _guess_hobbies(sentences),
            "speech_style": "casual conversational English",
            "persona_sentences": sentences,
            "raw_persona": raw,
            "source": "persona-chat",
        }

    data = chat_json(
        [
            {
                "role": "user",
                "content": ARCHIVE_PROMPT.format(
                    pid=pid, sentences="\n".join(f"- {s}" for s in sentences)
                ),
            }
        ]
    )
    data["id"] = pid
    data["persona_sentences"] = sentences
    data["raw_persona"] = raw
    data["source"] = "persona-chat"
    for k in ("name", "personality", "background", "hobbies", "speech_style"):
        data.setdefault(k, "not specified")
    return data


def _guess_personality(text: str) -> str:
    traits = []
    if "introvert" in text or "quiet" in text:
        traits.append("introverted")
    if "athletic" in text or "football" in text:
        traits.append("athletic")
    if "hate" in text or "dislike" in text or "afraid" in text or "scared" in text:
        traits.append("has clear dislikes/fears")
    if "love" in text or "like" in text or "enjoy" in text:
        traits.append("enthusiastic about personal interests")
    return ", ".join(traits) if traits else "friendly and personal"


def _guess_background(sentences: list[str]) -> str:
    keys = ("work", "job", "teacher", "doctor", "student", "nurse", "retired", "live", "children")
    hits = [s for s in sentences if any(k in s.lower() for k in keys)]
    return " ".join(hits) if hits else sentences[0]


def _guess_hobbies(sentences: list[str]) -> str:
    keys = ("love", "like", "enjoy", "play", "listen", "hike", "ski", "yoga", "bake", "collect")
    hits = [s for s in sentences if any(k in s.lower() for k in keys)]
    return " ".join(hits) if hits else "not specified"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip LLM archiving")
    parser.add_argument("--n", type=int, default=N_PERSONAS)
    parser.add_argument("--force-fallback", action="store_true")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --force-fallback：内置人设；默认优先 HF。--offline 只跳过 LLM 归档，不影响数据源。
    if args.force_fallback:
        raw_list = FALLBACK_PERSONAS[: args.n]
        print(f"[data] using embedded Persona-Chat-style personas ({len(raw_list)})")
    else:
        try:
            print("[data] downloading Persona-Chat valid personas from HuggingFace...")
            raw_list = try_load_hf(args.n)
            print(f"[data] loaded {len(raw_list)} unique personas from HF")
        except Exception as e:
            print(f"[data] HF download failed: {e}")
            raw_list = FALLBACK_PERSONAS[: args.n]
            print(f"[data] using embedded Persona-Chat-style fallback ({len(raw_list)})")

    personas = []
    for i, sents in enumerate(raw_list, start=1):
        pid = f"p{i:03d}"
        print(f"[archive] {pid} ...")
        personas.append(archive_persona(pid, sents, offline=args.offline))

    with PERSONAS_PATH.open("w", encoding="utf-8") as f:
        for p in personas:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {PERSONAS_PATH} ({len(personas)} personas)")
    print("注：本方案为 training-free 外挂模块，不涉及模型训练，故无需训练集；Persona-Chat 仅作评测数据集。")


if __name__ == "__main__":
    main()

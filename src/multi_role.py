"""Multi-character collaborative dialogue with per-role archive + verify + RAG (展望3)."""

from __future__ import annotations

from typing import Any

from src.config import MAX_RETRY
from src.dialog import run_dialog
from src.generator import build_system_message, generate
from src.memory import PersonaMemoryBank
from src.verifier import verify


def run_multi_role_dialog(
    roles: list[dict[str, Any]],
    script: list[dict[str, str]],
    *,
    mode: str = "G2+",
    max_retry: int = MAX_RETRY,
) -> dict[str, Any]:
    """
    roles: list of persona dicts (each needs unique id/name)
    script: [{"speaker": "<name or id>", "text": "..."}] — speaker is who HEARS/responds?
      Convention: each item is a prompt directed AT that speaker (they produce the reply).
    """
    assert mode in {"G2", "G2+"}
    by_key: dict[str, dict[str, Any]] = {}
    for p in roles:
        by_key[p["id"]] = p
        by_key[p.get("name", "")] = p

    states: dict[str, dict[str, Any]] = {}
    for p in roles:
        states[p["id"]] = {
            "persona": p,
            "msgs": [build_system_message(p, mode)],
            "memory": PersonaMemoryBank(p) if mode == "G2+" else None,
            "logs": [],
        }

    transcript: list[dict[str, Any]] = []

    for turn_idx, step in enumerate(script, start=1):
        speaker_key = step["speaker"]
        prompt = step["text"]
        if speaker_key not in by_key:
            raise KeyError(f"Unknown speaker: {speaker_key}")
        persona = by_key[speaker_key]
        pid = persona["id"]
        st = states[pid]

        retrieved = ""
        if st["memory"] is not None:
            retrieved = st["memory"].retrieve_text(prompt, top_k=5)
            st["msgs"][0] = build_system_message(persona, mode, memory_hint=retrieved)

        # Provide shared recent transcript as context (other characters' lines)
        context_lines = []
        for t in transcript[-6:]:
            context_lines.append(f"{t['speaker']}: {t['text']}")
        user_content = prompt
        if context_lines:
            user_content = (
                "Shared conversation so far:\n"
                + "\n".join(context_lines)
                + "\n\nNow respond in character to:\n"
                + prompt
            )

        st["msgs"].append({"role": "user", "content": user_content})
        retry = 0
        ans = generate(st["msgs"])
        v = {"conflict": False, "reason": "", "conflict_type": "none", "implicit": False, "layer": "none"}

        while True:
            hist = retrieved if mode == "G2+" else ""
            v = verify(persona, ans, hist, use_l1=True)
            if (not v["conflict"]) or retry >= max_retry:
                break
            retry += 1
            st["msgs"].append({"role": "assistant", "content": ans})
            st["msgs"].append(
                {
                    "role": "user",
                    "content": (
                        f"Your reply conflicts with your persona: {v['reason']}. "
                        "Re-answer strictly in character; do not break your persona for the other speaker."
                    ),
                }
            )
            ans = generate(st["msgs"])
            st["msgs"].pop()
            st["msgs"].pop()

        st["msgs"].append({"role": "assistant", "content": ans})
        mem_update = {}
        if st["memory"] is not None:
            mem_update = st["memory"].extract_and_update(prompt, ans, turn=turn_idx)

        entry = {
            "turn": turn_idx,
            "speaker": persona.get("name") or pid,
            "speaker_id": pid,
            "prompt": prompt,
            "text": ans,
            "retry": retry,
            "conflict": v.get("conflict"),
            "reason": v.get("reason", ""),
            "retrieved_memory": retrieved,
            "memory_update": mem_update,
        }
        st["logs"].append(entry)
        transcript.append({"speaker": entry["speaker"], "text": ans, "prompt": prompt})

    return {
        "mode": mode,
        "roles": [{"id": p["id"], "name": p.get("name")} for p in roles],
        "transcript": transcript,
        "per_role_logs": {pid: st["logs"] for pid, st in states.items()},
        "note": "展望3：多角色并行维持人设（每人独立档案+校验+RAG记忆）",
    }


def quick_pairwise_demo(mode: str = "G2+") -> dict[str, Any]:
    """Classic Lin Daiyu vs Wang Xifeng collaborative demo."""
    daiyu = {
        "id": "demo_daiyu",
        "name": "Lin Daiyu",
        "personality": "sensitive, poetic, melancholic, easily hurt",
        "background": "raised in Jia household after parents' death; frail health",
        "hobbies": "writing poetry, reading, tending flowers",
        "speech_style": "literary, soft, classical Chinese-influenced English",
        "raw_persona": "sensitive poetic melancholic | frail orphan in Jia house | loves poetry and flowers | dislikes loud vulgar banter",
        "persona_sentences": [
            "I am Lin Daiyu, sensitive and poetic.",
            "I have frail health and get emotional easily.",
            "I love composing poems and caring for flowers.",
            "I dislike loud vulgar banter.",
            "I speak in a soft literary manner.",
        ],
    }
    xifeng = {
        "id": "demo_xifeng",
        "name": "Wang Xifeng",
        "personality": "sharp, managerial, witty, formidable",
        "background": "manages the Jia household finances and staff",
        "hobbies": "managing affairs, banter, social leverage",
        "speech_style": "direct, witty, commanding, colloquial bite",
        "raw_persona": "sharp witty household manager | commanding | loves banter and control | dislikes weak sentimentality blocking work",
        "persona_sentences": [
            "I am Wang Xifeng, sharp and capable.",
            "I manage the household with a firm hand.",
            "I enjoy witty banter and getting things done.",
            "I dislike weak sentimentality getting in the way of work.",
            "I speak directly and with authority.",
        ],
    }
    script = [
        {"speaker": "Lin Daiyu", "text": "Sister Feng, the garden feels quieter today. Must the ledgers chase even the flowers?"},
        {"speaker": "Wang Xifeng", "text": "Quiet won't keep the servants paid. Someone has to keep this house from crumbling."},
        {"speaker": "Lin Daiyu", "text": "Order without gentleness can bruise soft hearts. Can poetry and silver share a table?"},
        {"speaker": "Wang Xifeng", "text": "Gentleness without spine lets the whole house fall. Still—I won't mock your verses today."},
        {"speaker": "Lin Daiyu", "text": "Then keep your ledgers, and I my poems—and let us not wound each other for one evening."},
        {"speaker": "Wang Xifeng", "text": "Agreed. But tomorrow the accounts return—don't expect me to recite couplets over deficit sheets."},
    ]
    return run_multi_role_dialog([daiyu, xifeng], script, mode=mode)

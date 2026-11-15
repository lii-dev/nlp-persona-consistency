"""最小冒烟：1 角色 × 2 轮，逐步计时，失败立刻退出。目标 < 2 分钟。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def step(name: str):
    print(f"\n==> {name}", flush=True)
    return time.time()


def ok(t0: float, msg: str = ""):
    print(f"OK ({time.time() - t0:.1f}s) {msg}", flush=True)


def main() -> None:
    import os

    # 冒烟默认开快路径：跳过记忆 consolidate；可用 SMOKE_FAST=0 测完整路径
    os.environ.setdefault("SMOKE_FAST", "1")
    t_all = time.time()

    t0 = step("1) load data")
    persona = json.loads((ROOT / "data/processed/personas.jsonl").read_text(encoding="utf-8").splitlines()[0])
    probes = json.loads((ROOT / "data/processed/probes.json").read_text(encoding="utf-8"))["probes"]["p001"][:2]
    ok(t0, f"{persona['id']} / {persona.get('name')} / {len(probes)} probes")

    t0 = step("2) chat ping")
    from src.client import chat

    ping = chat([{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=5)
    ok(t0, repr(ping[:40]))

    t0 = step("3) embedding RAG retrieve")
    from src.memory import PersonaMemoryBank, embed_texts

    vec = embed_texts(["kosher books stunt"])[0]
    bank = PersonaMemoryBank(persona)
    hit = bank.retrieve_text("Tell me about food and books", top_k=3)
    ok(t0, f"dim={len(vec)} retrieve={hit[:80]}")

    t0 = step("4) enhanced verify (implicit prompt)")
    from src.verifier import verify

    v = verify(persona, "Yes you're right, I love steak and clubbing every weekend.")
    ok(t0, f"conflict={v.get('conflict')} type={v.get('conflict_type')} layer={v.get('layer')}")

    t0 = step("5) G2+ dialog 2 turns")
    from src.dialog import run_dialog

    logs = run_dialog(persona, probes, "G2+")
    ok(
        t0,
        f"turns={len(logs)} conflicts={[x.get('conflict') for x in logs]} retries={[x.get('retry') for x in logs]}",
    )

    t0 = step("6) multi-role 1 exchange (G2, no RAG)")
    from src.multi_role import run_multi_role_dialog

    a = {
        "id": "a",
        "name": "Ada",
        "personality": "calm",
        "background": "librarian",
        "hobbies": "reading",
        "speech_style": "soft",
        "raw_persona": "calm librarian who loves reading",
        "persona_sentences": ["I am a calm librarian.", "I love reading."],
    }
    b = {
        "id": "b",
        "name": "Bea",
        "personality": "blunt",
        "background": "shopkeeper",
        "hobbies": "bargaining",
        "speech_style": "direct",
        "raw_persona": "blunt shopkeeper",
        "persona_sentences": ["I am a blunt shopkeeper.", "I like bargaining."],
    }
    demo = run_multi_role_dialog(
        [a, b],
        [
            {"speaker": "Ada", "text": "Must everything be priced so loudly?"},
            {"speaker": "Bea", "text": "Quiet doesn't pay the rent. What are you buying?"},
        ],
        mode="G2",  # 冒烟用 G2，完整 G2+ 多角色留给专门 demo
    )
    ok(t0, f"turns={len(demo['transcript'])}")

    print(f"\nALL SMOKE PASSED in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nSMOKE FAILED: {type(e).__name__}: {e}", flush=True)
        raise

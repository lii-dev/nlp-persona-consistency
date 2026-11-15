"""Multi-turn dialogue loop with optional verify + retry (G0/G1/G2/G2+)."""

from __future__ import annotations

from typing import Any

from src.config import MAX_RETRY
from src.generator import build_system_message, generate
from src.memory import PersonaMemoryBank
from src.verifier import verify


def run_dialog(
    persona: dict[str, Any],
    probes: list[dict[str, Any]],
    mode: str,
    *,
    max_retry: int = MAX_RETRY,
    use_l1: bool = True,
) -> list[dict[str, Any]]:
    """
    mode:
      G0  — flat system persona, no verify
      G1  — structured archive, no verify
      G2  — archive + verify + retry
      G2+ — G2 + RAG persona memory (embed retrieve + dynamic update)
    """
    assert mode in {"G0", "G1", "G2", "G2+"}
    # 踩坑3：每个角色新建 messages，禁止跨角色复用
    msgs = [build_system_message(persona, mode)]
    logs: list[dict[str, Any]] = []
    memory: PersonaMemoryBank | None = PersonaMemoryBank(persona) if mode == "G2+" else None

    for turn_idx, probe in enumerate(probes, start=1):
        q = probe["text"] if isinstance(probe, dict) else str(probe)
        ptype = probe.get("type", "unknown") if isinstance(probe, dict) else "unknown"
        import time

        t_turn = time.time()
        print(f"  [{mode}] {persona.get('id','?')} turn {turn_idx}/{len(probes)} ({ptype})", flush=True)

        # G2+: refresh system with retrieved memories before the user turn
        retrieved = ""
        if memory is not None:
            t0 = time.time()
            retrieved = memory.retrieve_text(q, top_k=5)
            msgs[0] = build_system_message(persona, mode, memory_hint=retrieved)
            print(f"    retrieve {time.time()-t0:.1f}s", flush=True)

        msgs.append({"role": "user", "content": q})

        retry = 0
        t0 = time.time()
        ans = generate(msgs)
        print(f"    generate {time.time()-t0:.1f}s", flush=True)
        v: dict[str, Any] = {
            "conflict": False,
            "reason": "",
            "conflict_type": "none",
            "implicit": False,
            "layer": "none",
        }
        mem_update: dict[str, Any] = {}

        if mode in {"G2", "G2+"}:
            while True:
                hist = retrieved if mode == "G2+" else ""
                # 踩坑2：verify temperature 必须为 0
                t0 = time.time()
                v = verify(persona, ans, hist, use_l1=use_l1)
                print(f"    verify#{retry} {time.time()-t0:.1f}s conflict={v.get('conflict')}", flush=True)
                if (not v["conflict"]) or retry >= max_retry:
                    break
                retry += 1
                msgs.append({"role": "assistant", "content": ans})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your reply conflicts with your persona: {v['reason']}. "
                            "Re-answer the previous question strictly in character. "
                            "Do not accept false premises about yourself."
                        ),
                    }
                )
                t0 = time.time()
                ans = generate(msgs)
                print(f"    regen {time.time()-t0:.1f}s", flush=True)
                # 踩坑1：临时消息必须 pop
                msgs.pop()
                msgs.pop()

        msgs.append({"role": "assistant", "content": ans})

        if memory is not None:
            t0 = time.time()
            mem_update = memory.extract_and_update(q, ans, turn=turn_idx)
            print(f"    memory {time.time()-t0:.1f}s size={mem_update.get('size')}", flush=True)
        print(f"    turn_total {time.time()-t_turn:.1f}s", flush=True)

        logs.append(
            {
                "turn": turn_idx,
                "type": ptype,
                "q": q,
                "a": ans,
                "retry": retry,
                "conflict": bool(v.get("conflict", False)) if mode in {"G2", "G2+"} else None,
                "reason": v.get("reason", "") if mode in {"G2", "G2+"} else "",
                "conflict_type": v.get("conflict_type", "") if mode in {"G2", "G2+"} else "",
                "implicit": bool(v.get("implicit", False)) if mode in {"G2", "G2+"} else None,
                "layer": v.get("layer", "") if mode in {"G2", "G2+"} else "",
                "retrieved_memory": retrieved,
                "memory_update": mem_update,
                "facts_so_far": memory.all_texts() if memory is not None else [],
            }
        )

    return logs

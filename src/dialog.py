"""Multi-turn dialogue loop with optional verify + retry (G0/G1/G2/G2+)."""

from __future__ import annotations

from typing import Any

from src.config import MAX_RETRY
from src.generator import build_system_message, generate
from src.memory import extract_facts, retrieve_relevant, summarize
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
      G2+ — G2 + historical persona fact memory for verify
    """
    assert mode in {"G0", "G1", "G2", "G2+"}
    # 踩坑3：每个角色新建 messages，禁止跨角色复用
    msgs = [build_system_message(persona, mode)]
    logs: list[dict[str, Any]] = []
    facts: list[str] = []

    for turn_idx, probe in enumerate(probes, start=1):
        q = probe["text"] if isinstance(probe, dict) else str(probe)
        ptype = probe.get("type", "unknown") if isinstance(probe, dict) else "unknown"
        msgs.append({"role": "user", "content": q})

        retry = 0
        ans = generate(msgs)
        v: dict[str, Any] = {
            "conflict": False,
            "reason": "",
            "conflict_type": "none",
            "layer": "none",
        }

        if mode in {"G2", "G2+"}:
            while True:
                hist = ""
                if mode == "G2+":
                    hist = retrieve_relevant(facts, q) or summarize(facts)
                # 踩坑2：verify temperature 必须为 0（在 client/config 中固定）
                v = verify(persona, ans, hist, use_l1=use_l1)
                if (not v["conflict"]) or retry >= max_retry:
                    break
                retry += 1
                msgs.append({"role": "assistant", "content": ans})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your reply conflicts with your persona: {v['reason']}. "
                            "Re-answer the previous question strictly in character."
                        ),
                    }
                )
                ans = generate(msgs)
                # 踩坑1：临时消息必须 pop，否则历史越来越脏
                msgs.pop()
                msgs.pop()

        msgs.append({"role": "assistant", "content": ans})

        if mode == "G2+":
            facts.extend(extract_facts(q, ans))

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
                "layer": v.get("layer", "") if mode in {"G2", "G2+"} else "",
                "facts_so_far": list(facts) if mode == "G2+" else [],
            }
        )

    return logs

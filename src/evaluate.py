"""Metrics + LLM-as-judge scoring."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src import prompts
from src.client import chat_json
from src.config import JUDGE_MODEL, JUDGE_TEMPERATURE
from src.verifier import verify


def drift_rate(logs: list[dict[str, Any]]) -> float | None:
    """conflict count / total; for G0/G1 run offline verify if conflict is None."""
    if not logs:
        return None
    conflicts = [x for x in logs if x.get("conflict") is True]
    known = [x for x in logs if x.get("conflict") is not None]
    if not known:
        return None
    return len(conflicts) / len(known)


def retry_rate(logs: list[dict[str, Any]]) -> float:
    if not logs:
        return 0.0
    return sum(1 for x in logs if (x.get("retry") or 0) > 0) / len(logs)


def fix_success_rate(logs: list[dict[str, Any]]) -> float | None:
    """Among turns that retried, fraction that ended conflict=False."""
    retried = [x for x in logs if (x.get("retry") or 0) > 0]
    if not retried:
        return None
    return sum(1 for x in retried if not x.get("conflict")) / len(retried)


def annotate_conflicts(
    persona: dict[str, Any],
    logs: list[dict[str, Any]],
    *,
    use_l1: bool = True,
) -> list[dict[str, Any]]:
    """Offline verify for G0/G1 so drift rate is comparable across groups."""
    out = []
    for row in logs:
        row = dict(row)
        if row.get("conflict") is None:
            try:
                v = verify(persona, row["a"], use_l1=use_l1)
                row["conflict"] = v["conflict"]
                row["reason"] = v["reason"]
                row["conflict_type"] = v.get("conflict_type", "")
                row["layer"] = v.get("layer", "")
            except Exception as e:
                row["conflict"] = None
                row["reason"] = f"verify_failed: {e}"
                row["conflict_type"] = ""
                row["layer"] = "error"
        out.append(row)
    return out


def judge_consistency(persona: dict[str, Any], q: str, a: str) -> dict[str, Any]:
    persona_str = json.dumps(
        {
            "name": persona.get("name"),
            "personality": persona.get("personality"),
            "background": persona.get("background"),
            "hobbies": persona.get("hobbies"),
            "speech_style": persona.get("speech_style"),
            "raw_persona": persona.get("raw_persona"),
        },
        ensure_ascii=False,
    )
    return chat_json(
        [
            {
                "role": "user",
                "content": prompts.CONSISTENCY_JUDGE.format(persona=persona_str, q=q, a=a),
            }
        ],
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
    )


def judge_fluency(a: str) -> dict[str, Any]:
    return chat_json(
        [{"role": "user", "content": prompts.FLUENCY_JUDGE.format(a=a)}],
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
    )


def score_logs(persona: dict[str, Any], logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in logs:
        c = judge_consistency(persona, row["q"], row["a"])
        f = judge_fluency(row["a"])
        scored.append(
            {
                **row,
                "consistency": int(c.get("score", 3)),
                "consistency_reason": c.get("reason", ""),
                "fluency": int(f.get("score", 3)),
                "fluency_reason": f.get("reason", ""),
            }
        )
    return scored


def summarize_run(all_logs: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list] = defaultdict(list)
    for x in all_logs:
        by_type[x.get("type", "unknown")].append(x)

    def avg(key: str) -> float | None:
        vals = [x[key] for x in all_logs if key in x and x[key] is not None]
        return sum(vals) / len(vals) if vals else None

    known = [x for x in all_logs if x.get("conflict") is not None]
    drift = (sum(1 for x in known if x["conflict"]) / len(known)) if known else None

    return {
        "n": len(all_logs),
        "drift_rate": drift,
        "consistency_avg": avg("consistency"),
        "fluency_avg": avg("fluency"),
        "retry_rate": retry_rate(all_logs),
        "fix_success_rate": fix_success_rate(all_logs),
        "by_type_drift": {
            t: (sum(1 for x in rows if x.get("conflict")) / len(rows) if rows else None)
            for t, rows in by_type.items()
            if any(x.get("conflict") is not None for x in rows)
        },
    }


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

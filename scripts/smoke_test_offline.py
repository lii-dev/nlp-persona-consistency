"""Offline smoke test: archive/prompt/L1/probe wiring without calling the API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PERSONAS_PATH, PROBES_PATH
from src.prompts import sys_prompt
from src.verifier import l1_rule_check


def main() -> None:
    personas = [json.loads(l) for l in PERSONAS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    assert len(personas) == 20
    assert probes["meta"]["total"] == 200

    p = personas[6]  # vegan / no sweets
    assert "G0" in sys_prompt(p, "G0") or "Traits" in sys_prompt(p, "G0")
    assert "Personality:" in sys_prompt(p, "G1")

    hit = l1_rule_check(p, "I love sweets and cake every day!")
    print("L1 on sweets conflict:", hit)
    print("p007 raw:", p["raw_persona"])
    print("probe types p001:", [x["type"] for x in probes["probes"]["p001"]])
    print("[ok] offline smoke passed")


if __name__ == "__main__":
    main()

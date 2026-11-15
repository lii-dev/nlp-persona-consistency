"""
抽取人工评测表（默认 40 条，四组各约 10 条）。
输出 results/eval/human_check_template.csv —— 你只需填 human_conflict 列（true/false）。
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

random.seed(42)
MODE_DIR = {"G0": "G0", "G1": "G1", "G2": "G2", "G2+": "G2plus"}
OUT = ROOT / "results" / "eval" / "human_check_template.csv"


def load_rows(mode: str, n: int = 10) -> list[dict]:
    folder = ROOT / "results" / MODE_DIR[mode]
    rows = []
    for fp in sorted(folder.glob("p*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for log in data["logs"]:
            rows.append(
                {
                    "mode": mode,
                    "persona_id": data["persona_id"],
                    "turn": log["turn"],
                    "type": log.get("type", ""),
                    "q": log["q"],
                    "a": log["a"],
                    "model_conflict": log.get("conflict"),
                }
            )
    if len(rows) > n:
        # 尽量覆盖诱导类
        hard = [r for r in rows if r["type"] in {"style_induction", "false_memory", "anti_induction", "setting_boundary", "long_range"}]
        easy = [r for r in rows if r not in hard]
        take_hard = min(len(hard), max(5, n // 2))
        picked = random.sample(hard, take_hard) + random.sample(easy, n - take_hard)
        random.shuffle(picked)
        return picked
    return rows


def main() -> None:
    all_rows = []
    for m in MODE_DIR:
        all_rows.extend(load_rows(m, 10))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "mode",
        "persona_id",
        "turn",
        "type",
        "q",
        "a",
        "model_conflict",
        "human_conflict",  # 你填 true/false
        "note",
    ]
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(all_rows, 1):
            w.writerow(
                {
                    "id": i,
                    **{k: r[k] for k in ["mode", "persona_id", "turn", "type", "q", "a", "model_conflict"]},
                    "human_conflict": "",
                    "note": "",
                }
            )
    print(f"wrote {OUT} ({len(all_rows)} rows)")
    print("填完 human_conflict 后运行: python scripts/score_human_check.py")


if __name__ == "__main__":
    main()

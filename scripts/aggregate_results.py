"""Aggregate G0/G1/G2/G2+ summaries into a comparison table for the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR
from src.evaluate import dump_json

MODES = [("G0", "G0"), ("G1", "G1"), ("G2", "G2"), ("G2+", "G2plus")]


def main() -> None:
    table = {}
    for label, folder in MODES:
        p = RESULTS_DIR / folder / "summary.json"
        judge = RESULTS_DIR / "eval" / f"{folder}_judge_summary.json"
        row = {"drift_rate": None, "consistency": None, "fluency": None, "retry_rate": None, "fix_success_rate": None}
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            row["drift_rate"] = s.get("drift_rate")
            row["retry_rate"] = s.get("retry_rate")
            row["fix_success_rate"] = s.get("fix_success_rate")
        if judge.exists():
            j = json.loads(judge.read_text(encoding="utf-8"))
            row["consistency"] = j.get("consistency_avg")
            row["fluency"] = j.get("fluency_avg")
            if row["drift_rate"] is None:
                row["drift_rate"] = j.get("drift_rate")
        table[label] = row

    out = RESULTS_DIR / "eval" / "comparison_table.json"
    dump_json(out, table)

    print("| 指标 | G0 | G1 | G2 | G2+ |")
    print("|---|---|---|---|---|")
    keys = [
        ("漂移率↓", "drift_rate", True),
        ("一致性↑", "consistency", False),
        ("流畅度", "fluency", False),
        ("重试率", "retry_rate", True),
        ("修正成功率", "fix_success_rate", True),
    ]
    for name, key, pct in keys:
        cells = []
        for label, _ in MODES:
            v = table[label].get(key)
            if v is None:
                cells.append("—")
            elif pct:
                cells.append(f"{v*100:.1f}%")
            else:
                cells.append(f"{v:.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()

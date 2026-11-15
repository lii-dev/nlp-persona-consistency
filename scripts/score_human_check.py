"""根据 human_check_template.csv 计算人工与模型一致率。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "eval" / "human_check_template.csv"
OUT = ROOT / "results" / "eval" / "human_agreement.json"


def parse_bool(x: str):
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "冲突", "是"}:
        return True
    if s in {"0", "false", "no", "n", "不冲突", "否"}:
        return False
    return None


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH}; run sample_human_check.py first")

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))
    labeled = []
    for r in rows:
        h = parse_bool(r.get("human_conflict", ""))
        if h is None:
            continue
        m = r.get("model_conflict")
        if m in ("", None):
            continue
        m_bool = str(m).lower() in {"true", "1"}
        labeled.append({"mode": r["mode"], "agree": h == m_bool})

    if not labeled:
        raise SystemExit("尚未填写 human_conflict 列")

    overall = sum(x["agree"] for x in labeled) / len(labeled)
    by_mode = {}
    for m in ["G0", "G1", "G2", "G2+"]:
        sub = [x for x in labeled if x["mode"] == m]
        by_mode[m] = (sum(x["agree"] for x in sub) / len(sub)) if sub else None

    result = {"n_labeled": len(labeled), "agreement": overall, "by_mode": by_mode}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"答辩可用：人工与 judge/校验一致率 = {overall*100:.1f}% (n={len(labeled)})")


if __name__ == "__main__":
    main()

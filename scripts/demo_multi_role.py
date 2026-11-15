"""展望3：多角色协同对话演示（独立档案 + 校验 + RAG 记忆）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR
from src.evaluate import dump_json
from src.multi_role import quick_pairwise_demo


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "G2+"
    result = quick_pairwise_demo(mode=mode)
    out = RESULTS_DIR / "demos" / "multi_role.json"
    dump_json(out, result)

    conflicts = sum(
        1
        for logs in result["per_role_logs"].values()
        for row in logs
        if row.get("conflict")
    )
    retries = sum(row.get("retry", 0) for logs in result["per_role_logs"].values() for row in logs)
    print(json.dumps({"out": str(out), "turns": len(result["transcript"]), "conflicts": conflicts, "retries": retries}, ensure_ascii=False))
    for t in result["transcript"]:
        print(f"[{t['speaker']}] {t['text'][:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()

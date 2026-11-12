"""
Score consistency + fluency with qwen-plus (LLM-as-judge).

Usage:
  python scripts/run_judge.py --mode G0
  python scripts/run_judge.py --mode G2 --sample 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PERSONAS_PATH, RESULTS_DIR
from src.evaluate import dump_json, judge_consistency, judge_fluency, summarize_run

MODE_DIR = {"G0": "G0", "G1": "G1", "G2": "G2", "G2+": "G2plus"}
random.seed(42)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=list(MODE_DIR))
    parser.add_argument("--sample", type=int, default=0, help="Optional subsample size")
    args = parser.parse_args()

    personas = {p["id"]: p for p in load_jsonl(PERSONAS_PATH)}
    in_dir = RESULTS_DIR / MODE_DIR[args.mode]
    files = sorted(in_dir.glob("p*.json"))
    if not files:
        raise SystemExit(f"No results in {in_dir}. Run run_experiment.py first.")

    rows: list[dict] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        pid = data["persona_id"]
        for log in data["logs"]:
            rows.append({"persona_id": pid, **log})

    if args.sample > 0 and args.sample < len(rows):
        rows = random.sample(rows, args.sample)

    scored = []
    for row in tqdm(rows, desc=f"judge-{args.mode}"):
        persona = personas[row["persona_id"]]
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

    out_dir = RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / f"{MODE_DIR[args.mode]}_scored.json", scored)
    summary = summarize_run(scored)
    summary["mode"] = args.mode
    dump_json(out_dir / f"{MODE_DIR[args.mode]}_judge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

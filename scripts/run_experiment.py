"""
Run G0 / G1 / G2 / G2+ experiments on the evaluation probe set.

Usage:
  python scripts/run_experiment.py --mode G0
  python scripts/run_experiment.py --mode G2 --limit 2   # smoke test 2 personas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PERSONAS_PATH, PROBES_PATH, RESULTS_DIR
from src.dialog import run_dialog
from src.evaluate import annotate_conflicts, dump_json, summarize_run


MODE_DIR = {
    "G0": "G0",
    "G1": "G1",
    "G2": "G2",
    "G2+": "G2plus",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["G0", "G1", "G2", "G2+"])
    parser.add_argument("--limit", type=int, default=0, help="Only first N personas (0=all)")
    parser.add_argument("--persona-id", type=str, default="", help="Run a single persona id")
    parser.add_argument("--offline-annotate", action="store_true",
                        help="For G0/G1, also run verifier offline to compute drift")
    parser.add_argument("--no-l1", action="store_true", help="Disable L1 rule layer")
    parser.add_argument("--resume", action="store_true", help="Skip personas that already have result files")
    args = parser.parse_args()

    personas = load_jsonl(PERSONAS_PATH)
    probes_all = json.loads(PROBES_PATH.read_text(encoding="utf-8"))["probes"]

    if args.persona_id:
        personas = [p for p in personas if p["id"] == args.persona_id]
    if args.limit > 0:
        personas = personas[: args.limit]

    out_dir = RESULTS_DIR / MODE_DIR[args.mode]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_logs: list[dict] = []
    t0 = time.time()

    for persona in tqdm(personas, desc=f"mode={args.mode}"):
        pid = persona["id"]
        out_fp = out_dir / f"{pid}.json"
        if args.resume and out_fp.exists():
            prev = json.loads(out_fp.read_text(encoding="utf-8"))
            for row in prev.get("logs", []):
                all_logs.append({"persona_id": pid, **row})
            continue
        probes = probes_all[pid]
        try:
            logs = run_dialog(persona, probes, args.mode, use_l1=not args.no_l1)
            if args.mode in {"G0", "G1"} and args.offline_annotate:
                logs = annotate_conflicts(persona, logs, use_l1=not args.no_l1)
        except Exception as e:
            dump_json(out_dir / f"{pid}.error.json", {"persona_id": pid, "error": str(e)})
            print(f"[warn] {pid} failed: {e}")
            continue

        payload = {
            "persona_id": pid,
            "mode": args.mode,
            "persona": {
                "name": persona.get("name"),
                "raw_persona": persona.get("raw_persona"),
            },
            "logs": logs,
        }
        dump_json(out_dir / f"{pid}.json", payload)
        for row in logs:
            all_logs.append({"persona_id": pid, **row})

    summary = summarize_run(all_logs)
    summary["mode"] = args.mode
    summary["n_personas"] = len(personas)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    dump_json(out_dir / "summary.json", summary)
    dump_json(out_dir / "all_logs.jsonl".replace(".jsonl", "_meta.json"), {"n": len(all_logs)})

    # also write jsonl for easy aggregation
    with (out_dir / "all_logs.jsonl").open("w", encoding="utf-8") as f:
        for row in all_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[ok] results -> {out_dir}")


if __name__ == "__main__":
    main()

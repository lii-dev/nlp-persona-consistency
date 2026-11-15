"""生成报告用 4 张图 -> results/figures/"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 中文标签在部分环境缺字体时用英文，保证能出图
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

MODES = ["G0", "G1", "G2", "G2+"]
MODE_DIR = {"G0": "G0", "G1": "G1", "G2": "G2", "G2+": "G2plus"}

TYPE_ORDER = [
    "self_intro",
    "detail_recall",
    "anti_induction",
    "style_induction",
    "false_memory",
    "setting_boundary",
    "long_range",
]
TYPE_LABEL = {
    "self_intro": "Self-intro",
    "detail_recall": "Detail",
    "anti_induction": "Anti-induct",
    "style_induction": "Style",
    "false_memory": "False-mem",
    "setting_boundary": "Boundary",
    "long_range": "Long-range",
}


def load_comparison() -> dict:
    return json.loads((ROOT / "results/eval/comparison_table.json").read_text(encoding="utf-8"))


def load_summary(mode: str) -> dict:
    p = ROOT / "results" / MODE_DIR[mode] / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fig1_drift(comp: dict) -> None:
    vals = [comp[m]["drift_rate"] * 100 for m in MODES]
    colors = ["#6B7280", "#9CA3AF", "#2563EB", "#059669"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(MODES, vals, color=colors, width=0.62)
    ax.set_ylabel("Persona Drift Rate (%)")
    ax.set_title("Fig.1 Overall Drift Rate (lower is better)")
    ax.set_ylim(0, max(vals) * 1.35 + 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_drift.png", dpi=180)
    plt.close(fig)


def fig2_by_type() -> None:
    # G0 vs G2 vs G2+
    data = {m: load_summary(m).get("by_type_drift", {}) for m in ["G0", "G2", "G2+"]}
    x = np.arange(len(TYPE_ORDER))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (m, color) in enumerate([("G0", "#6B7280"), ("G2", "#2563EB"), ("G2+", "#059669")]):
        vals = [float(data[m].get(t, 0) or 0) * 100 for t in TYPE_ORDER]
        ax.bar(x + (i - 1) * w, vals, width=w, label=m, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABEL[t] for t in TYPE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Drift Rate by Probe Type (%)")
    ax.set_title("Fig.2 Drift by Probe Type (G0 vs G2 vs G2+)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_by_type.png", dpi=180)
    plt.close(fig)


def fig3_retry_fix(comp: dict) -> None:
    retry = [((comp[m]["retry_rate"] or 0) * 100) for m in MODES]
    fix = [((comp[m]["fix_success_rate"] or 0) * 100) if comp[m]["fix_success_rate"] is not None else 0 for m in MODES]
    x = np.arange(len(MODES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, retry, width=w, label="Retry rate", color="#F59E0B")
    ax.bar(x + w / 2, fix, width=w, label="Fix success", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Fig.3 Retry Rate & Fix Success (G2/G2+)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_retry_fix.png", dpi=180)
    plt.close(fig)


def fig4_scores(comp: dict) -> None:
    cons = [comp[m]["consistency"] for m in MODES]
    flu = [comp[m]["fluency"] for m in MODES]
    x = np.arange(len(MODES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, cons, width=w, label="Consistency (1/3/5)", color="#2563EB")
    ax.bar(x + w / 2, flu, width=w, label="Fluency (1/3/5)", color="#7C3AED")
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Judge Score")
    ax.set_ylim(1, 5.4)
    ax.set_title("Fig.4 Consistency & Fluency (sample=40/mode)")
    ax.legend(frameon=False, loc="lower right")
    ax.axhline(5, color="#E5E7EB", lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_scores.png", dpi=180)
    plt.close(fig)


def main() -> None:
    comp = load_comparison()
    fig1_drift(comp)
    fig2_by_type()
    fig3_retry_fix(comp)
    fig4_scores(comp)
    print("wrote:")
    for p in sorted(FIG.glob("fig*.png")):
        print(" ", p)


if __name__ == "__main__":
    main()

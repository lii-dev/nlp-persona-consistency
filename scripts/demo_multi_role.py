"""
展望3：多角色协同小规模演示（不进指标）。

默认：林黛玉 vs 王熙凤 对话 5 轮，展示档案可切换 / 并行维持。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR
from src.dialog import run_dialog
from src.evaluate import dump_json

DAI_YU = {
    "id": "demo_daiyu",
    "name": "Lin Daiyu",
    "personality": "sensitive, poetic, melancholic, easily hurt",
    "background": "raised in Jia household after parents' death; frail health",
    "hobbies": "writing poetry, reading, tending flowers",
    "speech_style": "literary, soft, classical Chinese-influenced English",
    "raw_persona": "sensitive poetic melancholic | frail orphan in Jia house | loves poetry and flowers",
    "persona_sentences": [
        "I am Lin Daiyu, sensitive and poetic.",
        "I have frail health and get emotional easily.",
        "I love composing poems and caring for flowers.",
        "I dislike loud vulgar banter.",
        "I speak in a soft literary manner.",
    ],
}

XI_FENG = {
    "id": "demo_xifeng",
    "name": "Wang Xifeng",
    "personality": "sharp, managerial, witty, formidable",
    "background": "manages the Jia household finances and staff",
    "hobbies": "managing affairs, banter, social leverage",
    "speech_style": "direct, witty, commanding, colloquial bite",
    "raw_persona": "sharp witty household manager | commanding | loves banter and control",
    "persona_sentences": [
        "I am Wang Xifeng, sharp and capable.",
        "I manage the household with a firm hand.",
        "I enjoy witty banter and getting things done.",
        "I dislike weak sentimentality getting in the way of work.",
        "I speak directly and with authority.",
    ],
}


SEED_TURNS = [
    ("Lin Daiyu", "The garden feels quieter today. Do you ever tire of managing everyone?"),
    ("Wang Xifeng", "Quiet won't keep the household fed. Someone has to keep order."),
    ("Lin Daiyu", "Order without gentleness can bruise soft hearts."),
    ("Wang Xifeng", "And gentleness without spine lets the whole house fall apart."),
    ("Lin Daiyu", "Then perhaps we each guard a different kind of fragility."),
]


def run_case(mode: str = "G2") -> dict:
    # Each character answers from their own archive on alternating user prompts.
    dy_probes = [{"type": "multi_role", "text": t} for who, t in SEED_TURNS if who == "Wang Xifeng"]
    # Better: feed the partner line as user turn to each character separately.
    logs = {"Lin Daiyu": [], "Wang Xifeng": []}

    # Construct 5-turn interleaved conversation by prompting each side with the other's line.
    # Turn plan: Xifeng line -> Daiyu replies; Daiyu line -> Xifeng replies; etc.
    dialogue = []
    # Start with Xifeng addressing Daiyu
    xifeng_open = "Sister Lin, stop sighing in the garden — the accounts won't balance themselves."
    daiyu_logs = run_dialog(DAI_YU, [{"type": "multi_role", "text": xifeng_open}], mode)
    dialogue.append({"speaker": "Wang Xifeng", "text": xifeng_open})
    dialogue.append({"speaker": "Lin Daiyu", "text": daiyu_logs[0]["a"], "meta": daiyu_logs[0]})

    followups = [
        ("Wang Xifeng", "Poetry is lovely, but can poetry pay the servants?"),
        ("Lin Daiyu", "Must every breath be measured in silver?"),
        ("Wang Xifeng", "In this house, yes — or the silver runs out."),
        ("Lin Daiyu", "Then I will keep my verses, and you your ledgers."),
    ]
    # Alternate: odd prompts to Daiyu, even to Xifeng based on who should reply
    # Simpler fixed 5-reply demo:
    prompts_to_daiyu = [
        xifeng_open,
        "Poetry is lovely, but can poetry pay the servants?",
        "In this house, yes — or the silver runs out.",
    ]
    prompts_to_xifeng = [
        "The garden is my only quiet place; must you interrupt even that?",
        "Must every breath be measured in silver?",
        "Then I will keep my verses, and you your ledgers. Can we at least not wound each other today?",
    ]

    daiyu_logs = run_dialog(
        DAI_YU, [{"type": "multi_role", "text": t} for t in prompts_to_daiyu[:3]], mode
    )
    xifeng_logs = run_dialog(
        XI_FENG, [{"type": "multi_role", "text": t} for t in prompts_to_xifeng[:2]], mode
    )

    transcript = []
    for i in range(5):
        if i % 2 == 0:
            # Xifeng speaks (from her logs or seed), then Daiyu
            xi = xifeng_logs[i // 2] if i // 2 < len(xifeng_logs) else None
            if xi:
                transcript.append({"turn": i + 1, "speaker": "Wang Xifeng", "text": xi["a"], "retry": xi["retry"]})
            dy = daiyu_logs[i // 2]
            transcript.append({"turn": i + 1, "speaker": "Lin Daiyu", "text": dy["a"], "retry": dy["retry"]})
        else:
            pass

    # Cleaner linear transcript for report
    clean = []
    clean.append({"speaker": "Wang Xifeng (prompt)", "text": prompts_to_daiyu[0]})
    clean.append({"speaker": "Lin Daiyu", "text": daiyu_logs[0]["a"]})
    clean.append({"speaker": "Lin Daiyu (prompt to Xifeng)", "text": prompts_to_xifeng[0]})
    clean.append({"speaker": "Wang Xifeng", "text": xifeng_logs[0]["a"]})
    clean.append({"speaker": "Wang Xifeng (prompt)", "text": prompts_to_daiyu[1]})
    clean.append({"speaker": "Lin Daiyu", "text": daiyu_logs[1]["a"]})
    clean.append({"speaker": "Lin Daiyu (prompt to Xifeng)", "text": prompts_to_xifeng[1]})
    clean.append({"speaker": "Wang Xifeng", "text": xifeng_logs[1]["a"]})
    clean.append({"speaker": "Wang Xifeng (prompt)", "text": prompts_to_daiyu[2]})
    clean.append({"speaker": "Lin Daiyu", "text": daiyu_logs[2]["a"]})

    return {
        "mode": mode,
        "note": "展望3 case study：多角色档案切换；不进入主指标。",
        "personas": {"Lin Daiyu": DAI_YU, "Wang Xifeng": XI_FENG},
        "transcript": clean,
        "daiyu_logs": daiyu_logs,
        "xifeng_logs": xifeng_logs,
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "G2"
    result = run_case(mode)
    out = RESULTS_DIR / "demos" / "multi_role.json"
    dump_json(out, result)
    print(json.dumps({"out": str(out), "turns": len(result["transcript"])}, ensure_ascii=False))
    for turn in result["transcript"]:
        print(f"[{turn['speaker']}] {turn['text'][:120]}")


if __name__ == "__main__":
    main()

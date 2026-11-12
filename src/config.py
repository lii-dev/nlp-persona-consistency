"""Global configuration. Training-free pipeline; no training set required."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

GEN_MODEL = os.getenv("GEN_MODEL", "qwen2.5-7b-instruct")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "qwen2.5-7b-instruct")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen-plus")

GEN_TEMPERATURE = 0.7
VERIFY_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "350"))  # 控长度，加速评测、减少跑题

MAX_RETRY = 2
N_PERSONAS = 20
PROBES_PER_PERSONA = 10  # 6 normal + 4 inductive

PERSONAS_PATH = PROCESSED_DIR / "personas.jsonl"
PROBES_PATH = PROCESSED_DIR / "probes.json"


def require_api_key() -> str:
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("sk-xxx"):
        raise RuntimeError(
            "请在项目根目录创建 .env，并设置 DASHSCOPE_API_KEY。"
            "可复制 .env.example 后填入阿里云 DashScope Key。"
        )
    return DASHSCOPE_API_KEY

"""RAG persona memory: embedding retrieval + dynamic consolidate/update (展望2)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from src import prompts
from src.client import chat_json, get_client
from src.config import EXTRACT_MODEL, MEMORY_CONSOLIDATE, VERIFY_MODEL, VERIFY_TEMPERATURE

EMBED_MODEL = "text-embedding-v3"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_client()
    # DashScope embedding API accepts batch input
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # ensure order by index
    data = sorted(resp.data, key=lambda d: d.index)
    return [list(d.embedding) for d in data]


@dataclass
class MemoryItem:
    text: str
    grounded: bool = True
    embedding: list[float] = field(default_factory=list)
    turn: int = 0


class PersonaMemoryBank:
    """Per-character memory bank with embedding RAG + LLM consolidate."""

    def __init__(self, persona: dict[str, Any], *, max_facts: int = 24):
        self.persona = persona
        self.max_facts = max_facts
        self.items: list[MemoryItem] = []
        # seed with original persona sentences
        seeds = persona.get("persona_sentences") or []
        raw = persona.get("raw_persona") or ""
        if not seeds and raw:
            seeds = [s.strip() for s in raw.split("|") if s.strip()]
        for s in seeds:
            self.add_fact(s, grounded=True, turn=0, embed=False)
        self._reembed_missing()

    def add_fact(self, text: str, *, grounded: bool = True, turn: int = 0, embed: bool = True) -> None:
        text = text.strip()
        if not text:
            return
        # near-dup skip (keyword)
        for it in self.items:
            if it.text.lower() == text.lower():
                return
        item = MemoryItem(text=text, grounded=grounded, turn=turn)
        if embed:
            try:
                item.embedding = embed_texts([text])[0]
            except Exception:
                item.embedding = []
        self.items.append(item)

    def _reembed_missing(self) -> None:
        missing = [it for it in self.items if not it.embedding]
        if not missing:
            return
        try:
            vecs = embed_texts([it.text for it in missing])
            for it, v in zip(missing, vecs):
                it.embedding = v
        except Exception:
            pass

    def retrieve(self, query: str, *, top_k: int = 5) -> list[str]:
        if not self.items:
            return []
        # embedding RAG
        try:
            qv = embed_texts([query])[0]
            scored = [(_cosine(qv, it.embedding), it) for it in self.items if it.embedding]
            scored.sort(key=lambda x: x[0], reverse=True)
            picked = [it.text for score, it in scored[:top_k] if score >= 0.25]
            if picked:
                return picked
        except Exception:
            pass
        # keyword fallback
        q_tokens = set(_tokenize(query))
        scored_kw: list[tuple[int, MemoryItem]] = []
        for it in self.items:
            score = len(q_tokens & set(_tokenize(it.text)))
            scored_kw.append((score, it))
        scored_kw.sort(key=lambda x: (x[0], x[1].turn), reverse=True)
        picked = [it.text for s, it in scored_kw[:top_k] if s > 0]
        return picked or [it.text for it in self.items[-top_k:]]

    def retrieve_text(self, query: str, *, top_k: int = 5) -> str:
        return " | ".join(self.retrieve(query, top_k=top_k))

    def extract_and_update(
        self,
        question: str,
        answer: str,
        *,
        turn: int,
        consolidate: bool | None = None,
    ) -> dict[str, Any]:
        """Extract facts from turn, consolidate into bank (dynamic update)."""
        import os

        # SMOKE_FAST=1：跳过 LLM 抽取；正式跑：qwen-flash 抽取，consolidate 默认关
        fast = os.getenv("SMOKE_FAST", "0") == "1"
        if consolidate is None:
            consolidate = MEMORY_CONSOLIDATE and not fast
        persona_sketch = self.persona.get("raw_persona") or ""
        new_facts: list[dict[str, Any]] = []

        if fast:
            snippet = " ".join(answer.split())[:120]
            if snippet:
                new_facts = [{"text": f"I said: {snippet}", "grounded": False}]
        else:
            try:
                data = chat_json(
                    [
                        {
                            "role": "user",
                            "content": prompts.FACT_EXTRACT_PROMPT.format(
                                q=question,
                                a=answer[:500],
                                persona=persona_sketch[:300],
                            ),
                        }
                    ],
                    model=EXTRACT_MODEL,
                    temperature=VERIFY_TEMPERATURE,
                    max_tokens=200,
                )
                raw = data.get("facts") or []
                for f in raw:
                    if isinstance(f, dict):
                        new_facts.append(
                            {
                                "text": str(f.get("text", "")).strip(),
                                "grounded": bool(f.get("grounded", False)),
                            }
                        )
                    else:
                        new_facts.append({"text": str(f).strip(), "grounded": False})
            except Exception as e:
                print(f"    [warn] fact extract failed: {e}", flush=True)
                new_facts = []

        dropped: list[str] = []
        if not new_facts:
            return {"added": [], "dropped_drift": [], "size": len(self.items)}

        for f in new_facts:
            if f["text"]:
                self.add_fact(f["text"], grounded=f["grounded"], turn=turn, embed=True)

        # consolidate 默认关闭；开启时仅每 5 轮或超限触发
        need_consolidate = consolidate and (turn % 5 == 0 or len(self.items) > self.max_facts)
        if need_consolidate:
            existing = [it.text for it in self.items]
            try:
                cons = chat_json(
                    [
                        {
                            "role": "user",
                            "content": prompts.MEMORY_CONSOLIDATE_PROMPT.format(
                                max_facts=self.max_facts,
                                persona=persona_sketch[:400],
                                existing="\n".join(f"- {x}" for x in existing[:20]) or "(empty)",
                                new_facts="\n".join(
                                    f"- {f['text']} (grounded={f['grounded']})" for f in new_facts
                                ),
                            ),
                        }
                    ],
                    model=EXTRACT_MODEL,
                    temperature=VERIFY_TEMPERATURE,
                    max_tokens=400,
                )
                memories = [str(m).strip() for m in (cons.get("memories") or []) if str(m).strip()]
                dropped = [str(m).strip() for m in (cons.get("dropped_drift") or []) if str(m).strip()]
                if memories:
                    grounded_set = {it.text.lower() for it in self.items if it.grounded}
                    for f in new_facts:
                        if f["grounded"]:
                            grounded_set.add(f["text"].lower())
                    self.items = []
                    for m in memories[: self.max_facts]:
                        self.add_fact(
                            m,
                            grounded=(m.lower() in grounded_set)
                            or any(m.lower() in s or s in m.lower() for s in grounded_set),
                            turn=turn,
                            embed=False,
                        )
                    self._reembed_missing()
            except Exception:
                if len(self.items) > self.max_facts:
                    grounded = [it for it in self.items if it.grounded]
                    others = [it for it in self.items if not it.grounded]
                    self.items = (grounded + others)[: self.max_facts]
        elif len(self.items) > self.max_facts:
            grounded = [it for it in self.items if it.grounded]
            others = [it for it in self.items if not it.grounded]
            self.items = (grounded + others)[: self.max_facts]

        return {"added": [f["text"] for f in new_facts], "dropped_drift": dropped, "size": len(self.items)}

    def all_texts(self) -> list[str]:
        return [it.text for it in self.items]


# ---- backward-compatible helpers used by older call sites ----

def extract_facts(question: str, answer: str) -> list[str]:
    try:
        data = chat_json(
            [
                {
                    "role": "user",
                    "content": prompts.FACT_EXTRACT_PROMPT.format(
                        q=question, a=answer, persona=""
                    ),
                }
            ],
            model=VERIFY_MODEL,
            temperature=VERIFY_TEMPERATURE,
        )
        out = []
        for f in data.get("facts") or []:
            if isinstance(f, dict):
                out.append(str(f.get("text", "")).strip())
            else:
                out.append(str(f).strip())
        return [x for x in out if x]
    except Exception:
        return []


def summarize(facts: list[str], *, top_k: int = 12) -> str:
    if not facts:
        return ""
    return " | ".join(facts[-top_k:])


def retrieve_relevant(facts: list[str], query: str, *, top_k: int = 5) -> str:
    if not facts:
        return ""
    try:
        bank = PersonaMemoryBank({"persona_sentences": facts, "raw_persona": ""}, max_facts=len(facts) + 5)
        # clear seeds duplication — PersonaMemoryBank seeds from persona_sentences
        return bank.retrieve_text(query, top_k=top_k)
    except Exception:
        q_tokens = set(_tokenize(query))
        scored = []
        for f in facts:
            scored.append((len(q_tokens & set(_tokenize(f))), f))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [f for s, f in scored[:top_k] if s > 0] or facts[-top_k:]
        return " | ".join(picked)

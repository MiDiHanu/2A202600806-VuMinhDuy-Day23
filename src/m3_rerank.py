"""M3 Rerank — stub for Lab 24 (copy full implementation from Day 18)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RerankedResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class CrossEncoderReranker:
    """Cross-encoder reranker stub."""

    def rerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[RerankedResult]:
        if not docs:
            return []
        q_words = set(query.lower().split())
        scored = []
        for doc in docs:
            overlap = len(q_words & set(doc["text"].lower().split()))
            scored.append(RerankedResult(
                text=doc["text"],
                score=overlap / max(len(q_words), 1),
                metadata=doc.get("metadata", {}),
            ))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

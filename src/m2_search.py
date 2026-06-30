"""M2 Search — stub for Lab 24 (copy full implementation from Day 18)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridSearch:
    """Hybrid BM25 + Dense search stub."""

    def __init__(self):
        self._chunks: list[dict] = []

    def index(self, chunks: list[dict]) -> None:
        self._chunks = chunks

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        if not self._chunks:
            return []
        # Simple keyword fallback
        q_words = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            overlap = len(q_words & set(chunk["text"].lower().split()))
            scored.append(SearchResult(
                text=chunk["text"],
                score=overlap / max(len(q_words), 1),
                metadata=chunk.get("metadata", {}),
            ))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

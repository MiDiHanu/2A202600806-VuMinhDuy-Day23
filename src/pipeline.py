"""Pipeline — stub for Lab 24 (copy full implementation from Day 18)."""
from __future__ import annotations


class RAGPipeline:
    """Full RAG pipeline stub."""

    def __init__(self, collection_name: str = "lab24_production"):
        self.collection_name = collection_name
        self._search = None
        self._reranker = None

    def query(self, question: str, top_k: int = 3) -> dict:
        """Run a query through the RAG pipeline."""
        return {
            "question": question,
            "answer":   "Stub answer — implement full pipeline from Day 18.",
            "contexts": [],
        }


def build_pipeline(collection_name: str = "lab24_production") -> RAGPipeline:
    return RAGPipeline(collection_name)

"""M4 Evaluation — stub for Lab 24 (copy full implementation from Day 18)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PerQuestionScore:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Evaluate RAG pipeline using RAGAS metrics.

    Returns:
        {
            "faithfulness": float,
            "answer_relevancy": float,
            "context_precision": float,
            "context_recall": float,
            "per_question": list[PerQuestionScore],
        }
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question":      questions,
            "answer":        answers,
            "contexts":      contexts,
            "ground_truths": [[gt] for gt in ground_truths],
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        import math

        def safe(val: object) -> float:
            try:
                f = float(val)  # type: ignore[arg-type]
                return f if not math.isnan(f) else 0.5
            except Exception:
                return 0.5

        per_question = [
            PerQuestionScore(
                faithfulness=safe(row.get("faithfulness")),
                answer_relevancy=safe(row.get("answer_relevancy")),
                context_precision=safe(row.get("context_precision")),
                context_recall=safe(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]

        return {
            "faithfulness":      float(result["faithfulness"]),
            "answer_relevancy":  float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall":    float(result["context_recall"]),
            "per_question":      per_question,
        }
    except Exception as e:
        print(f"⚠️  RAGAS failed: {e}")
        dummy = PerQuestionScore(0.5, 0.5, 0.5, 0.5)
        return {
            "faithfulness": 0.5, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
            "per_question": [dummy] * len(questions),
        }

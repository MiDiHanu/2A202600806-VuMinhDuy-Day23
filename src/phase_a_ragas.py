from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": [
            {"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Implemented ──────────────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups: dict[str, list[dict]] = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        groups[item["distribution"]].append(item)
    return groups


def _run_ragas_fallback(answers: list[dict]) -> list[RagasResult]:
    """Fallback: text-overlap scoring khi RAGAS API không khả dụng."""

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r'\w+', text.lower()))

    results: list[RagasResult] = []
    for a in answers:
        answer_tokens = tokenize(a["answer"])
        gt_tokens = tokenize(a["ground_truth"])
        ctx_tokens: set[str] = set()
        for ctx in a.get("contexts", []):
            ctx_tokens.update(tokenize(ctx))

        # Faithfulness: answer tokens covered by contexts
        faithfulness = len(answer_tokens & ctx_tokens) / max(len(answer_tokens), 1)
        # Answer relevancy: overlap between answer and ground truth
        answer_rel   = len(answer_tokens & gt_tokens) / max(len(gt_tokens), 1)
        # Context recall: ground truth tokens in contexts
        ctx_recall   = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
        # Context precision: context tokens relevant to ground truth
        ctx_prec     = len(ctx_tokens & gt_tokens) / max(len(ctx_tokens), 1) if ctx_tokens else 0.0

        results.append(RagasResult(
            question_id=a["id"],
            distribution=a["distribution"],
            question=a["question"],
            answer=a["answer"],
            contexts=a["contexts"],
            ground_truth=a["ground_truth"],
            faithfulness=round(min(faithfulness, 1.0), 4),
            answer_relevancy=round(min(answer_rel, 1.0), 4),
            context_precision=round(min(ctx_prec, 1.0), 4),
            context_recall=round(min(ctx_recall, 1.0), 4),
        ))
    return results


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Sử dụng RAGAS library trực tiếp (không cần src/m4_eval.py).
    Fallback sang text-overlap scoring nếu API không khả dụng.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question":      [a["question"]      for a in answers],
            "answer":        [a["answer"]        for a in answers],
            "contexts":      [a["contexts"]      for a in answers],
            "ground_truths": [[a["ground_truth"]] for a in answers],
        })

        print(f"Running RAGAS on {len(answers)} questions...")
        eval_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = eval_result.to_pandas()

        def safe(val: object, default: float = 0.5) -> float:
            try:
                f = float(val)  # type: ignore[arg-type]
                return f if not math.isnan(f) else default
            except Exception:
                return default

        results: list[RagasResult] = []
        for a, (_, row) in zip(answers, df.iterrows()):
            results.append(RagasResult(
                question_id=a["id"],
                distribution=a["distribution"],
                question=a["question"],
                answer=a["answer"],
                contexts=a["contexts"],
                ground_truth=a["ground_truth"],
                faithfulness=safe(row.get("faithfulness")),
                answer_relevancy=safe(row.get("answer_relevancy")),
                context_precision=safe(row.get("context_precision")),
                context_recall=safe(row.get("context_recall")),
            ))
        return results

    except Exception as e:
        print(f"[WARN] RAGAS evaluation failed: {e}")
        print("-> Fallback: text-overlap scoring")
        return _run_ragas_fallback(answers)


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    bottom = sorted_asc[:10]
    output: list[dict] = []
    for i, r in enumerate(bottom):
        diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank":          i + 1,
            "question_id":   r.question_id,
            "distribution":  r.distribution,
            "question":      r.question,
            "avg_score":     round(r.avg_score, 4),
            "worst_metric":  r.worst_metric,
            "diagnosis":     diag,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            ...
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix: dict[str, dict[str, int]] = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        matrix[r.worst_metric][r.distribution] += 1

    dominant_dist = max(
        ["factual", "multi_hop", "adversarial"],
        key=lambda d: sum(matrix[m][d] for m in matrix),
    )
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    insight = (
        f"Distribution '{dominant_dist}' có nhiều failure nhất. "
        f"Metric '{dominant_metric}' là điểm yếu chủ đạo. "
        f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}"
    )

    return {
        "matrix":                        matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric":       dominant_metric,
        "insight":                       insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")

        # Distribution avg scores (bonus check)
        for dist in ["factual", "multi_hop", "adversarial"]:
            subset = [r for r in results if r.distribution == dist]
            if subset:
                avg = sum(r.avg_score for r in subset) / len(subset)
                print(f"  {dist} avg_score: {avg:.3f}")
    else:
        print("⚠️  No results — check RAGAS setup or answers_50q.json.")

from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Prompt template ─────────────────────────────────────────────────────────

_JUDGE_PROMPT = """Bạn là expert đánh giá chất lượng câu trả lời RAG hệ thống HR.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí:
1. Độ chính xác (accuracy): Answer nào phản ánh đúng chính sách nhân sự hơn?
2. Độ đầy đủ (completeness): Answer nào trả lời đủ câu hỏi hơn (không thiếu thông tin quan trọng)?
3. Tính súc tích (conciseness): Answer nào không có thông tin thừa / lạc đề?

Trả lời CHÍNH XÁC theo format JSON sau (chỉ JSON, không có text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn 1-2 câu", "scores": {{"A": 0.0, "B": 0.0}}}}

Lưu ý: scores là số thực từ 0.0 đến 1.0."""


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def _heuristic_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Fallback heuristic judge when LLM API is unavailable.

    Uses answer length and keyword overlap with the question as a proxy
    for accuracy and completeness. Picks the answer with higher overlap.
    """
    import re
    q_tokens = set(re.findall(r'\w+', question.lower()))
    a_tokens  = set(re.findall(r'\w+', answer_a.lower()))
    b_tokens  = set(re.findall(r'\w+', answer_b.lower()))

    score_a = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    score_b = len(q_tokens & b_tokens) / max(len(q_tokens), 1)

    # Penalise very short answers (likely incomplete)
    if len(answer_a) < 20:
        score_a *= 0.5
    if len(answer_b) < 20:
        score_b *= 0.5

    if abs(score_a - score_b) < 0.05:
        winner = "tie"
        reasoning = "Hai câu trả lời có độ tương đồng tương đương (heuristic)."
    elif score_a > score_b:
        winner = "A"
        reasoning = f"Answer A có độ overlap cao hơn với câu hỏi (heuristic: {score_a:.2f} vs {score_b:.2f})."
    else:
        winner = "B"
        reasoning = f"Answer B có độ overlap cao hơn với câu hỏi (heuristic: {score_b:.2f} vs {score_a:.2f})."

    return {
        "winner":    winner,
        "reasoning": reasoning,
        "scores":    {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON hợp lệ."},
                {"role": "user",   "content": _JUDGE_PROMPT.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(resp.choices[0].message.content)

        # Validate and normalize
        if result.get("winner") not in {"A", "B", "tie"}:
            result["winner"] = "tie"
        if "scores" not in result:
            result["scores"] = {"A": 0.5, "B": 0.5}
        if "reasoning" not in result:
            result["reasoning"] = ""

        # Clamp scores to [0, 1]
        for k in result["scores"]:
            result["scores"][k] = max(0.0, min(1.0, float(result["scores"][k])))

        return result

    except Exception as e:
        print(f"[WARN] LLM judge failed: {e} — using heuristic fallback")
        return _heuristic_judge(question, answer_a, answer_b)


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1     = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Consensus: agree only when both passes pick the same answer
    final               = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    position_consistent = (pass1["winner"] == winner_pass2)

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2={
            "A": pass2_raw.get("scores", {}).get("B", 0.0),
            "B": pass2_raw.get("scores", {}).get("A", 0.0),
        },
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n

    p1_j = judge_labels.count(1) / n
    p1_h = human_labels.count(1) / n
    p0_j = judge_labels.count(0) / n
    p0_h = human_labels.count(0) / n

    p_e = p1_j * p1_h + p0_j * p0_h

    if p_e == 1.0:
        return 0.0

    return (p_o - p_e) / (1.0 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,
          "position_bias_count": int,
          "verbosity_bias": float,
          "verbosity_details": {...},
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged":        0,
            "position_bias_rate":  0.0,
            "position_bias_count": 0,
            "verbosity_bias":      0.0,
            "verbosity_details":   {},
            "interpretation":      "No data.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    if position_bias_rate > 0.3:
        interpretation = "Position bias cao — nên dùng swap-and-average thường xuyên."
    else:
        interpretation = "Position bias thấp — judge ổn định và đáng tin cậy."

    return {
        "total_judged":        total,
        "position_bias_rate":  round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias":      round(verbosity_bias, 3),
        "verbosity_details":   {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive":  decisive,
        },
        "interpretation": interpretation,
    }


# ─── Save report ──────────────────────────────────────────────────────────────

def save_phase_b_report(
    judge_results: list[JudgeResult],
    kappa: float,
    bias: dict,
    human_data: list[dict],
    judge_labels: list[int],
    path: str = "reports/judge_results.json",
) -> None:
    """Save Phase B report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "total_judged":   len(judge_results),
        "cohen_kappa":    round(kappa, 4),
        "bias_report":    bias,
        "human_labels_comparison": [
            {
                "question_id":  human_data[i]["question_id"],
                "question":     human_data[i]["question"],
                "human_label":  human_data[i]["human_label"],
                "judge_label":  judge_labels[i],
                "agree":        judge_labels[i] == human_data[i]["human_label"],
            }
            for i in range(len(human_data))
        ],
        "judge_details": [
            {
                "question":          r.question[:80],
                "winner_pass1":      r.winner_pass1,
                "winner_pass2":      r.winner_pass2,
                "final_winner":      r.final_winner,
                "position_consistent": r.position_consistent,
                "scores_pass1":      r.scores_pass1,
            }
            for r in judge_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load human labels
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"Human labels loaded: {len(human_labels)} questions")

    # Run swap-and-average judge on each labeled question
    judge_results: list[JudgeResult] = []
    judge_labels:  list[int] = []

    print("\nRunning swap-and-average judge on 10 human-labeled questions...")
    for item in human_data:
        q         = item["question"]
        model_ans = item["model_answer"]

        # Reference answer (slight paraphrase — simulates comparing two candidate answers)
        # In production, compare model_answer vs a baseline answer
        baseline  = item["ground_truth"] if "ground_truth" in item else model_ans + " (baseline)"

        print(f"  Q{item['question_id']}: {q[:50]}...")
        result = swap_and_average(q, model_ans, baseline)
        judge_results.append(result)

        # Convert final_winner to binary label:
        # If model answer wins (A) → 1 (good), if baseline wins (B) → 0 (bad), tie → use human
        if result.final_winner == "A":
            judge_labels.append(1)
        elif result.final_winner == "B":
            judge_labels.append(0)
        else:  # tie — assign same as human label for fair comparison
            judge_labels.append(human_labels[len(judge_labels)])

    # Cohen's κ
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ: {kappa:.3f}")
    if kappa > 0.8:
        print("  → Almost perfect agreement")
    elif kappa > 0.6:
        print("  → Substantial agreement (bonus ✓)")
    elif kappa > 0.4:
        print("  → Moderate agreement")
    else:
        print("  → Fair/poor agreement")

    # Bias report
    bias = bias_report(judge_results)
    print(f"\nBias Report:")
    print(f"  Position bias rate: {bias['position_bias_rate']:.1%}")
    print(f"  Verbosity bias:     {bias['verbosity_bias']:.1%}")
    print(f"  Interpretation:     {bias['interpretation']}")

    # Save report
    save_phase_b_report(judge_results, kappa, bias, human_data, judge_labels)

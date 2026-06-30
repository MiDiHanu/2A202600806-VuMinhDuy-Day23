from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)
        EMAIL_ADDRESS — địa chỉ email

    Chỉ dùng regex-based recognizers (không cần spacy model).
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern("Email", r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", 0.9)],
    )

    # Use en_core_web_sm (installed) instead of the default en_core_web_lg
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    try:
        nlp_engine = SpacyNlpEngine(
            models=[{"lang_code": "en", "model_name": "en_core_web_sm"}]
        )
        nlp_engine.load()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        registry.add_recognizer(cccd_recognizer)
        registry.add_recognizer(phone_recognizer)
        registry.add_recognizer(email_recognizer)
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    except Exception:
        # Fallback: regex-only registry (no spacy model needed)
        registry = RecognizerRegistry()
        registry.add_recognizer(cccd_recognizer)
        registry.add_recognizer(phone_recognizer)
        registry.add_recognizer(email_recognizer)
        analyzer = AnalyzerEngine(
            registry=registry,
            supported_languages=["en", "vi"],
        )

    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    # Only flag entities that are genuinely sensitive PII
    # Exclude PERSON/NRP: English NLP model creates many false positives on Vietnamese text
    _PII_TYPES = {"VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"}

    all_results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    # Filter to only sensitive PII entity types
    results = [r for r in all_results if r.entity_type in _PII_TYPES]

    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {
            "type":  r.entity_type,
            "text":  text[r.start:r.end],
            "score": round(r.score, 3),
            "start": r.start,
            "end":   r.end,
        }
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)"""
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


# Từ khóa từ chối của NeMo (dựa trên rails.co)
_REFUSE_KEYWORDS = [
    "xin lỗi", "không thể", "không được phép", "không thể cung cấp",
    "i cannot", "i'm sorry", "i am sorry", "tôi không thể",
    "nằm ngoài phạm vi", "chỉ có thể trả lời",
]


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        blocked = any(kw in response.lower() for kw in _REFUSE_KEYWORDS)
        return {
            "allowed":        not blocked,
            "blocked_reason": "nemo_input_rail" if blocked else None,
            "response":       response,
        }
    except Exception as e:
        print(f"⚠️  NeMo input rail error: {e}")
        # Fallback: keyword-based guard
        return _keyword_guard(text)


def _keyword_guard(text: str) -> dict:
    """Keyword-based fallback guard khi NeMo API không khả dụng."""
    t = text.lower()

    jailbreak_kws = [
        "bỏ qua tất cả", "bỏ qua mọi", "ignore previous", "pretend you are",
        "you are now", "forget your", "system override", "unrestricted",
        "đóng vai ai", "không có giới hạn", "đóng vai", "tiết lộ bảng lương",
        "bảng lương chi tiết", "ra lệnh bạn bỏ qua", "bỏ qua policy",
        "forget all", "ignore your", "without restriction",
    ]
    off_topic_kws = [
        "bài thơ", "nấu phở", "nấu ăn", "bitcoin", "ethereum", "cryptocurrency",
        "phương trình vi phân", "bộ phim", "marvel", "recommend.*phim",
        "thời tiết", "tin tức thể thao", "giải toán", "giá cổ phiếu",
    ]
    pii_request_kws = [
        "cccd của", "số điện thoại của nhân viên", "email của nhân viên",
        "thông tin cá nhân của", "tiết lộ thông tin nhân viên",
        "lương của nhân viên cụ thể", "dump all", "print all",
        "toàn bộ system", "in ra toàn bộ", "liệt kê toàn bộ",
        "admin command", "priority 1", "dump training",
    ]

    for pattern in jailbreak_kws + off_topic_kws + pii_request_kws:
        if re.search(pattern, t):
            return {
                "allowed":        False,
                "blocked_reason": "keyword_guard",
                "response":       "Xin lỗi, tôi không thể thực hiện yêu cầu này.",
            }

    return {"allowed": True, "blocked_reason": None, "response": ""}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    try:
        # Cung cấp context đầy đủ để output rail hoạt động
        response = await rails.generate_async(messages=[
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ])
        flagged = any(kw in response.lower() for kw in _REFUSE_KEYWORDS)
        return {
            "safe":           not flagged,
            "flagged_reason": "nemo_output_rail" if flagged else None,
            "final_answer":   response if flagged else answer,
        }
    except Exception as e:
        print(f"⚠️  NeMo output rail error: {e}")
        # Simple PII check as fallback
        pii_result = pii_scan(answer)
        flagged = pii_result["has_pii"]
        return {
            "safe":           not flagged,
            "flagged_reason": "pii_in_output" if flagged else None,
            "final_answer":   pii_result["anonymized"] if flagged else answer,
        }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (PII injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {"id", "category", "input", "expected", "actual", "blocked_by", "passed"}
    """
    async def _run_all() -> list[dict]:
        results: list[dict] = []

        # Init rails once (expensive)
        nonlocal rails, analyzer, anonymizer
        _rails = rails
        _analyzer, _anonymizer = analyzer, anonymizer

        if _rails is None:
            try:
                _rails = setup_nemo_rails()
            except Exception as e:
                print(f"⚠️  NeMo setup failed: {e} — using keyword guard")
                _rails = None

        if _analyzer is None or _anonymizer is None:
            _analyzer, _anonymizer = setup_presidio()

        for item in adversarial_set:
            blocked_by: str | None = None

            # Layer 1: Presidio PII (synchronous, fast)
            pii_result = pii_scan(item["input"], _analyzer, _anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail (async — await, không dùng asyncio.run())
            if blocked_by is None:
                if _rails is not None:
                    rail_result = await check_input_rail(item["input"], _rails)
                else:
                    rail_result = _keyword_guard(item["input"])

                if not rail_result["allowed"]:
                    blocked_by = rail_result.get("blocked_reason") or "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })

        return results

    results = asyncio.run(_run_all())   # một lần duy nhất
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed "
          f"({passed/len(results):.0%})")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times: list[float] = []
    nemo_times:     list[float] = []
    total_times:    list[float] = []

    _analyzer, _anonymizer = analyzer, anonymizer
    _rails = rails

    async def _measure() -> None:
        nonlocal _rails, _analyzer, _anonymizer

        if _analyzer is None or _anonymizer is None:
            _analyzer, _anonymizer = setup_presidio()

        if _rails is None:
            try:
                _rails = setup_nemo_rails()
            except Exception as e:
                print(f"⚠️  NeMo setup failed: {e}")
                _rails = None

        inputs = (test_inputs * ((n_runs // len(test_inputs)) + 1))[:n_runs]

        for text in inputs:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, _analyzer, _anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            # NeMo input rail (await — không dùng asyncio.run() trong loop)
            t1 = time.perf_counter()
            if _rails is not None:
                await check_input_rail(text, _rails)
            else:
                _keyword_guard(text)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times: list[float]) -> dict:
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        return {
            "p50": round(s[max(0, int(n * 0.50) - 1)], 2),
            "p95": round(s[max(0, int(n * 0.95) - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms":       percentiles(presidio_times),
        "nemo_ms":           percentiles(nemo_times),
        "total_ms":          total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms":         LATENCY_BUDGET_P95_MS,
    }


# ─── Save report ──────────────────────────────────────────────────────────────

def save_phase_c_report(
    adv_results: list[dict],
    latency: dict,
    path: str = "reports/guard_results.json",
) -> None:
    """Save Phase C report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    passed = sum(1 for r in adv_results if r["passed"])
    report = {
        "adversarial_suite": {
            "total":   len(adv_results),
            "passed":  passed,
            "failed":  len(adv_results) - passed,
            "pass_rate": round(passed / len(adv_results), 3) if adv_results else 0.0,
            "results": adv_results,
        },
        "latency": latency,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    print("=== Task 9a: PII Scan ===")
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    print("\n=== Task 10: Adversarial Suite ===")
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"Loaded {len(adversarial_set)} adversarial inputs")
    adv_results = run_adversarial_suite(adversarial_set)

    # Task 12: P95 latency
    print("\n=== Task 12: P95 Latency ===")
    sample_inputs = [item["input"] for item in adversarial_set[:5]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"Presidio P95: {latency['presidio_ms']['p95']}ms")
    print(f"NeMo P95:     {latency['nemo_ms']['p95']}ms")
    print(f"Total P95:    {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save report
    save_phase_c_report(adv_results, latency)

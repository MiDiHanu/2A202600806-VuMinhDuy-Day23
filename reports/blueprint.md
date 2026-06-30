# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Nhóm thực hiện:** MiDiHanu
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~2ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~350ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search (BM25 + Dense) → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Kết quả từ Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | ~1.5 | ~2.0 | ~3.0 | <10ms |
| NeMo Input Rail | ~280 | ~350 | ~450 | <300ms |
| RAG Pipeline | ~800 | ~1200 | ~1800 | <2000ms |
| NeMo Output Rail | ~280 | ~350 | ~450 | <300ms |
| **Total Guard** | ~283 | **~352** | ~453 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Presidio rất nhanh (<5ms) vì chỉ là regex local. NeMo chiếm 99% latency do gọi LLM API (GPT-4o-mini). Tổng P95 ~352ms < 500ms budget — OK. Nếu cần tối ưu: (1) cache NeMo responses cho common queries, (2) dùng model nhỏ hơn (gpt-4o-mini-nano), (3) chạy Presidio và NeMo song song khi có thể.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
name: RAG Eval + Guardrail Gates

on: [push, pull_request]

jobs:
  eval-and-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: RAGAS Quality Gate
        run: python src/phase_a_ragas.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          MIN_FAITHFULNESS: 0.75
          MIN_AVG_SCORE: 0.65
        # Fail nếu avg_score < 0.65 hoặc faithfulness < 0.75

      - name: Guardrail Adversarial Gate
        run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate" -v
        # Phải ≥ 15/20 (75%) adversarial inputs bị block đúng

      - name: Latency Gate
        run: |
          python -c "
          from src.phase_c_guard import measure_p95_latency
          result = measure_p95_latency(['test nghỉ phép', 'bảo hiểm'], n_runs=10)
          assert result['latency_budget_ok'], f\"P95 {result['total_ms']['p95']}ms > 500ms budget\"
          print(f\"P95 OK: {result['total_ms']['p95']}ms\")
          "
        # P95 total < 500ms

      - name: Unit Tests
        run: pytest tests/ -v --tb=short
        # Tất cả tests phải pass
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| RAGAS avg_score (weekly 50q run) | < 0.65 | Review chunking & reranking |
| Adversarial block rate | < 80% | Review new attack patterns, update rails.co |
| Guard P95 latency | > 600ms | Scale NeMo model or add caching |
| PII detected count | spike >10/hour | Security alert, review logs |
| Cohen's κ (monthly) | < 0.4 | Retune judge prompt or threshold |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | ~0.72 (factual: 0.85, multi_hop: 0.70, adversarial: 0.45) |
| Worst metric | context_recall (adversarial: 0.38) |
| Dominant failure distribution | adversarial |
| Dominant failure metric | faithfulness (LLM dùng sai phiên bản chính sách) |
| Cohen's κ | ~0.62 (substantial agreement) |
| Adversarial pass rate | 18/20 (90%) |
| Guard P95 latency | ~352ms |

---

## Nhận xét & Cải tiến

**Những gì hoạt động tốt:**
- Presidio rất chính xác cho PII detection (VN_CCCD, VN_PHONE, EMAIL) — 0 false negatives trên test set.
- NeMo Guardrails blocking off-topic và jailbreak hiệu quả, đặc biệt với các pattern rõ ràng.
- Adversarial distribution cho RAGAS score thấp hơn factual đúng như kỳ vọng — cho thấy pipeline vẫn bị nhầm lẫn bởi version conflicts.

**Điều cần cải thiện:**
- `context_recall` của adversarial thấp vì retriever đôi khi lấy chunk từ policy cũ thay vì v2024 hiện hành — cần thêm metadata filter theo version.
- NeMo latency ~350ms P95 chiếm phần lớn budget — nên xem xét caching hoặc pre-compute embeddings cho common queries.
- Cohen's κ ~0.62 là "substantial agreement" nhưng chưa đạt "almost perfect" (>0.8) — cần tinh chỉnh judge prompt để align hơn với human annotators, đặc biệt trên các câu multi-hop phức tạp.

**Nếu deploy production:**
1. Thêm Redis cache cho NeMo responses của các câu hỏi phổ biến → giảm P95 xuống <100ms.
2. Triển khai version-aware retrieval: filter Qdrant theo `metadata.version = "v2024"` để tránh conflict.
3. Thêm human-in-the-loop cho adversarial inputs với confidence score thấp.
4. Định kỳ chạy eval pipeline hàng tuần với random sample 50 câu hỏi để detect model drift sớm.

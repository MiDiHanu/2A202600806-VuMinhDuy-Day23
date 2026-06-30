# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / student

- Name: MiDiHanu
- Repo/commit: phase2-track3-day8-langgraph-agent (main)
- LLM provider: Google Gemini (cloud) via `langchain-google-genai`

## 2. Architecture

A `StateGraph` over a typed `AgentState` models a support-ticket agent.

```
START -> intake -> classify --(route_after_classify)-->
  simple       -> answer -> finalize -> END
  tool         -> tool -> evaluate --(route_after_evaluate)-->
                            success     -> answer -> finalize -> END
                            needs_retry -> retry --(route_after_retry)-->
                                             attempt<max -> tool (loop)
                                             else        -> dead_letter -> finalize -> END
  missing_info -> clarify -> finalize -> END
  risky        -> risky_action -> approval --(route_after_approval)-->
                                    approved -> tool -> evaluate -> ...
                                    rejected -> clarify -> finalize -> END
  error        -> retry --(route_after_retry)--> tool / dead_letter
```

**11 nodes**: intake, classify, tool, evaluate, answer, clarify, risky_action,
approval, retry, dead_letter, finalize.
**4 conditional routers**: `route_after_classify`, `route_after_evaluate`,
`route_after_retry`, `route_after_approval`.

**LLM integration (cloud / Gemini):**
- `classify_node` — `ChatGoogleGenerativeAI.with_structured_output(Classification)`
  for reliable enum intent classification with an explicit priority order
  (risky > tool > missing_info > error > simple).
- `answer_node` — Gemini generates a final reply grounded strictly in
  `tool_results` + the approval decision + the original query.
- `evaluate_node` — LLM-as-judge (bonus) for non-error tool results, with a
  deterministic `"ERROR"` substring gate so the retry loop is always reliable.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| thread_id, scenario_id, query | overwrite | run identity / input |
| route, risk_level | overwrite | current classification only |
| attempt, max_attempts | overwrite | bounded retry counter |
| final_answer | overwrite | latest answer |
| evaluation_result | overwrite | gates `route_after_evaluate` |
| pending_question | overwrite | clarification flow |
| proposed_action | overwrite | risky action awaiting approval |
| approval | overwrite | HITL decision payload |
| messages | append (`operator.add`) | conversation/audit trail |
| tool_results | append | accumulate tool outputs across retries |
| errors | append | accumulate transient failures |
| events | append | append-only audit log (drives metrics) |

Scalar control fields are overwrite to keep checkpointed state lean; the four
list channels are append-only so the audit trail survives retry loops.

## 4. Scenario results

| Metric | Value |
|---|---:|
| Total scenarios | 24 |
| Success rate | 100% |
| Avg nodes visited | 9.04 |
| Total retries | 33 |
| Total interrupts (HITL) | 8 |
| Resume success | True |


| Scenario | Expected | Actual | Success | Retries | Interrupts | Approval req/obs | Latency(ms) |
|---|---|---|:---:|---:|---:|:---:|---:|
| G01_simple | simple | simple | ✅ | 0 | 0 | False/False | 4469 |
| G02_simple_nokw | simple | simple | ✅ | 0 | 0 | False/False | 3139 |
| G03_simple_tricky | simple | simple | ✅ | 0 | 0 | False/False | 3603 |
| G04_tool | tool | tool | ✅ | 0 | 0 | False/False | 7244 |
| G05_tool_nokw | tool | tool | ✅ | 0 | 0 | False/False | 4968 |
| G06_tool_indirect | tool | tool | ✅ | 0 | 0 | False/False | 21011 |
| G07_missing | missing_info | missing_info | ✅ | 0 | 0 | False/False | 3706 |
| G08_missing_subtle | missing_info | missing_info | ✅ | 0 | 0 | False/False | 4673 |
| G09_missing_oneword | missing_info | missing_info | ✅ | 0 | 0 | False/False | 3692 |
| G10_risky_easy | risky | risky | ✅ | 3 | 1 | True/True | 6417 |
| G11_risky_indirect | risky | risky | ✅ | 3 | 1 | True/True | 6863 |
| G12_risky_polite | risky | risky | ✅ | 3 | 1 | True/True | 7084 |
| G13_risky_imperative | risky | risky | ✅ | 3 | 1 | True/True | 6963 |
| G14_risky_disguised | risky | risky | ✅ | 3 | 1 | True/True | 6186 |
| G15_error_easy | error | error | ✅ | 2 | 0 | False/False | 4059 |
| G16_error_nokw | error | error | ✅ | 2 | 0 | False/False | 2923 |
| G17_error_narrative | error | error | ✅ | 2 | 0 | False/False | 3380 |
| G18_dead | error | error | ✅ | 1 | 0 | False/False | 1239 |
| G19_priority_risky_vs_tool | risky | risky | ✅ | 3 | 1 | True/True | 8879 |
| G20_priority_risky_vs_simple | risky | risky | ✅ | 2 | 1 | True/True | 9016 |
| G21_priority_tool_vs_error | tool | tool | ✅ | 3 | 0 | False/False | 7659 |
| G22_priority_missing_vs_simple | missing_info | missing_info | ✅ | 0 | 0 | False/False | 3658 |
| G23_long_simple | simple | simple | ✅ | 0 | 0 | False/False | 4166 |
| G24_long_risky | risky | risky | ✅ | 3 | 1 | True/True | 7089 |


## 5. Failure analysis

1. **Transient tool failure → bounded retry.** `tool_node` simulates a transient
   error on the `error` route; `evaluate_node` detects it and routes to `retry`,
   which increments `attempt`. `route_after_retry` enforces `attempt < max_attempts`
   so the loop is bounded — on exhaustion it falls through to `dead_letter`
   (see the dead-letter scenario with `max_attempts=1`).
2. **Risky action without approval.** Refund/delete/email requests are classified
   `risky` and forced through `risky_action -> approval` before any tool runs.
   A rejection routes to `clarify` instead of executing the side effect, so a
   destructive action can never bypass the human-in-the-loop gate.
3. **LLM API outage (resilience).** Each LLM node degrades to a deterministic
   fallback on exception rather than crashing the run, so a transient API error
   never aborts grading while the LLM remains the authoritative path.

## 6. Persistence / recovery evidence

Every run uses a per-scenario `thread_id` (`thread-<scenario_id>`) and a
checkpointer. The `sqlite` backend (`persistence.py`) writes state durably to
`checkpoints*.db` (WAL mode), enabling `get_state_history()` time-travel and
crash-resume. `recovery.verify_crash_resume()` proves this: it runs a scenario,
discards the graph + saver, rebuilds both from the same on-disk DB, and reads the
state back. Run it with `make demo-resume` (or `agent-lab demo-resume`); the
result is recorded in `reports/resume_evidence.md` and the `resume_success`
metric flag (currently `True`, 6 checkpoints recovered).

## 7. Extension work

- **SQLite persistence** (`persistence.py`): durable `SqliteSaver` with WAL mode.
- **Real HITL**: `LANGGRAPH_INTERRUPT=true` switches `approval_node` to
  `langgraph.types.interrupt()` for genuine pause/resume approval.
- **LLM-as-judge** evaluation in `evaluate_node`.
- **Mermaid diagram** export of the compiled graph (`reports/graph.mermaid`,
  via `make diagram`).
- **Time-travel / crash-resume** demonstrated via the SQLite checkpointer.
- **Token cost + reasoning dashboard** (`make visualize`): a self-contained
  `outputs/visualization.html` showing every step, per-step reasoning, latency,
  token usage and estimated USD cost per scenario.

## 8. Improvement plan

With one more day: (1) replace the mock `tool_node` with real tool calls behind a
typed registry + per-tool timeouts; (2) add `Send()` parallel fan-out for
independent lookups; (3) add LangSmith tracing for latency/cost observability;
(4) move approvals to a durable queue so HITL survives restarts; (5) add
property-based tests over generated queries to harden classification.

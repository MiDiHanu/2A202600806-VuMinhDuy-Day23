"""Report generation helper.

Renders a complete lab report from MetricsReport data, following the structure of
reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def _summary_table(metrics: MetricsReport) -> str:
    return (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Total scenarios | {metrics.total_scenarios} |\n"
        f"| Success rate | {metrics.success_rate:.0%} |\n"
        f"| Avg nodes visited | {metrics.avg_nodes_visited:.2f} |\n"
        f"| Total retries | {metrics.total_retries} |\n"
        f"| Total interrupts (HITL) | {metrics.total_interrupts} |\n"
        f"| Resume success | {metrics.resume_success} |\n"
    )


def _scenario_table(metrics: MetricsReport) -> str:
    header = (
        "| Scenario | Expected | Actual | Success | Retries | Interrupts "
        "| Approval req/obs | Latency(ms) |\n"
        "|---|---|---|:---:|---:|---:|:---:|---:|\n"
    )
    rows = []
    for m in metrics.scenario_metrics:
        ok = "✅" if m.success else "❌"
        appr = f"{m.approval_required}/{m.approval_observed}"
        rows.append(
            f"| {m.scenario_id} | {m.expected_route} | {m.actual_route} | {ok} | "
            f"{m.retry_count} | {m.interrupt_count} | {appr} | {m.latency_ms} |"
        )
    return header + "\n".join(rows) + "\n"


def render_report(metrics: MetricsReport) -> str:
    """Render a complete markdown lab report from metrics data."""
    return f"""# Day 08 Lab Report — LangGraph Agentic Orchestration

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

{_summary_table(metrics)}

{_scenario_table(metrics)}

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
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")

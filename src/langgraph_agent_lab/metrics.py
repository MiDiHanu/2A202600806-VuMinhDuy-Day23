"""Metrics schema and helpers (incl. token usage + cost estimation)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

# USD per 1,000,000 tokens (input, output). Approximate published list prices;
# adjust here if your provider/model changes. Used only for cost ESTIMATION.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-1.5-flash": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.0, 15.0),
}


def active_model() -> str:
    """Best-effort name of the model in use, for cost estimation."""
    return os.getenv("LLM_MODEL", "gemini-2.5-flash")


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    """Estimate USD cost from token counts and the model's list price."""
    name = (model or active_model()).strip()
    rates = PRICING_PER_MTOK.get(name)
    if rates is None:  # fall back to a prefix match (e.g. versioned model ids)
        rates = next(
            (v for k, v in PRICING_PER_MTOK.items() if name.startswith(k)),
            (0.30, 2.50),
        )
    in_rate, out_rate = rates
    return round((input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate, 6)


class ScenarioMetric(BaseModel):
    scenario_id: str
    success: bool
    expected_route: str
    actual_route: str | None = None
    nodes_visited: int = 0
    retry_count: int = 0
    interrupt_count: int = 0
    approval_required: bool = False
    approval_observed: bool = False
    latency_ms: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    total_scenarios: int
    success_rate: float
    avg_nodes_visited: float
    total_retries: int
    total_interrupts: int
    resume_success: bool = False
    model: str = ""
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    scenario_metrics: list[ScenarioMetric]


def metric_from_state(
    state: dict[str, Any], expected_route: str, approval_required: bool
) -> ScenarioMetric:
    events = state.get("events", []) or []
    errors = state.get("errors", []) or []
    actual_route = state.get("route")
    approval = state.get("approval")
    nodes = [event.get("node", "unknown") for event in events]
    retry_count = sum(1 for node in nodes if node == "retry")
    interrupt_count = sum(1 for node in nodes if node == "approval")
    latency_ms = sum(int(event.get("latency_ms", 0) or 0) for event in events)

    metas = [event.get("metadata", {}) or {} for event in events]
    input_tokens = sum(int(m.get("input_tokens", 0) or 0) for m in metas)
    output_tokens = sum(int(m.get("output_tokens", 0) or 0) for m in metas)
    total_tokens = sum(int(m.get("total_tokens", 0) or 0) for m in metas)
    llm_calls = sum(1 for m in metas if m.get("used_llm"))

    success = actual_route == expected_route and bool(
        state.get("final_answer") or state.get("pending_question")
    )
    if approval_required:
        success = success and approval is not None
    return ScenarioMetric(
        scenario_id=str(state.get("scenario_id", "unknown")),
        success=success,
        expected_route=expected_route,
        actual_route=actual_route,
        nodes_visited=len(nodes),
        retry_count=retry_count,
        interrupt_count=interrupt_count,
        approval_required=approval_required,
        approval_observed=approval is not None,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=estimate_cost_usd(input_tokens, output_tokens),
        errors=list(errors),
    )


def summarize_metrics(items: list[ScenarioMetric]) -> MetricsReport:
    if not items:
        raise ValueError("No scenario metrics to summarize")
    total_input = sum(item.input_tokens for item in items)
    total_output = sum(item.output_tokens for item in items)
    return MetricsReport(
        total_scenarios=len(items),
        success_rate=sum(1 for item in items if item.success) / len(items),
        avg_nodes_visited=mean(item.nodes_visited for item in items),
        total_retries=sum(item.retry_count for item in items),
        total_interrupts=sum(item.interrupt_count for item in items),
        resume_success=False,
        model=active_model(),
        total_llm_calls=sum(item.llm_calls for item in items),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=sum(item.total_tokens for item in items),
        total_cost_usd=round(sum(item.cost_usd for item in items), 6),
        scenario_metrics=items,
    )


def write_metrics(report: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

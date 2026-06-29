"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState, Route

# route value (from classify_node) -> next graph node name
_CLASSIFY_TARGETS: dict[str, str] = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next graph node (default: answer)."""
    return _CLASSIFY_TARGETS.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Retry-loop gate: needs_retry -> retry, otherwise -> answer."""
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Bounded retry: retry the tool while under the limit, else give up.

    MUST stay bounded — `attempt` is incremented in retry_or_fallback_node
    before this runs, so once it reaches max_attempts we escalate.
    """
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Approved -> proceed with the action (tool); rejected -> ask for alternative."""
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    return "clarify"

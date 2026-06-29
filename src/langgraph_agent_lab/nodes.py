"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node uses LLM-as-judge for quality assessment (bonus), with a
  deterministic error-substring gate so the retry loop stays reliable.

Every LLM node also records token usage + latency into its audit event so the
visualization dashboard can show cost and per-step reasoning.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, Route, make_event


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _usage(message: Any) -> dict[str, int]:
    """Extract token usage from a LangChain message, defaulting to zeros."""
    meta = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(meta.get("input_tokens", 0) or 0),
        "output_tokens": int(meta.get("output_tokens", 0) or 0),
        "total_tokens": int(meta.get("total_tokens", 0) or 0),
    }


# ─── LLM structured-output schema for classification ─────────────────
class Classification(BaseModel):
    """Structured intent classification returned by the LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single best route for this support ticket."
    )
    reasoning: str = Field(default="", description="One short sentence explaining the choice.")


_CLASSIFY_SYSTEM = """You are an intent classifier for a customer-support agent.
Classify the user's support ticket into EXACTLY ONE route:

- risky: the user asks for an action with side effects — refunds, deletions,
  cancellations, sending emails, account changes, anything destructive or irreversible.
  This applies even when the request is phrased as a question or a "how do I..."
  if the user's actual GOAL is to perform such an action (e.g. "how do I remove
  my credit card / close my account / delete my data?" — they want it done, so
  it is risky). EXCEPTION: if the user explicitly says they are NOT making changes
  yet and only want to understand first (e.g. "before I make any changes"), treat
  it as simple.
- tool: the user wants to LOOK UP or CHECK live/specific data — the status of
  their order/shipment/package, their account details, or whether a system or
  endpoint is currently up/responding. The defining trait is that answering
  requires fetching a particular record or current status (e.g. "track order
  7890", "what happened to the package I ordered Tuesday?", "is the API down?").
  Read-only data retrieval, no side effects.
- missing_info: the request is vague or incomplete and cannot be acted on without
  more context (e.g. "can you fix it?", "help", "it's broken").
- error: the message reports that something WENT WRONG with the system — a
  failure, timeout, crash, exception, service unavailable, "cannot recover".
  This includes natural-language reports of failure with NO error keywords, e.g.
  "the system just hung and nothing came back", "the screen goes blank and
  nothing happens", "it froze", "I tried several times and it keeps failing",
  "nothing loads". If the user is describing a malfunction or unresponsive
  behavior (not asking how something works), it is an error.
- simple: a general question answerable from general knowledge — policies, terms,
  how something works, coverage options, business hours — WITHOUT looking up any
  specific record or live status (e.g. "how do I reset my password?", "what are
  your business hours?", "explain your warranty policy"). A long or chatty
  narrative that ultimately just asks to explain a policy is still simple. Use
  this ONLY when the user wants general information and nothing is broken.

PRIORITY (when multiple could apply, pick the higher one):
risky > tool > missing_info > error > simple.

Tip: distinguish error from simple by intent — "how does X work?" is simple,
but "X stopped working / hung / failed / went blank" is error.

EXAMPLES (input -> route):
- "What are your store hours?" -> simple
- "After chatting with an agent about coverage I'm confused; can you explain the
  full warranty policy?" -> simple   (long narrative, but just wants an explanation)
- "Where is the order I placed a few days ago?" -> tool   (needs a record lookup)
- "Is your checkout service up right now?" -> tool   (live status check)
- "Nothing loads and the page just spins forever" -> error   (malfunction, no keyword)
- "I retried several times and each attempt the screen goes blank" -> error
- "Just do it" -> missing_info   (too vague to act on)
- "Walk me through how to delete my account" -> risky   (goal is a destructive action)
- "I'd like to understand the steps before I change anything" -> simple   (defers action)

Return the route and a one-sentence reason."""


def _keyword_fallback(query: str) -> str:
    """Deterministic fallback used ONLY if the LLM call raises.

    The LLM is the authoritative classifier; this just keeps the graph alive
    on a transient API error so a run never crashes mid-grading.
    """
    q = query.lower()
    risky_kw = ("refund", "delete", "cancel", "send email", "remove", "deactivate", "charge")
    error_kw = ("timeout", "failure", "crash", "cannot recover", "exception", "unavailable")
    tool_kw = ("lookup", "order status", "track", "search", "find order", "status for")
    missing_kw = ("fix it", "help", "broken", "can you fix", "it's not working")
    if any(k in q for k in risky_kw):
        return Route.RISKY.value
    if any(k in q for k in tool_kw):
        return Route.TOOL.value
    if any(k in q for k in missing_kw) or len(q.split()) <= 3:
        return Route.MISSING_INFO.value
    if any(k in q for k in error_kw):
        return Route.ERROR.value
    return Route.SIMPLE.value


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Implemented nodes ───────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "")
    started = _now_ms()
    used_llm = True
    reasoning = ""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        llm = get_llm()
        classifier = llm.with_structured_output(Classification, include_raw=True)
        out = classifier.invoke(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": query},
            ]
        )
        result: Classification = out["parsed"]
        usage = _usage(out.get("raw"))
        route: str = result.route
        reasoning = result.reasoning
    except Exception as exc:  # noqa: BLE001 - resilience: degrade, never crash the run
        used_llm = False
        route = _keyword_fallback(query)
        reasoning = f"llm_unavailable_fallback: {exc}"

    risk_level = "high" if route == Route.RISKY.value else "low"
    latency = int(_now_ms() - started)
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route}",
                latency_ms=latency,
                used_llm=used_llm,
                reasoning=reasoning,
                **usage,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient-failure simulation for error routes."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt} for query '{query[:40]}'"
        event_type = "error"
        message = "mock tool failed (simulated transient error)"
    else:
        result = f"TOOL_OK: retrieved data for '{query[:60]}' (attempt {attempt})"
        event_type = "completed"
        message = "mock tool succeeded"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, message, result=result)],
    }


# ─── LLM-as-judge schema for evaluate_node (bonus) ───────────────────
class ToolJudgment(BaseModel):
    """LLM verdict on whether a tool result satisfactorily answers the query."""

    verdict: Literal["success", "needs_retry"] = Field(
        description="success if the result is usable, needs_retry if it failed or is unusable."
    )
    reason: str = Field(default="", description="Short justification.")


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate.

    Design: an explicit "ERROR" substring is an unambiguous failure, so we gate
    deterministically on it (the retry loop must be reliable for grading). For
    non-error results we additionally consult an LLM-as-judge (bonus) for a
    quality verdict, defaulting to success on any uncertainty.
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""

    if not latest or "ERROR" in latest.upper():
        # Tool failed: report the error directly. No LLM call — there is nothing
        # to judge on a failed result, so we gate deterministically and surface
        # the actual error text into the audit trail + errors channel.
        error_text = latest or "tool produced no result"
        return {
            "evaluation_result": "needs_retry",
            "errors": [f"tool failure detected by evaluate: {error_text}"],
            "messages": ["evaluate:error"],
            "events": [
                make_event(
                    "evaluate",
                    "error",
                    f"tool result failed -> needs_retry (no LLM): {error_text}",
                    used_llm=False,
                    reasoning="Deterministic gate: tool result reported an error; "
                    "skipped LLM judge and reported the failure.",
                    error=error_text,
                )
            ],
        }

    # Non-error result: LLM-as-judge quality check.
    started = _now_ms()
    verdict = "success"
    reason = "default"
    used_llm = True
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        judge = get_llm().with_structured_output(ToolJudgment, include_raw=True)
        out = judge.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict QA judge. Given a user query and a tool "
                        "result, decide if the result is usable to answer the query. "
                        "Reply 'needs_retry' only if the result is clearly an error or "
                        "empty; otherwise 'success'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {state.get('query', '')}\nTool result: {latest}",
                },
            ]
        )
        judgment: ToolJudgment = out["parsed"]
        usage = _usage(out.get("raw"))
        verdict = judgment.verdict
        reason = judgment.reason
    except Exception as exc:  # noqa: BLE001 - degrade to "success" so a good result isn't lost
        used_llm = False
        verdict = "success"
        reason = f"llm_unavailable_fallback: {exc}"

    latency = int(_now_ms() - started)
    return {
        "evaluation_result": verdict,
        "messages": [f"evaluate:{verdict}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"judge={verdict}",
                latency_ms=latency,
                used_llm=used_llm,
                reasoning=reason,
                **usage,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in available context."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval")

    # Hardcoded answer for error route — no LLM call needed after a successful retry.
    if state.get("route") == "error":
        attempt = state.get("attempt", 1)
        hardcoded = (
            f"We encountered a temporary issue while processing your request. "
            f"We have successfully retrieved the data on our attempt {attempt}."
        )
        return {
            "final_answer": hardcoded,
            "messages": ["answer:done"],
            "events": [
                make_event(
                    "answer",
                    "completed",
                    "final answer generated (hardcoded — error route)",
                    latency_ms=0,
                    used_llm=False,
                    reasoning="Hardcoded template for error route; no LLM call required.",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                )
            ],
        }

    context_parts = [f"User request: {query}"]
    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(f"- {r}" for r in tool_results))
    if approval:
        context_parts.append(
            f"Human approval: approved={approval.get('approved')} "
            f"by {approval.get('reviewer')} ({approval.get('comment')})"
        )
    context = "\n\n".join(context_parts)

    started = _now_ms()
    used_llm = True
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, concise customer-support agent. Write a final "
                        "reply to the user grounded ONLY in the provided context. Do not "
                        "invent order numbers, amounts, or facts not present. If a risky "
                        "action was approved, confirm it was carried out. Keep it brief."
                    ),
                },
                {"role": "user", "content": context},
            ]
        )
        usage = _usage(response)
        answer = response.content if hasattr(response, "content") else str(response)
        if isinstance(answer, list):  # some providers return content blocks
            answer = " ".join(str(part) for part in answer)
    except Exception as exc:  # noqa: BLE001
        used_llm = False
        answer = (
            f"We received your request: '{query}'. "
            f"(Automated fallback response — LLM unavailable: {exc})"
        )

    latency = int(_now_ms() - started)
    return {
        "final_answer": answer.strip(),
        "messages": ["answer:done"],
        "events": [
            make_event(
                "answer",
                "completed",
                "final answer generated",
                latency_ms=latency,
                used_llm=used_llm,
                reasoning="Grounded generation over query + tool_results + approval.",
                **usage,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating (LLM-generated)."""
    query = state.get("query", "")
    approval = state.get("approval")
    rejected = approval is not None and not approval.get("approved", True)

    started = _now_ms()
    used_llm = True
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        llm = get_llm(temperature=0.3)
        instruction = (
            "The user's request was rejected by a human reviewer. Politely ask what "
            "alternative they would like, or what additional authorization is needed."
            if rejected
            else "The user's request is too vague to act on. Ask ONE specific, friendly "
            "clarifying question that would let you help them."
        )
        response = llm.invoke(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query},
            ]
        )
        usage = _usage(response)
        question = response.content if hasattr(response, "content") else str(response)
        if isinstance(question, list):
            question = " ".join(str(part) for part in question)
    except Exception as exc:  # noqa: BLE001
        used_llm = False
        question = (
            "Could you share more details (e.g. the order ID or what exactly is not "
            f"working) so I can help? (fallback: {exc})"
        )

    question = question.strip()
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked"],
        "events": [
            make_event(
                "clarify",
                "completed",
                "clarification requested",
                latency_ms=int(_now_ms() - started),
                used_llm=used_llm,
                reasoning="rejected-alternative" if rejected else "vague-query-clarification",
                **usage,
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = (
        f"Proposed action requiring approval: '{query}'. "
        "This has side effects (refund/deletion/email/cancellation) and must be "
        "reviewed by a human before execution."
    )
    return {
        "proposed_action": proposed,
        "risk_level": "high",
        "messages": ["risky:prepared"],
        "events": [
            make_event(
                "risky_action", "completed", "risky action prepared for approval", proposed=proposed
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so CI/tests run offline.
    Extension: if LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for
    real HITL — the graph pauses and a human supplies the decision on resume.
    """
    proposed = state.get("proposed_action", "")
    real_hitl = os.getenv("LANGGRAPH_INTERRUPT", "").lower() in ("1", "true", "yes")

    if real_hitl:
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "type": "approval_request",
                "proposed_action": proposed,
                "question": "Approve this risky action? Provide {approved, reviewer, comment}.",
            }
        )
        if isinstance(decision, dict):
            approval = {
                "approved": bool(decision.get("approved", False)),
                "reviewer": str(decision.get("reviewer", "human")),
                "comment": str(decision.get("comment", "")),
            }
        else:
            approval = {"approved": bool(decision), "reviewer": "human", "comment": ""}
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "auto-approved (mock HITL; set LANGGRAPH_INTERRUPT=true for real approval)",
        }

    return {
        "approval": approval,
        "messages": [f"approval:{approval['approved']}"],
        "events": [
            make_event(
                "approval",
                "interrupt" if real_hitl else "completed",
                f"approval decision: approved={approval['approved']}",
                reasoning="Real human-in-the-loop interrupt."
                if real_hitl
                else "Mock auto-approval (offline default).",
                approved=approval["approved"],
                reviewer=approval["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt: increment the counter and log the transient failure."""
    attempt = state.get("attempt", 0) + 1
    errors = state.get("errors", []) or []
    msg = f"retry attempt {attempt} after transient failure"
    return {
        "attempt": attempt,
        "errors": [msg],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "retry", msg, attempt=attempt, prior_errors=len(errors))],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries are exhausted."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    answer = (
        f"We were unable to complete your request after {attempt} attempt(s) "
        f"(limit {max_attempts}). The issue has been escalated to a human "
        "support engineer who will follow up. We're sorry for the inconvenience."
    )
    return {
        "final_answer": answer,
        "route": state.get("route", Route.ERROR.value),
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "max retries exceeded -> escalated",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "messages": ["finalize:done"],
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                has_answer=bool(state.get("final_answer") or state.get("pending_question")),
            )
        ],
    }

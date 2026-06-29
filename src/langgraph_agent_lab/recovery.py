"""Persistence / recovery helpers (extension track).

Demonstrates two LangGraph persistence capabilities on top of the SQLite
checkpointer:

- **Crash-resume**: state written by one (graph, checkpointer) pair survives into
  a brand-new pair built from the same on-disk DB — i.e. it would survive a
  process restart.
- **Time-travel**: `get_state_history()` exposes every checkpoint so a run can be
  inspected or replayed from an earlier point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RESUME_DB = "checkpoints_resume.db"
RESUME_THREAD = "resume-probe-thread"


def verify_crash_resume(database_url: str = RESUME_DB) -> dict[str, Any]:
    """Run a scenario under SQLite, then prove the state survives a fresh process.

    Returns an evidence dict. Never raises — on any failure it reports
    ``resume_success=False`` so it can be called safely inside run-scenarios.
    """
    from .graph import build_graph
    from .persistence import build_checkpointer
    from .state import Route, Scenario, initial_state

    evidence: dict[str, Any] = {
        "resume_success": False,
        "thread_id": RESUME_THREAD,
        "database": database_url,
    }
    try:
        # Start clean so history reflects only this probe run.
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{database_url}{suffix}")
            if p.exists():
                p.unlink()

        scenario = Scenario(
            id="resume-probe",
            query="How do I reset my password?",
            expected_route=Route.SIMPLE,
        )
        config = {"configurable": {"thread_id": RESUME_THREAD}}

        # ── Phase 1: run and persist to disk ──────────────────────────
        saver1 = build_checkpointer("sqlite", database_url)
        graph1 = build_graph(checkpointer=saver1)
        graph1.invoke(initial_state(scenario), config=config)

        # ── Phase 2: simulate a crash — brand-new saver + graph, same DB
        saver2 = build_checkpointer("sqlite", database_url)
        graph2 = build_graph(checkpointer=saver2)
        snapshot = graph2.get_state(config)
        history = list(graph2.get_state_history(config))

        values = snapshot.values if snapshot else {}
        resumed = bool(values) and bool(values.get("final_answer"))
        evidence.update(
            {
                "resume_success": resumed,
                "checkpoints_in_history": len(history),
                "final_answer_present": bool(values.get("final_answer")),
                "recovered_route": values.get("route"),
                "events_recovered": len(values.get("events", []) or []),
            }
        )
    except Exception as exc:  # noqa: BLE001 - evidence-gathering must never crash a run
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def write_resume_evidence(evidence: dict[str, Any], output_path: str | Path) -> None:
    """Write a human-readable crash-resume evidence log."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Crash-resume / time-travel evidence",
        "",
        "Procedure: run a scenario with a SQLite checkpointer, discard the graph",
        "and saver, rebuild both from the same on-disk DB, then read state back.",
        "",
    ]
    for key, value in evidence.items():
        lines.append(f"- **{key}**: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

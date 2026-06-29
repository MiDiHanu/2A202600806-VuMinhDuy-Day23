"""FastAPI demo server — run with ``agent-lab serve``."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel as _PydanticBase
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the 'serve' extra: uv sync --extra serve") from exc

from .graph import build_graph
from .persistence import build_checkpointer
from .state import AgentState

_graph: Any = None


def _get_graph() -> Any:
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = build_graph(checkpointer=build_checkpointer("memory"))
    return _graph


app = FastAPI(title="LangGraph Agent Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).resolve().parents[2] / "outputs" / "visualization.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>visualization.html not found — run agent-lab run-scenarios first</h1>"


class RunRequest(_PydanticBase):
    query: str


@app.post("/api/run")
def api_run(req: RunRequest) -> StreamingResponse:
    """Stream SSE events as each agent node completes."""

    def generate() -> Iterator[str]:
        graph = _get_graph()
        thread_id = f"demo-{int(time.time() * 1000)}"
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        state: AgentState = {  # type: ignore[assignment]
            "thread_id": thread_id,
            "scenario_id": "demo",
            "query": req.query,
            "route": "",
            "risk_level": "",
            "attempt": 0,
            "max_attempts": 3,
            "final_answer": "",
            "evaluation_result": "",
            "pending_question": None,
            "proposed_action": None,
            "approval": None,
            "messages": [],
            "tool_results": [],
            "errors": [],
            "events": [],
        }

        try:
            for chunk in graph.stream(state, config=config):
                for _node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    for ev in update.get("events", []):
                        yield f"data: {json.dumps({'type': 'step', 'event': ev})}\n\n"

            snap = graph.get_state(config)
            vals = snap.values
            yield f"data: {json.dumps({'type': 'done', 'route': vals.get('route', ''), 'final_answer': vals.get('final_answer', ''), 'pending_question': vals.get('pending_question')})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

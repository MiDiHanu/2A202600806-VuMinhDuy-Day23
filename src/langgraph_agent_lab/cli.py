"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .recovery import verify_crash_resume, write_resume_evidence
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state
from .visualize import build_trace, write_visualization

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
    verify_resume: Annotated[bool, typer.Option("--verify-resume/--no-verify-resume")] = True,
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    final_states = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        final_states.append(final_state)
        metrics.append(
            metric_from_state(
                final_state, scenario.expected_route.value, scenario.requires_approval
            )
        )
    report = summarize_metrics(metrics)

    # Extension: demonstrate crash-resume / time-travel on a SQLite checkpointer
    # and record the result in the metrics + an evidence log.
    if verify_resume:
        evidence = verify_crash_resume()
        report.resume_success = bool(evidence.get("resume_success"))
        write_resume_evidence(evidence, "reports/resume_evidence.md")
        typer.echo(f"Resume verification: {evidence}")

    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])

    # Build the visualization dashboard (token cost, steps, per-step reasoning).
    trace = build_trace(final_states, scenarios, report)
    write_visualization(trace, "outputs/visualization.html", "outputs/trace.json")
    typer.echo("Wrote visualization to outputs/visualization.html")
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("export-diagram")
def export_diagram(
    output: Annotated[Path, typer.Option("--output")] = Path("reports/graph.mermaid"),
) -> None:
    """Export the compiled graph as a Mermaid diagram (offline, no LLM)."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    mermaid = graph.get_graph().draw_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(mermaid, encoding="utf-8")
    typer.echo(f"Wrote Mermaid diagram to {output}")


@app.command("visualize")
def visualize(
    trace: Annotated[Path, typer.Option("--trace")] = Path("outputs/trace.json"),
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/visualization.html"),
) -> None:
    """Re-render the HTML dashboard from a saved trace.json (no LLM calls)."""
    from .visualize import render_html

    data = json.loads(trace.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")
    typer.echo(f"Wrote visualization to {output}")


@app.command("serve")
def serve(
    port: Annotated[int, typer.Option("--port")] = 8000,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
) -> None:
    """Start the live demo server (requires 'serve' extra)."""
    try:
        import uvicorn
        from .server import app as fastapi_app
    except ImportError:
        typer.echo("Install the 'serve' extra: uv sync --extra serve", err=True)
        raise typer.Exit(1)

    typer.echo(f"Live demo server -> http://{host}:{port}")
    typer.echo("Open the URL above, go to the Demo tab, and type any query.")
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command("demo-resume")
def demo_resume() -> None:
    """Run the crash-resume / time-travel demonstration and print evidence."""
    evidence = verify_crash_resume()
    write_resume_evidence(evidence, "reports/resume_evidence.md")
    typer.echo(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    app()

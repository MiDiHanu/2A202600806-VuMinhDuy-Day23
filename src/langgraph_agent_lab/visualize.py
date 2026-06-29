"""Self-contained HTML dashboard for LangGraph agent runs.

Generates a multi-tab interactive dashboard with:
- Overview: KPI grid, token cost breakdown, architecture graph
- Scenarios: expandable traces with step reasoning
- Analytics: per-scenario bar charts and comparison table
- Demo: interactive JS-based query routing simulator
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .metrics import MetricsReport, active_model

NODE_ICON = {
    "intake": "📥",
    "classify": "🧭",
    "tool": "🔧",
    "evaluate": "⚖️",
    "answer": "💬",
    "clarify": "❓",
    "risky_action": "⚠️",
    "approval": "🧑‍⚖️",
    "retry": "🔁",
    "dead_letter": "💀",
    "finalize": "✅",
}

ROUTE_COLOR = {
    "simple": "#3fb950",
    "tool": "#58a6ff",
    "missing_info": "#e3b341",
    "risky": "#f78166",
    "error": "#f85149",
}


def build_trace(
    states: list[dict[str, Any]],
    scenarios: list[Any],
    report: MetricsReport,
) -> dict[str, Any]:
    """Assemble a JSON-serializable trace from final states + scenarios + report."""
    by_id = {s.id: s for s in scenarios}
    metrics_by_id = {m.scenario_id: m for m in report.scenario_metrics}
    scenario_traces = []
    for state in states:
        sid = str(state.get("scenario_id", "unknown"))
        scenario = by_id.get(sid)
        metric = metrics_by_id.get(sid)
        expected_route = scenario.expected_route.value if scenario is not None else None
        steps = []
        for event in state.get("events", []) or []:
            md = event.get("metadata", {}) or {}
            steps.append(
                {
                    "node": event.get("node", "unknown"),
                    "event_type": event.get("event_type", ""),
                    "message": event.get("message", ""),
                    "latency_ms": int(event.get("latency_ms", 0) or 0),
                    "used_llm": bool(md.get("used_llm")),
                    "reasoning": md.get("reasoning", ""),
                    "input_tokens": int(md.get("input_tokens", 0) or 0),
                    "output_tokens": int(md.get("output_tokens", 0) or 0),
                    "total_tokens": int(md.get("total_tokens", 0) or 0),
                }
            )
        scenario_traces.append(
            {
                "scenario_id": sid,
                "query": state.get("query", ""),
                "expected_route": expected_route,
                "actual_route": state.get("route", ""),
                "success": bool(metric.success) if metric else None,
                "requires_approval": bool(getattr(scenario, "requires_approval", False)),
                "final_answer": state.get("final_answer"),
                "pending_question": state.get("pending_question"),
                "input_tokens": metric.input_tokens if metric else 0,
                "output_tokens": metric.output_tokens if metric else 0,
                "total_tokens": metric.total_tokens if metric else 0,
                "cost_usd": metric.cost_usd if metric else 0.0,
                "latency_ms": metric.latency_ms if metric else 0,
                "llm_calls": metric.llm_calls if metric else 0,
                "steps": steps,
            }
        )
    return {
        "model": report.model or active_model(),
        "summary": report.model_dump(exclude={"scenario_metrics"}),
        "scenarios": scenario_traces,
    }


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def render_html(trace: dict[str, Any]) -> str:  # noqa: PLR0914
    """Render the full standalone HTML dashboard."""
    s = trace["summary"]
    model = trace["model"]
    total_tokens = s.get("total_tokens", 0)
    in_tok = s.get("total_input_tokens", 0)
    out_tok = s.get("total_output_tokens", 0)
    cost = s.get("total_cost_usd", 0.0)
    success_rate = s.get("success_rate", 0.0)
    scenarios = trace["scenarios"]
    n = len(scenarios) or 1
    avg_cost = cost / n
    proj_1k = cost / n * 1000
    in_pct = round((in_tok / total_tokens * 100) if total_tokens else 0, 1)
    out_pct = round(100 - in_pct if total_tokens else 0, 1)

    data_json = json.dumps(trace, ensure_ascii=False)

    scenario_rows_html = _build_scenario_rows(scenarios)
    analytics_html = _build_analytics(scenarios, s)
    demo_presets_html = _build_demo_presets(scenarios)
    first_query = next((sc.get("query", "") for sc in scenarios if sc.get("query")), "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LangGraph Agent Dashboard</title>
<meta name="description" content="Interactive visualization of LangGraph agent run: token cost, routing decisions, step-by-step reasoning, and live demo."/>
<style>
/* ── Reset & tokens ─────────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0d1117; --surface:#161b22; --surface2:#21262d; --border:#30363d;
  --ink:#e6edf3; --muted:#7d8590; --subtle:#484f58;
  --blue:#58a6ff; --green:#3fb950; --red:#f85149;
  --amber:#e3b341; --pink:#f78166; --purple:#bc8cff;
  --route-simple:#3fb950; --route-tool:#58a6ff;
  --route-missing:#e3b341; --route-risky:#f78166; --route-error:#f85149;
  --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --radius:8px; --radius-lg:12px;
}}
html{{scroll-behavior:smooth}}
body{{
  background:var(--bg); color:var(--ink); font:14px/1.6 var(--sans);
  min-height:100vh; -webkit-font-smoothing:antialiased;
}}

/* ── Layout ─────────────────────────────────────────────── */
.app{{display:flex; flex-direction:column; min-height:100vh}}
.topbar{{
  position:sticky; top:0; z-index:100;
  background:color-mix(in srgb,var(--bg) 85%,transparent);
  backdrop-filter:blur(12px); border-bottom:1px solid var(--border);
  padding:0 24px;
}}
.topbar-inner{{
  max-width:1100px; margin:0 auto; display:flex; align-items:center; gap:16px; height:56px;
}}
.logo{{display:flex;align-items:center;gap:10px;text-decoration:none}}
.logo-icon{{
  width:30px;height:30px;border-radius:7px;
  background:linear-gradient(135deg,#1f6feb,#388bfd);
  display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0
}}
.logo-text{{font-size:14px;font-weight:600;color:var(--ink);letter-spacing:-.01em}}
.logo-text span{{color:var(--muted);font-weight:400}}
.topbar-sep{{width:1px;height:20px;background:var(--border)}}
.nav-tabs{{display:flex;gap:2px;flex:1}}
.nav-tab{{
  padding:6px 14px; border-radius:6px; border:none; cursor:pointer;
  font:13px/1 var(--sans); color:var(--muted); background:transparent;
  transition:color .15s, background .15s;
}}
.nav-tab:hover{{color:var(--ink);background:var(--surface2)}}
.nav-tab.active{{color:var(--ink);background:var(--surface2)}}
.model-badge{{
  display:flex; align-items:center; gap:6px; padding:4px 10px;
  border-radius:20px; border:1px solid var(--border); background:var(--surface);
  font:11px/1 var(--mono); color:var(--muted); white-space:nowrap;
}}
.model-badge::before{{content:"";width:6px;height:6px;border-radius:50%;background:var(--green);flex-shrink:0}}

.content{{max-width:1100px;margin:0 auto;padding:32px 24px 80px;flex:1}}

/* ── Tabs ────────────────────────────────────────────────── */
.tab-panel{{display:none}}
.tab-panel.active{{display:block}}

/* ── Section heading ─────────────────────────────────────── */
.section-head{{
  display:flex; align-items:baseline; gap:12px; margin-bottom:20px; padding-bottom:12px;
  border-bottom:1px solid var(--border);
}}
.section-head h2{{font-size:16px;font-weight:600;letter-spacing:-.01em}}
.section-head .count{{
  font:11px/1 var(--mono); color:var(--muted);
  background:var(--surface2); padding:2px 8px; border-radius:20px;
}}

/* ── KPI grid ────────────────────────────────────────────── */
.kpi-grid{{
  display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border); border-radius:var(--radius-lg);
  overflow:hidden; margin-bottom:24px;
}}
@media(max-width:640px){{.kpi-grid{{grid-template-columns:repeat(2,1fr)}}}}
.kpi{{background:var(--surface); padding:18px 20px}}
.kpi-val{{font:600 24px/1 var(--mono); font-variant-numeric:tabular-nums; letter-spacing:-.02em}}
.kpi-val.green{{color:var(--green)}} .kpi-val.blue{{color:var(--blue)}}
.kpi-val.amber{{color:var(--amber)}} .kpi-val.purple{{color:var(--purple)}}
.kpi-label{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:6px}}
.kpi-sub{{font:11px/1.3 var(--mono);color:var(--subtle);margin-top:4px}}

/* ── Token bar card ──────────────────────────────────────── */
.token-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg);
  padding:20px; margin-bottom:24px;
}}
.token-card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}}
.token-card-head h3{{font:13px/1 var(--sans);font-weight:600}}
.token-card-head .total{{font:12px/1 var(--mono);color:var(--muted)}}
.tok-bar{{height:12px; border-radius:6px; overflow:hidden; display:flex; background:var(--surface2)}}
.tok-bar-in{{background:#1f6feb;transition:width .4s ease}}
.tok-bar-out{{background:#8957e5;transition:width .4s ease}}
.tok-legend{{display:flex;gap:24px;margin-top:10px}}
.tok-legend span{{font:12px/1.4 var(--mono);color:var(--muted);display:flex;align-items:center;gap:7px}}
.tok-dot{{width:8px;height:8px;border-radius:2px;display:inline-block;flex-shrink:0}}

/* ── Architecture ────────────────────────────────────────── */
.arch-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg);
  padding:24px; margin-bottom:24px; overflow-x:auto;
}}
.arch-card h3{{font-size:13px;font-weight:600;margin-bottom:20px;color:var(--ink)}}
.arch-svg{{width:100%;min-width:500px;max-width:860px;display:block;margin:0 auto}}

/* ── Scenario cards ──────────────────────────────────────── */
.scenario-list{{display:flex;flex-direction:column;gap:10px}}
.scenario-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg);
  overflow:hidden; transition:border-color .15s;
}}
.scenario-card:hover{{border-color:var(--subtle)}}
.scenario-header{{
  display:flex; align-items:center; gap:10px; padding:14px 16px;
  cursor:pointer; user-select:none; list-style:none;
}}
.scenario-header::-webkit-details-marker{{display:none}}
.route-tag{{
  padding:3px 9px; border-radius:5px; font:700 10px/1.4 var(--mono);
  text-transform:uppercase; letter-spacing:.06em; flex-shrink:0;
}}
.sc-id{{font:600 13px/1 var(--mono); flex-shrink:0}}
.sc-query{{color:var(--muted);font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}}
.sc-badges{{display:flex;gap:6px;align-items:center;flex-shrink:0}}
.badge{{
  padding:2px 8px; border-radius:4px; font:700 10px/1.4 var(--mono);
  text-transform:uppercase; letter-spacing:.06em;
}}
.badge-pass{{background:color-mix(in srgb,var(--green) 15%,transparent);color:var(--green);border:1px solid color-mix(in srgb,var(--green) 35%,transparent)}}
.badge-fail{{background:color-mix(in srgb,var(--red) 15%,transparent);color:var(--red);border:1px solid color-mix(in srgb,var(--red) 35%,transparent)}}
.badge-hitl{{background:color-mix(in srgb,var(--purple) 15%,transparent);color:var(--purple);border:1px solid color-mix(in srgb,var(--purple) 35%,transparent)}}
.chevron{{color:var(--muted);font-size:11px;transition:transform .2s;margin-left:4px;flex-shrink:0}}
details[open] .chevron{{transform:rotate(90deg)}}
.sc-body{{border-top:1px solid var(--border)}}
.sc-meta-bar{{
  display:flex;flex-wrap:wrap;gap:6px 20px;padding:10px 16px 8px;
  border-bottom:1px solid var(--border); background:var(--surface2);
}}
.sc-meta-bar span{{font:11px/1.4 var(--mono);color:var(--muted)}}
.sc-meta-bar b{{color:var(--ink);font-variant-numeric:tabular-nums}}
.sc-meta-bar .ok{{color:var(--green)}} .sc-meta-bar .bad{{color:var(--red)}}

/* ── Timeline ────────────────────────────────────────────── */
.timeline{{padding:12px 16px 6px;position:relative}}
.timeline::before{{
  content:""; position:absolute; left:31px; top:26px; bottom:28px;
  width:1px; background:var(--border);
}}
.step{{display:flex;gap:12px;padding:6px 0;position:relative}}
.step-icon{{
  flex:0 0 30px; height:30px; border-radius:7px;
  background:var(--surface2); border:1px solid var(--border);
  display:flex; align-items:center; justify-content:center;
  font-size:14px; z-index:1; flex-shrink:0;
}}
.step-err .step-icon{{border-color:var(--red);box-shadow:0 0 0 3px color-mix(in srgb,var(--red) 15%,transparent)}}
.step-body{{flex:1;padding-top:4px}}
.step-row{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.step-name{{font:600 12px/1 var(--mono);color:var(--ink)}}
.step-msg{{font-size:12px;color:var(--muted)}}
.chips{{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}}
.chip{{
  font:11px/1.3 var(--mono); padding:2px 7px; border-radius:4px;
  background:var(--surface2); border:1px solid var(--border); color:var(--muted);
  font-variant-numeric:tabular-nums;
}}
.chip-llm{{
  background:color-mix(in srgb,var(--blue) 12%,transparent);
  color:var(--blue); border-color:color-mix(in srgb,var(--blue) 35%,transparent);
}}
.chip-tok{{color:var(--ink)}}
.reasoning{{
  margin-top:7px; font-size:12px; line-height:1.5; color:#c9d1d9;
  background:var(--surface2); border:1px solid var(--border);
  border-left:2px solid var(--blue); border-radius:5px; padding:7px 10px;
}}
.reasoning-label{{
  font:10px/1 var(--mono); text-transform:uppercase; letter-spacing:.1em;
  color:var(--blue); margin-right:8px;
}}
.final-answer{{
  margin:4px 16px 14px; padding:12px 14px; border-radius:7px;
  background:var(--surface2); border:1px solid var(--border);
}}
.final-label{{
  display:block; font:10px/1 var(--mono); text-transform:uppercase;
  letter-spacing:.1em; color:var(--muted); margin-bottom:6px;
}}
.final-text{{font-size:13px; color:var(--ink); line-height:1.5}}

/* ── Analytics ───────────────────────────────────────────── */
.analytics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
@media(max-width:680px){{.analytics-grid{{grid-template-columns:1fr}}}}
.chart-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg); padding:20px;
}}
.chart-card h3{{font-size:13px;font-weight:600;margin-bottom:16px;color:var(--ink)}}
.bar-chart{{display:flex;flex-direction:column;gap:8px}}
.bar-row{{display:flex;align-items:center;gap:10px}}
.bar-label{{font:11px/1 var(--mono);color:var(--muted);width:90px;text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar-track{{flex:1;height:18px;background:var(--surface2);border-radius:4px;overflow:hidden;position:relative}}
.bar-fill{{height:100%;border-radius:4px;transition:width .6s ease}}
.bar-val{{font:11px/1 var(--mono);color:var(--muted);width:60px;flex-shrink:0;font-variant-numeric:tabular-nums}}
.table-wrap{{overflow-x:auto}}
table.data{{width:100%;border-collapse:collapse;font-size:12px}}
table.data th{{
  font:10px/1 var(--mono);text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted); padding:6px 10px; text-align:left;
  border-bottom:1px solid var(--border);
}}
table.data td{{
  padding:8px 10px; border-bottom:1px solid var(--border);
  font-variant-numeric:tabular-nums; color:var(--ink); font-size:12px;
}}
table.data tr:last-child td{{border-bottom:none}}
table.data tr:hover td{{background:var(--surface2)}}
.pill{{
  display:inline-block; padding:1px 7px; border-radius:20px;
  font:10px/1.6 var(--mono); text-transform:uppercase; letter-spacing:.04em;
}}

/* ── Demo ────────────────────────────────────────────────── */
.demo-layout{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;align-items:start}}
@media(max-width:720px){{.demo-layout{{grid-template-columns:minmax(0,1fr)}}}}
.demo-input-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg); padding:22px;
  min-width:0; overflow:hidden;
}}
.demo-input-card h3{{font-size:14px;font-weight:600;margin-bottom:6px}}
.demo-input-card p{{font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.5}}
.query-input{{
  width:100%; padding:10px 14px; border-radius:var(--radius);
  border:1px solid var(--border); background:var(--surface2);
  color:var(--ink); font:14px/1 var(--sans); outline:none;
  transition:border-color .15s; margin-bottom:10px;
}}
.query-input:focus{{border-color:var(--blue);box-shadow:0 0 0 3px color-mix(in srgb,var(--blue) 20%,transparent)}}
.demo-presets{{
  display:flex; flex-direction:column; gap:6px; margin-bottom:14px;
  max-height:248px; overflow-y:auto; padding-right:4px;
}}
.demo-presets::-webkit-scrollbar{{width:8px}}
.demo-presets::-webkit-scrollbar-thumb{{background:var(--border);border-radius:4px}}
.preset-btn{{
  display:flex; align-items:center; gap:8px;
  padding:7px 10px; border-radius:6px; border:1px solid var(--border);
  background:var(--surface2); color:var(--muted); font-size:12px;
  cursor:pointer; text-align:left; transition:all .15s;
  width:100%; min-width:0; overflow:hidden;
}}
.preset-btn:hover{{border-color:var(--blue);color:var(--ink);background:color-mix(in srgb,var(--blue) 8%,var(--surface2))}}
.preset-route{{
  flex:0 0 auto; font:9px/1.5 var(--mono); text-transform:uppercase; letter-spacing:.04em;
  padding:1px 6px; border-radius:20px; border:1px solid var(--border);
}}
.preset-text{{flex:1 1 auto; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}}
.preset-id{{flex:0 0 auto; font:9px/1 var(--mono); color:var(--muted); opacity:.7; white-space:nowrap}}
.run-btn{{
  width:100%; padding:10px; border-radius:var(--radius); border:none;
  background:var(--blue); color:#fff; font:600 13px/1 var(--sans);
  cursor:pointer; transition:opacity .15s;
}}
.run-btn:hover{{opacity:.85}} .run-btn:disabled{{opacity:.4;cursor:not-allowed}}
.demo-output-card{{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg);
  padding:22px; min-height:300px; min-width:0; overflow:hidden;
}}
.demo-output-card h3{{font-size:14px;font-weight:600;margin-bottom:16px}}
.demo-placeholder{{color:var(--muted);font-size:13px;text-align:center;padding:60px 0}}
.demo-placeholder p{{margin-top:8px}}
.demo-steps{{display:flex;flex-direction:column;gap:0}}
.demo-step{{
  display:flex; gap:12px; padding:10px 0;
  border-left:2px solid var(--border); margin-left:15px; padding-left:16px;
  position:relative; opacity:0; transform:translateY(6px);
  transition:opacity .3s ease, transform .3s ease;
}}
.demo-step.visible{{opacity:1;transform:translateY(0)}}
.demo-step-dot{{
  position:absolute; left:-9px; top:14px; width:16px; height:16px;
  border-radius:50%; background:var(--surface2); border:2px solid var(--border);
  display:flex; align-items:center; justify-content:center; font-size:9px;
}}
.demo-step-dot.active{{border-color:var(--blue);background:color-mix(in srgb,var(--blue) 20%,var(--surface2))}}
.demo-step-name{{font:600 12px/1 var(--mono);color:var(--ink)}}
.demo-step-desc{{font-size:12px;color:var(--muted);margin-top:2px}}
.demo-step-reason{{
  font-size:12px;line-height:1.4;color:#c9d1d9;
  background:var(--surface2);border-radius:4px;padding:5px 8px;margin-top:5px;
  border-left:2px solid var(--blue);
}}
.demo-route-banner{{
  margin-top:14px; padding:12px 16px; border-radius:var(--radius);
  border:1px solid var(--border); text-align:center;
  font:600 13px/1 var(--mono); letter-spacing:.05em; text-transform:uppercase;
}}
.spinner{{
  display:inline-block; width:14px; height:14px; border-radius:50%;
  border:2px solid var(--border); border-top-color:var(--blue);
  animation:spin .6s linear infinite; vertical-align:middle; margin-right:6px;
}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.live-badge{{
  display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:20px;
  font:10px/1.4 var(--mono); text-transform:uppercase; letter-spacing:.06em;
}}
.live-badge.offline{{background:color-mix(in srgb,var(--muted) 14%,transparent);color:var(--muted);border:1px solid var(--border)}}
.live-badge.live{{background:color-mix(in srgb,#3fb950 15%,transparent);color:#3fb950;border:1px solid #3fb950}}
.final-answer-live{{
  margin-top:14px; padding:14px 16px; border-radius:var(--radius);
  background:color-mix(in srgb,var(--blue) 8%,var(--surface2));
  border:1px solid color-mix(in srgb,var(--blue) 30%,var(--border));
  font-size:13px; line-height:1.55; color:var(--ink);
}}
.final-answer-live .fa-label{{
  font:10px/1 var(--mono); text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); display:block; margin-bottom:6px;
}}
.demo-input-card code{{
  font:11px/1 var(--mono); background:var(--surface2); color:var(--blue);
  padding:2px 5px; border-radius:4px; border:1px solid var(--border);
}}
</style>
</head>
<body>
<div class="app">

<!-- ── Top bar ── -->
<nav class="topbar">
  <div class="topbar-inner">
    <a class="logo" href="#">
      <span class="logo-icon">🤖</span>
      <span class="logo-text">LangGraph <span>Agent Dashboard</span></span>
    </a>
    <div class="topbar-sep"></div>
    <div class="nav-tabs">
      <button class="nav-tab" onclick="switchTab('overview',this)">Overview</button>
      <button class="nav-tab" onclick="switchTab('scenarios',this)">Scenarios</button>
      <button class="nav-tab" onclick="switchTab('analytics',this)">Analytics</button>
      <button class="nav-tab active" onclick="switchTab('demo',this)">Demo ✨</button>
    </div>
    <span class="model-badge">{_esc(model)}</span>
  </div>
</nav>

<div class="content">

<!-- ════════════════════════════════ OVERVIEW ═════════════════════════════════ -->
<div id="tab-overview" class="tab-panel">

  <div class="section-head">
    <h2>Run Overview</h2>
    <span class="count">{s.get("total_scenarios",0)} scenarios · {_esc(model)}</span>
  </div>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-val green">{s.get("total_scenarios",0)}</div>
      <div class="kpi-label">Scenarios</div>
      <div class="kpi-sub">{s.get("avg_nodes_visited",0):.1f} nodes avg</div>
    </div>
    <div class="kpi">
      <div class="kpi-val green">{success_rate:.0%}</div>
      <div class="kpi-label">Success Rate</div>
      <div class="kpi-sub">{int(s.get("total_scenarios",0)*success_rate)}/{s.get("total_scenarios",0)} passed</div>
    </div>
    <div class="kpi">
      <div class="kpi-val blue">{total_tokens:,}</div>
      <div class="kpi-label">Total Tokens</div>
      <div class="kpi-sub">{s.get("total_llm_calls",0)} LLM calls</div>
    </div>
    <div class="kpi">
      <div class="kpi-val amber">${cost:.5f}</div>
      <div class="kpi-label">Est. Cost</div>
      <div class="kpi-sub">~${proj_1k:.2f} per 1k runs</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">{in_tok:,}</div>
      <div class="kpi-label">Input Tokens</div>
      <div class="kpi-sub">{in_pct}% of total</div>
    </div>
    <div class="kpi">
      <div class="kpi-val purple">{out_tok:,}</div>
      <div class="kpi-label">Output Tokens</div>
      <div class="kpi-sub">{out_pct}% of total</div>
    </div>
    <div class="kpi">
      <div class="kpi-val amber">{s.get("total_retries",0)}</div>
      <div class="kpi-label">Retries</div>
      <div class="kpi-sub">bounded retry loops</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">{"✅" if s.get("resume_success") else "—"}</div>
      <div class="kpi-label">Crash Resume</div>
      <div class="kpi-sub">SQLite WAL checkpt</div>
    </div>
  </div>

  <div class="token-card">
    <div class="token-card-head">
      <h3>Token Usage Breakdown</h3>
      <span class="total">{total_tokens:,} total · ${avg_cost:.6f} avg/scenario</span>
    </div>
    <div class="tok-bar">
      <div class="tok-bar-in" style="width:{in_pct}%"></div>
      <div class="tok-bar-out" style="width:{out_pct}%"></div>
    </div>
    <div class="tok-legend">
      <span><span class="tok-dot" style="background:#1f6feb"></span>Input: {in_tok:,} tokens ({in_pct}%)</span>
      <span><span class="tok-dot" style="background:#8957e5"></span>Output: {out_tok:,} tokens ({out_pct}%)</span>
    </div>
  </div>

  <div class="arch-card">
    <h3>Graph Architecture — 11 nodes · 4 conditional routers</h3>
    {_build_arch_svg()}
  </div>

</div>

<!-- ════════════════════════════════ SCENARIOS ════════════════════════════════ -->
<div id="tab-scenarios" class="tab-panel">
  <div class="section-head">
    <h2>Scenario Traces</h2>
    <span class="count">{s.get("total_scenarios",0)} runs</span>
  </div>
  <div class="scenario-list">
{scenario_rows_html}
  </div>
</div>

<!-- ════════════════════════════════ ANALYTICS ════════════════════════════════ -->
<div id="tab-analytics" class="tab-panel">
  <div class="section-head">
    <h2>Run Analytics</h2>
    <span class="count">per-scenario breakdown</span>
  </div>
{analytics_html}
</div>

<!-- ════════════════════════════════ DEMO ═════════════════════════════════════ -->
<div id="tab-demo" class="tab-panel active">
  <div class="section-head">
    <h2>Interactive Demo</h2>
    <span id="live-badge" class="live-badge offline">◌ Simulated</span>
  </div>
  <p id="demo-subtitle" style="font-size:13px;color:var(--muted);margin-bottom:20px;max-width:70ch">
    Type any support query below and watch it route step-by-step through the graph —
    classification, tool calls, approval gates, and final answer. Right now this is an
    offline JavaScript simulation. To run the <b>real</b> agent (Gemini-powered),
    start the server with <code>uv run agent-lab serve</code> and open
    <code>http://127.0.0.1:8000</code> — this badge will switch to “● Live”.
  </p>
  <div class="demo-layout">
    <div class="demo-input-card">
      <h3>Your query</h3>
      <p>Pick a scenario from this run ({s.get("total_scenarios",0)}) or write your own:</p>
      <div class="demo-presets">
        {demo_presets_html}
      </div>
      <input id="demo-query" class="query-input" type="text"
             placeholder="Type your support query here…"
             value="{_esc(first_query)}"
             onkeydown="if(event.key==='Enter')runDemo()"/>
      <button class="run-btn" id="run-btn" onclick="runDemo()">▶ Simulate Routing</button>
    </div>
    <div class="demo-output-card">
      <h3>Agent trace</h3>
      <div id="demo-output">
        <div class="demo-placeholder">
          <div style="font-size:32px">🤖</div>
          <p>Pick a scenario or type a query, then run it to watch the agent work</p>
        </div>
      </div>
    </div>
  </div>
</div>

</div><!-- /.content -->
</div><!-- /.app -->

<script>
/* ── Tab switching ── */
function switchTab(name, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}

/* ── Embedded trace data ── */
const TRACE = {data_json};

/* ── Demo simulator ── */
function setQuery(btn) {{
  document.getElementById('demo-query').value = (btn.dataset.query || btn.textContent).trim();
}}

const ROUTES = {{
  simple: {{
    color: '#3fb950',
    steps: [
      {{node:'intake', icon:'📥', desc:'Query normalized and validated'}},
      {{node:'classify', icon:'🧭', desc:'Intent: simple question', reason:'No external data needed — answerable from general knowledge.', llm:true}},
      {{node:'answer', icon:'💬', desc:'LLM generates final answer', reason:'Grounded generation over query context.', llm:true}},
      {{node:'finalize', icon:'✅', desc:'Workflow complete'}}
    ]
  }},
  tool: {{
    color: '#58a6ff',
    steps: [
      {{node:'intake', icon:'📥', desc:'Query normalized and validated'}},
      {{node:'classify', icon:'🧭', desc:'Intent: tool call required', reason:'Query references specific data (order, account) requiring lookup.', llm:true}},
      {{node:'tool', icon:'🔧', desc:'Mock tool executed — data retrieved'}},
      {{node:'evaluate', icon:'⚖️', desc:'LLM judges tool result quality', reason:'Tool returned valid data — verdict: success.', llm:true}},
      {{node:'answer', icon:'💬', desc:'LLM generates grounded answer', reason:'Grounded on tool_results from lookup.', llm:true}},
      {{node:'finalize', icon:'✅', desc:'Workflow complete'}}
    ]
  }},
  missing_info: {{
    color: '#e3b341',
    steps: [
      {{node:'intake', icon:'📥', desc:'Query normalized and validated'}},
      {{node:'classify', icon:'🧭', desc:'Intent: missing information', reason:'Query is too vague to act on without more context.', llm:true}},
      {{node:'clarify', icon:'❓', desc:'Clarification question formulated', llm:true}},
      {{node:'finalize', icon:'✅', desc:'Awaiting user response'}}
    ]
  }},
  risky: {{
    color: '#f78166',
    steps: [
      {{node:'intake', icon:'📥', desc:'Query normalized and validated'}},
      {{node:'classify', icon:'🧭', desc:'Intent: risky action detected', reason:'Request involves destructive or high-impact side-effect (refund/delete/email).', llm:true}},
      {{node:'risky_action', icon:'⚠️', desc:'Risky action flagged — preparing approval request'}},
      {{node:'approval', icon:'🧑‍⚖️', desc:'Human-in-the-loop gate — approval granted (simulated)'}},
      {{node:'tool', icon:'🔧', desc:'Tool executed after approval'}},
      {{node:'evaluate', icon:'⚖️', desc:'Result evaluated', llm:true}},
      {{node:'answer', icon:'💬', desc:'Final answer with approval context', llm:true}},
      {{node:'finalize', icon:'✅', desc:'Workflow complete'}}
    ]
  }},
  error: {{
    color: '#f85149',
    steps: [
      {{node:'intake', icon:'📥', desc:'Query normalized and validated'}},
      {{node:'classify', icon:'🧭', desc:'Intent: error / malformed request', reason:'Query is unrecognizable or triggers an error state.', llm:true}},
      {{node:'retry', icon:'🔁', desc:'Retry attempt 1 of 3'}},
      {{node:'tool', icon:'🔧', desc:'Re-attempting tool call'}},
      {{node:'evaluate', icon:'⚖️', desc:'Evaluate retry result', llm:true}},
      {{node:'answer', icon:'💬', desc:'Best-effort answer generated', llm:true}},
      {{node:'finalize', icon:'✅', desc:'Workflow complete'}}
    ]
  }}
}};

function classify(query) {{
  const q = query.toLowerCase();
  const risky = ['refund','delete','remove','cancel','terminate','erase','wipe','send email','charge'];
  const tools  = ['order','status','lookup','check','find','track','invoice','account','balance','history'];
  const vague  = ['fix it','help me','broken','issue','problem','something wrong','not working'];
  if (risky.some(w => q.includes(w))) return 'risky';
  if (tools.some(w => q.includes(w))) return 'tool';
  if (vague.some(w => q.includes(w))) return 'missing_info';
  if (q.split(' ').length < 3) return 'missing_info';
  return 'simple';
}}

async function runDemo() {{
  if (LIVE_MODE) {{ return runLive(); }}
  const query = document.getElementById('demo-query').value.trim();
  if (!query) return;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Routing…';
  const out = document.getElementById('demo-output');
  out.innerHTML = '';

  const route = classify(query);
  const plan = ROUTES[route];

  for (let i = 0; i < plan.steps.length; i++) {{
    await delay(260 + i * 80);
    const s = plan.steps[i];
    const div = document.createElement('div');
    div.className = 'demo-step';
    div.innerHTML = `
      <div class="demo-step-dot active">${{s.icon}}</div>
      <div>
        <div class="demo-step-name">${{s.node}}</div>
        <div class="demo-step-desc">${{s.desc}}${{s.llm ? ' <span class="chip chip-llm" style="font-size:10px">LLM</span>' : ''}}</div>
        ${{s.reason ? `<div class="demo-step-reason">${{s.reason}}</div>` : ''}}
      </div>`;
    out.appendChild(div);
    requestAnimationFrame(() => requestAnimationFrame(() => div.classList.add('visible')));
  }}

  await delay(200);
  const banner = document.createElement('div');
  banner.className = 'demo-route-banner';
  banner.style.cssText = `border-color:${{plan.color}};color:${{plan.color}};background:color-mix(in srgb,${{plan.color}} 10%,transparent)`;
  banner.textContent = `→ routed as: ${{route}}`;
  out.appendChild(banner);
  requestAnimationFrame(() => requestAnimationFrame(() => banner.style.opacity='1'));

  btn.disabled = false;
  btn.innerHTML = '▶ Simulate Routing';
}}

function delay(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

/* ── Live mode: run the REAL agent via the demo server (agent-lab serve) ── */
let LIVE_MODE = false;
const NODE_ICONS = {{intake:'📥',classify:'🧭',tool:'🔧',evaluate:'⚖️',answer:'💬',clarify:'❓',risky_action:'⚠️',approval:'🧑‍⚖️',retry:'🔁',dead_letter:'☠️',finalize:'✅'}};
const ROUTE_COLORS = {{simple:'#3fb950',tool:'#58a6ff',missing_info:'#e3b341',risky:'#f78166',error:'#f85149',dead_letter:'#f85149'}};

async function checkLive() {{
  try {{
    const r = await fetch('/health', {{cache:'no-store'}});
    if (!r.ok) return;
    LIVE_MODE = true;
    const b = document.getElementById('live-badge');
    b.className = 'live-badge live';
    b.textContent = '● Live (Gemini)';
    document.getElementById('run-btn').textContent = '▶ Run Live';
    const sub = document.getElementById('demo-subtitle');
    if (sub) sub.innerHTML = 'Connected to the live agent server — your query runs through the <b>real</b> LangGraph graph powered by Gemini. Type anything and press Enter or click “▶ Run Live”.';
  }} catch (e) {{ /* offline — keep the JS simulator */ }}
}}

function _appendStep(out, node, desc, reason, llm, isErr) {{
  const div = document.createElement('div');
  div.className = 'demo-step';
  const icon = NODE_ICONS[node] || '•';
  div.innerHTML = `
    <div class="demo-step-dot active">${{icon}}</div>
    <div>
      <div class="demo-step-name">${{node}}</div>
      <div class="demo-step-desc" style="${{isErr ? 'color:#f85149' : ''}}">${{desc}}${{llm ? ' <span class="chip chip-llm" style="font-size:10px">LLM</span>' : ''}}</div>
      ${{reason ? `<div class="demo-step-reason">${{reason}}</div>` : ''}}
    </div>`;
  out.appendChild(div);
  requestAnimationFrame(() => requestAnimationFrame(() => div.classList.add('visible')));
}}

function _appendBanner(out, route) {{
  const color = ROUTE_COLORS[route] || '#58a6ff';
  const banner = document.createElement('div');
  banner.className = 'demo-route-banner';
  banner.style.cssText = `border-color:${{color}};color:${{color}};background:color-mix(in srgb,${{color}} 10%,transparent)`;
  banner.textContent = `→ routed as: ${{route}}`;
  out.appendChild(banner);
  requestAnimationFrame(() => requestAnimationFrame(() => banner.style.opacity='1'));
}}

function _appendAnswer(out, text) {{
  if (!text) return;
  const div = document.createElement('div');
  div.className = 'final-answer-live';
  div.innerHTML = `<span class="fa-label">Final answer</span>${{text}}`;
  out.appendChild(div);
}}

async function runLive() {{
  const query = document.getElementById('demo-query').value.trim();
  if (!query) return;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running…';
  const out = document.getElementById('demo-output');
  out.innerHTML = '';
  try {{
    const resp = await fetch('/api/run', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{query}})
    }});
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {{stream:true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!raw.startsWith('data:')) continue;
        let msg;
        try {{ msg = JSON.parse(raw.slice(5).trim()); }} catch (e) {{ continue; }}
        if (msg.type === 'step') {{
          const ev = msg.event || {{}};
          const meta = ev.metadata || {{}};
          const isErr = (ev.event_type === 'error');
          _appendStep(out, ev.node, ev.message || '', meta.reasoning || '', !!meta.used_llm, isErr);
        }} else if (msg.type === 'done') {{
          await delay(150);
          _appendBanner(out, msg.route || '');
          _appendAnswer(out, msg.final_answer || msg.pending_question || '');
        }} else if (msg.type === 'error') {{
          _appendStep(out, 'dead_letter', 'server error: ' + msg.message, '', false, true);
        }}
      }}
    }}
  }} catch (e) {{
    _appendStep(out, 'dead_letter', 'connection failed — is the server running? ' + e, '', false, true);
  }}
  btn.disabled = false;
  btn.innerHTML = '▶ Run Live';
}}

checkLive();
</script>
</body>
</html>"""


def _build_arch_svg() -> str:
    """Return inline SVG of the graph architecture."""
    return """<svg class="arch-svg" viewBox="0 0 860 340" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#484f58"/>
    </marker>
    <marker id="arr-b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#58a6ff"/>
    </marker>
    <marker id="arr-g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#3fb950"/>
    </marker>
    <marker id="arr-r" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#f85149"/>
    </marker>
    <marker id="arr-a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#e3b341"/>
    </marker>
    <marker id="arr-p" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#f78166"/>
    </marker>
  </defs>

  <!-- START -->
  <circle cx="60" cy="170" r="18" fill="#21262d" stroke="#30363d" stroke-width="1.5"/>
  <text x="60" y="175" text-anchor="middle" font-size="10" fill="#7d8590" font-family="monospace">START</text>

  <!-- intake -->
  <rect x="100" y="152" width="72" height="36" rx="7" fill="#161b22" stroke="#30363d" stroke-width="1.5"/>
  <text x="136" y="175" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">📥 intake</text>

  <!-- classify -->
  <rect x="200" y="152" width="78" height="36" rx="7" fill="#1f2937" stroke="#58a6ff" stroke-width="1.5"/>
  <text x="239" y="175" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">🧭 classify</text>

  <!-- answer (simple path) -->
  <rect x="340" y="40" width="72" height="36" rx="7" fill="#161b22" stroke="#3fb950" stroke-width="1.5"/>
  <text x="376" y="63" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">💬 answer</text>

  <!-- tool -->
  <rect x="340" y="112" width="68" height="36" rx="7" fill="#1f2937" stroke="#58a6ff" stroke-width="1.5"/>
  <text x="374" y="135" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">🔧 tool</text>

  <!-- evaluate -->
  <rect x="440" y="112" width="78" height="36" rx="7" fill="#1f2937" stroke="#58a6ff" stroke-width="1.5"/>
  <text x="479" y="135" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">⚖️ evaluate</text>

  <!-- retry -->
  <rect x="440" y="168" width="68" height="36" rx="7" fill="#161b22" stroke="#e3b341" stroke-width="1.5"/>
  <text x="474" y="191" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">🔁 retry</text>

  <!-- dead_letter -->
  <rect x="440" y="224" width="88" height="36" rx="7" fill="#161b22" stroke="#f85149" stroke-width="1.5"/>
  <text x="484" y="247" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">💀 dead_letter</text>

  <!-- clarify -->
  <rect x="340" y="196" width="72" height="36" rx="7" fill="#161b22" stroke="#e3b341" stroke-width="1.5"/>
  <text x="376" y="219" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">❓ clarify</text>

  <!-- risky_action -->
  <rect x="340" y="268" width="90" height="36" rx="7" fill="#161b22" stroke="#f78166" stroke-width="1.5"/>
  <text x="385" y="291" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">⚠️ risky</text>

  <!-- approval -->
  <rect x="460" y="268" width="80" height="36" rx="7" fill="#1f2937" stroke="#bc8cff" stroke-width="1.5"/>
  <text x="500" y="291" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">🧑 approval</text>

  <!-- finalize -->
  <rect x="600" y="152" width="78" height="36" rx="7" fill="#161b22" stroke="#3fb950" stroke-width="1.5"/>
  <text x="639" y="175" text-anchor="middle" font-size="12" fill="#e6edf3" font-family="monospace">✅ finalize</text>

  <!-- END -->
  <circle cx="730" cy="170" r="18" fill="#0d1117" stroke="#3fb950" stroke-width="2"/>
  <circle cx="730" cy="170" r="12" fill="#3fb950"/>
  <text x="730" y="174" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold" font-family="monospace">END</text>

  <!-- Arrows -->
  <!-- START -> intake -->
  <line x1="78" y1="170" x2="98" y2="170" stroke="#484f58" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- intake -> classify -->
  <line x1="172" y1="170" x2="198" y2="170" stroke="#484f58" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- classify -> answer (simple) -->
  <path d="M 239 152 Q 239 58 338 58" stroke="#3fb950" stroke-width="1.5" fill="none" marker-end="url(#arr-g)"/>
  <text x="270" y="90" font-size="9" fill="#3fb950" font-family="monospace">simple</text>

  <!-- classify -> tool -->
  <path d="M 278 163 Q 310 148 338 130" stroke="#58a6ff" stroke-width="1.5" fill="none" marker-end="url(#arr-b)"/>
  <text x="290" y="143" font-size="9" fill="#58a6ff" font-family="monospace">tool</text>

  <!-- classify -> clarify -->
  <path d="M 278 177 Q 310 207 338 214" stroke="#e3b341" stroke-width="1.5" fill="none" marker-end="url(#arr-a)"/>
  <text x="284" y="208" font-size="9" fill="#e3b341" font-family="monospace">missing</text>

  <!-- classify -> risky_action -->
  <path d="M 239 188 Q 239 276 338 286" stroke="#f78166" stroke-width="1.5" fill="none" marker-end="url(#arr-p)"/>
  <text x="247" y="254" font-size="9" fill="#f78166" font-family="monospace">risky</text>

  <!-- classify -> retry (error) -->
  <path d="M 278 170 Q 380 192 438 186" stroke="#f85149" stroke-width="1.5" stroke-dasharray="4,3" fill="none" marker-end="url(#arr-r)"/>
  <text x="330" y="185" font-size="9" fill="#f85149" font-family="monospace">error</text>

  <!-- tool -> evaluate -->
  <line x1="408" y1="130" x2="438" y2="130" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr-b)"/>

  <!-- evaluate -> answer (success) -->
  <path d="M 479 112 Q 479 58 522 58" stroke="#3fb950" stroke-width="1.5" fill="none" marker-end="url(#arr-g)"/>
  <text x="490" y="82" font-size="9" fill="#3fb950" font-family="monospace">success</text>

  <!-- evaluate -> retry -->
  <path d="M 479 148 L 479 166" stroke="#e3b341" stroke-width="1.5" marker-end="url(#arr-a)"/>
  <text x="483" y="163" font-size="9" fill="#e3b341" font-family="monospace">retry</text>

  <!-- retry -> tool loop -->
  <path d="M 440 178 Q 400 160 408 136" stroke="#e3b341" stroke-width="1.5" stroke-dasharray="4,3" fill="none" marker-end="url(#arr-a)"/>

  <!-- retry -> dead_letter -->
  <line x1="474" y1="204" x2="474" y2="222" stroke="#f85149" stroke-width="1.5" marker-end="url(#arr-r)"/>
  <text x="478" y="218" font-size="9" fill="#f85149" font-family="monospace">max</text>

  <!-- risky_action -> approval -->
  <line x1="430" y1="286" x2="458" y2="286" stroke="#f78166" stroke-width="1.5" marker-end="url(#arr-p)"/>

  <!-- approval -> tool (approved) -->
  <path d="M 500 268 Q 500 148 536 148" stroke="#3fb950" stroke-width="1.5" stroke-dasharray="4,3" fill="none" marker-end="url(#arr-g)"/>
  <text x="504" y="214" font-size="9" fill="#3fb950" font-family="monospace">ok</text>

  <!-- approval -> clarify (rejected) -->
  <path d="M 500 304 Q 500 322 380 322 Q 380 232 376 234" stroke="#f85149" stroke-width="1.5" stroke-dasharray="4,3" fill="none" marker-end="url(#arr-r)"/>
  <text x="430" y="320" font-size="9" fill="#f85149" font-family="monospace">reject</text>

  <!-- answer -> finalize -->
  <path d="M 412 58 Q 540 58 598 162" stroke="#3fb950" stroke-width="1.5" fill="none" marker-end="url(#arr-g)"/>

  <!-- clarify -> finalize -->
  <path d="M 412 214 Q 560 214 598 178" stroke="#484f58" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

  <!-- dead_letter -> finalize -->
  <path d="M 528 242 Q 565 242 598 168" stroke="#484f58" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

  <!-- finalize -> END -->
  <line x1="678" y1="170" x2="710" y2="170" stroke="#3fb950" stroke-width="1.5" marker-end="url(#arr-g)"/>

  <!-- Legend -->
  <rect x="620" y="22" width="220" height="76" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="632" y="38" font-size="10" fill="#7d8590" font-family="monospace">LEGEND</text>
  <line x1="632" y1="48" x2="652" y2="48" stroke="#3fb950" stroke-width="1.5"/>
  <text x="658" y="52" font-size="10" fill="#7d8590" font-family="monospace">success path</text>
  <line x1="632" y1="62" x2="652" y2="62" stroke="#58a6ff" stroke-width="1.5"/>
  <text x="658" y="66" font-size="10" fill="#7d8590" font-family="monospace">tool path</text>
  <line x1="632" y1="76" x2="652" y2="76" stroke="#e3b341" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="658" y="80" font-size="10" fill="#7d8590" font-family="monospace">conditional / retry</text>
  <line x1="632" y1="90" x2="652" y2="90" stroke="#f85149" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="658" y="94" font-size="10" fill="#7d8590" font-family="monospace">error / rejection</text>
</svg>"""


def _build_scenario_rows(scenarios: list[dict[str, Any]]) -> str:
    return "\n".join(_scenario_card(sc) for sc in scenarios)


def _build_demo_presets(scenarios: list[dict[str, Any]]) -> str:
    """Build the demo preset buttons from the scenarios actually run.

    Each button carries the real query (in data-query) and a route pill, so the
    Interactive Demo stays in sync with whatever scenario set was executed.
    """
    if not scenarios:
        return '<p style="font-size:12px;color:var(--muted)">No scenarios in this run.</p>'
    buttons = []
    for sc in scenarios:
        query = sc.get("query", "")
        if not query:
            continue
        route = sc.get("expected_route") or sc.get("actual_route") or "?"
        color = ROUTE_COLOR.get(route, "#7d8590")
        sid = sc.get("scenario_id", "")
        buttons.append(
            f'<button class="preset-btn" data-query="{_esc(query)}" onclick="setQuery(this)">'
            f'<span class="preset-route" style="color:{color};border-color:color-mix(in srgb,{color} 45%,transparent)">{_esc(route)}</span>'
            f'<span class="preset-text" title="{_esc(query)}">{_esc(query)}</span>'
            f'<span class="preset-id">{_esc(sid)}</span>'
            f"</button>"
        )
    return "\n        ".join(buttons)


def _scenario_card(sc: dict[str, Any]) -> str:
    route = sc.get("actual_route") or "?"
    exp = sc.get("expected_route") or "?"
    ok = sc.get("success")
    color = ROUTE_COLOR.get(route, "#7d8590")
    match_cls = "ok" if exp == route else "bad"
    ok_badge = '<span class="badge badge-pass">PASS</span>' if ok else '<span class="badge badge-fail">FAIL</span>'
    hitl_badge = '<span class="badge badge-hitl">HITL</span>' if sc.get("requires_approval") else ""
    steps_html = "\n".join(_step_html(s) for s in sc.get("steps", []))
    answer = sc.get("final_answer") or sc.get("pending_question") or ""
    cost = sc.get("cost_usd", 0.0)

    return f"""    <details class="scenario-card" open>
      <summary class="scenario-header">
        <span class="route-tag" style="background:color-mix(in srgb,{color} 15%,transparent);color:{color};border:1px solid color-mix(in srgb,{color} 40%,transparent)">{_esc(route)}</span>
        <span class="sc-id">{_esc(sc["scenario_id"])}</span>
        <span class="sc-query">{_esc(sc.get("query",""))}</span>
        <span class="sc-badges">{hitl_badge}{ok_badge}</span>
        <span class="chevron">▶</span>
      </summary>
      <div class="sc-body">
        <div class="sc-meta-bar">
          <span>route: <b class="{match_cls}">{_esc(exp)} → {_esc(route)}</b></span>
          <span>LLM calls: <b>{sc.get("llm_calls",0)}</b></span>
          <span>tokens: <b>{sc.get("total_tokens",0)}</b> ({sc.get("input_tokens",0)}↓ / {sc.get("output_tokens",0)}↑)</span>
          <span>cost: <b>${cost:.6f}</b></span>
          <span>latency: <b>{sc.get("latency_ms",0):,} ms</b></span>
          <span>nodes: <b>{len(sc.get("steps",[]))}</b></span>
        </div>
        <ol class="timeline">
{steps_html}
        </ol>
        <div class="final-answer">
          <span class="final-label">final answer / clarification</span>
          <div class="final-text">{_esc(answer)}</div>
        </div>
      </div>
    </details>"""


def _step_html(step: dict[str, Any]) -> str:
    node = step["node"]
    icon = NODE_ICON.get(node, "•")
    llm_chip = '<span class="chip chip-llm">LLM</span>' if step.get("used_llm") else ""
    err_cls = " step-err" if step.get("event_type") == "error" else ""
    lat = f'<span class="chip">{step["latency_ms"]} ms</span>' if step.get("latency_ms") else ""
    toks = ""
    if step.get("total_tokens"):
        toks = (
            f'<span class="chip">⬇ {step["input_tokens"]}</span>'
            f'<span class="chip">⬆ {step["output_tokens"]}</span>'
            f'<span class="chip chip-tok">{step["total_tokens"]} tok</span>'
        )
    reasoning = ""
    if step.get("reasoning"):
        reasoning = (
            f'<div class="reasoning"><span class="reasoning-label">reasoning</span>'
            f' {_esc(step["reasoning"])}</div>'
        )
    return f"""          <li class="step{err_cls}">
            <div class="step-icon">{icon}</div>
            <div class="step-body">
              <div class="step-row"><span class="step-name">{_esc(node)}</span>{llm_chip}
                <span class="step-msg">{_esc(step.get("message",""))}</span></div>
              <div class="chips">{lat}{toks}</div>
              {reasoning}
            </div>
          </li>"""


def _build_analytics(scenarios: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    max_tok = max((sc.get("total_tokens", 0) for sc in scenarios), default=1) or 1
    max_lat = max((sc.get("latency_ms", 0) for sc in scenarios), default=1) or 1

    tok_rows = ""
    for sc in scenarios:
        pct = round(sc.get("total_tokens", 0) / max_tok * 100, 1)
        sid = sc["scenario_id"]
        route = sc.get("actual_route", "?")
        color = ROUTE_COLOR.get(route, "#7d8590")
        tok_rows += (
            f'<div class="bar-row">'
            f'<span class="bar-label" title="{_esc(sid)}">{_esc(sid)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="bar-val">{sc.get("total_tokens",0)}</span>'
            f'</div>\n'
        )

    lat_rows = ""
    for sc in scenarios:
        pct = round(sc.get("latency_ms", 0) / max_lat * 100, 1)
        sid = sc["scenario_id"]
        route = sc.get("actual_route", "?")
        color = ROUTE_COLOR.get(route, "#7d8590")
        lat_ms = sc.get("latency_ms", 0)
        lat_rows += (
            f'<div class="bar-row">'
            f'<span class="bar-label" title="{_esc(sid)}">{_esc(sid)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="bar-val">{lat_ms:,} ms</span>'
            f'</div>\n'
        )

    table_rows = ""
    for sc in scenarios:
        route = sc.get("actual_route", "?")
        color = ROUTE_COLOR.get(route, "#7d8590")
        ok_icon = "✅" if sc.get("success") else "❌"
        table_rows += (
            f'<tr>'
            f'<td><code style="font-size:11px">{_esc(sc["scenario_id"])}</code></td>'
            f'<td><span class="pill" style="background:color-mix(in srgb,{color} 15%,transparent);color:{color}">{_esc(route)}</span></td>'
            f'<td style="text-align:center">{ok_icon}</td>'
            f'<td>{sc.get("total_tokens",0)}</td>'
            f'<td>${sc.get("cost_usd",0.0):.6f}</td>'
            f'<td>{sc.get("latency_ms",0):,}</td>'
            f'<td>{sc.get("llm_calls",0)}</td>'
            f'<td>{sc.get("retry_count",0) if "retry_count" in sc else "—"}</td>'
            f'</tr>\n'
        )

    return f"""  <div class="analytics-grid">
    <div class="chart-card">
      <h3>Token usage per scenario</h3>
      <div class="bar-chart">
{tok_rows}      </div>
    </div>
    <div class="chart-card">
      <h3>Latency per scenario</h3>
      <div class="bar-chart">
{lat_rows}      </div>
    </div>
  </div>

  <div class="chart-card" style="margin-bottom:24px">
    <h3>Full comparison table</h3>
    <div class="table-wrap">
      <table class="data">
        <thead>
          <tr>
            <th>Scenario</th><th>Route</th><th>Pass</th>
            <th>Tokens</th><th>Cost USD</th><th>Latency ms</th>
            <th>LLM calls</th><th>Retries</th>
          </tr>
        </thead>
        <tbody>
{table_rows}        </tbody>
      </table>
    </div>
  </div>"""


def write_visualization(
    trace: dict[str, Any],
    html_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    """Write visualization HTML (and optionally the raw trace JSON)."""
    hp = Path(html_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(render_html(trace), encoding="utf-8")
    if json_path:
        jp = Path(json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )

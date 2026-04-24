"""
app.py
------
Gradio demo app for the Agentic Research Planning Framework.
MSc AI & Machine Learning — Midterm Demo

Usage:
    pip install gradio
    python app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import json
import time
import threading
from datetime import datetime
from typing import Generator

import config
from agents import librarian_agent, planner_agent, reviewer_agent
from models.schemas import ReviewResult


# ── Colour constants ──────────────────────────────────────────────────────────
ACCENT   = "#2563EB"
SUCCESS  = "#16A34A"
WARNING  = "#D97706"
DANGER   = "#DC2626"
MUTED    = "#6B7280"


# ── Pipeline runner with streaming logs ──────────────────────────────────────

def format_log(msg: str, level: str = "info") -> str:
    icons = {"info": "◆", "success": "✓", "warning": "⚠", "error": "✗", "agent": "→"}
    icon  = icons.get(level, "◆")
    ts    = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}]  {icon}  {msg}"


def run_pipeline_streaming(topic: str) -> Generator:
    """
    Runs the full pipeline and yields (log, results_html, summary_html)
    tuples at each step so Gradio can stream updates live.
    """
    if not topic.strip():
        yield "Please enter a research topic.", "", ""
        return

    log_lines = []

    def log(msg, level="info"):
        log_lines.append(format_log(msg, level))
        return "\n".join(log_lines)

    # ── Step 1+2: Librarian ───────────────────────────────────────
    yield log(f"Starting pipeline for: '{topic}'"), "", ""
    yield log("Librarian Agent — fetching papers from arXiv...", "agent"), "", ""

    try:
        papers, gaps = librarian_agent.run(topic)
    except Exception as e:
        yield log(f"Pipeline error: {e}", "error"), "", ""
        return

    yield log(f"Retrieved {len(papers)} papers. Concepts extracted.", "success"), "", ""
    yield log(f"Identified {len(gaps)} research gap(s).", "success"), "", ""

    if not gaps:
        yield log("No gaps found. Try a more specific topic.", "warning"), "", ""
        return

    for g in gaps:
        yield log(f"Gap: {g.description[:70]}...", "info"), "", ""

    # ── Step 3: Planner ───────────────────────────────────────────
    yield log("Planner Agent — generating candidate research plans...", "agent"), "", ""

    try:
        plans = planner_agent.run(gaps, papers)
    except Exception as e:
        yield log(f"Planner error: {e}", "error"), "", ""
        return

    yield log(f"Generated {len(plans)} candidate plan(s) across {len(gaps)} gap(s).", "success"), "", ""

    # ── Step 4+5: Reviewer ────────────────────────────────────────
    yield log("Reviewer Agent — scoring novelty and checking feasibility...", "agent"), "", ""

    try:
        results = reviewer_agent.run(plans, papers)
    except Exception as e:
        import agents.reviewer_agent as _rev
        results = getattr(_rev, '_partial_results', [])
        yield log(f"Rate limit hit — saved {len(results)} partial result(s).", "warning"), "", ""

    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    yield log(f"Review complete: {len(accepted)} accepted, {len(rejected)} rejected.", "success"), "", ""

    # ── Revision loop ─────────────────────────────────────────────
    rejected_results = [r for r in results if not r.accepted]
    if rejected_results:
        yield log(f"Revision loop: retrying {len(rejected_results)} rejected plan(s)...", "agent"), "", ""

        seen = set()
        unique_gaps = []
        for r in rejected_results:
            if r.plan.source_gap and r.plan.source_gap.description not in seen:
                seen.add(r.plan.source_gap.description)
                unique_gaps.append(r.plan.source_gap)

        for attempt in range(config.MAX_REVISION_ATTEMPTS):
            if not unique_gaps:
                break
            yield log(f"Revision attempt {attempt+1}/{config.MAX_REVISION_ATTEMPTS}...", "info"), "", ""
            try:
                revised      = planner_agent.run(unique_gaps, papers)
                re_reviewed  = reviewer_agent.run(revised, papers)
                newly        = [r for r in re_reviewed if r.accepted]
                still        = [r for r in re_reviewed if not r.accepted]
                results     += newly
                accepted    += newly
                yield log(f"Revision {attempt+1}: {len(newly)} newly accepted.", "success"), "", ""
                accepted_descs = {r.plan.source_gap.description for r in newly if r.plan.source_gap}
                unique_gaps    = [g for g in unique_gaps if g.description not in accepted_descs]
            except Exception as e:
                yield log(f"Revision error: {e}", "warning"), "", ""
                break

    # ── Save results ──────────────────────────────────────────────
    try:
        import os, json as _json
        from datetime import datetime as _dt
        path = config.OUTPUT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {"runs": []}
        if os.path.exists(path):
            with open(path) as f:
                try:
                    existing = _json.load(f)
                    if "runs" not in existing:
                        existing = {"runs": [existing]}
                except:
                    pass
        run_entry = {
            "topic": topic,
            "timestamp": _dt.now().isoformat(),
            "summary": {
                "total_reviewed": len(results),
                "accepted": len(accepted),
                "rejected": len(results) - len(accepted),
            },
            "results": [
                {
                    "accepted": r.accepted,
                    "novelty_score": r.novelty_score,
                    "feasibility_passed": r.feasibility_passed,
                    "feasibility_notes": r.feasibility_notes,
                    "suggested_title": r.suggested_title,
                    "research_direction": r.research_direction,
                    "experimental_blueprint": r.experimental_blueprint,
                    "plan": {
                        "research_question": r.plan.research_question,
                        "proposed_method": r.plan.proposed_method,
                        "dataset": r.plan.dataset,
                        "evaluation_metric": r.plan.evaluation_metric,
                        "source_gap": r.plan.source_gap.description if r.plan.source_gap else "",
                    }
                } for r in results
            ]
        }
        existing["runs"].append(run_entry)
        with open(path, "w") as f:
            _json.dump(existing, f, indent=2)
        yield log(f"Results saved to {path} (run #{len(existing['runs'])}).", "success"), "", ""
    except Exception as e:
        yield log(f"Could not save results: {e}", "warning"), "", ""

    # ── Build output HTML ─────────────────────────────────────────
    summary_html = _build_summary(topic, results, accepted, papers, gaps)
    results_html = _build_results(accepted, rejected)
    yield "\n".join(log_lines), results_html, summary_html


# ── HTML builders ─────────────────────────────────────────────────────────────

def _build_summary(topic, results, accepted, papers, gaps):
    total    = len(results)
    n_acc    = len(accepted)
    rate     = round(100 * n_acc / total) if total else 0
    scores   = [r.novelty_score for r in accepted]
    avg_nov  = round(sum(scores) / len(scores), 2) if scores else 0

    def stat(label, value, color="#111"):
        return f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:18px 22px;text-align:center;">
          <div style="font-size:28px;font-weight:700;color:{color}">{value}</div>
          <div style="font-size:12px;color:#6b7280;margin-top:4px;font-weight:500;
                      text-transform:uppercase;letter-spacing:.05em">{label}</div>
        </div>"""

    rate_color = SUCCESS if rate >= 60 else (WARNING if rate >= 30 else DANGER)

    return f"""
    <div style="font-family:'IBM Plex Sans',system-ui,sans-serif;padding:4px 0">
      <h3 style="font-size:15px;font-weight:600;color:#111;margin:0 0 14px">
        Topic: <span style="color:{ACCENT}">{topic}</span>
      </h3>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
        {stat("Papers retrieved", len(papers))}
        {stat("Gaps identified", len(gaps))}
        {stat("Plans reviewed", total)}
        {stat("Accept rate", f"{rate}%", rate_color)}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        {stat("Accepted proposals", n_acc, SUCCESS)}
        {stat("Avg novelty (accepted)", avg_nov, ACCENT)}
      </div>
    </div>"""


def _novelty_bar(score):
    pct   = min(100, int(score * 10))
    color = SUCCESS if score >= 6 else (ACCENT if score >= 4 else WARNING)
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
      <div style="flex:1;background:#e5e7eb;border-radius:4px;height:6px">
        <div style="width:{pct}%;background:{color};height:6px;border-radius:4px"></div>
      </div>
      <span style="font-size:12px;font-weight:600;color:{color};min-width:32px">{score}</span>
    </div>"""


def _build_results(accepted, rejected):
    if not accepted and not rejected:
        return "<p style='color:#6b7280;font-size:14px'>No results yet. Run the pipeline above.</p>"

    html = "<div style=\"font-family:'IBM Plex Sans',system-ui,sans-serif\">"

    if accepted:
        html += f"<h3 style='font-size:15px;font-weight:600;color:{SUCCESS};margin:0 0 12px'>" \
                f"✓  {len(accepted)} Accepted Proposal(s)</h3>"
        for i, r in enumerate(accepted, 1):
            gap_label = r.plan.source_gap.description[:55] + "..." if r.plan.source_gap else "N/A"
            html += f"""
            <div style="border:1px solid #d1fae5;border-radius:12px;padding:18px 20px;
                        margin-bottom:14px;background:#f0fdf4">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;
                          margin-bottom:10px;gap:12px">
                <div style="font-size:14px;font-weight:600;color:#111;flex:1">
                  {i}. {r.suggested_title or r.plan.research_question[:80]}
                </div>
                <span style="background:#dcfce7;color:#16a34a;font-size:11px;font-weight:600;
                             padding:3px 10px;border-radius:20px;white-space:nowrap">
                  ACCEPTED
                </span>
              </div>

              <div style="font-size:13px;color:#374151;margin-bottom:10px;line-height:1.6">
                {r.research_direction or ""}
              </div>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                          margin-bottom:10px;font-size:12px">
                <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px">
                  <div style="color:#6b7280;font-weight:500;margin-bottom:3px">Dataset</div>
                  <div style="color:#111;font-weight:600">{r.plan.dataset}</div>
                </div>
                <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px">
                  <div style="color:#6b7280;font-weight:500;margin-bottom:3px">Metric</div>
                  <div style="color:#111;font-weight:600">{r.plan.evaluation_metric}</div>
                </div>
              </div>

              <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px;
                          margin-bottom:10px;font-size:12px;color:#374151;line-height:1.6">
                <span style="font-weight:600;color:#6b7280">Blueprint: </span>
                {r.experimental_blueprint[:300]}{"..." if len(r.experimental_blueprint) > 300 else ""}
              </div>

              <div style="font-size:12px;color:#6b7280">
                <span style="font-weight:500">Novelty score</span>
                {_novelty_bar(r.novelty_score)}
              </div>
              <div style="font-size:11px;color:#9ca3af;margin-top:6px">
                Gap: {gap_label}
              </div>
            </div>"""

    if rejected:
        html += f"<h3 style='font-size:14px;font-weight:600;color:{DANGER};" \
                f"margin:16px 0 10px'>✗  {len(rejected)} Rejected Plan(s)</h3>"
        for r in rejected:
            reasons = []
            if r.novelty_score < config.NOVELTY_THRESHOLD:
                reasons.append(f"Novelty {r.novelty_score:.2f} below threshold")
            if not r.feasibility_passed:
                short = r.feasibility_notes[:90] + "..." if len(r.feasibility_notes) > 90 else r.feasibility_notes
                reasons.append(short)
            html += f"""
            <div style="border:1px solid #fee2e2;border-radius:10px;padding:14px 16px;
                        margin-bottom:8px;background:#fff5f5">
              <div style="font-size:13px;color:#374151;font-weight:500;margin-bottom:4px">
                {r.plan.research_question[:90]}...
              </div>
              <div style="font-size:11px;color:{DANGER}">
                {" | ".join(reasons)}
              </div>
            </div>"""

    html += "</div>"
    return html


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLE_TOPICS = [
    "federated learning privacy",
    "sign language recognition Indian regional languages",
    "large language models hallucination detection",
    "vision transformers medical image segmentation",
    "knowledge graph completion low resource",
    "quantum computing error correction",
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.gr-button-primary {
    background: #2563EB !important;
    border: none !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover {
    background: #1d4ed8 !important;
}
#log-box textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
    background: #0f172a !important;
    color: #94a3b8 !important;
}
.gr-panel {
    border-radius: 12px !important;
}
"""

HEADER_HTML = """
<div style="font-family:'IBM Plex Sans',system-ui,sans-serif;
            padding:28px 0 20px;border-bottom:2px solid #e5e7eb;margin-bottom:24px">
  <div style="font-size:11px;font-weight:600;letter-spacing:.12em;color:#2563EB;
              text-transform:uppercase;margin-bottom:8px">
    MSc AI &amp; Machine Learning — Major Project
  </div>
  <h1 style="font-size:28px;font-weight:700;color:#0f172a;margin:0 0 8px;line-height:1.2">
    Agentic Research Planning Framework
  </h1>
  <p style="font-size:14px;color:#6b7280;margin:0;max-width:620px;line-height:1.6">
    Enter a research topic to automatically retrieve papers from arXiv,
    identify research gaps, generate candidate plans, and evaluate them
    for novelty and feasibility.
  </p>
</div>
"""

with gr.Blocks(title="Agentic Research Planner") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=3):
            topic_input = gr.Textbox(
                label="Research Topic",
                placeholder='e.g. "federated learning privacy" or "sign language recognition"',
                lines=1,
            )
        with gr.Column(scale=1, min_width=140):
            run_btn = gr.Button("▶  Run Pipeline", variant="primary", size="lg")

    gr.Examples(
        examples=[[t] for t in EXAMPLE_TOPICS],
        inputs=topic_input,
        label="Example topics",
    )

    summary_out = gr.HTML(label="Summary", value="")

    with gr.Row():
        with gr.Column(scale=2):
            results_out = gr.HTML(
                label="Proposals",
                value="<p style='color:#9ca3af;font-size:13px;padding:12px 0'>"
                      "Results will appear here after the pipeline runs.</p>"
            )
        with gr.Column(scale=1):
            log_out = gr.Textbox(
                label="Pipeline log",
                lines=28,
                interactive=False,
                elem_id="log-box",
            )

    run_btn.click(
        fn=run_pipeline_streaming,
        inputs=topic_input,
        outputs=[log_out, results_out, summary_out],
    )

    gr.HTML("""
    <div style="font-family:'IBM Plex Sans',system-ui,sans-serif;
                border-top:1px solid #e5e7eb;padding:16px 0 4px;margin-top:16px;
                display:flex;justify-content:space-between;align-items:center">
      <span style="font-size:12px;color:#9ca3af">
        Agents: Librarian → Planner → Reviewer &nbsp;|&nbsp;
        LLM: Groq llama-3.3-70b-versatile &nbsp;|&nbsp;
        Embeddings: all-MiniLM-L6-v2
      </span>
      <span style="font-size:12px;color:#9ca3af">
        Novelty threshold: {threshold} &nbsp;|&nbsp; Max papers: {papers}
      </span>
    </div>
    """.format(threshold=config.NOVELTY_THRESHOLD, papers=config.MAX_PAPERS))


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )
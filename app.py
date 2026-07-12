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
from datetime import datetime
from typing import Generator

import config
from agents import librarian_agent, planner_agent, reviewer_agent
from main import save_results, _rejection_reasons
from models.schemas import ReviewResult
from tools.feasibility_checker import get_topic_datasets
from tools.pdf_exporter import generate_pdf
from tools.topic_validator import validate_topic


# ── Colour constants ──────────────────────────────────────────────────────────
ACCENT   = "#2563EB"
SUCCESS  = "#16A34A"
WARNING  = "#D97706"
DANGER   = "#DC2626"
MUTED    = "#6B7280"

# ── State: holds last completed run for PDF download ──────────────────────────
_last_run: dict = {}  # keys: results, papers, gaps, topic, run_number


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
        yield "Please enter a research topic.", "", "", gr.update(visible=False)
        return

    log_lines = []

    def log(msg, level="info"):
        log_lines.append(format_log(msg, level))
        return "\n".join(log_lines)

    # ── Topic scope validation ────────────────────────────────────
    yield log(f"Validating topic scope: '{topic}'", "info"), "", "", gr.update(visible=False)
    validation = validate_topic(topic)
    if not validation["in_scope"]:
        reason = validation["reason"]
        suggestion = validation.get("suggested_topic", "")
        suggestion_html = (
            f"<p style='margin:10px 0 0;font-size:13px;color:#374151'>"
            f"Try instead: <b>{suggestion}</b></p>"
            if suggestion else ""
        )
        error_html = f"""
        <div style="font-family:'IBM Plex Sans',system-ui,sans-serif;
                    border:1px solid #fecaca;border-radius:12px;padding:20px 24px;
                    background:#fff5f5;margin:8px 0">
          <div style="font-size:15px;font-weight:600;color:#dc2626;margin-bottom:8px">
            Topic out of scope
          </div>
          <p style="font-size:13px;color:#374151;margin:0">{reason}</p>
          {suggestion_html}
          <p style="font-size:12px;color:#9ca3af;margin:12px 0 0">
            This tool is scoped to Computer Science and Machine Learning research.
            Please enter a CS/ML topic to continue.
          </p>
        </div>"""
        yield log(f"Topic rejected: {reason}", "error"), error_html, "", gr.update(visible=False)
        return

    yield log("Topic validated — within CS/ML scope.", "success"), "", "", gr.update(visible=False)

    # ── Step 1+2: Librarian ───────────────────────────────────────
    yield log(f"Starting pipeline for: '{topic}'"), "", "", gr.update(visible=False)
    yield log("Librarian Agent — fetching papers from arXiv...", "agent"), "", "", gr.update(visible=False)

    try:
        papers, gaps = librarian_agent.run(topic)
    except Exception as e:
        yield log(f"Pipeline error: {e}", "error"), "", "", gr.update(visible=False)

        return

    yield log(f"Retrieved {len(papers)} papers. Concepts extracted.", "success"), "", "", gr.update(visible=False)

    yield log(f"Identified {len(gaps)} research gap(s).", "success"), "", "", gr.update(visible=False)


    if not gaps:
        yield log("No gaps found. Try a more specific topic.", "warning"), "", "", gr.update(visible=False)

        return

    for g in gaps:
        yield log(f"Gap: {g.description[:70]}...", "info"), "", "", gr.update(visible=False)

    # ── Dataset grounding: verified HF Hub datasets for this topic ──
    suggested_datasets = get_topic_datasets(topic)
    if suggested_datasets:
        yield log(f"HF Hub datasets found: {', '.join(suggested_datasets[:4])}...", "info"), "", "", gr.update(visible=False)
    else:
        yield log("No HF Hub datasets for topic — using generic benchmarks.", "info"), "", "", gr.update(visible=False)

    # ── Step 3: Planner ───────────────────────────────────────────
    yield log("Planner Agent — generating candidate research plans...", "agent"), "", "", gr.update(visible=False)

    try:
        plans = planner_agent.run(gaps, papers, suggested_datasets=suggested_datasets)
    except Exception as e:
        yield log(f"Planner error: {e}", "error"), "", "", gr.update(visible=False)

        return

    yield log(f"Generated {len(plans)} candidate plan(s) across {len(gaps)} gap(s).", "success"), "", "", gr.update(visible=False)


    # ── Step 4+5: Reviewer ────────────────────────────────────────
    yield log("Reviewer Agent — scoring novelty and checking feasibility...", "agent"), "", "", gr.update(visible=False)

    rate_limited = False
    try:
        results = reviewer_agent.run(plans, papers)
    except Exception as e:
        results = getattr(reviewer_agent, '_partial_results', [])
        if "rate_limit" in str(e).lower() or "429" in str(e):
            rate_limited = True
            yield log(f"Rate limit hit — recovered {len(results)} partial result(s).", "warning"), "", "", gr.update(visible=False)
        else:
            yield log(f"Reviewer error: {e} — recovered {len(results)} partial result(s).", "error"), "", "", gr.update(visible=False)
        if not results:
            return

    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    yield log(f"Review complete: {len(accepted)} accepted, {len(rejected)} rejected.", "success"), "", "", gr.update(visible=False)


    # ── Revision loop ─────────────────────────────────────────────
    # Skipped after a rate limit — further LLM calls would fail too.
    rejected_results = [] if rate_limited else rejected
    if rejected_results:
        yield log(f"Revision loop: retrying {len(rejected_results)} rejected plan(s)...", "agent"), "", "", gr.update(visible=False)


        seen = set()
        unique_gaps = []
        rejection_feedback: dict = {}  # gap description -> [reason strings]
        for r in rejected_results:
            if r.plan.source_gap:
                desc = r.plan.source_gap.description
                if desc not in seen:
                    seen.add(desc)
                    unique_gaps.append(r.plan.source_gap)
                rejection_feedback.setdefault(desc, []).extend(_rejection_reasons(r))

        for attempt in range(config.MAX_REVISION_ATTEMPTS):
            if not unique_gaps:
                break
            yield log(f"Revision attempt {attempt+1}/{config.MAX_REVISION_ATTEMPTS}...", "info"), "", "", gr.update(visible=False)
            try:
                revised      = planner_agent.run(unique_gaps, papers,
                                                 suggested_datasets=suggested_datasets,
                                                 rejection_feedback=rejection_feedback)
                re_reviewed  = reviewer_agent.run(revised, papers)
                newly        = [r for r in re_reviewed if r.accepted]
                still        = [r for r in re_reviewed if not r.accepted]
                results     += newly
                accepted    += newly
                yield log(f"Revision {attempt+1}: {len(newly)} newly accepted.", "success"), "", "", gr.update(visible=False)

                accepted_descs = {r.plan.source_gap.description for r in newly if r.plan.source_gap}
                unique_gaps    = [g for g in unique_gaps if g.description not in accepted_descs]
                for r in still:
                    if r.plan.source_gap:
                        rejection_feedback.setdefault(r.plan.source_gap.description, []).extend(_rejection_reasons(r))
            except Exception as e:
                yield log(f"Revision error: {e}", "warning"), "", "", gr.update(visible=False)
                break

    # ── Save results (same format and path as the CLI pipeline) ──
    try:
        path, run_num = save_results(results, topic)
        _last_run["results"]    = results
        _last_run["papers"]     = papers
        _last_run["gaps"]       = gaps
        _last_run["topic"]      = topic
        _last_run["run_number"] = run_num
        yield log(f"Results saved to {path} (run #{run_num}).", "success"), "", "", gr.update(visible=False)

    except Exception as e:
        yield log(f"Could not save results: {e}", "warning"), "", "", gr.update(visible=False)

    # ── Build output HTML ─────────────────────────────────────────
    summary_html = _build_summary(topic, results, accepted, papers, gaps)
    results_html = _build_results(accepted, rejected)
    yield "\n".join(log_lines), results_html, summary_html, gr.update(visible=True)


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
            if r.novelty_score < r.novelty_threshold:
                reasons.append(f"Novelty {r.novelty_score:.2f} below threshold {r.novelty_threshold:.2f}")
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




# ── PDF generation ────────────────────────────────────────────────────────────

def generate_pdf_report():
    """Called when the user clicks Download PDF. Generates and returns the file path."""
    if not _last_run:
        return None
    try:
        path = generate_pdf(
            topic      = _last_run["topic"],
            results    = _last_run["results"],
            papers     = _last_run["papers"],
            gaps       = _last_run["gaps"],
            run_number = _last_run.get("run_number", 1),
            output_dir = "outputs",
        )
        return path
    except Exception as e:
        print(f"[pdf] Error generating PDF: {e}")
        return None

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
    Enter a <b>Computer Science or Machine Learning</b> research topic to automatically
    retrieve papers from arXiv, identify research gaps, generate candidate plans,
    and evaluate them for novelty and feasibility.
  </p>
</div>
"""

with gr.Blocks(title="Agentic Research Planner") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=3):
            topic_input = gr.Textbox(
                label="Research Topic (CS / ML only)",
                placeholder='e.g. "federated learning privacy" or "vision transformers medical imaging"',
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
        download_btn = gr.Button(
            "⬇  Download PDF Report",
            variant="secondary",
            size="sm",
            visible=False,
        )
        pdf_file = gr.File(label="PDF Report", visible=False, interactive=False)

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
        outputs=[log_out, results_out, summary_out, download_btn],
    )

    def _do_pdf():
        path = generate_pdf_report()
        if path:
            return gr.update(value=path, visible=True)
        return gr.update(visible=False)

    download_btn.click(
        fn=_do_pdf,
        inputs=None,
        outputs=pdf_file,
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
        Novelty threshold: computed per-run from corpus &nbsp;|&nbsp; Max papers: {papers}
      </span>
    </div>
    """.format(papers=config.MAX_PAPERS))


if __name__ == "__main__":
    # Gradio 6.x takes css on launch(); on Gradio <6 it belongs on
    # gr.Blocks(...) instead.
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )
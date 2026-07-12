"""
agents/reviewer_agent.py
-------------------------
Reviewer Agent for the Agentic Research Planning Framework.

Responsibilities:
    1. Score each candidate plan for novelty using sentence embeddings
       and cosine similarity against retrieved paper abstracts.
    2. Validate each plan for feasibility using rule-based checks
       on dataset names and evaluation metrics.
    3. For plans that pass both checks, call the LLM to generate a
       suggested research title and experimental blueprint.
    4. Assemble and return a list of ReviewResult objects.
    5. Accept a plan only if novelty_score >= the corpus-derived novelty
       threshold for this run (see compute_novelty_threshold) AND
       feasibility passes.

Usage:
    from agents.reviewer_agent import run
    results = run(plans, papers)
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import RateLimitError

import config
from models.schemas import Paper, ResearchPlan, ReviewResult
from tools.embedder import get_embeddings
from tools.feasibility_checker import check_feasibility
from tools.llm import call_llm, parse_json


# -- Module-level partial results store
# Updated during run() so main.py can recover results if a rate limit
# exception interrupts the loop mid-execution (see thesis §4.7)
_partial_results: List[ReviewResult] = []


# -- Internal helpers

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Make a single Groq API call via the shared LLM client."""
    return call_llm(system_prompt, user_prompt)


def _parse_json(raw: str, label: str):
    """Parse LLM JSON output via the shared helper. Returns None on failure."""
    return parse_json(raw, label, tag="reviewer_agent")


# -- Phase A upgrade: score_novelty with top-K discrimination

def score_novelty(plan: ResearchPlan, papers: List[Paper],
                  abstract_embs: Optional[np.ndarray] = None) -> float:
    """
    Score a research plan's novelty against existing paper abstracts.

    Phase A upgrade: uses a top-K similarity approach rather than mean
    across all papers. This makes the scorer more discriminative —
    a plan is penalised if it is too similar to ANY of the K most
    similar papers, not just the average paper in the corpus.

    Scoring uses a weighted blend of method and question embeddings:
        - 70% proposed_method   (most distinctive, least domain-generic)
        - 30% research_question (broader context signal)

    Top-K formula:
        1. Compute weighted similarity against all retrieved abstracts
        2. Select the K highest similarity scores (K = TOP_K_PAPERS)
        3. Take the mean of those K scores (worst-case neighbourhood)
        4. novelty_score = (1 - topk_mean) * 10, clamped to [0.0, 10.0]

    Acceptance is decided against the corpus-derived per-run threshold
    (see compute_novelty_threshold), not a fixed band. As reference
    points: ~0 means near-copy of existing work, mid-range (3–6) means
    meaningful differentiation, and high scores (8+) indicate strong
    conceptual distance from the retrieved corpus.

    Args:
        plan          : The ResearchPlan to score.
        papers        : Retrieved papers used as the novelty comparison baseline.
        abstract_embs : Optional precomputed abstract embeddings, shape (n, 384).
                        Passed by run() so the corpus is embedded once per run
                        instead of once per plan. Computed here if omitted.

    Returns:
        Float novelty score in range [0.0, 10.0].
    """
    if abstract_embs is None:
        abstract_embs = get_embeddings([p.abstract for p in papers])  # (n, 384)

    # Embed method and question together in one encoder call
    plan_embs    = get_embeddings([plan.proposed_method, plan.research_question])
    method_emb   = plan_embs[0:1]                                    # shape: (1, 384)
    question_emb = plan_embs[1:2]                                    # shape: (1, 384)

    method_sims   = cosine_similarity(method_emb,   abstract_embs)  # shape: (1, n)
    question_sims = cosine_similarity(question_emb, abstract_embs)  # shape: (1, n)

    # Weighted similarity: method contributes more as the distinctive signal
    weighted_sims = 0.7 * method_sims + 0.3 * question_sims         # shape: (1, n)

    # Top-K: select the K most similar papers and score against those
    k = min(config.TOP_K_PAPERS, len(papers))
    top_k_sims = np.sort(weighted_sims[0])[-k:]                     # K highest values
    topk_mean  = float(top_k_sims.mean())

    score = round((1.0 - topk_mean) * 10, 2)
    return max(0.0, min(10.0, score))                               # clamp to valid range


# -- Percentile-based threshold computation

def compute_novelty_threshold(papers: List[Paper],
                              abstract_embs: Optional[np.ndarray] = None) -> float:
    """
    Compute a principled novelty threshold from the retrieved paper corpus.

    Rather than using a fixed arbitrary threshold, this function derives
    the threshold from the actual corpus used in each run. It measures
    how similar existing papers are to each other (inter-paper similarity),
    then sets the acceptance threshold at the 25th percentile of that
    baseline distribution.

    Rationale: a generated research plan must be more novel than at least
    75% of existing papers are relative to each other. This anchors the
    threshold to the specific domain and corpus, not to a hand-tuned number.

    Algorithm:
        1. For each paper, embed its abstract.
        2. Compute top-K similarity against all other papers in the corpus.
        3. Convert to novelty score: (1 - topk_mean) * 10.
        4. Collect all inter-paper novelty scores into a baseline distribution.
        5. Return the 25th percentile as the threshold.

    Args:
        papers        : Retrieved papers from the Librarian Agent.
        abstract_embs : Optional precomputed abstract embeddings, shape (n, 384).
                        Passed by run() so the corpus is embedded once per run.
                        Computed here if omitted.

    Returns:
        Float threshold in range [0.0, 10.0].
        Falls back to config.NOVELTY_THRESHOLD_STRICT if computation fails.
    """
    if len(papers) < 3:
        print(f"[reviewer_agent] WARNING: Too few papers ({len(papers)}) for "
              f"percentile threshold — using fallback {config.NOVELTY_THRESHOLD_STRICT}")
        return config.NOVELTY_THRESHOLD_STRICT

    try:
        if abstract_embs is None:
            abstract_embs = get_embeddings([p.abstract for p in papers])  # (n, 384)
        k = min(config.TOP_K_PAPERS, len(papers) - 1)

        baseline_scores = []
        for i in range(len(papers)):
            # Build similarity matrix excluding the paper itself
            other_embs = np.delete(abstract_embs, i, axis=0)  # shape: (n-1, 384)
            sims       = cosine_similarity(
                abstract_embs[i].reshape(1, -1), other_embs
            )                                                   # shape: (1, n-1)
            top_k_sims  = np.sort(sims[0])[-k:]
            topk_mean   = float(top_k_sims.mean())
            score       = round((1.0 - topk_mean) * 10, 2)
            baseline_scores.append(max(0.0, min(10.0, score)))

        threshold = float(np.percentile(baseline_scores, 25))
        threshold = round(max(1.0, min(8.0, threshold)), 2)   # safety clamp

        print(f"[reviewer_agent] Novelty threshold computed from corpus "
              f"(25th percentile of {len(papers)} papers): {threshold:.2f}")
        print(f"[reviewer_agent] Baseline score range: "
              f"min={min(baseline_scores):.2f}, "
              f"max={max(baseline_scores):.2f}, "
              f"mean={sum(baseline_scores)/len(baseline_scores):.2f}")

        return threshold

    except Exception as e:
        print(f"[reviewer_agent] WARNING: Threshold computation failed: {e}. "
              f"Using fallback {config.NOVELTY_THRESHOLD_STRICT}")
        return config.NOVELTY_THRESHOLD_STRICT


# -- Task 4.4: generate_output_fields

def generate_output_fields(plan: ResearchPlan) -> tuple[str, str, str]:
    """
    Generate presentational output fields for an accepted research plan.

    Called ONLY after a plan passes both novelty and feasibility checks,
    to avoid wasting API calls on rejected plans.

    Returns a tuple of (suggested_title, research_direction, experimental_blueprint).
    Falls back to placeholder strings if the LLM call fails.

    Args:
        plan: A ResearchPlan that has passed all checks.

    Returns:
        Tuple of (suggested_title: str, research_direction: str,
                  experimental_blueprint: str).
    """
    system_prompt = (
        "You are a research proposal writer. "
        "Given a research plan, generate three short output fields. "
        "Return ONLY a valid JSON object with exactly these keys: "
        "'suggested_title' (a concise, specific research paper title, string), "
        "'research_direction' (1-2 sentences describing the research opportunity, string), "
        "'experimental_blueprint' (3-5 sentences describing how the idea could be tested, string). "
        "No extra keys, no markdown."
    )
    user_prompt = (
        f"Research question: {plan.research_question}\n"
        f"Proposed method: {plan.proposed_method}\n"
        f"Dataset: {plan.dataset}\n"
        f"Evaluation metric: {plan.evaluation_metric}"
    )

    raw    = _call_llm(system_prompt, user_prompt)
    parsed = _parse_json(raw, f"generate_output_fields:{plan.research_question[:40]}")

    if parsed and all(k in parsed for k in ("suggested_title", "research_direction", "experimental_blueprint")):
        return (
            str(parsed["suggested_title"]),
            str(parsed["research_direction"]),
            str(parsed["experimental_blueprint"]),
        )

    # Fallback — do not crash if LLM output is malformed
    print(f"[reviewer_agent] WARNING: Could not generate output fields for plan. Using fallbacks.")
    return (
        plan.research_question[:80],
        f"Investigate: {plan.research_question}",
        f"Apply {plan.proposed_method} on {plan.dataset}, evaluated by {plan.evaluation_metric}.",
    )


# -- Task 4.5 + 4.6: run

def run(plans: List[ResearchPlan], papers: List[Paper]) -> List[ReviewResult]:
    """
    Full Reviewer Agent pipeline: score novelty, check feasibility, generate outputs.

    For each plan:
        1. Score novelty via embedding cosine similarity.
        2. Check feasibility via rule-based dataset and metric validation.
        3. Mark as accepted only if BOTH checks pass.
        4. Call LLM for title and blueprint ONLY for accepted plans.
        5. Assemble a ReviewResult for every plan (accepted or rejected).

    Args:
        plans  : Candidate ResearchPlan objects from the Planner Agent.
        papers : Retrieved papers used as the novelty comparison baseline.

    Returns:
        List[ReviewResult] — one per input plan, accepted or rejected.
    """
    results: List[ReviewResult] = []
    # Expose partial results at module level so main.py can
    # recover them if a rate limit exception interrupts the loop
    global _partial_results
    _partial_results = results

    # Embed the corpus once per run — reused by the threshold computation
    # and by every score_novelty() call below.
    abstract_embs = get_embeddings([p.abstract for p in papers]) if papers else None

    # Compute principled percentile-based novelty threshold from the corpus.
    # This replaces the fixed NOVELTY_THRESHOLD_STRICT with a data-driven
    # value anchored to inter-paper similarity in the retrieved corpus.
    novelty_threshold = compute_novelty_threshold(papers, abstract_embs)

    print(f"[reviewer_agent] Reviewing {len(plans)} plan(s)...")

    for i, plan in enumerate(plans):
        print(f"\n[reviewer_agent] Plan {i+1}/{len(plans)}: '{plan.research_question[:55]}...'")

        # Step 1: Novelty scoring (top-K embedding-based, Phase A upgrade)
        novelty_score = score_novelty(plan, papers, abstract_embs)
        novelty_ok    = novelty_score >= novelty_threshold
        print(f"[reviewer_agent]   Novelty score   : {novelty_score:.2f} "
              f"(threshold={novelty_threshold:.2f}) "
              f"{'PASS' if novelty_ok else 'FAIL - below threshold'}")

        # Step 2: Feasibility check (5-component, Phase B upgrade)
        feasibility_ok, feasibility_notes, feasibility_components = check_feasibility(plan)
        print(f"[reviewer_agent]   Feasibility     : {'PASS' if feasibility_ok else 'FAIL'}")
        if not feasibility_ok:
            # Show which components failed
            for comp_name, (comp_ok, comp_note) in feasibility_components.items():
                if not comp_ok:
                    label = comp_name.replace("_", " ").title()
                    print(f"[reviewer_agent]     x {label}: {comp_note[:80]}")
        else:
            # Show component pass summary in compact form
            comp_labels = {
                "data_availability":         "Data",
                "metric_validity":           "Metric",
                "computational_feasibility": "Compute",
                "methodological_practicality":"Method",
                "time_feasibility":          "Time",
            }
            passed_labels = " | ".join(
                comp_labels[k] for k in comp_labels
                if feasibility_components.get(k, (True, ""))[0]
            )
            print(f"[reviewer_agent]   Components PASS : {passed_labels}")

        # Step 3: Accept only if both checks pass
        accepted = novelty_ok and feasibility_ok

        # Step 4: Generate title + blueprint ONLY for accepted plans
        if accepted:
            print(f"[reviewer_agent]   Generating title and blueprint...")
            try:
                suggested_title, research_direction, experimental_blueprint = \
                    generate_output_fields(plan)
            except RateLimitError as e:
                print(f"[reviewer_agent] WARNING: Rate limit hit during output generation.")
                print(f"[reviewer_agent] Saving {len(results)} result(s) collected so far...")
                # Mark this plan as accepted but with empty output fields
                # so partial results are not lost
                suggested_title        = plan.research_question[:80]
                research_direction     = f"Investigate: {plan.research_question}"
                experimental_blueprint = f"Apply {plan.proposed_method} on {plan.dataset}, evaluated by {plan.evaluation_metric}."
                results.append(ReviewResult(
                    plan=plan,
                    novelty_score=novelty_score,
                    novelty_threshold=novelty_threshold,
                    feasibility_passed=feasibility_ok,
                    feasibility_notes=feasibility_notes,
                    feasibility_components=feasibility_components,
                    suggested_title=suggested_title,
                    research_direction=research_direction,
                    experimental_blueprint=experimental_blueprint,
                    accepted=accepted,
                ))
                raise  # re-raise so main.py can catch and save
        else:
            suggested_title        = ""
            research_direction     = ""
            experimental_blueprint = ""

        print(f"[reviewer_agent]   Decision        : {'ACCEPTED' if accepted else 'REJECTED'}")

        results.append(ReviewResult(
            plan=plan,
            novelty_score=novelty_score,
            novelty_threshold=novelty_threshold,
            feasibility_passed=feasibility_ok,
            feasibility_notes=feasibility_notes,
            feasibility_components=feasibility_components,
            suggested_title=suggested_title,
            research_direction=research_direction,
            experimental_blueprint=experimental_blueprint,
            accepted=accepted,
        ))

    accepted_count = sum(1 for r in results if r.accepted)
    print(f"\n[reviewer_agent] Review complete. "
          f"{accepted_count}/{len(plans)} plan(s) accepted.")

    return results
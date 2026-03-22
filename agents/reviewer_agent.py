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
    5. Accept a plan only if novelty_score >= NOVELTY_THRESHOLD AND
       feasibility passes.

Usage:
    from agents.reviewer_agent import run
    results = run(plans, papers)
"""

import json
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

import config
from models.schemas import Paper, ResearchPlan, ReviewResult
from tools.embedder import get_embeddings
from tools.feasibility_checker import check_feasibility


# -- Groq client — initialised once at module level
_client = Groq(api_key=config.GROQ_API_KEY)


# -- Internal helpers

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Make a single Groq API call and return raw response text."""
    response = _client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )
    return response.choices[0].message.content


def _parse_json(raw: str, label: str) -> any:
    """Parse JSON from LLM response. Logs warning and returns None on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[reviewer_agent] WARNING: JSON parse failed for '{label}'. Error: {e}")
        return None


# -- Task 4.3: score_novelty

def score_novelty(plan: ResearchPlan, papers: List[Paper]) -> float:
    """
    Score a research plan's novelty against existing paper abstracts.

    Uses the proposed_method as the primary signal — this is the most
    distinctive part of a plan and the least likely to share vocabulary
    with abstracts simply due to domain terminology.

    Scoring uses a weighted blend:
        - 70% based on proposed_method similarity (most discriminating)
        - 30% based on research_question similarity (broader context)

    Uses MEAN similarity rather than MAX to avoid penalising plans for
    being in the same domain as a single highly-similar paper.

    Formula: novelty_score = (1 - weighted_mean_similarity) * 10
    Clamped to [0.0, 10.0].

    Score interpretation (domain-specific, calibrated):
        0.0 – 1.4 : Near-copy of existing abstract. Rejected.
        1.5 – 2.9 : On-topic with novel combination. Accepted.
        3.0 – 5.0 : Good conceptual distance. Accepted.
        5.0+      : Strongly differentiated. Accepted.

    Args:
        plan   : The ResearchPlan to score.
        papers : Retrieved papers used as the novelty comparison baseline.

    Returns:
        Float novelty score in range [0.0, 10.0].
    """
    abstract_emb = get_embeddings([p.abstract for p in papers])   # shape: (n, 384)

    # Embed method and question separately for weighted scoring
    method_emb   = get_embeddings([plan.proposed_method])          # shape: (1, 384)
    question_emb = get_embeddings([plan.research_question])        # shape: (1, 384)

    method_sims   = cosine_similarity(method_emb, abstract_emb)    # shape: (1, n)
    question_sims = cosine_similarity(question_emb, abstract_emb)  # shape: (1, n)

    # Weighted mean similarity: method weighted higher as more discriminating
    weighted_sims = 0.7 * method_sims + 0.3 * question_sims        # shape: (1, n)
    mean_sim = float(weighted_sims.mean())

    score = round((1.0 - mean_sim) * 10, 2)
    return max(0.0, min(10.0, score))   # clamp to valid range


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

    print(f"[reviewer_agent] Reviewing {len(plans)} plan(s)...")

    for i, plan in enumerate(plans):
        print(f"\n[reviewer_agent] Plan {i+1}/{len(plans)}: '{plan.research_question[:55]}...'")

        # Step 1: Novelty scoring (embedding-based)
        novelty_score = score_novelty(plan, papers)
        novelty_ok    = novelty_score >= config.NOVELTY_THRESHOLD
        print(f"[reviewer_agent]   Novelty score   : {novelty_score:.2f} "
              f"({'PASS' if novelty_ok else 'FAIL - below threshold'})")

        # Step 2: Feasibility check (rule-based)
        feasibility_ok, feasibility_notes = check_feasibility(plan)
        print(f"[reviewer_agent]   Feasibility     : {'PASS' if feasibility_ok else 'FAIL'}")
        if not feasibility_ok:
            print(f"[reviewer_agent]   Feasibility notes: {feasibility_notes}")

        # Step 3: Accept only if both checks pass
        accepted = novelty_ok and feasibility_ok

        # Step 4: Generate title + blueprint ONLY for accepted plans
        if accepted:
            print(f"[reviewer_agent]   Generating title and blueprint...")
            suggested_title, research_direction, experimental_blueprint = \
                generate_output_fields(plan)
        else:
            suggested_title        = ""
            research_direction     = ""
            experimental_blueprint = ""

        print(f"[reviewer_agent]   Decision        : {'ACCEPTED' if accepted else 'REJECTED'}")

        results.append(ReviewResult(
            plan=plan,
            novelty_score=novelty_score,
            feasibility_passed=feasibility_ok,
            feasibility_notes=feasibility_notes,
            suggested_title=suggested_title,
            research_direction=research_direction,
            experimental_blueprint=experimental_blueprint,
            accepted=accepted,
        ))

    accepted_count = sum(1 for r in results if r.accepted)
    print(f"\n[reviewer_agent] Review complete. "
          f"{accepted_count}/{len(plans)} plan(s) accepted.")

    return results

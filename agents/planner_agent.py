"""
agents/planner_agent.py
------------------------
Planner Agent for the Agentic Research Planning Framework.

Responsibilities:
    1. For each research gap, select the most relevant paper abstracts as context.
    2. Call the LLM to generate exactly 3 candidate research plans per gap.
    3. Validate that every plan has all 4 required fields.
    4. Re-prompt once if the LLM returns fewer than 3 valid plans.
    5. Return a flat list of all ResearchPlan objects across all gaps.

Usage:
    from agents.planner_agent import run
    plans = run(gaps, papers)
"""

import json
from typing import List

from groq import Groq

import config
from models.schemas import Paper, ResearchGap, ResearchPlan


# -- Groq client — initialised once at module level
_client = Groq(api_key=config.GROQ_API_KEY)

# Required fields every plan must contain
_REQUIRED_FIELDS = {"research_question", "proposed_method", "dataset", "evaluation_metric"}


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
    """
    Parse JSON from LLM response. Logs warning and returns None on failure.
    Never crashes the pipeline.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[planner_agent] WARNING: JSON parse failed for '{label}'. Error: {e}")
        print(f"[planner_agent] Raw response: {raw[:200]}")
        return None


def _is_valid_plan(plan_dict: dict) -> bool:
    """
    Check that a plan dictionary has all 4 required fields, each non-empty.
    """
    return (
        isinstance(plan_dict, dict)
        and _REQUIRED_FIELDS.issubset(plan_dict.keys())
        and all(str(plan_dict[f]).strip() for f in _REQUIRED_FIELDS)
    )


def _dict_to_plan(plan_dict: dict, gap: ResearchGap) -> ResearchPlan:
    """Convert a validated plan dictionary into a ResearchPlan dataclass."""
    return ResearchPlan(
        research_question=str(plan_dict["research_question"]).strip(),
        proposed_method=str(plan_dict["proposed_method"]).strip(),
        dataset=str(plan_dict["dataset"]).strip(),
        evaluation_metric=str(plan_dict["evaluation_metric"]).strip(),
        source_gap=gap,
    )


# -- Task 3.3: select_context_papers

def select_context_papers(gap: ResearchGap, papers: List[Paper],
                          n: int = None) -> List[Paper]:
    """
    Select the n most relevant papers for a given gap by concept overlap.

    Scores each paper by counting how many of its concepts appear in the
    gap's related_concepts list. Returns the top-n papers sorted by score.
    Case-insensitive matching to handle capitalisation differences from the LLM.

    Args:
        gap    : The ResearchGap to find context for.
        papers : Full list of retrieved papers with concepts populated.
        n      : Number of papers to return. Defaults to PLANNER_CONTEXT_PAPERS.

    Returns:
        List of the most relevant Paper objects (length <= n).
    """
    if n is None:
        n = config.PLANNER_CONTEXT_PAPERS

    gap_concepts_lower = {c.lower() for c in gap.related_concepts}

    scores = []
    for paper in papers:
        paper_concepts_lower = {c.lower() for c in paper.concepts}
        overlap = len(gap_concepts_lower & paper_concepts_lower)
        scores.append((overlap, paper))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scores[:n]]


# -- Task 3.1 + 3.2: generate_plans

def generate_plans(gap: ResearchGap, context_papers: List[Paper]) -> List[ResearchPlan]:
    """
    Generate exactly 3 candidate research plans for a given research gap.

    Sends the gap description, related concepts, and selected paper abstracts
    to the LLM. If fewer than 3 valid plans are returned, re-prompts once
    with an explicit correction request before giving up.

    Args:
        gap            : The ResearchGap to generate plans for.
        context_papers : Most relevant papers to use as grounding context.

    Returns:
        List of ResearchPlan objects. May contain fewer than 3 if the LLM
        repeatedly fails to return valid structured output.
    """
    system_prompt = (
        "You are a research planning assistant. "
        "Given a research gap and relevant paper abstracts, generate exactly 3 "
        "candidate research plans. "
        "Return ONLY a valid JSON object with a single key 'plans' whose value is "
        "an array of exactly 3 objects. Each object must contain these keys: "
        "'research_question' (string), "
        "'proposed_method' (string), "
        "'dataset' (string), "
        "'evaluation_metric' (string). "
        "Each plan must propose a concrete method - do not simply restate the gap. "
        "CRITICAL RULE FOR DATASET: The 'dataset' field must contain ONLY the "
        "short official name of a real, publicly available benchmark dataset — "
        "for example: 'CIFAR-10', 'MNIST', 'ImageNet', 'SQuAD', 'COCO', "
        "'FEMNIST', 'Penn Treebank', 'MovieLens', 'MIMIC-III'. "
        "Do NOT describe how the dataset will be used, partitioned, or modified. "
        "Do NOT write phrases like 'Non-IID CIFAR-10', 'MNIST with noise', "
        "'a custom dataset', or 'benchmark datasets with injected attacks'. "
        "Just write the base dataset name. Describe the experimental setup in "
        "'proposed_method' instead. "
        "No explanation, no extra keys, no markdown."
    )

    context_abstracts = "\n\n".join(
        f"Paper: {p.title}\nAbstract: {p.abstract[:400]}"
        for p in context_papers
    )

    user_prompt = (
        f"Research gap: {gap.description}\n\n"
        f"Related concepts: {gap.related_concepts}\n\n"
        f"Relevant abstracts:\n{context_abstracts}"
    )

    def _attempt(prompt_system, prompt_user) -> List[ResearchPlan]:
        raw = _call_llm(prompt_system, prompt_user)
        parsed = _parse_json(raw, f"generate_plans:{gap.description[:40]}")
        if not parsed:
            return []

        # Handle both {"plans": [...]} and a bare list fallback
        plan_list = parsed.get("plans", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(plan_list, list):
            return []

        valid = []
        for item in plan_list:
            if _is_valid_plan(item):
                valid.append(_dict_to_plan(item, gap))
            else:
                missing = _REQUIRED_FIELDS - set(item.keys()) if isinstance(item, dict) else _REQUIRED_FIELDS
                print(f"[planner_agent] WARNING: Plan discarded - missing fields: {missing}")
        return valid

    # First attempt
    plans = _attempt(system_prompt, user_prompt)

    # Task 3.1: re-prompt once if fewer than 3 valid plans returned
    if len(plans) < config.MAX_PLANS_PER_GAP:
        print(f"[planner_agent] Only {len(plans)} valid plan(s) returned. Re-prompting...")
        retry_system = system_prompt + (
            " IMPORTANT: Your previous response did not contain exactly 3 valid plans. "
            "You must return a JSON object with key 'plans' containing exactly 3 items, "
            "each with all 4 required fields. "
            "Remember: 'dataset' must be a short official benchmark name only "
            "(e.g. 'CIFAR-10', 'FEMNIST', 'MNIST') — no descriptions, no modifications. "
            "'evaluation_metric' must be a standard metric name only "
            "(e.g. 'accuracy', 'F1', 'AUC', 'attack success rate') — no descriptions."
        )
        plans = _attempt(retry_system, user_prompt)

    return plans


# -- Task 3.4: run

def run(gaps: List[ResearchGap], papers: List[Paper]) -> List[ResearchPlan]:
    """
    Full Planner Agent pipeline: for each gap, select context and generate plans.

    Returns a flat list of all valid ResearchPlan objects across all gaps.
    source_gap is set on every plan for use in the main.py revision loop.

    Args:
        gaps   : Research gaps identified by the Librarian Agent.
        papers : Full list of retrieved papers with concepts populated.

    Returns:
        Flat List[ResearchPlan] - not nested by gap.
    """
    all_plans: List[ResearchPlan] = []

    for i, gap in enumerate(gaps):
        print(f"[planner_agent] Gap {i+1}/{len(gaps)}: '{gap.description[:60]}...'")

        context = select_context_papers(gap, papers)
        print(f"[planner_agent]   Selected {len(context)} context paper(s) by concept overlap.")

        plans = generate_plans(gap, context)
        print(f"[planner_agent]   Generated {len(plans)} valid plan(s).")

        all_plans.extend(plans)

    print(f"[planner_agent] Total plans generated: {len(all_plans)} across {len(gaps)} gap(s).")
    return all_plans
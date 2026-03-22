"""
agents/librarian_agent.py
--------------------------
Librarian Agent for the Agentic Research Planning Framework.

Responsibilities:
    1. Retrieve research papers from arXiv for a given topic.
    2. Extract key technical concepts from each abstract via LLM.
    3. Build a concept frequency map across all papers.
    4. Identify research gaps — weakly explored concept combinations.

Usage:
    from agents.librarian_agent import run
    papers, gaps = run("continual learning")
"""

import json
from collections import Counter
from typing import List, Tuple

from groq import Groq

import config
from models.schemas import Paper, ResearchGap
from tools.arxiv_tool import fetch_papers


# ── Groq client — initialised once at module level ────────────────────────────
_client = Groq(api_key=config.GROQ_API_KEY)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Make a single Groq API call and return the raw response text.
    Uses JSON mode to enforce structured output.
    """
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
    Parse a JSON string returned by the LLM.
    Logs a warning and returns None on failure — never crashes the pipeline.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[librarian_agent] WARNING: JSON parse failed for '{label}'. Error: {e}")
        print(f"[librarian_agent] Raw response was: {raw[:200]}")
        return None


# ── Task 2.2: extract_concepts ────────────────────────────────────────────────

def extract_concepts(paper: Paper) -> List[str]:
    """
    Extract 5–8 key technical concepts from a paper's abstract using an LLM.

    The LLM is instructed to return a JSON object with a single key 'concepts'
    containing an array of short concept strings. JSON mode on Groq requires
    a top-level object (not a bare array), so we wrap accordingly.

    Args:
        paper: A Paper object with a populated abstract.

    Returns:
        List of concept strings. Empty list if extraction fails.
    """
    system_prompt = (
        "You extract technical concepts from research paper abstracts. "
        "Return ONLY a valid JSON object with a single key 'concepts' "
        "whose value is an array of 5 to 8 short concept strings. "
        "No explanation, no markdown, no extra keys."
    )
    user_prompt = f"Abstract: {paper.abstract}"

    raw = _call_llm(system_prompt, user_prompt)
    parsed = _parse_json(raw, f"extract_concepts:{paper.arxiv_id}")

    if parsed and isinstance(parsed.get("concepts"), list):
        return [str(c) for c in parsed["concepts"]]

    print(f"[librarian_agent] WARNING: Could not extract concepts for paper '{paper.title[:60]}'")
    return []


# ── Task 2.3: identify_gaps ───────────────────────────────────────────────────

def identify_gaps(papers: List[Paper]) -> List[ResearchGap]:
    """
    Identify research gaps from the aggregated concept landscape of all papers.

    Builds a concept frequency map (how often each concept appears across papers)
    and a paper-concept index (which concepts each paper covers), then asks the
    LLM to identify concept combinations that are weakly covered or isolated.

    Args:
        papers: List of Paper objects with concepts already populated.

    Returns:
        List of ResearchGap objects. Empty list if identification fails.
    """
    # Build concept frequency map
    all_concepts = [concept for paper in papers for concept in paper.concepts]
    concept_freq = dict(Counter(all_concepts).most_common())

    # Build paper → concept index (using arxiv_id as key)
    paper_concept_index = {
        paper.arxiv_id: paper.concepts
        for paper in papers
        if paper.concepts
    }

    if not concept_freq:
        print("[librarian_agent] WARNING: No concepts available — cannot identify gaps.")
        return []

    system_prompt = (
        "You identify research gaps from a scientific concept frequency map. "
        "A gap is a concept or concept combination that appears rarely or in isolation "
        "compared to the broader literature. "
        "Return ONLY a valid JSON object with a single key 'gaps' whose value is an array. "
        "Each gap object must have exactly these keys: "
        "'description' (string), 'related_concepts' (array of strings), "
        "'supporting_paper_ids' (array of strings). "
        "Return between 2 and 5 gaps. No explanation, no extra keys."
    )
    user_prompt = (
        f"Concept frequency map: {json.dumps(concept_freq)}\n\n"
        f"Paper IDs and their concepts: {json.dumps(paper_concept_index)}"
    )

    raw = _call_llm(system_prompt, user_prompt)
    parsed = _parse_json(raw, "identify_gaps")

    if not parsed or not isinstance(parsed.get("gaps"), list):
        print("[librarian_agent] WARNING: Could not identify gaps from concept map.")
        return []

    gaps = []
    for item in parsed["gaps"]:
        try:
            gaps.append(ResearchGap(
                description=str(item["description"]),
                related_concepts=list(item.get("related_concepts", [])),
                supporting_paper_ids=list(item.get("supporting_paper_ids", [])),
            ))
        except (KeyError, TypeError) as e:
            print(f"[librarian_agent] WARNING: Skipping malformed gap entry. Error: {e}")

    return gaps


# ── Task 2.4: run ─────────────────────────────────────────────────────────────

def run(topic: str) -> Tuple[List[Paper], List[ResearchGap]]:
    """
    Full Librarian Agent pipeline: fetch → extract concepts → identify gaps.

    Args:
        topic: Research topic string (e.g. 'continual learning').

    Returns:
        Tuple of (papers, gaps) where:
          - papers is a List[Paper] with concepts populated on each
          - gaps   is a List[ResearchGap] identified from the concept landscape
    """
    print(f"[librarian_agent] Fetching papers for topic: '{topic}'")
    papers = fetch_papers(topic, max_results=config.MAX_PAPERS)

    if not papers:
        print("[librarian_agent] ERROR: No papers retrieved. Cannot continue.")
        return [], []

    print(f"[librarian_agent] Retrieved {len(papers)} papers. Extracting concepts...")

    for i, paper in enumerate(papers):
        concepts = extract_concepts(paper)
        paper.concepts = concepts
        print(f"[librarian_agent]   [{i+1}/{len(papers)}] '{paper.title[:55]}...' → {concepts}")

    populated = [p for p in papers if p.concepts]
    print(f"[librarian_agent] Concepts extracted for {len(populated)}/{len(papers)} papers.")

    print("[librarian_agent] Identifying research gaps...")
    gaps = identify_gaps(papers)
    print(f"[librarian_agent] Identified {len(gaps)} research gap(s).")

    return papers, gaps

"""
tools/arxiv_tool.py
--------------------
arXiv API wrapper for the Agentic Research Planning Framework.

Responsibilities:
    - Accept a research topic string and optional result limit.
    - Query arXiv using the official Python client.
    - Return a list of Paper objects with title, abstract, and arxiv_id populated.
    - Leave Paper.concepts empty — the Librarian Agent populates these later.

Usage:
    from tools.arxiv_tool import fetch_papers
    papers = fetch_papers("federated learning", max_results=10)
"""

import arxiv
from typing import List

from models.schemas import Paper


def fetch_papers(query: str, max_results: int = 20) -> List[Paper]:
    """
    Query arXiv and return a list of Paper objects.

    Papers are sorted by relevance to the query string.
    Paper.concepts will be empty — populated later by the Librarian Agent.

    Args:
        query       : Research topic or keyword string (e.g. 'federated learning').
        max_results : Maximum number of papers to retrieve. Defaults to 20.

    Returns:
        List of Paper objects. Empty list if the query returns no results
        or a network error occurs.
    """
    papers: List[Paper] = []

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for result in search.results():
            papers.append(Paper(
                title=result.title,
                abstract=result.summary,
                arxiv_id=result.entry_id,
                concepts=[]
            ))

    except Exception as e:
        print(f"[arxiv_tool] WARNING: Failed to retrieve papers for query '{query}'. Error: {e}")

    if not papers:
        print(f"[arxiv_tool] WARNING: No papers returned for query '{query}'.")

    return papers

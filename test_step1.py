"""
test_step1.py
-------------
Validation test for Step 1: arXiv Retrieval.

Run this from the project root after completing Step 1:
    python test_step1.py

All five checks from the implementation plan validation checklist are covered.
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from tools.arxiv_tool import fetch_papers
from models.schemas import Paper


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def run_tests():
    print("=" * 60)
    print("Step 1 Validation — arXiv Retrieval")
    print("=" * 60)

    # ── Test 1: Basic retrieval returns results ───────────────────
    print("\nTest 1: fetch_papers returns a non-empty list")
    papers = fetch_papers("federated learning", max_results=5)
    check("Returns a non-empty list", len(papers) > 0)

    if not papers:
        print("\nCannot continue — no papers returned. Check network connection.")
        return

    # ── Test 2: Paper fields are populated ────────────────────────
    print("\nTest 2: Each Paper has non-empty title, abstract, arxiv_id")
    all_fields_ok = all(
        p.title and p.abstract and p.arxiv_id
        for p in papers
    )
    check("All papers have title, abstract, arxiv_id", all_fields_ok)

    # ── Test 3: concepts list is empty (intentionally) ────────────
    print("\nTest 3: concepts list is empty on retrieval (populated later by Librarian)")
    all_concepts_empty = all(p.concepts == [] for p in papers)
    check("concepts list is empty on all papers", all_concepts_empty)

    # ── Test 4: max_results is respected ──────────────────────────
    print("\nTest 4: max_results parameter limits returned papers")
    limited = fetch_papers("graph neural networks", max_results=3)
    check("Returns <= 3 papers when max_results=3", len(limited) <= 3)

    # ── Test 5: graceful handling of bad query ────────────────────
    print("\nTest 5: Handles unusual query without crashing")
    try:
        result = fetch_papers("xyzzy_nonexistent_topic_12345", max_results=3)
        check("No exception raised on unusual query", True)
        check("Returns a list (possibly empty)", isinstance(result, list))
    except Exception as e:
        check(f"No exception raised — FAILED with: {e}", False)

    # ── Print sample output ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sample Output — First 3 papers retrieved:")
    print("=" * 60)
    for i, p in enumerate(papers[:3], 1):
        print(f"\n  Paper {i}:")
        print(f"    Title    : {p.title}")
        print(f"    arXiv ID : {p.arxiv_id}")
        print(f"    Abstract : {p.abstract[:120]}...")
        print(f"    Concepts : {p.concepts}  ← empty until Librarian Agent runs")

    print("\n" + "=" * 60)
    print("Step 1 validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

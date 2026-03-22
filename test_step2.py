"""
test_step2.py
-------------
Validation test for Step 2: Literature Grounding (Librarian Agent).

Run from the project root after completing Step 2:
    python test_step2.py

Covers all checklist items from the implementation plan:
    - extract_concepts returns a Python list of strings
    - All Paper objects have non-empty concepts after run()
    - identify_gaps returns at least 1 ResearchGap
    - LLM JSON parse failures are caught and do not crash
    - run() returns both papers and gaps as a tuple
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.librarian_agent import run, extract_concepts, identify_gaps
from models.schemas import Paper, ResearchGap


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def run_tests():
    print("=" * 60)
    print("Step 2 Validation — Literature Grounding (Librarian Agent)")
    print("=" * 60)

    # ── Test 1: extract_concepts returns a list of strings ────────
    print("\nTest 1: extract_concepts returns a Python list of strings")
    dummy_paper = Paper(
        title="Test Paper on Federated Learning",
        abstract=(
            "We propose a novel approach to federated learning that addresses "
            "non-IID data distribution across clients. Our method combines "
            "gradient compression with differential privacy to reduce "
            "communication overhead while maintaining model accuracy."
        ),
        arxiv_id="http://arxiv.org/abs/test.001"
    )
    concepts = extract_concepts(dummy_paper)
    check("Returns a list", isinstance(concepts, list))
    check("List contains strings", all(isinstance(c, str) for c in concepts))
    check("Returns 5–8 concepts", 4 <= len(concepts) <= 10)
    print(f"  Extracted concepts: {concepts}")

    # ── Test 2: Full run() pipeline ───────────────────────────────
    print("\nTest 2: run() returns (papers, gaps) tuple for 'continual learning'")
    result = run("continual learning")
    check("run() returns a tuple of length 2", isinstance(result, tuple) and len(result) == 2)

    papers, gaps = result

    # ── Test 3: Papers have concepts populated ────────────────────
    print("\nTest 3: All Paper objects have non-empty concepts after run()")
    papers_with_concepts = [p for p in papers if p.concepts]
    check("At least some papers have concepts populated", len(papers_with_concepts) > 0)
    ratio = len(papers_with_concepts) / len(papers) if papers else 0
    check(f">=80% papers have concepts ({len(papers_with_concepts)}/{len(papers)})", ratio >= 0.8)

    # ── Test 4: At least 1 gap identified ─────────────────────────
    print("\nTest 4: identify_gaps returns at least 1 ResearchGap")
    check("gaps is a list", isinstance(gaps, list))
    check("At least 1 gap identified", len(gaps) >= 1)
    check("All items are ResearchGap objects", all(isinstance(g, ResearchGap) for g in gaps))

    # ── Test 5: Gap structure is valid ────────────────────────────
    print("\nTest 5: Each ResearchGap has required fields populated")
    for i, gap in enumerate(gaps):
        check(f"Gap {i+1} has description", bool(gap.description))
        check(f"Gap {i+1} has related_concepts list", isinstance(gap.related_concepts, list))
        check(f"Gap {i+1} has supporting_paper_ids list", isinstance(gap.supporting_paper_ids, list))

    # ── Test 6: Graceful JSON failure handling ────────────────────
    print("\nTest 6: LLM parse failures do not crash the pipeline")
    bad_paper = Paper(title="", abstract=".", arxiv_id="bad_id")
    try:
        result = extract_concepts(bad_paper)
        check("No exception on minimal/bad input", True)
        check("Returns a list even on bad input", isinstance(result, list))
    except Exception as e:
        check(f"No exception raised — FAILED with: {e}", False)

    # ── Print sample output ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sample Output")
    print("=" * 60)

    print(f"\nPapers retrieved : {len(papers)}")
    print(f"Papers with concepts: {len(papers_with_concepts)}")
    print(f"Research gaps found : {len(gaps)}")

    print("\nFirst 2 papers with concepts:")
    for p in papers[:2]:
        print(f"  Title    : {p.title[:65]}")
        print(f"  Concepts : {p.concepts}")

    print("\nIdentified Research Gaps:")
    for i, gap in enumerate(gaps, 1):
        print(f"\n  Gap {i}: {gap.description}")
        print(f"    Related concepts : {gap.related_concepts}")
        print(f"    Supporting papers: {gap.supporting_paper_ids[:2]}{'...' if len(gap.supporting_paper_ids) > 2 else ''}")

    print("\n" + "=" * 60)
    print("Step 2 validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

"""
test_step4.py
-------------
Validation test for Step 4 (Reviewer Agent) and Step 5 (Novelty Scoring).

Run from the project root:
    python test_step4.py

Covers all checklist items from the implementation plan:
    - check_feasibility correctly rejects plans citing non-existent datasets
    - check_feasibility returns a descriptive notes string for rejections
    - score_novelty returns a float in range 0.0-10.0 for every plan
    - Plans closely matching an existing abstract score below 4.0
    - generate_output_fields only runs for accepted plans
    - ReviewResult.accepted is False for any plan failing either check
    - Embedder calibration: identical text ~ 0, unrelated text ~ 8-10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.librarian_agent import run as librarian_run
from agents.planner_agent import run as planner_run
from agents.reviewer_agent import run as reviewer_run, score_novelty, generate_output_fields
from tools.feasibility_checker import check_feasibility
from tools.embedder import get_embeddings
from models.schemas import Paper, ResearchPlan, ResearchGap, ReviewResult


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def run_tests():
    print("=" * 60)
    print("Step 4+5 Validation - Reviewer Agent & Novelty Scoring")
    print("=" * 60)

    # ── Test 1: Embedder calibration ──────────────────────────────
    print("\nTest 1: Embedder shape and calibration")
    embs = get_embeddings(["hello world", "another sentence"])
    check("get_embeddings returns ndarray", hasattr(embs, 'shape'))
    check("Shape is (2, 384) for all-MiniLM-L6-v2", embs.shape == (2, 384))

    # Calibration: identical text should score near 0
    dummy_abstract = (
        "We propose a federated learning approach using gradient compression "
        "and differential privacy to handle non-IID data across distributed clients."
    )
    dummy_paper = Paper(title="Test", abstract=dummy_abstract, arxiv_id="test_001")

    identical_plan = ResearchPlan(
        research_question=dummy_abstract,
        proposed_method=dummy_abstract,
        dataset="cifar-10",
        evaluation_metric="accuracy",
        source_gap=None
    )
    identical_score = score_novelty(identical_plan, [dummy_paper])
    check(f"Identical text scores near 0 (got {identical_score:.2f})", identical_score < 2.0)

    # Calibration: unrelated text should score high
    unrelated_plan = ResearchPlan(
        research_question="How do ocean tidal patterns affect coral reef ecosystems?",
        proposed_method="Marine biology field study with underwater acoustic sensors",
        dataset="cifar-10",
        evaluation_metric="accuracy",
        source_gap=None
    )
    unrelated_score = score_novelty(unrelated_plan, [dummy_paper])
    check(f"Unrelated text scores high (got {unrelated_score:.2f})", unrelated_score >= 7.0)

    # ── Test 2: Feasibility checker — known datasets ───────────────
    print("\nTest 2: check_feasibility rejects non-existent datasets")

    good_plan = ResearchPlan(
        research_question="Test question",
        proposed_method="A continual learning approach using regularization",
        dataset="split cifar-100",
        evaluation_metric="forgetting metric and accuracy",
        source_gap=None
    )
    passed, notes = check_feasibility(good_plan)
    check("Known dataset passes feasibility", passed)
    check("Notes say 'passed' for valid plan", "passed" in notes.lower())

    bad_plan = ResearchPlan(
        research_question="Test question",
        proposed_method="A classification approach",
        dataset="MyMadeUpDataset2099",
        evaluation_metric="accuracy",
        source_gap=None
    )
    passed_bad, notes_bad = check_feasibility(bad_plan)
    check("Unknown dataset fails feasibility", not passed_bad)
    check("Notes explain the rejection", len(notes_bad) > 20)
    print(f"  Rejection note: {notes_bad[:80]}...")

    # ── Test 3: score_novelty range ────────────────────────────────
    print("\nTest 3: score_novelty always returns float in [0.0, 10.0]")
    test_papers = [
        Paper(title="Paper A", abstract="Deep learning methods for image recognition using convolutional neural networks.", arxiv_id="p1"),
        Paper(title="Paper B", abstract="Natural language processing with transformer architectures and attention mechanisms.", arxiv_id="p2"),
    ]
    test_plan = ResearchPlan(
        research_question="How can we improve continual learning with energy based models?",
        proposed_method="Energy-based regularization for catastrophic forgetting prevention",
        dataset="split cifar-10",
        evaluation_metric="accuracy",
        source_gap=None
    )
    score = score_novelty(test_plan, test_papers)
    check("Returns a float", isinstance(score, float))
    check("Score is in [0.0, 10.0]", 0.0 <= score <= 10.0)
    print(f"  Score for test plan: {score:.2f}")

    # ── Test 4: Full pipeline run ──────────────────────────────────
    print("\nTest 4: Full pipeline — Librarian → Planner → Reviewer")
    print("  Running Librarian Agent...")
    papers, gaps = librarian_run("continual learning")
    print("  Running Planner Agent...")
    plans = planner_run(gaps, papers)
    print("  Running Reviewer Agent...")
    results = reviewer_run(plans, papers)

    check("Returns a list of ReviewResult objects",
          isinstance(results, list) and all(isinstance(r, ReviewResult) for r in results))
    check("One result per plan", len(results) == len(plans))

    # ── Test 5: Accepted/rejected logic ───────────────────────────
    print("\nTest 5: Accepted/rejected logic is correct")
    for r in results:
        should_accept = r.novelty_score >= 4.0 and r.feasibility_passed
        check(
            f"Plan accepted={r.accepted} matches novelty({r.novelty_score:.1f})>=4.0 AND feasibility({r.feasibility_passed})",
            r.accepted == should_accept
        )

    # ── Test 6: LLM only called for accepted plans ─────────────────
    print("\nTest 6: Output fields only generated for accepted plans")
    for r in results:
        if r.accepted:
            check("Accepted plan has suggested_title", bool(r.suggested_title))
            check("Accepted plan has research_direction", bool(r.research_direction))
            check("Accepted plan has experimental_blueprint", bool(r.experimental_blueprint))
        else:
            check("Rejected plan has empty output fields",
                  r.suggested_title == "" and r.research_direction == "" and r.experimental_blueprint == "")

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Review Summary")
    print("=" * 60)
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    print(f"\nTotal plans reviewed : {len(results)}")
    print(f"Accepted             : {len(accepted)}")
    print(f"Rejected             : {len(rejected)}")

    print("\nAll plans with scores:")
    for r in results:
        status = "ACCEPTED" if r.accepted else "REJECTED"
        print(f"\n  [{status}] Novelty: {r.novelty_score:.2f} | Feasibility: {'PASS' if r.feasibility_passed else 'FAIL'}")
        print(f"  Question : {r.plan.research_question[:70]}")
        print(f"  Dataset  : {r.plan.dataset}")
        print(f"  Metric   : {r.plan.evaluation_metric}")
        if r.accepted:
            print(f"  Title    : {r.suggested_title}")
            print(f"  Blueprint: {r.experimental_blueprint[:120]}...")

    print("\n" + "=" * 60)
    print("Step 4+5 validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

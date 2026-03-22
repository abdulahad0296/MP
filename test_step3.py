"""
test_step3.py
-------------
Validation test for Step 3: Planner Agent.

Run from the project root after completing Step 3:
    python test_step3.py

Covers all checklist items from the implementation plan:
    - generate_plans returns exactly 3 plans (re-prompts once if fewer)
    - Every ResearchPlan has all 4 required fields populated and non-empty
    - run() returns a flat List[ResearchPlan] - not nested by gap
    - source_gap is set on every plan
    - Plans propose concrete methods, not just gap restatements
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.librarian_agent import run as librarian_run
from agents.planner_agent import run as planner_run, select_context_papers, generate_plans
from models.schemas import Paper, ResearchGap, ResearchPlan


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def run_tests():
    print("=" * 60)
    print("Step 3 Validation - Planner Agent")
    print("=" * 60)

    # Setup: reuse Librarian output as input
    print("\nSetup: Running Librarian Agent to get gaps and papers...")
    papers, gaps = librarian_run("continual learning")

    if not gaps:
        print("ERROR: No gaps returned by Librarian. Cannot test Planner.")
        return
    print(f"  Using {len(papers)} papers and {len(gaps)} gaps as input.\n")

    # Test 1: select_context_papers
    print("Test 1: select_context_papers selects by concept overlap")
    gap = gaps[0]
    context = select_context_papers(gap, papers, n=4)
    check("Returns a list", isinstance(context, list))
    check("Returns <= 4 papers", len(context) <= 4)
    check("Returns Paper objects", all(isinstance(p, Paper) for p in context))
    print(f"  Gap concepts   : {gap.related_concepts}")
    print(f"  Context papers : {[p.title[:45] for p in context]}")

    # Test 2: generate_plans for a single gap
    print("\nTest 2: generate_plans returns exactly 3 plans for one gap")
    plans = generate_plans(gap, context)
    check("Returns a list", isinstance(plans, list))
    check("Returns exactly 3 plans", len(plans) == 3)
    check("All items are ResearchPlan objects", all(isinstance(p, ResearchPlan) for p in plans))

    # Test 3: All 4 required fields are populated
    print("\nTest 3: Every plan has all 4 required fields non-empty")
    for i, plan in enumerate(plans):
        check(f"Plan {i+1} has research_question", bool(plan.research_question.strip()))
        check(f"Plan {i+1} has proposed_method",   bool(plan.proposed_method.strip()))
        check(f"Plan {i+1} has dataset",            bool(plan.dataset.strip()))
        check(f"Plan {i+1} has evaluation_metric",  bool(plan.evaluation_metric.strip()))

    # Test 4: source_gap is set on every plan
    print("\nTest 4: source_gap is set on every plan")
    check("All plans have source_gap set", all(p.source_gap is not None for p in plans))
    check("source_gap is a ResearchGap", all(isinstance(p.source_gap, ResearchGap) for p in plans))

    # Test 5: Full run() returns flat list
    print("\nTest 5: run() returns a flat List[ResearchPlan] across all gaps")
    all_plans = planner_run(gaps, papers)
    check("Returns a list", isinstance(all_plans, list))
    check("List is not nested (all items are ResearchPlan)", all(isinstance(p, ResearchPlan) for p in all_plans))
    check(f"Returns plans for all gaps ({len(gaps)} gaps x ~3 plans)", len(all_plans) >= len(gaps))
    check("Every plan has source_gap set", all(p.source_gap is not None for p in all_plans))

    # Test 6: Plans are concrete, not gap restatements
    print("\nTest 6: Plans propose concrete methods (not just gap restatements)")
    for i, plan in enumerate(plans[:3]):
        gap_words = set(gap.description.lower().split())
        plan_words = set(plan.proposed_method.lower().split())
        overlap_ratio = len(gap_words & plan_words) / max(len(gap_words), 1)
        check(f"Plan {i+1} proposed_method is not a copy of gap (<60% word overlap)",
              overlap_ratio < 0.6)

    # Print sample output
    print("\n" + "=" * 60)
    print("Sample Output - Plans for first 2 gaps")
    print("=" * 60)

    gap_plans = {}
    for plan in all_plans:
        key = plan.source_gap.description[:55]
        gap_plans.setdefault(key, []).append(plan)

    for gap_desc, gap_plan_list in list(gap_plans.items())[:2]:
        print(f"\nGap: {gap_desc}...")
        for j, plan in enumerate(gap_plan_list, 1):
            print(f"\n  Plan {j}:")
            print(f"    Research Question : {plan.research_question}")
            print(f"    Proposed Method   : {plan.proposed_method}")
            print(f"    Dataset           : {plan.dataset}")
            print(f"    Evaluation Metric : {plan.evaluation_metric}")

    print(f"\nTotal plans across all gaps: {len(all_plans)}")
    print("\n" + "=" * 60)
    print("Step 3 validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
